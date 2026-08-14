r"""
Run CARDIAC-FM on your own ECGs and get a risk score per recording.

Unlike `common/train_eval/ecg_test.py` -- which is an *evaluation* script: it requires ground-truth
labels, keeps only the recordings that have one, and reports AUROC -- this script needs no labels.
Give it a manifest of ECGs and a fine-tuned checkpoint, and it scores every recording in the
manifest.

    python infer.py \
      --ecg_dir  /path/to/manifest_dir \       # holds <split>.tsv, see docs/DATA_FORMAT.md
      --ckpt     /path/to/af5_ecg.pth \        # from huggingface.co/lst627/CARDIAC-FM
      --ecg_ckpt /path/to/ecgfm_mimic_iv_physionet.pt \
      --out      predictions.csv

Output CSV: `id,risk_score`, where risk_score is the sigmoid of the model logit, in (0, 1).
It is a *relative* score for ranking, not a calibrated absolute probability for your population.

Optional:
  --labels <dir with y.npy>   also report AUROC / AUPRC on the subset that has labels
  --mode ecg_mri --mri_dir D  use an *_ecg_mri.pth checkpoint; D holds per-subject
                              vst_{2ch,4ch,sa}.npy (see docs/DATA_FORMAT.md for why producing
                              those from raw DICOM/NIfTI is not something this repo can do for you)

Uses a GPU when one is visible, otherwise CPU.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in ("common", os.path.join("common", "mri_encoder"), os.path.join("common", "data"),
           os.path.join("UKBB", "downstream")):
    sys.path.insert(0, os.path.join(_ROOT, _p))

from fairseq_signals.data.ecg.raw_ecg_dataset import FileECGDataset      # noqa: E402
from downstream_ecgmri_cinema import ECGMRIDownstreamModel               # noqa: E402

# The published m75 checkpoints were all trained with exactly this architecture. Changing any of it
# silently mis-loads weights, so it is pinned here rather than exposed as a flag.
M75_CFG = dict(
    img_size=224, n_frames=8,
    tube_t=2, patch_h=16, patch_w=16,
    encoder_dim=768, encoder_depth=12, encoder_heads=12,
    decoder_dim=192, decoder_depth=4, decoder_heads=6,   # decoder is unused at inference
    mask_ratio=0.9, norm_pix_loss=True,
    view_encoder="conv", view_depth=2, n_sa_slices=3,
)
M75_POOL = "per_view"


class InferenceDataset(Dataset):
    """Every recording in the manifest, with no label requirement."""

    def __init__(self, ecg_dir, split, mode="ecg", mri_dir=None, img_size=224, n_frames=8):
        self.mode, self.mri_dir = mode, mri_dir
        self.img_size, self.n_frames = img_size, n_frames

        manifest_path = os.path.join(ecg_dir, f"{split}.tsv")
        self.ecg_data = FileECGDataset(
            manifest_path=manifest_path,
            sample_rate=None, max_sample_size=None, min_sample_size=None,
            pad=True, pad_leads=False, leads_to_load=None,
            label=False, filter=False, normalize=False,
            mean_path=None, std_path=None, num_buckets=0,
            compute_mask_indices=False, leads_bucket=None,
            bucket_selection="uniform", training=False,
        )
        # manifest: line 1 is the data root, then one "<file>\t<n_samples>" per recording
        ids = []
        with open(manifest_path) as f:
            f.readline()
            for line in f:
                if line.strip():
                    ids.append(os.path.splitext(line.strip().split("\t")[0])[0])

        if mode == "ecg_mri":
            keep = [(i, e) for i, e in enumerate(ids)
                    if all(os.path.exists(os.path.join(mri_dir, e, f))
                           for f in ("vst_2ch.npy", "vst_4ch.npy", "vst_sa.npy"))]
            dropped = len(ids) - len(keep)
            if dropped:
                print(f"  skipping {dropped} recording(s) with no MRI under {mri_dir}", flush=True)
            self.samples = keep
        else:
            self.samples = list(enumerate(ids))

    def __len__(self):
        return len(self.samples)

    def _center_frames(self, x):
        T = x.shape[1]
        if T <= self.n_frames:
            return x
        start = (T - self.n_frames) // 2          # center crop; never random at inference
        return x[:, start:start + self.n_frames]

    def _load_mri(self, eid):
        subj = os.path.join(self.mri_dir, eid)

        def load_resize(fname):
            x = torch.from_numpy(np.load(os.path.join(subj, fname)).astype(np.float32))
            S, T, H, W = x.shape
            if H != self.img_size or W != self.img_size:
                x = x.reshape(S * T, 1, H, W)
                x = F.interpolate(x, (self.img_size, self.img_size),
                                  mode="bilinear", align_corners=False)
                x = x.reshape(S, T, self.img_size, self.img_size)
            return self._center_frames(x)

        return {"v2ch": load_resize("vst_2ch.npy"),
                "v4ch": load_resize("vst_4ch.npy"),
                "vsa":  load_resize("vst_sa.npy")}

    def __getitem__(self, idx):
        ecg_idx, eid = self.samples[idx]
        ecg = self.ecg_data[ecg_idx]
        mri = self._load_mri(eid) if self.mode == "ecg_mri" else None
        return ecg, mri

    @staticmethod
    def collate_fn(batch):
        # Padding matches DownstreamDataset.collate_fn exactly; the model's masked mean-pool
        # depends on padding_mask being True where padded.
        ecg_list, mri_list = zip(*batch)
        sources = [s["source"] for s in ecg_list]
        max_len = max(s.shape[-1] for s in sources)
        padded = torch.zeros(len(sources), sources[0].shape[0], max_len)
        masks = torch.ones(len(sources), max_len, dtype=torch.bool)
        for i, s in enumerate(sources):
            padded[i, :, :s.shape[-1]] = s
            masks[i, :s.shape[-1]] = False
        ecg_collated = {"net_input": {"source": padded, "padding_mask": masks}}
        mri_collated = ({k: torch.stack([m[k] for m in mri_list]) for k in mri_list[0]}
                        if mri_list[0] is not None else None)
        return ecg_collated, mri_collated


def main():
    ap = argparse.ArgumentParser(
        description="Score your own ECGs with a fine-tuned CARDIAC-FM checkpoint. No labels needed.")
    ap.add_argument("--ecg_dir", required=True, help="directory holding <split>.tsv")
    ap.add_argument("--ckpt", required=True, help="fine-tuned checkpoint, e.g. af5_ecg.pth")
    ap.add_argument("--ecg_ckpt", required=True, help="ECG-FM backbone .pt")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", default="ecg", choices=["ecg", "ecg_mri"])
    ap.add_argument("--mri_dir", default=None)
    ap.add_argument("--labels", default=None, help="optional dir with y.npy, to report AUROC")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda when available)")
    args = ap.parse_args()

    if args.mode == "ecg_mri" and not args.mri_dir:
        ap.error("--mode ecg_mri requires --mri_dir")
    manifest = os.path.join(args.ecg_dir, f"{args.split}.tsv")
    if not os.path.exists(manifest):
        ap.error(f"manifest not found: {manifest}\n"
                 f"  --ecg_dir must contain <split>.tsv -- see docs/DATA_FORMAT.md")

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    model = ECGMRIDownstreamModel(cl_ckpt=None, ecg_ckpt=args.ecg_ckpt, mae_cfg=M75_CFG,
                                  mode=args.mode, training_type="full", pool=M75_POOL)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # In --mode ecg the checkpoint still carries mri_encoder weights; they are simply unused.
    real_missing = [k for k in missing if not k.startswith("mri_")] if args.mode == "ecg" else missing
    if real_missing:
        print(f"  WARNING: {len(real_missing)} missing keys, e.g. {real_missing[:4]}", flush=True)
    model.to(device).eval()

    ds = InferenceDataset(args.ecg_dir, args.split, mode=args.mode, mri_dir=args.mri_dir,
                          img_size=M75_CFG["img_size"], n_frames=M75_CFG["n_frames"])
    if len(ds) == 0:
        sys.exit(f"no recordings to score in {manifest}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=InferenceDataset.collate_fn)
    print(f"scoring {len(ds)} recording(s) from {manifest}", flush=True)

    scores = []
    with torch.no_grad():
        for ecgs, mris in loader:
            ecgs["net_input"]["source"] = ecgs["net_input"]["source"].to(device)
            ecgs["net_input"]["padding_mask"] = ecgs["net_input"]["padding_mask"].to(device)
            if mris is not None:
                mris = {k: v.to(device) for k, v in mris.items()}
            logits = model(ecgs, mris)
            scores.extend(torch.sigmoid(logits).float().cpu().numpy().reshape(-1).tolist())

    out = pd.DataFrame({"id": [e for _, e in ds.samples], "risk_score": scores})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows -> {args.out}", flush=True)
    print(f"  risk_score  min={out.risk_score.min():.4f}  "
          f"median={out.risk_score.median():.4f}  max={out.risk_score.max():.4f}", flush=True)

    if args.labels:
        from sklearn.metrics import roc_auc_score, average_precision_score
        y = np.load(os.path.join(args.labels, "y.npy")).squeeze()
        idx = np.array([i for i, _ in ds.samples])
        if idx.max() >= len(y):
            print("  labels file is shorter than the manifest; skipping evaluation", flush=True)
            return
        y = y[idx]
        keep = ~np.isnan(y)
        if keep.sum() and len(np.unique(y[keep])) > 1:
            s = out.risk_score.values
            print(f"  AUROC={roc_auc_score(y[keep], s[keep]):.4f}  "
                  f"AUPRC={average_precision_score(y[keep], s[keep]):.4f}  "
                  f"(n={int(keep.sum())}, events={int(y[keep].sum())})", flush=True)
        else:
            print("  labels present but not evaluable (all NaN or a single class)", flush=True)


if __name__ == "__main__":
    main()
