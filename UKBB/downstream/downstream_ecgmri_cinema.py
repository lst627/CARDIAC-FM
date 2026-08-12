"""
Downstream ECG+MRI prediction using Stage-1 CL pretrained encoders (CineMA variant).

Mirrors downstream_ecgmri.py, but the MRI tower is the CineMA cross-fusion encoder:
its fusion is internal, so it yields one fused (B, embed_dim) vector and the MRI
projection is Linear(embed_dim, 512) — not Linear(3*embed_dim, 512).

Consumes the Stage-1 checkpoint written by stage1_CL_cinema.py (key prefixes
mri_encoder.mae.*, mri_projection.*, ecg_encoder.*, ecg_projection.*).

Supports two modes:
  --mode ecg      : ECG embedding (512) -> Linear(512, 1)
  --mode ecg_mri  : concat(ECG 512, MRI 512) -> Linear(1024, 1)

Usage:
    python downstream_ecgmri_cinema.py \
        --outcome af5 --mode ecg_mri --training_type partial \
        --cl_ckpt  /gpfs/projects/trend/bojun/mri/CineMA/checkpoints/stage1_cinema/stage1_cinema_best.pth \
        --ecg_ckpt /gpfs/projects/trend/bojun/multimodal/mimic_iv_ecg_physionet_pretrained.pt \
        --mri_dir  /gpfs/projects/trend/data/UKBB/MRI/cropped_new \
        --ecg_dir  /gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual/stage1/ecg_tsv \
        --csv_train .../MRI_train/af5.csv --csv_val .../MRI_valid_new/af5.csv --csv_test .../MRI_test_new/af5.csv \
        --out_dir  /gpfs/projects/trend/bojun/mri/CineMA/checkpoints/downstream_ecg_mri
"""

import os
import sys
import json
import argparse
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

from fairseq_signals.models import build_model_from_checkpoint
from fairseq_signals.data.ecg.raw_ecg_dataset import FileECGDataset

# CineMA cross-fusion MRI encoder
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # cardiacfm_new/
sys.path.insert(0, os.path.join(_ROOT, "common", "mri_encoder"))
from encoder import CineMAEncoder


# ── model ─────────────────────────────────────────────────────────────────────

class ECGMRIDownstreamModel(nn.Module):
    """
    Loads ECG + MRI encoders and projection heads from Stage-1 CL checkpoint,
    adds a binary prediction head.

    mode = "ecg"     : ECG embedding (512) -> Linear(512, 1)
    mode = "ecg_mri" : concat(ECG 512, MRI 512) -> Linear(1024, 1)

    training_type:
      "full"      : all parameters trainable
      "senc_proj" : freeze backbones, train enc_norm + projections + head
      "partial"   : freeze all MRI except last `unfreeze_blocks` enc_blocks + enc_norm
      "linear"    : freeze encoders + projections, train only the head
    """
    def __init__(self, cl_ckpt, ecg_ckpt, mae_cfg, mode="ecg_mri",
                 training_type="senc_proj", unfreeze_blocks=2, pool="mean"):
        super().__init__()
        self.mode = mode

        # ── MRI encoder + projection ───────────────────────────────────────
        # readout dim depends on pooling: 'mean'/'cls' -> encoder_dim, 'per_view' -> 3x.
        self.mri_encoder = CineMAEncoder(mae_cfg=mae_cfg, ckpt=None, freeze=False, pool=pool)
        mri_dim = self.mri_encoder.embed_dim
        self.mri_projection = nn.Sequential(
            nn.LayerNorm(mri_dim),
            nn.Dropout(0.1),
            nn.Linear(mri_dim, 512),
        )

        # ── ECG encoder + projection ───────────────────────────────────────
        self.ecg_encoder = build_model_from_checkpoint(ecg_ckpt)
        self.ecg_projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
        )

        # ── load Stage-1 CL weights by prefix ──────────────────────────────
        if cl_ckpt is not None:
            state = torch.load(cl_ckpt, map_location="cpu", weights_only=False)
            sd = state.get("model", state)
            self._load_part(self.mri_encoder.mae, sd, "mri_encoder.mae.")
            self._load_part(self.mri_projection,  sd, "mri_projection.")
            self._load_part(self.ecg_encoder,     sd, "ecg_encoder.")
            self._load_part(self.ecg_projection,  sd, "ecg_projection.")
            print(f"[downstream] loaded CL weights from {cl_ckpt}")

        # ── freeze / unfreeze per training_type ────────────────────────────
        if training_type in ("senc_proj", "partial", "linear"):
            for p in self.mri_encoder.mae.parameters():
                p.requires_grad_(False)
            for p in self.ecg_encoder.parameters():
                p.requires_grad_(False)
        if training_type in ("senc_proj", "partial"):
            for p in self.mri_encoder.mae.enc_norm.parameters():
                p.requires_grad_(True)
        if training_type == "partial":
            for blk in self.mri_encoder.mae.enc_blocks[-unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
        if training_type == "linear":
            for p in self.mri_projection.parameters():
                p.requires_grad_(False)
            for p in self.ecg_projection.parameters():
                p.requires_grad_(False)

        # ── prediction head ────────────────────────────────────────────────
        in_dim = 1024 if mode == "ecg_mri" else 512
        self.head = nn.Linear(in_dim, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.0)

    @staticmethod
    def _load_part(module, full_sd, prefix):
        sub_sd = {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}
        if sub_sd:
            missing, unexpected = module.load_state_dict(sub_sd, strict=False)
            if missing:
                print(f"  [{prefix.rstrip('.')}] missing keys: {len(missing)}")
            if unexpected:
                print(f"  [{prefix.rstrip('.')}] unexpected keys: {len(unexpected)}")
        else:
            print(f"  [{prefix.rstrip('.')}] WARNING: no keys matched this prefix")

    def _encode_ecg(self, ecgs):
        feats = self.ecg_encoder.extract_features(
            source=ecgs["net_input"]["source"],
            padding_mask=ecgs["net_input"]["padding_mask"])
        x = feats["x"]
        pad_mask = feats.get("padding_mask", ecgs["net_input"]["padding_mask"])
        if pad_mask is not None:
            non_pad = ~pad_mask
            x = (x * non_pad.unsqueeze(-1).float()).sum(1) / non_pad.float().sum(1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(1)
        return self.ecg_projection(x)          # (B, 512)

    def _encode_mri(self, mris):
        fused = self.mri_encoder(mris["v2ch"], mris["v4ch"], mris["vsa"])  # (B, embed_dim)
        return self.mri_projection(fused)       # (B, 512)

    def forward(self, ecgs, mris=None):
        ecg_emb = self._encode_ecg(ecgs)
        if self.mode == "ecg_mri":
            mri_emb = self._encode_mri(mris)
            feat = torch.cat([ecg_emb, mri_emb], dim=-1)
        else:
            feat = ecg_emb
        return self.head(feat).squeeze(-1)      # (B,) logits


# ── dataset ───────────────────────────────────────────────────────────────────

class DownstreamDataset(Dataset):
    """Loads paired ECG + (optionally) MRI + binary outcome label."""
    def __init__(self, outcome_csv, ecg_dir, mri_dir, split, outcome_col,
                 mode="ecg_mri", img_size=224, n_frames=8, manifest_split=None):
        self.mri_dir  = mri_dir
        self.img_size = img_size
        self.n_frames = n_frames
        self.augment  = (split == "train")
        self.mode     = mode
        # ECG manifest is keyed by split name ({ecg_dir}/{split}.tsv). manifest_split lets an eval slot
        # read a *different* manifest (e.g. predict on the train set) while keeping augment/training off.
        ms = manifest_split or split

        df = pd.read_csv(outcome_csv)
        df["eid_visit"] = df["eid_visit"].astype(str)
        df = df.dropna(subset=[outcome_col])
        label_map = dict(zip(df["eid_visit"], df[outcome_col].astype(float)))

        manifest_path = os.path.join(ecg_dir, f"{ms}.tsv")
        self.ecg_data = FileECGDataset(
            manifest_path=manifest_path,
            sample_rate=None, max_sample_size=None, min_sample_size=None,
            pad=True, pad_leads=False, leads_to_load=None,
            label=False, filter=False, normalize=False,
            mean_path=None, std_path=None, num_buckets=0,
            compute_mask_indices=False, leads_bucket=None,
            bucket_selection="uniform", training=(split == "train"),
        )
        ecg_eid_to_idx = {}
        with open(manifest_path) as f:
            f.readline()
            for i, line in enumerate(f):
                eid = os.path.splitext(line.strip().split("\t")[0])[0]
                ecg_eid_to_idx[eid] = i

        valid = set(ecg_eid_to_idx) & set(label_map)
        if mode == "ecg_mri":
            mri_eids = {
                d for d in os.listdir(mri_dir)
                if os.path.isdir(os.path.join(mri_dir, d)) and
                all(os.path.exists(os.path.join(mri_dir, d, f))
                    for f in ["vst_2ch.npy", "vst_4ch.npy", "vst_sa.npy"])
            }
            valid &= mri_eids

        self.samples = [(eid, ecg_eid_to_idx[eid], label_map[eid]) for eid in sorted(valid)]
        pos = sum(l for _, _, l in self.samples)
        print(f"[{split}] n={len(self.samples)}  pos={int(pos)}  neg={len(self.samples)-int(pos)}")

    def __len__(self):
        return len(self.samples)

    def _crop_frames(self, x):
        T = x.shape[1]
        if T <= self.n_frames:
            return x
        start = torch.randint(0, T - self.n_frames + 1, (1,)).item() if self.augment \
                else (T - self.n_frames) // 2
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
            return self._crop_frames(x)
        return {"v2ch": load_resize("vst_2ch.npy"),
                "v4ch": load_resize("vst_4ch.npy"),
                "vsa":  load_resize("vst_sa.npy")}

    def __getitem__(self, idx):
        eid, ecg_idx, label = self.samples[idx]
        ecg = self.ecg_data[ecg_idx]
        mri = self._load_mri(eid) if self.mode == "ecg_mri" else None
        return ecg, mri, torch.tensor(label, dtype=torch.float32)

    @staticmethod
    def collate_fn(batch):
        ecg_list, mri_list, labels = zip(*batch)
        labels = torch.stack(labels)
        sources = [s["source"] for s in ecg_list]
        max_len = max(s.shape[-1] for s in sources)
        padded = torch.zeros(len(sources), sources[0].shape[0], max_len)
        masks  = torch.ones(len(sources), max_len, dtype=torch.bool)
        for i, s in enumerate(sources):
            padded[i, :, :s.shape[-1]] = s
            masks[i, :s.shape[-1]] = False
        ecg_collated = {"net_input": {"source": padded, "padding_mask": masks}}
        mri_collated = {k: torch.stack([m[k] for m in mri_list]) for k in mri_list[0]} \
                       if mri_list[0] is not None else None
        return ecg_collated, mri_collated, labels


# ── training ──────────────────────────────────────────────────────────────────

def set_seed(seed):
    """Make a single run reproducible (init, shuffle, MRI frame-crop augmentation)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id):
    """Seed numpy/random per DataLoader worker (torch's worker RNG is auto-seeded by PyTorch)."""
    s = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(s)
    random.seed(s)


def cosine_lr(optimizer, warmup_length, steps):
    initial_lrs = [g["lr"] for g in optimizer.param_groups]
    def _lr_adjuster(step):
        if step < warmup_length:
            factor = (step + 1) / warmup_length
        else:
            e  = step - warmup_length
            es = steps - warmup_length
            factor = 0.5 * (1 + math.cos(math.pi * e / es))
        for g, lr0 in zip(optimizer.param_groups, initial_lrs):
            g["lr"] = lr0 * factor
    return _lr_adjuster


def run_epoch(loader, model, optimizer, scheduler, device, step_offset, train, return_preds=False):
    model.train() if train else model.eval()
    losses, preds, trues = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, (ecgs, mris, labels) in enumerate(loader):
            if train and scheduler is not None:
                scheduler(i + step_offset)
            ecgs["net_input"]["source"]       = ecgs["net_input"]["source"].to(device)
            ecgs["net_input"]["padding_mask"] = ecgs["net_input"]["padding_mask"].to(device)
            if mris is not None:
                mris = {k: v.to(device) for k, v in mris.items()}
            labels = labels.to(device)
            # bf16 autocast: ~halves activation memory (with flash attention this lets
            # full fine-tuning fit), but unlike fp16 it has fp32's exponent range so it
            # cannot overflow to NaN — and needs no GradScaler.
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ecgs, mris)
                loss   = F.binary_cross_entropy_with_logits(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            losses.append(loss.item())
            preds.extend(torch.sigmoid(logits.float()).detach().cpu().numpy())
            trues.extend(labels.cpu().numpy())
            if train and i % 20 == 0:
                print(f"  step {i}/{len(loader)}  loss={loss.item():.4f}", flush=True)
    auroc = roc_auc_score(trues, preds) if len(set(trues)) > 1 else float("nan")
    if return_preds:
        return np.mean(losses), auroc, trues, preds
    return np.mean(losses), auroc


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome",       required=True)
    parser.add_argument("--mode",          default="ecg_mri", choices=["ecg", "ecg_mri"])
    parser.add_argument("--training_type", default="senc_proj",
                        choices=["full", "senc_proj", "partial", "linear"])
    parser.add_argument("--unfreeze_blocks", type=int, default=2,
                        help="top MRI encoder blocks to unfreeze when --training_type partial")
    parser.add_argument("--pool",          default="mean", choices=["mean", "cls", "per_view"])
    parser.add_argument("--view_encoder",  default="none", choices=["none", "conv", "vit"],
                        help="must match the pretrained/contrastive checkpoint's per-view stage")
    parser.add_argument("--view_depth",    type=int, default=2)
    parser.add_argument("--n_sa_slices",   type=int, default=3)
    parser.add_argument("--cl_ckpt",       required=True)
    parser.add_argument("--ecg_ckpt",      required=True)
    parser.add_argument("--mri_dir",       required=True)
    parser.add_argument("--ecg_dir",       required=True)
    parser.add_argument("--csv_train",     required=True)
    parser.add_argument("--csv_val",       required=True)
    parser.add_argument("--csv_test",      required=True)
    parser.add_argument("--out_dir",       required=True)
    parser.add_argument("--lr",            type=float, default=5e-6)
    parser.add_argument("--lr_ecg",        type=float, default=None,
                        help="LR for ECG encoder+projection (defaults to --lr)")
    parser.add_argument("--lr_mri",        type=float, default=None,
                        help="LR for MRI encoder+projection (defaults to --lr)")
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--img_size",      type=int,   default=224)
    parser.add_argument("--n_frames",      type=int,   default=8)
    parser.add_argument("--embed_dim",     type=int,   default=384)
    parser.add_argument("--select_by",     choices=["val_loss", "val_auroc"], default="val_loss",
                        help="model-selection + early-stop monitor (val_loss = smoother/less-overfit)")
    parser.add_argument("--encoder_depth", type=int,   default=8, help="must match pretraining (base=12)")
    parser.add_argument("--encoder_heads", type=int,   default=6, help="must match pretraining (base=12)")
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--seed",          type=int,   default=1,
                        help="seed for reproducible init/shuffle/MRI-crop; set for the reproducible reruns")
    parser.add_argument("--eval_ckpt",     default="",
                        help="eval-only: load this downstream ckpt, dump per-sample result.csv on the "
                             "test set (id,y_true,y_pred) and skip training. For bootstrap CIs.")
    parser.add_argument("--eval_manifest", default="test", choices=["train", "valid", "test"],
                        help="which ECG manifest the eval (test) slot reads. Set to 'train' to dump "
                             "per-sample preds on the (paired) TRAIN set, e.g. to fit the +Risk glm. "
                             "Pair with --csv_test pointing at the matching label csv (MRI_train/<outc>.csv).")
    args = parser.parse_args()

    set_seed(args.seed)
    g = torch.Generator(); g.manual_seed(args.seed)

    lr_ecg = args.lr_ecg if args.lr_ecg is not None else args.lr
    lr_mri = args.lr_mri if args.lr_mri is not None else args.lr
    lr_head = max(lr_ecg, lr_mri)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  mode: {args.mode}  outcome: {args.outcome}  type: {args.training_type}  "
          f"lr_ecg={lr_ecg:.2e}  lr_mri={lr_mri:.2e}")

    train_ds = DownstreamDataset(args.csv_train, args.ecg_dir, args.mri_dir, "train",
                                 outcome_col=args.outcome, mode=args.mode,
                                 img_size=args.img_size, n_frames=args.n_frames)
    val_ds   = DownstreamDataset(args.csv_val,   args.ecg_dir, args.mri_dir, "valid",
                                 outcome_col=args.outcome, mode=args.mode,
                                 img_size=args.img_size, n_frames=args.n_frames)
    test_ds  = DownstreamDataset(args.csv_test,  args.ecg_dir, args.mri_dir, "test",
                                 outcome_col=args.outcome, mode=args.mode,
                                 img_size=args.img_size, n_frames=args.n_frames,
                                 manifest_split=args.eval_manifest)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g,
                              collate_fn=DownstreamDataset.collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, worker_init_fn=seed_worker,
                              collate_fn=DownstreamDataset.collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, worker_init_fn=seed_worker,
                              collate_fn=DownstreamDataset.collate_fn)

    mae_cfg = dict(
        img_size=args.img_size, n_frames=args.n_frames,
        tube_t=2, patch_h=16, patch_w=16,
        encoder_dim=args.embed_dim, encoder_depth=args.encoder_depth, encoder_heads=args.encoder_heads,
        decoder_dim=192, decoder_depth=4, decoder_heads=6,
        mask_ratio=0.9, norm_pix_loss=True,
        view_encoder=args.view_encoder, view_depth=args.view_depth,
        n_sa_slices=args.n_sa_slices,
    )
    model = ECGMRIDownstreamModel(args.cl_ckpt, args.ecg_ckpt, mae_cfg,
                                  mode=args.mode, training_type=args.training_type,
                                  unfreeze_blocks=args.unfreeze_blocks, pool=args.pool)
    model = nn.DataParallel(model)
    model.to(device)
    print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Eval-only: load a trained downstream checkpoint, dump per-sample test predictions, exit.
    if args.eval_ckpt:
        ck = torch.load(args.eval_ckpt, map_location=device, weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        (model.module if hasattr(model, "module") else model).load_state_dict(sd, strict=True)
        _, auc, trues, preds = run_epoch(test_loader, model, None, None, device, 0,
                                         train=False, return_preds=True)
        # test_loader is shuffle=False / no drop_last, so preds align with test_ds.samples order.
        ids = [str(s[0]) for s in test_ds.samples]
        n = min(len(ids), len(trues))
        df = pd.DataFrame({"id": ids[:n], "y_true": np.asarray(trues).ravel()[:n],
                           "y_pred": np.asarray(preds).ravel()[:n]})
        df.to_csv(os.path.join(args.out_dir, "result.csv"), index=False)
        print(f"[eval_only] {args.outcome} {args.mode} test AUROC={auc:.4f}  n={len(df)}  "
              f"pos={int(df.y_true.sum())}  wrote {args.out_dir}/result.csv")
        return

    optimizer = torch.optim.AdamW(
        [{"params": model.module.head.parameters(),           "lr": lr_head},
         {"params": model.module.ecg_projection.parameters(), "lr": lr_ecg},
         {"params": model.module.ecg_encoder.parameters(),    "lr": lr_ecg},
         {"params": model.module.mri_projection.parameters(), "lr": lr_mri},
         {"params": model.module.mri_encoder.parameters(),    "lr": lr_mri}],
        betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)

    n_batches = math.ceil(len(train_ds) / args.batch_size)
    scheduler = cosine_lr(optimizer, warmup_length=50, steps=args.epochs * n_batches)

    # Model selection + early stopping monitor: --select_by val_loss | val_auroc.
    # val_loss is smoother and its minimum tends to pick the least-overfit epoch
    # (higher mean test AUROC across prior runs); val_auroc aligns with the eval metric
    # but is noisier on small/imbalanced val and can land on overfit spikes.
    best_val_loss, best_val_auc = float("inf"), 0.0
    best_test_auc, best_epoch   = float("nan"), -1
    patience, patience_counter  = 3, 0
    step_offset = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_auc = run_epoch(train_loader, model, optimizer, scheduler, device, step_offset, train=True)
        val_loss,  val_auc  = run_epoch(val_loader,  model, None, None, device, 0, train=False)
        test_loss, test_auc = run_epoch(test_loader, model, None, None, device, 0, train=False)
        step_offset += n_batches

        print(f"\nEpoch {epoch+1}/{args.epochs}  "
              f"train loss={train_loss:.4f} auc={train_auc:.4f}  "
              f"val loss={val_loss:.4f} auc={val_auc:.4f}  "
              f"test loss={test_loss:.4f} auc={test_auc:.4f}  "
              f"time={(time.time()-t0)/60:.1f}min\n", flush=True)

        improved = (val_loss < best_val_loss) if args.select_by == "val_loss" else (val_auc > best_val_auc)
        if improved:
            best_val_loss, best_val_auc, best_test_auc, best_epoch = val_loss, val_auc, test_auc, epoch + 1
            patience_counter = 0
            m = model.module if hasattr(model, "module") else model
            torch.save({"epoch": epoch+1, "model": m.state_dict(),
                        "val_loss": val_loss, "val_auc": val_auc, "test_auc": test_auc},
                       os.path.join(args.out_dir, f"downstream_{args.outcome}_{args.mode}_best.pth"))
            print(f"  saved best (by {args.select_by}: val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  test_auc={test_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (monitor={args.select_by}).")
                break

    print(f"Done (select_by={args.select_by}). Best epoch {best_epoch}: "
          f"val_loss={best_val_loss:.4f}  val_auc={best_val_auc:.4f}  test_auc={best_test_auc:.4f}")

    results = {
        "outcome": args.outcome, "mode": args.mode, "training_type": args.training_type,
        "batch_size": args.batch_size, "lr": args.lr, "lr_ecg": lr_ecg, "lr_mri": lr_mri,
        "epochs": args.epochs, "cl_ckpt": args.cl_ckpt,
        "select_by": args.select_by, "best_epoch": best_epoch,
        "best_val_loss": best_val_loss, "best_val_auroc": best_val_auc,
        "test_auroc": best_test_auc,
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.out_dir}/results.json")


if __name__ == "__main__":
    main()
