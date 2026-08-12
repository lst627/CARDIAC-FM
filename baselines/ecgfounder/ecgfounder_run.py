"""
ECGFounder baseline harness — fine-tune on UKB, infer on UKB test + CHS/MESA zero-shot.

Produces result.csv (id, y_true, y_pred) in the SAME layout as our ECG-FM / CL arms, so the
existing bootstrap / summary_report tooling compares it with no changes.

This file owns every part of the pipeline that is OURS (manifest reading, idx->label join,
train loop, val-AUROC model selection, build_result). It leaves exactly TWO integration points
to fill from the cloned ECGFounder repo — both marked `# >>> ECGFounder INTEGRATION`:

  (A) build_model(n_classes)  -> their net1d loader + pretrained checkpoint
  (B) preprocess(feats)       -> their dataset.py signal transform (filter + z-score)

Data contract (verified against CHS_MESA/scripts/ecg_test.py):
  - manifest {split}.tsv : col0 = "<file>.mat", the column-1 HEADER = the .mat root dir
  - each .mat            : feats (12,5000) float64 @ 500 Hz, plus int `idx`
  - label               : {label_dir}/y.npy, full-cohort array; this sample's label = y[idx]
  - id                  : filename without ".mat"
  - NaN labels are skipped in train/val and dropped from result.csv

Usage
  # train on UKB (writes best.pth chosen by val AUROC)
  python ecgfounder_run.py train --outcome af5 \
      --ecg_tsv_dir  $UKB/ECG_manifest_moretest \
      --label_dir    $UKB/ECG_label/af5 \
      --ckpt_out     $OUT/ecgfounder/af5/best.pth \
      --pretrained   $HOME/ECGFounder/checkpoint/12_lead_ECGFounder.pth

  # infer on any split/cohort (writes result.csv)
  python ecgfounder_run.py test  --outcome af5 --split test \
      --ecg_tsv_dir  $UKB/ECG_manifest_moretest --label_dir $UKB/ECG_label/af5 \
      --ckpt_in      $OUT/ecgfounder/af5/best.pth \
      --save_dir     $EVAL/ukb_test/ecgfounder/af5
"""
import os, sys, argparse, random
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score


def seed_everything(seed):
    """Make the fine-tune reproducible: fresh-head init, shuffle order, and any torch RNG use."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True        # remove cuDNN conv nondeterminism (job is short)
    torch.backends.cudnn.benchmark = False

# Point this at your ECGFounder clone so `import net1d` works.
ECGFOUNDER_REPO = os.environ.get("ECGFOUNDER_REPO", os.path.expanduser("~/ECGFounder"))
sys.path.insert(0, ECGFOUNDER_REPO)


# ----------------------------------------------------------------------------- (B) preprocessing
# Wired to ECGFounder's own dataset.py + util.filter_bandpass (verified against the repo, 2026-07-22).
#   - filter_bandpass: 50 Hz notch (iirnotch Q=30) + 0.67-40 Hz Butterworth order-4, per lead
#   - z-score: GLOBAL over the whole (12,5000) array (their z_score_normalization), NOT per-lead
# LEAD ORDER: their dataset swaps aVF/aVL because their raw feed (MIMIC/HEEDB) is
#   I,II,III,aVR,aVF,aVL,V1..V6. UKB .mat feats are already standard I,II,III,aVR,aVL,aVF,V1..V6,
#   so NO swap is applied. If a lead-order check ever shows ours is MIMIC-ordered, set SWAP_AVF_AVL=1.
SWAP_AVF_AVL = int(os.environ.get("ECGF_SWAP_AVF_AVL", "0"))
from util import filter_bandpass                     # noqa: E402  (ECGFounder repo)


def preprocess(feats):
    """feats: (12, 5000) float64 @ 500 Hz -> (12, 5000) float32, exactly as ECGFounder trains."""
    x = np.asarray(feats, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0)
    if SWAP_AVF_AVL:                                  # indices 4,5 = aVL,aVF in standard order
        x = x.copy(); x[[4, 5]] = x[[5, 4]]
    x = filter_bandpass(x, 500)                       # their exact filter
    x = (x - x.mean()) / (x.std() + 1e-8)             # their GLOBAL z-score
    return x.astype(np.float32)


# ----------------------------------------------------------------------------- (A) model
def build_model(n_classes, pretrained_path, device):
    """Load ECGFounder Net1D backbone + a fresh Linear(1024, n_classes) head.

    Replicates finetune_model.ft_12lead_ECGFounder exactly, except torch.load uses
    weights_only=False (the checkpoint stores numpy scalars; torch>=2.6 defaults to True and would
    refuse it). Verified: checkpoint['state_dict'], no 'module.' prefix, dense = Linear(1024,150)."""
    from net1d import Net1D                           # noqa: E402  (ECGFounder repo)
    model = Net1D(in_channels=12, base_filters=64, ratio=1,
                  filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                  m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2,
                  groups_width=16, verbose=False, use_bn=False, use_do=False, n_classes=n_classes)
    sd = torch.load(pretrained_path, map_location=device, weights_only=False)["state_dict"]
    sd = {k: v for k, v in sd.items() if not k.startswith("dense.")}   # drop 150-class head
    model.load_state_dict(sd, strict=False)                           # backbone only
    model.dense = nn.Linear(model.dense.in_features, n_classes)       # fresh binary head
    return model.to(device)


# ----------------------------------------------------------------------------- dataset (OURS)
class ECGFounderDataset(Dataset):
    def __init__(self, ecg_tsv_dir, label_dir, split):
        tsv = pd.read_csv(f"{ecg_tsv_dir}/{split}.tsv", sep="\t")
        self.files = tsv.iloc[:, 0].tolist()
        self.root = tsv.columns[1]                    # column-1 header = .mat dir
        self.label = np.load(f"{label_dir}/y.npy").squeeze()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        mat = loadmat(os.path.join(self.root, self.files[i]))
        x = preprocess(mat["feats"])                  # (12, 5000) float32
        idx = int(mat["idx"].squeeze())
        y = self.label[idx]
        y = np.float32(y) if y is not None else np.float32("nan")
        return torch.from_numpy(x), torch.tensor(y), self.files[i], idx


def loaders(ecg_tsv_dir, label_dir, split, bs, shuffle, seed=1):
    ds = ECGFounderDataset(ecg_tsv_dir, label_dir, split)
    nw = int(os.environ.get("ECGF_WORKERS", "8"))   # per-sample scipy filtering is the bottleneck
    g = torch.Generator().manual_seed(seed) if shuffle else None   # deterministic shuffle order
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw, drop_last=False, generator=g)


# ----------------------------------------------------------------------------- eval / result
def infer(model, loader, device):
    """returns per-sample (file, idx, y_true, y_pred) for ALL rows (NaN labels kept -> dropped later)."""
    model.eval()
    files, idxs, ys, ps = [], [], [], []
    with torch.no_grad():
        for x, y, f, idx in loader:
            logit = model(x.to(device)).squeeze(-1)
            p = torch.sigmoid(logit).cpu().numpy().ravel()
            ps.extend(p); ys.extend(y.numpy().ravel())
            files.extend(f); idxs.extend([int(v) for v in idx])
    return pd.DataFrame({"id": [f.replace(".mat", "") for f in files],
                         "idx": idxs, "y_true": ys, "y_pred": ps})


def auroc(df):
    m = df["y_true"].notna() & df["y_pred"].notna()
    return roc_auc_score(df.loc[m, "y_true"], df.loc[m, "y_pred"])


# ----------------------------------------------------------------------------- train
def run_train(args, device):
    seed_everything(args.seed)                       # BEFORE build_model (fresh head init) and shuffle
    tr = loaders(args.ecg_tsv_dir, args.label_dir, "train", args.batch_size, True, seed=args.seed)
    va = loaders(args.ecg_tsv_dir, args.label_dir, "valid", args.batch_size, False)
    model = build_model(1, args.pretrained, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, best_state, since = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        for x, y, _, _ in tr:
            y = y.to(device)
            mask = ~torch.isnan(y)
            if mask.sum() == 0:
                continue
            logit = model(x.to(device)).squeeze(-1)
            loss = loss_fn(logit[mask], y[mask])
            opt.zero_grad(); loss.backward(); opt.step()
        va_auc = auroc(infer(model, va, device))
        print(f"  epoch {ep}: val AUROC = {va_auc:.4f}", flush=True)
        if va_auc > best_auc:                          # select on val AUROC (matches our protocol)
            best_auc, best_state, since = va_auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:                                          # early stop: keeps the selected model on the stable
            since += 1                                 # plateau instead of a late val-AUROC noise spike
            if since >= args.patience:
                print(f"  early stop @ epoch {ep}: no val improvement for {args.patience} epochs "
                      f"(best {best_auc:.4f})", flush=True)
                break
    if best_state is None:                           # rare few-shot outcome: val AUROC never defined
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [warn] val AUROC never computable (too few val positives) -> saving last epoch")
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    torch.save(best_state, args.ckpt_out)
    with open(os.path.join(os.path.dirname(args.ckpt_out), "best_val.txt"), "w") as fh:
        fh.write(f"{best_auc:.6f}\n")               # sidecar for automated lr selection
    print(f"[best val AUROC {best_auc:.4f}] -> {args.ckpt_out}")


# ----------------------------------------------------------------------------- test
def run_test(args, device):
    model = build_model(1, args.pretrained, device)
    model.load_state_dict(torch.load(args.ckpt_in, map_location=device))
    ld = loaders(args.ecg_tsv_dir, args.label_dir, args.split, args.batch_size, False)
    df = infer(model, ld, device)
    out = df.loc[df["y_true"].notna(), ["id", "y_true", "y_pred"]].reset_index(drop=True)
    os.makedirs(args.save_dir, exist_ok=True)
    out.to_csv(f"{args.save_dir}/result.csv", index=False)
    print(f"[{args.split}] n={len(out)}  AUROC={auroc(out):.4f}  -> {args.save_dir}/result.csv")


# ----------------------------------------------------------------------------- cli
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "test"])
    ap.add_argument("--outcome", required=True)
    ap.add_argument("--ecg_tsv_dir", required=True)
    ap.add_argument("--label_dir", required=True)
    ap.add_argument("--pretrained", default=os.path.join(ECGFOUNDER_REPO, "checkpoint/12_lead_ECGFounder.pth"))
    ap.add_argument("--seed", type=int, default=1)         # fixed for reproducibility (matches our seed labels)
    ap.add_argument("--lr", type=float, default=1e-4)      # ECGFounder finetune lr; sweep if needed
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)     # early stop after N epochs w/o val improvement
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--split", default="test")            # test mode
    ap.add_argument("--ckpt_out")                          # train mode
    ap.add_argument("--ckpt_in")                           # test mode
    ap.add_argument("--save_dir")                          # test mode
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)
    if args.mode == "train":
        assert args.ckpt_out, "--ckpt_out required for train"
        run_train(args, dev)
    else:
        assert args.ckpt_in and args.save_dir, "--ckpt_in and --save_dir required for test"
        run_test(args, dev)
