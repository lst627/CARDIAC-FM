"""
Fine-tune the DeepECG-SSL backbone (SSL_pretrained.pt, wav2vec2_cmsc, never fine-tuned on any
downstream task) on our UKB af5/hf5 — the SAME treatment ECG-FM and ECGFounder got, so DeepECG-SSL
becomes a fair peer baseline (and gains an HF arm the off-the-shelf AF model could not provide).

Matched protocol (== ECGFounder): seed 1, lr 5e-6, AdamW, BCEWithLogits, select on val AUROC,
patience-3 early stop, 20-epoch cap. Preprocessing (== DeepECG probe): 250 Hz (feats[:, ::2]) +
per-lead z-score.

Head: encoder.extract_features -> mean-pool over time -> Linear(768,1). The whole encoder is
trainable (full fine-tune), matching how we fine-tuned ECG-FM / ECGFounder.

  train:  python deepssl_finetune.py train --outcome af5 --ckpt_out .../af5/best.pth
  test:   python deepssl_finetune.py test  --outcome af5 --split test --ckpt_in .../af5/best.pth \
                 --ecg_tsv_dir <dir> --label_dir <dir/af5> --save_dir <eval/.../deepssl/af5>
"""
import os, sys, argparse, random
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from fairseq_signals.models import build_model_from_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
BACKBONE = f"{HERE}/SSL_pretrained.pt"
UKB = "/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual"


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def prep(feats):
    x = feats[:, ::2]                                   # 500 -> 250 Hz
    if os.environ.get("SSL_NORM", "zscore") == "raw":   # diagnostic: raw units vs per-lead z-score
        return x.astype(np.float32)
    return ((x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)).astype(np.float32)


class DS(Dataset):
    def __init__(self, ecg_tsv_dir, label_dir, split):
        t = pd.read_csv(f"{ecg_tsv_dir}/{split}.tsv", sep="\t")
        self.files = t.iloc[:, 0].tolist(); self.root = t.columns[1]
        self.lab = np.load(f"{label_dir}/y.npy").squeeze()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        mat = loadmat(os.path.join(self.root, self.files[i]))
        y = self.lab[int(mat["idx"].squeeze())]
        return (torch.from_numpy(prep(mat["feats"])),
                torch.tensor(np.float32(y) if y is not None else np.float32("nan")),
                self.files[i])


class DeepSSLClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = build_model_from_checkpoint(BACKBONE)   # wav2vec2_cmsc, trainable
        self.head = nn.Linear(768, 1)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        f = self.enc.extract_features(source=x, padding_mask=None)
        h = f["x"] if isinstance(f, dict) else f          # (B, T, 768)
        h = h.mean(dim=1)                                 # mean-pool over time
        return self.head(h).squeeze(-1)                   # (B,)


def loaders(d, lab, split, bs, shuffle, seed=1):
    g = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(DS(d, lab, split), batch_size=bs, shuffle=shuffle,
                      num_workers=int(os.environ.get("W", "8")), generator=g)


def auroc(y, p):
    m = np.isfinite(y)
    return roc_auc_score(y[m], p[m]) if m.sum() and len(np.unique(y[m])) > 1 else np.nan


def infer(model, loader, dev):
    model.eval(); ys, ps, fs = [], [], []
    with torch.no_grad():
        for x, y, f in loader:
            ps.append(torch.sigmoid(model(x.to(dev))).cpu().numpy().ravel())
            ys.append(y.numpy().ravel()); fs.extend(f)
    return np.concatenate(ys), np.concatenate(ps), fs


def run_train(a, dev):
    seed_all(a.seed)
    tr = loaders(a.ecg_tsv_dir, a.label_dir, "train", a.batch_size, True, a.seed)
    va = loaders(a.ecg_tsv_dir, a.label_dir, "valid", a.batch_size, False)
    model = DeepSSLClassifier().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, since = -1.0, None, 0
    for ep in range(a.epochs):
        model.train()
        for x, y, _ in tr:
            y = y.to(dev); mask = ~torch.isnan(y)
            if mask.sum() == 0:
                continue
            opt.zero_grad()
            loss = lossf(model(x.to(dev))[mask], y[mask]); loss.backward(); opt.step()
        yv, pv, _ = infer(model, va, dev); va_auc = auroc(yv, pv)
        print(f"  epoch {ep}: val AUROC = {va_auc:.4f}", flush=True)
        if np.isfinite(va_auc) and va_auc > best:
            best, best_state, since = va_auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            since += 1
            if since >= a.patience:
                print(f"  early stop @ {ep} (best {best:.4f})", flush=True); break
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    os.makedirs(os.path.dirname(a.ckpt_out), exist_ok=True)
    torch.save(best_state, a.ckpt_out)
    print(f"[best val AUROC {best:.4f}] -> {a.ckpt_out}", flush=True)


def run_test(a, dev):
    model = DeepSSLClassifier().to(dev)
    model.load_state_dict(torch.load(a.ckpt_in, map_location=dev))
    y, p, f = infer(model, loaders(a.ecg_tsv_dir, a.label_dir, a.split, a.batch_size, False), dev)
    df = pd.DataFrame({"id": [x.replace(".mat", "") for x in f], "y_true": y, "y_pred": p})
    df = df[np.isfinite(df.y_true)].reset_index(drop=True)
    os.makedirs(a.save_dir, exist_ok=True); df.to_csv(f"{a.save_dir}/result.csv", index=False)
    print(f"[{a.split}] n={len(df)} AUROC={auroc(df.y_true.values, df.y_pred.values):.4f} "
          f"-> {a.save_dir}/result.csv", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "test"])
    ap.add_argument("--outcome", required=True)
    ap.add_argument("--ecg_tsv_dir", default=f"{UKB}/ECG_manifest_moretest")
    ap.add_argument("--label_dir", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt_out"); ap.add_argument("--ckpt_in"); ap.add_argument("--save_dir")
    a = ap.parse_args()
    if a.label_dir is None:
        a.label_dir = f"{UKB}/ECG_label/{a.outcome}"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, flush=True)
    if a.mode == "train":
        assert a.ckpt_out; run_train(a, dev)
    else:
        assert a.ckpt_in and a.save_dir; run_test(a, dev)
