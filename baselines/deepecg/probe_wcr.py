"""
Probe the DeepECG-SSL (WCR) checkpoint: it is a fairseq_signals `ecg_transformer_classifier`
(NOT TorchScript, unlike the EfficientNetV2 models), so it builds with the same
build_model_from_checkpoint helper our ECG-FM arms already use.

Determines: (a) that it builds and loads, (b) the accepted input shape / sample length,
(c) the correct amplitude scaling — validated by zero-shot AUROC on our af5 labels
(this checkpoint is already fine-tuned for incident AF at 5 years, so ~0.70 = right, ~0.50 = wrong).
"""
import os, numpy as np, pandas as pd, torch
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from fairseq_signals.models import build_model_from_checkpoint

UKB = "/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual"
HERE = os.path.dirname(os.path.abspath(__file__))
N = 1500

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model_from_checkpoint(os.path.join(HERE, "wcr_afib_5y.pt")).to(dev).eval()
print("built OK:", type(model).__name__, flush=True)


def fwd(x):
    """try the plausible fairseq_signals call signatures; return logits or raise."""
    with torch.no_grad():
        try:
            out = model(source=x)
        except TypeError:
            out = model(**{"source": x})
    if isinstance(out, dict):
        for k in ("out", "x", "logits", "encoder_out"):
            if k in out:
                out = out[k]; break
    return out


# --- (b) input length probe -------------------------------------------------
for T in (2500, 5000):
    try:
        o = fwd(torch.randn(2, 12, T, device=dev))
        print(f"  ACCEPTS (2,12,{T}) -> {tuple(o.shape)}", flush=True)
    except Exception as e:
        print(f"  rejects (2,12,{T}): {str(e).splitlines()[0][:110]}", flush=True)

# --- load real data ---------------------------------------------------------
tsv = pd.read_csv(f"{UKB}/ECG_manifest_moretest/test.tsv", sep="\t")
root = tsv.columns[1]
lab = np.load(f"{UKB}/ECG_label/af5/y.npy").squeeze()
X5, Y = [], []
for i in range(len(tsv)):
    if len(Y) >= N:
        break
    mat = loadmat(os.path.join(root, tsv.iloc[i, 0]))
    y = lab[int(mat["idx"].squeeze())]
    if not np.isfinite(y):
        continue
    X5.append(mat["feats"]); Y.append(float(y))     # keep full 500 Hz; decimate per-variant below
X5 = np.stack(X5); Y = np.array(Y)
print(f"\nn={len(Y)} pos={int(Y.sum())} raw range [{X5.min():.0f},{X5.max():.0f}]", flush=True)


def run(x):
    out = []
    for i in range(0, len(x), 32):
        b = torch.tensor(x[i:i + 32], dtype=torch.float32, device=dev)
        o = fwd(b)
        out.append(torch.sigmoid(o).float().cpu().numpy().reshape(len(b), -1)[:, 0])
    return np.concatenate(out)


X25 = X5[:, :, ::2]                                  # 250 Hz variant
variants = {
    "500Hz raw (12,5000)":   X5,
    "250Hz raw (12,2500)":   X25,
    "500Hz per-lead z":      (X5 - X5.mean(2, keepdims=True)) / (X5.std(2, keepdims=True) + 1e-8),
    "250Hz per-lead z":      (X25 - X25.mean(2, keepdims=True)) / (X25.std(2, keepdims=True) + 1e-8),
    "500Hz /100":            X5 / 100.0,
    "250Hz /100":            X25 / 100.0,
}
print(f"\n{'variant':26s} {'AUROC':>7s}   mean_p", flush=True)
best = (None, 0.0)
for k, v in variants.items():
    try:
        p = run(v.astype(np.float32))
        a = roc_auc_score(Y, p)
        print(f"  {k:24s} {a:7.4f}   {p.mean():.4f}", flush=True)
        if a > best[1]:
            best = (k, a)
    except Exception as e:
        print(f"  {k:24s} FAILED: {str(e).splitlines()[0][:90]}", flush=True)
print(f"\nBEST: {best[0]}  AUROC={best[1]:.4f}", flush=True)
