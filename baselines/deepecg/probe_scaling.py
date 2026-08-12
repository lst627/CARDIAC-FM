"""
Determine the correct input SCALING for the DeepECG models on our ECGs.

Format is already settled empirically: the TorchScript model accepts (batch, 12, 2500) — channels
first — so our (12,5000) @ 500 Hz becomes feats[:, ::2] (exact 2x decimation to 250 Hz), no
transpose. The open question is amplitude units: the SL checkpoint is named "..._unscaled.pt" and
the docs never state units.

CORRECTNESS CHECK: afib_5y.pt is a genuine 5-year incident-AF model, so its zero-shot AUROC on our
af5 labels tells us whether the input is right. ~0.70+ = correct; ~0.50 = wrong scaling/shape.
Whichever variant is sane is the one to use for the real run.
"""
import torch, numpy as np, pandas as pd, os
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

UKB = "/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual"
HERE = os.path.dirname(os.path.abspath(__file__))
N = 2000

tsv = pd.read_csv(f"{UKB}/ECG_manifest_moretest/test.tsv", sep="\t")
root = tsv.columns[1]
lab = np.load(f"{UKB}/ECG_label/af5/y.npy").squeeze()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = torch.jit.load(f"{HERE}/afib_5y.pt", map_location=dev).eval().to(dev)

X, Y = [], []
for i in range(len(tsv)):
    if len(Y) >= N:
        break
    mat = loadmat(os.path.join(root, tsv.iloc[i, 0]))
    y = lab[int(mat["idx"].squeeze())]
    if not np.isfinite(y):
        continue
    X.append(mat["feats"][:, ::2])          # 500 Hz -> 250 Hz, keeps (12, 2500)
    Y.append(float(y))
X = np.stack(X); Y = np.array(Y)
print(f"device={dev}  n={len(Y)}  pos={int(Y.sum())}  raw range [{X.min():.1f}, {X.max():.1f}]", flush=True)


def run(x):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            b = torch.tensor(x[i:i + 64], dtype=torch.float32, device=dev)
            out.append(torch.sigmoid(m(b)).cpu().numpy().ravel())
    return np.concatenate(out)


variants = {
    "raw (.mat units)":      X,
    "/10":                   X / 10.0,
    "/100  (-> ~mV)":        X / 100.0,
    "/1000 (mV if uV)":      X / 1000.0,
    "*10":                   X * 10.0,
    "per-lead z-score":      (X - X.mean(2, keepdims=True)) / (X.std(2, keepdims=True) + 1e-8),
    "global z-score":        (X - X.mean((1, 2), keepdims=True)) / (X.std((1, 2), keepdims=True) + 1e-8),
}
print(f"\n{'scaling':24s} {'AUROC':>7s}   mean_p   (want AUROC >= ~0.70)", flush=True)
best = (None, 0.0)
for k, v in variants.items():
    p = run(v.astype(np.float32))
    a = roc_auc_score(Y, p)
    print(f"  {k:22s} {a:7.4f}   {p.mean():.4f}", flush=True)
    if a > best[1]:
        best = (k, a)
print(f"\nBEST: {best[0]}  AUROC={best[1]:.4f}", flush=True)
