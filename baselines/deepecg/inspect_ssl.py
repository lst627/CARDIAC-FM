"""
Inspect the DeepECG-SSL PRETRAINING backbone (SSL_pretrained.pt = base_ssl.pt, never fine-tuned on
any downstream task) so we can fine-tune it ourselves on UKB af5/hf5 — the same treatment ECG-FM
and ECGFounder got, making DeepECG-SSL a fair peer baseline (and giving an HF arm).

Determines: (a) how it loads, (b) its feature-extraction interface + hidden dim, (c) that a
forward pass on our 250Hz/per-lead-z input works. Also runs a quick FROZEN-feature linear probe on
af5 as a pipeline sanity check (not the final number — that comes from full fine-tuning).
"""
import os, numpy as np, pandas as pd, torch
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# --- repo path configuration (see docs/PATHS.md) ---
import os as _os, sys as _sys
_pd = _os.path.dirname(_os.path.abspath(__file__))
while _pd != "/" and not _os.path.isdir(_os.path.join(_pd, "common")):
    _pd = _os.path.dirname(_pd)
_sys.path.insert(0, _os.path.join(_pd, "common"))
from paths import P


HERE = os.path.dirname(os.path.abspath(__file__))
UKB = P("UKB_ECG_ROOT")
CKPT = f"{HERE}/SSL_pretrained.pt"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prep(feats):
    x = feats[:, ::2]                                  # 500 -> 250 Hz
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)   # per-lead z


# --- load -------------------------------------------------------------------
print("=== raw checkpoint structure ===", flush=True)
raw = torch.load(CKPT, map_location="cpu", weights_only=False)
print("top keys:", list(raw.keys())[:12] if isinstance(raw, dict) else type(raw), flush=True)
if isinstance(raw, dict) and "cfg" in raw:
    for sec in ("model", "task", "criterion"):
        v = raw["cfg"].get(sec) if hasattr(raw["cfg"], "get") else None
        print(f"  cfg.{sec}._name = {v.get('_name') if hasattr(v,'get') else v}", flush=True)
    sd = raw.get("model", {})
    ks = list(sd.keys()) if hasattr(sd, "keys") else []
    print(f"  state_dict n={len(ks)} first={ks[:3]} last={ks[-3:]}", flush=True)

print("\n=== build_model_from_checkpoint ===", flush=True)
from fairseq_signals.models import build_model_from_checkpoint
model = build_model_from_checkpoint(CKPT).to(dev).eval()
print("built:", type(model).__name__, flush=True)

# --- feature interface probe ------------------------------------------------
x = torch.randn(2, 12, 2500, device=dev)
feat_dim = None
with torch.no_grad():
    if hasattr(model, "extract_features"):
        try:
            f = model.extract_features(source=x, padding_mask=None)
            xf = f["x"] if isinstance(f, dict) else f
            print(f"extract_features -> x shape {tuple(xf.shape)}", flush=True)
            feat_dim = xf.shape[-1]
        except Exception as e:
            print("extract_features failed:", str(e).splitlines()[0][:120], flush=True)
    try:
        o = model(source=x)
        print("forward(source=) ->", {k: tuple(v.shape) for k, v in o.items()} if isinstance(o, dict)
              else tuple(o.shape), flush=True)
    except Exception as e:
        print("forward(source=) failed:", str(e).splitlines()[0][:120], flush=True)
print("feat_dim =", feat_dim, flush=True)


# --- frozen-feature linear probe on af5 (pipeline sanity) -------------------
def embed(files, root):
    out = []
    for i in range(0, len(files), 32):
        buf = np.stack([prep(loadmat(os.path.join(root, f))["feats"]) for f in files[i:i+32]])
        with torch.no_grad():
            f = model.extract_features(source=torch.tensor(buf, dtype=torch.float32, device=dev),
                                       padding_mask=None)
            xf = f["x"] if isinstance(f, dict) else f
            emb = xf.mean(1)                            # mean-pool over time
        out.append(emb.cpu().numpy())
    return np.concatenate(out)


def collect(split, n):
    tsv = pd.read_csv(f"{UKB}/ECG_manifest_moretest/{split}.tsv", sep="\t")
    root = tsv.columns[1]; lab = np.load(f"{UKB}/ECG_label/af5/y.npy").squeeze()
    fs, ys = [], []
    for i in range(len(tsv)):
        if len(ys) >= n:
            break
        idx = int(loadmat(os.path.join(root, tsv.iloc[i, 0]))["idx"].squeeze())
        y = lab[idx]
        if np.isfinite(y):
            fs.append(tsv.iloc[i, 0]); ys.append(float(y))
    return fs, np.array(ys), root


if feat_dim:
    print("\n=== frozen-feature linear probe (af5, sanity only) ===", flush=True)
    ftr, ytr, root = collect("valid", 1500)            # small train fold
    fte, yte, _ = collect("test", 1500)
    Etr, Ete = embed(ftr, root), embed(fte, root)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(Etr, ytr)
    p = clf.predict_proba(Ete)[:, 1]
    print(f"linear-probe af5 AUROC = {roc_auc_score(yte, p):.4f}  (full fine-tune will differ)", flush=True)
print("\nDONE", flush=True)
