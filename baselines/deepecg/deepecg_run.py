"""
DeepECG zero-shot AF benchmark: apply the published 5-year incident-AF models to our af5 data.

  DeepECG-SL  = heartwise/EfficientNetV2_AFIB_5y  (afib_5y.pt,      ~7 MB)
  DeepECG-SSL = heartwise/WCR_AFIB_5Y             (wcr_afib_5y.pt, ~2.2 GB)

Both are already fine-tuned for incident AF at 5 years — i.e. exactly our `af5` endpoint — so this
is a true ZERO-SHOT benchmark: no fine-tuning at all, the cleanest possible external comparison.
(They are AF-specific, so there is no hf5 arm.)

They load DIFFERENTLY: the SL checkpoint is TorchScript (torch.jit.load), while the SSL/WCR
checkpoint is a fairseq_signals `ecg_transformer_classifier` (build_model_from_checkpoint) — the
same framework our ECG-FM arms already use, so no new dependency.

INPUT RECIPE — settled empirically (probe_scaling.py / probe_wcr.py, 2026-07-23).
Both models want 250 Hz, i.e. feats[:, ::2] -> (12, 2500), channels-first, lead order
I,II,III,aVR,aVL,aVF,V1..V6 (matches our .mat). They differ in NORMALISATION:

  SL  (EfficientNetV2, TorchScript)   raw .mat units   UKB af5 AUROC 0.690
       (per-lead z gave 0.519 -> wrong; checkpoint is literally named "..._unscaled")
  SSL (WCR, fairseq_signals)          per-lead z-score UKB af5 AUROC 0.756
       (250Hz raw 0.725; the model is globally scale-invariant -- raw and /100 are bit-identical --
        but responds to per-lead scaling. 500 Hz gives only 0.60-0.61, confirming 250 Hz.)

Choosing each model's best input makes the BASELINE stronger, i.e. it is conservative for us.
Per the paper's Methods, CHS amplitudes were already rescaled to UKB reference values and MESA was
already comparable, so the same recipe applies to all three cohorts.

Writes the standard result.csv (id, y_true, y_pred) so the existing bootstrap/report/figure tooling
picks these up as additional baselines with no changes.

Usage: python deepecg_run.py --model afib_5y.pt --tag deepecg_sl
"""
import os, argparse
import numpy as np, pandas as pd, torch
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

EVAL = "/gpfs/projects/trend/bojun/multimodal_rep/eval"
UKB = "/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual"
CHS = "/gpfs/projects/trend/bojun/CHS_MESA/data_train_valid_test_individual_CHS"
MESA = "/gpfs/projects/trend/bojun/CHS_MESA/data_train_valid_test_individual_MESA"


def prep(feats, norm):
    """(12,5000) @500Hz raw -> (12,2500) @250Hz, with the model's required normalisation."""
    x = feats[:, ::2]
    if norm == "perlead_z":
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
    return x


def predict(model, tsv_dir, label_dir, split, dev, norm, fwd, bs=64):
    """returns DataFrame(id, y_true, y_pred) over the split; rows with no label are dropped."""
    tsv = pd.read_csv(f"{tsv_dir}/{split}.tsv", sep="\t")
    root = tsv.columns[1]
    lab = np.load(f"{label_dir}/y.npy").squeeze()
    ids, ys, buf, preds = [], [], [], []

    def flush():
        if not buf:
            return
        x = torch.tensor(np.stack(buf), dtype=torch.float32, device=dev)
        with torch.no_grad():
            o = fwd(model, x)
            preds.append(torch.sigmoid(o).float().cpu().numpy().reshape(len(buf), -1)[:, 0])
        buf.clear()

    for i in range(len(tsv)):
        fn = tsv.iloc[i, 0]
        mat = loadmat(os.path.join(root, fn))
        y = lab[int(mat["idx"].squeeze())]
        if not np.isfinite(y):
            continue                                  # unlabeled -> skip (matches our other arms)
        ids.append(fn.replace(".mat", "")); ys.append(float(y))
        buf.append(prep(mat["feats"], norm))          # 250 Hz + model-specific norm
        if len(buf) == bs:
            flush()
    flush()
    return pd.DataFrame({"id": ids, "y_true": ys, "y_pred": np.concatenate(preds)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="TorchScript .pt (afib_5y.pt | wcr_afib_5y.pt)")
    ap.add_argument("--tag", required=True, help="output arm name, e.g. deepecg_sl / deepecg_ssl")
    ap.add_argument("--loader", choices=["jit", "fairseq"], default="jit",
                    help="jit = TorchScript (EfficientNetV2); fairseq = fairseq_signals (WCR)")
    ap.add_argument("--norm", choices=["raw", "perlead_z"], default="raw")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, a.model)
    if a.loader == "jit":
        m = torch.jit.load(path, map_location=dev).eval().to(dev)
        fwd = lambda model, x: model(x)
    else:
        from fairseq_signals.models import build_model_from_checkpoint
        m = build_model_from_checkpoint(path).to(dev).eval()

        def fwd(model, x):
            out = model(source=x)
            if isinstance(out, dict):
                for k in ("out", "x", "logits", "encoder_out"):
                    if k in out:
                        return out[k]
            return out
    print(f"[{a.tag}] loaded {a.model} ({a.loader}, norm={a.norm}) on {dev}", flush=True)

    cells = [("UKB  test", f"{UKB}/ECG_manifest_moretest", f"{UKB}/ECG_label/af5", "test",
              f"{EVAL}/ukb_test/{a.tag}/af5"),
             ("CHS  zs",   f"{CHS}/ECG_manifest",          f"{CHS}/ECG_label/af5", "test",
              f"{EVAL}/zeroshot/CHS/{a.tag}/af5"),
             ("MESA zs",   f"{MESA}/ECG_manifest",         f"{MESA}/ECG_label/af5", "test",
              f"{EVAL}/zeroshot/MESA/{a.tag}/af5")]

    for name, tsv_dir, label_dir, split, save in cells:
        if not os.path.exists(f"{tsv_dir}/{split}.tsv"):
            print(f"  [skip] {name}: no {split}.tsv"); continue
        df = predict(m, tsv_dir, label_dir, split, dev, a.norm, fwd)
        os.makedirs(save, exist_ok=True)
        df.to_csv(f"{save}/result.csv", index=False)
        auc = roc_auc_score(df.y_true, df.y_pred) if df.y_true.nunique() > 1 else float("nan")
        print(f"  {name} af5: n={len(df)} pos={int(df.y_true.sum())} AUROC={auc:.4f} -> {save}/result.csv",
              flush=True)


if __name__ == "__main__":
    main()
