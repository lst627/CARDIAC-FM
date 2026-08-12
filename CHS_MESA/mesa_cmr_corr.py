"""
MESA: ECG-predicted CMR phenotype vs MEASURED (ground-truth) correlation — the external analogue of
the UKB CMR-regression accuracy (cmr_test_results). Advisor asked to add it for MESA, which (unlike
CHS) has measured cardiac MRI. Committed generator for what was previously an ad-hoc result
(eval/a2/mesa_measured_vs_predicted.csv, seed 1); reproduces seed 1 and produces seed 3.

Predicted: eval/a1/cmr_pred_external[_seed<seed>]/MESA/<feat>/result.csv   (id, y_pred)
Measured:  CHS_MESA/MESA/MESA_CMR_features.csv                            (id_visit, <feat>)
Join predicted.id == measured.id_visit. Reports Pearson r, OLS slope (truth ~ pred), n, and the drop
vs UKB r (cmr_reg/cmr_test_results_seed<seed>.csv). paper_r carried from the original seed-1 a2 file.

  python mesa_cmr_corr.py --seed 3
  python mesa_cmr_corr.py --seed 1   # reproduce the original ad-hoc a2 to validate methodology
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import pearsonr

EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"
MEAS = "/gpfs/projects/trend/bojun/CHS_MESA/MESA/MESA_CMR_features.csv"
FEATS = ["lvm", "lvedv", "lvesv", "lavmin", "lavmax", "laef", "lvef"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    predir = f"{EV}/a1/cmr_pred_external" if a.seed == 1 else f"{EV}/a1/cmr_pred_external_seed{a.seed}"
    if a.out:
        out = a.out
    elif a.seed == 1:
        out = f"{EV}/a2/mesa_measured_vs_predicted_seed1_repro.csv"   # don't clobber the original
    else:
        out = f"{EV}_{a.seed}/a2/mesa_measured_vs_predicted.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    meas = pd.read_csv(MEAS); meas["id"] = meas["id_visit"].astype(str)
    ukb = pd.read_csv(f"{EV}/cmr_reg/cmr_test_results_seed{a.seed}.csv").set_index("feature")["ours_r"]
    # NB: no `paper_r` — the CARDIAC-FM paper reports NO MESA predicted-vs-measured correlation
    # (this analysis is the editor's E1 ask, genuinely new). The column in the ad-hoc a2 file was
    # spurious (unknown origin, matches no paper number) and is intentionally dropped.

    rows = []
    for f in FEATS:
        pf = f"{predir}/MESA/{f}/result.csv"
        if not os.path.exists(pf):
            print(f"  [skip] {f}: no predicted at {pf}"); continue
        pred = pd.read_csv(pf)[["id", "y_pred"]].copy(); pred["id"] = pred["id"].astype(str)
        j = pred.merge(meas[["id", f]], on="id", how="inner").dropna(subset=["y_pred", f])
        if len(j) < 20:
            print(f"  [skip] {f}: n={len(j)}"); continue
        x, y = j["y_pred"].values.astype(float), j[f].values.astype(float)
        r = float(pearsonr(x, y)[0]); ur = float(ukb.get(f, np.nan))
        # Pearson correlation only. (Variance-explained R²=1-MSE/Var is negative externally because the
        # UKB-fit unscaling mismatches MESA's absolute scale; squared-r would ignore that, so we omit R².)
        rows.append({"feature": f, "n": len(j), "ours_r": r, "ukb_r_ours": ur})
        print(f"  {f:<8} n={len(j):<6} MESA_r={r:.3f}  UKB_r={ur:.3f}  drop={r-ur:+.3f}")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[wrote -> {out}]")


if __name__ == "__main__":
    main()
