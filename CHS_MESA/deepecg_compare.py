"""
DeepECG comparison — published 5-year AF models applied ZERO-SHOT vs our arms (af5 only).

Kept as its OWN report, deliberately NOT folded into summary_report_0719's "beats N/5" tally,
because the comparison is not like-for-like:

  our CL / ECG-FM / ECGFounder : fine-tuned on UKB (ECGFounder + ours with a matched protocol)
  DeepECG-SL / DeepECG-SSL     : NEVER saw our data; published weights, frozen, applied as-is

DeepECG is also AF-only (no HF checkpoint exists), so there is no hf5 arm here.

Both metrics are rank-based (AUROC and average-precision/AUPRC are invariant to monotone rescaling),
so the DeepECG models' known miscalibration on our cohorts does NOT distort either number.

Per cell every arm is inner-joined on `id` -> one common population, and the paired bootstrap
compares OUR arm against each DeepECG model on the same resample.

Usage: python deepecg_compare.py [--B 2000]
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

# --- repo path configuration (see docs/PATHS.md) ---
import os as _os, sys as _sys
_pd = _os.path.dirname(_os.path.abspath(__file__))
while _pd != "/" and not _os.path.isdir(_os.path.join(_pd, "common")):
    _pd = _os.path.dirname(_pd)
_sys.path.insert(0, _os.path.join(_pd, "common"))
from paths import P


EV = P("EVAL_ROOT")
OUT = f"{EV}/deepecg"


def fauc(y, s):
    npos = int(y.sum()); nneg = len(y) - npos
    if npos < 3 or nneg < 3:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def aupr(y, s):
    return average_precision_score(y, s) if 0 < int(y.sum()) < len(y) else np.nan


def load(f):
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)[["id", "y_true", "y_pred"]].dropna()
    d["id"] = d["id"].astype(str)
    return d


def cells():
    """(label, {arm: path}, our_arms_to_test). Two DeepECG regimes are shown per outcome:
      *(zs)*  = published DeepECG AF model applied off-the-shelf (af5 only; their fine-tuning)
      *(ft)*  = DeepECG-SSL backbone fine-tuned BY US on UKB, matched to ECGFounder (af5 + hf5)
    Only the (ft) arm is a like-for-like peer of our model / ECG-FM / ECGFounder."""
    out = []
    for o in ["af5", "hf5"]:
        ukb = {
            "CL(ECG) [ours]":     f"{EV}/ukb_test/m75_ecgfull/ecg/{o}/result.csv",
            "CL(ECG+MRI) [ours]": f"{EV}/ukb_test/m75_ecgfull/ecg_mri/{o}/result.csv",
            "ECG-FM (ft)":        f"{EV}/ecgfm_ukb_paired/seed1/test/{o}/result.csv",
            "ECGFounder (ft)":    f"{EV}/ukb_test/ecgfounder/{o}/result.csv",
            "DeepECG-SSL (ft)":   f"{EV}/ukb_test/deepssl_ft/{o}/result.csv",
        }
        if o == "af5":                              # off-the-shelf AF models exist for af5 only
            ukb["DeepECG-SL (zs)"] = f"{EV}/ukb_test/deepecg_sl/{o}/result.csv"
            ukb["DeepECG-SSL (zs)"] = f"{EV}/ukb_test/deepecg_ssl/{o}/result.csv"
        out.append((f"UKB test — {o}", ukb, ["CL(ECG) [ours]", "CL(ECG+MRI) [ours]"]))
        for coh in ["CHS", "MESA"]:
            d = {
                "CL(ECG) [ours]":   f"{EV}/zeroshot/{coh}/m75_ecgfull/{o}/result.csv",
                "ECG-FM (ft)":      f"{EV}/ecgfm_zeroshot_provided/seed1/zeroshot/{coh}/{o}/result.csv",
                "ECGFounder (ft)":  f"{EV}/zeroshot/{coh}/ecgfounder/{o}/result.csv",
                "DeepECG-SSL (ft)": f"{EV}/zeroshot/{coh}/deepssl_ft/{o}/result.csv",
            }
            if o == "af5":
                d["DeepECG-SL (zs)"] = f"{EV}/zeroshot/{coh}/deepecg_sl/{o}/result.csv"
                d["DeepECG-SSL (zs)"] = f"{EV}/zeroshot/{coh}/deepecg_ssl/{o}/result.csv"
            out.append((f"{coh} zero-shot — {o}", d, ["CL(ECG) [ours]"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(OUT, exist_ok=True)

    L = ["# DeepECG vs our arms — two regimes", "",
         "**(ft) = the fair, like-for-like comparison.** The DeepECG-SSL backbone "
         "(`SSL_pretrained.pt`, wav2vec2-CMSC, never fine-tuned on any task) was fine-tuned BY US on "
         "UKB af5/hf5 with the SAME protocol as ECGFounder (seed 1, lr 5e-6, patience-3 early stop), "
         "then applied zero-shot to CHS/MESA — exactly how our CL arms / ECG-FM / ECGFounder are "
         "treated. This is a peer baseline and covers **both** outcomes.", "",
         "**(zs) = off-the-shelf published AF models** (`EfficientNetV2_AFIB_5y` = SL, `WCR_AFIB_5Y` "
         "= SSL) applied as-is: THEIR fine-tuning on THEIR (North-American) data, no adaptation to "
         "our cohorts. af5 only (no HF checkpoint exists). Useful as a deployable-model benchmark but "
         "NOT like-for-like — their external AF edge reflects fine-tuning-domain match, not "
         "pretraining: the same backbone fine-tuned on UKB (the (ft) arm) transfers far worse.", "",
         "Both DeepECG variants are kept OUT of the main report's `beats N/5` tally.",
         "Both metrics are rank-based, so DeepECG's miscalibration on our cohorts does not affect them.",
         f"Stratified bootstrap 95% CI, B={a.B}; Δ and p are PAIRED (same resample) vs our arm.", ""]
    rows = []

    for label, arms, ours_list in cells():
        loaded = {k: load(v) for k, v in arms.items()}
        miss = [k for k, v in loaded.items() if v is None]
        loaded = {k: v for k, v in loaded.items() if v is not None}
        if not loaded:
            continue
        base = list(loaded.values())[0][["id", "y_true"]]
        m = base.copy()
        for k, d in loaded.items():
            m = m.merge(d[["id", "y_pred"]].rename(columns={"y_pred": k}), on="id")
        m = m.dropna()
        y = m["y_true"].values.astype(int)
        names = list(loaded)

        # one shared resample -> CI for every arm + paired deltas
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        dr_auc = {k: np.empty(a.B) for k in names}
        dr_apr = {k: np.empty(a.B) for k in names}
        for b in range(a.B):
            idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
            yy = y[idx]
            for k in names:
                v = m[k].values[idx]
                dr_auc[k][b] = fauc(yy, v); dr_apr[k][b] = aupr(yy, v)
        pt_auc = {k: fauc(y, m[k].values) for k in names}
        pt_apr = {k: aupr(y, m[k].values) for k in names}

        L += [f"\n## {label}  (n={len(y)}, events={int(y.sum())})\n"]
        if miss:
            L.append(f"> missing arms: {', '.join(miss)}\n")
        L += ["| arm | AUROC [95% CI] | AUPRC [95% CI] |", "|---|---|---|"]
        for k in names:
            ca = np.nanpercentile(dr_auc[k], [2.5, 97.5]); cp = np.nanpercentile(dr_apr[k], [2.5, 97.5])
            L.append(f"| {k} | {pt_auc[k]:.4f} [{ca[0]:.4f}, {ca[1]:.4f}] "
                     f"| {pt_apr[k]:.4f} [{cp[0]:.4f}, {cp[1]:.4f}] |")
            rows.append(dict(cell=label, arm=k, n=len(y), events=int(y.sum()),
                             auroc=pt_auc[k], auroc_lo=ca[0], auroc_hi=ca[1],
                             auprc=pt_apr[k], auprc_lo=cp[0], auprc_hi=cp[1]))

        L += ["", "| comparison | ΔAUROC [95% CI] | p | ΔAUPRC [95% CI] | p |", "|---|---|---|---|---|"]
        for ours in ours_list:
            if ours not in names:
                continue
            for dk in [k for k in names if k.startswith("DeepECG")]:
                da = dr_auc[ours] - dr_auc[dk]; dp = dr_apr[ours] - dr_apr[dk]
                pa = min(2 * min(np.nanmean(da >= 0), np.nanmean(da <= 0)), 1.0)
                pp = min(2 * min(np.nanmean(dp >= 0), np.nanmean(dp <= 0)), 1.0)
                qa = np.nanpercentile(da, [2.5, 97.5]); qp = np.nanpercentile(dp, [2.5, 97.5])
                st = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
                L.append(f"| {ours} − {dk} | {pt_auc[ours]-pt_auc[dk]:+.4f} [{qa[0]:+.4f}, {qa[1]:+.4f}] "
                         f"| {pa:.4f} {st(pa)} | {pt_apr[ours]-pt_apr[dk]:+.4f} [{qp[0]:+.4f}, {qp[1]:+.4f}] "
                         f"| {pp:.4f} {st(pp)} |")
                rows.append(dict(cell=label, arm=f"{ours} vs {dk}", n=len(y), events=int(y.sum()),
                                 d_auroc=pt_auc[ours]-pt_auc[dk], d_auroc_lo=qa[0], d_auroc_hi=qa[1],
                                 p_auroc=pa, d_auprc=pt_apr[ours]-pt_apr[dk],
                                 d_auprc_lo=qp[0], d_auprc_hi=qp[1], p_auprc=pp))
        print(f"  {label} done", flush=True)

    open(f"{OUT}/deepecg_compare.md", "w").write("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(f"{OUT}/deepecg_compare.csv", index=False)
    print(f"[written to {OUT}/deepecg_compare.md]")


if __name__ == "__main__":
    main()
