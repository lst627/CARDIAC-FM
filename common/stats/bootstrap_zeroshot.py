"""
Zero-shot bootstrap: our CL models (m75/m90) vs our in-pipeline ECG-FM, on the CHS/MESA zero-shot
af5/hf5 cells. Per-model 95% CI (AUROC + AUPRC) and paired-bootstrap Δ + p-value vs ECG-FM
(same test set, paired on id). Stratified resampling, percentile method.
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
RES = P("EXT_RESULTS")
# paper ECG-FM (provided ECGFM_ft checkpoints) zero-shot, seed 1 — matched seed-1 vs seed-1 comparison
EFROOT = P("EVAL_ROOT", "ecgfm_zeroshot_provided")


def fauc(y, s):  # fast rank-based AUROC (ties handled by rankdata)
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def load(f):
    return pd.read_csv(f)[["id", "y_true", "y_pred"]] if os.path.exists(f) else None


def ci(y, s, metric, B, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    st = np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        st[b] = metric(y[idx], s[idx])
    return metric(y, s), *np.percentile(st, [2.5, 97.5])


def paired(y, a, bb, metric, B, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    df = np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        yb = y[idx]
        df[b] = metric(yb, a[idx]) - metric(yb, bb[idx])
    p = 2 * min((df >= 0).mean(), (df <= 0).mean())
    return metric(y, a) - metric(y, bb), *np.percentile(df, [2.5, 97.5]), min(p, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=f"{EV}/bootstrap_zeroshot.md")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    apr = lambda y, s: average_precision_score(y, s)

    L = [f"# Zero-shot bootstrap — CL (m75/m90) vs paper ECG-FM seed1 (B={args.B}, seed {args.seed})",
         "Per-model point [95% CI]; paired Δ = CL − ECG-FM [95% CI] with p-value (same test set).\n"]
    for coh in ["CHS", "MESA"]:
        for o in ["af5", "hf5"]:
            ef = load(f"{EFROOT}/seed1/zeroshot/{coh}/{o}/result.csv")
            m75 = load(f"{EV}/zeroshot/{coh}/m75_ecgfull/{o}/result.csv")
            m90 = load(f"{EV}/zeroshot/{coh}/m90_ecgfull/{o}/result.csv")
            if ef is None or m75 is None or m90 is None:
                L.append(f"\n## {coh} {o}: missing result.csv — skipped"); continue
            m = (ef.merge(m75.rename(columns={"y_pred": "p75"})[["id", "p75"]], on="id")
                   .merge(m90.rename(columns={"y_pred": "p90"})[["id", "p90"]], on="id")
                   .dropna(subset=["y_true", "y_pred", "p75", "p90"]))
            y = m.y_true.values.astype(int); pe, p75, p90 = m.y_pred.values, m.p75.values, m.p90.values
            L.append(f"\n## {coh} {o}  (n={len(y)}, pos={int(y.sum())})\n")
            L.append("| model | AUROC [95% CI] | AUPRC [95% CI] |")
            L.append("|---|---|---|")
            for nm, pr in [("ECG-FM", pe), ("m75", p75), ("m90", p90)]:
                a = ci(y, pr, fauc, args.B, rng); p = ci(y, pr, apr, args.B, rng)
                L.append(f"| {nm} | {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | {p[0]:.3f} [{p[1]:.3f}, {p[2]:.3f}] |")
            L.append("\n| vs ECG-FM | ΔAUROC [95% CI] p | ΔAUPRC [95% CI] p |")
            L.append("|---|---|---|")
            for nm, pr in [("m75", p75), ("m90", p90)]:
                dr = paired(y, pr, pe, fauc, args.B, rng); dp = paired(y, pr, pe, apr, args.B, rng)
                L.append(f"| {nm} | {dr[0]:+.3f} [{dr[1]:+.3f}, {dr[2]:+.3f}] p={dr[3]:.3f} | "
                         f"{dp[0]:+.3f} [{dp[1]:+.3f}, {dp[2]:+.3f}] p={dp[3]:.3f} |")
    txt = "\n".join(L)
    with open(args.out, "w") as f:
        f.write(txt + "\n")
    print(txt); print(f"\n[written to {args.out}]")


if __name__ == "__main__":
    main()
