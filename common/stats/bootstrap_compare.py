"""
Paired-bootstrap comparison of two models on the same test set, for AUROC AND AUPRC.
Resamples test indices (stratified by class), recomputes the DIFFERENCE metric(A)-metric(B) each
draw -> difference point estimate, 95% CI, and two-sided p-value. Works for AUROC and AUPRC alike.
Predictions are paired on `id`, so this is a within-subject comparison.

Default use: OUR model (m75/m90, few-shot 20%) vs the original ECG-FM / CARDIAC-FM, aggregated over
their 4 seeds. p<0.05 with positive Δ = ours significantly better on that metric.

Usage: python bootstrap_compare.py [--ours m90_ecgfull] [--seeds "1 2 3 4"] [--B 2000] [--seed 42]
"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = "/gpfs/projects/trend/bojun/CHS_MESA/results"
EXT = "/gpfs/projects/trend/bojun/CHS_MESA/result_finetune_from_pretrain_fewshot"
# overlap outcomes (both our CL and the paper's ECG-FM/CARDIAC-FM few-shot predictions exist)
OUTCOMES = ["af5", "hf5", "mi5", "mi10", "is5", "is10", "ces10", "cvddth5", "cvddth10", "dth5", "dth10"]
NAME = {"af5": "AF", "hf5": "HF", "mi5": "MI5", "mi10": "MI10", "is5": "IS5", "is10": "IS10",
        "ces10": "CES10", "cvddth5": "CVdth5", "cvddth10": "CVdth10", "dth5": "Dth5", "dth10": "Dth10"}


def load_ours(coh, o, model):
    f = f"{OUT}/fewshot/{coh}/train_valid_10_10/{model}/{o}/result.csv"
    return pd.read_csv(f)[["id", "y_true", "y_pred"]] if os.path.exists(f) else None


def load_ext(coh, o, model, seed):
    f = f"{EXT}/{coh}_{o.upper()}_{model}{seed}.csv"
    return pd.read_csv(f)[["id", "y_true", "y_pred"]] if os.path.exists(f) else None


def paired_boot(y, pa, pb, metric, B, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    point = metric(y, pa) - metric(y, pb)
    diffs = np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        yb = y[idx]
        diffs[b] = metric(yb, pa[idx]) - metric(yb, pb[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    return (point, lo, hi, min(p, 1.0))


def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "·"


def agg(od, coh, o, base, seeds, B, rng):
    roc, pr = [], []
    for s in seeds:
        x = load_ext(coh, o, base, s)
        if x is None:
            continue
        m = od.merge(x, on="id", suffixes=("_o", "_x")).dropna(subset=["y_true_o", "y_pred_o", "y_pred_x"])
        y = m["y_true_o"].values.astype(int)
        if len(np.unique(y)) < 2:
            continue
        roc.append(paired_boot(y, m["y_pred_o"].values, m["y_pred_x"].values, roc_auc_score, B, rng))
        pr.append(paired_boot(y, m["y_pred_o"].values, m["y_pred_x"].values, average_precision_score, B, rng))
    if not roc:
        return None

    def summ(rs):
        d = np.array([r[0] for r in rs]); p = np.array([r[3] for r in rs])
        return np.median(d), int(((p < 0.05) & (d > 0)).sum()), int(((p < 0.05) & (d < 0)).sum()), len(rs)
    return summ(roc), summ(pr)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="m90_ecgfull")
    ap.add_argument("--seeds", default="1 2 3 4")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ours_root", default=OUT,
                    help="root for our CL result.csv (e.g. /gpfs/.../multimodal_rep/eval)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    OUT = args.ours_root                                   # our-CL predictions read from here
    seeds = [int(s) for s in args.seeds.split()]
    rng = np.random.default_rng(args.seed)
    out = args.out or f"{OUT}/bootstrap_compare_{args.ours}.md"

    lines = [f"# Paired-bootstrap: {args.ours} vs ECG-FM / CARDIAC-FM (B={args.B}, seeds {seeds})",
             "Δmed = median AUC(ours)−AUC(theirs) over baseline seeds. sig+/− = # seeds where ours is",
             "significantly better/worse (paired-bootstrap p<0.05). Reported for AUROC and AUPRC.\n"]
    for base in ["ECGFM", "CARDIACFM"]:
        lines.append(f"\n## vs {base}\n")
        lines.append("| cohort | outcome | ΔAUROC med | AUROC sig+ | AUROC sig− | ΔAUPRC med | AUPRC sig+ | AUPRC sig− |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for coh in ["CHS", "MESA"]:
            for o in OUTCOMES:
                od = load_ours(coh, o, args.ours)
                if od is None:
                    continue
                a = agg(od, coh, o, base, seeds, args.B, rng)
                if a is None:
                    continue
                (dr, rp, rm, rk), (dp, pp, pm, pk) = a
                lines.append(f"| {coh} | {NAME[o]} | {dr:+.3f} | {rp}/{rk} | {rm}/{rk} | "
                             f"{dp:+.3f} | {pp}/{pk} | {pm}/{pk} |")
    txt = "\n".join(lines)
    with open(out, "w") as f:
        f.write(txt + "\n")
    print(txt)
    print(f"\n[written to {out}]")


if __name__ == "__main__":
    main()
