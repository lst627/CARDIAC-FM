"""
Bootstrap 95% CIs for AUROC & AUPRC from the saved result.csv files (no retraining).
Stratified (class-preserving) resampling, percentile method. Reports our m75/m90 point + CI per
cell and, where we have the published CARDIAC-FM(ECG) interval, an overlap verdict.

Usage:
  python bootstrap_ci.py [--B 2000] [--seed 42] [--regime zs|fs|both] [--out <md path>]
"""
import argparse, os, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# --- repo path configuration (see docs/PATHS.md) ---
import os as _os, sys as _sys
_pd = _os.path.dirname(_os.path.abspath(__file__))
while _pd != "/" and not _os.path.isdir(_os.path.join(_pd, "common")):
    _pd = _os.path.dirname(_pd)
_sys.path.insert(0, _os.path.join(_pd, "common"))
from paths import P


OUT = P("EXT_RESULTS")


def load(coh, outc, model, regime):
    if regime == "ukb":
        f = f"{OUT}/ukb_test/{model}/{outc}/result.csv"
    elif regime == "zs":
        f = f"{OUT}/zeroshot/{coh}/{model}/{outc}/result.csv"
    else:
        f = f"{OUT}/fewshot/{coh}/train_valid_10_10/{model}/{outc}/result.csv"
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f).dropna(subset=["y_true", "y_pred"])
    return d["y_true"].values.astype(int), d["y_pred"].values


def boot_ci(y, s, metric, B, rng):
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = metric(y, s)
    stats = np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        stats[b] = metric(y[idx], s[idx])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return (point, lo, hi)


# published CARDIAC-FM(ECG) model-only AUROC [ci_lo, ci_hi] (from paper screenshots; verify)
PUB = {
    ("CHS", "zs", "af5"): (.676, .657, .694), ("CHS", "zs", "hf5"): (.686, .625, .748),
    ("CHS", "fs", "af5"): (.717, .708, .726), ("CHS", "fs", "hf5"): (.775, .767, .784),
    ("CHS", "fs", "ces10"): (.706, .669, .742), ("CHS", "fs", "cvddth10"): (.737, .729, .746),
    ("CHS", "fs", "cvddth5"): (.746, .732, .761), ("CHS", "fs", "dth10"): (.691, .681, .701),
    ("CHS", "fs", "dth5"): (.687, .668, .706), ("CHS", "fs", "is10"): (.642, .629, .654),
    ("CHS", "fs", "is5"): (.638, .625, .651), ("CHS", "fs", "mi10"): (.664, .636, .693),
    ("CHS", "fs", "mi5"): (.673, .664, .681),
    ("MESA", "zs", "af5"): (.691, .685, .697), ("MESA", "zs", "hf5"): (.722, .671, .773),
    ("MESA", "fs", "af5"): (.729, .727, .732), ("MESA", "fs", "hf5"): (.800, .789, .811),
    ("MESA", "fs", "ces10"): (.679, .582, .775), ("MESA", "fs", "cvddth10"): (.742, .720, .764),
    ("MESA", "fs", "cvddth5"): (.699, .662, .735), ("MESA", "fs", "dth10"): (.708, .671, .744),
    ("MESA", "fs", "dth5"): (.637, .565, .709), ("MESA", "fs", "is10"): (.634, .598, .670),
    ("MESA", "fs", "is5"): (.645, .628, .663), ("MESA", "fs", "mi10"): (.607, .572, .643),
    ("MESA", "fs", "mi5"): (.588, .568, .608),
}
NAME = {"af5": "AF", "hf5": "HF", "ces10": "CES10", "cvddth5": "CVdth5", "cvddth10": "CVdth10",
        "dth5": "Dth5", "dth10": "Dth10", "mi5": "MI5", "mi10": "MI10", "is5": "IS5", "is10": "IS10"}
FS_OUTCOMES = ["af5", "hf5", "ces10", "cvddth5", "cvddth10", "dth5", "dth10", "mi5", "mi10", "is5", "is10"]


def verdict(lo, hi, pub):
    if pub is None or np.isnan(lo):
        return "-"
    p, plo, phi = pub
    if lo > phi:  return "WIN*"      # our CI entirely above their CI
    if hi < plo:  return "LOSS*"     # our CI entirely below their CI
    return "tie"                     # overlap


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regime", choices=["zs", "fs", "ukb", "both", "all"], default="both")
    ap.add_argument("--out_root", default=OUT,
                    help="input root to read result.csv from (e.g. $EVAL_ROOT)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    OUT = args.out_root                                    # redirect all reads to this root
    out_path = args.out or f"{OUT}/bootstrap_ci.md"
    rng = np.random.default_rng(args.seed)

    regimes = ([("zs", ["af5", "hf5"])] if args.regime in ("zs", "both", "all") else []) + \
              ([("fs", FS_OUTCOMES)] if args.regime in ("fs", "both", "all") else [])
    lines = [f"# Bootstrap 95% CIs (stratified percentile, B={args.B}, seed={args.seed})",
             "AUROC & AUPRC per cell from result.csv. WIN*/LOSS* = our 95% CI entirely above/below the",
             "published CARDIAC-FM(ECG) AUROC CI (non-overlap); tie = overlap. (Their CI is 4-seed, not",
             "bootstrap — overlap is a conservative comparison, not a paired test.)\n"]
    for reg, outs in regimes:
        lines.append(f"\n## {reg.upper()}  (AUROC point [95% CI]  |  AUPRC point [95% CI])\n")
        lines.append("| cohort | outcome | pubCFM AUROC[CI] | m75 AUROC[CI] | m90 AUROC[CI] | vs pub | m75 AUPRC[CI] | m90 AUPRC[CI] |")
        lines.append("|---|---|---|---|---|---|---|---|")
        cohorts = ["UKB"] if reg == "ukb" else ["CHS", "MESA"]
        for coh in cohorts:
            for o in outs:
                cells = {}
                for model in ["m75_ecgfull", "m90_ecgfull"]:
                    d = load(coh, o, model, reg)
                    if d is None:
                        cells[model] = None; continue
                    y, s = d
                    cells[model] = {"roc": boot_ci(y, s, roc_auc_score, args.B, rng),
                                    "pr": boot_ci(y, s, average_precision_score, args.B, rng)}
                pub = PUB.get((coh, reg, o))
                def fmt(t): return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]" if t and not np.isnan(t[0]) else "—"
                m75, m90 = cells.get("m75_ecgfull"), cells.get("m90_ecgfull")
                pubs = f"{pub[0]:.3f} [{pub[1]:.3f}, {pub[2]:.3f}]" if pub else "—"
                # verdict on the better model
                v = "-"
                if pub and m75 and m90:
                    best = m75 if m75["roc"][0] >= m90["roc"][0] else m90
                    v = verdict(best["roc"][1], best["roc"][2], pub)
                lines.append(f"| {coh} | {NAME[o]} | {pubs} | {fmt(m75['roc']) if m75 else '—'} | "
                             f"{fmt(m90['roc']) if m90 else '—'} | {v} | "
                             f"{fmt(m75['pr']) if m75 else '—'} | {fmt(m90['pr']) if m90 else '—'} |")
    # UKB internal test: ecg vs ecg+mri side by side (from 05_ukb_eval_downstream.sh)
    if args.regime in ("ukb", "all"):
        lines.append("\n## UKB internal test — ecg vs ecg+mri (AUROC & AUPRC point [95% CI])\n")
        lines.append("| outcome | model | mode | AUROC [95% CI] | AUPRC [95% CI] | pos/n |")
        lines.append("|---|---|---|---|---|---|")
        fmt = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]" if not np.isnan(t[0]) else "—"
        for o in ["af5", "hf5"]:
            for model in ["m75_ecgfull", "m90_ecgfull"]:
                for mode in ["ecg", "ecg_mri"]:
                    f = f"{OUT}/ukb_test/{model}/{mode}/{o}/result.csv"
                    if not os.path.exists(f):
                        lines.append(f"| {NAME[o]} | {model} | {mode} | — | — | — |"); continue
                    d = pd.read_csv(f).dropna(subset=["y_true", "y_pred"])
                    y = d.y_true.values.astype(int); s = d.y_pred.values
                    roc = boot_ci(y, s, roc_auc_score, args.B, rng)
                    pr = boot_ci(y, s, average_precision_score, args.B, rng)
                    lines.append(f"| {NAME[o]} | {model} | {mode} | {fmt(roc)} | {fmt(pr)} | {int(y.sum())}/{len(y)} |")

    txt = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(txt + "\n")
    print(txt)
    print(f"\n[written to {out_path}]")


if __name__ == "__main__":
    main()
