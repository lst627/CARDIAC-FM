"""
Time-to-event (Cox/DeepSurv) comparison figure: test-set C-index for CARDIAC-FM (ECG) vs ECG-FM,
per outcome (AF, HF). Reads each model's cox test result.csv (logh, T, E), computes Harrell's
C-index + a bootstrap 95% CI, and draws grouped bars. Also runs a paired bootstrap for the
CARDIAC-FM - ECG-FM ΔC-index p-value.

  python fig_cox_compare.py [--B 1000] [--outdir <figures>]
"""
import argparse, os, json
import numpy as np, pandas as pd
from lifelines.utils import concordance_index
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COX = "/gpfs/projects/trend/bojun/multimodal_rep/eval/cox"
# CARDIAC-FM first (the reference); then the fine-tuned baselines, matched Okabe-Ito colours.
MODELS = [("CARDIAC-FM (ECG)", "m75_seed1", "#009E73"),
          ("ECG-FM", "ecgfm", "#E69F00"),
          ("ECGFounder", "ecgfounder", "#56B4E9"),
          ("DeepECG-SSL", "deepssl_ft", "#D55E00"),
          ("DeepECG-SL", "deepsl_ft", "#CC79A7")]
SIG_SYM = {"ECG-FM": "*", "ECGFounder": "†", "DeepECG-SSL": "‡", "DeepECG-SL": "§"}
OUTCOMES = [("af", "Atrial Fibrillation"), ("hf", "Heart Failure")]


def load(tag, outc):
    f = f"{COX}/{tag}/{outc}/test/result.csv"
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    return d["logh"].values, d["T"].values, d["E"].values.astype(int)


def cindex(T, s, E):
    return concordance_index(T, -s, E)     # higher hazard -> shorter survival


def boot_ci(T, s, E, B, rng):
    n = len(T); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            vals.append(cindex(T[idx], s[idx], E[idx]))
        except Exception:
            pass
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


def compute_stats(B, rng):
    """Bootstrap every (outcome, model) C-index + the paired ΔC-index tests -> JSON-serialisable
    dict. This is the only expensive step; render() draws purely from it."""
    bars, marks, tests = {}, {}, {}
    print(f"{'outcome':<20}{'model':<20}{'C-index [95% CI]':<24}{'events/n'}")
    for outc, _ in OUTCOMES:
        for mname, tag, _col in MODELS:
            r = load(tag, outc)
            if r is None:
                print(f"  [skip] {mname} {outc}: missing"); continue
            s, T, E = r
            pt = cindex(T, s, E); lo, hi = boot_ci(T, s, E, B, rng)
            bars[f"{outc}|{tag}"] = {"cindex": float(pt), "ci": [float(lo), float(hi)],
                                     "events": int(E.sum()), "n": int(len(E))}
            print(f"{outc.upper():<20}{mname:<20}{f'{pt:.3f} [{lo:.3f}, {hi:.3f}]':<24}{int(E.sum())}/{len(E)}")
    # paired bootstrap ΔC-index: CARDIAC-FM vs EACH baseline; cache Δ + p per comparison and mark
    # CARDIAC-FM with every symbol it significantly beats.
    print("\nPaired ΔC-index (CARDIAC-FM − baseline):")
    for outc, _ in OUTCOMES:
        r0 = load(MODELS[0][1], outc)
        if r0 is None:
            continue
        beaten = []; tests[outc] = {}
        for mname, tag, _c in MODELS[1:]:
            rb = load(tag, outc)
            if rb is None:
                continue
            n = min(len(r0[1]), len(rb[1]))
            T, E, s0, s1 = r0[1][:n], r0[2][:n], r0[0][:n], rb[0][:n]
            d = cindex(T, s0, E) - cindex(T, s1, E)
            db = []
            for _ in range(B):
                idx = rng.integers(0, n, n)
                try:
                    db.append(cindex(T[idx], s0[idx], E[idx]) - cindex(T[idx], s1[idx], E[idx]))
                except Exception:
                    pass
            dba = np.array(db)
            p = 2 * min((dba >= 0).mean(), (dba <= 0).mean()) if db else np.nan
            sig = bool(np.isfinite(p) and p < 0.05 and d > 0)
            ci = [float(x) for x in np.percentile(dba, [2.5, 97.5])] if db else [np.nan, np.nan]
            tests[outc][mname] = {"delta_cindex": float(d), "delta_ci": ci,
                                  "p": float(p) if np.isfinite(p) else None, "sig": sig}
            print(f"  {outc.upper()} vs {mname:<13}: Δ={d:+.4f}  p={p:.3f}{' *' if sig else ''}")
            if sig:
                beaten.append(SIG_SYM[mname])
        marks[outc] = "".join(beaten)
    return {"bars": bars, "marks": marks, "tests": tests}


def render(stats, outdir):
    """Draw the grouped-bar C-index figure from cached stats. No recomputation, no title."""
    bars, marks = stats["bars"], stats["marks"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ng = len(MODELS); w = 0.8 / ng
    for gi, (mname, tag, col) in enumerate(MODELS):
        xs, ys, los, his = [], [], [], []
        for oi, (outc, _) in enumerate(OUTCOMES):
            b = bars.get(f"{outc}|{tag}")
            if b is None:
                continue
            pt, (lo, hi) = b["cindex"], b["ci"]
            x = oi + (gi - (ng - 1) / 2) * w
            xs.append(x); ys.append(pt); los.append(pt - lo); his.append(hi - pt)
        ax.bar(xs, ys, width=w, color=col, edgecolor="white", label=mname, zorder=2)
        ax.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor="#333333", elinewidth=1, capsize=3, zorder=3)
    # significance marks on the CARDIAC-FM bar
    for oi, (outc, _) in enumerate(OUTCOMES):
        b0 = bars.get(f"{outc}|{MODELS[0][1]}")
        if b0 and marks.get(outc):
            x = oi + (0 - (ng - 1) / 2) * w
            ax.text(x, b0["ci"][1] + 0.008, marks[outc], ha="center", va="bottom", fontsize=11)

    ax.axhline(0.5, color="#999999", ls=":", lw=1, zorder=1)
    ax.set_xticks(range(len(OUTCOMES)))
    ax.set_xticklabels([t for _, t in OUTCOMES], fontsize=10)
    ax.set_ylabel("Harrell's C-index (UKB test, time-to-event)")
    ax.set_ylim(0.5, 0.85); ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.07))
    fig.text(0.5, 0.005, "Marks on the CARDIAC-FM bar = baselines it significantly beats (paired "
             "bootstrap p<0.05):  * ECG-FM   † ECGFounder   ‡ DeepECG-SSL   § DeepECG-SL",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    for ext in ["png", "pdf"]:
        fig.savefig(f"{outdir}/cox_cindex_compare.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[figure -> {outdir}/cox_cindex_compare.png/pdf]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="/gpfs/projects/trend/bojun/multimodal_rep/eval/figures")
    ap.add_argument("--stats", default=None, help="cached bootstrap JSON (default <outdir>/cox_cindex_stats.json)")
    ap.add_argument("--recompute", action="store_true", help="force re-run the bootstrap even if cache exists")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    stats_path = a.stats or f"{a.outdir}/cox_cindex_stats.json"

    if os.path.exists(stats_path) and not a.recompute:
        stats = json.load(open(stats_path))
        print(f"[loaded cached stats <- {stats_path}] (pass --recompute to re-run)")
    else:
        rng = np.random.default_rng(a.seed)
        stats = compute_stats(a.B, rng)
        stats["_meta"] = {"B": a.B, "seed": a.seed}
        json.dump(stats, open(stats_path, "w"), indent=2)
        print(f"[wrote stats -> {stats_path}]")
    render(stats, a.outdir)


if __name__ == "__main__":
    main()
