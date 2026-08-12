"""
Zero-shot survival (time-to-event) comparison on CHS / MESA: the UKB Cox/DeepSurv-fine-tuned models
applied WITHOUT refit to the external cohorts. Harrell's C-index per model, bootstrap 95% CI, and a
paired bootstrap ΔC-index vs CARDIAC-FM (marks = baselines it significantly beats). Mirrors
fig_cox_compare.py (UKB) but over 4 cells: CHS/MESA × AF/HF.

Two-stage: compute -> cache cox_zeroshot_stats.json ; render reads the cache (no recompute).
  python fig_cox_zeroshot.py [--B 2000] [--recompute] [--outdir <figures>]
"""
import argparse, os, json
import numpy as np, pandas as pd
from lifelines.utils import concordance_index
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ZS = "/gpfs/projects/trend/bojun/multimodal_rep/eval/cox_zeroshot"
MODELS = [("CARDIAC-FM (ECG)", "m75_seed1", "#009E73"),
          ("ECG-FM", "ecgfm", "#E69F00"),
          ("ECGFounder", "ecgfounder", "#56B4E9"),
          ("DeepECG-SSL", "deepssl_ft", "#D55E00"),
          ("DeepECG-SL", "deepsl_ft", "#CC79A7")]
SIG_SYM = {"ECG-FM": "*", "ECGFounder": "†", "DeepECG-SSL": "‡", "DeepECG-SL": "§"}
CELLS = [("CHS", "af"), ("CHS", "hf"), ("MESA", "af"), ("MESA", "hf")]
CELL_LABEL = {("CHS", "af"): "CHS — AF", ("CHS", "hf"): "CHS — HF",
              ("MESA", "af"): "MESA — AF", ("MESA", "hf"): "MESA — HF"}


def load(coh, tag, outc):
    f = f"{ZS}/{coh}/{tag}/{outc}/result.csv"
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    return d["logh"].values, d["T"].values, d["E"].values.astype(int)


def cindex(T, s, E):
    return concordance_index(T, -s, E)          # higher hazard -> shorter survival


def boot_ci(T, s, E, B, rng):
    n = len(T); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            vals.append(cindex(T[idx], s[idx], E[idx]))
        except Exception:
            pass
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def compute_stats(B, rng):
    bars, marks, tests = {}, {}, {}
    print(f"{'cell':<12}{'model':<20}{'C-index [95% CI]':<24}{'events/n'}")
    for coh, outc in CELLS:
        key = f"{coh}|{outc}"
        for mname, tag, _c in MODELS:
            r = load(coh, tag, outc)
            if r is None:
                print(f"  [skip] {mname} {coh} {outc}: missing"); continue
            s, T, E = r
            pt = cindex(T, s, E); lo, hi = boot_ci(T, s, E, B, rng)
            bars[f"{key}|{tag}"] = {"cindex": float(pt), "ci": [lo, hi], "events": int(E.sum()), "n": int(len(E))}
            print(f"{key:<12}{mname:<20}{f'{pt:.3f} [{lo:.3f}, {hi:.3f}]':<24}{int(E.sum())}/{len(E)}")
        # paired ΔC-index vs CARDIAC-FM
        r0 = load(coh, MODELS[0][1], outc)
        if r0 is None:
            continue
        beaten = []; tests[key] = {}
        for mname, tag, _c in MODELS[1:]:
            rb = load(coh, tag, outc)
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
            tests[key][mname] = {"delta_cindex": float(d), "delta_ci": ci,
                                 "p": float(p) if np.isfinite(p) else None, "sig": sig}
            print(f"  {key} vs {mname:<13}: Δ={d:+.4f} p={p:.3f}{' *' if sig else ''}")
            if sig:
                beaten.append(SIG_SYM[mname])
        marks[key] = "".join(beaten)
    return {"bars": bars, "marks": marks, "tests": tests}


def render(stats, outdir):
    bars, marks = stats["bars"], stats["marks"]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ng = len(MODELS); w = 0.8 / ng
    for gi, (mname, tag, col) in enumerate(MODELS):
        xs, ys, los, his = [], [], [], []
        for ci_, (coh, outc) in enumerate(CELLS):
            b = bars.get(f"{coh}|{outc}|{tag}")
            if b is None:
                continue
            pt, (lo, hi) = b["cindex"], b["ci"]
            x = ci_ + (gi - (ng - 1) / 2) * w
            xs.append(x); ys.append(pt); los.append(pt - lo); his.append(hi - pt)
        # diverging bars anchored at the 0.5 chance line: above-chance go up, below-chance go down
        ax.bar(xs, [y - 0.5 for y in ys], width=w, bottom=0.5, color=col, edgecolor="white",
               label=mname, zorder=2)
        ax.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor="#333333", elinewidth=1, capsize=3, zorder=3)
    for ci_, (coh, outc) in enumerate(CELLS):
        b0 = bars.get(f"{coh}|{outc}|{MODELS[0][1]}")
        if b0 and marks.get(f"{coh}|{outc}"):
            x = ci_ + (0 - (ng - 1) / 2) * w
            ax.text(x, b0["ci"][1] + 0.008, marks[f"{coh}|{outc}"], ha="center", va="bottom", fontsize=11)
    ax.axhline(0.5, color="#666666", ls="--", lw=1, zorder=4)   # chance line (bars diverge from here)
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([CELL_LABEL[c] for c in CELLS], fontsize=10)
    ax.set_ylabel("Harrell's C-index (zero-shot, time-to-event)")
    ax.set_ylim(0.44, 0.72); ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.07))
    fig.text(0.5, 0.005, "Marks on the CARDIAC-FM bar = baselines it significantly beats (paired "
             "bootstrap p<0.05):  * ECG-FM   † ECGFounder   ‡ DeepECG-SSL   § DeepECG-SL",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    for ext in ["png", "pdf"]:
        fig.savefig(f"{outdir}/cox_cindex_zeroshot.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[figure -> {outdir}/cox_cindex_zeroshot.png/pdf]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="/gpfs/projects/trend/bojun/multimodal_rep/eval/figures")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    stats_path = a.stats or f"{a.outdir}/cox_cindex_zeroshot_stats.json"
    if os.path.exists(stats_path) and not a.recompute:
        stats = json.load(open(stats_path)); print(f"[loaded {stats_path}]")
    else:
        stats = compute_stats(a.B, np.random.default_rng(a.seed))
        stats["_meta"] = {"B": a.B, "seed": a.seed}
        json.dump(stats, open(stats_path, "w"), indent=2); print(f"[wrote {stats_path}]")
    render(stats, a.outdir)


if __name__ == "__main__":
    main()
