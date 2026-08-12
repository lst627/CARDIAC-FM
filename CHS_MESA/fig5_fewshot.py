"""
Fig 5 — broad-outcome FEW-SHOT comparison (CHS / MESA).

CARDIAC-FM (ECG) vs the four fine-tuned baselines (ECG-FM, ECGFounder, DeepECG-SSL, DeepECG-SL),
each adapted by the SAME few-shot protocol: fine-tune on the cohort's 10% (train_valid_5_5) or
20% (train_valid_10_10) fraction, then test on the full test split. Broad outcomes have no UKB
label (few-shot is the only option), no risk score (so no +RS arm) and no ECG+MRI mode externally,
so every outcome is a single ECG-only bar per model.

Metric: rank-based AUROC. CIs and tests: paired stratified bootstrap on the COMMON population
(inner-join on id across every arm in the cell). A per-baseline significance symbol is placed on
OUR bar when it beats that baseline (ΔAUROC>0 and paired p<0.05):
  * ECG-FM   † ECGFounder   ‡ DeepECG-SSL   § DeepECG-SL
Only our bar carries markers; a marker string like "*†" means our model beats ECG-FM AND ECGFounder.

Outcomes and models are auto-discovered from disk, so the figure degrades gracefully to whatever
arms exist (e.g. before the DeepECG few-shot job lands, only 3 baseline bars appear).

  python fig5_fewshot.py [--ours_root .../eval] [--B 2000] [--outdir .../eval/figures]
"""
import os, argparse, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
import matplotlib

# --- repo path configuration (see docs/PATHS.md) ---
import os as _os, sys as _sys
_pd = _os.path.dirname(_os.path.abspath(__file__))
while _pd != "/" and not _os.path.isdir(_os.path.join(_pd, "common")):
    _pd = _os.path.dirname(_pd)
_sys.path.insert(0, _os.path.join(_pd, "common"))
from paths import P

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

EV = P("EVAL_ROOT")
FRACS = [("train_valid_5_5", "10%"), ("train_valid_10_10", "20%")]

# our model first, then the four baselines: (fewshot subdir tag, family label, colour, sig symbol)
OURS = ("m75_ecgfull", "CARDIAC-FM (ECG)", "#009E73")
BASELINES = [("ecgfm",      "ECG-FM",      "#E69F00", "*"),
             ("ecgfounder", "ECGFounder",  "#56B4E9", "†"),   # dagger
             ("deepssl_ft", "DeepECG-SSL", "#D55E00", "‡"),   # double dagger
             ("deepsl_ft",  "DeepECG-SL",  "#CC79A7", "§")]   # section
FAM_COLOR = {OURS[1]: OURS[2], **{b[1]: b[2] for b in BASELINES}}
SIG_SYM = {b[1]: b[3] for b in BASELINES}
ORDER = [OURS[1]] + [b[1] for b in BASELINES]


def auroc(y, s):
    y = np.asarray(y); npos = int(y.sum()); nneg = len(y) - npos
    if npos < 3 or nneg < 3:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def load(path):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)[["id", "y_true", "y_pred"]].dropna()
    d["id"] = d["id"].astype(str)
    return d


def cell(OR, coh, frdir, o):
    """Load our + every available baseline for one outcome; inner-join to the common population.
    Returns (y, {family: aligned_scores}) or None if our arm missing / too small."""
    ourp = load(f"{OR}/fewshot/{coh}/{frdir}/{OURS[0]}/{o}/result.csv")
    if ourp is None:
        return None
    frames = {OURS[1]: ourp}
    for tag, fam, _, _ in BASELINES:
        d = load(f"{EV}/fewshot/{coh}/{frdir}/{tag}/{o}/result.csv")
        if d is not None:
            frames[fam] = d
    m = ourp[["id", "y_true"]].copy()
    for fam, d in frames.items():
        m = m.merge(d[["id", "y_pred"]].rename(columns={"y_pred": fam}), on="id")
    m = m.dropna()
    y = m["y_true"].values.astype(int)
    if int(y.sum()) < 3 or int((y == 0).sum()) < 3 or len(m) < 20:
        return None
    return y, {fam: m[fam].values for fam in frames}


def boot(y, P, B, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    fams = list(P)
    pt = {f: auroc(y, P[f]) for f in fams}
    draws = {f: np.empty(B) for f in fams}
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        yy = y[idx]
        for f in fams:
            draws[f][b] = auroc(yy, P[f][idx])
    ci = {f: np.nanpercentile(draws[f], [2.5, 97.5]) for f in fams}
    return pt, ci, draws


def beats_stats(draws, our_fam, base_fam):
    """paired bootstrap our-vs-base: ΔAUROC, its 95% CI, p = 2·min(Pr(Δ≥0),Pr(Δ≤0)), and the
    significance flag (Δ>0 and p<0.05). Returned as a dict so Δ/p land in the cache, not just the mark."""
    d = draws[our_fam] - draws[base_fam]
    delta = float(np.nanmean(d))
    p = 2 * min((d >= 0).mean(), (d <= 0).mean())
    ci = [float(x) for x in np.nanpercentile(d, [2.5, 97.5])]
    return {"delta_auroc": delta, "delta_ci": ci, "p": float(p), "sig": bool(delta > 0 and p < 0.05)}


# Advisor: present only the non-rare broad outcomes, 10-year horizon only.
KEEP_OUTCOMES = ["mi10", "is10", "cvddth10", "dth10"]
OUTC_LABEL = {"mi10": "MI (10y)", "is10": "Ischemic stroke (10y)",
              "cvddth10": "CV death (10y)", "dth10": "All-cause death (10y)"}


def discover_outcomes(OR, coh, frdir):
    """non-rare broad outcomes (10-yr) our model has a few-shot result for."""
    d = f"{OR}/fewshot/{coh}/{frdir}/{OURS[0]}"
    if not os.path.isdir(d):
        return []
    return [o for o in KEEP_OUTCOMES if os.path.exists(f"{d}/{o}/result.csv")]


def compute_panel(OR, coh, frdir, B, rng):
    """Bootstrap one cohort x fraction panel -> a JSON-serialisable list of per-outcome dicts
    (npos, nn, per-family AUROC point + 95% CI, and the significance-mark string for OUR bar).
    This is the ONLY expensive step; render_panel() below draws purely from its output."""
    results = []
    for o in discover_outcomes(OR, coh, frdir):
        c = cell(OR, coh, frdir, o)
        if c is None:
            continue
        y, P = c
        pt, ci, draws = boot(y, P, B, rng)
        tests = {b[1]: beats_stats(draws, OURS[1], b[1]) for b in BASELINES if b[1] in P}
        marks = "".join(SIG_SYM[fam] for fam in (b[1] for b in BASELINES)
                        if fam in tests and tests[fam]["sig"])
        results.append({"o": o, "npos": int(y.sum()), "nn": int(len(y)),
                        "pt": {f: float(pt[f]) for f in pt},
                        "ci": {f: [float(ci[f][0]), float(ci[f][1])] for f in ci},
                        "marks": marks, "tests": tests})
    return results


def render_panel(ax, results):
    """Draw one panel from cached compute_panel() output. No recomputation, no title."""
    if not results:
        ax.set_visible(False)
        return 0
    fams_present = [f for f in ORDER if any(f in r["pt"] for r in results)]
    n = len(results); g = len(fams_present); w = 0.8 / g
    for gi, fam in enumerate(fams_present):
        xs, ys, los, his = [], [], [], []
        for i, r in enumerate(results):
            if fam not in r["pt"]:
                continue
            x = i + (gi - (g - 1) / 2) * w
            xs.append(x); ys.append(r["pt"][fam])
            los.append(r["pt"][fam] - r["ci"][fam][0]); his.append(r["ci"][fam][1] - r["pt"][fam])
        ax.bar(xs, ys, width=w, color=FAM_COLOR[fam], edgecolor="white", linewidth=0.5, label=fam, zorder=2)
        ax.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.5, zorder=3)
    # significance marks above OUR bar
    for i, r in enumerate(results):
        if r["marks"] and OURS[1] in r["ci"]:
            gi = fams_present.index(OURS[1])
            xo = i + (gi - (g - 1) / 2) * w
            ax.text(xo, r["ci"][OURS[1]][1] + 0.012, r["marks"], ha="center", va="bottom", fontsize=8, zorder=4)
    ax.axhline(0.5, color="#999999", lw=0.8, ls=":", zorder=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{OUTC_LABEL.get(r['o'], r['o'])}\n({r['npos']}/{r['nn']})" for r in results], fontsize=8)
    ax.set_ylabel("AUROC"); ax.set_ylim(0.40, 1.0)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_root", default=EV, help="root for OUR few-shot preds (few-shot is seed-1, in eval)")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=f"{EV}/figures")
    ap.add_argument("--stats", default=None, help="cached bootstrap JSON (default <outdir>/fig5_stats.json)")
    ap.add_argument("--recompute", action="store_true", help="force re-run the bootstrap even if the cache exists")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    stats_path = a.stats or f"{a.outdir}/fig5_stats.json"

    # ---- compute (bootstrap) once and cache; on later format tweaks just reload the JSON ----
    if os.path.exists(stats_path) and not a.recompute:
        stats = json.load(open(stats_path))
        print(f"[loaded cached stats <- {stats_path}] (pass --recompute to re-run the bootstrap)", flush=True)
    else:
        rng = np.random.default_rng(a.seed)
        stats = {"_meta": {"B": a.B, "seed": a.seed, "ours_root": a.ours_root}}
        for coh in ["CHS", "MESA"]:
            for frdir, frac_lbl in FRACS:
                res = compute_panel(a.ours_root, coh, frdir, a.B, rng)
                stats[f"{coh}|{frdir}"] = res
                print(f"[computed {coh} {frac_lbl}: {len(res)} outcomes]", flush=True)
        json.dump(stats, open(stats_path, "w"), indent=2)
        print(f"[wrote stats -> {stats_path}]", flush=True)

    # ---- render: one SEPARATE figure per cohort x fraction (CHS/MESA x 10%/20%) -> 4 figures ----
    for coh in ["CHS", "MESA"]:
        for frdir, frac_lbl in FRACS:
            results = stats.get(f"{coh}|{frdir}", [])
            if not results:
                print(f"[{coh} {frac_lbl}] no cached outcomes, skipping", flush=True); continue
            fig, ax = plt.subplots(figsize=(max(9, len(results) * 1.9), 5.2))
            render_panel(ax, results)
            present = [f for f in ORDER if any(f in r["pt"] for r in results)]
            handles = [Patch(facecolor=FAM_COLOR[f], edgecolor="white", label=f) for f in present]
            symnote = "   ".join(f"{s} beats {f}" for f, s in SIG_SYM.items() if f in present)
            fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8,
                       frameon=False, bbox_to_anchor=(0.5, 1.02))
            fig.text(0.5, 0.005, f"Marks on the CARDIAC-FM bar = baselines it significantly beats "
                     f"(paired bootstrap p<0.05):  {symnote}", ha="center", fontsize=8)
            fig.tight_layout(rect=[0, 0.04, 1, 0.98])
            tag = frac_lbl.replace("%", "pct")                # 10% -> 10pct, 20% -> 20pct
            for ext in ["png", "pdf"]:
                fig.savefig(f"{a.outdir}/fig5_fewshot_{coh}_{tag}.{ext}", dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[{coh} {frac_lbl}] wrote fig5_fewshot_{coh}_{tag}.png/pdf", flush=True)


if __name__ == "__main__":
    main()
