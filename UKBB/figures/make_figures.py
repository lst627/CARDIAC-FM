"""
Reproduce paper Fig 2 (UK Biobank) and Fig 3 (CHS/MESA external), with three changes vs the paper:
  1. error bars are STRATIFIED BOOTSTRAP 95% CI over subjects (not mean +/- 1.96 SD over 4 seeds);
  2. a * above a bar = that bar significantly beats the ECG-FM reference (paired bootstrap p<0.05;
     +Risk bars are compared to ECG-FM+Risk). Set STAR_REF below to change the reference.
  3. ECGFounder is added as a baseline (seed 1, matched protocol: lr 5e-6, patience-3 early stop).

All bars in a panel are inner-joined on `id` -> one common population, so every bar and every
paired test is on the same subjects. Everything is seed 1 (single-seed comparison).

Encoding (dataviz): COLOR = model family (5, Okabe-Ito colour-blind-safe, fixed order);
TEXTURE (hatch) = +Risk Score. This is the paper's own solid/hatched convention and keeps the
categorical colour count at 5.

A SINGLE run uses ONE seed of our model and ONE seed of ECG-FM (not a mean/ensemble):
  --our_seed   {0,1,2,3}   our m75 seed  (1 = canonical eval/, 0/2/3 = eval_<S>/)
  --ecgfm_seed {1,2,3,4}   ECG-FM seed
ECGFounder is single-seed (seed 1) and Risk Score is fixed, so those bars are constant.
Output filenames carry the seed pair, e.g. fig2_ukb_our1_ecgfm1.png.

Usage: python make_figures.py [--our_seed 1] [--ecgfm_seed 1] [--B 2000]
"""
import os, argparse, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")

EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"
RS = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/computed"
RISK_COL = {"af5": "charge_mean", "hf5": "prevent_mean"}
OUT = f"{EV}/figures"; os.makedirs(OUT, exist_ok=True)
B = 2000; RNG = np.random.default_rng(42)
STAR_REF = "ECG-FM"        # star = significantly beats this family (its +RS form for +RS bars)


def our_root(seed):
    """our m75 predictions live in eval/ for seed 1 (canonical) and eval_<S>/ for 0/2/3."""
    return EV if seed == 1 else f"/gpfs/projects/trend/bojun/multimodal_rep/eval_{seed}"

# model family -> colour (Okabe-Ito, fixed order; Risk Score = neutral grey)
FAM_COLOR = {"Risk Score": "#9A9A9A", "ECG-FM": "#E69F00", "ECGFounder": "#56B4E9",
             "DeepECG-SSL": "#D55E00", "DeepECG-SL": "#CC79A7",
             "CARDIAC-FM (ECG)": "#009E73", "CARDIAC-FM (ECG+MRI)": "#0072B2"}   # Okabe-Ito, fixed order
HATCH = "///"              # +Risk Score bars
# fine-tuned foundation-model baselines; each gets its OWN significance symbol so a CARDIAC-FM bar
# shows exactly WHICH baselines it beats (paired bootstrap p<0.05, risk-tier-matched).
BASELINE_FAMILIES = ["ECG-FM", "ECGFounder", "DeepECG-SSL", "DeepECG-SL"]
SIG_SYM = {"ECG-FM": "*", "ECGFounder": "†", "DeepECG-SSL": "‡", "DeepECG-SL": "§"}


# ----------------------------------------------------------------------------- metrics
def auroc(y, s):
    y = np.asarray(y); npos = int(y.sum()); nneg = len(y) - npos
    if npos < 3 or nneg < 3:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def auprc(y, s):
    if int(np.sum(y)) < 3:
        return np.nan
    return average_precision_score(y, s)


def load_pred(path):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)[["id", "y_true", "y_pred"]].dropna()
    d["id"] = d["id"].astype(str)
    return d


def load_risk(cohort_key, outc):
    r = pd.read_csv(f"{RS}/{cohort_key}_riskscore.csv")[["id", RISK_COL[outc]]]
    r = r.rename(columns={RISK_COL[outc]: "y_pred"}); r["id"] = r["id"].astype(str)
    return r


# ----------------------------------------------------------------------------- bar sets
def fig2_bars(outc, oseed, eseed):
    """UKB paired test. (label, family, is_rs, source). ours from our_root(oseed); ECG-FM seed eseed;
    ECGFounder + Risk Score fixed."""
    orr = our_root(oseed)
    return [
        ("Risk Score",              "Risk Score",           False, ("risk", "UKBB")),
        ("ECG-FM",                  "ECG-FM",               False, f"{EV}/ecgfm_ukb_paired/seed{eseed}/test/{outc}/result.csv"),
        ("ECGFounder",              "ECGFounder",           False, f"{EV}/ukb_test/ecgfounder/{outc}/result.csv"),
        ("DeepECG-SSL",             "DeepECG-SSL",          False, f"{EV}/ukb_test/deepssl_ft/{outc}/result.csv"),
        ("DeepECG-SL",              "DeepECG-SL",           False, f"{EV}/ukb_test/deepsl_ft/{outc}/result.csv"),
        ("CARDIAC-FM (ECG)",        "CARDIAC-FM (ECG)",     False, f"{orr}/ukb_test/m75_ecgfull/ecg/{outc}/result.csv"),
        ("CARDIAC-FM (ECG+MRI)",    "CARDIAC-FM (ECG+MRI)", False, f"{orr}/ukb_test/m75_ecgfull/ecg_mri/{outc}/result.csv"),
        ("ECG-FM +RS",             "ECG-FM",               True,  f"{EV}/ukb_test/ecgfm_RS/seed{eseed}/{outc}/result.csv"),
        ("ECGFounder +RS",         "ECGFounder",           True,  f"{EV}/ukb_test/ecgfounder_RS/{outc}/result.csv"),
        ("DeepECG-SSL +RS",        "DeepECG-SSL",          True,  f"{EV}/ukb_test/deepssl_ft_RS/{outc}/result.csv"),
        ("DeepECG-SL +RS",         "DeepECG-SL",           True,  f"{EV}/ukb_test/deepsl_ft_RS/{outc}/result.csv"),
        ("CARDIAC-FM (ECG) +RS",   "CARDIAC-FM (ECG)",     True,  f"{orr}/ukb_test/m75_ecgfull_RS/ecg/{outc}/result.csv"),
        ("CARDIAC-FM (ECG+MRI) +RS","CARDIAC-FM (ECG+MRI)", True,  f"{orr}/ukb_test/m75_ecgfull_RS/ecg_mri/{outc}/result.csv"),
    ]


def fig3_bars(coh, outc, oseed, eseed):
    """External zero-shot (ECG-only; no ECG+MRI mode externally)."""
    orr = our_root(oseed)
    return [
        ("Risk Score",           "Risk Score",       False, ("risk", coh)),
        ("ECG-FM",               "ECG-FM",           False, f"{EV}/ecgfm_zeroshot_provided/seed{eseed}/zeroshot/{coh}/{outc}/result.csv"),
        ("ECGFounder",           "ECGFounder",       False, f"{EV}/zeroshot/{coh}/ecgfounder/{outc}/result.csv"),
        ("DeepECG-SSL",          "DeepECG-SSL",      False, f"{EV}/zeroshot/{coh}/deepssl_ft/{outc}/result.csv"),
        ("DeepECG-SL",           "DeepECG-SL",       False, f"{EV}/zeroshot/{coh}/deepsl_ft/{outc}/result.csv"),
        ("CARDIAC-FM (ECG)",     "CARDIAC-FM (ECG)", False, f"{orr}/zeroshot/{coh}/m75_ecgfull/{outc}/result.csv"),
        ("ECG-FM +RS",          "ECG-FM",           True,  f"{EV}/zeroshot/{coh}/ecgfm_RS/seed{eseed}/{outc}/result.csv"),
        ("ECGFounder +RS",      "ECGFounder",       True,  f"{EV}/zeroshot/{coh}/ecgfounder_RS/{outc}/result.csv"),
        ("DeepECG-SSL +RS",     "DeepECG-SSL",      True,  f"{EV}/zeroshot/{coh}/deepssl_ft_RS/{outc}/result.csv"),
        ("DeepECG-SL +RS",      "DeepECG-SL",       True,  f"{EV}/zeroshot/{coh}/deepsl_ft_RS/{outc}/result.csv"),
        ("CARDIAC-FM (ECG) +RS","CARDIAC-FM (ECG)", True,  f"{orr}/zeroshot/{coh}/m75_ecgfull_RS/{outc}/result.csv"),
    ]


# ----------------------------------------------------------------------------- assemble one panel
def assemble(bars, outc):
    """Load every bar, inner-join on id -> common population. Returns (labels, fams, rs, Y, P) where
    Y is the shared y_true and P[label] is that bar's aligned score vector."""
    frames = {}
    for label, fam, is_rs, src in bars:
        d = load_risk(src[1], outc) if isinstance(src, tuple) else load_pred(src)
        if d is None:
            print(f"  [skip] missing: {label} ({outc})"); continue
        frames[label] = d
    if "ECG-FM" not in frames:
        return None
    # y_true from a model arm (risk file has no y_true)
    base = next(frames[l][["id", "y_true"]] for l, *_ in bars if l in frames and "y_true" in frames[l])
    m = base.copy()
    for label in list(frames):
        m = m.merge(frames[label][["id", "y_pred"]].rename(columns={"y_pred": label}), on="id")
    m = m.dropna()
    y = m["y_true"].values.astype(int)
    P = {label: m[label].values for label in frames}
    kept = [(l, f, r) for (l, f, r, _) in bars if l in frames]
    return kept, y, P


def boot(y, P, metric):
    """point + 95% CI per bar, and the full bootstrap draw matrix (for paired tests vs any ref)."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    labels = list(P)
    pt = {l: metric(y, P[l]) for l in labels}
    draws = {l: np.empty(B) for l in labels}
    for b in range(B):
        idx = np.concatenate([RNG.choice(pos, len(pos), True), RNG.choice(neg, len(neg), True)])
        yy = y[idx]
        for l in labels:
            draws[l][b] = metric(yy, P[l][idx])
    ci = {l: np.nanpercentile(draws[l], [2.5, 97.5]) for l in labels}
    return pt, ci, draws


OUR_FAMILIES = {"CARDIAC-FM (ECG)", "CARDIAC-FM (ECG+MRI)"}   # only OUR bars carry significance marks


def refs_for(kept, ref_family):
    """significance reference for OUR model bars only (baselines never get a marker), MATCHED on
    risk tier: a solid (no-RS) CARDIAC-FM bar is tested against the no-RS reference; a +RS bar
    against the +RS reference."""
    have = {l for l, *_ in kept}
    refs = {}
    for l, fam, is_rs in kept:
        if fam not in OUR_FAMILIES:
            refs[l] = None
        else:
            r = f"{ref_family} +RS" if is_rs else ref_family
            refs[l] = r if r in have else None
    return refs


def stars_vs(draws, pt, refs):
    """paired-bootstrap p<0.05 AND higher point estimate, per bar, vs its matched reference."""
    out = {}
    for l in draws:
        ref = refs.get(l)
        if ref is None or ref not in draws:
            out[l] = False; continue
        d = draws[l] - draws[ref]
        p = 2 * min(np.nanmean(d >= 0), np.nanmean(d <= 0))
        out[l] = (p < 0.05) and (pt[l] > pt[ref])
    return out


# ----------------------------------------------------------------------------- drawing
def draw_group(ax, kept, y, P, metric, x0):
    """draw one outcome group of bars starting at x0; returns next x and the tick centre.
    Significance: a single * on a CARDIAC-FM bar = it beats EVERY fine-tuned foundation-model
    baseline present (ECG-FM, ECGFounder, DeepECG-SSL, DeepECG-SL), each risk-tier-matched,
    paired bootstrap p<0.05. Consolidated to one mark since there are now four baselines."""
    pt, ci, draws = boot(y, P, metric)
    have = {f for _, f, _ in kept}
    star_by_fam = {fam: stars_vs(draws, pt, refs_for(kept, fam)) for fam in BASELINE_FAMILIES if fam in have}
    w = 0.8
    xs = x0 + np.arange(len(kept))
    for x, (l, fam, is_rs) in zip(xs, kept):
        lo, hi = ci[l]
        ax.bar(x, pt[l], width=w, color=FAM_COLOR[fam], edgecolor="white", linewidth=0.6,
               hatch=HATCH if is_rs else None, zorder=2)
        ax.errorbar(x, pt[l], yerr=[[pt[l]-lo], [hi-pt[l]]], fmt="none", ecolor="#333333",
                    elinewidth=1.0, capsize=2.2, zorder=3)
        if fam in OUR_FAMILIES:                          # one symbol per baseline this bar beats
            mk = "".join(SIG_SYM[f] for f in BASELINE_FAMILIES
                         if f in star_by_fam and star_by_fam[f].get(l, False))
            if mk:
                ax.text(x, hi + 0.006, mk, ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color="#222222", zorder=4)
    return xs[-1] + 1.4, xs.mean()


def panel(ax, groups, metric, ylabel, title, ymin):
    """groups = list of (group_label, kept, y, P)."""
    x = 0.0; centres = []
    for glabel, kept, y, P in groups:
        x, c = draw_group(ax, kept, y, P, metric, x)
        centres.append((c, glabel))
    ax.set_xticks([c for c, _ in centres])
    ax.set_xticklabels([g for _, g in centres], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.set_ylim(ymin, None)
    ax.margins(x=0.02)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#888"); ax.spines["bottom"].set_color("#888")


def legend_handles(present):
    """only show families actually drawn (external has no ECG+MRI bar)."""
    h = [Patch(facecolor=c, edgecolor="white", label=f) for f, c in FAM_COLOR.items() if f in present]
    h.append(Patch(facecolor="#CCCCCC", edgecolor="white", hatch=HATCH, label="+ Risk Score"))
    h.append(Patch(facecolor="none", edgecolor="none",
                   label="CARDIAC-FM beats (p<0.05):  * ECG-FM   † ECGFounder   ‡ DeepECG-SSL   § DeepECG-SL"))
    return h


def families_in(panels):
    fams = set()
    for p in panels.values():
        if p:
            for _, f, _ in p[0]:
                fams.add(f)
    return fams


# ----------------------------------------------------------------------------- figures
def make_fig2(oseed, eseed):
    tag = f"our{oseed}_ecgfm{eseed}"
    panels = {o: assemble(fig2_bars(o, oseed, eseed), o) for o in ["af5", "hf5"]}
    fig, axes = plt.subplots(1, 2, figsize=(19, 5.4))
    for ax, (metric, ylab, ttl, ymin) in zip(
            axes, [(auroc, "AUROC", "Discrimination (AUROC)", 0.5),
                   (auprc, "AUPRC", "Precision–recall (AUPRC)", 0.0)]):
        groups = [(nm, *panels[o]) for o, nm in [("af5", "Atrial Fibrillation"), ("hf5", "Heart Failure")]]
        panel(ax, groups, metric, ylab, ttl, ymin)
    fig.suptitle(f"Figure 2 — UK Biobank: 5-year incident AF and HF (paired test)   "
                 f"[CARDIAC-FM seed {oseed}, ECG-FM seed {eseed}]",
                 fontsize=12.5, fontweight="bold", x=0.02, ha="left")
    fig.legend(handles=legend_handles(families_in(panels)), loc="lower center", ncol=8, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig2_ukb_{tag}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] -> {OUT}/fig2_ukb_{tag}.png|pdf")


def make_fig3(oseed, eseed):
    tag = f"our{oseed}_ecgfm{eseed}"
    fig, axes = plt.subplots(2, 2, figsize=(19, 10.5))
    allfams = set()
    for row, coh in enumerate(["CHS", "MESA"]):
        panels = {o: assemble(fig3_bars(coh, o, oseed, eseed), o) for o in ["af5", "hf5"]}
        allfams |= families_in(panels)
        for col, (metric, ylab, ymin) in enumerate(
                [(auroc, "AUROC", 0.5), (auprc, "AUPRC", 0.0)]):
            groups = [(nm, *panels[o]) for o, nm in [("af5", "Atrial Fibrillation"), ("hf5", "Heart Failure")]]
            panel(axes[row, col], groups, metric, ylab,
                  f"({chr(97+row)}) {coh} — {ylab} (zero-shot)", ymin)
    fig.suptitle(f"Figure 3 — External validation (zero-shot, ECG-only): CHS and MESA   "
                 f"[CARDIAC-FM seed {oseed}, ECG-FM seed {eseed}]",
                 fontsize=12.5, fontweight="bold", x=0.02, ha="left")
    fig.legend(handles=legend_handles(allfams), loc="lower center", ncol=8, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig3_external_{tag}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] -> {OUT}/fig3_external_{tag}.png|pdf")


# ----------------------------------------------------------------------------- numbers dump
def panel_stats(kept, y, P):
    """Every plotted number for one panel: per-bar AUROC+AUPRC point + 95% CI, and each OUR bar's
    paired-bootstrap ΔmetriC + p vs every baseline family (risk-tier matched). Reseeds per panel so
    the dump is reproducible on its own. This is the machine-readable source another person plots from."""
    global RNG
    out = {"n": int(len(y)), "pos": int(y.sum()), "neg": int((y == 0).sum())}
    for mname, metric in [("auroc", auroc), ("auprc", auprc)]:
        RNG = np.random.default_rng(42)
        pt, ci, draws = boot(y, P, metric)
        bars = {l: {"point": float(pt[l]), "ci_lo": float(ci[l][0]), "ci_hi": float(ci[l][1]),
                    "family": fam, "is_rs": bool(rs)} for (l, fam, rs) in kept}
        pairs = []
        for (l, fam, rs) in kept:
            if fam not in OUR_FAMILIES:
                continue
            for bfam in BASELINE_FAMILIES:
                ref = f"{bfam} +RS" if rs else bfam
                if ref not in draws:
                    continue
                d = draws[l] - draws[ref]
                p = 2 * min(np.nanmean(d >= 0), np.nanmean(d <= 0))
                pairs.append({"bar": l, "vs": ref, "delta": float(np.nanmean(d)),
                              "ci_lo": float(np.nanpercentile(d, 2.5)), "ci_hi": float(np.nanpercentile(d, 97.5)),
                              "p": float(p), "sig": bool(p < 0.05 and pt[l] > pt[ref])})
        out[mname] = {"bars": bars, "pairs": pairs}
    return out


def dump_numbers(oseed, eseed, path):
    """Write fig2 (UKB) + fig3 (external CHS/MESA) plotting numbers to one JSON."""
    stats = {"_meta": {"B": B, "seed": 42, "our_seed": oseed, "ecgfm_seed": eseed,
                       "note": "UKB +RS bars use the risk score; see ukb-risk-score-mapping-bug"}}
    for o in ["af5", "hf5"]:
        pan = assemble(fig2_bars(o, oseed, eseed), o)
        if pan:
            stats[f"UKB|{o}"] = panel_stats(*pan)
    for coh in ["CHS", "MESA"]:
        for o in ["af5", "hf5"]:
            pan = assemble(fig3_bars(coh, o, oseed, eseed), o)
            if pan:
                stats[f"{coh}|{o}"] = panel_stats(*pan)
    json.dump(stats, open(path, "w"), indent=2)
    print(f"[fig2/fig3 numbers -> {path}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--our_seed", type=int, default=1, choices=[0, 1, 2, 3])
    ap.add_argument("--ecgfm_seed", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    B = a.B
    make_fig2(a.our_seed, a.ecgfm_seed)
    make_fig3(a.our_seed, a.ecgfm_seed)
    dump_numbers(a.our_seed, a.ecgfm_seed, f"{OUT}/fig23_numbers_our{a.our_seed}_ecgfm{a.ecgfm_seed}.json")
