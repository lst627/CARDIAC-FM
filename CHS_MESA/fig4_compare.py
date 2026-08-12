"""
Fig 4 arm comparison, CHS + MESA — LIKE-FOR-LIKE (single split / single seed).

Arms (all zero-shot, full follow-up = the paper's actual behaviour):
  RiskScore   : our CHARGE-AF / PREVENT-HF (charge_mean / prevent_mean)
  ours_seed1  : our m75 zero-shot + Risk Score fusion           (m75_ecgfull_RS)

Tertile HRs are now ADJUSTED for baseline covariates (advisor comment): Cox(Surv ~ tertile_int +
tertile_high + covariates), MICE-imputed (m=4) and pooled by Rubin's rules. The paper reports each
arm's HR/CI but never tests the difference between arms, so we add:
  (a) paired bootstrap on  Delta = log HR_high(A) - log HR_high(B)  [tertiles RE-CUT per resample]
  (b) paired bootstrap on  Delta C-index  using the CONTINUOUS scores
Resampling is stratified by event status. p = 2*min(Pr(D>=0), Pr(D<=0)).

Two-stage pipeline (so re-formatting the figure never re-runs the bootstrap):
  compute -> cache all numbers to <outdir>/fig4_stats_<seedtag>.json
  render  -> print the tables + draw the figure PURELY from that JSON
Pass --recompute to force the bootstrap to re-run.

Usage: python fig4_compare.py [--B 1000] [--ours_root .../eval_3] [--figdir .../figures] [--recompute]
"""
import argparse, os, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
warnings.filterwarnings("ignore")

# baseline covariates for the ADJUSTED tertile Cox (advisor: adjust for covariates, not univariable).
CHS_RF = ["age", "chol", "hdl", "sbp", "bmiimp", "egfr", "dm", "cursmk", "lipid", "htnmed", "gender"]
MESA_RF = ["agec", "sbpc", "bmic", "cepgfrc", "dm03c", "cigc", "htnmedc", "gender"]
CP = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/CHARGE-PREVENT"

IDD = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/csv_train_valid_test_individual_id_disease"
RS = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/computed"
EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"


def tert(x):
    q1, q2 = np.nanquantile(x, [1 / 3, 2 / 3])
    g = np.zeros(len(x), int)                 # 0 = low (reference)
    g[(x > q1) & (x <= q2)] = 1               # intermediate
    g[x > q2] = 2                             # high
    return g


def loghr_high(T, E, g):
    """univariable Cox on the tertile factor; returns log HR for high-vs-low (nan if unfittable)."""
    d = pd.DataFrame({"T": T, "E": E, "intermediate": (g == 1).astype(int), "high": (g == 2).astype(int)})
    d = d[(d["T"] > 0)]
    if d["E"].sum() < 5 or d["high"].sum() < 5:
        return np.nan
    try:
        return CoxPHFitter().fit(d, "T", "E").params_["high"]
    except Exception:
        return np.nan


def mice(raw, cols, binary, m=4):
    """multiple imputation of the covariate table -> m completed frames (keyed by id)."""
    imps = []
    for s in range(m):
        im = IterativeImputer(max_iter=10, random_state=s, sample_posterior=True)
        X = pd.DataFrame(im.fit_transform(raw[cols]), columns=cols, index=raw.index)
        for b in binary:
            if b in X:
                X[b] = X[b].round().clip(0, 1)
        X.insert(0, "id", raw["id"].values)
        imps.append(X)
    return imps


def adj_hr_ci(ids, T, E, s, cov_imps, cov_cols, m):
    """ADJUSTED tertile HR: Cox(Surv ~ tertile_int + tertile_high + covariates) per imputation,
    tertiles cut on the score s; pool the tertile HRs across imputations by Rubin's rules."""
    g = tert(s)
    base = pd.DataFrame({"id": np.asarray(ids), "T": np.asarray(T, float), "E": np.asarray(E, int),
                         "intermediate": (g == 1).astype(int), "high": (g == 2).astype(int)})
    betas = {"intermediate": [], "high": []}; vars = {"intermediate": [], "high": []}
    for imp in cov_imps:
        d = base.merge(imp, on="id", how="inner")
        d = d[d["T"] > 0].copy()
        if d["E"].sum() < 5 or d["high"].sum() < 5:
            continue
        for c in cov_cols:                       # z-score covariates (HR per 1 SD for them; tertile HRs unaffected)
            sd = d[c].std(); d[c] = (d[c] - d[c].mean()) / (sd if sd > 0 else 1.0)
        X = d[["intermediate", "high"] + cov_cols].copy()
        X["T"] = d["T"].values; X["E"] = d["E"].values
        try:
            fit = CoxPHFitter().fit(X, "T", "E")
            for lv in ["intermediate", "high"]:
                betas[lv].append(fit.params_[lv]); vars[lv].append(fit.standard_errors_[lv] ** 2)
        except Exception:
            continue
    out = {}
    for lv in ["intermediate", "high"]:
        if not betas[lv]:
            out[lv] = (np.nan, np.nan, np.nan); continue
        b = np.array(betas[lv]); w = np.array(vars[lv])
        bbar = b.mean(); Tvar = w.mean() + (1 + 1 / m) * (b.var(ddof=1) if len(b) > 1 else 0.0)
        se = np.sqrt(Tvar)
        out[lv] = (np.exp(bbar), np.exp(bbar - 1.96 * se), np.exp(bbar + 1.96 * se))
    return out


def unadj_hr_ci(T, E, s):
    """UNADJUSTED tertile HR: univariable Cox on the tertile factor (intermediate + high vs low),
    analytic 95% CI. The pre-covariate-adjustment version of adj_hr_ci."""
    g = tert(s)
    d = pd.DataFrame({"T": np.asarray(T, float), "E": np.asarray(E, int),
                      "intermediate": (g == 1).astype(int), "high": (g == 2).astype(int)})
    d = d[d["T"] > 0]
    out = {}
    try:
        su = CoxPHFitter().fit(d[["T", "E", "intermediate", "high"]], "T", "E").summary
        for lv in ["intermediate", "high"]:
            out[lv] = (su.loc[lv, "exp(coef)"], su.loc[lv, "exp(coef) lower 95%"], su.loc[lv, "exp(coef) upper 95%"])
    except Exception:
        for lv in ["intermediate", "high"]:
            out[lv] = (np.nan, np.nan, np.nan)
    return out


def paired_boot(T, E, sA, sB, B, rng):
    """returns (dLogHR, ci_lo, ci_hi, p, dC, ci_lo, ci_hi, p) -- tertiles re-cut inside each resample."""
    pos, neg = np.where(E == 1)[0], np.where(E == 0)[0]
    obs_h = loghr_high(T, E, tert(sA)) - loghr_high(T, E, tert(sB))
    obs_c = concordance_index(T, -sA, E) - concordance_index(T, -sB, E)
    dh, dc = [], []
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        Tb, Eb = T[idx], E[idx]
        a = loghr_high(Tb, Eb, tert(sA[idx])) - loghr_high(Tb, Eb, tert(sB[idx]))
        if np.isfinite(a):
            dh.append(a)
        dc.append(concordance_index(Tb, -sA[idx], Eb) - concordance_index(Tb, -sB[idx], Eb))
    dh, dc = np.array(dh), np.array(dc)
    ph = 2 * min((dh >= 0).mean(), (dh <= 0).mean())
    pc = 2 * min((dc >= 0).mean(), (dc <= 0).mean())
    return (obs_h, *np.percentile(dh, [2.5, 97.5]), min(ph, 1.0),
            obs_c, *np.percentile(dc, [2.5, 97.5]), min(pc, 1.0))


# ------------------------------------------------------------------ compute (bootstrap; cached) ----
def analyse_cell(coh, outc, ids, T, E, arms, cov_imps, cov_cols, B, rng, m=4):
    """One (cohort, outcome): adjusted tertile HRs per arm + the paired ours-vs-RiskScore test.
    Returns (rows, comparison) as plain JSON-serialisable dicts."""
    rows = []
    for nm, s in arms.items():
        radj = adj_hr_ci(ids, T, E, s, cov_imps, cov_cols, m)
        rune = unadj_hr_ci(T, E, s)
        c = concordance_index(T, -s, E)
        rows.append({"coh": coh, "outc": outc, "arm": nm,
                     "hr_adj":   {lv: [float(v) for v in radj[lv]] for lv in radj},
                     "hr_unadj": {lv: [float(v) for v in rune[lv]] for lv in rune},
                     "c": float(c), "n": int(len(T)), "ev": int(E.sum())})
    pr = paired_boot(T, E, arms["ours_seed1"], arms["RiskScore"], B, rng)
    comp = {"coh": coh, "outc": outc,
            "dloghr": float(pr[0]), "dloghr_ci": [float(pr[1]), float(pr[2])], "p_hr": float(pr[3]),
            "dc": float(pr[4]), "dc_ci": [float(pr[5]), float(pr[6])], "p_c": float(pr[7])}
    return rows, comp


def compute_chs(ourroot, B, rng):
    rows, comps = [], []
    chs = pd.read_csv(f"{IDD}/CHS_split1.csv"); chs["id"] = chs["id"].astype(str)
    rs = pd.read_csv(f"{RS}/CHS_riskscore.csv"); rs["id"] = rs["id"].astype(str)
    chs = chs.merge(rs[["id", "charge_mean", "prevent_mean"]], on="id", how="left")
    chs_rf = pd.read_csv(f"{IDD}/CHS_rf.csv"); chs_rf["id"] = chs_rf["id"].astype(str)
    chs_cov = mice(chs_rf, CHS_RF, ["dm", "cursmk", "lipid", "htnmed", "gender"], 4)
    # subtypes (hfpef/hfref) reuse the HF (hf5) ECG+RS score against their OWN endpoint, PREVENT-HF risk.
    for outc, oc, riskcol in [("af", "af5", "charge_mean"), ("hf", "hf5", "prevent_mean"),
                              ("hfpef", "hf5", "prevent_mean"), ("hfref", "hf5", "prevent_mean")]:
        ours = pd.read_csv(f"{ourroot}/zeroshot/CHS/m75_ecgfull_RS/{oc}/result.csv")[["id", "y_pred"]]
        ours["id"] = ours["id"].astype(str); ours = ours.rename(columns={"y_pred": "ours"})
        d = chs.merge(ours, on="id", how="inner")
        t, e, p = f"tto{outc}", f"inc{outc}", f"prev{outc}"
        d = d[(d[p] == 0) & d[t].notna() & d[e].notna() & d[riskcol].notna() & d["ours"].notna()]
        T, E = d[t].values.astype(float), d[e].values.astype(int)
        arms = {"RiskScore": d[riskcol].values, "ours_seed1": d["ours"].values}
        r, cmp = analyse_cell("CHS", outc, d["id"].values, T, E, arms, chs_cov, CHS_RF, B, rng, 4)
        rows += r; comps.append(cmp)
        print(f"  [computed CHS {outc.upper()}  n={len(d)} ev={int(E.sum())}]", flush=True)
    return rows, comps


def compute_mesa(ourroot, B, rng):
    """MESA survival from MESA_disease.csv, keyed idno_visit (== our prediction id), carrying the HF
    SUBTYPE endpoints (pef=HFpEF, ref=HFrEF). Each idno_visit is its own time origin."""
    rows, comps = [], []
    DIS = "/gpfs/projects/trend/bojun/CHS_MESA/MESA/MESA_disease.csv"
    surv = pd.read_csv(DIS); surv["id"] = surv["idno_visit"].astype(str)
    rs = pd.read_csv(f"{RS}/MESA_riskscore.csv"); rs["id"] = rs["id"].astype(str)
    mrf = pd.read_csv(f"{CP}/charge_prevent_MESA.csv"); mrf["id"] = mrf["idno_visit"].astype(str)
    mesa_cov = mice(mrf, MESA_RF, ["dm03c", "cigc", "htnmedc", "gender"], 4)
    for outc, oc, riskcol, t, e, p in [
            ("af", "af5", "charge_mean", "ttoaf", "incaf", "prevaf"),
            ("hf", "hf5", "prevent_mean", "ttohf", "inchf", "prevhf"),
            ("hfpef", "hf5", "prevent_mean", "ttopef", "incpef", "prevpef"),
            ("hfref", "hf5", "prevent_mean", "ttoref", "incref", "prevref")]:
        ours = pd.read_csv(f"{ourroot}/zeroshot/MESA/m75_ecgfull_RS/{oc}/result.csv")[["id", "y_pred"]]
        ours["id"] = ours["id"].astype(str); ours = ours.rename(columns={"y_pred": "ours"})
        j = rs.merge(ours, on="id", how="inner").merge(surv[["id", t, e, p]], on="id", how="inner")
        d = j[(j[p] == 0) & j[t].notna() & j[e].notna() & j[riskcol].notna() & j["ours"].notna()].copy()
        if len(d) == 0 or d[e].sum() < 5:
            print(f"  MESA {outc.upper()}: too few events — skipped", flush=True); continue
        T, E = d[t].astype(float).values, d[e].astype(int).values
        arms = {"RiskScore": d[riskcol].values, "ours_seed1": d["ours"].values}
        r, cmp = analyse_cell("MESA", outc, d["id"].values, T, E, arms, mesa_cov, MESA_RF, B, rng, 4)
        rows += r; comps.append(cmp)
        print(f"  [computed MESA {outc.upper()}  n={len(d)} ev={int(E.sum())}]", flush=True)
    return rows, comps


# ------------------------------------------------------------------ render (from cache) ----
def print_tables(stats):
    """Print the HR / comparison tables from the cached numbers (no recomputation)."""
    rows, comps = stats["rows"], stats["comparisons"]
    bar = "=" * 84
    by_cell = {}
    for r in rows:
        by_cell.setdefault((r["coh"], r["outc"]), []).append(r)
    for (coh, outc), rs in by_cell.items():
        n, ev = rs[0]["n"], rs[0]["ev"]
        print(f"\n{bar}\n{coh} {outc.upper()}  (n={n}, events={ev}) — full follow-up, zero-shot\n{bar}")
        for which, key in [("UNADJUSTED", "hr_unadj"), ("covariate-ADJUSTED", "hr_adj")]:
            print(f"-- {which} tertile HR --")
            print(f"{'arm':<14}{'HR int/low [95% CI]':<26}{'HR high/low [95% CI]':<26}{'C-index'}")
            for r in rs:
                hi, hl = r[key]["intermediate"], r[key]["high"]
                print(f"{r['arm']:<14}{hi[0]:.2f} [{hi[1]:.2f}, {hi[2]:.2f}]".ljust(40)
                      + f"{hl[0]:.2f} [{hl[1]:.2f}, {hl[2]:.2f}]".ljust(26) + f"{r['c']:.3f}")
        c = next((x for x in comps if (x["coh"], x["outc"]) == (coh, outc)), None)
        if c:
            print(f"\n{'comparison':<28}{'Δ log HR_high [95% CI]':<30}{'p':<9}{'Δ C-index [95% CI]':<28}{'p'}")
            print(f"{'ours_seed1 vs RiskScore':<28}"
                  f"{c['dloghr']:+.3f} [{c['dloghr_ci'][0]:+.3f}, {c['dloghr_ci'][1]:+.3f}]".ljust(58)
                  + f"{c['p_hr']:.3f}".ljust(9)
                  + f"{c['dc']:+.4f} [{c['dc_ci'][0]:+.4f}, {c['dc_ci'][1]:+.4f}]".ljust(28)
                  + f"{c['p_c']:.3f}")


def make_fig4(figdir, ourroot, rows, which="adjusted"):
    """Paper Fig-4 layout: per cohort, one facet per outcome; at each arm (Risk Score, ECG+RS
    zero-shot) plot BOTH tertile hazard ratios vs the low-risk reference -- Intermediate/Low (red)
    and High/Low (teal) -- with 95% CI. Both cohorts include the two HF subtypes (HFpEF/HFrEF).
    Linear y-axis, dashed null at HR=1. NO figure title (per-facet outcome labels only). Data = rows.
    `which` selects the covariate-'adjusted' or 'unadjusted' tertile HRs (both are cached)."""
    os.makedirs(figdir, exist_ok=True)
    from matplotlib.lines import Line2D
    hrkey = "hr_adj" if which == "adjusted" else "hr_unadj"
    seedtag = "our" + (ourroot.rstrip("/").split("_")[-1] if "eval_" in ourroot else "1")
    cells = {}
    for r in rows:
        cells.setdefault((r["coh"], r["outc"]), {})[r["arm"]] = r
    ARM_ORDER = ["RiskScore", "ours_seed1"]
    ARM_LABEL = {"RiskScore": "Risk Score", "ours_seed1": "ECG + Risk Score"}
    OUTC_TITLE = {"af": "Atrial Fibrillation", "hf": "Heart Failure",
                  "hfpef": "HF with preserved EF (HFpEF)", "hfref": "HF with reduced EF (HFrEF)"}
    SERIES = [("intermediate", "Intermediate/Low", "#F8766D"),   # red
              ("high", "High/Low", "#00BFC4")]                    # teal
    layouts = {"CHS": (["af", "hf", "hfpef", "hfref"], 2, 2, (11, 8)),
               "MESA": (["af", "hf", "hfpef", "hfref"], 2, 2, (11, 8))}
    for coh, (outcs, nr, nc, figsize) in layouts.items():
        if not any((coh, o) in cells for o in outcs):
            continue
        fig, axes = plt.subplots(nr, nc, figsize=figsize, squeeze=False)
        axes = axes.reshape(-1)
        for ax, outc in zip(axes, outcs):
            if (coh, outc) not in cells:
                ax.set_visible(False); continue
            armd = cells[(coh, outc)]
            arms = [a for a in ARM_ORDER if a in armd]
            for xi, arm in enumerate(arms):
                hrd = armd[arm][hrkey]
                for si, (key, _, col) in enumerate(SERIES):
                    hr, lo, hi = hrd[key]
                    xp = xi + (si - 0.5) * 0.24
                    ax.plot([xp, xp], [lo, hi], color=col, lw=2, zorder=2)
                    ax.plot(xp, hr, "o", color=col, ms=7, zorder=3)
            ax.axhline(1.0, color="#444444", ls="--", lw=1, zorder=1)
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels([ARM_LABEL[a] for a in arms], fontsize=9)
            ax.set_xlim(-0.5, len(arms) - 0.5)
            ev0 = armd[arms[0]]["ev"]
            ax.set_title(f"{OUTC_TITLE[outc]}  (events={ev0})", fontsize=10)
            ax.set_ylabel("Hazard ratio (vs. low tertile)", fontsize=8)
            ax.grid(axis="y", alpha=0.25, zorder=0)
            ax.set_ylim(bottom=0)
        for ax in axes[len(outcs):]:
            ax.set_visible(False)
        handles = [Line2D([0], [0], color=col, marker="o", lw=2, label=lab) for _, lab, col in SERIES]
        fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        for ext in ["png", "pdf"]:
            fig.savefig(f"{figdir}/fig4_riskstrat_{coh}_{seedtag}_{which}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig4 {coh} {which} -> {figdir}/fig4_riskstrat_{coh}_{seedtag}_{which}.png/pdf]", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ours_root", default=EV, help="root for OUR model preds, e.g. .../eval_3 for seed 3")
    ap.add_argument("--figdir", default=None, help="if set, also save the risk-stratification figure here")
    ap.add_argument("--stats", default=None, help="cached numbers JSON (default <figdir|.>/fig4_stats_<seedtag>.json)")
    ap.add_argument("--recompute", action="store_true", help="force re-run the MICE + bootstrap even if cache exists")
    a = ap.parse_args()
    seedtag = "our" + (a.ours_root.rstrip("/").split("_")[-1] if "eval_" in a.ours_root else "1")
    stats_path = a.stats or f"{a.figdir or '.'}/fig4_stats_{seedtag}.json"

    if os.path.exists(stats_path) and not a.recompute:
        stats = json.load(open(stats_path))
        print(f"[loaded cached stats <- {stats_path}] (pass --recompute to re-run)", flush=True)
    else:
        rng = np.random.default_rng(a.seed)
        print(f"[computing fig4 stats (B={a.B}, seed={a.seed}, {a.ours_root})]", flush=True)
        cr, cc = compute_chs(a.ours_root, a.B, rng)
        mr, mc = compute_mesa(a.ours_root, a.B, rng)
        stats = {"_meta": {"B": a.B, "seed": a.seed, "ours_root": a.ours_root, "seedtag": seedtag},
                 "rows": cr + mr, "comparisons": cc + mc}
        os.makedirs(os.path.dirname(os.path.abspath(stats_path)), exist_ok=True)
        json.dump(stats, open(stats_path, "w"), indent=2)
        print(f"[wrote stats -> {stats_path}]", flush=True)

    print_tables(stats)
    if a.figdir:
        make_fig4(a.figdir, a.ours_root, stats["rows"], "adjusted")
        make_fig4(a.figdir, a.ours_root, stats["rows"], "unadjusted")


if __name__ == "__main__":
    main()
