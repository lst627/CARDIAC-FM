"""
A1 (external) — Cox on ECG-PREDICTED CMR features in CHS and MESA, risk-factor adjusted,
pooled over 4 MICE imputations by Rubin's rules.

This is the paper's headline use of the CMR regression (Suppl Fig 1-2, Tables 13/16):
**CHS has no cardiac MRI at all**, so ECG-inferred phenotypes are the only way to study
cardiac structure-disease associations there. CHS additionally has the HF subtypes
(HFpEF / HFrEF), where the paper reports the most pronounced pattern.

Method identical to the UKB version (`cmr_feature_cox.py`): z-score features + covariates
(HR per 1 SD), `coxph(Surv(t,e) ~ feature + covariates)` per imputation, pool with
Tvar = W + (1+1/m)B.

⚠️ Predicted features only — neither cohort's measurements are available to us (CHS has no MRI;
MESA's measured CMR is not in our tree). So this analysis *assumes* the predicted≈measured
agreement established in UKB rather than testing it.

Usage: python cmr_feature_cox_external.py
"""
import argparse, os, json, warnings
import numpy as np, pandas as pd
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
from scipy.stats import norm
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from lifelines import CoxPHFitter
warnings.filterwarnings("ignore")

EV = P("EVAL_ROOT")
IDD = P("RISK_ROOT", "csv_train_valid_test_individual_id_disease")
CP = P("RISK_ROOT", "CHARGE-PREVENT")
MD = P("MESA_ECG_ROOT")
FEATS = ["lvef", "laef", "lvm", "lvedv", "lvesv", "lavmin", "lavmax"]
CHS_RF = ["age", "chol", "hdl", "sbp", "bmiimp", "egfr", "dm", "cursmk", "lipid", "htnmed", "gender"]
MESA_RF = ["agec", "sbpc", "bmic", "cepgfrc", "dm03c", "cigc", "htnmedc", "gender"]


def z(df, cols):
    out = df.copy()
    for c in cols:
        s = out[c].astype(float)
        out[c] = (s - s.mean()) / (s.std() if s.std() > 0 else 1.0)
    return out


def mice(raw, cols, binary, m=4):
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


def pool(rows, m):
    df = pd.DataFrame(rows)
    g = df.groupby(["outcome", "feature"])
    p = g.agg(beta_bar=("beta", "mean"), W=("var", "mean"), B=("beta", "var"),
              n=("n", "first"), events=("events", "first")).reset_index()
    p["Tvar"] = p["W"] + (1 + 1 / m) * p["B"]
    p["HR"] = np.exp(p["beta_bar"])
    p["HR_low"] = np.exp(p["beta_bar"] - 1.96 * np.sqrt(p["Tvar"]))
    p["HR_high"] = np.exp(p["beta_bar"] + 1.96 * np.sqrt(p["Tvar"]))
    p["p"] = 2 * norm.cdf(-np.abs(p["beta_bar"] / np.sqrt(p["Tvar"])))
    return p


def fit_all(base_builder, outcomes, rf_cols, imps, m):
    rows = []
    for i, rfi in enumerate(imps, start=1):
        base = base_builder(rfi)
        for outc, (tv, ev, pv) in outcomes.items():
            d0 = base.dropna(subset=[tv, ev])
            if pv and pv in d0.columns:
                d0 = d0[d0[pv] == 0]
            d0 = d0[d0[tv] > 0]
            for f in FEATS:
                col = f"{f}_pred"
                d = d0.dropna(subset=[col] + rf_cols).copy()
                if len(d) < 50 or d[ev].sum() < 10:
                    continue
                d = z(d, [col] + rf_cols)
                X = d[[col] + rf_cols].copy()
                X["T"] = d[tv].astype(float).values
                X["E"] = d[ev].astype(int).values
                try:
                    fit = CoxPHFitter().fit(X, "T", "E")
                    rows.append(dict(imp=i, outcome=outc, feature=f, beta=fit.params_[col],
                                     var=fit.standard_errors_[col] ** 2, n=len(d),
                                     events=int(X["E"].sum())))
                except Exception as e:
                    print(f"  !! {outc} {col} imp{i}: {str(e)[:60]}")
        print(f"  imputation {i} done", flush=True)
    return pool(rows, m) if rows else None


def load_pred(coh, extdir):
    out = None
    for f in FEATS:
        p = pd.read_csv(f"{extdir}/{coh}/{f}/result.csv")[["id", "y_pred"]]
        p["id"] = p["id"].astype(str)
        p = p.rename(columns={"y_pred": f"{f}_pred"}).dropna(subset=[f"{f}_pred"])
        out = p if out is None else out.merge(p, on="id", how="outer")
    return out


def make_supp_cox(series, cohort, outcomes, figdir, seedtag, fignum):
    """Supp Fig 1/2: per outcome (facet), HR per 1 SD of each CMR feature with 95% CI. `series` is a
    list of (pooled_df, label, colour): one for CHS (ECG-predicted), two for MESA where measured
    cardiac MRI is available (ECG-predicted vs MRI-measured, dodged side-by-side)."""
    from matplotlib.lines import Line2D
    series = [(p, lab, col) for (p, lab, col) in series if p is not None]
    if not series:
        return
    os.makedirs(figdir, exist_ok=True)
    xorder = ["laef", "lavmax", "lavmin", "lvedv", "lvesv", "lvm", "lvef"]
    OUT_TITLE = {"af": "AF", "hf": "HF", "hfref": "HFrEF", "hfpef": "HFpEF"}
    allout = set().union(*[set(p.outcome.unique()) for p, _, _ in series])
    outs = [o for o in outcomes if o in allout]
    ns = len(series)
    fig, axes = plt.subplots(len(outs), 1, figsize=(8, 2.1 * len(outs)), squeeze=False, sharex=True)
    axes = axes.reshape(-1)
    for ax, outc in zip(axes, outs):
        feats_present = None
        for si, (p, lab, col) in enumerate(series):
            s = p[p.outcome == outc].set_index("feature")
            feats = [f for f in xorder if f in s.index]
            feats_present = feats_present or feats
            for xi, f in enumerate(feats):
                r = s.loc[f]
                xp = xi + ((si - (ns - 1) / 2) * 0.18 if ns > 1 else 0)
                ax.plot([xp, xp], [r.HR_low, r.HR_high], color=col, lw=2, zorder=2)
                ax.plot(xp, r.HR, "o", color=col, ms=6, zorder=3)
        ax.axhline(1.0, color="#444444", ls="--", lw=1, zorder=1)
        ff = feats_present or []
        ax.set_xticks(range(len(ff))); ax.set_xticklabels([f.upper() for f in ff], fontsize=8, rotation=30)
        ax.set_ylabel("HR (per 1 SD)", fontsize=8); ax.set_title(OUT_TITLE.get(outc, outc), fontsize=9, loc="right")
        ax.grid(axis="y", alpha=0.2, zorder=0)
    if ns > 1:
        handles = [Line2D([0], [0], color=col, marker="o", lw=2, label=lab) for _, lab, col in series]
        fig.legend(handles=handles, loc="upper center", ncol=ns, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.96 if ns > 1 else 0.99])  # no figure title (per-facet outcome labels only)
    for ext in ["png", "pdf"]:
        fig.savefig(f"{figdir}/supp{fignum}_cmr_cox_{cohort}_{seedtag}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[supp{fignum} {cohort} -> {figdir}/supp{fignum}_cmr_cox_{cohort}_{seedtag}.png/pdf]", flush=True)


def emit(L, title, p, note=""):
    L.append(f"\n## {title}\n")
    if p is None:
        L.append("*(no fits — insufficient events)*"); return
    if note:
        L.append(note + "\n")
    for outc in p.outcome.unique():
        s = p[p.outcome == outc]
        L.append(f"\n### {outc.upper()}  (n={int(s.n.iloc[0])}, events={int(s.events.iloc[0])})\n")
        L.append("| feature | HR per 1 SD [95% CI] | p |")
        L.append("|---|---|---|")
        for f in FEATS:
            r = s[s.feature == f]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            L.append(f"| {f.upper()} | {r.HR:.2f} [{r.HR_low:.2f}, {r.HR_high:.2f}] | {r.p:.3g} |")


def compute_pooled(extdir, m):
    """Run the MICE + pooled Cox for all three series (CHS predicted, MESA predicted, MESA measured).
    Returns {name: pooled DataFrame or None}. This is the only expensive step."""
    # ---------------- CHS ----------------
    print("CHS ...", flush=True)
    surv = pd.read_csv(f"{IDD}/CHS_split1.csv"); surv["id"] = surv["id"].astype(str)
    rf = pd.read_csv(f"{IDD}/CHS_rf.csv"); rf["id"] = rf["id"].astype(str)
    pred = load_pred("CHS", extdir)
    imps = mice(rf, CHS_RF, ["dm", "cursmk", "lipid", "htnmed", "gender"], m)
    outcomes = {"af": ("ttoaf", "incaf", "prevaf"), "hf": ("ttohf", "inchf", "prevhf"),
                "hfpef": ("ttohfpef", "inchfpef", "prevhfpef"), "hfref": ("ttohfref", "inchfref", "prevhfref")}
    p = fit_all(lambda rfi: surv.merge(pred, on="id").merge(rfi, on="id"), outcomes, CHS_RF, imps, m)

    # ---------------- MESA ----------------
    print("MESA ...", flush=True)
    # MESA_disease.csv is keyed by idno_visit (== our prediction id) and carries the HF SUBTYPE
    # endpoints (pef = HFpEF, ref = HFrEF) that the split files lack -> reproduces Supp Fig 2 subtypes.
    DIS = P("MESA_TABLES", "MESA_disease.csv")
    OUTC_COLS = ["ttoaf", "incaf", "prevaf", "ttohf", "inchf", "prevhf",
                 "ttopef", "incpef", "prevpef", "ttoref", "incref", "prevref"]
    msurv = pd.read_csv(DIS); msurv["id"] = msurv["idno_visit"].astype(str)
    msurv = msurv[["id"] + OUTC_COLS]
    mrf = pd.read_csv(f"{CP}/charge_prevent_MESA.csv")
    mrf["id"] = mrf["idno_visit"].astype(str)
    mimps = mice(mrf, MESA_RF, ["dm03c", "cigc", "htnmedc", "gender"], m)
    mpred = load_pred("MESA", extdir)
    mout = {"af": ("ttoaf", "incaf", "prevaf"), "hf": ("ttohf", "inchf", "prevhf"),
            "hfpef": ("ttopef", "incpef", "prevpef"), "hfref": ("ttoref", "incref", "prevref")}
    mp = fit_all(lambda rfi: mpred.merge(rfi, on="id").merge(msurv, on="id"), mout, MESA_RF, mimps, m)

    # advisor: MESA HAS measured cardiac MRI -> also run the Cox on the MRI-MEASURED CMR features
    # (MESA_CMR_features.csv, keyed by id_visit == our prediction id). Second series in Supp Fig 2.
    meas = pd.read_csv(P("MESA_TABLES", "MESA_CMR_features.csv"))
    meas["id"] = meas["id_visit"].astype(str)
    meas = meas.rename(columns={f: f"{f}_pred" for f in FEATS})[["id"] + [f"{f}_pred" for f in FEATS]]
    mp_meas = fit_all(lambda rfi: meas.merge(rfi, on="id").merge(msurv, on="id"), mout, MESA_RF, mimps, m)

    return {"CHS_pred": p, "MESA_pred": mp, "MESA_meas": mp_meas}


def render(tables, out, figdir, seedtag):
    """Write the markdown tables + Supp Fig 1/2 purely from the cached pooled tables."""
    p, mp, mp_meas = tables["CHS_pred"], tables["MESA_pred"], tables["MESA_meas"]
    L = ["# CHS / MESA — Cox on ECG-PREDICTED CMR features (risk-factor adjusted)", "",
         "HR per 1 SD. Pooled over 4 MICE imputations (Rubin's rules). Prevalent cases excluded.",
         "CHS has no cardiac MRI (ECG-predicted only). MESA DOES have measured cardiac MRI, so its "
         "section reports BOTH ECG-predicted and MRI-measured CMR features side by side."]
    emit(L, "CHS — no cardiac MRI in this cohort", p,
         f"Adjusted for: {', '.join(CHS_RF)}. Full follow-up (CHS `tto*` is untruncated, ~26 yr).")
    emit(L, "MESA — ECG-predicted CMR (subtypes incl.)", mp,
         f"Adjusted for: {', '.join(MESA_RF)}. Each exam is its own time origin; a participant with "
         "both exams contributes twice (within-person correlation unmodelled → CIs slightly optimistic).")
    emit(L, "MESA — MRI-MEASURED CMR (subtypes incl.)", mp_meas,
         f"Same Cox, adjusted for {', '.join(MESA_RF)}, but using MEASURED cardiac-MRI features "
         "(validates the ECG-predicted associations against ground truth).")
    if figdir:
        make_supp_cox([(p, "ECG-predicted", "#1b9e9e")], "CHS", ["af", "hf", "hfref", "hfpef"],
                      figdir, seedtag, 1)
        make_supp_cox([(mp, "ECG-predicted", "#1b9e9e"), (mp_meas, "MRI-measured", "#d95f02")],
                      "MESA", ["af", "hf", "hfref", "hfpef"], figdir, seedtag, 2)
    txt = "\n".join(L)
    open(out, "w").write(txt + "\n")
    print(txt); print(f"\n[written to {out}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1, help="which CMR-regression seed's external preds to use")
    ap.add_argument("--figdir", default=None, help="if set, also write Supp Fig 1 (CHS) + Supp Fig 2 (MESA)")
    ap.add_argument("--stats", default=None, help="cached pooled-Cox JSON (default alongside --out)")
    ap.add_argument("--recompute", action="store_true", help="force re-run the MICE + Cox even if cache exists")
    a = ap.parse_args()
    seedtag = f"seed{a.seed}"
    extdir = f"{EV}/a1/cmr_pred_external" if a.seed == 1 else f"{EV}/a1/cmr_pred_external_seed{a.seed}"
    if a.out is None:
        a.out = f"{EV}/a1/cmr_feature_cox_external{'' if a.seed == 1 else f'_seed{a.seed}'}.md"
    stats_path = a.stats or a.out.replace(".md", "_stats.json")

    # ---- compute (MICE + Cox) once and cache; on later format tweaks just reload the JSON ----
    if os.path.exists(stats_path) and not a.recompute:
        raw = json.load(open(stats_path))
        tables = {k: (pd.DataFrame(v) if v else None) for k, v in raw.items() if not k.startswith("_")}
        print(f"[loaded cached pooled tables <- {stats_path}] (pass --recompute to re-run)", flush=True)
    else:
        tables = compute_pooled(extdir, a.m)
        raw = {k: (v.to_dict("records") if v is not None else None) for k, v in tables.items()}
        raw["_meta"] = {"m": a.m, "seed": a.seed, "extdir": extdir}
        json.dump(raw, open(stats_path, "w"), indent=2)
        print(f"[wrote pooled tables -> {stats_path}]", flush=True)

    render(tables, a.out, a.figdir, seedtag)


if __name__ == "__main__":
    main()
