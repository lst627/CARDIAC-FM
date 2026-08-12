"""
A1 (UKB part) — Cox on CMR features, MEASURED vs ECG-PREDICTED, adjusted for traditional risk
factors, pooled over 4 MICE imputations by Rubin's rules.

Reproduces `Hazard Ratio.Rmd` ("HR for CMR feature"):
  * features and risk factors are Z-SCORED  -> HR is per 1 SD (why the paper reports ~1.4-1.6)
  * per imputation i:  coxph(Surv(time, event) ~ <cmr_feature> + <risk factors>)
  * Rubin's rules:  beta_bar = mean(beta);  W = mean(var);  B = var(beta)
                    Tvar = W + (1 + 1/m)*B;  HR = exp(beta_bar);  CI = exp(beta_bar +/- 1.96*sqrt(Tvar))
                    z = beta_bar/sqrt(Tvar);  p = 2*Phi(-|z|)
  * fitted separately for `true` (measured) and `predicted` features -> lets us ask whether the
    ECG-inferred phenotype carries the same prognostic signal as the measured one.

Population: paired UKB test (`csv_HR/ecgmri.csv`), prevalent cases excluded per outcome.
Our predictions are seed 1 (the paper averages 4 splits before z-scoring).

Usage: python cmr_feature_cox.py [--out <md>]
"""
import argparse, warnings
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from lifelines import CoxPHFitter

# --- repo path configuration (see docs/PATHS.md) ---
import os as _os, sys as _sys
_pd = _os.path.dirname(_os.path.abspath(__file__))
while _pd != "/" and not _os.path.isdir(_os.path.join(_pd, "common")):
    _pd = _os.path.dirname(_pd)
_sys.path.insert(0, _os.path.join(_pd, "common"))
from paths import P

warnings.filterwarnings("ignore")

EV = P("EVAL_ROOT")
CP = P("RISK_ROOT", "CHARGE-PREVENT")
HR = P("RISK_ROOT", "csv_HR")
FEATS = ["lvef", "laef", "lvm", "lvedv", "lvesv", "lavmin", "lavmax"]
# traditional risk factors available in charge_prevent_UKBB.csv (the paper additionally had
# cholesterol/HDL/lipid-lowering, which are not in our file)
RF = ["agecmr", "sex", "sbp", "bmi", "egfrcr", "dm", "cursmoke", "htnmed"]
OUTCOMES = {"af": ("ttoaf", "afcens", "prevaf"), "hf": ("ttohf", "hfcens", "prevhf")}


def z(df, cols):
    out = df.copy()
    for c in cols:
        s = out[c].astype(float)
        out[c] = (s - s.mean()) / (s.std() if s.std() > 0 else 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{EV}/a1/cmr_feature_cox.md")
    ap.add_argument("--m", type=int, default=4, help="number of MICE imputations")
    a = ap.parse_args()

    # ---- survival + prevalent flags (paired UKB test) ----
    surv = pd.read_csv(f"{HR}/ecgmri.csv"); surv["eid_visit"] = surv["eid_visit"].astype(str)
    keep = ["eid_visit"] + sorted({c for t, e, p in OUTCOMES.values() for c in (t, e, p)})
    surv = surv[[c for c in keep if c in surv.columns]]

    # ---- measured CMR (truth) ----
    truth = pd.read_csv(f"{EV}/UKBB_cmr_true_test.csv"); truth["eid_visit"] = truth["eid_visit"].astype(str)

    # ---- our ECG-predicted CMR (seed 1) ----
    pred = truth[["eid_visit"]].copy()
    for f in FEATS:
        p = pd.read_csv(f"{EV}/cmr_reg/m75_seed1/{f}/test/result.csv")[["id", "y_pred"]]
        p["eid_visit"] = p["id"].astype(str)
        pred = pred.merge(p[["eid_visit", "y_pred"]].rename(columns={"y_pred": f"{f}_pred"}),
                          on="eid_visit", how="left")

    # ---- risk factors: MICE x m ----
    raw = pd.read_csv(f"{CP}/charge_prevent_UKBB.csv")
    raw["eid_visit"] = raw["eid_visit"].astype(str)
    imps = []
    for s in range(a.m):
        imp = IterativeImputer(max_iter=10, random_state=s, sample_posterior=True)
        X = pd.DataFrame(imp.fit_transform(raw[RF]), columns=RF, index=raw.index)
        for b in ["sex", "dm", "cursmoke", "htnmed"]:
            X[b] = X[b].round().clip(0, 1)
        X.insert(0, "eid_visit", raw["eid_visit"].values)
        imps.append(X)
        print(f"  imputation {s+1}/{a.m} done", flush=True)

    rows = []
    for i, rfi in enumerate(imps, start=1):
        base = (surv.merge(truth[["eid_visit"] + FEATS], on="eid_visit")
                    .merge(pred, on="eid_visit").merge(rfi, on="eid_visit"))
        for outc, (tv, ev, pv) in OUTCOMES.items():
            d0 = base[(base[pv] == 0)].dropna(subset=[tv, ev]) if pv in base else base.dropna(subset=[tv, ev])
            d0 = d0[d0[tv] > 0]
            for f in FEATS:
                for col, typ in [(f, "measured"), (f"{f}_pred", "predicted")]:
                    d = d0.dropna(subset=[col] + RF).copy()
                    d = z(d, [col] + RF)
                    X = d[[col] + RF].copy()
                    X["T"] = d[tv].astype(float).values
                    X["E"] = d[ev].astype(int).values
                    try:
                        fit = CoxPHFitter().fit(X, "T", "E")
                        rows.append(dict(imp=i, outcome=outc, feature=f, type=typ,
                                         beta=fit.params_[col], var=fit.standard_errors_[col] ** 2,
                                         n=len(d), events=int(X["E"].sum())))
                    except Exception as e:
                        print(f"  !! {outc} {col} imp{i}: {str(e)[:60]}")
        print(f"  Cox fits done for imputation {i}", flush=True)

    df = pd.DataFrame(rows)
    g = df.groupby(["outcome", "feature", "type"])
    pool = g.agg(m=("beta", "size"), beta_bar=("beta", "mean"), W=("var", "mean"),
                 B=("beta", "var"), n=("n", "first"), events=("events", "first")).reset_index()
    pool["Tvar"] = pool["W"] + (1 + 1 / pool["m"]) * pool["B"]
    pool["HR"] = np.exp(pool["beta_bar"])
    pool["HR_low"] = np.exp(pool["beta_bar"] - 1.96 * np.sqrt(pool["Tvar"]))
    pool["HR_high"] = np.exp(pool["beta_bar"] + 1.96 * np.sqrt(pool["Tvar"]))
    pool["p"] = 2 * norm.cdf(-np.abs(pool["beta_bar"] / np.sqrt(pool["Tvar"])))

    L = ["# UKB — Cox on CMR features (measured vs ECG-predicted), risk-factor adjusted",
         "",
         f"Per 1 SD. Adjusted for: {', '.join(RF)}. Pooled over {a.m} MICE imputations "
         "(Rubin's rules). Paired UKB test; prevalent cases excluded per outcome. "
         "Our predictions are **seed 1**.", ""]
    for outc in OUTCOMES:
        s = pool[pool.outcome == outc]
        L.append(f"\n## {outc.upper()}  (n={int(s.n.iloc[0])}, events={int(s.events.iloc[0])})\n")
        L.append("| feature | measured HR [95% CI] | p | predicted HR [95% CI] | p |")
        L.append("|---|---|---|---|---|")
        for f in FEATS:
            r = {t: s[(s.feature == f) & (s.type == t)] for t in ["measured", "predicted"]}
            if any(len(v) == 0 for v in r.values()):
                continue
            m_, p_ = r["measured"].iloc[0], r["predicted"].iloc[0]
            L.append(f"| {f.upper()} | {m_.HR:.2f} [{m_.HR_low:.2f}, {m_.HR_high:.2f}] | {m_.p:.3g} "
                     f"| {p_.HR:.2f} [{p_.HR_low:.2f}, {p_.HR_high:.2f}] | {p_.p:.3g} |")
    txt = "\n".join(L)
    open(a.out, "w").write(txt + "\n")
    pool.to_csv(a.out.replace(".md", ".csv"), index=False)
    print(txt); print(f"\n[written to {a.out}]")


if __name__ == "__main__":
    main()
