"""
Compute CHARGE-AF and PREVENT-HF clinical risk scores from raw factors, reproducing the paper's R
pipeline (risk_score/analysis/*.Rmd): per-cohort column maps, MICE imputation (4 sets), fixed
published coefficients. Outputs a per-cohort risk-score CSV (keyed by id) for the standalone
"Risk Score" bars and for the +Risk late fusion.

Usage: python risk_score.py --cohort CHS|MESA|UKBB [--eval]
"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, average_precision_score

# --- repo path configuration (see docs/PATHS.md) ---
import os as _os, sys as _sys
_pd = _os.path.dirname(_os.path.abspath(__file__))
while _pd != "/" and not _os.path.isdir(_os.path.join(_pd, "common")):
    _pd = _os.path.dirname(_pd)
_sys.path.insert(0, _os.path.join(_pd, "common"))
from paths import P


CP = P("RISK_ROOT", "CHARGE-PREVENT")
OUTDIR = P("RISK_ROOT", "computed")

# per-cohort raw-column -> canonical name (canonical: age race height weight sbp dbp cursmoke htnmed
# prevdm prevhf prevmi sex bmi egfr). Matches the R Rmd term mappings exactly.
MAPS = {
    "CHS":  dict(id="seqid", age="age", race="race01", height="htimp", weight="weight", sbp="sbp",
                 dbp="dbp", cursmoke="cursmk", htnmed="htnmed", prevdm="dm", prevhf="prevhf",
                 prevmi="prevmi", sex="gend01", bmi="bmiimp", egfr="egfr"),
    "MESA": dict(id="idno_visit", age="agec", race="racec", height="htcm", weight="wtlb", sbp="sbpc",
                 dbp="dbpc", cursmoke="cigc", htnmed="htnmedc", prevdm="dm03c", prevhf="prevhf",
                 prevmi="prevmi", sex="gender", bmi="bmic", egfr="cepgfrc"),
    "UKBB": dict(id="eid_visit", age="agecmr", race="ethnicity", height="ht", weight="wt", sbp="sbp",
                 dbp="dbp", cursmoke="cursmoke", htnmed="htnmed", prevdm="dm", prevhf="prevhf",
                 prevmi="hxmi", sex="sex", bmi="bmi", egfr="egfrcr"),
}
NUMERIC = ["age", "height", "weight", "sbp", "dbp", "bmi", "egfr"]          # MICE-imputed
BINARY = ["cursmoke", "htnmed", "prevdm", "prevhf", "prevmi", "sex"]        # imputed then rounded to 0/1
# race is a CATEGORICAL CODE, not 0/1: UKB ethnicity 1-6, CHS race01 1-5, MESA racec 0/1. CHARGE-AF
# uses (race == 1) [white], matching the paper's R (`ifelse(ethnicity/race01/racec == 1, 0.465, 0)`).
# It must NOT be clipped to [0,1] -- that collapses every non-missing code to 1, making the race term a
# constant for UKB/CHS and destroying its contribution. (Bug found 2026-07-19.)
CATEG = ["race"]


def charge_af(d):  # CHARGE-AF; d has canonical columns
    # s = linear predictor (Alonso et al., J Am Heart Assoc 2013, Table 4 coefficients).
    s = (d.age/5*0.508 + (d.race == 1)*0.465 + d.height/10*0.248 + d.weight/15*0.115
         + d.sbp/20*0.197 + d.dbp/10*(-0.101) + (d.cursmoke == 1)*0.359 + (d.htnmed == 1)*0.349
         + (d.prevdm == 1)*0.237 + (d.prevhf == 1)*0.701 + (d.prevmi == 1)*0.496)
    # 5-year risk = 1 - S0^exp(s - mean), S0 = 0.9718412736, mean = 12.5815600.
    # The exp() is NOT optional: without it the exponent goes negative whenever s < 12.5816
    # (37.9% of MESA, and a similar share of CHS/UKB), and since S0 < 1 that makes S0^(negative)
    # > 1, i.e. a NEGATIVE "probability" -- observed down to -10.6%. The earlier R pipeline
    # (risk_score/analysis/CHARGE-AF*.Rmd) and the old model.py both omitted it, and this port
    # inherited the omission. It is invisible to AUROC/AUPRC/HR/C-index -- both forms are
    # monotone increasing in s, so every rank-based metric is bit-identical -- but it makes any
    # ABSOLUTE risk wrong, which is what calibration, IDI/NRI and decision-curve analysis read.
    # (Bug found 2026-08-12.)
    return 1 - 0.9718412736**np.exp(s - 12.5815600)


def prevent_hf(d):  # AHA PREVENT-HF (sex-specific), verbatim from model.py / PREVENT-HF.Rmd
    male = d.sex == 1
    a = (d.age-55)/10
    sl = (np.minimum(d.sbp, 110)-110)/20;  sh = (np.maximum(d.sbp, 110)-130)/20
    bl = (np.minimum(d.bmi, 30)-25)/5;      bh = (np.maximum(d.bmi, 30)-30)/5
    el = (np.minimum(d.egfr, 60)-60)/(-15); eh = (np.maximum(d.egfr, 60)-90)/(-15)
    dm = (d.prevdm == 1).astype(float); sm = (d.cursmoke == 1).astype(float); hm = (d.htnmed == 1).astype(float)
    lw = (-4.310409 + 0.8998235*a - 0.4559771*sl + 0.3576505*sh + 1.038346*dm + 0.583916*sm
          - 0.0072294*bl + 0.2997706*bh + 0.7451638*el + 0.0557087*eh + 0.3534442*hm
          - 0.0981511*hm*sh - 0.0946663*a*sh - 0.3581041*a*dm - 0.1159453*a*sm - 0.0038780*a*bh - 0.1884289*a*el)
    lm = (-3.946391 + 0.8972642*a - 0.6811466*sl + 0.3634461*sh + 0.923776*dm + 0.5023736*sm
          - 0.0485841*bl + 0.3726929*bh + 0.6926917*el + 0.0251827*eh + 0.2980922*hm
          - 0.0497731*hm*sh - 0.1289201*a*sh - 0.3040924*a*dm - 0.1401688*a*sm + 0.0068126*a*bh - 0.1797778*a*el)
    lo = np.where(male, lm, lw)
    return 1/(1+np.exp(-lo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["CHS", "MESA", "UKBB"])
    ap.add_argument("--eval", action="store_true", help="also print standalone AUROC/AUPRC vs af5/hf5")
    args = ap.parse_args()
    m = MAPS[args.cohort]
    raw = pd.read_csv(f"{CP}/charge_prevent_{args.cohort}.csv")
    d = raw.rename(columns={v: k for k, v in m.items()})[list(m.keys())].copy()
    d["id"] = d["id"].astype(str)

    # MICE x4 (like the paper's 4 imputed datasets); binary cols rounded/clipped after imputation
    scores = {"charge": [], "prevent": []}
    feats = NUMERIC + BINARY + CATEG
    for s in range(4):
        imp = IterativeImputer(max_iter=10, random_state=s, sample_posterior=True)
        X = pd.DataFrame(imp.fit_transform(d[feats]), columns=feats, index=d.index)
        for b in BINARY:
            X[b] = X[b].round().clip(0, 1)
        for c in CATEG:
            X[c] = X[c].round()          # keep the code (1=white); (race == 1) is applied in charge_af
        scores["charge"].append(charge_af(X).values)
        scores["prevent"].append(prevent_hf(X))
    out = pd.DataFrame({"id": d["id"].values})
    for k in ["charge", "prevent"]:
        arr = np.vstack(scores[k])                      # 4 x n
        for i in range(4):
            out[f"{k}_{i+1}"] = arr[i]
        out[f"{k}_mean"] = arr.mean(axis=0)
    os.makedirs(OUTDIR, exist_ok=True)
    fout = f"{OUTDIR}/{args.cohort}_riskscore.csv"
    out.to_csv(fout, index=False)
    print(f"[{args.cohort}] wrote {len(out)} rows -> {fout}")

    if args.eval:  # standalone Risk Score bars: CHARGE-AF vs af5, PREVENT-HF vs hf5
        base = (P("EVAL_ROOT", "zeroshot")
                if args.cohort in ("CHS", "MESA") else None)
        if base is None:
            print("  (--eval standalone only wired for CHS/MESA external)"); return
        for outc, col in [("af5", "charge_mean"), ("hf5", "prevent_mean")]:
            r = pd.read_csv(f"{base}/{args.cohort}/m75_ecgfull/{outc}/result.csv")
            r["id"] = r["id"].astype(str)
            j = r.merge(out[["id", col]], on="id").dropna(subset=["y_true", col])
            y = j.y_true.values
            print(f"  {args.cohort} {outc} standalone {col.split('_')[0]}: "
                  f"AUROC={roc_auc_score(y, j[col]):.4f}  AUPRC={average_precision_score(y, j[col]):.4f}  (n={len(j)})")


if __name__ == "__main__":
    main()
