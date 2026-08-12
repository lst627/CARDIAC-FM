"""
Reviewer-2 analysis: does the ECG add value over a REFIT clinical model (not the fixed published
CHARGE-AF / PREVENT-HF formula)?

Design mirrors the ECG model's own protocol so the comparison is fair:
  - fit on UKB train, evaluate on UKB test;
  - apply the UKB-fit model ZERO-SHOT to CHS and MESA (no in-cohort refit) -- exactly how the ECG
    model is transferred externally.

Per outcome (af5 -> CHARGE-AF variable set; hf5 -> PREVENT-HF variable set) we fit, on UKB train:
  clinical LR : logistic(outcome ~ individual variables)          [the refit baseline]
  full LR     : logistic(outcome ~ individual variables + ECG)    [does ECG add?]
then report, on UKB test + zero-shot CHS/MESA: AUROC of {clinical, ECG-only, full}, and the paired
bootstrap ΔAUROC (full - clinical) and (ECG - clinical) with p. This answers the reviewer in their
own metric, against the stronger refit baseline they requested.

NOTE: a refit LR estimates its own coefficients, so it is robust to the UKB unit/scale mis-mapping
that broke the fixed CHARGE-AF/PREVENT-HF score (ukb-risk-score-mapping-bug) -- unit errors are
absorbed into the fitted weights.

  python refit_clinical_compare.py [--B 2000] [--ours_root .../eval_3] [--out <json>]
"""
import argparse, os, sys, json, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

# --- repo path configuration (see docs/PATHS.md) ---
import os as _os, sys as _sys
_pd = _os.path.dirname(_os.path.abspath(__file__))
while _pd != "/" and not _os.path.isdir(_os.path.join(_pd, "common")):
    _pd = _os.path.dirname(_pd)
_sys.path.insert(0, _os.path.join(_pd, "common"))
from paths import P

warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # cardiacfm_new/
sys.path.insert(0, os.path.join(_ROOT, "common", "risk"))
from risk_score import MAPS                        # per-cohort raw->canonical column maps

CP = P("RISK_ROOT", "CHARGE-PREVENT")
EV = P("EVAL_ROOT")
CANON = ["age", "race", "height", "weight", "sbp", "dbp", "cursmoke", "htnmed",
         "prevdm", "prevhf", "prevmi", "sex", "bmi", "egfr"]
CHARGE = ["age", "white", "height", "weight", "sbp", "dbp", "cursmoke", "htnmed", "prevdm", "prevhf", "prevmi"]
PREVENT = ["age", "sbp", "bmi", "egfr", "prevdm", "cursmoke", "htnmed", "sex"]
VARS = {"af5": CHARGE, "hf5": PREVENT}


def load_vars(cohort):
    """raw charge_prevent_{cohort}.csv -> canonical columns + white indicator, keyed by str id."""
    m = MAPS[cohort]
    raw = pd.read_csv(f"{CP}/charge_prevent_{cohort}.csv")
    d = pd.DataFrame({"id": raw[m["id"]].astype(str)})
    for c in CANON:
        d[c] = pd.to_numeric(raw[m[c]], errors="coerce") if m[c] in raw.columns else np.nan
    d["white"] = (d["race"] == 1).astype(float)      # CHARGE-AF race term: white=1 (matches risk_score.py)
    return d


def load_preds(path):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)[["id", "y_true", "y_pred"]].copy()
    d["id"] = d["id"].astype(str)
    return d.dropna(subset=["y_true"])


def align_ids(vars_df, pred_ids):
    """UKB preds may be keyed by eid while vars are eid_visit (or vice-versa). Pick the join that
    covers the most prediction rows; strip a trailing _visit suffix if that helps."""
    pid = set(pred_ids)
    cov0 = np.mean([i in pid for i in vars_df["id"]]) if len(vars_df) else 0
    stripped = vars_df["id"].str.split("_").str[0]
    covs = np.mean([i in pid for i in stripped]) if len(vars_df) else 0
    if covs > cov0:
        vars_df = vars_df.copy(); vars_df["id"] = stripped
    return vars_df


def boot_deltas(y, sa, sb, B, rng):
    """paired bootstrap of BOTH ΔAUROC and ΔAUPRC for sa - sb (shared resamples), stratified by
    event. Returns {'auroc': {delta, ci, p}, 'auprc': {delta, ci, p}}."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    o_roc = roc_auc_score(y, sa) - roc_auc_score(y, sb)
    o_pr = average_precision_score(y, sa) - average_precision_score(y, sb)
    if B <= 0:
        nan = {"delta": float("nan"), "ci": [float("nan"), float("nan")], "p": float("nan")}
        return {"auroc": {**nan, "delta": float(o_roc)}, "auprc": {**nan, "delta": float(o_pr)}}
    dr, dp = np.empty(B), np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        yy = y[idx]
        dr[b] = roc_auc_score(yy, sa[idx]) - roc_auc_score(yy, sb[idx])
        dp[b] = average_precision_score(yy, sa[idx]) - average_precision_score(yy, sb[idx])

    def summ(o, d):
        p = 2 * min((d >= 0).mean(), (d <= 0).mean())
        return {"delta": float(o), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "p": float(min(p, 1.0))}
    return {"auroc": summ(o_roc, dr), "auprc": summ(o_pr, dp)}


def boot_metric(y, s, B, rng):
    """Stratified bootstrap 95% CI for a SINGLE arm's AUROC and AUPRC (point + ci)."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    o_roc = float(roc_auc_score(y, s)); o_pr = float(average_precision_score(y, s))
    if B <= 0:
        nan = [float("nan"), float("nan")]
        return {"auroc": o_roc, "auprc": o_pr, "auroc_ci": nan, "auprc_ci": nan}
    rr, pp = np.empty(B), np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        yy = y[idx]
        rr[b] = roc_auc_score(yy, s[idx]); pp[b] = average_precision_score(yy, s[idx])
    return {"auroc": o_roc, "auprc": o_pr,
            "auroc_ci": [float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))],
            "auprc_ci": [float(np.percentile(pp, 2.5)), float(np.percentile(pp, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ours_root", default=f"{EV}_3", help="root for OUR preds (seed 3 = eval_3)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    mrng = np.random.default_rng(a.seed + 12345)   # independent stream for per-arm CIs (keeps deltas byte-identical)
    OR = a.ours_root
    out_path = a.out or f"{OR}/figures/refit_clinical_compare.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    varcache = {c: load_vars(c) for c in ["UKBB", "CHS", "MESA"]}
    results = []
    for outc in ["af5", "hf5"]:
        cols = VARS[outc]
        # ---- UKB train: labels + ECG score to FIT on (OUR model = ours_root, seed 3) ----
        tr = load_preds(f"{OR}/ukb_train_preds/m75_ecgfull/{outc}/result.csv")
        vU = align_ids(varcache["UKBB"], set(tr["id"]))
        trj = tr.merge(vU, on="id", how="inner")
        print(f"[{outc}] UKB-train fit n={len(trj)} (of {len(tr)} preds), events={int(trj.y_true.sum())}", flush=True)

        imp = SimpleImputer(strategy="median").fit(trj[cols])
        sc = StandardScaler().fit(imp.transform(trj[cols]))
        def X(df):  # clinical design matrix
            return sc.transform(imp.transform(df[cols]))
        ecg_mu, ecg_sd = trj["y_pred"].mean(), trj["y_pred"].std() or 1.0
        def Xe(df):  # clinical + standardized ECG score
            return np.column_stack([X(df), (df["y_pred"].values - ecg_mu) / ecg_sd])

        ytr = trj["y_true"].values.astype(int)
        clin = LogisticRegression(max_iter=1000).fit(X(trj), ytr)
        full = LogisticRegression(max_iter=1000).fit(Xe(trj), ytr)

        # ---- evaluate: UKB test, then zero-shot CHS / MESA ----
        evalsets = [("UKB", load_preds(f"{OR}/ukb_test/m75_ecgfull/ecg/{outc}/result.csv"), "UKBB"),
                    ("CHS", load_preds(f"{OR}/zeroshot/CHS/m75_ecgfull/{outc}/result.csv"), "CHS"),
                    ("MESA", load_preds(f"{OR}/zeroshot/MESA/m75_ecgfull/{outc}/result.csv"), "MESA")]
        for coh, pred, vkey in evalsets:
            if pred is None:
                print(f"  [skip] {coh} {outc}: no preds"); continue
            v = align_ids(varcache[vkey], set(pred["id"]))
            j = pred.merge(v, on="id", how="inner").dropna(subset=["y_pred"])
            y = j["y_true"].values.astype(int)
            if y.sum() < 5 or (y == 0).sum() < 5:
                print(f"  [skip] {coh} {outc}: too few events"); continue
            s_clin = clin.predict_proba(X(j))[:, 1]
            s_ecg = j["y_pred"].values
            s_full = full.predict_proba(Xe(j))[:, 1]
            met = {k: boot_metric(y, s, a.B, mrng)   # per-arm point + 95% CI (adds *_ci; point fields unchanged)
                   for k, s in [("clinical", s_clin), ("ecg", s_ecg), ("full", s_full)]}
            df_c = boot_deltas(y, s_full, s_clin, a.B, rng)   # full - clinical (does ECG add?)
            de_c = boot_deltas(y, s_ecg, s_clin, a.B, rng)    # ecg  - clinical
            row = {"cohort": coh, "outcome": outc, "n": int(len(y)), "events": int(y.sum()),
                   "metrics": met, "d_full_minus_clinical": df_c, "d_ecg_minus_clinical": de_c}
            results.append(row)
            print(f"  {coh} {outc}: [AUROC clin={met['clinical']['auroc']:.3f} full={met['full']['auroc']:.3f} "
                  f"Δ={df_c['auroc']['delta']:+.3f} p={df_c['auroc']['p']:.3f}]  "
                  f"[AUPRC clin={met['clinical']['auprc']:.3f} full={met['full']['auprc']:.3f} "
                  f"Δ={df_c['auprc']['delta']:+.3f} p={df_c['auprc']['p']:.3f}]", flush=True)

    json.dump({"_meta": {"B": a.B, "seed": a.seed, "ours_root": OR,
                         "note": "refit clinical LR fit on UKB train; CHS/MESA zero-shot"},
               "results": results}, open(out_path, "w"), indent=2)
    print(f"\n[wrote -> {out_path}]", flush=True)


if __name__ == "__main__":
    main()
