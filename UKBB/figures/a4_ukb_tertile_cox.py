"""
A4 — UKB tertile risk-stratification Cox (the paper's supplementary `Hazard Ratio.Rmd` -> UKBB_HR.csv).

Three arms, as in their Methods ("risk scores alone, CARDIAC-FM (ECG + risk scores), and
CARDIAC-FM (ECG + MRI + risk scores)"): predictions -> tertiles (low/intermediate/high by the
1/3 and 2/3 quantiles) -> univariable Cox with LOW as reference -> HR high/low and intermediate/low.

Population: paired UKB test (`csv_HR/ecgmri.csv`), prevalent cases excluded per outcome.
NOTE this is NOT Fig 4 (which is CHS/MESA); it is the UKB supplementary version, and unlike Fig 4
it includes the ECG+MRI arm, which only exists in UKB.
"""
import argparse, os, warnings
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
warnings.filterwarnings("ignore")

EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"
OR = EV  # our-model root; --ours_root overrides (e.g. .../eval_3 for seed 3). Risk Score/survival stay under EV.
HRD = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/csv_HR"
RS = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/computed"
RISK_COL = {"af": "charge_mean", "hf": "prevent_mean"}
OUT = {"af": ("ttoaf", "afcens", "prevaf", "af5"), "hf": ("ttohf", "hfcens", "prevhf", "hf5")}


def tert(x):
    q1, q2 = np.nanquantile(x, [1/3, 2/3])
    g = np.full(len(x), None, dtype=object); ok = ~np.isnan(x)
    g[ok & (x <= q1)] = "low"; g[ok & (x > q1) & (x <= q2)] = "intermediate"; g[ok & (x > q2)] = "high"
    return g


def cox(T, E, g):
    d = pd.DataFrame({"T": np.asarray(T, float), "E": np.asarray(E, int),
                      "intermediate": (g == "intermediate").astype(int),
                      "high": (g == "high").astype(int)})
    d = d[(d["T"] > 0)]
    if d["E"].sum() < 10:
        return None
    s = CoxPHFitter().fit(d, "T", "E").summary
    return {lv: (s.loc[lv, "exp(coef)"], s.loc[lv, "exp(coef) lower 95%"],
                 s.loc[lv, "exp(coef) upper 95%"], s.loc[lv, "p"]) for lv in ["intermediate", "high"]}, \
           int(d["E"].sum()), len(d)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{EV}/a4/ukb_tertile_cox.md")
    ap.add_argument("--ours_root", default=EV, help="root for OUR model preds, e.g. .../eval_3 for seed 3")
    a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    global OR; OR = a.ours_root
    surv = pd.read_csv(f"{HRD}/ecgmri.csv"); surv["id"] = surv["eid_visit"].astype(str)
    risk = pd.read_csv(f"{RS}/UKBB_riskscore.csv"); risk["id"] = risk["id"].astype(str)

    L = ["# A4 — UKB tertile risk stratification (supplementary Cox; NOT Fig 4)", "",
         "Tertiles of the predicted risk (low / intermediate / high by the 1/3, 2/3 quantiles);",
         "univariable Cox with **low as reference**. Paired UKB test; prevalent cases excluded.",
         "Includes the **ECG+MRI** arm, which exists only in UKB.", ""]
    rows = []
    for outc, (tv, ev, pv, oc) in OUT.items():
        arms = {"RiskScore": risk[["id", RISK_COL[outc]]].rename(columns={RISK_COL[outc]: "s"}),
                "CL(ECG)+RS": None, "CL(ECG+MRI)+RS": None}
        for nm, mode in [("CL(ECG)+RS", "ecg"), ("CL(ECG+MRI)+RS", "ecg_mri")]:
            f = f"{OR}/ukb_test/m75_ecgfull_RS/{mode}/{oc}/result.csv"
            if os.path.exists(f):
                d = pd.read_csv(f)[["id", "y_pred"]].rename(columns={"y_pred": "s"})
                d["id"] = d["id"].astype(str); arms[nm] = d
        # ---- restrict every arm to the COMMON population, else the HRs are not comparable:
        # the risk score exists for the whole paired test, our CL predictions only for the
        # labelled subset, so an unrestricted comparison would contrast different populations.
        common = None
        for nm, d in arms.items():
            if d is None:
                continue
            ids = set(d.dropna(subset=["s"])["id"])
            common = ids if common is None else (common & ids)
        L.append(f"\n## {outc.upper()}\n")
        L.append(f"*All arms restricted to the common population (n={len(common)} with predictions "
                 f"from every arm).*\n")
        L.append("| arm | n | events | HR intermediate/low [95% CI] | HR high/low [95% CI] | p (high) |")
        L.append("|---|---|---|---|---|---|")
        for nm, d in arms.items():
            if d is None:
                continue
            d = d[d["id"].isin(common)]
            j = surv[["id", tv, ev, pv]].merge(d, on="id").dropna(subset=[tv, ev, "s"])
            j = j[j[pv] == 0]
            r = cox(j[tv].values, j[ev].values, tert(j["s"].values))
            if r is None:
                L.append(f"| {nm} | — | — | (too few events) | | |"); continue
            res, e_, n_ = r
            i_, h_ = res["intermediate"], res["high"]
            L.append(f"| {nm} | {n_} | {e_} | {i_[0]:.2f} [{i_[1]:.2f}, {i_[2]:.2f}] "
                     f"| {h_[0]:.2f} [{h_[1]:.2f}, {h_[2]:.2f}] | {h_[3]:.3g} |")
            rows.append(dict(outcome=outc, arm=nm, n=n_, events=e_,
                             hr_int=i_[0], hr_int_lo=i_[1], hr_int_hi=i_[2],
                             hr_high=h_[0], hr_high_lo=h_[1], hr_high_hi=h_[2], p_high=h_[3]))
    txt = "\n".join(L); open(a.out, "w").write(txt + "\n")
    pd.DataFrame(rows).to_csv(a.out.replace(".md", ".csv"), index=False)
    print(txt); print(f"\n[written to {a.out}]")


if __name__ == "__main__":
    main()
