"""
A3 — subgroup analysis by AGE and SEX (paper: Suppl Figs 3-5, Tables 10/11/14).

AUROC with stratified bootstrap 95% CI within each subgroup, for every model arm, in all three
cohorts. Answers Reviewer #1's point 5 as well (inconsistent age thresholds):

  AGE CUTOFF is NOT arbitrary. CHS enrolled adults aged >=65 by design, so a <65 stratum is EMPTY
  (frac<65 = 0.000). We therefore use <65/>=65 for UKB and MESA (46% below the cut in both) and
  <75/>=75 for CHS (75 is the CHS median -> 56/44 split). This should be stated in the paper.

Usage: python a3_subgroups.py [--B 1000]
"""
import argparse, os, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata

# --- repo path configuration (see docs/PATHS.md) ---
import os as _os, sys as _sys
_pd = _os.path.dirname(_os.path.abspath(__file__))
while _pd != "/" and not _os.path.isdir(_os.path.join(_pd, "common")):
    _pd = _os.path.dirname(_pd)
_sys.path.insert(0, _os.path.join(_pd, "common"))
from paths import P

warnings.filterwarnings("ignore")

EV = P("EVAL_ROOT")
OR = EV  # our-model root; --ours_root overrides (e.g. .../eval_3 for seed 3). Baselines stay under EV.
CP = P("RISK_ROOT", "CHARGE-PREVENT")
RS = P("RISK_ROOT", "computed")
DEMO = {"UKBB": ("eid_visit", "agecmr", "sex"), "CHS": ("seqid", "age", "gend01"),
        "MESA": ("idno_visit", "agec", "gender")}
AGE_CUT = {"UKBB": 65, "CHS": 75, "MESA": 65}       # see docstring
RISK_COL = {"af5": "charge_mean", "hf5": "prevent_mean"}


def fauc(y, s):
    y = np.asarray(y); npos = y.sum(); nneg = len(y) - npos
    if npos < 5 or nneg < 5:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def ci(y, s, B, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) < 5 or len(neg) < 5:
        return np.nan, np.nan, np.nan
    st = np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        st[b] = fauc(y[idx], s[idx])
    return fauc(y, s), *np.percentile(st, [2.5, 97.5])


def demo(coh):
    idc, ac, sc = DEMO[coh]
    d = pd.read_csv(f"{CP}/charge_prevent_{coh}.csv")
    out = pd.DataFrame({"id": d[idc].astype(str),
                        "age": pd.to_numeric(d[ac], errors="coerce"),
                        "sex": pd.to_numeric(d[sc], errors="coerce")})
    return out.dropna(subset=["age"])


def load(f, name):
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)[["id", "y_true", "y_pred"]].dropna()
    d["id"] = d["id"].astype(str)
    return d.rename(columns={"y_pred": name})


def arms_for(coh, outc):
    """returns {arm_name: path}"""
    if coh == "UKBB":
        a = {"ECG-FM": f"{EV}/ukb_test/ecgfm/{outc}/result.csv",
             "CL(ECG)": f"{OR}/ukb_test/m75_ecgfull/ecg/{outc}/result.csv",
             "CL(ECG+MRI)": f"{OR}/ukb_test/m75_ecgfull/ecg_mri/{outc}/result.csv",
             "CL(ECG)+RS": f"{OR}/ukb_test/m75_ecgfull_RS/ecg/{outc}/result.csv",
             "CL(ECG+MRI)+RS": f"{OR}/ukb_test/m75_ecgfull_RS/ecg_mri/{outc}/result.csv"}
    else:
        a = {"CL(ECG)": f"{OR}/zeroshot/{coh}/m75_ecgfull/{outc}/result.csv",
             "CL(ECG)+RS": f"{OR}/zeroshot/{coh}/m75_ecgfull_RS/{outc}/result.csv"}
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=f"{EV}/a3/subgroups.md")
    ap.add_argument("--ours_root", default=EV, help="root for OUR model preds, e.g. .../eval_3 for seed 3")
    a = ap.parse_args()
    global OR; OR = a.ours_root
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    L = ["# A3 — subgroup analysis by age and sex", "",
         "AUROC [stratified bootstrap 95% CI], within subgroup.", "",
         "> **Age cutoff is not arbitrary (answers Reviewer #1 point 5).** CHS enrolled adults aged "
         "**≥65 by design**, so a <65 stratum is *empty* (frac<65 = 0.000). We use **<65/≥65 for UKB "
         "and MESA** (46% below the cut in both) and **<75/≥75 for CHS** (75 = CHS median, 56/44 "
         "split). This should be stated explicitly in the paper.", ""]
    rows = []
    for coh in ["UKBB", "CHS", "MESA"]:
        dm = demo(coh); cut = AGE_CUT[coh]
        for outc in ["af5", "hf5"]:
            arms = arms_for(coh, outc)
            # risk score as an arm too
            rsf = f"{RS}/{coh}_riskscore.csv"
            base = None
            merged = {}
            for nm, f in arms.items():
                d = load(f, nm)
                if d is None:
                    continue
                merged[nm] = d
                base = d[["id", "y_true"]] if base is None else base
            if base is None:
                continue
            j = base.merge(dm, on="id", how="inner")
            for nm, d in merged.items():
                j = j.merge(d[["id", nm]], on="id", how="left")
            if os.path.exists(rsf):
                r = pd.read_csv(rsf)[["id", RISK_COL[outc]]].rename(columns={RISK_COL[outc]: "RiskScore"})
                r["id"] = r["id"].astype(str)
                j = j.merge(r, on="id", how="left")
            arm_names = [c for c in (["RiskScore"] + list(merged)) if c in j.columns]

            groups = {"ALL": j,
                      f"age<{cut}": j[j.age < cut], f"age>={cut}": j[j.age >= cut],
                      "female": j[j.sex == 0], "male": j[j.sex == 1]}
            L.append(f"\n## {coh} — {outc}\n")
            L.append("| subgroup | n | events | " + " | ".join(arm_names) + " |")
            L.append("|---" * (3 + len(arm_names)) + "|")
            for gname, g in groups.items():
                if len(g) < 50:
                    continue
                cells = []
                for nm in arm_names:
                    gg = g.dropna(subset=["y_true", nm])
                    y = gg["y_true"].values.astype(int)
                    pt, lo, hi = ci(y, gg[nm].values, a.B, rng)
                    cells.append("—" if not np.isfinite(pt) else f"{pt:.3f} [{lo:.3f}, {hi:.3f}]")
                    rows.append(dict(cohort=coh, outcome=outc, subgroup=gname, arm=nm,
                                     n=len(gg), events=int(y.sum()), auroc=pt, lo=lo, hi=hi))
                ev = int(g["y_true"].sum())
                L.append(f"| {gname} | {len(g)} | {ev} | " + " | ".join(cells) + " |")
            print(f"  {coh} {outc} done", flush=True)

    txt = "\n".join(L)
    open(a.out, "w").write(txt + "\n")
    pd.DataFrame(rows).to_csv(a.out.replace(".md", ".csv"), index=False)
    print(f"[written to {a.out}]")


if __name__ == "__main__":
    main()
