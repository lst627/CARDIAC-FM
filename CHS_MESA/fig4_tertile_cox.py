"""
Fig 4 — risk stratification by tertiles of a predicted risk score (CHS / MESA).

Reproduces the paper's `Hazard Ratio {CHS,MESA}.Rmd`:
  1. per-person score  ->  2. tertiles via quantile(1/3, 2/3): low / intermediate / high
  3. UNIVARIABLE Cox `Surv(tto, inc==1) ~ tertile_group` with **low as reference**
     (a single 3-level factor, so one model yields BOTH high-vs-low and intermediate-vs-low)
  4. HR = exp(coef), CI = exp(coef +/- 1.96 SE)
Prevalent cases are excluded (`prev<outcome> == 0`), matching their `fit_one()`.

TIME-SCALE NOTE (verified 2026-07-19): CHS `tto*` is in DAYS and runs to ~9,510 (~26 yr) --
their code applies NO truncation, so their Fig 4 CHS HRs are over FULL follow-up even though the
caption says "5-year incident". We therefore report BOTH: `full` (their way, comparable to the
paper) and `5yr` (administratively censored at 1826.25 d) so the discrepancy is visible.
MESA has explicit `tto*_5yr`/`inc*_5yr` columns (exactly 1826.2 d) -- used directly.

Usage: python fig4_tertile_cox.py [--arm risk]
"""
import argparse, os
import numpy as np, pandas as pd
from lifelines import CoxPHFitter

# --- repo path configuration (see docs/PATHS.md) ---
import os as _os, sys as _sys
_pd = _os.path.dirname(_os.path.abspath(__file__))
while _pd != "/" and not _os.path.isdir(_os.path.join(_pd, "common")):
    _pd = _os.path.dirname(_pd)
_sys.path.insert(0, _os.path.join(_pd, "common"))
from paths import P


RS = P("RISK_ROOT", "computed")
IDD = P("RISK_ROOT", "csv_train_valid_test_individual_id_disease")
MESA = P("MESA_ECG_ROOT", "test.csv")
DAYS_5YR = 1826.25
RISK_COL = {"af": "charge_mean", "hf": "prevent_mean",
            "hfpef": "prevent_mean", "hfref": "prevent_mean"}


def tertiles(x):
    """low / intermediate / high by the 1/3 and 2/3 quantiles (matches R case_when order)."""
    q1, q2 = np.nanquantile(x, [1 / 3, 2 / 3])
    g = np.full(len(x), None, dtype=object)
    ok = ~np.isnan(x)
    g[ok & (x <= q1)] = "low"
    g[ok & (x > q1) & (x <= q2)] = "intermediate"
    g[ok & (x > q2)] = "high"
    return g


def cox_tertile(time, event, group):
    """univariable Cox on the 3-level tertile factor, low = reference."""
    d = pd.DataFrame({
        "T": np.asarray(time, float), "E": np.asarray(event, int),
        "intermediate": (group == "intermediate").astype(int),
        "high": (group == "high").astype(int),
    })
    d = d[(d["T"] > 0) & d["T"].notna()]
    if d["E"].sum() < 5 or d["high"].sum() == 0:
        return None
    cph = CoxPHFitter().fit(d, duration_col="T", event_col="E")
    s = cph.summary
    return {lv: (s.loc[lv, "exp(coef)"], s.loc[lv, "exp(coef) lower 95%"],
                 s.loc[lv, "exp(coef) upper 95%"], s.loc[lv, "p"])
            for lv in ["intermediate", "high"]}, int(d["E"].sum()), len(d)


def report(rows, title):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(f"{'cohort':<6}{'outcome':<8}{'horizon':<9}{'n':>7}{'events':>8}   "
          f"{'HR int/low [95% CI]':<26}{'HR high/low [95% CI]':<26}")
    for r in rows:
        if r["res"] is None:
            print(f"{r['coh']:<6}{r['outc']:<8}{r['hz']:<9}{'-':>7}{'-':>8}   (too few events)"); continue
        res, ev, n = r["res"]
        i, h = res["intermediate"], res["high"]
        print(f"{r['coh']:<6}{r['outc']:<8}{r['hz']:<9}{n:>7}{ev:>8}   "
              f"{i[0]:.2f} [{i[1]:.2f}, {i[2]:.2f}]{'':<8}{h[0]:.2f} [{h[1]:.2f}, {h[2]:.2f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="risk", choices=["risk"])
    args = ap.parse_args()
    rows = []

    # ---------------- CHS ----------------
    chs = pd.read_csv(f"{IDD}/CHS_split1.csv"); chs["id"] = chs["id"].astype(str)
    rs = pd.read_csv(f"{RS}/CHS_riskscore.csv"); rs["id"] = rs["id"].astype(str)
    chs = chs.merge(rs[["id", "charge_mean", "prevent_mean"]], on="id", how="left")
    for outc in ["af", "hf", "hfpef", "hfref"]:
        t, e, p = f"tto{outc}", f"inc{outc}", f"prev{outc}"
        col = RISK_COL[outc]
        d = chs[(chs[p] == 0) & chs[t].notna() & chs[e].notna() & chs[col].notna()].copy()
        g = tertiles(d[col].values)
        for hz in ["full", "5yr"]:
            T, E = d[t].values.astype(float), d[e].values.astype(int)
            if hz == "5yr":
                E = np.where(T > DAYS_5YR, 0, E); T = np.minimum(T, DAYS_5YR)
            rows.append({"coh": "CHS", "outc": outc, "hz": hz, "res": cox_tertile(T, E, g)})

    # ---------------- MESA ----------------
    # MESA survival is keyed by `idno` (participant) with PER-EXAM columns; our scores are keyed by
    # `idno_visit` where the suffix IS the exam number (1 or 5). So visit v maps to tto<outc><v> /
    # inc<outc><v> / prev<outc><v> -- each exam is its own time origin (no duplicated follow-up).
    D = os.path.dirname(MESA)
    surv = pd.concat([pd.read_csv(f"{D}/{f}") for f in ["test.csv", "train1.csv", "valid1.csv"]],
                     ignore_index=True).drop_duplicates(subset="idno")
    surv["idno"] = surv["idno"].astype(str)
    rs = pd.read_csv(f"{RS}/MESA_riskscore.csv"); rs["id"] = rs["id"].astype(str)
    rs["idno"] = rs["id"].str.split("_").str[0]
    rs["visit"] = rs["id"].str.split("_").str[1]
    j = rs.merge(surv, on="idno", how="inner")
    for outc in ["af", "hf"]:
        col = RISK_COL[outc]
        parts = []
        for v in ["1", "5"]:
            t, e, p = f"tto{outc}{v}", f"inc{outc}{v}", f"prev{outc}{v}"
            if t not in j or e not in j:
                continue
            d = j[(j["visit"] == v) & j[t].notna() & j[e].notna() & j[col].notna()].copy()
            if p in d.columns:
                d = d[d[p] == 0]
            parts.append(pd.DataFrame({"T": d[t].astype(float).values,
                                       "E": d[e].astype(int).values, "S": d[col].values}))
        if not parts:
            rows.append({"coh": "MESA", "outc": outc, "hz": "5yr", "res": None}); continue
        a = pd.concat(parts, ignore_index=True)
        g = tertiles(a["S"].values)          # tertiles across the pooled exam-visits
        for hz in ["full", "5yr"]:
            T, E = a["T"].values.copy(), a["E"].values.copy()
            if hz == "5yr":
                E = np.where(T > DAYS_5YR, 0, E); T = np.minimum(T, DAYS_5YR)
            rows.append({"coh": "MESA", "outc": outc, "hz": hz, "res": cox_tertile(T, E, g)})

    report(rows, "Fig 4 — RISK SCORE ONLY (CHARGE-AF / PREVENT-HF), tertile Cox, low = reference")
    print("\nNOTE: CHS 'full' = their code's actual behaviour (no truncation, ~26 yr follow-up);")
    print("      CHS '5yr'  = administratively censored at 1826.25 d, matching the figure caption.")


if __name__ == "__main__":
    main()
