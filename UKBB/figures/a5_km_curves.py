"""
A5 — Kaplan-Meier curves by predicted-risk tertile (the paper's `KM.Rmd`).

Same tertile groups as the Cox analyses (A4 for UKB, Fig 4 for CHS): low / intermediate / high by
the 1/3 and 2/3 quantiles of the predicted risk. Plots cumulative incidence with 95% CI bands.
Purely descriptive - it visualises the separation that the tertile HRs quantify.
"""
import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
warnings.filterwarnings("ignore")

EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"
OR = EV  # our-model root; --ours_root overrides (e.g. .../eval_3 for seed 3). Risk Score/survival stay under EV.
HRD = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/csv_HR"
IDD = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/csv_train_valid_test_individual_id_disease"
RS = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/computed"
OUTD = f"{EV}/a5"
COLORS = {"low": "#2c7fb8", "intermediate": "#f0a30a", "high": "#d7191c"}


def tert(x):
    q1, q2 = np.nanquantile(x, [1/3, 2/3])
    g = np.full(len(x), None, dtype=object); ok = ~np.isnan(x)
    g[ok & (x <= q1)] = "low"; g[ok & (x > q1) & (x <= q2)] = "intermediate"; g[ok & (x > q2)] = "high"
    return g


def km_panel(ax, T, E, g, title, xlabel):
    for lv in ["low", "intermediate", "high"]:
        m = (g == lv)
        if m.sum() < 20:
            continue
        k = KaplanMeierFitter().fit(T[m], E[m], label=f"{lv} (n={int(m.sum())})")
        k.plot_cumulative_density(ax=ax, color=COLORS[lv], ci_show=True, lw=2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel); ax.set_ylabel("cumulative incidence")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=.3)


def main():
    import argparse
    global OR, OUTD
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_root", default=EV, help="root for OUR model preds, e.g. .../eval_3 for seed 3")
    ap.add_argument("--outd", default=OUTD, help="output dir for km_*.png (e.g. .../eval_3/a5)")
    a = ap.parse_args()
    OR = a.ours_root; OUTD = a.outd
    os.makedirs(OUTD, exist_ok=True)

    # ---------- UKB ----------
    surv = pd.read_csv(f"{HRD}/ecgmri.csv"); surv["id"] = surv["eid_visit"].astype(str)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for r, (outc, tv, ev, pv, oc, rc) in enumerate(
            [("AF", "ttoaf", "afcens", "prevaf", "af5", "charge_mean"),
             ("HF", "ttohf", "hfcens", "prevhf", "hf5", "prevent_mean")]):
        risk = pd.read_csv(f"{RS}/UKBB_riskscore.csv")[["id", rc]].rename(columns={rc: "s"})
        risk["id"] = risk["id"].astype(str)
        arms = {"Risk Score": risk}
        for nm, mode in [("CL(ECG)+RS", "ecg"), ("CL(ECG+MRI)+RS", "ecg_mri")]:
            f = f"{OR}/ukb_test/m75_ecgfull_RS/{mode}/{oc}/result.csv"
            if os.path.exists(f):
                d = pd.read_csv(f)[["id", "y_pred"]].rename(columns={"y_pred": "s"})
                d["id"] = d["id"].astype(str); arms[nm] = d
        common = None
        for d in arms.values():
            ids = set(d.dropna(subset=["s"])["id"]); common = ids if common is None else common & ids
        for c, (nm, d) in enumerate(arms.items()):
            j = surv[["id", tv, ev, pv]].merge(d[d["id"].isin(common)], on="id").dropna(subset=[tv, ev, "s"])
            j = j[(j[pv] == 0) & (j[tv] > 0)]
            km_panel(axes[r, c], j[tv].values / 365.25, j[ev].values.astype(int),
                     tert(j["s"].values), f"UKB {outc} — {nm}", "years")
    fig.suptitle("A5 — UKB: cumulative incidence by predicted-risk tertile (paired test)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUTD}/km_ukb.png", dpi=150); plt.close(fig)
    print("wrote km_ukb.png", flush=True)

    # ---------- CHS ----------
    chs = pd.read_csv(f"{IDD}/CHS_split1.csv"); chs["id"] = chs["id"].astype(str)
    rs = pd.read_csv(f"{RS}/CHS_riskscore.csv"); rs["id"] = rs["id"].astype(str)
    chs = chs.merge(rs[["id", "charge_mean", "prevent_mean"]], on="id", how="left")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for r, (outc, tv, ev, pv, oc, rc) in enumerate(
            [("AF", "ttoaf", "incaf", "prevaf", "af5", "charge_mean"),
             ("HF", "ttohf", "inchf", "prevhf", "hf5", "prevent_mean")]):
        arms = {"Risk Score": chs[["id", rc]].rename(columns={rc: "s"})}
        f = f"{OR}/zeroshot/CHS/m75_ecgfull_RS/{oc}/result.csv"
        if os.path.exists(f):
            d = pd.read_csv(f)[["id", "y_pred"]].rename(columns={"y_pred": "s"})
            d["id"] = d["id"].astype(str); arms["CL(ECG)+RS (zero-shot)"] = d
        common = None
        for d in arms.values():
            ids = set(d.dropna(subset=["s"])["id"]); common = ids if common is None else common & ids
        for c, (nm, d) in enumerate(arms.items()):
            j = chs[["id", tv, ev, pv]].merge(d[d["id"].isin(common)], on="id").dropna(subset=[tv, ev, "s"])
            j = j[(j[pv] == 0) & (j[tv] > 0)]
            km_panel(axes[r, c], j[tv].values / 365.25, j[ev].values.astype(int),
                     tert(j["s"].values), f"CHS {outc} — {nm}", "years")
    fig.suptitle("A5 — CHS: cumulative incidence by predicted-risk tertile (zero-shot)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUTD}/km_chs.png", dpi=150); plt.close(fig)
    print("wrote km_chs.png", flush=True)


if __name__ == "__main__":
    main()
