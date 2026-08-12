"""
CMR-feature regression: TEST-set evaluation, reproducing the paper's UKBB_R2_Corr.Rmd exactly.

  corr = pearson(y_true, y_pred)
  R2   = 1 - MSE / Var(y_true)      <- variance-explained, NOT squared correlation; can be negative
  restricted to `healthy == 1`      <- verified to mean NOT(prevalent AF or prevalent HF); 96.6%

Reports OUR model and (on the identical individuals) the PAPER's own predictions from
ECGstage1_UKBB_continuous_split{i}.csv, so the comparison is like-for-like rather than against
their published numbers on a different population.

Validation-set r is a model-selection diagnostic and is deliberately NOT reported here.

Usage: python cmr_eval.py
"""
import os, argparse
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

EV = P("EVAL_ROOT")
IDD = P("RISK_ROOT", "csv_train_valid_test_individual_id_disease")
TRUTH = f"{EV}/UKBB_cmr_true_test.csv"
FEATS = ["lvm", "lvedv", "lvesv", "lavmin", "lavmax", "laef", "lvef"]
PAPER_R = {"lvm": 0.79, "lvedv": 0.72, "lvesv": 0.71, "lavmin": 0.67,
           "lavmax": 0.57, "laef": 0.55, "lvef": 0.51}


def metrics(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    r = np.corrcoef(yt, yp)[0, 1]
    r2 = 1 - np.mean((yt - yp) ** 2) / np.var(yt)
    return r, r2


def make_fig6(rows, figdir, seedtag):
    """Supp Fig 6: ECG-predicted CMR-feature accuracy -- Pearson r (left) and R^2 (right) bars."""
    os.makedirs(figdir, exist_ok=True)
    order = ["lavmin", "lavmax", "laef", "lvef", "lvedv", "lvesv", "lvm"]     # paper's x order
    rr = sorted((r for r in rows if r["feature"] in order), key=lambda r: order.index(r["feature"]))
    feats = [r["feature"].upper() for r in rr]
    fig, (axr, axr2) = plt.subplots(1, 2, figsize=(12, 4.4))
    axr.bar(feats, [r["ours_r"] for r in rr], color="#2c5f8a", edgecolor="white")
    axr.set_ylabel("Pearson correlation"); axr.set_ylim(0, 1); axr.set_title("Pearson r")
    axr2.bar(feats, [r["ours_R2"] for r in rr], color="#2c5f8a", edgecolor="white")
    axr2.set_ylabel("R²"); axr2.set_ylim(0, max(0.7, max((r["ours_R2"] for r in rr), default=0.7) * 1.1))
    axr2.set_title("R² (variance explained)")
    for ax in (axr, axr2):
        ax.tick_params(axis="x", rotation=45); ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Supp Fig 6 — ECG-predicted CMR features, UKB test ({seedtag})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["png", "pdf"]:
        fig.savefig(f"{figdir}/supp6_cmr_accuracy_{seedtag}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[supp6 -> {figdir}/supp6_cmr_accuracy_{seedtag}.png/pdf]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1, help="which CMR-regression seed to evaluate")
    ap.add_argument("--figdir", default=None, help="if set, also write the Supp-6 accuracy figure")
    a = ap.parse_args()
    truth = pd.read_csv(TRUTH); truth["eid_visit"] = truth["eid_visit"].astype(str)
    paper = pd.read_csv(f"{IDD}/ECGstage1_UKBB_continuous_split1.csv")
    paper["eid_visit"] = paper["eid_visit"].astype(str)

    print(f"{'feature':<9}{'n':>7}{'ours r':>9}{'ours R2':>9}"
          f"{'paper-split1 r':>16}{'R2':>8}   {'their published r':>18}")
    print("-" * 78)
    rows = []
    for f in FEATS:
        pf = f"{EV}/cmr_reg/m75_seed{a.seed}/{f}/test/result.csv"
        if not os.path.exists(pf):
            print(f"{f:<9}{'—':>7}  (test result.csv missing — inference not finished)"); continue
        ours = pd.read_csv(pf)[["id", "y_pred"]].rename(columns={"id": "eid_visit", "y_pred": "ours"})
        ours["eid_visit"] = ours["eid_visit"].astype(str)
        j = (truth[["eid_visit", f, "healthy"]]
             .merge(ours, on="eid_visit", how="inner")
             .merge(paper[["eid_visit", f"{f}_pred"]], on="eid_visit", how="left"))
        j = j[(j["healthy"] == 1)].dropna(subset=[f, "ours"])
        r_o, r2_o = metrics(j[f], j["ours"])
        sub = j.dropna(subset=[f"{f}_pred"])
        r_p, r2_p = metrics(sub[f], sub[f"{f}_pred"]) if len(sub) > 10 else (np.nan, np.nan)
        print(f"{f:<9}{len(j):>7}{r_o:>9.3f}{r2_o:>9.3f}{r_p:>16.3f}{r2_p:>8.3f}   {PAPER_R[f]:>18.2f}")
        rows.append(dict(feature=f, n=len(j), ours_r=r_o, ours_R2=r2_o,
                         paper_split1_r=r_p, paper_split1_R2=r2_p, paper_published_r=PAPER_R[f]))
    if rows:
        out = f"{EV}/cmr_reg/cmr_test_results_seed{a.seed}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n[written to {out}]")
        print("\nNOTE: 'paper-split1' is THEIR prediction evaluated by US on the same individuals "
              "(single split);\n      'their published r' is their 4-seed mean on their own population "
              "— not directly comparable.")
        if a.figdir:
            make_fig6(rows, a.figdir, f"seed{a.seed}")


if __name__ == "__main__":
    main()
