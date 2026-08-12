"""
Supplementary Figs 3-5 -- subgroup analysis (AUROC by age / sex).

Reads a3_subgroups.py's `subgroups.csv` and draws one dot-and-CI figure per cohort: subgroups on
the y-axis, AUROC on the x-axis, AF | HF facets, one colour per model arm. No recompute.
  UKB    (Supp 3): CARDIAC-FM (ECG + RS)  vs  (ECG + MRI + RS)
  CHS    (Supp 4): CARDIAC-FM (ECG)  and its +Risk-Score variant   (zero-shot; no few-shot arm)
  MESA   (Supp 5): same as CHS

  python fig_subgroups.py --csv <a3/subgroups.csv> --outdir <figures> [--tag our3]
"""
import argparse, os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ARMS = {"UKBB": ["CL(ECG)+RS", "CL(ECG+MRI)+RS"],
        "CHS":  ["CL(ECG)", "CL(ECG)+RS"],
        "MESA": ["CL(ECG)", "CL(ECG)+RS"]}
ARM_COLOR = {"CL(ECG)": "#009E73", "CL(ECG)+RS": "#D55E00", "CL(ECG+MRI)+RS": "#0072B2"}
ARM_LABEL = {"CL(ECG)": "CARDIAC-FM (ECG)", "CL(ECG)+RS": "CARDIAC-FM (ECG + RS)",
             "CL(ECG+MRI)+RS": "CARDIAC-FM (ECG + MRI + RS)"}
OUTC_TITLE = {"af5": "Atrial Fibrillation", "hf5": "Heart Failure"}


def order_subs(subs):
    o = [s for s in ["ALL"] if s in subs]
    o += sorted(s for s in subs if s.startswith("age<"))
    o += sorted(s for s in subs if s.startswith("age>="))
    o += [s for s in ["female", "male"] if s in subs]
    return o


def sub_label(s):
    return {"ALL": "Overall", "female": "Female", "male": "Male"}.get(
        s, s.replace("age<", "Age <").replace("age>=", "Age ≥"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_csv(a.csv)
    suffix = f"_{a.tag}" if a.tag else ""
    for coh in ["UKBB", "CHS", "MESA"]:
        sub = df[df.cohort == coh]
        if sub.empty:
            continue
        arms = [x for x in ARMS[coh] if x in sub.arm.unique()]
        outcs = [o for o in ["af5", "hf5"] if o in sub.outcome.unique()]
        fig, axes = plt.subplots(1, len(outcs), figsize=(6 * len(outcs), 4.6), squeeze=False)
        axes = axes.reshape(-1)
        for ax, outc in zip(axes, outcs):
            so = sub[sub.outcome == outc]
            subs = order_subs(list(so.subgroup.unique()))
            ypos = {s: i for i, s in enumerate(reversed(subs))}     # first in list -> top
            for ai, arm in enumerate(arms):
                for _, row in so[so.arm == arm].iterrows():
                    if row.subgroup not in ypos or not np.isfinite(row.auroc):
                        continue
                    y = ypos[row.subgroup] + (ai - (len(arms) - 1) / 2) * 0.18
                    ax.plot([row.lo, row.hi], [y, y], color=ARM_COLOR[arm], lw=2, zorder=2)
                    ax.plot(row.auroc, y, "o", color=ARM_COLOR[arm], ms=6, zorder=3)
            ax.set_yticks(range(len(subs)))
            ax.set_yticklabels([sub_label(s) for s in reversed(subs)], fontsize=9)
            ax.set_ylim(-0.5, len(subs) - 0.5)
            ax.set_xlabel("AUROC"); ax.set_title(OUTC_TITLE.get(outc, outc), fontsize=10)
            ax.axvline(0.5, color="#999999", ls=":", lw=0.8, zorder=1)
            ax.grid(axis="x", alpha=0.25, zorder=0); ax.set_xlim(0.5, 0.9)
        handles = [Line2D([0], [0], color=ARM_COLOR[x], marker="o", lw=2, label=ARM_LABEL[x]) for x in arms]
        fig.legend(handles=handles, loc="upper center", ncol=len(arms), fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, 1.03))
        fig.suptitle(f"Subgroup AUROC by age/sex -- {coh}" + (f"  ({a.tag})" if a.tag else ""),
                     y=1.06, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        for ext in ["png", "pdf"]:
            fig.savefig(f"{a.outdir}/supp_subgroups_{coh}{suffix}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[{coh}] wrote supp_subgroups_{coh}{suffix}.png/pdf", flush=True)


if __name__ == "__main__":
    main()
