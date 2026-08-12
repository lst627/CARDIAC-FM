"""
Ensemble comparison (advisor request, 2026-07-20).

Instead of comparing our model against each ECG-FM seed separately and counting wins, build ONE
ECG-FM prediction by averaging the 4 seeds' predicted probabilities per subject, compute a single
AUROC from that, and run one paired bootstrap against our model.

  ensemble AUROC  =  AUROC( mean_s p_s(x) )        <-- what we now report
  mean AUROC      =  mean_s AUROC( p_s(x) )        <-- reported alongside, NOT the same number

Averaging predictions before scoring is the stronger baseline: seed-to-seed noise cancels, so the
ensemble scores above the mean of the individual AUROCs. BOTH sides are ensembled over 4 seeds
(ours are 0-3, ECG-FM's are 1-4) so the comparison is symmetric. Any arm for which our per-seed
files do not exist yet falls back to seed 1 and is labelled "[seed1 only]" in the output -- those
rows ARE asymmetric and must not be quoted as final.

Usage: python ensemble_compare.py [--B 2000] [--ours_root .../eval] [--out ...]
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
SEEDS = [1, 2, 3, 4]


def fauc(y, s):
    """rank-based AUROC (fast enough for thousands of bootstrap reps)."""
    npos = int(y.sum()); nneg = len(y) - npos
    if npos < 3 or nneg < 3:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def load(f):
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)[["id", "y_true", "y_pred"]].dropna()
    d["id"] = d["id"].astype(str)
    return d


def ecgfm_paths(setting, outc, rs):
    """the 4 seed prediction files for a given setting."""
    if setting == "UKB":
        return [f"{EV}/ukb_test/ecgfm_RS/seed{s}/{outc}/result.csv" if rs else
                f"{EV}/ecgfm_ukb_paired/seed{s}/test/{outc}/result.csv" for s in SEEDS]
    return [f"{EV}/zeroshot/{setting}/ecgfm_RS/seed{s}/{outc}/result.csv" if rs else
            f"{EV}/ecgfm_zeroshot_provided/seed{s}/zeroshot/{setting}/{outc}/result.csv"
            for s in SEEDS]


OUR_SEEDS = [0, 1, 2, 3]          # our downstream seeds are 0-3; ECG-FM's are 1-4


def ours_paths(setting, outc, rs, root):
    """{arm: [4 seed paths]} — ours is ensembled the same way ECG-FM is.

    Seed 1 lives in the canonical (unsuffixed) location because it was produced first and every
    earlier report is built on it; seeds 0/2/3 live under *_clseeds. UKB has ecg + ecg_mri;
    external cohorts are ECG-only (no MRI at inference in the zero-shot setting)."""
    tag = "m75_ecgfull_RS" if rs else "m75_ecgfull"

    def ukb(mode, s):
        if rs:      # +Risk fusion was regenerated for ALL four seeds (risk_fuse --regime fig2_clseeds)
            return f"{root}/ukb_test_clseeds_RS/seed{s}/m75_ecgfull/{mode}/{outc}/result.csv"
        if s == 1:
            return f"{root}/ukb_test/{tag}/{mode}/{outc}/result.csv"
        return f"{root}/ukb_test_clseeds/seed{s}/{tag}/{mode}/{outc}/result.csv"

    def ext(s):
        if rs:      # risk_fuse --regime zs_clseeds writes the UNsuffixed tag under a _RS root
            return f"{root}/zeroshot_clseeds_RS/seed{s}/{setting}/m75_ecgfull/{outc}/result.csv"
        return f"{root}/zeroshot_clseeds/seed{s}/{setting}/m75_ecgfull/{outc}/result.csv"

    if setting == "UKB":
        return {"CL(ECG)": [ukb("ecg", s) for s in OUR_SEEDS],
                "CL(ECG+MRI)": [ukb("ecg_mri", s) for s in OUR_SEEDS]}
    return {"CL(ECG)": [ext(s) for s in OUR_SEEDS]}


def single_seed1(setting, outc, rs, root):
    """fallback so the table still renders where our per-seed files are not yet generated."""
    tag = "m75_ecgfull_RS" if rs else "m75_ecgfull"
    if setting == "UKB":
        return {"CL(ECG)": f"{root}/ukb_test/{tag}/ecg/{outc}/result.csv",
                "CL(ECG+MRI)": f"{root}/ukb_test/{tag}/ecg_mri/{outc}/result.csv"}
    return {"CL(ECG)": f"{root}/zeroshot/{setting}/{tag}/{outc}/result.csv"}


def build_ensemble(paths):
    """average the seeds' predicted probabilities on the ids common to all 4."""
    ds = [load(p) for p in paths]
    if any(d is None for d in ds):
        return None, None
    m = ds[0][["id", "y_true"]].copy()
    for i, d in enumerate(ds):
        m = m.merge(d[["id", "y_pred"]].rename(columns={"y_pred": f"s{i}"}), on="id")
    cols = [f"s{i}" for i in range(len(ds))]
    m["y_pred"] = m[cols].mean(axis=1)
    return m[["id", "y_true", "y_pred"]], m[cols + ["y_true"]]


def paired_boot(y, a, b, B, rng):
    """stratified paired bootstrap: both models scored on the SAME resample."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    da = np.empty(B); db = np.empty(B)
    for i in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        yy = y[idx]
        da[i] = fauc(yy, a[idx]); db[i] = fauc(yy, b[idx])
    d = da - db
    p = 2 * min((d >= 0).mean(), (d <= 0).mean())
    return (np.nanpercentile(da, [2.5, 97.5]), np.nanpercentile(db, [2.5, 97.5]),
            float(np.nanmean(d)), np.nanpercentile(d, [2.5, 97.5]), min(p, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ours_root", default=EV)
    ap.add_argument("--label", default="m75, 4 seeds (0-3)")
    ap.add_argument("--out", default=f"{EV}/ensemble/ensemble_vs_ours.md")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    L = [f"# 4-seed ENSEMBLE vs ENSEMBLE — ours ({a.label}) vs ECG-FM", "",
         "Each arm's prediction = **mean of its 4 seeds' predicted probabilities per subject**, "
         "scored once. This is not the same as averaging the 4 AUROCs; both are shown so the "
         "difference is visible. Our seeds are 0-3, ECG-FM's are 1-4.", "",
         "> ⚠️ Rows marked **[seed1 only]** do not yet have our per-seed predictions and compare a "
         "single model against a 4-model ensemble. They are biased against us and are provisional.",
         "",
         f"Paired stratified bootstrap, B={a.B}; both arms scored on the same resample, so shared "
         "test-set noise cancels. p = 2·min(Pr(Δ≥0), Pr(Δ≤0)).", ""]
    rows = []

    for rs in [False, True]:
        tagname = "+Risk Score" if rs else "model only"
        L += [f"\n## {tagname}\n",
              "| setting | outcome | our arm | ours ENS [95% CI] | ours mean-of-seeds | "
              "ECG-FM ENS [95% CI] | ECG-FM mean-of-seeds | Δ (ours−ECG-FM) [95% CI] | p |",
              "|---|---|---|---|---|---|---|---|---|"]
        for setting in ["UKB", "CHS", "MESA"]:
            for outc in ["af5", "hf5"]:
                ens, percol = build_ensemble(ecgfm_paths(setting, outc, rs))
                if ens is None:
                    L.append(f"| {setting} | {outc} | — | *ECG-FM seed files missing* | | | | |")
                    continue
                # individual-seed AUROCs on the ensemble's id set, for the mean-of-AUROC column
                ys = percol["y_true"].values.astype(int)
                indiv = [fauc(ys, percol[f"s{i}"].values) for i in range(len(SEEDS))]
                fallback = single_seed1(setting, outc, rs, a.ours_root)
                for armname, plist in ours_paths(setting, outc, rs, a.ours_root).items():
                    o, ourcol = build_ensemble(plist)
                    nseed = len(OUR_SEEDS)
                    if o is None:                      # per-seed files not generated yet
                        o, ourcol, nseed = load(fallback[armname]), None, 1
                    if o is None:
                        continue
                    armlabel = armname if nseed > 1 else f"{armname} [seed1 only]"
                    our_indiv = ([fauc(ourcol["y_true"].values.astype(int), ourcol[f"s{i}"].values)
                                  for i in range(nseed)] if ourcol is not None else [np.nan])
                    j = o.rename(columns={"y_pred": "ours"}).merge(
                        ens[["id", "y_pred"]].rename(columns={"y_pred": "ens"}), on="id")
                    if len(j) < 50:
                        continue
                    y = j["y_true"].values.astype(int)
                    A, B_ = j["ours"].values, j["ens"].values
                    pa, pb = fauc(y, A), fauc(y, B_)
                    cia, cib, dm, cid, p = paired_boot(y, A, B_, a.B, rng)
                    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    om = np.nanmean(our_indiv)
                    omtxt = "—" if not np.isfinite(om) else f"{om:.4f}"
                    L.append(f"| {setting} | {outc} | {armlabel} | {pa:.4f} [{cia[0]:.4f}, {cia[1]:.4f}] "
                             f"| {omtxt} | {pb:.4f} [{cib[0]:.4f}, {cib[1]:.4f}] | {np.mean(indiv):.4f} "
                             f"| {pa-pb:+.4f} [{cid[0]:+.4f}, {cid[1]:+.4f}] | {p:.4f}{star} |")
                    rows.append(dict(setting=setting, outcome=outc, arm=armname, risk=rs, n=len(j),
                                     events=int(y.sum()), our_nseed=nseed, ours_ens=pa,
                                     ours_mean_of_seeds=om, ecgfm_ens=pb,
                                     ecgfm_mean_of_auroc=float(np.mean(indiv)), delta=pa - pb,
                                     d_lo=cid[0], d_hi=cid[1], p=p))
                print(f"  {setting} {outc} rs={rs} done", flush=True)

    open(a.out, "w").write("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(a.out.replace(".md", ".csv"), index=False)
    print(f"[written to {a.out}]")


if __name__ == "__main__":
    main()
