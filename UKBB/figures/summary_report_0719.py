"""
Summary table (2026-07-19): our m75 (seed 1) vs EACH of the paper's 4 ECG-FM seeds, in three settings:
  1. UKB test        (paired UKB test; m75 arms: ecg and ecg_mri)
  2. Zero-shot       (CHS/MESA, UKB-trained predictor applied directly)
  3. Few-shot 20%    (CHS/MESA, train_valid_10_10)
No risk-score fusion.

Per cell: all models inner-joined on `id` so every number is on ONE common population. Reports our
point AUROC + stratified bootstrap 95% CI, each ECG-FM seed's AUROC, and the paired-bootstrap
Δ = ours − ECG-FM_seed with a two-sided p-value (paired on id, so shared test noise cancels).

NOTE: our model is FIXED at seed 1, so these p-values ask "does our model beat THIS ECG-FM model?"
They do NOT account for our own seed variance -> not a claim about the method in general.

Usage: python summary_report_0719.py [--B 2000] [--out <path.md>]
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

# --- repo path configuration (see docs/PATHS.md) ---
import os as _os, sys as _sys
_pd = _os.path.dirname(_os.path.abspath(__file__))
while _pd != "/" and not _os.path.isdir(_os.path.join(_pd, "common")):
    _pd = _os.path.dirname(_pd)
_sys.path.insert(0, _os.path.join(_pd, "common"))
from paths import P


EV = P("EVAL_ROOT")
EXT = P("FEWSHOT_RESULTS")
SEEDS = [1, 2, 3, 4]
OURS = "m75_ecgfull"


def with_ecgfounder(ef_dict, ecgf_path, deepssl_path=None, deepsl_path=None):
    """append the fine-tuned single-model baselines next to the 4 ECG-FM seeds:
      ECGFounder    (lr 5e-6, patience 3)
      DeepECG-SSL   (its SSL wav2vec2-CMSC backbone fine-tuned by us on UKB, SAME protocol)
      DeepECG-SL    (its EfficientNet1DV2 77-class backbone fine-tuned by us on UKB, SAME protocol)
    All model-independent (identical across our seeds), read from canonical EV. The OFF-THE-SHELF
    DeepECG AF models (zs) are NOT here -- AF-only and not like-for-like; they live in
    eval/deepecg/deepecg_compare.md."""
    d = {**ef_dict, "ECGFounder": ecgf_path}
    for name, path in [("DeepECG-SSL", deepssl_path), ("DeepECG-SL", deepsl_path)]:
        if path is not None and os.path.exists(path):
            d[name] = path
    return d


def fewshot_baselines(OR, coh, frdir, frac, o):
    """section-4 baselines for one cell: our fraction-matched ECG-FM, plus ECGFounder at the same
    fraction IF it was run for this outcome (matched Fig-5 protocol). Existence-checked so a failed
    ECGFounder outcome degrades to the ECG-FM-only comparison instead of dropping the whole cell."""
    base = {f"ECG-FM ({frac})": f"{OR}/fewshot/{coh}/{frdir}/ecgfm/{o}/result.csv"}
    for label, tag in [("ECGFounder", "ecgfounder"), ("DeepECG-SSL", "deepssl_ft"), ("DeepECG-SL", "deepsl_ft")]:
        p = f"{OR}/fewshot/{coh}/{frdir}/{tag}/{o}/result.csv"
        if os.path.exists(p):
            base[f"{label} ({frac})"] = p
    return base
# few-shot outcomes present in BOTH our fewshot/ and the paper's ECG-FM files.
# (ces5 is ours-only; the paper's ATH_LAC is a combined form with no match on our side)
# af5/hf5 are deliberately EXCLUDED here: they have UKB labels, so they are reported zero-shot
# (section 2). The broad outcomes have no UKB label, so few-shot is the only option for them.
FS_OUTCOMES = ["mi5", "mi10", "is5", "is10", "ces10", "cvddth5", "cvddth10", "dth5", "dth10"]
# Ours-only few-shot outcomes the paper never evaluated (no paper ECG-FM baseline, so they cannot
# go in section 3). They appear ONLY in section 4 vs our OWN fraction-matched ECG-FM. Cohort
# availability varies (e.g. hfpef/hfref exist for CHS, ce/laa for MESA), so cells are emitted only
# where both result.csv exist.
NEW_FS_OUTCOMES = ["ath5", "ath10", "ces5", "hfpef5", "hfpef10", "hfref5", "hfref10",
                   "lac5", "lac10", "ce5", "ce10", "laa5", "laa10"]


def fauc(y, s):  # fast rank-based AUROC
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def aupr(y, s):  # AUPRC (average precision)
    if int(np.sum(y)) < 1 or int(np.sum(y)) == len(y):
        return np.nan
    return average_precision_score(y, s)


def load(f):
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)[["id", "y_true", "y_pred"]]
    d["id"] = d["id"].astype(str)
    return d


def ci(y, s, B, rng):
    """AUROC and AUPRC point + 95% CI in ONE shared resample pass."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    ro = np.empty(B); pr = np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        yy = y[idx]; ss = s[idx]
        ro[b] = fauc(yy, ss); pr[b] = aupr(yy, ss)
    return ((fauc(y, s), *np.percentile(ro, [2.5, 97.5])),
            (aupr(y, s), *np.percentile(pr, [2.5, 97.5])))


def paired_boot(y, pa, pb, B, rng):
    """Δ = A - B for AUROC and AUPRC, paired on the SAME resample. Returns two (Δ,lo,hi,p) tuples."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    do = np.empty(B); dp = np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        yy = y[idx]
        do[b] = fauc(yy, pa[idx]) - fauc(yy, pb[idx])
        dp[b] = aupr(yy, pa[idx]) - aupr(yy, pb[idx])
    po = 2.0 * min((do >= 0).mean(), (do <= 0).mean())
    pp = 2.0 * min((dp >= 0).mean(), (dp <= 0).mean())
    return ((fauc(y, pa) - fauc(y, pb), *np.percentile(do, [2.5, 97.5]), min(po, 1.0)),
            (aupr(y, pa) - aupr(y, pb), *np.percentile(dp, [2.5, 97.5]), min(pp, 1.0)))


def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def cell(ours_f, ef_files, B, rng):
    """returns (n, pos, (ours_auc_ci, ours_apr_ci), [rows]) or None. Each row:
    (seed, ef_auc, ef_apr, (dAUROC,lo,hi,p), (dAUPRC,lo,hi,p))."""
    od = load(ours_f)
    efs = {s: load(f) for s, f in ef_files.items()}
    if od is None or any(v is None for v in efs.values()):
        return None
    m = od.rename(columns={"y_pred": "p_ours"})[["id", "y_true", "p_ours"]]
    for s, e in efs.items():
        m = m.merge(e.rename(columns={"y_pred": f"e{s}"})[["id", f"e{s}"]], on="id")
    m = m.dropna(subset=["y_true", "p_ours"] + [f"e{s}" for s in efs])
    if m.empty or m["y_true"].nunique() < 2:
        return None
    y = m["y_true"].values.astype(int); po = m["p_ours"].values
    rows = []
    for s in efs:
        pe = m[f"e{s}"].values
        dauc, dapr = paired_boot(y, po, pe, B, rng)
        rows.append((s, fauc(y, pe), aupr(y, pe), dauc, dapr))
    return len(y), int(y.sum()), ci(y, po, B, rng), rows


def emit(L, title, cells, B, rng):
    L.append(f"\n## {title}\n")
    L.append("| cell | n (pos) | **m75 AUROC [95% CI]** | baseline | bl AUROC | ΔAUROC [95% CI] | p "
             "| **m75 AUPRC [95% CI]** | bl AUPRC | ΔAUPRC [95% CI] | p |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for label, ours_f, ef_files in cells:
        r = cell(ours_f, ef_files, B, rng)
        if r is None:
            L.append(f"| {label} | — | *missing* | | | | | | | | |"); continue
        n, pos, ((oa, olo, ohi), (opa, oplo, ophi)), rows = r
        wins = sum(1 for _, _, _, (d, _, _, p), _ in rows if p < 0.05 and d > 0)
        winp = sum(1 for _, _, _, _, (d, _, _, p) in rows if p < 0.05 and d > 0)
        for i, (s, ea, epa, (d, lo, hi, p), (dp, lpp, hpp, pp)) in enumerate(rows):
            head = (f"| **{label}** | {n} ({pos}) | **{oa:.3f} [{olo:.3f}, {ohi:.3f}]** "
                    if i == 0 else "| | | ")
            aprc = f"**{opa:.3f} [{oplo:.3f}, {ophi:.3f}]**" if i == 0 else ""
            nm = f"seed{s}" if isinstance(s, int) else str(s)
            L.append(head + f"| {nm} | {ea:.3f} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | {p:.3f} {sig(p)} "
                     f"| {aprc} | {epa:.3f} | {dp:+.3f} [{lpp:+.3f}, {hpp:+.3f}] | {pp:.3f} {sig(pp)} |")
        L.append(f"| | | *beats {wins}/{len(rows)} AUROC* | | | | | *beats {winp}/{len(rows)} AUPRC* | | | |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ours_root", default=None,
                    help="root holding OUR predictions (e.g. eval_unseed). ECG-FM baselines always "
                         "come from the canonical eval/ since they are model-independent.")
    ap.add_argument("--skip_fewshot", action="store_true")
    ap.add_argument("--only_fewshot", action="store_true",
                    help="emit ONLY the few-shot sections (3, 4a-d); skip UKB-test + zero-shot "
                         "(sections 1-2), which live in the main comparison report.")
    ap.add_argument("--label", default="seed 1", help="how to describe OUR model in the title")
    ap.add_argument("--risk", action="store_true",
                    help="+Risk (late-fusion) variant. AF/HF only — CHARGE-AF/PREVENT-HF are "
                         "outcome-specific, so the broad few-shot outcomes have no risk score.")
    ap.add_argument("--ecgfm_seed", type=int, default=None,
                    help="restrict the ECG-FM baseline to a SINGLE seed (e.g. 1) instead of all 4. "
                         "Use for a representative-seed report (our seed vs one ECG-FM seed).")
    args = ap.parse_args()
    if args.ecgfm_seed is not None:
        global SEEDS
        SEEDS = [args.ecgfm_seed]
    if args.out is None:
        args.out = f"{EV}/summary_report_0719{'_risk' if args.risk else ''}.md"
    rng = np.random.default_rng(args.seed)
    OR = args.ours_root or EV          # our predictions
    # (EV stays the ECG-FM baseline root throughout)

    if args.risk:
        L = [f"# **+Risk**: m75+RS ({args.label}) vs 4 ECG-FM+RS seeds + ECGFounder+RS + DeepECG-SSL+RS + DeepECG-SL+RS (B={args.B})",
             "",
             "Late fusion `outcome ~ model_score + risk_score`, following the paper "
             "(`CHARGE_AF_CHS.Rmd`): **one glm per seed**, fit on that seed's UKB-train predictions, "
             "then applied to both the UKB test set and the external cohorts.",
             "Risk score = CHARGE-AF (af5) / PREVENT-HF (hf5), mean over the 4 MICE imputations.",
             "",
             "> **AF/HF only.** CHARGE-AF and PREVENT-HF are outcome-specific; there is no validated "
             "risk score for MI/stroke/CES/death, so the broad few-shot outcomes cannot have a +Risk arm.",
             "> Our model is fixed at **seed 1**; ECG-FM seeds verified to match the paper "
             "(r=1.0000 vs its `af{v}_pred` columns)."]
        cells = []
        for mode in ["ecg", "ecg_mri"]:
            for o in ["af5", "hf5"]:
                cells.append((f"UKB {mode} {o}",
                              f"{OR}/ukb_test/{OURS}_RS/{mode}/{o}/result.csv",
                              with_ecgfounder({s: f"{EV}/ukb_test/ecgfm_RS/seed{s}/{o}/result.csv" for s in SEEDS},
                                              f"{EV}/ukb_test/ecgfounder_RS/{o}/result.csv",
                                              f"{EV}/ukb_test/deepssl_ft_RS/{o}/result.csv",
                                              f"{EV}/ukb_test/deepsl_ft_RS/{o}/result.csv")))
        emit(L, "1. UKB test +Risk (paired UKB test set)", cells, args.B, rng)
        cells = [(f"{coh} {o}",
                  f"{OR}/zeroshot/{coh}/{OURS}_RS/{o}/result.csv",
                  with_ecgfounder({s: f"{EV}/zeroshot/{coh}/ecgfm_RS/seed{s}/{o}/result.csv" for s in SEEDS},
                                  f"{EV}/zeroshot/{coh}/ecgfounder_RS/{o}/result.csv",
                                  f"{EV}/zeroshot/{coh}/deepssl_ft_RS/{o}/result.csv",
                                  f"{EV}/zeroshot/{coh}/deepsl_ft_RS/{o}/result.csv"))
                 for coh in ["CHS", "MESA"] for o in ["af5", "hf5"]]
        emit(L, "2. Zero-shot +Risk (CHS / MESA)", cells, args.B, rng)
        txt = "\n".join(L)
        with open(args.out, "w") as f:
            f.write(txt + "\n")
        print(txt); print(f"\n[written to {args.out}]")
        return

    _title = (f"# Broad-outcome FEW-SHOT: m75 ({args.label}) vs ECG-FM, ECGFounder, DeepECG-SSL, DeepECG-SL — AUROC/AUPRC, bootstrap CI, paired p (B={args.B})"
              if args.only_fewshot else
              f"# m75 ({args.label}) vs 4 ECG-FM seeds + ECGFounder + DeepECG-SSL + DeepECG-SL — AUROC, bootstrap CI, paired p (B={args.B})")
    L = [_title,
         "*Few-shot only. UKB-test and external zero-shot (AF/HF) live in the main comparison report; excluded here.*" if args.only_fewshot else "",
         "",
         "All models inner-joined on `id` per cell, so every row is on one common population.",
         "Δ is a **paired** bootstrap (same resample for both models), so shared test-set noise cancels.",
         "**ECGFounder** (seed-1, matched protocol: lr 5e-6, patience-3 early stop) is a baseline in the "
         "AF/HF cells; the few-shot broad-outcome sections (3, 4a-d) compare our model against all four "
         "fine-tuned baselines (ECG-FM, ECGFounder, DeepECG-SSL, DeepECG-SL), each adapted by the same "
         "few-shot protocol (fine-tune on the cohort fraction, test on the full test split).",
         "",
         f"> **Scope:** our model is **{args.label}**. These p-values ask *\"does our model beat THIS "
         "ECG-FM model?\"* — they do **not** include our own seed variance, so they are not a claim that "
         "the *method* is better in general. ECG-FM's seed SD is large (~0.05–0.07), so a seed-aware "
         "test would be stricter.",
         "> **No risk-score fusion** in any cell here."]

    # 1. UKB test (paired) — m75 ecg and ecg_mri; ECG-FM is ECG-only
    cells = []
    for mode in ["ecg", "ecg_mri"]:
        for o in ["af5", "hf5"]:
            cells.append((f"UKB {mode} {o}",
                          f"{OR}/ukb_test/{OURS}/{mode}/{o}/result.csv",
                          with_ecgfounder({s: f"{EV}/ecgfm_ukb_paired/seed{s}/test/{o}/result.csv" for s in SEEDS},
                                          f"{EV}/ukb_test/ecgfounder/{o}/result.csv",
                                          f"{EV}/ukb_test/deepssl_ft/{o}/result.csv",
                                          f"{EV}/ukb_test/deepsl_ft/{o}/result.csv")))
    if not args.only_fewshot:
        emit(L, "1. UKB test (paired UKB test set)", cells, args.B, rng)

    # 2. Zero-shot
    cells = [(f"{coh} {o}",
              f"{OR}/zeroshot/{coh}/{OURS}/{o}/result.csv",
              with_ecgfounder({s: f"{EV}/ecgfm_zeroshot_provided/seed{s}/zeroshot/{coh}/{o}/result.csv" for s in SEEDS},
                              f"{EV}/zeroshot/{coh}/ecgfounder/{o}/result.csv",
                              f"{EV}/zeroshot/{coh}/deepssl_ft/{o}/result.csv",
                              f"{EV}/zeroshot/{coh}/deepsl_ft/{o}/result.csv"))
             for coh in ["CHS", "MESA"] for o in ["af5", "hf5"]]
    if not args.only_fewshot:
        emit(L, "2. Zero-shot (CHS / MESA)", cells, args.B, rng)

    # 3. Few-shot 20% (train_valid_10_10) vs the paper's ECG-FM few-shot preds
    cells = [(f"{coh} {o}",
              f"{EV}/fewshot/{coh}/train_valid_10_10/{OURS}/{o}/result.csv",
              {s: f"{EXT}/{coh}_{o.upper()}_ECGFM{s}.csv" for s in SEEDS})
             for coh in ["CHS", "MESA"] for o in FS_OUTCOMES]
    if args.skip_fewshot:
        cells = []
    emit(L, "3. Few-shot 20% (CHS / MESA) — 9 broad outcomes vs the paper's 4 ECG-FM seeds "
            "(AF/HF are zero-shot, reported separately)",
         cells, args.B, rng)

    # 4. Data efficiency: 10% vs 20%, each against the FRACTION-MATCHED ECG-FM.
    # For the broad (Fig-5) outcomes our ECG-FM arm starts from RAW pretrained ECG-FM weights and is
    # fine-tuned on the same cohort fraction (03_fewshot.sh line 71, empty init arg) -- it never
    # touches the retired 60/20/20 UKB predictor, so this comparison is clean and fraction-matched.
    if not args.skip_fewshot:
        for frac, frdir in [("10%", "train_valid_5_5"), ("20%", "train_valid_10_10")]:
            cells = [(f"{coh} {o}",
                      f"{OR}/fewshot/{coh}/{frdir}/{OURS}/{o}/result.csv",
                      fewshot_baselines(OR, coh, frdir, frac, o))
                     for coh in ["CHS", "MESA"] for o in FS_OUTCOMES]
            emit(L, f"4{'a' if frac=='10%' else 'b'}. Few-shot {frac} — vs 4 fraction-matched fine-tuned baselines (ECG-FM, ECGFounder, DeepECG-SSL, DeepECG-SL)",
                 cells, args.B, rng)
        L.append("\n> **Section 4 uses a different baseline from section 3.** Section 3 compares "
                 "against the **paper's** ECG-FM, which exists only at 20%. Section 4 compares against "
                 "**our own** ECG-FM fine-tuned at the *same* fraction (raw pretrained weights + that "
                 "cohort fraction), so 10% and 20% are directly comparable to each other.")

        # 4c / 4d: outcomes the PAPER never evaluated (no paper baseline). Only vs our own ECG-FM.
        for frac, frdir, tag in [("10%", "train_valid_5_5", "4c"), ("20%", "train_valid_10_10", "4d")]:
            cells = []
            for coh in ["CHS", "MESA"]:
                for o in NEW_FS_OUTCOMES:
                    of = f"{OR}/fewshot/{coh}/{frdir}/{OURS}/{o}/result.csv"
                    ef = f"{OR}/fewshot/{coh}/{frdir}/ecgfm/{o}/result.csv"
                    if os.path.exists(of) and os.path.exists(ef):   # emit only available comparisons
                        cells.append((f"{coh} {o}", of, fewshot_baselines(OR, coh, frdir, frac, o)))
            emit(L, f"{tag}. Few-shot {frac} — NEW outcomes not in the paper (vs 4 fraction-matched fine-tuned baselines (ECG-FM, ECGFounder, DeepECG-SSL, DeepECG-SL))",
                 cells, args.B, rng)
        L.append("\n> **Sections 4c/4d** cover outcomes the paper never benchmarked (e.g. HFpEF/HFrEF "
                 "subtypes, atherosclerosis). No paper ECG-FM baseline exists for them, so they appear "
                 "only here, against our own fraction-matched ECG-FM. Cohort coverage varies with which "
                 "outcomes each cohort recorded.")

    txt = "\n".join(L)
    with open(args.out, "w") as f:
        f.write(txt + "\n")
    print(txt); print(f"\n[written to {args.out}]")


if __name__ == "__main__":
    main()
