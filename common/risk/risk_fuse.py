"""
+Risk late fusion (zero-shot): fit logistic regression  outcome ~ model_score + risk_score  on the
UKB TRAIN split (our CL model's UKB-train predictions + UKB CHARGE-AF/PREVENT-HF), then APPLY it to
the external cohort (CL zero-shot predictions + external risk scores) -> fused +Risk prediction.
Writes a normal result.csv (id,y_true,y_pred) so the existing bootstrap tools give its CI.

Reproduces the R pipeline's fit-on-UKB-train fusion. Uses the mean-over-4-MICE risk score
(charge_mean / prevent_mean) as the single risk covariate.

Usage: python risk_fuse.py --tag m75_ecgfull [--regime zs]
"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

EVAL = "/gpfs/projects/trend/bojun/multimodal_rep/eval"   # overridden by --eval_root
RS = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/computed"
RISK_COL = {"af5": "charge_mean", "hf5": "prevent_mean"}


def load_pred(f):
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)[["id", "y_true", "y_pred"]]
    d["id"] = d["id"].astype(str)
    return d


def load_risk(cohort, outc):
    r = pd.read_csv(f"{RS}/{cohort}_riskscore.csv")[["id", RISK_COL[outc]]].rename(columns={RISK_COL[outc]: "risk"})
    r["id"] = r["id"].astype(str)
    return r


def fit_glm(u, tag, outc, label=""):
    clf = LogisticRegression(C=1e12, max_iter=1000)   # ~unregularized, like glm
    clf.fit(u[["y_pred", "risk"]].values, u["y_true"].values.astype(int))
    print(f"[{tag} {outc}{label}] glm fit on UKB train n={len(u)}  coef(model,risk)={clf.coef_[0].round(3)} "
          f"intercept={clf.intercept_[0]:.3f}")
    return clf


def apply_glm(clf, p, r, sd, tag, outc, label):
    """merge preds+risk, fuse, write result.csv, print model-only vs +Risk AUROC."""
    e = p.merge(r, on="id").dropna(subset=["y_true", "y_pred", "risk"])
    e["y_pred_fused"] = clf.predict_proba(e[["y_pred", "risk"]].values)[:, 1]
    out = e[["id", "y_true", "y_pred_fused"]].rename(columns={"y_pred_fused": "y_pred"})
    os.makedirs(sd, exist_ok=True)
    out.to_csv(f"{sd}/result.csv", index=False)
    y = out.y_true.values
    print(f"    {label} {outc}: +Risk AUROC={roc_auc_score(y, out.y_pred):.4f} "
          f"AUPRC={average_precision_score(y, out.y_pred):.4f}  "
          f"(model-only {roc_auc_score(e.y_true, e.y_pred):.4f})  -> {sd}/result.csv")


EFPAIR = "/gpfs/projects/trend/bojun/multimodal_rep/eval/ecgfm_ukb_paired/seed1"


def _cl_ids(kind, outc):
    """paired id set the CL arms used (kind=train|test) — restrict ECG-FM to the same population."""
    sub = "ukb_train_paired" if kind == "train" else "ukb_test"
    d = load_pred(f"{EVAL}/{sub}/m75_ecgfull/ecg/{outc}/result.csv")
    return set(d["id"]) if d is not None else None


def run_fig2_ecgfm(args):
    """Fig 2 ECG-FM+RS (third bar): fit glm on ECG-FM paired-TRAIN preds, apply to paired-TEST preds,
    both restricted to the CL paired id sets so all three Fig 2 bars share one population."""
    ukb_r = {o: load_risk("UKBB", o) for o in ["af5", "hf5"]}
    for outc in ["af5", "hf5"]:
        trp = load_pred(f"{EFPAIR}/train/{outc}/result.csv")
        tep = load_pred(f"{EFPAIR}/test/{outc}/result.csv")
        if trp is None or tep is None:
            print(f"[ecgfm {outc}] paired ECG-FM preds missing -> skip"); continue
        tr_ids, te_ids = _cl_ids("train", outc), _cl_ids("test", outc)
        trp = trp[trp["id"].isin(tr_ids)].dropna(subset=["y_true", "y_pred"])
        tep = tep[tep["id"].isin(te_ids)].dropna(subset=["y_true", "y_pred"])
        u = trp.merge(ukb_r[outc], on="id").dropna(subset=["y_true", "y_pred", "risk"])
        clf = fit_glm(u, "ecgfm", outc)
        # model-only ECG-FM on the paired test (for the matched bar)
        mo = tep.merge(ukb_r[outc], on="id").dropna(subset=["y_true", "y_pred", "risk"])
        os.makedirs(f"{EVAL}/ukb_test/ecgfm/{outc}", exist_ok=True)
        tep[["id", "y_true", "y_pred"]].to_csv(f"{EVAL}/ukb_test/ecgfm/{outc}/result.csv", index=False)
        sd = f"{EVAL}/ukb_test/ecgfm_RS/{outc}"
        apply_glm(clf, tep, ukb_r[outc], sd, "ecgfm", outc, label="UKB-test ecgfm")


def run_ecgfm_rs(args):
    """ECG-FM +Risk for ALL 4 seeds, following the paper (CHARGE_AF_CHS.Rmd):
    ONE glm per seed, fit on that seed's UKB-train preds + risk, then applied to BOTH
    the UKB test set and the external zero-shot cohorts (the paper's ratio=0 case).
    Verified: our ECGFM_downstream{v} == the paper's seed v (r=1.0000 vs af{v}_pred)."""
    ukb_r = {o: load_risk("UKBB", o) for o in ["af5", "hf5"]}
    for S in [1, 2, 3, 4]:
        for outc in ["af5", "hf5"]:
            root = f"{EVAL}/ecgfm_ukb_paired/seed{S}"
            trp, tep = load_pred(f"{root}/train/{outc}/result.csv"), load_pred(f"{root}/test/{outc}/result.csv")
            if trp is None or tep is None:
                print(f"[ecgfm seed{S} {outc}] preds missing -> skip"); continue
            tr_ids, te_ids = _cl_ids("train", outc), _cl_ids("test", outc)
            trp = trp[trp["id"].isin(tr_ids)].dropna(subset=["y_true", "y_pred"])
            tep = tep[tep["id"].isin(te_ids)].dropna(subset=["y_true", "y_pred"])
            u = trp.merge(ukb_r[outc], on="id").dropna(subset=["y_true", "y_pred", "risk"])
            clf = fit_glm(u, f"ecgfm seed{S}", outc)
            # (a) UKB test
            apply_glm(clf, tep, ukb_r[outc], f"{EVAL}/ukb_test/ecgfm_RS/seed{S}/{outc}",
                      f"ecgfm seed{S}", outc, label=f"UKB-test seed{S}")
            # (b) external zero-shot — same glm, per the paper
            for coh in ["CHS", "MESA"]:
                zp = load_pred(f"{EVAL}/ecgfm_zeroshot_provided/seed{S}/zeroshot/{coh}/{outc}/result.csv")
                if zp is None:
                    continue
                apply_glm(clf, zp.dropna(subset=["y_true", "y_pred"]), load_risk(coh, outc),
                          f"{EVAL}/zeroshot/{coh}/ecgfm_RS/seed{S}/{outc}",
                          f"ecgfm seed{S}", outc, label=f"{coh} zs seed{S}")


def run_fig2(args):
    """UKB Fig 2 +Risk: fit glm on paired UKB-TRAIN preds (per mode), apply to paired UKB-TEST preds."""
    ukb_r = {o: load_risk("UKBB", o) for o in ["af5", "hf5"]}
    for mode in ["ecg", "ecg_mri"]:
        for outc in ["af5", "hf5"]:
            trp = load_pred(f"{EVAL}/ukb_train_paired/{args.tag}/{mode}/{outc}/result.csv")
            tep = load_pred(f"{EVAL}/ukb_test/{args.tag}/{mode}/{outc}/result.csv")
            if trp is None or tep is None:
                print(f"[{args.tag} {mode} {outc}] train/test preds missing -> skip"); continue
            u = trp.merge(ukb_r[outc], on="id").dropna(subset=["y_true", "y_pred", "risk"])
            clf = fit_glm(u, args.tag, outc, label=f" {mode}")
            sd = f"{EVAL}/ukb_test/{args.tag}_RS/{mode}/{outc}"
            apply_glm(clf, tep, ukb_r[outc], sd, args.tag, outc, label=f"UKB-test {mode}")


def run_fig2_clseeds(args):
    """Per-seed UKB +Risk for the 4-seed ensemble comparison. One glm PER SEED, fit on that seed's
    paired-train preds and applied to that seed's paired-test preds -- mirroring how each ECG-FM
    seed got its own glm in run_ecgfm_rs(). Seed 1's inputs live in the canonical (unsuffixed)
    dirs; 0/2/3 live under *_clseeds. Output is uniform: ukb_test_clseeds_RS/seed<S>/..."""
    ukb_r = {o: load_risk("UKBB", o) for o in ["af5", "hf5"]}
    for S in [0, 1, 2, 3]:
        for mode in ["ecg", "ecg_mri"]:
            for outc in ["af5", "hf5"]:
                if S == 1:
                    trf = f"{EVAL}/ukb_train_paired/{args.tag}/{mode}/{outc}/result.csv"
                    tef = f"{EVAL}/ukb_test/{args.tag}/{mode}/{outc}/result.csv"
                else:
                    trf = f"{EVAL}/ukb_train_paired_clseeds/seed{S}/{args.tag}/{mode}/{outc}/result.csv"
                    tef = f"{EVAL}/ukb_test_clseeds/seed{S}/{args.tag}/{mode}/{outc}/result.csv"
                trp, tep = load_pred(trf), load_pred(tef)
                if trp is None or tep is None:
                    print(f"[seed{S} {mode} {outc}] missing -> skip"); continue
                u = trp.merge(ukb_r[outc], on="id").dropna(subset=["y_true", "y_pred", "risk"])
                clf = fit_glm(u, f"{args.tag} seed{S}", outc, label=f" {mode}")
                sd = f"{EVAL}/ukb_test_clseeds_RS/seed{S}/{args.tag}/{mode}/{outc}"
                apply_glm(clf, tep, ukb_r[outc], sd, args.tag, outc, label=f"UKB-test seed{S} {mode}")


def run_zs_clseeds(args):
    """Per-seed EXTERNAL zero-shot +Risk, for the 4-seed ensemble. One glm per seed, fit on that
    seed's UKB-train ECG-only preds, applied to that seed's CHS/MESA zero-shot preds -- mirroring
    run_ecgfm_rs(). Seed 1's zero-shot preds live in zeroshot_clseeds/seed1 (verified identical to
    the canonical eval/zeroshot/<coh>/m75_ecgfull), so all four seeds are read uniformly."""
    for S in [0, 1, 2, 3]:
        for outc in ["af5", "hf5"]:
            trf = f"{EVAL}/ukb_train_preds_clseeds/seed{S}/{args.tag}/{outc}/result.csv"
            trp = load_pred(trf)
            if trp is None:
                print(f"[seed{S} {outc}] UKB-train preds missing -> skip"); continue
            u = trp.merge(load_risk("UKBB", outc), on="id").dropna(subset=["y_true", "y_pred", "risk"])
            clf = fit_glm(u, f"{args.tag} seed{S}", outc)
            for coh in ["CHS", "MESA"]:
                zf = f"{EVAL}/zeroshot_clseeds/seed{S}/{coh}/{args.tag}/{outc}/result.csv"
                zp = load_pred(zf)
                if zp is None:
                    print(f"[seed{S} {coh} {outc}] zero-shot preds missing -> skip"); continue
                apply_glm(clf, zp.dropna(subset=["y_true", "y_pred"]), load_risk(coh, outc),
                          f"{EVAL}/zeroshot_clseeds_RS/seed{S}/{coh}/{args.tag}/{outc}",
                          f"{args.tag} seed{S}", outc, label=f"{coh} zs seed{S}")


def run_ecgfounder_rs(args):
    """ECGFounder +Risk. ECGFounder is a single seed-1 baseline trained on the FULL UKB ECG split
    (not paired), so: fit ONE glm on its UKB-train preds + UKB risk, apply to its UKB-test preds
    (UKB risk) and to its CHS/MESA zero-shot preds (external risk). Mirrors run_ecgfm_rs but for a
    single model. Outputs: ukb_test/ecgfounder_RS, zeroshot/<coh>/ecgfounder_RS."""
    ukb_r = {o: load_risk("UKBB", o) for o in ["af5", "hf5"]}
    for outc in ["af5", "hf5"]:
        trp = load_pred(f"{EVAL}/ukb_train_preds/ecgfounder/{outc}/result.csv")
        if trp is None:
            print(f"[ecgfounder {outc}] UKB-train preds missing -> skip"); continue
        u = trp.merge(ukb_r[outc], on="id").dropna(subset=["y_true", "y_pred", "risk"])
        clf = fit_glm(u, "ecgfounder", outc)
        tep = load_pred(f"{EVAL}/ukb_test/ecgfounder/{outc}/result.csv")
        if tep is not None:
            apply_glm(clf, tep.dropna(subset=["y_true", "y_pred"]), ukb_r[outc],
                      f"{EVAL}/ukb_test/ecgfounder_RS/{outc}", "ecgfounder", outc, label="UKB-test")
        for coh in ["CHS", "MESA"]:
            zp = load_pred(f"{EVAL}/zeroshot/{coh}/ecgfounder/{outc}/result.csv")
            if zp is None:
                continue
            apply_glm(clf, zp.dropna(subset=["y_true", "y_pred"]), load_risk(coh, outc),
                      f"{EVAL}/zeroshot/{coh}/ecgfounder_RS/{outc}", "ecgfounder", outc, label=f"{coh} zs")


def run_single_rs(args):
    """+Risk for ANY single-model arm given --tag (fit glm on its UKB-train preds, apply to its
    UKB-test + CHS/MESA zero-shot preds). Generalises run_ecgfounder_rs. Handles af5 AND hf5."""
    tag = args.tag
    ukb_r = {o: load_risk("UKBB", o) for o in ["af5", "hf5"]}
    for outc in ["af5", "hf5"]:
        trp = load_pred(f"{EVAL}/ukb_train_preds/{tag}/{outc}/result.csv")
        if trp is None:
            print(f"[{tag} {outc}] UKB-train preds missing -> skip"); continue
        u = trp.merge(ukb_r[outc], on="id").dropna(subset=["y_true", "y_pred", "risk"])
        clf = fit_glm(u, tag, outc)
        tep = load_pred(f"{EVAL}/ukb_test/{tag}/{outc}/result.csv")
        if tep is not None:
            apply_glm(clf, tep.dropna(subset=["y_true", "y_pred"]), ukb_r[outc],
                      f"{EVAL}/ukb_test/{tag}_RS/{outc}", tag, outc, label="UKB-test")
        for coh in ["CHS", "MESA"]:
            zp = load_pred(f"{EVAL}/zeroshot/{coh}/{tag}/{outc}/result.csv")
            if zp is None:
                continue
            apply_glm(clf, zp.dropna(subset=["y_true", "y_pred"]), load_risk(coh, outc),
                      f"{EVAL}/zeroshot/{coh}/{tag}_RS/{outc}", tag, outc, label=f"{coh} zs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m75_ecgfull")
    ap.add_argument("--regime", default="zs", choices=["zs", "fig2", "fig2_ecgfm", "ecgfm_rs", "fig2_clseeds", "zs_clseeds", "ecgfounder_rs", "single_rs"])
    ap.add_argument("--eval_root", default=None,
                    help="root for OUR predictions + outputs (e.g. eval_unseed). ECG-FM arms are "
                         "unaffected; they live in the canonical eval/.")
    args = ap.parse_args()
    if args.eval_root:
        global EVAL
        EVAL = args.eval_root

    if args.regime == "fig2":
        run_fig2(args); return
    if args.regime == "fig2_ecgfm":
        run_fig2_ecgfm(args); return
    if args.regime == "ecgfm_rs":
        run_ecgfm_rs(args); return
    if args.regime == "fig2_clseeds":
        run_fig2_clseeds(args); return
    if args.regime == "zs_clseeds":
        run_zs_clseeds(args); return
    if args.regime == "ecgfounder_rs":
        run_ecgfounder_rs(args); return
    if args.regime == "single_rs":
        run_single_rs(args); return

    for outc in ["af5", "hf5"]:
        # ---- fit glm on UKB train: outcome ~ model_score + risk_score ----
        ukb_p = load_pred(f"{EVAL}/ukb_train_preds/{args.tag}/{outc}/result.csv")
        if ukb_p is None:
            print(f"[{args.tag} {outc}] UKB-train preds missing -> skip"); continue
        ukb_r = load_risk("UKBB", outc)
        u = ukb_p.merge(ukb_r, on="id").dropna(subset=["y_true", "y_pred", "risk"])
        clf = LogisticRegression(C=1e12, max_iter=1000)   # ~unregularized, like glm
        clf.fit(u[["y_pred", "risk"]].values, u["y_true"].values.astype(int))
        print(f"[{args.tag} {outc}] glm fit on UKB train n={len(u)}  coef(model,risk)={clf.coef_[0].round(3)} "
              f"intercept={clf.intercept_[0]:.3f}")

        # ---- apply to external zero-shot ----
        for coh in ["CHS", "MESA"]:
            ext_p = load_pred(f"{EVAL}/zeroshot/{coh}/{args.tag}/{outc}/result.csv")
            if ext_p is None:
                continue
            ext_r = load_risk(coh, outc)
            e = ext_p.merge(ext_r, on="id").dropna(subset=["y_true", "y_pred", "risk"])
            e["y_pred_fused"] = clf.predict_proba(e[["y_pred", "risk"]].values)[:, 1]
            out = e[["id", "y_true", "y_pred_fused"]].rename(columns={"y_pred_fused": "y_pred"})
            sd = f"{EVAL}/zeroshot/{coh}/{args.tag}_RS/{outc}"
            os.makedirs(sd, exist_ok=True)
            out.to_csv(f"{sd}/result.csv", index=False)
            v = out.dropna(subset=["y_true", "y_pred"]); y = v.y_true.values
            m0 = e.dropna(subset=["y_true", "y_pred"])
            print(f"    {coh} {outc}: +Risk AUROC={roc_auc_score(y, v.y_pred):.4f} "
                  f"AUPRC={average_precision_score(y, v.y_pred):.4f}  "
                  f"(model-only was {roc_auc_score(m0.y_true, m0.y_pred):.4f})  -> {sd}/result.csv")


if __name__ == "__main__":
    main()
