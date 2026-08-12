"""
B2 / B3 / B4 — clinical-utility metrics for Reviewer #3 point 2(b),(c).

  B2  calibration : Brier score, calibration slope & intercept, decile calibration plot
  B3  IDI + NRI   : reclassification / discrimination-slope improvement (bootstrap CIs)
  B4  DCA         : decision-curve net benefit across threshold probabilities

Comparison throughout: **Risk Score alone  vs  CL(+Risk Score)** — this is precisely the gain the
reviewer questions ("ECG + Risk Score yields only a 2-3 point AUROC gain over Risk Score alone").

⚠️ ALL THREE ASSUME CALIBRATED PROBABILITIES.
  * the risk scores (CHARGE-AF / PREVENT-HF) are genuine 5-year risk probabilities
  * the +Risk fused predictions come from a logistic regression -> reasonably calibrated
  * RAW model scores are sigmoid outputs of a fine-tuned network and are NOT calibrated, so IDI/NRI
    on them would partly measure calibration rather than added information. We therefore run these
    metrics on the FUSED arms and report calibration (B2) alongside so the reader can judge.
  * externally (CHS/MESA) the fusion glm was fitted on UKB, so some calibration drift is expected --
    that is itself worth showing.

NOTE on NRI: continuous NRI is known to be inflated by miscalibration and can be positive for a
useless marker (Pepe, Kerr). We report the EVENT and NON-EVENT components separately, never just
the sum, and treat DCA as the primary clinical-utility metric.

Usage: python clinical_utility.py [--B 1000]
"""
import argparse, os, warnings
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
warnings.filterwarnings("ignore")

EV = P("EVAL_ROOT")
OR = EV  # our-model root; --ours_root overrides (e.g. .../eval_3 for seed 3). Risk Score stays under EV.
RS = P("RISK_ROOT", "computed")
RISK_COL = {"af5": "charge_mean", "hf5": "prevent_mean"}
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calib(y, p):
    """Brier, calibration intercept & slope (logistic recalibration of y on logit(p))."""
    from sklearn.linear_model import LogisticRegression
    brier = float(np.mean((p - y) ** 2))
    lr = LogisticRegression(C=1e12, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    slope = float(lr.coef_[0][0]); inter = float(lr.intercept_[0])
    return brier, inter, slope


def platt_cv(y, p, k=5, seed=0):
    """Cross-fitted Platt recalibration: p_cal = sigmoid(a*logit(p)+b), with (a,b) fit out-of-fold
    so every sample's calibrated probability comes from a fit that never saw it (no leakage, full n).
    Monotone -> AUROC/AUPRC unchanged; only the probability scale is corrected. This is the honest
    'recalibrate then deploy' step (fit-on-held-out mirrors fitting on a target-cohort fraction)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    z = logit(np.clip(p, EPS, 1 - EPS)).reshape(-1, 1)
    kk = min(k, int(y.sum()), int((y == 0).sum()))     # guard rare outcomes
    if kk < 2:
        return p
    out = np.empty(len(p), float)
    for tr, te in StratifiedKFold(n_splits=kk, shuffle=True, random_state=seed).split(z, y):
        lr = LogisticRegression(C=1e12, max_iter=1000).fit(z[tr], y[tr])
        out[te] = lr.predict_proba(z[te])[:, 1]
    return out


def idi(y, p_new, p_old):
    e, n = y == 1, y == 0
    return float((p_new[e].mean() - p_old[e].mean()) - (p_new[n].mean() - p_old[n].mean()))


def nri_components(y, p_new, p_old):
    """category-free NRI, reported as separate event / non-event components."""
    e, n = y == 1, y == 0
    up_e = float(np.mean(p_new[e] > p_old[e])); dn_e = float(np.mean(p_new[e] < p_old[e]))
    up_n = float(np.mean(p_new[n] > p_old[n])); dn_n = float(np.mean(p_new[n] < p_old[n]))
    return (up_e - dn_e), (dn_n - up_n)          # event component, non-event component


def net_benefit(y, p, thresholds):
    n = len(y); out = []
    for pt in thresholds:
        flag = p >= pt
        tp = float(np.sum(flag & (y == 1))); fp = float(np.sum(flag & (y == 0)))
        out.append(tp / n - (fp / n) * (pt / (1 - pt)))
    return np.array(out)


def boot_ci(fn, y, a, b, B, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        vals.append(fn(y[idx], a[idx], b[idx]))
    v = np.array(vals, dtype=object)
    if v.ndim == 1 and np.isscalar(v[0]):
        arr = np.array(vals, float); return np.percentile(arr, [2.5, 97.5])
    arr = np.array([list(x) for x in vals], float)
    return np.percentile(arr, [2.5, 97.5], axis=0)


def load(f, col="y_pred"):
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)[["id", "y_true", col]].dropna()
    d["id"] = d["id"].astype(str)
    return d


def cells():
    """(label, outcome, {arm: path}, riskkey) — RiskScore is always the reference arm.

    RAW (unfused) model arms are included: whether the network's sigmoid output is calibrated is
    exactly what a calibration analysis should report, and it shows what the fusion fixes.
    They are used for B2 (calibration) and B4 (DCA); B3 (NRI/IDI) uses only the fused arms, since
    those metrics require comparable probability scales."""
    out = []
    for o in ["af5", "hf5"]:
        arms = {"CL(ECG)": f"{OR}/ukb_test/m75_ecgfull/ecg/{o}/result.csv",
                "CL(ECG+MRI)": f"{OR}/ukb_test/m75_ecgfull/ecg_mri/{o}/result.csv",
                "CL(ECG)+RS": f"{OR}/ukb_test/m75_ecgfull_RS/ecg/{o}/result.csv",
                "CL(ECG+MRI)+RS": f"{OR}/ukb_test/m75_ecgfull_RS/ecg_mri/{o}/result.csv"}
        out.append(("UKB", o, arms, "UKBB"))
        for coh in ["CHS", "MESA"]:
            out.append((coh, o, {"CL(ECG)": f"{OR}/zeroshot/{coh}/m75_ecgfull/{o}/result.csv",
                                 "CL(ECG)+RS": f"{OR}/zeroshot/{coh}/m75_ecgfull_RS/{o}/result.csv"}, coh))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recalibrate", action="store_true",
                    help="cross-fitted Platt-recalibrate EVERY arm (incl. Risk Score) before B2/B3/B4; "
                         "writes to b2_recal/b3_recal/b4_recal, leaving the assessment-only version intact.")
    ap.add_argument("--ours_root", default=EV, help="root for OUR model preds, e.g. .../eval_3 for seed 3")
    ap.add_argument("--out_root", default=EV, help="root for B2/B3/B4 outputs, e.g. .../eval_3 for seed 3")
    a = ap.parse_args()
    global OR; OR = a.ours_root
    OB = a.out_root
    seedtag = "seed" + (a.ours_root.rstrip("/").split("_")[-1] if "eval_" in a.ours_root else "1")
    rng = np.random.default_rng(a.seed)
    suf = "_recal" if a.recalibrate else ""
    for d in ["b2", "b3", "b4"]:
        os.makedirs(f"{OB}/{d}{suf}", exist_ok=True)

    thr = np.arange(0.01, 0.31, 0.01)
    Lc = [f"# B2 — Calibration ({seedtag}){' — RECALIBRATED (5-fold cross-fit Platt)' if a.recalibrate else ''}", "",
          "Brier (lower better); calibration **intercept** 0 = no systematic over/under-prediction; "
          "**slope** 1 = correct spread (<1 = over-extreme predictions).", "",
          "| cohort | outcome | arm | n | events | Brier | intercept | slope |",
          "|---|---|---|---|---|---|---|---|"]
    Li = [f"# B3 — IDI and NRI vs Risk Score alone ({seedtag}){' — RECALIBRATED (5-fold cross-fit Platt)' if a.recalibrate else ''}", "",
          "IDI = improvement in discrimination slope. NRI components reported **separately** "
          "(the sum hides offsetting movement, and continuous NRI is inflated by miscalibration — "
          "see script header). Bootstrap 95% CI.", "",
          "| cohort | outcome | arm | IDI [95% CI] | NRI events | NRI non-events |",
          "|---|---|---|---|---|---|"]
    rows_c, rows_i = [], []

    for coh, o, arms, rskey in cells():
        r = pd.read_csv(f"{RS}/{rskey}_riskscore.csv")[["id", RISK_COL[o]]].rename(
            columns={RISK_COL[o]: "risk"})
        r["id"] = r["id"].astype(str)
        loaded = {nm: load(f) for nm, f in arms.items()}
        loaded = {k: v for k, v in loaded.items() if v is not None}
        if not loaded:
            continue
        base = list(loaded.values())[0][["id", "y_true"]]
        j = base.merge(r, on="id")
        for nm, d in loaded.items():
            j = j.merge(d[["id", "y_pred"]].rename(columns={"y_pred": nm}), on="id")
        j = j.dropna()
        if len(j) < 100:
            continue
        y = j["y_true"].values.astype(int)

        if a.recalibrate:                              # Platt-recalibrate EVERY arm + Risk Score
            for nm in ["risk"] + list(loaded):
                j[nm] = platt_cv(y, j[nm].values.astype(float), seed=a.seed)

        # ---- B2 calibration (risk score + each arm)
        for nm in ["risk"] + list(loaded):
            p = np.clip(j[nm].values.astype(float), EPS, 1 - EPS)
            b, i0, s0 = calib(y, p)
            disp = "RiskScore" if nm == "risk" else nm
            Lc.append(f"| {coh} | {o} | {disp} | {len(y)} | {int(y.sum())} | {b:.4f} | {i0:+.3f} | {s0:.3f} |")
            rows_c.append(dict(cohort=coh, outcome=o, arm=disp, n=len(y), events=int(y.sum()),
                               brier=b, intercept=i0, slope=s0))

        # ---- B3 IDI / NRI, and B4 DCA
        p_old = np.clip(j["risk"].values.astype(float), EPS, 1 - EPS)
        plt.figure(figsize=(6, 4.2))
        plt.plot(thr, net_benefit(y, p_old, thr), label="Risk Score", color="#7b3294", lw=2)
        for nm in loaded:
            p_new = np.clip(j[nm].values.astype(float), EPS, 1 - EPS)
            if not nm.endswith("+RS"):        # NRI/IDI need comparable probability scales
                plt.plot(thr, net_benefit(y, p_new, thr), label=nm, lw=1.5, ls=":")
                continue
            v = idi(y, p_new, p_old)
            lo, hi = boot_ci(idi, y, p_new, p_old, a.B, rng)
            ne, nn = nri_components(y, p_new, p_old)
            (nlo, nnlo), (nhi, nnhi) = boot_ci(nri_components, y, p_new, p_old, a.B, rng)
            Li.append(f"| {coh} | {o} | {nm} | {v:+.4f} [{lo:+.4f}, {hi:+.4f}] "
                      f"| {ne:+.3f} [{nlo:+.3f}, {nhi:+.3f}] | {nn:+.3f} [{nnlo:+.3f}, {nnhi:+.3f}] |")
            rows_i.append(dict(cohort=coh, outcome=o, arm=nm, idi=v, idi_lo=lo, idi_hi=hi,
                               nri_event=ne, nri_nonevent=nn))
            plt.plot(thr, net_benefit(y, p_new, thr), label=nm, lw=2)
        # treat-all / treat-none references
        prev = y.mean()
        plt.plot(thr, [prev - (1 - prev) * (t / (1 - t)) for t in thr], "--", color="gray",
                 label="treat all", lw=1)
        plt.axhline(0, color="k", lw=1, label="treat none")
        plt.ylim(min(-0.005, -0.002), None); plt.xlabel("threshold probability")
        plt.ylabel("net benefit"); plt.title(f"DCA — {coh} {o} (seed 1)", fontsize=10)
        plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(f"{OB}/b4{suf}/dca_{coh}_{o}.png", dpi=150); plt.close()
        print(f"  {coh} {o} done", flush=True)

    open(f"{OB}/b2{suf}/calibration.md", "w").write("\n".join(Lc) + "\n")
    pd.DataFrame(rows_c).to_csv(f"{OB}/b2{suf}/calibration.csv", index=False)
    open(f"{OB}/b3{suf}/idi_nri.md", "w").write("\n".join(Li) + "\n")
    pd.DataFrame(rows_i).to_csv(f"{OB}/b3{suf}/idi_nri.csv", index=False)
    print(f"\n[B2 -> {OB}/b2{suf}/  B3 -> {OB}/b3{suf}/  B4 -> {OB}/b4{suf}/]")


if __name__ == "__main__":
    main()
