# Figure numbers — machine-readable plotting data

Every main/supplementary figure's numbers are cached here so the figures can be **re-plotted directly
from these files without re-running any bootstrap/MICE**. One subfolder per figure; the JSON/CSV inside
is exactly what that figure's render step reads.

Headline config: our model = **m75 seed 3**, ECG-FM baseline = **seed 1**. Bootstraps use **B=2000**
(fig4 B=1000), seed 42. Each figure script uses a **compute → cache → render** split — these files are
the cached compute; rendering reads them in seconds.

> ⚠️ **UKB `+Risk` numbers are provisional** — the UKB CHARGE-AF/PREVENT-HF risk score is mis-mapped.
> Any UKB **+RS** bar/cell (fig2) is affected. All **model-only** and **all CHS/MESA** risk numbers are correct.

| subfolder | file | figure | generating script |
|---|---|---|---|
| `fig2_ukb/` | `fig23_numbers_our3_ecgfm1.json` | Fig 2 — UKB AF/HF model bars | `UKBB/figures/make_figures.py` |
| `fig3_external_zeroshot/` | `fig23_numbers_our3_ecgfm1.json` (same file, keys `CHS\|*`, `MESA\|*`) | Fig 3 — external zero-shot | `UKBB/figures/make_figures.py` |
| `fig4_risk_stratification/` | `fig4_stats_our3.json` | Fig 4 — CHS/MESA tertile Cox (adj + unadj) | `CHS_MESA/fig4_compare.py` |
| `fig5_fewshot/` | `fig5_stats.json` | Fig 5 — few-shot (10%/20% × CHS/MESA) | `CHS_MESA/fig5_fewshot.py` |
| `survival_ukb/` | `cox_cindex_stats.json` | Survival C-index, UKB (5 models) | `UKBB/figures/fig_cox_compare.py` |
| `survival_zeroshot/` | `cox_cindex_zeroshot_stats.json` | Survival C-index, zero-shot CHS/MESA | `CHS_MESA/fig_cox_zeroshot.py` |
| `supp_cmr_feature_cox/` | `cmr_feature_cox_external_seed3_stats.json` | Supp — CMR-feature Cox (CHS/MESA) | `CHS_MESA/cmr_feature_cox_external.py` |
| `supp_cmr_regression/` | `cmr_test_results_seed3.csv` | Supp — CMR regression accuracy (UKB) | `UKBB/figures/cmr_eval.py` |
| `supp_mesa_cmr_corr/` | `mesa_measured_vs_predicted.csv` | MESA predicted-vs-measured CMR correlation (editor E1) | `CHS_MESA/mesa_cmr_corr.py` |
| `supp_subgroups/` | `subgroups.csv` | Supp — subgroup AUROC | `CHS_MESA/fig_subgroups.py` |
| `reviewer2_refit_clinical/` | `refit_clinical_compare.json` + risk-score refit tables + `seed1_vs_seed3_comparison.csv` | Reviewer 2 — refit clinical incremental value | `CHS_MESA/refit_clinical_compare.py` |

## Key schemas (per file)

- **`fig23_numbers_our3_ecgfm1.json`** — keys `UKB|af5`, `UKB|hf5` (fig2) and `CHS|*`, `MESA|*` (fig3).
  Each: `n`, `pos`, `neg`, and per metric (`auroc`, `auprc`): `bars` = `{label:{point,ci_lo,ci_hi,family,is_rs}}`;
  `pairs` = `[{bar,vs,delta,ci_lo,ci_hi,p,sig}]` (each OUR bar vs each baseline, risk-tier-matched paired bootstrap).
- **`fig4_stats_our3.json`** — `rows` = `[{coh,outc,arm, hr_adj:{intermediate:[hr,lo,hi],high:[…]}, hr_unadj:{…}, c,n,ev}]`
  (`hr_unadj` = univariable tertile HR; `hr_adj` = covariate-adjusted, MICE/Rubin). `comparisons` =
  `[{coh,outc,dloghr,dloghr_ci,p_hr,dc,dc_ci,p_c}]` (ours vs RiskScore, paired bootstrap).
- **`fig5_stats.json`** — keys `CHS|train_valid_5_5` (10%), `…_10_10` (20%), `MESA|…`. Each → outcome list
  `{o,npos,nn, pt:{family:auroc}, ci:{family:[lo,hi]}, marks, tests:{baseline:{delta_auroc,delta_ci,p,sig}}}`.
- **`cox_cindex_stats.json` / `cox_cindex_zeroshot_stats.json`** — `bars`={`"af|tag"`:{cindex,ci:[lo,hi],events,n}},
  `marks`, `tests`={outcome:{baseline:{delta_cindex,delta_ci,p,sig}}}. Zero-shot keys are `CHS|af`,`CHS|hf`,`MESA|af`,`MESA|hf`.
- **`cmr_feature_cox_external_seed3_stats.json`** — keys `CHS_pred`,`MESA_pred`,`MESA_meas`, each list of
  `{outcome,feature,HR,HR_low,HR_high,p,n,events,beta_bar,Tvar,…}` (per 1 SD, Rubin-pooled).
- **`mesa_measured_vs_predicted.csv`** — `feature, n, ours_r (MESA Pearson), ukb_r_ours (UKB reference)`.
- **`refit_clinical_compare.json`** — `results` = `[{cohort,outcome,n,events, metrics:{clinical,ecg,full each {auroc,auprc}},
  d_full_minus_clinical, d_ecg_minus_clinical}]` where each `d_*` = `{auroc:{delta,ci,p}, auprc:{delta,ci,p}}` (paired bootstrap).
- **`subgroups.csv`** — `cohort,outcome,subgroup,arm,n,events,auroc,lo,hi`.
- **`cmr_test_results_seed3.csv`** — `feature,n,ours_r,ours_R2,paper_split1_r,paper_split1_R2,paper_published_r`.
