# Configuring paths

The scripts in this repository were developed on the authors' SLURM cluster and still carry **absolute
paths** (`/gpfs/projects/trend/...`), declared as module-level constants at the top of each `.py` file
and as shell variables in each `.sh` file. They will not resolve on your system.

This document maps every one of them so you can repoint them. There are 65 distinct absolute paths, but
they reduce to the **12 logical roots** below — set those, and the rest follow.

> These are plain constants, not CLI arguments. There is no environment-variable indirection yet;
> repointing means editing the constant. A refactor to read them from the environment is a known
> outstanding item.

## The logical roots

| root | what it holds | who writes it |
|---|---|---|
| `UKB_MRI_DIR` | UK Biobank cardiac MRI, per subject: `vst_2ch.npy`, `vst_4ch.npy`, `vst_sa.npy` | your preprocessing (see README → Data) |
| `UKB_PHENO_DIR` | UKB phenotype/outcome CSVs, split into `MRI_train/`, `MRI_valid_new/`, `MRI_test_new/`, one `<outcome>.csv` each | your cohort definition |
| `UKB_ECG_ROOT` | UKB ECG: `ECG_manifest*/` (`.tsv` manifests), `ECG_label*/`, `ECG_label_surv/<outcome>/`, and `stage1/` (alignment splits + `ecg_tsv/`) | your preprocessing |
| `MRI_SPLITS` | subject-ID lists for MAE pretraining, `train/` and `val/` | your split definition |
| `CHS_ECG_ROOT` / `MESA_ECG_ROOT` | external cohort ECG, same layout as `UKB_ECG_ROOT` (`ECG_manifest/`, `ECG_label_surv/<outcome>/`) | your preprocessing |
| `RISK_ROOT` | clinical risk-score inputs and outputs: `CHARGE-PREVENT/` (raw risk-factor tables `charge_prevent_<cohort>.csv`), `computed/` (MICE-imputed scores), `csv_HR/`, `csv_train_valid_test_individual_id_disease/` | `common/risk/risk_score.py` writes `computed/` |
| `MESA_TABLES` | MESA measured-CMR and disease tables (`MESA_CMR_features.csv`, `MESA_disease.csv`) | MESA data release |
| `CKPT_ROOT` | all model checkpoints (MAE, stage-1, downstream runs) | training scripts |
| `ECG_CKPT` | the ECG-FM backbone file (`mimic_iv_ecg_physionet_pretrained.pt`) | upstream release — see `weights/README.md` |
| `EVAL_ROOT` | prediction/eval outputs consumed by every figure script: `<run>/result.csv`, `cox/`, `cox_zeroshot/`, `figures/`, `logs/` | eval scripts |
| `FEWSHOT_RESULTS` / `EXT_RESULTS` | external few-shot and aggregate result tables | `CHS_MESA/` scripts |
| `CONDA_ENV` | the conda environment path used by the `.sh` files (`conda activate <path>`) | your environment |

### Stale roots

Three roots in the `.sh` files refer to the authors' pre-reorganization tree and are **obsolete** —
they point at directories that are now inside this repository:

| old path | now |
|---|---|
| `.../bojun/multimodal` | `UKBB/contrastive/` and `UKBB/downstream/` |
| `.../bojun/mri/CineMA/pretrain` | `UKBB/pretrain_mri/` |
| `.../bojun/multimodal_rep/ECGFounder_DeepSSL` | `baselines/` |
| `.../bojun/CHS_MESA/scripts` | `common/train_eval/` (the `SCRIPTS` variable) |

Every `cd <one of these>` in a `.sh` file should become a `cd` into the corresponding repository
directory (or be dropped, since the scripts resolve `common/` via `sys.path` anyway).

## The missing `config.sh`

`UKBB/downstream/run_cox.sh`, `UKBB/downstream/run_cox_ecgfm.sh`, and `CHS_MESA/run_cox_zeroshot.sh`
each begin with

```bash
source /gpfs/projects/trend/bojun/CHS_MESA/scripts/config.sh
```

That file was cluster-local and is **not** part of this repository. A working template is provided at
[`env/config.example.sh`](../env/config.example.sh) — copy it, fill in your paths, and repoint the
`source` line. It must define `SCRIPTS`, `ECGFM`, `CKPT_ROOT`, `CHS`, `MESA`, and the `pick_ckpt`
helper.

## Per-file reference

Every line in the repository containing an absolute path, grouped by file. The `root` column says
which logical root above it belongs to.

### `CHS_MESA/a3_subgroups.py`

| line | root | current value |
|---|---|---|
| 18 | `EVAL_ROOT` | `EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"` |
| 20 | `RISK_ROOT` | `CP = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/CHARGE-PREVENT"` |
| 21 | `RISK_ROOT` | `RS = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/computed"` |

### `CHS_MESA/clinical_utility.py`

| line | root | current value |
|---|---|---|
| 33 | `EVAL_ROOT` | `EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"` |
| 35 | `RISK_ROOT` | `RS = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/computed"` |

### `CHS_MESA/cmr_feature_cox.py`

| line | root | current value |
|---|---|---|
| 27 | `EVAL_ROOT` | `EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"` |
| 28 | `RISK_ROOT` | `CP = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/CHARGE-PREVENT"` |
| 29 | `RISK_ROOT` | `HR = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/csv_HR"` |

### `CHS_MESA/cmr_feature_cox_external.py`

| line | root | current value |
|---|---|---|
| 31 | `EVAL_ROOT` | `EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"` |
| 32 | `RISK_ROOT` | `IDD = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/csv_train_valid_test_individual_id_disease"` |
| 33 | `RISK_ROOT` | `CP = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/CHARGE-PREVENT"` |
| 34 | `MESA_ECG_ROOT` | `MD = "/gpfs/projects/trend/bojun/CHS_MESA/data_train_valid_test_individual_MESA"` |
| 191 | `MESA_TABLES` | `DIS = "/gpfs/projects/trend/bojun/CHS_MESA/MESA/MESA_disease.csv"` |
| 206 | `MESA_TABLES` | `meas = pd.read_csv("/gpfs/projects/trend/bojun/CHS_MESA/MESA/MESA_CMR_features.csv")` |

### `CHS_MESA/deepecg_compare.py`

| line | root | current value |
|---|---|---|
| 25 | `EVAL_ROOT` | `EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"` |

### `CHS_MESA/ensemble_compare.py`

| line | root | current value |
|---|---|---|
| 24 | `EVAL_ROOT` | `EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"` |

### `CHS_MESA/fig4_compare.py`

| line | root | current value |
|---|---|---|
| 36 | `RISK_ROOT` | `CP = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/CHARGE-PREVENT"` |
| 38 | `RISK_ROOT` | `IDD = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/csv_train_valid_test_individual_id_disease"` |
| 39 | `RISK_ROOT` | `RS = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/computed"` |
| 40 | `EVAL_ROOT` | `EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"` |
| 195 | `MESA_TABLES` | `DIS = "/gpfs/projects/trend/bojun/CHS_MESA/MESA/MESA_disease.csv"` |

### `CHS_MESA/fig4_tertile_cox.py`

| line | root | current value |
|---|---|---|
| 23 | `RISK_ROOT` | `RS = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/computed"` |
| 24 | `RISK_ROOT` | `IDD = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/csv_train_valid_test_individual_id_disease"` |
| 25 | `MESA_ECG_ROOT` | `MESA = "/gpfs/projects/trend/bojun/CHS_MESA/data_train_valid_test_individual_MESA/test.csv"` |

### `CHS_MESA/fig5_fewshot.py`

| line | root | current value |
|---|---|---|
| 29 | `EVAL_ROOT` | `EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"` |

### `CHS_MESA/fig_cox_zeroshot.py`

| line | root | current value |
|---|---|---|
| 17 | `EVAL_ROOT` | `ZS = "/gpfs/projects/trend/bojun/multimodal_rep/eval/cox_zeroshot"` |
| 139 | `EVAL_ROOT` | `ap.add_argument("--outdir", default="/gpfs/projects/trend/bojun/multimodal_rep/eval/figures")` |

### `CHS_MESA/mesa_cmr_corr.py`

| line | root | current value |
|---|---|---|
| 19 | `EVAL_ROOT` | `EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"` |
| 20 | `MESA_TABLES` | `MEAS = "/gpfs/projects/trend/bojun/CHS_MESA/MESA/MESA_CMR_features.csv"` |

### `CHS_MESA/refit_clinical_compare.py`

| line | root | current value |
|---|---|---|
| 34 | `RISK_ROOT` | `CP = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/CHARGE-PREVENT"` |
| 35 | `EVAL_ROOT` | `EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"` |

### `CHS_MESA/run_cox_zeroshot.sh`

| line | root | current value |
|---|---|---|
| 9 | `EVAL_ROOT` | `#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/cox_zs_%A_%a.out` |
| 10 | `EVAL_ROOT` | `#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/cox_zs_%A_%a.err` |
| 13 | `(obsolete: old script dir)` | `source /gpfs/projects/trend/bojun/CHS_MESA/scripts/config.sh` |
| 15 | `EVAL_ROOT` | `COX=/gpfs/projects/trend/bojun/multimodal_rep/eval/cox` |
| 16 | `EVAL_ROOT` | `ZS=/gpfs/projects/trend/bojun/multimodal_rep/eval/cox_zeroshot` |
| 17 | `(obsolete: now baselines/)` | `DE=/gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL` |

### `UKBB/contrastive/stage1_CL_cinema.py`

| line | root | current value |
|---|---|---|
| 12 | `UKB_MRI_DIR` | `--mri_dir /gpfs/projects/trend/data/UKBB/MRI/cropped_new \` |
| 13 | `UKB_ECG_ROOT` | `--ecg_dir /gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual/stage1/ecg_tsv \` |
| 15 | `CKPT_ROOT` | `--mae_ckpt /gpfs/projects/trend/bojun/mri/CineMA/checkpoints/cinema_224_100ep_mg/cinema_best.pth \` |
| 16 | `ECG_CKPT` | `--ecg_ckpt /gpfs/projects/trend/bojun/multimodal/mimic_iv_ecg_physionet_pretrained.pt \` |
| 17 | `CKPT_ROOT` | `--out_dir  /gpfs/projects/trend/bojun/mri/CineMA/checkpoints/stage1_cinema` |

### `UKBB/contrastive/submit_stage1_cinema_base_m75.sh`

| line | root | current value |
|---|---|---|
| 9 | `LOG_DIR` | `#SBATCH --output=/gpfs/projects/trend/bojun/mri/CineMA/logs/stage1_cinema_base_m75_%x_%j.out` |
| 10 | `LOG_DIR` | `#SBATCH --error=/gpfs/projects/trend/bojun/mri/CineMA/logs/stage1_cinema_base_m75_%x_%j.err` |
| 25 | `CONDA_ENV` | `conda activate /gpfs/projects/trend/bojun/mri_env` |
| 26 | `CONDA_ENV` | `export LD_LIBRARY_PATH=/gpfs/projects/trend/bojun/mri_env/lib:$LD_LIBRARY_PATH` |
| 34 | `UKB_MRI_DIR` | `MRI_DIR=/gpfs/projects/trend/data/UKBB/MRI/cropped_new` |
| 35 | `UKB_ECG_ROOT` | `ECG_DIR=/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual/stage1/ecg_tsv` |
| 36 | `UKB_ECG_ROOT` | `SPLITS=/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual/stage1` |
| 37 | `CKPT_ROOT` | `MAE_CKPT=/gpfs/projects/trend/bojun/mri/CineMA/checkpoints/cinema_base_conv_224_m75_mg/cinema_bes...` |
| 38 | `ECG_CKPT` | `ECG_CKPT=/gpfs/projects/trend/bojun/multimodal/mimic_iv_ecg_physionet_pretrained.pt` |
| 39 | `CKPT_ROOT` | `SAVE=/gpfs/projects/trend/bojun/mri/CineMA/checkpoints/stage1_cinema_base_m75_ecg${ECG_MODE}` |
| 43 | `(obsolete: old code dir)` | `cd /gpfs/projects/trend/bojun/multimodal` |

### `UKBB/downstream/downstream_ecgmri_cinema.py`

| line | root | current value |
|---|---|---|
| 18 | `CKPT_ROOT` | `--cl_ckpt  /gpfs/projects/trend/bojun/mri/CineMA/checkpoints/stage1_cinema/stage1_cinema_best.pth \` |
| 19 | `ECG_CKPT` | `--ecg_ckpt /gpfs/projects/trend/bojun/multimodal/mimic_iv_ecg_physionet_pretrained.pt \` |
| 20 | `UKB_MRI_DIR` | `--mri_dir  /gpfs/projects/trend/data/UKBB/MRI/cropped_new \` |
| 21 | `UKB_ECG_ROOT` | `--ecg_dir  /gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual/stage1/ecg_tsv \` |
| 23 | `CKPT_ROOT` | `--out_dir  /gpfs/projects/trend/bojun/mri/CineMA/checkpoints/downstream_ecg_mri` |

### `UKBB/downstream/run_cox.sh`

| line | root | current value |
|---|---|---|
| 9 | `EVAL_ROOT` | `#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/cox_ft_%A_%a.out` |
| 10 | `EVAL_ROOT` | `#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/cox_ft_%A_%a.err` |
| 12 | `(obsolete: old script dir)` | `source /gpfs/projects/trend/bojun/CHS_MESA/scripts/config.sh` |
| 13 | `UKB_ECG_ROOT` | `UKB=/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual` |
| 21 | `EVAL_ROOT` | `--save_dir "/gpfs/projects/trend/bojun/multimodal_rep/eval/cox/m75_seed1/$O" \` |

### `UKBB/downstream/run_cox_ecgfm.sh`

| line | root | current value |
|---|---|---|
| 9 | `EVAL_ROOT` | `#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/cox_ecgfm_%A_%a.out` |
| 10 | `EVAL_ROOT` | `#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/cox_ecgfm_%A_%a.err` |
| 13 | `(obsolete: old script dir)` | `source /gpfs/projects/trend/bojun/CHS_MESA/scripts/config.sh` |
| 14 | `UKB_ECG_ROOT` | `UKB=/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual` |
| 16 | `EVAL_ROOT` | `D=/gpfs/projects/trend/bojun/multimodal_rep/eval/cox/ecgfm/$O` |

### `UKBB/downstream/submit_downstream_cinema.sh`

| line | root | current value |
|---|---|---|
| 8 | `LOG_DIR` | `#SBATCH --output=/gpfs/projects/trend/bojun/mri/CineMA/logs/downstream_cinema_%x_%j.out` |
| 9 | `LOG_DIR` | `#SBATCH --error=/gpfs/projects/trend/bojun/mri/CineMA/logs/downstream_cinema_%x_%j.err` |
| 37 | `CONDA_ENV` | `conda activate /gpfs/projects/trend/bojun/mri_env` |
| 38 | `CONDA_ENV` | `export LD_LIBRARY_PATH=/gpfs/projects/trend/bojun/mri_env/lib:$LD_LIBRARY_PATH` |
| 43 | `CKPT_ROOT` | `CKPT_ROOT=/gpfs/projects/trend/bojun/mri/CineMA/checkpoints` |
| 45 | `ECG_CKPT` | `ECG_CKPT=/gpfs/projects/trend/bojun/multimodal/mimic_iv_ecg_physionet_pretrained.pt` |
| 46 | `UKB_MRI_DIR` | `MRI_DIR=/gpfs/projects/trend/data/UKBB/MRI/cropped_new` |
| 47 | `UKB_ECG_ROOT` | `ECG_DIR=/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual/stage1/ecg_tsv` |
| 48 | `UKB_PHENO_DIR` | `PHENO_DIR=/gpfs/projects/trend/data/UKBB/Phenotype` |
| 55 | `(obsolete: old code dir)` | `cd /gpfs/projects/trend/bojun/multimodal` |

### `UKBB/figures/a4_ukb_tertile_cox.py`

| line | root | current value |
|---|---|---|
| 17 | `EVAL_ROOT` | `EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"` |
| 19 | `RISK_ROOT` | `HRD = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/csv_HR"` |
| 20 | `RISK_ROOT` | `RS = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/computed"` |

### `UKBB/figures/a5_km_curves.py`

| line | root | current value |
|---|---|---|
| 16 | `EVAL_ROOT` | `EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"` |
| 18 | `RISK_ROOT` | `HRD = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/csv_HR"` |
| 19 | `RISK_ROOT` | `IDD = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/csv_train_valid_test_individual_id_disease"` |
| 20 | `RISK_ROOT` | `RS = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/computed"` |

### `UKBB/figures/cmr_eval.py`

| line | root | current value |
|---|---|---|
| 22 | `EVAL_ROOT` | `EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"` |
| 23 | `RISK_ROOT` | `IDD = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/csv_train_valid_test_individual_id_disease"` |

### `UKBB/figures/fig_cox_compare.py`

| line | root | current value |
|---|---|---|
| 17 | `EVAL_ROOT` | `COX = "/gpfs/projects/trend/bojun/multimodal_rep/eval/cox"` |
| 144 | `EVAL_ROOT` | `ap.add_argument("--outdir", default="/gpfs/projects/trend/bojun/multimodal_rep/eval/figures")` |

### `UKBB/figures/make_figures.py`

| line | root | current value |
|---|---|---|
| 33 | `EVAL_ROOT` | `EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"` |
| 34 | `RISK_ROOT` | `RS = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/computed"` |
| 43 | `EVAL_ROOT` | `return EV if seed == 1 else f"/gpfs/projects/trend/bojun/multimodal_rep/eval_{seed}"` |

### `UKBB/figures/summary_report_0719.py`

| line | root | current value |
|---|---|---|
| 22 | `EVAL_ROOT` | `EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"` |
| 23 | `FEWSHOT_RESULTS` | `EXT = "/gpfs/projects/trend/bojun/CHS_MESA/result_finetune_from_pretrain_fewshot"` |

### `UKBB/pretrain_mri/submit_mae_multigpu_base_m75.sh`

| line | root | current value |
|---|---|---|
| 9 | `LOG_DIR` | `#SBATCH --output=/gpfs/projects/trend/bojun/mri/CineMA/logs/cinema_base_m75_%j.out` |
| 10 | `LOG_DIR` | `#SBATCH --error=/gpfs/projects/trend/bojun/mri/CineMA/logs/cinema_base_m75_%j.err` |
| 19 | `CONDA_ENV` | `conda activate /gpfs/projects/trend/bojun/mri_env` |
| 24 | `UKB_MRI_DIR` | `DATA=/gpfs/projects/trend/data/UKBB/MRI/cropped_new` |
| 25 | `CKPT_ROOT` | `CKPT=/gpfs/projects/trend/bojun/mri/CineMA/checkpoints/cinema_base_conv_224_m75_mg` |
| 26 | `LOG_DIR` | `LOGS=/gpfs/projects/trend/bojun/mri/CineMA/logs` |
| 27 | `MRI_SPLITS` | `TR=/gpfs/projects/trend/bojun/mri/splits/train` |
| 28 | `MRI_SPLITS` | `VA=/gpfs/projects/trend/bojun/mri/splits/val` |
| 32 | `(obsolete: old code dir)` | `cd /gpfs/projects/trend/bojun/mri/CineMA/pretrain` |

### `baselines/deepecg/deepecg_run.py`

| line | root | current value |
|---|---|---|
| 39 | `EVAL_ROOT` | `EVAL = "/gpfs/projects/trend/bojun/multimodal_rep/eval"` |
| 40 | `UKB_ECG_ROOT` | `UKB = "/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual"` |
| 41 | `CHS_ECG_ROOT` | `CHS = "/gpfs/projects/trend/bojun/CHS_MESA/data_train_valid_test_individual_CHS"` |
| 42 | `MESA_ECG_ROOT` | `MESA = "/gpfs/projects/trend/bojun/CHS_MESA/data_train_valid_test_individual_MESA"` |

### `baselines/deepecg/deepsl_finetune.py`

| line | root | current value |
|---|---|---|
| 27 | `UKB_ECG_ROOT` | `UKB = "/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual"` |

### `baselines/deepecg/deepssl_finetune.py`

| line | root | current value |
|---|---|---|
| 26 | `UKB_ECG_ROOT` | `UKB = "/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual"` |

### `baselines/deepecg/inspect_ssl.py`

| line | root | current value |
|---|---|---|
| 16 | `UKB_ECG_ROOT` | `UKB = "/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual"` |

### `baselines/deepecg/inspect_ssl.sh`

| line | root | current value |
|---|---|---|
| 8 | `EVAL_ROOT` | `#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/ssl_inspect_%j.out` |
| 9 | `EVAL_ROOT` | `#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/ssl_inspect_%j.err` |
| 11 | `CONDA_ENV` | `ENV=/gpfs/projects/trend/bojun/mri_env` |
| 13 | `(obsolete: now baselines/)` | `cd /gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL/deepecg` |

### `baselines/deepecg/probe_scaling.py`

| line | root | current value |
|---|---|---|
| 17 | `UKB_ECG_ROOT` | `UKB = "/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual"` |

### `baselines/deepecg/probe_scaling.sh`

| line | root | current value |
|---|---|---|
| 8 | `EVAL_ROOT` | `#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepecg_probe_%j.out` |
| 9 | `EVAL_ROOT` | `#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepecg_probe_%j.err` |
| 10 | `CONDA_ENV` | `ENV=/gpfs/projects/trend/bojun/mri_env` |
| 12 | `(obsolete: now baselines/)` | `cd /gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL/deepecg` |

### `baselines/deepecg/probe_wcr.py`

| line | root | current value |
|---|---|---|
| 15 | `UKB_ECG_ROOT` | `UKB = "/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual"` |

### `baselines/deepecg/probe_wcr.sh`

| line | root | current value |
|---|---|---|
| 8 | `EVAL_ROOT` | `#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepecg_wcr_%j.out` |
| 9 | `EVAL_ROOT` | `#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepecg_wcr_%j.err` |
| 10 | `CONDA_ENV` | `ENV=/gpfs/projects/trend/bojun/mri_env` |
| 12 | `(obsolete: now baselines/)` | `cd /gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL/deepecg` |

### `baselines/deepecg/run_cox_deepecg.sh`

| line | root | current value |
|---|---|---|
| 9 | `EVAL_ROOT` | `#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/cox_deepecg_%A_%a.out` |
| 10 | `EVAL_ROOT` | `#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/cox_deepecg_%A_%a.err` |
| 14 | `(obsolete: now baselines/)` | `D=/gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL/deepecg` |
| 15 | `CONDA_ENV` | `ENV=/gpfs/projects/trend/bojun/mri_env; export PATH="$ENV/bin:$PATH"` |
| 17 | `UKB_ECG_ROOT` | `UKB=/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual` |
| 20 | `EVAL_ROOT` | `CKD=/gpfs/projects/trend/bojun/multimodal_rep/eval/cox/${M}_ft/$O` |

### `baselines/deepecg/run_deepecg.sh`

| line | root | current value |
|---|---|---|
| 8 | `EVAL_ROOT` | `#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepecg_zs_%j.out` |
| 9 | `EVAL_ROOT` | `#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepecg_zs_%j.err` |
| 13 | `CONDA_ENV` | `ENV=/gpfs/projects/trend/bojun/mri_env` |
| 15 | `(obsolete: now baselines/)` | `cd /gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL/deepecg` |

### `baselines/deepecg/run_deepecg_fewshot.sh`

| line | root | current value |
|---|---|---|
| 9 | `EVAL_ROOT` | `#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepecg_fewshot_%A_%a.out` |
| 10 | `EVAL_ROOT` | `#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepecg_fewshot_%A_%a.err` |
| 21 | `(obsolete: now baselines/)` | `D=/gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL/deepecg` |
| 22 | `CONDA_ENV` | `ENV=/gpfs/projects/trend/bojun/mri_env; export PATH="$ENV/bin:$PATH"` |
| 24 | `EVAL_ROOT` | `EVAL=/gpfs/projects/trend/bojun/multimodal_rep/eval` |
| 25 | `CHS_ECG_ROOT` | `CHS=/gpfs/projects/trend/bojun/CHS_MESA/data_train_valid_test_individual_CHS` |
| 26 | `MESA_ECG_ROOT` | `MESA=/gpfs/projects/trend/bojun/CHS_MESA/data_train_valid_test_individual_MESA` |

### `baselines/deepecg/run_deepecg_ssl.sh`

| line | root | current value |
|---|---|---|
| 8 | `EVAL_ROOT` | `#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepecg_ssl_%j.out` |
| 9 | `EVAL_ROOT` | `#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepecg_ssl_%j.err` |
| 14 | `CONDA_ENV` | `ENV=/gpfs/projects/trend/bojun/mri_env` |
| 16 | `(obsolete: now baselines/)` | `cd /gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL/deepecg` |

### `baselines/deepecg/run_deepsl_ft.sh`

| line | root | current value |
|---|---|---|
| 9 | `EVAL_ROOT` | `#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepsl_ft_%A_%a.out` |
| 10 | `EVAL_ROOT` | `#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepsl_ft_%A_%a.err` |
| 14 | `(obsolete: now baselines/)` | `D=/gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL/deepecg` |
| 15 | `CONDA_ENV` | `ENV=/gpfs/projects/trend/bojun/mri_env; export PATH="$ENV/bin:$PATH"` |
| 17 | `EVAL_ROOT` | `EVAL=/gpfs/projects/trend/bojun/multimodal_rep/eval` |
| 18 | `UKB_ECG_ROOT` | `UKB=/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual` |
| 19 | `CHS_ECG_ROOT` | `CHS=/gpfs/projects/trend/bojun/CHS_MESA/data_train_valid_test_individual_CHS` |
| 20 | `MESA_ECG_ROOT` | `MESA=/gpfs/projects/trend/bojun/CHS_MESA/data_train_valid_test_individual_MESA` |

### `baselines/deepecg/run_deepssl_ft.sh`

| line | root | current value |
|---|---|---|
| 9 | `EVAL_ROOT` | `#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepssl_ft_%A_%a.out` |
| 10 | `EVAL_ROOT` | `#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepssl_ft_%A_%a.err` |
| 14 | `(obsolete: now baselines/)` | `D=/gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL/deepecg` |
| 15 | `CONDA_ENV` | `ENV=/gpfs/projects/trend/bojun/mri_env; export PATH="$ENV/bin:$PATH"` |
| 17 | `EVAL_ROOT` | `EVAL=/gpfs/projects/trend/bojun/multimodal_rep/eval` |
| 18 | `UKB_ECG_ROOT` | `UKB=/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual` |
| 19 | `CHS_ECG_ROOT` | `CHS=/gpfs/projects/trend/bojun/CHS_MESA/data_train_valid_test_individual_CHS` |
| 20 | `MESA_ECG_ROOT` | `MESA=/gpfs/projects/trend/bojun/CHS_MESA/data_train_valid_test_individual_MESA` |

### `baselines/ecgfounder/cox_ecgfounder.py`

| line | root | current value |
|---|---|---|
| 13 | `(obsolete: now baselines/)` | `"/gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL/ECGFounder")` |

### `common/risk/risk_fuse.py`

| line | root | current value |
|---|---|---|
| 17 | `EVAL_ROOT` | `EVAL = "/gpfs/projects/trend/bojun/multimodal_rep/eval"   # overridden by --eval_root` |
| 18 | `RISK_ROOT` | `RS = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/computed"` |
| 57 | `EVAL_ROOT` | `EFPAIR = "/gpfs/projects/trend/bojun/multimodal_rep/eval/ecgfm_ukb_paired/seed1"` |

### `common/risk/risk_score.py`

| line | root | current value |
|---|---|---|
| 15 | `RISK_ROOT` | `CP = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/CHARGE-PREVENT"` |
| 16 | `RISK_ROOT` | `OUTDIR = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/computed"` |
| 98 | `EVAL_ROOT` | `base = ("/gpfs/projects/trend/bojun/multimodal_rep/eval/zeroshot"` |

### `common/stats/bootstrap_ci.py`

| line | root | current value |
|---|---|---|
| 13 | `EXT_RESULTS` | `OUT = "/gpfs/projects/trend/bojun/CHS_MESA/results"` |
| 81 | `?` | `help="input root to read result.csv from (e.g. /gpfs/.../multimodal_rep/eval)")` |

### `common/stats/bootstrap_compare.py`

| line | root | current value |
|---|---|---|
| 16 | `EXT_RESULTS` | `OUT = "/gpfs/projects/trend/bojun/CHS_MESA/results"` |
| 17 | `FEWSHOT_RESULTS` | `EXT = "/gpfs/projects/trend/bojun/CHS_MESA/result_finetune_from_pretrain_fewshot"` |
| 82 | `?` | `help="root for our CL result.csv (e.g. /gpfs/.../multimodal_rep/eval)")` |

### `common/stats/bootstrap_zeroshot.py`

| line | root | current value |
|---|---|---|
| 11 | `EVAL_ROOT` | `EV = "/gpfs/projects/trend/bojun/multimodal_rep/eval"` |
| 12 | `EXT_RESULTS` | `RES = "/gpfs/projects/trend/bojun/CHS_MESA/results"` |
| 14 | `EVAL_ROOT` | `EFROOT = "/gpfs/projects/trend/bojun/multimodal_rep/eval/ecgfm_zeroshot_provided"` |

### `common/train_eval/build_surv_labels.py`

| line | root | current value |
|---|---|---|
| 20 | `UKB_ECG_ROOT` | `UKB = "/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual"` |
| 22 | `RISK_ROOT` | `SURV = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/csv_HR/UKBB_CMR_AF-HF-IS_WithDem_Analytic....` |

### `common/train_eval/build_surv_labels_external.py`

| line | root | current value |
|---|---|---|
| 18 | `CHS_MESA_ROOT` | `BASE = "/gpfs/projects/trend/bojun/CHS_MESA"` |
