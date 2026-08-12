# Configuring paths

Nothing in this repository hard-codes a machine-specific location. Every data, checkpoint, and output
location is read from an **environment variable**, resolved through a single helper
([`common/paths.py`](../common/paths.py)) that Python and bash both use.

## Quick start

```bash
cp env/paths.example.sh env/paths.local.sh
$EDITOR env/paths.local.sh          # fill in your paths
python common/paths.py              # print what resolved, and what is still unset
```

`env/paths.local.sh` is **gitignored** — it is the one file that holds absolute paths for your
machine, and it never gets committed.

## How resolution works

For each variable, in order:

1. the **process environment** — `export EVAL_ROOT=/my/eval` wins over everything;
2. **`env/paths.local.sh`**, if it exists.

If neither supplies a value, `P()` raises an error naming the missing variable and pointing at this
document, rather than failing later with a confusing "file not found".

**Python:**

```python
from paths import P

EV  = P("EVAL_ROOT")                                                 # the root itself
IDD = P("RISK_ROOT", "csv_train_valid_test_individual_id_disease")   # a path under it
```

Each script bootstraps this by walking up to the repository root and adding `common/` to `sys.path`,
so scripts run correctly from any working directory.

**Bash:** the `.sh` run scripts source the same file, so both languages read identical values:

```bash
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_repo" != "/" ] && [ ! -d "$_repo/common" ]; do _repo="$(dirname "$_repo")"; done
source "$_repo/env/paths.local.sh"
```

## The variables

| variable | what it holds | who writes it |
|---|---|---|
| `UKB_MRI_DIR` | UK Biobank cardiac MRI, per subject: `vst_2ch.npy`, `vst_4ch.npy`, `vst_sa.npy` | your preprocessing (see README → Data) |
| `UKB_PHENO_DIR` | UKB phenotype/outcome CSVs: `MRI_train/`, `MRI_valid_new/`, `MRI_test_new/`, one `<outcome>.csv` each | your cohort definition |
| `UKB_ECG_ROOT` | UKB ECG: `ECG_manifest*/` (`.tsv` manifests), `ECG_label*/`, `ECG_label_surv/<outcome>/`, `stage1/` (alignment splits + `ecg_tsv/`) | your preprocessing |
| `MRI_SPLITS` | subject-ID lists for MAE pretraining: `train/` and `val/` | your split definition |
| `CHS_ECG_ROOT` / `MESA_ECG_ROOT` | external cohort ECG, same layout as `UKB_ECG_ROOT` | your preprocessing |
| `CHS_MESA_ROOT` | parent of the CHS/MESA cohort directories | — |
| `RISK_ROOT` | risk-score inputs and outputs: `CHARGE-PREVENT/` (raw tables `charge_prevent_<cohort>.csv`), `computed/` (imputed scores), `csv_HR/`, `csv_train_valid_test_individual_id_disease/` | `common/risk/risk_score.py` writes `computed/` |
| `MESA_TABLES` | MESA measured-CMR and disease tables (`MESA_CMR_features.csv`, `MESA_disease.csv`) | MESA data release |
| `FEWSHOT_RESULTS` / `EXT_RESULTS` | external few-shot and aggregate result tables | `CHS_MESA/` scripts |
| `CKPT_ROOT` | all model checkpoints (MAE, stage-1, downstream) | training scripts |
| `LOG_DIR` | training log directory | training scripts |
| `ECG_CKPT` | the ECG-FM backbone **file** (see [`weights/README.md`](../weights/README.md)) | upstream release |
| `EVAL_ROOT` | prediction/eval outputs consumed by every figure script: `<run>/result.csv`, `cox/`, `cox_zeroshot/`, `figures/` | eval scripts |
| `ECGFOUNDER_REPO` | checkout of the ECGFounder repository — `baselines/` only | third party |

`env/paths.local.sh` additionally defines a few bash-only helpers used by the survival run scripts:

| name | purpose |
|---|---|
| `SCRIPTS` | directory holding the generic train/eval engines (`common/train_eval`); the run scripts `cd` there and call `cox_finetune.py` / `cox_test.py` by bare filename |
| `ECGFM`, `CHS`, `MESA` | aliases for `ECG_CKPT`, `CHS_ECG_ROOT`, `MESA_ECG_ROOT`, kept so the run scripts read naturally |
| `pick_ckpt()` | returns the best checkpoint in a Cox output directory — `cox_finetune.py` writes `epoch_<N>.pth` only when validation C-index improves, so the highest-numbered file is the best one |

These replace a cluster-local `config.sh` that three survival scripts used to `source` but which was
never part of the repository.

## SLURM log paths

`#SBATCH --output=` / `--error=` directives are read by SLURM **before** the shell runs, so a shell
variable there would never expand. They are therefore written relative to the submit directory:

```
#SBATCH --output=logs/cox_ft_%A_%a.out
```

Create `logs/` in the directory you submit from, or override at submission time with
`sbatch -o /somewhere/else/%j.out`.

## Checking your configuration

```bash
$ python common/paths.py
  UKB_MRI_DIR      /data/ukbb/mri/cropped_new
  EVAL_ROOT        <UNSET>   -- prediction/eval outputs consumed by the figure scripts
  ...
local defaults file: /path/to/CARDIAC-FM/env/paths.local.sh (found)
15/16 configured
```

Not every variable is needed for every task — `ECGFOUNDER_REPO` matters only for `baselines/`,
`MRI_SPLITS` only for MAE pretraining. Configure what your workflow touches.
