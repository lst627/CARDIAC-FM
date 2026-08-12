#!/bin/bash
# PATH CONFIGURATION TEMPLATE -- copy to env/paths.local.sh and fill in.
#
# Every path below must point at your own data. Nothing in the repository hard-codes a location;
# both Python (common/paths.py) and the .sh run scripts read this file, so there is a single
# source of truth.
#
#   bash:   source env/paths.local.sh
#   python: automatic -- common/paths.py reads it when a variable is not already in the environment
#
# See docs/PATHS.md for what each variable means.

# --- cohort data -------------------------------------------------------------
export UKB_MRI_DIR=/path/to/UKBB/MRI/cropped_new
export UKB_PHENO_DIR=/path/to/UKBB/Phenotype
export UKB_ECG_ROOT=/path/to/UKB/ecg_root
export MRI_SPLITS=/path/to/mri/splits

export CHS_MESA_ROOT=/path/to/CHS_MESA
export CHS_ECG_ROOT=${CHS_MESA_ROOT}/data_train_valid_test_individual_CHS
export MESA_ECG_ROOT=${CHS_MESA_ROOT}/data_train_valid_test_individual_MESA
export MESA_TABLES=${CHS_MESA_ROOT}/MESA

# --- risk scores -------------------------------------------------------------
export RISK_ROOT=${CHS_MESA_ROOT}/risk_score

# --- results -----------------------------------------------------------------
export EVAL_ROOT=/path/to/eval
export FEWSHOT_RESULTS=${CHS_MESA_ROOT}/result_finetune_from_pretrain_fewshot
export EXT_RESULTS=${CHS_MESA_ROOT}/results

# --- checkpoints / logs ------------------------------------------------------
export CKPT_ROOT=/path/to/checkpoints
export LOG_DIR=/path/to/logs
export ECG_CKPT=/path/to/weights/ecgfm_mimic_iv_physionet.pt

# --- baselines (optional; only needed for baselines/) ------------------------
export ECGFOUNDER_REPO=/path/to/ECGFounder

# --- environment (used by the SLURM .sh scripts) -----------------------------
export CONDA_ENV=/path/to/conda/env

# --- where the generic train/eval engines live (was $SCRIPTS in the old config.sh) ---
export SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/train_eval"

# Legacy aliases kept so the survival run scripts work unchanged
export ECGFM="$ECG_CKPT"
export CHS="$CHS_ECG_ROOT"
export MESA="$MESA_ECG_ROOT"

# Best checkpoint in a Cox fine-tune output dir: cox_finetune.py writes epoch_<N>.pth only when
# validation C-index improves, so the highest-numbered file is the best one.
pick_ckpt() { ls -1v "$1"/epoch_*.pth 2>/dev/null | tail -n1; }
