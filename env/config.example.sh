#!/bin/bash
# Template for the cluster-local config.sh that the survival run scripts expect.
#
#   UKBB/downstream/run_cox.sh
#   UKBB/downstream/run_cox_ecgfm.sh
#   CHS_MESA/run_cox_zeroshot.sh
#
# each start with `source .../config.sh`. That file was local to the authors' cluster and is not
# part of this repository. Copy this template, fill in your paths, and repoint the `source` line:
#
#   cp env/config.example.sh env/config.sh      # config.sh is gitignored
#   $EDITOR env/config.sh
#
# See docs/PATHS.md for what each root means.

# --- where the generic train/eval engines live -------------------------------
# The run scripts `cd "$SCRIPTS"` and then call `python cox_finetune.py` / `cox_test.py`
# by bare filename, so this must be the directory that contains them.
SCRIPTS=/path/to/CARDIAC-FM/common/train_eval

# --- checkpoints -------------------------------------------------------------
# ECG-FM backbone (see weights/README.md)
ECGFM=/path/to/weights/ecgfm_mimic_iv_physionet.pt

# Root under which training runs write their checkpoints. run_cox.sh builds
#   $CKPT_ROOT/stage1_cinema_base_m75_ecgfull/stage1_cinema_best.pth
# from this, so the aligned stage-1 model must be reachable at that sub-path
# (or edit the CL= line in run_cox.sh to point straight at your file).
CKPT_ROOT=/path/to/checkpoints

# --- external cohort data ----------------------------------------------------
# Each must contain ECG_manifest/ (.tsv) and ECG_label_surv/<outcome>/
CHS=/path/to/data_train_valid_test_individual_CHS
MESA=/path/to/data_train_valid_test_individual_MESA

# --- helper ------------------------------------------------------------------
# Return the best checkpoint in a Cox fine-tune output directory.
# common/train_eval/cox_finetune.py writes "epoch_<N>.pth" ONLY when validation
# C-index improves, so the highest-numbered epoch file is the best one.
pick_ckpt() {
  ls -1v "$1"/epoch_*.pth 2>/dev/null | tail -n1
}

export SCRIPTS ECGFM CKPT_ROOT CHS MESA
