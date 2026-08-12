#!/bin/bash
#SBATCH --job-name=deepecg_fewshot
#SBATCH --partition=gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepecg_fewshot_%A_%a.out
#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepecg_fewshot_%A_%a.err
# DeepECG-SL & DeepECG-SSL FEW-SHOT on CHS/MESA broad outcomes, matched to the our-model / ECGFounder
# few-shot protocol (seed 1, lr 5e-6, patience-3, 20-epoch cap; full fine-tune from the backbone).
# Fine-tune on each cohort's fraction (train_valid_5_5=10%, train_valid_10_10=20%), then test on the
# full test split. Writes checkpoints to fewshot_ft/ and predictions to fewshot/ (same tree as the
# other few-shot arms). Outcomes = exactly those our model has few-shot preds for, minus af5/hf5
# (those are the zero-shot Fig-3 outcomes, not broad few-shot).
#
# Array over (model x cohort):  0=deepsl/CHS  1=deepsl/MESA  2=deepssl/CHS  3=deepssl/MESA
# Resumable: any cell whose result.csv already exists is skipped.
set -u
D=/gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL/deepecg
ENV=/gpfs/projects/trend/bojun/mri_env; export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}" W=8
EVAL=/gpfs/projects/trend/bojun/multimodal_rep/eval
CHS=/gpfs/projects/trend/bojun/CHS_MESA/data_train_valid_test_individual_CHS
MESA=/gpfs/projects/trend/bojun/CHS_MESA/data_train_valid_test_individual_MESA
FRACTIONS="train_valid_5_5 train_valid_10_10"
mkdir -p "$EVAL/logs"

MODELS=(deepsl deepsl deepssl deepssl)
COHS=(CHS   MESA   CHS    MESA)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
COH=${COHS[$SLURM_ARRAY_TASK_ID]}
C=$([ "$COH" = CHS ] && echo "$CHS" || echo "$MESA")
SCRIPT=$([ "$M" = deepsl ] && echo deepsl_finetune.py || echo deepssl_finetune.py)
TAG=${M}_ft                                   # deepsl_ft / deepssl_ft (matches the main-benchmark arm names)
cd "$D"
echo "===== $M ($SCRIPT) few-shot on $COH  ->  arm '$TAG' ====="

for FR in $FRACTIONS; do
  # match our model's broad-outcome set exactly (drop af5/hf5, which are zero-shot)
  OUTS=$(ls "$EVAL/fewshot/$COH/$FR/m75_ecgfull/" 2>/dev/null | grep -vE '^(af5|hf5)$')
  for O in $OUTS; do
    [ -d "$C/ECG_label/$O" ] || { echo "  skip $O (cohort lacks label)"; continue; }
    FT="$EVAL/fewshot_ft/$COH/$FR/$TAG/$O"
    PRED="$EVAL/fewshot/$COH/$FR/$TAG/$O"
    if [ -s "$PRED/result.csv" ]; then echo "  [$COH $FR $O] already done, skip"; continue; fi
    echo ">>> [$COH $FR $O] TRAIN"
    python "$SCRIPT" train --outcome "$O" --seed 1 \
      --ecg_tsv_dir "$C/ECG_manifest/$FR" --label_dir "$C/ECG_label/$O" \
      --ckpt_out "$FT/best.pth" || { echo "  TRAIN FAIL $O"; continue; }
    echo ">>> [$COH $FR $O] TEST"
    python "$SCRIPT" test --outcome "$O" --split test --seed 1 \
      --ecg_tsv_dir "$C/ECG_manifest" --label_dir "$C/ECG_label/$O" \
      --ckpt_in "$FT/best.pth" --save_dir "$PRED" || echo "  TEST FAIL $O"
  done
done
echo "===== [$M/$COH] ALL DONE ====="
