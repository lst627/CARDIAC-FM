#!/bin/bash
#SBATCH --job-name=cox_ft
#SBATCH --partition=gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/cox_ft_%A_%a.out
#SBATCH --error=logs/cox_ft_%A_%a.err
# Time-to-event (Cox/DeepSurv) fine-tune of CARDIAC-FM ECG on UKB survival. Array over outcome.

# --- repo path configuration (see docs/PATHS.md) ---
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_repo" != "/" ] && [ ! -d "$_repo/common" ]; do _repo="$(dirname "$_repo")"; done
source "$_repo/env/paths.local.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

UKB=${UKB_ECG_ROOT}
CL="$CKPT_ROOT/stage1_cinema_base_m75_ecgfull/stage1_cinema_best.pth"
OUTS=(af hf); O=${OUTS[$SLURM_ARRAY_TASK_ID]}
cd "$SCRIPTS"
echo "===== Cox fine-tune: outcome=$O ====="
python cox_finetune.py --seed 1 --epochs 20 --model_name CARDIACFM \
  --ecgfm_ckpt "$ECGFM" --cardiacfm_pretrained_ckpt "$CL" \
  --ecg_tsv_dir "$UKB/ECG_manifest_moretest" --label_dir "$UKB/ECG_label_surv/$O" \
  --save_dir "${EVAL_ROOT}/cox/m75_seed1/$O" \
  --batch_size 64
echo "done $O"
