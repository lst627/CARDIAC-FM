#!/bin/bash
#SBATCH --job-name=cox_zs
#SBATCH --partition=gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-19
#SBATCH --output=logs/cox_zs_%A_%a.out
#SBATCH --error=logs/cox_zs_%A_%a.err
# Zero-shot survival: UKB Cox-fine-tuned models applied to CHS/MESA (no refit) -> C-index.
# Array over (model x cohort x outcome): idx = model*4 + cohort*2 + outcome.

# --- repo path configuration (see docs/PATHS.md) ---
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_repo" != "/" ] && [ ! -d "$_repo/common" ]; do _repo="$(dirname "$_repo")"; done
source "$_repo/env/paths.local.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export W=8
COX=${EVAL_ROOT}/cox
ZS=${EVAL_ROOT}/cox_zeroshot
DE="$_repo/baselines"
MODELS=(m75_seed1 ecgfm deepsl_ft deepssl_ft ecgfounder); COHORTS=(CHS MESA); OUTS=(af hf)
i=$SLURM_ARRAY_TASK_ID
M=$((i/4)); rem=$((i%4)); C=$((rem/2)); O=$((rem%2))
model=${MODELS[$M]}; cohd=${COHORTS[$C]}; outc=${OUTS[$O]}
if [ "$cohd" = CHS ]; then CD=$CHS; else CD=$MESA; fi
MAN=$CD/ECG_manifest; LAB=$CD/ECG_label_surv/$outc; SAVE=$ZS/$cohd/$model/$outc
echo "===== $model / $cohd / $outc ====="
case $model in
  m75_seed1)  CK=$(pick_ckpt $COX/m75_seed1/$outc)
    (cd "$SCRIPTS" && python cox_test.py --outcome $outc --ckpt "$CK" --ecgfm_ckpt "$ECGFM" \
       --ecg_tsv_dir "$MAN" --label_dir "$LAB" --split test --save_dir "$SAVE");;
  ecgfm)      CK=$(pick_ckpt $COX/ecgfm/$outc)
    (cd "$SCRIPTS" && python cox_test.py --outcome $outc --model_name ECGFM --ckpt "$CK" --ecgfm_ckpt "$ECGFM" \
       --ecg_tsv_dir "$MAN" --label_dir "$LAB" --split test --save_dir "$SAVE");;
  deepsl_ft)  (cd "$DE/deepecg" && python cox_deepecg.py test --model deepsl --outcome $outc --split test \
       --ckpt_in "$COX/deepsl_ft/$outc/best.pth" --ecg_tsv_dir "$MAN" --label_dir "$LAB" --save_dir "$SAVE");;
  deepssl_ft) (cd "$DE/deepecg" && python cox_deepecg.py test --model deepssl --outcome $outc --split test \
       --ckpt_in "$COX/deepssl_ft/$outc/best.pth" --ecg_tsv_dir "$MAN" --label_dir "$LAB" --save_dir "$SAVE");;
  ecgfounder) (cd "$DE/ecgfounder" && python cox_ecgfounder.py test --outcome $outc --split test \
       --ckpt_in "$COX/ecgfounder/$outc/best.pth" --ecg_tsv_dir "$MAN" --label_dir "$LAB" --save_dir "$SAVE");;
esac
echo "done $model $cohd $outc"
