#!/bin/bash
#SBATCH --job-name=deepsl_ft
#SBATCH --partition=gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/deepsl_ft_%A_%a.out
#SBATCH --error=logs/deepsl_ft_%A_%a.err
# Fine-tune DeepECG-SL (EfficientNetV2 77-class TorchScript) on UKB af5/hf5, matched to ECGFounder;
# then zero-shot external. Arm: deepsl_ft. Array over outcome.

# --- repo path configuration (see docs/PATHS.md) ---
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_repo" != "/" ] && [ ! -d "$_repo/common" ]; do _repo="$(dirname "$_repo")"; done
source "$_repo/env/paths.local.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -u
D="$HERE"
ENV=${CONDA_ENV}; export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}" W=8
EVAL=${EVAL_ROOT}
UKB=${UKB_ECG_ROOT}
CHS=${CHS_ECG_ROOT}
MESA=${MESA_ECG_ROOT}
OUTS=(af5 hf5); O=${OUTS[$SLURM_ARRAY_TASK_ID]}
CK=$EVAL/ukb_ft/deepsl_ft/$O/best.pth
cd $D
echo "########## [$O] TRAIN ##########"
python deepsl_finetune.py train --outcome $O --ckpt_out $CK || exit 1
echo "########## [$O] INFERENCE ##########"
python deepsl_finetune.py test --outcome $O --split test --ckpt_in $CK \
  --ecg_tsv_dir $UKB/ECG_manifest_moretest --label_dir $UKB/ECG_label/$O --save_dir $EVAL/ukb_test/deepsl_ft/$O
python deepsl_finetune.py test --outcome $O --split train --ckpt_in $CK \
  --ecg_tsv_dir $UKB/ECG_manifest_moretest --label_dir $UKB/ECG_label/$O --save_dir $EVAL/ukb_train_preds/deepsl_ft/$O
for COH in CHS MESA; do
  C=$([ $COH = CHS ] && echo $CHS || echo $MESA)
  python deepsl_finetune.py test --outcome $O --split test --ckpt_in $CK \
    --ecg_tsv_dir $C/ECG_manifest --label_dir $C/ECG_label/$O --save_dir $EVAL/zeroshot/$COH/deepsl_ft/$O
done
echo "########## [$O] DONE ##########"
