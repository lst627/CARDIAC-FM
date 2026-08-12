#!/bin/bash
#SBATCH --job-name=cox_ecgfm
#SBATCH --partition=gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/cox_ecgfm_%A_%a.out
#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/cox_ecgfm_%A_%a.err
# DeepSurv (Cox) fine-tune on the ECG-FM baseline (no multimodal pretraining), same protocol as our
# CARDIAC-FM Cox run. Train on UKB survival, then test-set C-index. Array over outcome.
source /gpfs/projects/trend/bojun/CHS_MESA/scripts/config.sh
UKB=/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual
OUTS=(af hf); O=${OUTS[$SLURM_ARRAY_TASK_ID]}
D=/gpfs/projects/trend/bojun/multimodal_rep/eval/cox/ecgfm/$O
cd "$SCRIPTS"

echo "===== [ECG-FM] Cox TRAIN outcome=$O ====="
python cox_finetune.py --seed 1 --epochs 20 --model_name ECGFM \
  --ecgfm_ckpt "$ECGFM" \
  --ecg_tsv_dir "$UKB/ECG_manifest_moretest" --label_dir "$UKB/ECG_label_surv/$O" \
  --save_dir "$D" --batch_size 64 || exit 1

CK=$(ls -v "$D"/epoch_*.pth 2>/dev/null | tail -1)
[ -z "$CK" ] && { echo "!! no checkpoint for $O"; exit 1; }
echo "===== [ECG-FM] Cox TEST outcome=$O  ckpt=$(basename "$CK") ====="
python cox_test.py --outcome "$O" --model_name ECGFM --ckpt "$CK" --ecgfm_ckpt "$ECGFM" \
  --ecg_tsv_dir "$UKB/ECG_manifest_moretest" --label_dir "$UKB/ECG_label_surv/$O" --save_dir "$D/test"
echo "done $O"
