#!/bin/bash
#SBATCH --job-name=cox_ft
#SBATCH --partition=gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/cox_ft_%A_%a.out
#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/cox_ft_%A_%a.err
# Time-to-event (Cox/DeepSurv) fine-tune of CARDIAC-FM ECG on UKB survival. Array over outcome.
source /gpfs/projects/trend/bojun/CHS_MESA/scripts/config.sh
UKB=/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual
CL="$CKPT_ROOT/stage1_cinema_base_m75_ecgfull/stage1_cinema_best.pth"
OUTS=(af hf); O=${OUTS[$SLURM_ARRAY_TASK_ID]}
cd "$SCRIPTS"
echo "===== Cox fine-tune: outcome=$O ====="
python cox_finetune.py --seed 1 --epochs 20 --model_name CARDIACFM \
  --ecgfm_ckpt "$ECGFM" --cardiacfm_pretrained_ckpt "$CL" \
  --ecg_tsv_dir "$UKB/ECG_manifest_moretest" --label_dir "$UKB/ECG_label_surv/$O" \
  --save_dir "/gpfs/projects/trend/bojun/multimodal_rep/eval/cox/m75_seed1/$O" \
  --batch_size 64
echo "done $O"
