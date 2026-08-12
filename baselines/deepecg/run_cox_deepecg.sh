#!/bin/bash
#SBATCH --job-name=cox_deepecg
#SBATCH --partition=gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/cox_deepecg_%A_%a.out
#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/cox_deepecg_%A_%a.err
# DeepECG-SL / DeepECG-SSL Cox (DeepSurv) fine-tune + test, as survival-downstream baselines.
# Array over (model x outcome): 0=deepsl/af 1=deepsl/hf 2=deepssl/af 3=deepssl/hf
set -u
D=/gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL/deepecg
ENV=/gpfs/projects/trend/bojun/mri_env; export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}" W=8
UKB=/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual
MODELS=(deepsl deepsl deepssl deepssl); OUTS=(af hf af hf)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}; O=${OUTS[$SLURM_ARRAY_TASK_ID]}
CKD=/gpfs/projects/trend/bojun/multimodal_rep/eval/cox/${M}_ft/$O
cd "$D"
echo "===== [$M / $O] Cox TRAIN ====="
python cox_deepecg.py train --model "$M" --outcome "$O" --seed 1 \
  --ecg_tsv_dir "$UKB/ECG_manifest_moretest" --label_dir "$UKB/ECG_label_surv/$O" \
  --ckpt_out "$CKD/best.pth" || exit 1
echo "===== [$M / $O] Cox TEST ====="
python cox_deepecg.py test --model "$M" --outcome "$O" --split test --ckpt_in "$CKD/best.pth" \
  --ecg_tsv_dir "$UKB/ECG_manifest_moretest" --label_dir "$UKB/ECG_label_surv/$O" --save_dir "$CKD/test"
echo "done $M $O"
