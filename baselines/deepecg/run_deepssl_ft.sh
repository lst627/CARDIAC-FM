#!/bin/bash
#SBATCH --job-name=deepssl_ft
#SBATCH --partition=gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepssl_ft_%A_%a.out
#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepssl_ft_%A_%a.err
# Fine-tune DeepECG-SSL backbone on UKB (af5 / hf5), matched to ECGFounder; then zero-shot external.
# Array over outcome so af5 and hf5 run in parallel. Arm name: deepssl_ft.
set -u
D=/gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL/deepecg
ENV=/gpfs/projects/trend/bojun/mri_env; export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}" W=8
EVAL=/gpfs/projects/trend/bojun/multimodal_rep/eval
UKB=/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual
CHS=/gpfs/projects/trend/bojun/CHS_MESA/data_train_valid_test_individual_CHS
MESA=/gpfs/projects/trend/bojun/CHS_MESA/data_train_valid_test_individual_MESA
OUTS=(af5 hf5); O=${OUTS[$SLURM_ARRAY_TASK_ID]}
CK=$EVAL/ukb_ft/deepssl_ft/$O/best.pth
cd $D

echo "########## [$O] TRAIN on UKB (seed1, lr5e-6, patience3) ##########"
python deepssl_finetune.py train --outcome $O --ckpt_out $CK || exit 1

echo "########## [$O] INFERENCE ##########"
python deepssl_finetune.py test --outcome $O --split test --ckpt_in $CK \
  --ecg_tsv_dir $UKB/ECG_manifest_moretest --label_dir $UKB/ECG_label/$O \
  --save_dir $EVAL/ukb_test/deepssl_ft/$O
python deepssl_finetune.py test --outcome $O --split train --ckpt_in $CK \
  --ecg_tsv_dir $UKB/ECG_manifest_moretest --label_dir $UKB/ECG_label/$O \
  --save_dir $EVAL/ukb_train_preds/deepssl_ft/$O
for COH in CHS MESA; do
  C=$([ $COH = CHS ] && echo $CHS || echo $MESA)
  python deepssl_finetune.py test --outcome $O --split test --ckpt_in $CK \
    --ecg_tsv_dir $C/ECG_manifest --label_dir $C/ECG_label/$O \
    --save_dir $EVAL/zeroshot/$COH/deepssl_ft/$O
done
echo "########## [$O] DONE ##########"
