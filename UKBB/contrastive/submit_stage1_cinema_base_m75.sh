#!/bin/bash
#SBATCH --job-name=stage1_cinema_base_m75
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/projects/trend/bojun/mri/CineMA/logs/stage1_cinema_base_m75_%x_%j.out
#SBATCH --error=/gpfs/projects/trend/bojun/mri/CineMA/logs/stage1_cinema_base_m75_%x_%j.err

# Contrastive stage 1 on the 0.75-mask ViT-BASE MRI encoder (cinema_base_conv_224_m75_mg).
# Identical to submit_stage1_cinema_base.sh except the MRI checkpoint (m75) and the SAVE
# dirs. MRI always full-tune; ECG tuning is the option:
#   sbatch submit_stage1_cinema_base_m75.sh full     -> whole ECG transformer (best ecg-only)
#   sbatch submit_stage1_cinema_base_m75.sh partial  -> last 2 ECG layers   (best ecg+mri)
ECG_MODE=${1:-full}     # full | partial
case "$ECG_MODE" in
  full)    ECG_UNFREEZE=-1 ;;
  partial) ECG_UNFREEZE=2  ;;
  *) echo "ECG_MODE must be 'full' or 'partial'"; exit 1 ;;
esac

module load conda
conda activate /gpfs/projects/trend/bojun/mri_env
export LD_LIBRARY_PATH=/gpfs/projects/trend/bojun/mri_env/lib:$LD_LIBRARY_PATH

# base architecture dims (MUST match the base pretrain and the downstream run)
VIEW_ENCODER=conv
EMBED_DIM=768
ENC_DEPTH=12
ENC_HEADS=12

MRI_DIR=/gpfs/projects/trend/data/UKBB/MRI/cropped_new
ECG_DIR=/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual/stage1/ecg_tsv
SPLITS=/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual/stage1
MAE_CKPT=/gpfs/projects/trend/bojun/mri/CineMA/checkpoints/cinema_base_conv_224_m75_mg/cinema_best.pth
ECG_CKPT=/gpfs/projects/trend/bojun/multimodal/mimic_iv_ecg_physionet_pretrained.pt
SAVE=/gpfs/projects/trend/bojun/mri/CineMA/checkpoints/stage1_cinema_base_m75_ecg${ECG_MODE}

mkdir -p $SAVE

cd /gpfs/projects/trend/bojun/multimodal

# stage1 runs fp32 (bf16 NaN'd on the base MRI encoder + full ECG). grad clip 5.0 is on.
# batch 32 on 4 GPUs (nodes=1) ~ the same as the 0.9 base runs; drop to 16 if OOM.
python stage1_CL_cinema.py \
  --mri_dir $MRI_DIR --ecg_dir $ECG_DIR \
  --csv_train $SPLITS/mri_train.csv --csv_val $SPLITS/mri_valid.csv \
  --mae_ckpt $MAE_CKPT --ecg_ckpt $ECG_CKPT --out_dir $SAVE \
  --view_encoder $VIEW_ENCODER --n_sa_slices 3 \
  --embed_dim $EMBED_DIM --encoder_depth $ENC_DEPTH --encoder_heads $ENC_HEADS \
  --lr 1e-5 --warmup_steps 50 --epochs 50 --batch_size 32 --n_frames 8 \
  --num_workers 4 --mri_tune full --ecg_unfreeze $ECG_UNFREEZE --pool per_view
