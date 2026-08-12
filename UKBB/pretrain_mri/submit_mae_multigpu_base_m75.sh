#!/bin/bash
#SBATCH --job-name=cinema_base_m75
#SBATCH --qos=normal
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cinema_base_m75_%j.out
#SBATCH --error=logs/cinema_base_m75_%j.err

# --- repo path configuration (see docs/PATHS.md) ---
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_repo" != "/" ] && [ ! -d "$_repo/common" ]; do _repo="$(dirname "$_repo")"; done
source "$_repo/env/paths.local.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# ViT-BASE CineMA pretrain, MASK RATIO 0.75 (vs the 0.9 default) — CineMA's ratio.
# Only change vs cinema_base_conv_224_mg is --mask_ratio, so it isolates the mask-rate
# effect. Note: 0.75 keeps ~25% of tokens visible to the encoder (vs ~10% at 0.9), i.e.
# ~2.5x more tokens through the base encoder -> more memory; if OOM, drop --batch_size
# to 8 and --accum_steps to 4 (keeps effective batch 128). Trains to its own ckpt dir.

module load conda
conda activate ${CONDA_ENV}

NGPUS=4
VIEW_ENCODER=conv

DATA=${UKB_MRI_DIR}
CKPT=${CKPT_ROOT}/cinema_base_conv_224_m75_mg
LOGS=${LOG_DIR}
TR=${MRI_SPLITS}/train
VA=${MRI_SPLITS}/val

mkdir -p $CKPT $LOGS

cd "$HERE"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((20000 + RANDOM % 20000))

torchrun --standalone --nproc_per_node=$NGPUS train_mae.py \
  --data_dir $DATA --train_split_dir $TR --val_split_dir $VA \
  --save_dir $CKPT --log_dir $LOGS \
  --view_encoder $VIEW_ENCODER --n_sa_slices 3 \
  --encoder_dim 768 --encoder_depth 12 --encoder_heads 12 \
  --decoder_dim 512 --decoder_depth 8 --decoder_heads 16 \
  --img_size 224 --batch_size 16 --accum_steps 2 \
  --mask_ratio 0.75 --epochs 200 --lr 1.5e-4 --warmup_epochs 10 --patience 7 \
  --num_workers 8 --log_name cinema_base_conv_224_m75_mg.log
