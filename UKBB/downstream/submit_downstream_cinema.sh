#!/bin/bash
#SBATCH --job-name=downstream_cinema
#SBATCH --partition=gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/downstream_cinema_%x_%j.out
#SBATCH --error=logs/downstream_cinema_%x_%j.err

# --- repo path configuration (see docs/PATHS.md) ---
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_repo" != "/" ] && [ ! -d "$_repo/common" ]; do _repo="$(dirname "$_repo")"; done
source "$_repo/env/paths.local.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# Usage: sbatch --job-name=ds_af5_ecg submit_downstream_cinema.sh af5 ecg_mri full [stage1_dir] [lr] [size]
# 4th arg = which contrastive checkpoint dir, e.g. stage1_cinema_base_ecgfull.
# 5th arg = downstream learning rate (default 1e-6).
# 6th arg = MRI encoder size (default base) — MUST match the checkpoint's architecture:
#   base  -> dim 768 / depth 12 / heads 12
#   small -> dim 384 / depth 8  / heads 6
# 7th arg = CL checkpoint file (default stage1_cinema_best.pth), e.g. stage1_cinema_ep008.pth
#   to evaluate an EARLY epoch (before the full-ECG run overfits).
# 8th arg = selection monitor (default val_loss): val_loss | val_auroc

OUTCOME=${1:-af5}
MODE=${2:-ecg_mri}      # ecg | ecg_mri
TYPE=${3:-full}      # full | senc_proj | partial | linear
STAGE1=${4:-stage1_cinema_base_ecgfull}   # contrastive checkpoint dir name
LR=${5:-1e-6}          # downstream learning rate
SIZE=${6:-base}        # base | small
CKPT_FILE=${7:-stage1_cinema_best.pth}   # which CL checkpoint, e.g. stage1_cinema_ep008.pth
SELECT_BY=${8:-val_loss}   # val_loss | val_auroc
case "$SIZE" in
  base)  EMBED_DIM=768; ENC_DEPTH=12; ENC_HEADS=12 ;;
  small) EMBED_DIM=384; ENC_DEPTH=8;  ENC_HEADS=6  ;;
  *) echo "SIZE must be 'base' or 'small'"; exit 1 ;;
esac
CKPT_TAG=${CKPT_FILE%.pth}   # for tagging the output dir

module load conda
conda activate ${CONDA_ENV}
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:$LD_LIBRARY_PATH

# --view_encoder and --pool MUST match the stage1 contrastive run that produced CL_CKPT.
VIEW_ENCODER=conv

CKPT_ROOT=${CKPT_ROOT}
CL_CKPT=$CKPT_ROOT/$STAGE1/$CKPT_FILE
ECG_CKPT=${ECG_CKPT}
MRI_DIR=${UKB_MRI_DIR}
ECG_DIR=${UKB_ECG_ROOT}/stage1/ecg_tsv
PHENO_DIR=${UKB_PHENO_DIR}
# tag output by outcome + stage1 source + ckpt + lr + selection so runs don't overwrite
# (OUTCOME was missing before -> af5/hf5 shared a dir and clobbered results.json)
SAVE=$CKPT_ROOT/downstream_${STAGE1}_${CKPT_TAG}_${OUTCOME}_${MODE}_lr${LR}_${SELECT_BY}

mkdir -p $SAVE

cd "$HERE"

python downstream_ecgmri_cinema.py \
  --outcome $OUTCOME --mode $MODE --training_type $TYPE \
  --cl_ckpt $CL_CKPT --ecg_ckpt $ECG_CKPT \
  --mri_dir $MRI_DIR --ecg_dir $ECG_DIR \
  --csv_train $PHENO_DIR/MRI_train/${OUTCOME}.csv \
  --csv_val   $PHENO_DIR/MRI_valid_new/${OUTCOME}.csv \
  --csv_test  $PHENO_DIR/MRI_test_new/${OUTCOME}.csv \
  --out_dir $SAVE \
  --view_encoder $VIEW_ENCODER --n_sa_slices 3 \
  --embed_dim $EMBED_DIM --encoder_depth $ENC_DEPTH --encoder_heads $ENC_HEADS \
  --select_by $SELECT_BY \
  --lr $LR --epochs 20 --batch_size 32 --img_size 224 \
  --n_frames 8 --pool per_view --num_workers 4
