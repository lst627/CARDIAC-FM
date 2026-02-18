cd cardiac_fm
export WEIGHTS_PATH=your_model_weights_path

python stage1_CL.py \
  --lr 1e-4 \
  --epochs 20 \
  --batch_size 32 \
  --mri_csv_path your_mri_csv_path \
  --cropped_mri_path your_cropped_mri_path \
  --ecg_tsv_path your_ecg_tsv_path \
  --save_path your_save_path \
  --pt_mri_path your_mri_pretrained_path \
  --pt_ecg_path your_ecg_pretrained_path \
  --wandb \
  --dry_run 