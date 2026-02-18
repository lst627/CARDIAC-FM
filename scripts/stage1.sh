cd cardiac_fm

python stage1_CL.py \
  --lr 1e-4 \
  --epochs 20 \
  --batch_size 32 \
  --mri_csv_path /chru/analysis/lifm6/multi-modal/data_train_valid_test_individual/stage1 \
  --cropped_mri_path /chru/data/UKBB/MRI/HeartMRI/cropped \
  --ecg_tsv_path /chru/analysis/lifm6/multi-modal/data_train_valid_test_individual/stage1/ecg_tsv \
  --save_path /chru/analysis/lifm6/multi-modal/model_state_dict_train_valid_test_individual/stage1_CL1e-4_test \
  --pt_mri_path /chru/analysis/lifm6/multi-modal/model_state_dict_train_valid_test_individual/cnn_lstm_lvef/model_epoch_70.pt \
  --pt_ecg_path /chru/analysis/lifm6/mimic_iv_ecg_physionet_pretrained.pt \
  --wandb \
  --dry_run 