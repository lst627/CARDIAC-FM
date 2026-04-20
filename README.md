# CARDIAC-FM

<a href="https://huggingface.co/lst627/CARDIAC-FM"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>

<p align="center">
    <img src="teaser.png" alt="Overview" height="600"/>
</p>

## Introduction

CARDIAC-FM is a multimodal foundation model for cardiovascular risk prediction using 12-lead electrocardiogram (ECG) and cardiac magnetic resonance imaging (MRI) data.

This repository provides code and pretrained models to:

- Train CARDIAC-FM on paired ECG–MRI data using contrastive learning  
- Fine-tune the model for downstream prediction tasks (e.g., atrial fibrillation, heart failure)  
- Run inference using different input settings depending on available data  

The model supports multiple input configurations:

- ECG only  
- ECG + MRI  
- ECG + clinical risk scores  
- ECG + MRI + clinical risk scores
  
## Environment
Ensure that you have NVIDIA GPUs and NCCL before installation.
```bash
git clone https://github.com/lst627/CARDIAC-FM
cd CARDIAC-FM

conda create -n cardiac python=3.9.21
conda activate cardiac
python3 -m pip install pip==24.0

pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102   --extra-index-url https://download.pytorch.org/whl/cu102
pip install -r requirements.txt

cd cardiac_fm/fairseq_signals_repo
pip install --editable ./
```
To test if installation of some packages is successful:
```bash
python -c "import numpy, pandas, nibabel, wfdb, torch, torchvision, fairseq_signals; print('ok')"
```
## Inference

## Finetuning

## Training

### Stage 1: Multi-modal Contrastive Pretraining

After filling in the paths for data and model weights in scripts/stage1.sh, you can use 

```bash
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

# --dry_run: runs a quick test using a small subset of data (remove for full training)
# --wandb: optional logging with Weights & Biases
```

for Stage 1 training. Note that (1) `--dry_run` is for testing the pipeline and will only use 100 samples for training, and (2) `--wandb` is for tracking the loss curves on wandb, which is not required in your training. Please remove `--dry_run` in actual training.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{li2026cardiacfm,
  title={CARDIAC-FM: A Multimodal Foundation Model for Cardiovascular Risk Prediction Using ECG and Cardiac MRI},
  author={Li, Fumin and Li, Siting and Qian, Yuhan and Chen, Bojun and Brody, Jennifer A and Yogeswaran, Vidhushei and Wiggins, Kerri L and Sitlani, Colleen M and Bis, Joshua C and Shojaie, Ali and others},
  journal={medRxiv},
  year={2026},
  doi={10.64898/2026.03.16.26348526}
}
