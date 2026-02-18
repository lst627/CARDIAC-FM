# CARDIAC-FM

<a href="https://huggingface.co/lst627/CARDIAC-FM"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>

<p align="center">
    <img src="teaser.png" alt="Overview" height="600"/>
</p>

## Introduction
CARDIAC-FM is a multimodal foundation model that integrates 12-lead electrocardiogram and cardiac magnetic resonance imaging data through contrastive learning. 
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
After filling in the paths for data and model weights in scripts/stage1.sh, you can use 
```bash
bash scripts/stage1.sh
```
for Stage 1 training. Note that (1) `--dry_run` is for testing the pipeline and will only use 100 samples for training, and (2) `--wandb` is for tracking the loss curves on wandb, which is not required in your training. Please remove `--dry_run` in actual training.
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
