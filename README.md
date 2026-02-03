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

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

cd cardiac_fm/fairseq_signals
pip install --editable ./
```
To test if installation of some packages is successful:
```bash
python -c "import numpy, pandas, nibabel, wfdb, torch, torchvision, fairseq_signals; print('ok')"
```
## Inference

## Finetuning

## Training

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation