#!/bin/bash
#SBATCH --job-name=ssl_inspect
#SBATCH --partition=gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/ssl_inspect_%j.out
#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/ssl_inspect_%j.err
set -u
ENV=/gpfs/projects/trend/bojun/mri_env
export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
cd /gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL/deepecg
if [ ! -s SSL_pretrained.pt ]; then
  echo "downloading SSL_pretrained.pt (DeepECG-SSL backbone, ~1GB) ..."
  curl -sL -o SSL_pretrained.pt "https://huggingface.co/heartwise/SSL_Pretrained_model/resolve/main/SSL_pretrained.pt"
  ls -la SSL_pretrained.pt
fi
python inspect_ssl.py
echo DONE
