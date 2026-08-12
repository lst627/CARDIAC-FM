#!/bin/bash
#SBATCH --job-name=deepecg_wcr
#SBATCH --partition=gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepecg_wcr_%j.out
#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepecg_wcr_%j.err
ENV=/gpfs/projects/trend/bojun/mri_env
export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
cd /gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL/deepecg
python probe_wcr.py
echo DONE
