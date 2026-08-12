#!/bin/bash
#SBATCH --job-name=deepecg_ssl
#SBATCH --partition=gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepecg_ssl_%j.out
#SBATCH --error=/gpfs/projects/trend/bojun/multimodal_rep/eval/logs/deepecg_ssl_%j.err
# DeepECG-SSL (WCR, fairseq_signals) zero-shot af5 on UKB test / CHS / MESA.
# 250 Hz + per-lead z-score (probe: 0.756 vs 0.725 raw; 500 Hz only 0.60).
# SL arm already produced by job 185381 with the identical raw recipe -- not re-run.
set -u
ENV=/gpfs/projects/trend/bojun/mri_env
export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
cd /gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL/deepecg
python deepecg_run.py --model wcr_afib_5y.pt --tag deepecg_ssl --loader fairseq --norm perlead_z
echo DONE
