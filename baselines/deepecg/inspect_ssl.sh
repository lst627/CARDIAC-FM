#!/bin/bash
#SBATCH --job-name=ssl_inspect
#SBATCH --partition=gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ssl_inspect_%j.out
#SBATCH --error=logs/ssl_inspect_%j.err

# --- repo path configuration (see docs/PATHS.md) ---
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_repo" != "/" ] && [ ! -d "$_repo/common" ]; do _repo="$(dirname "$_repo")"; done
source "$_repo/env/paths.local.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -u
ENV=${CONDA_ENV}
export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
cd "$HERE"
if [ ! -s SSL_pretrained.pt ]; then
  echo "downloading SSL_pretrained.pt (DeepECG-SSL backbone, ~1GB) ..."
  curl -sL -o SSL_pretrained.pt "https://huggingface.co/heartwise/SSL_Pretrained_model/resolve/main/SSL_pretrained.pt"
  ls -la SSL_pretrained.pt
fi
python inspect_ssl.py
echo DONE
