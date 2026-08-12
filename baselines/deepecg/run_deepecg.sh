#!/bin/bash
#SBATCH --job-name=deepecg_zs
#SBATCH --partition=gpu-h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/deepecg_zs_%j.out
#SBATCH --error=logs/deepecg_zs_%j.err
# Zero-shot DeepECG 5-year AF benchmark on UKB test / CHS / MESA (af5 only; these models are
# AF-specific). Downloads the SSL checkpoint first (ungated), then runs both SL and SSL.

# --- repo path configuration (see docs/PATHS.md) ---
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_repo" != "/" ] && [ ! -d "$_repo/common" ]; do _repo="$(dirname "$_repo")"; done
source "$_repo/env/paths.local.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -u
ENV=${CONDA_ENV}
export PATH="$ENV/bin:$PATH"; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
cd "$HERE"

if [ ! -s wcr_afib_5y.pt ]; then
  echo "downloading wcr_afib_5y.pt (DeepECG-SSL) ..."
  curl -sL -o wcr_afib_5y.pt "https://huggingface.co/heartwise/WCR_AFIB_5Y/resolve/main/wcr_afib_5y.pt"
  ls -la wcr_afib_5y.pt
fi

echo "########## DeepECG-SL (EfficientNetV2_AFIB_5y) ##########"
python deepecg_run.py --model afib_5y.pt --tag deepecg_sl
echo "########## DeepECG-SSL (WCR_AFIB_5Y) ##########"
python deepecg_run.py --model wcr_afib_5y.pt --tag deepecg_ssl
echo DONE
