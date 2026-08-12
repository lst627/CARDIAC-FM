# Environment setup

**Yes, you need to build your own environment** — the reference env (`mri_env`) is a conda env on the
authors' cluster and can't be shared directly. But it's small and reproducible. The only non-obvious part
is the **ECG side (`fairseq-signals`)**, a git install pinned to a specific commit, not a PyPI package.

> ⚠️ **Do NOT install facebookresearch/`fairseq`.** It is not used here (nothing imports it), and its
> pinned `hydra-core 1.0.7` / `omegaconf <2.1` break on Python 3.12 — this is the "hydra 1.0.7 / fairseq
> 0.12.2 conflict." `fairseq-signals` is self-contained and resolves `hydra-core` to 1.3.2, which works.

Reference env: **conda, Python 3.12.13**, CUDA GPU.

## What actually needs installing

| group | packages |
|---|---|
| core DL/numerical | torch, torchvision, numpy, scipy, pandas, scikit-learn, matplotlib, tqdm, pillow |
| survival | lifelines |
| ECG encoder + `.mat` reader | **fairseq-signals** (git, pinned commit) — self-contained, no facebookresearch `fairseq` |
| audio backend (fairseq-signals dep) | soundfile |

## Steps

```bash
# 1. fresh env
conda create -n cardiacfm python=3.12 -y
conda activate cardiacfm

# 2. torch FIRST, matching YOUR CUDA (do not blindly pin — pick the build for your GPU/driver
#    from https://pytorch.org). The reference env used torch 2.12.0 / torchvision 0.27.0, e.g.:
pip install torch==2.12.0 torchvision==0.27.0    # or the CUDA-specific index-url for your system

# 3. everything else, including the two pinned git installs
pip install -r requirements.txt
```

If `pip install -r` chokes on the `fairseq-signals` git line, install it explicitly and retry the rest:

```bash
pip install "git+https://github.com/Jwoo5/fairseq-signals.git@f8f0ff1c788a82c2059cb452cd5462898867489e"
```

(In the very unlikely event some code path asks for facebookresearch `fairseq` at runtime, install it
**without deps** so it can't drag in the incompatible hydra: `pip install --no-deps
"git+https://github.com/facebookresearch/fairseq.git@3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99"`. You
almost certainly won't need this.)

## Verify it imports

```bash
python - <<'PY'
import torch, lifelines, sklearn, pandas
import fairseq_signals                       # ECG encoder + FileECGDataset
from fairseq_signals.models import build_model_from_checkpoint
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("fairseq_signals OK")
PY
```

Then the in-repo modules (paths are wired to `common/`):

```bash
cd CARDIAC-FM
python UKBB/downstream/downstream_ecgmri_cinema.py --help
python common/train_eval/cox_test.py --help
```

## Notes

- **Exact reproduction:** `env/requirements-lock.txt` pins the PyPI packages of the reference env
  (122 pkgs) if you need to match versions precisely. Note it does **not** include the `fairseq-signals`
  git install — get that from `requirements.txt`. `requirements.txt` (repository root) is the curated
  minimal set, and the only place the git install is listed.
- **Missing `.so` at runtime** (e.g., a libstdc++/CUDA lib error): the authors prepend the env's lib dir
  to the loader path — `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`. Only needed if you
  hit such an error.
- **Weights** are not committed — see [`weights/README.md`](../weights/README.md) for what each checkpoint
  is and where to download it, then repoint the run scripts' `ECG_CKPT` / `CL_CKPT` at your copies.
  Minimum to run downstream: `ecgfm_mimic_iv_physionet.pt` + `stage1_cinema_m75.pth`.
- **Paths**: every data/checkpoint/output location comes from an environment variable. Run
  `cp env/paths.example.sh env/paths.local.sh`, fill it in, then check with `python common/paths.py`.
  See [`docs/PATHS.md`](../docs/PATHS.md).
