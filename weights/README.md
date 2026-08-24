# Weights

**No checkpoints are committed to this repository.** This file describes what each one is, what
produces it, and what consumes it.

| file | what | produced by | consumed by |
|---|---|---|---|
| `ecgfm_mimic_iv_physionet.pt` | ECG-FM backbone (McKeen et al.), loaded via `fairseq-signals` `build_model_from_checkpoint` | upstream ECG-FM / fairseq-signals release | every ECG script — builds the ECG encoder architecture before the aligned weights are loaded on top |
| `cinema_mae_m75.pth` | self-supervised MRI encoder: cross-view CineMA MAE, mask ratio 0.75, ViT-base | `UKBB/pretrain_mri/train_mae.py` | `UKBB/contrastive/stage1_CL_cinema.py` (`--mae_ckpt`) — only needed to re-run contrastive alignment from scratch |
| `stage1_cinema_m75.pth` | **the headline model** — ECG↔MRI contrastively aligned encoder (~2.2 GB) | `UKBB/contrastive/stage1_CL_cinema.py` | `UKBB/downstream/downstream_ecgmri_cinema.py` (`--cl_ckpt`), `run_cox.sh`, and all CHS/MESA zero-/few-shot evaluation |
| `downstream_m75/af5_ecg.pth`, `hf5_ecg.pth` | AF / HF classifier fine-tuned on UK Biobank, **ECG only** (~0.7 GB each) | `downstream_ecgmri_cinema.py --mode ecg` | `infer.py --mode ecg` — ready to score ECGs, no fine-tuning needed |
| `downstream_m75/af5_ecg_mri.pth`, `hf5_ecg_mri.pth` | the same, **ECG + MRI** (~0.7 GB each) | `downstream_ecgmri_cinema.py --mode ecg_mri` | `infer.py --mode ecg_mri` |

## Where to get them

- **`ecgfm_mimic_iv_physionet.pt`** — from the upstream
  [ECG-FM](https://github.com/bowang-lab/ECG-FM) release (MIMIC-IV + PhysioNet pretrained). This is a
  third-party checkpoint; follow that project's terms.
- **`cinema_mae_m75.pth`** and **`stage1_cinema_m75.pth`** — released alongside this repository at
  <https://huggingface.co/lst627/CARDIAC-FM>.

## Minimum to run

- **Score your own ECGs** — no training at all: `ecgfm_mimic_iv_physionet.pt` +
  one of `downstream_m75/*_ecg.pth`. See the Quick start in the top-level README.
- **Fine-tune on your own outcome**: `ecgfm_mimic_iv_physionet.pt` + `stage1_cinema_m75.pth`.
- **Full pipeline from stage 1**: add `cinema_mae_m75.pth`.
- **Full pipeline from scratch**: nothing beyond `ecgfm_mimic_iv_physionet.pt`, but you need UK
  Biobank MRI access to pretrain the MAE yourself.
- **Continue fine-tuning based on our fine-tuned model**: add the matching
  `downstream_m75/{af5,hf5}_{ecg,ecg_mri}.pth` checkpoint and pass it with `--finetuned_ckpt` as
  shown in the top-level README.

## Pointing the scripts at them

Checkpoint locations are environment variables like everything else. Set them in
`env/paths.local.sh` (copy it from `env/paths.example.sh`):

```bash
export ECG_CKPT=/path/to/CARDIAC-FM/weights/ecgfm_mimic_iv_physionet.pt
export CKPT_ROOT=/path/to/checkpoints          # where MAE / stage-1 / downstream runs live
```

See [`docs/PATHS.md`](../docs/PATHS.md) for every variable and how resolution works.

If you place the checkpoints in this directory, they are already covered by `.gitignore`.

## Baselines

Baseline checkpoints are **not** redistributed here — obtain them from their own releases. They are
only needed for `baselines/`, not for the main CARDIAC-FM pipeline.

| model | checkpoint | source |
|---|---|---|
| ECGFounder | `12_lead_ECGFounder.pth` | [ECGFounder release](https://github.com/PKUDigitalHealth/ECGFounder) |
| DeepECG-SL | `efficientnet_77.pt` | HeartWise / DeepECG release |
| DeepECG-SSL | `SSL_pretrained.pt` | HeartWise / DeepECG release |

`baselines/ecgfounder/ecgfounder_run.py` additionally needs the ECGFounder repository itself on
`sys.path`, via its `ECGFOUNDER_REPO` variable.

## Dependency note

`fairseq-signals` is required to load any ECG checkpoint — it provides both the encoder
(`build_model_from_checkpoint`) and the `.mat` reader (`FileECGDataset`).
See [`env/SETUP.md`](../env/SETUP.md).
