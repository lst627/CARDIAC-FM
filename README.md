# CARDIAC-FM

<a href="https://huggingface.co/lst627/CARDIAC-FM"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>

<p align="center">
    <img src="teaser.png" alt="Overview" height="600"/>
</p>

## Introduction

CARDIAC-FM is a multimodal foundation model for cardiovascular risk prediction using 12-lead
electrocardiogram (ECG) and cardiac magnetic resonance imaging (MRI) data.

Paper: ***[CARDIAC-FM: A Multimodal Foundation Model for Cardiovascular Risk Prediction Using ECG and Cardiac MRI](https://www.medrxiv.org/content/10.64898/2026.03.16.26348526v1)***

The headline model (**m75**) is a cross-view [CineMA](https://github.com/mathpluscode/CineMA) masked
autoencoder (ViT-base, mask ratio 0.75) pretrained on UK Biobank cardiac MRI, then contrastively
aligned (InfoNCE) with an [ECG-FM](https://github.com/bowang-lab/ECG-FM) encoder. It is developed and
evaluated on UK Biobank and validated externally on CHS and MESA.

This repository provides code to:

- Self-supervised pretrain the cardiac MRI encoder (CineMA MAE)
- Contrastively align ECG and MRI representations
- Fine-tune for downstream prediction (atrial fibrillation, heart failure, CMR phenotypes, survival)
- Reproduce every figure and number in the paper, including all external validation

Supported input configurations:

- ECG only
- ECG + MRI
- ECG + clinical risk scores (CHARGE-AF / PREVENT-HF)
- ECG + MRI + clinical risk scores

> **Note on model versions.** This is the CineMA-based implementation (m75). An earlier
> DenseNet+LSTM ("CNN-LSTM") MRI encoder was used in the initial release of this repository; it has
> been removed from `main` and remains available in the git history at tag/commit `277b3eb`. If you
> are loading checkpoints published before this change, use that revision instead — the two
> architectures are **not** checkpoint-compatible.

## Repository layout

```
common/            cohort-agnostic, reusable code
  ecg_encoder/       ECG-FM wrapper (model_ecg.py)
  mri_encoder/       CineMA MAE (cinema_mae.py) + CineMAEncoder wrapper (encoder.py, head.py)
  data/              ecg_dataset.py (wrapper over fairseq-signals FileECGDataset), mri_dataset.py
  train_eval/        generic fine-tune/test engines (classification / regression / Cox) + label builders
  risk/              CHARGE-AF / PREVENT-HF risk scores (MICE-imputed) + late fusion
  stats/             bootstrap utilities (CI, paired comparison, zero-shot)
UKBB/
  pretrain_mri/      CineMA MAE self-supervised pretraining
  contrastive/       stage-1 ECG<->MRI alignment -> the m75 checkpoint
  downstream/        fine-tune the aligned encoder on UKB (ECG-only and ECG+MRI)
  figures/           Fig 2 (UKB), Fig 3 (external), CMR accuracy, UKB tertile Cox / KM
CHS_MESA/          external validation: zero-/few-shot, risk stratification, survival, CMR,
                   refit-clinical, calibration/IDI/DCA, subgroups
baselines/         ECGFounder, DeepECG-SL/SSL comparison models
figure_numbers/    cached machine-readable numbers behind every figure (re-plot without recompute)
env/               environment setup + pinned lock file
weights/           how to obtain checkpoints (no large binaries committed)
docs/              supplementary documentation (see docs/PATHS.md)
```

Every entrypoint resolves its imports from `common/` by adding the relevant module directory to
`sys.path` at startup, so scripts can be run directly from any launcher. The `.sh` files are SLURM
examples — adapt them to your scheduler.

## Environment

Full instructions and pinned versions: **[`env/SETUP.md`](env/SETUP.md)**.

```bash
git clone https://github.com/lst627/CARDIAC-FM
cd CARDIAC-FM

conda create -n cardiacfm python=3.12 -y
conda activate cardiacfm

# install torch matching YOUR CUDA first (see https://pytorch.org)
pip install torch==2.12.0 torchvision==0.27.0

pip install -r requirements.txt
```

The only non-obvious dependency is the ECG side: **`fairseq-signals`** is a git install pinned to a
commit, not a PyPI package. It provides the ECG-FM encoder (`build_model_from_checkpoint`) and the
`.mat` reader (`FileECGDataset`); everything ECG-side depends on it. Do **not** install
facebookresearch/`fairseq` — it is unused here and its pinned `hydra-core 1.0.7` breaks on Python 3.12.

Verify:

```bash
python -c "import torch, lifelines, sklearn, pandas, fairseq_signals; print('ok')"
```

## Checkpoints

Three checkpoints are involved; none are committed to this repository. See
[`weights/README.md`](weights/README.md) for what each one is and how to obtain it.

| checkpoint | what | needed for |
|---|---|---|
| `ecgfm_mimic_iv_physionet.pt` | ECG-FM backbone (upstream release) | everything ECG-side |
| `cinema_mae_m75.pth` | self-supervised MRI encoder (CineMA MAE, mask 0.75) | re-running stage-1 alignment |
| `stage1_cinema_m75.pth` | **the headline aligned model** | all downstream / evaluation |

Minimum to run downstream evaluation: `ecgfm_mimic_iv_physionet.pt` + `stage1_cinema_m75.pth`.

## Pipeline

Run in this order. Each stage consumes the previous stage's checkpoint.

> The architecture dimensions (`--view_encoder`, `--embed_dim`, `--encoder_depth`,
> `--encoder_heads`, `--n_sa_slices`, `--pool`) **must be identical across all three stages**.
> The values below are the ones that produced the published m75 model.

### 1. MRI self-supervised pretraining

```bash
torchrun --standalone --nproc_per_node=4 UKBB/pretrain_mri/train_mae.py \
  --data_dir $MRI_DIR --train_split_dir $TRAIN_SPLIT --val_split_dir $VAL_SPLIT \
  --save_dir $CKPT_DIR --log_dir $LOG_DIR \
  --view_encoder conv --n_sa_slices 3 \
  --encoder_dim 768 --encoder_depth 12 --encoder_heads 12 \
  --decoder_dim 512 --decoder_depth 8 --decoder_heads 16 \
  --img_size 224 --batch_size 16 --accum_steps 2 \
  --mask_ratio 0.75 --epochs 200 --lr 1.5e-4 --warmup_epochs 10 --patience 7 \
  --num_workers 8
```

This produces `cinema_mae_m75.pth`. Note the script's own defaults are the ViT-small / mask-0.9
configuration — the flags above are what make it ViT-base at mask 0.75. Full SLURM version:
`UKBB/pretrain_mri/submit_mae_multigpu_base_m75.sh`.

### 2. Contrastive ECG-MRI alignment (stage 1)

```bash
python UKBB/contrastive/stage1_CL_cinema.py \
  --mri_dir $MRI_DIR --ecg_dir $ECG_TSV_DIR \
  --csv_train $SPLITS/mri_train.csv --csv_val $SPLITS/mri_valid.csv \
  --mae_ckpt $WEIGHTS/cinema_mae_m75.pth \
  --ecg_ckpt $WEIGHTS/ecgfm_mimic_iv_physionet.pt \
  --out_dir  $SAVE \
  --view_encoder conv --n_sa_slices 3 \
  --embed_dim 768 --encoder_depth 12 --encoder_heads 12 \
  --lr 1e-5 --warmup_steps 50 --epochs 50 --batch_size 32 --n_frames 8 \
  --mri_tune full --ecg_unfreeze -1 --pool per_view --num_workers 4
```

MRI is always fully fine-tuned; the ECG side is the option: `--ecg_unfreeze -1` tunes the whole ECG
transformer (best for ECG-only downstream), `--ecg_unfreeze 2` tunes only the last 2 layers (best for
ECG+MRI). Stage 1 runs in fp32 — bf16 produced NaNs on the base MRI encoder with a fully-unfrozen ECG
encoder. This produces `stage1_cinema_m75.pth`. SLURM version:
`UKBB/contrastive/submit_stage1_cinema_base_m75.sh {full|partial}`.

### 3. Downstream fine-tuning on UK Biobank

**ECG + MRI:**

```bash
python UKBB/downstream/downstream_ecgmri_cinema.py \
  --outcome $OUTCOME --mode ecg_mri --training_type senc_proj \
  --cl_ckpt  $WEIGHTS/stage1_cinema_m75.pth \
  --ecg_ckpt $WEIGHTS/ecgfm_mimic_iv_physionet.pt \
  --mri_dir  $MRI_DIR --ecg_dir $ECG_TSV_DIR \
  --csv_train $PHENO_DIR/MRI_train/${OUTCOME}.csv \
  --csv_val   $PHENO_DIR/MRI_valid_new/${OUTCOME}.csv \
  --csv_test  $PHENO_DIR/MRI_test_new/${OUTCOME}.csv \
  --out_dir  $SAVE \
  --view_encoder conv --n_sa_slices 3 \
  --embed_dim 768 --encoder_depth 12 --encoder_heads 12 \
  --lr 5e-6 --epochs 20 --batch_size 32 --img_size 224 \
  --n_frames 8 --pool per_view --select_by val_loss --num_workers 4
```

Use `--mode ecg` for the ECG-only arm. `--view_encoder` and `--pool` **must** match the stage-1 run
that produced `--cl_ckpt`.

**ECG-only classification / regression / survival** use the generic engines in `common/train_eval/`:

```bash
python common/train_eval/ecg_finetune.py     --help   # binary classification
python common/train_eval/ecg_finetune_reg.py --help   # CMR phenotype regression
python common/train_eval/cox_finetune.py     --help   # survival (Cox / DeepSurv)
```

Survival labels are built first with `common/train_eval/build_surv_labels.py` (UKB) and
`build_surv_labels_external.py` (CHS/MESA). See `UKBB/downstream/run_cox.sh` for a worked example.

### 4. External validation on CHS / MESA

Zero-shot applies the UKB-fine-tuned model directly; few-shot re-fits on 10%/20% of the external
cohort. Scripts live in `CHS_MESA/`:

| analysis | script |
|---|---|
| survival zero-shot | `fig_cox_zeroshot.py`, `run_cox_zeroshot.sh` |
| risk stratification (tertile Cox) | `fig4_compare.py`, `fig4_tertile_cox.py` |
| few-shot, broad outcomes | `fig5_fewshot.py` |
| CMR features (predicted + MESA measured) | `cmr_feature_cox_external.py`, `cmr_feature_cox.py`, `mesa_cmr_corr.py` |
| refit-clinical incremental value | `refit_clinical_compare.py` |
| calibration / IDI-NRI / decision curves | `clinical_utility.py` |
| subgroups | `a3_subgroups.py`, `fig_subgroups.py` |
| baseline comparisons | `deepecg_compare.py`, `ensemble_compare.py` |

### 5. Figures

Figure scripts use a **compute -> cache -> render** split. The cached compute for every figure is
committed in [`figure_numbers/`](figure_numbers/README.md), so figures can be re-plotted in seconds
without re-running bootstraps or MICE:

```bash
python UKBB/figures/make_figures.py     # Fig 2 (UKB) + Fig 3 (external)
python CHS_MESA/fig4_compare.py         # Fig 4
python CHS_MESA/fig5_fewshot.py         # Fig 5
```

Headline configuration: our model = **m75, seed 3**; ECG-FM baseline = **seed 1**. Bootstraps use
B=2000 (Fig 4: B=1000), seed 42.

## Data

No data is committed. The pipeline expects:

- **UKB ECG** — 12-lead, 500 Hz, LOWESS baseline-corrected, as `.mat` read through
  `fairseq-signals` `FileECGDataset` with `.tsv` manifests.
- **UKB MRI** — per-subject `vst_2ch.npy`, `vst_4ch.npy`, `vst_sa.npy`. Preprocessing follows
  Wang et al. (Nat Med 2024): resample to 0.994 mm, mid +/- 2 short-axis slices, segmentation crop to
  224, clip / rescale / z-score.
- **CHS / MESA ECG** — same `.mat` + `.tsv` format; CHS amplitudes are rescaled to match UKB.
- **Phenotypes / outcomes / risk-factor tables** — CSV, keyed by subject-visit ID.

Access to UK Biobank, CHS, and MESA requires separate application to each study.

## Configuring paths

Every data, checkpoint, and output location is read from an environment variable — nothing is
hard-coded to a particular machine. Configure them once:

```bash
cp env/paths.example.sh env/paths.local.sh
$EDITOR env/paths.local.sh          # fill in your paths
python common/paths.py              # print what resolved, and what is still unset
```

`env/paths.local.sh` is gitignored and is the single source of truth: Python reads it through
[`common/paths.py`](common/paths.py), and the `.sh` run scripts `source` the same file. A variable
set in the environment (`export EVAL_ROOT=...`) overrides it. Missing variables produce an error
naming the variable, not a confusing "file not found".

See [`docs/PATHS.md`](docs/PATHS.md) for what each variable means.

## Known issues

- **UK Biobank standalone risk scores are mis-mapped.** The `MAPS["UKBB"]` column mapping in
  `common/risk/risk_score.py` is incorrect, so any UKB **+Risk** cell (Fig 2) is *provisional*.
  CHS and MESA risk scores are correct, and the refit-clinical analysis is unaffected. This is
  flagged inline in `figure_numbers/README.md`.
- Absolute cluster paths, as described above.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This work builds on [ECG-FM](https://github.com/bowang-lab/ECG-FM) (McKeen et al.),
[fairseq-signals](https://github.com/Jwoo5/fairseq-signals), and
[CineMA](https://github.com/mathpluscode/CineMA). Baseline comparisons use
[ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder) and DeepECG-SL/SSL.

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
```
