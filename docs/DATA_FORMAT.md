# Data format

What CARDIAC-FM reads and how to produce it. `tools/prepare_ecg.py` and `tools/prepare_mri.py`
cover both modalities; the MRI path additionally requires segmentations, which you must supply.

## ECG

### Layout

```
<root>/
  ECG/
    <id>.mat                 one file per recording
  ECG_manifest/
    test.tsv                 one manifest per split
```

### `.mat` contents

Written by `scipy.io.savemat`, read by `fairseq-signals`' `FileECGDataset`.

| key | shape | dtype | meaning |
|---|---|---|---|
| `feats` | `(12, 5000)` | float64 | 12 leads × 5000 samples (10 s at 500 Hz) |
| `idx` | `(1, 1)` | int64 | row index into the label array `y.npy`, when labels are used |
| `org_sample_size` | `(1, 1)` | int64 | length before resampling |
| `curr_sample_size` | `(1, 1)` | int64 | length after resampling (5000) |
| `org_sample_rate` | `(1, 1)` | int64 | sampling rate before resampling |
| `curr_sample_rate` | `(1, 1)` | int64 | sampling rate after resampling (500) |

Only `feats` reaches the model. `idx` matters solely for label lookup, so it is irrelevant to
label-free inference.

### Manifest (`.tsv`)

```
<TAB>/absolute/path/to/ECG/
6045703.mat<TAB>5000
6094857.mat<TAB>5000
```

Line 1 is **a literal tab character followed by the absolute data root**, with a trailing slash.
Every later line is `<filename><TAB><number of samples>`. Paths are resolved against the root on
line 1, so moving the directory means rewriting that line.

### Producing it

`tools/prepare_ecg.py` converts WFDB, CSV, or NumPy inputs:

```bash
python tools/prepare_ecg.py --in_dir raw/ --format wfdb --out_root prepared/ --split test
python tools/prepare_ecg.py --in_dir raw/ --format csv --sample_rate 250 --out_root prepared/
python tools/prepare_ecg.py --in_dir raw/ --format npy --sample_rate 500 --out_root prepared/
```

It resamples to 500 Hz, center-crops or zero-pads to 5000 samples, reorders leads when the input
carries lead names, and writes both the `.mat` files and the manifest.

Verified round-trip: exporting real recordings, converting them, and reading them back through
`FileECGDataset` reproduces the signal bit-identically after the float32 cast the loader applies.

### What you are responsible for

The converter handles shape and sampling rate. It cannot check that your signals are
*physiologically comparable* to the training data, and getting any of the following wrong produces
confident, meaningless scores:

- **Lead order.** The model expects `I, II, III, aVR, aVL, aVF, V1–V6`. Lead names are used to
  reorder when present; CSV and NumPy inputs carry none, so the channel order is taken as-is.
- **Amplitude scale.** The training data are not in millivolts — recorded values span roughly
  ±300 in the source units. When CHS was added, its amplitudes had to be **rescaled to match UK
  Biobank**. If your device's units differ, the model sees out-of-distribution inputs. Use
  `--scale` and sanity-check that your per-lead amplitude distribution resembles the training range.
- **Baseline wander.** Training ECGs were baseline-corrected with LOWESS. The converter does not
  apply this.
- **Duration.** 10 s at 500 Hz. Shorter recordings are zero-padded, which the model never saw in
  training; longer ones are center-cropped.

A reasonable check before trusting anything: score a cohort where you already know the outcome and
confirm the AUROC is plausible (`infer.py --labels`).

### Compute

`infer.py` uses a GPU when one is visible and falls back to CPU otherwise. Measured on 32 CPU cores,
ECG-only, batch size 8:

| | cost |
|---|---|
| model load | ~42 s, one-off per run |
| forward | ~1.6 s per ECG |

So a few hundred recordings are comfortable on CPU (1,000 ECGs ≈ 26 min), while cohort-scale work is
not (100,000 ECGs ≈ 44 h) — use a GPU for that. The ECG+MRI path is substantially heavier: it adds a
ViT-base over 3 views × 8 frames at 224², so treat a GPU as required there.

## Labels

Only needed for training or fine-tuning. Inference (`infer.py`) requires none.

### Binary / regression targets: `y.npy`

```
<label_root>/<task>/y.npy      shape (N, 1) float64
```

**`y.npy` is one global array covering every recording across all splits, indexed by the `idx`
field stored inside each `.mat`.** It is not per-split and not aligned to manifest order:

```python
label = np.load(f"{label_dir}/y.npy").squeeze()
idx   = int(loadmat(mat_path)["idx"].squeeze())
y     = label[idx]
```

So `N` is the size of your whole dataset, and row `i` describes the recording whose `.mat` carries
`idx == i`. Use `NaN` for "no label" — those rows are dropped from loss and metrics, which is how
partially-labelled cohorts are handled. Binary tasks use `0.0` / `1.0`; regression tasks store the
value itself.

`tools/prepare_ecg.py` assigns `idx` in the order it converts files (alphabetical within the run),
so build `y.npy` in that same order — or set `idx` yourself and match it.

### Survival targets

`common/train_eval/build_surv_labels.py` (UKB) and `build_surv_labels_external.py` (CHS/MESA) write
the survival `y.npy`, which uses a signed-time encoding: `|y|` is the follow-up time and `sign(y)`
is the event indicator. Build these before running `cox_finetune.py`.

### Split / phenotype CSVs

Three scripts read subject lists from CSV, and all of them key on a column named **`eid_visit`**:

| consumer | file | required columns |
|---|---|---|
| `train_mae.py --train_split_dir / --val_split_dir` | every `*.csv` in the directory | `eid_visit` |
| `stage1_CL_cinema.py --csv_train / --csv_val` | one CSV | `eid_visit` |
| `downstream_ecgmri_cinema.py --csv_train / --csv_val / --csv_test` | one CSV per split | `eid_visit`, plus a column named after `--outcome` |

Example (`af5.csv`):

```csv
eid_visit,af5
1029163_2,1
1030481_2,0
```

`eid_visit` must match the MRI subject directory name and the `.mat` stem, since that is how the
three modalities are joined. Rows whose outcome is missing are dropped.

## MRI

### Layout

```
<mri_root>/
  <subject_id>/
    vst_2ch.npy    (1, 25, 224, 224) float32
    vst_4ch.npy    (1, 25, 224, 224) float32
    vst_sa.npy     (3, 25, 224, 224) float32
```

`(slices, frames, height, width)` — one slice for each long-axis view, three short-axis slices.
Inference center-crops the temporal axis to 8 frames.

### Producing it

`tools/prepare_mri.py` runs the exact preprocessing the published checkpoints were trained with:

```bash
python tools/prepare_mri.py --raw_dir raw/ --out_dir prepared_mri/ --workers 8
```

Expected input, one directory per subject:

```
raw/<subject_id>/
  la_2ch.nii.gz    seg_la_2ch.nii.gz
  la_4ch.nii.gz    seg_la_4ch.nii.gz
  sa.nii.gz        seg_sa.nii.gz
```

The pipeline: select short-axis slices (middle ± 2) → resample to 0.994 mm → bounding box from the
segmentation (union over selected slices and all frames, +10% margin) → crop, pad to square, resize
to 224 → temporal stride 2 (50 frames → 25) → clip to [0.1%, 99.9%], rescale to [1, 255], z-score.

It finishes with a self-check that compares the output shapes and per-volume statistics against the
training data. Requires `nibabel` and `scikit-image`.

### You must supply segmentations

**The crop is derived from the segmentation, so it is not optional.** Expected labels:

| file | labels |
|---|---|
| `seg_la_2ch.nii.gz` | `1` = left atrium |
| `seg_la_4ch.nii.gz` | `1` = left atrium, `2` = left ventricle |
| `seg_sa.nii.gz` | `1` = LV cavity, `2` = myocardium, `3` = RV |

Different label numbers? Remap, or pass `--labels_2ch` / `--labels_4ch` / `--labels_sa`.

If a label set matches nothing, the code falls back to a centred crop. **The output still has the
right shape and still passes the statistics self-check** — z-scoring a wrongly-cropped volume looks
exactly like z-scoring a correct one. The only signal is the run's own warning, which counts how
many views fell back. Do not ignore it.

UK Biobank distributes segmentations with its imaging data. If you bring your own MRI, you must
generate them. The obvious candidate — the Bai et al. (2018) `ukbb_cardiac` FCN — ships weights for
short-axis and 4-chamber only; there is **no 2-chamber model**, and the 2-chamber view is required,
so that route is incomplete on its own.

### Practical advice

If you already have segmentations, the multimodal path is straightforward: run
`tools/prepare_mri.py`, then `infer.py --mode ecg_mri`. The published ECG+MRI checkpoint reaches
test AUROC 0.820 for atrial fibrillation, against 0.770 for ECG alone.

If you do **not** have segmentations, use the ECG-only path. Producing cardiac segmentation for all
three views — including the 2-chamber view, which has no released model — is a project in its own
right, and a systematically different crop is a silent distribution shift: the model still returns a
plausible-looking number.
