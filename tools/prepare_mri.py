r"""
Convert raw cardiac MRI + segmentations into the vst_*.npy volumes CARDIAC-FM reads.

This is the preprocessing the published m75 checkpoints were trained with. The algorithm is
reproduced exactly -- resampling, bounding box, crop/pad/resize, temporal stride and intensity
normalisation are unchanged -- because the encoder is sensitive to the field of view it sees.

    python tools/prepare_mri.py --raw_dir raw/ --out_dir prepared_mri/ --workers 8

Expected input, one directory per subject:

    raw/
      <subject_id>/
        la_2ch.nii.gz      seg_la_2ch.nii.gz
        la_4ch.nii.gz      seg_la_4ch.nii.gz
        sa.nii.gz          seg_sa.nii.gz

Output:

    prepared_mri/
      <subject_id>/
        vst_2ch.npy   (1, 25, 224, 224) float32
        vst_4ch.npy   (1, 25, 224, 224) float32
        vst_sa.npy    (3, 25, 224, 224) float32

Then:

    python infer.py --mode ecg_mri --mri_dir prepared_mri/ \
      --ecg_dir prepared/ECG_manifest --split test \
      --ckpt af5_ecg_mri.pth --ecg_ckpt ecgfm_mimic_iv_physionet.pt --out predictions.csv

YOU MUST SUPPLY SEGMENTATIONS. The crop is derived from them, so they are not optional and their
quality directly sets what the model sees. Expected label values:

    seg_la_2ch.nii.gz   1 = left atrium
    seg_la_4ch.nii.gz   1 = left atrium, 2 = left ventricle
    seg_sa.nii.gz       1 = LV cavity, 2 = myocardium, 3 = RV

If your segmentation uses different label numbers, remap it first or pass --labels_2ch / --labels_4ch
/ --labels_sa. A label set that matches nothing falls back to a centred crop, which is almost
certainly wrong for your data -- the run reports how often that happened.

Requires: nibabel, scikit-image  (pip install nibabel scikit-image)
"""
import argparse
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")          # skimage precision warnings

try:
    import nibabel as nib
    from skimage.transform import resize
except ImportError as e:                    # noqa: BLE001
    sys.exit(f"missing dependency: {e}\n  pip install nibabel scikit-image")

TARGET_SPACING = 0.994      # mm, isotropic in-plane
TARGET_HW = 224
TEMPORAL_STRIDE = 2         # 50 frames -> 25
SA_N_SLICES = 3             # middle +/- 2
BBOX_MARGIN = 0.10


def load_nifti(path: Path):
    img = nib.load(str(path))
    return img.get_fdata(), float(img.header.get_zooms()[0])


def resample_volume(volume, src_spacing, tgt_spacing, order=1):
    scale = src_spacing / tgt_spacing
    new_H = round(volume.shape[0] * scale)
    new_W = round(volume.shape[1] * scale)
    n_s, n_t = volume.shape[2], volume.shape[3]
    out = np.zeros((new_H, new_W, n_s, n_t), dtype=np.float32)
    for s in range(n_s):
        for t in range(n_t):
            out[:, :, s, t] = resize(volume[:, :, s, t], (new_H, new_W), order=order,
                                     anti_aliasing=(order > 0), preserve_range=True, mode="reflect")
    return out


def compute_bbox(seg, selected_slices, labels, margin=BBOX_MARGIN):
    """Union of the requested labels across selected slices and all frames, plus a margin.

    Returns (x1, x2, y1, y2, used_fallback).
    """
    H, W = seg.shape[0], seg.shape[1]
    union = np.zeros((H, W), dtype=bool)
    for s in selected_slices:
        for label in labels:
            union |= (seg[:, :, s, :] == label).any(axis=-1)

    if not union.any():
        pad = int(min(H, W) * 0.15)
        return pad, H - pad, pad, W - pad, True

    rows = np.where(union.any(axis=1))[0]
    cols = np.where(union.any(axis=0))[0]
    x1, x2 = int(rows.min()), int(rows.max())
    y1, y2 = int(cols.min()), int(cols.max())
    dx, dy = int((x2 - x1) * margin), int((y2 - y1) * margin)
    return (max(0, x1 - dx), min(H - 1, x2 + dx),
            max(0, y1 - dy), min(W - 1, y2 + dy), False)


def crop_pad_resize(frame, x1, x2, y1, y2, target_hw=TARGET_HW):
    crop = frame[x1:x2 + 1, y1:y2 + 1]
    h, w = crop.shape
    if h < w:
        pad = (w - h) // 2
        crop = np.pad(crop, ((pad, w - h - pad), (0, 0)))
    elif w < h:
        pad = (h - w) // 2
        crop = np.pad(crop, ((0, 0), (pad, h - w - pad)))
    return resize(crop, (target_hw, target_hw), order=1, anti_aliasing=True,
                  preserve_range=True, mode="reflect").astype(np.float32)


def normalise(volume, p_low=0.1, p_high=99.9):
    """Clip to percentiles, rescale to [1, 255], then z-score the whole volume."""
    lo, hi = np.percentile(volume, p_low), np.percentile(volume, p_high)
    scaled = 1.0 + (np.clip(volume, lo, hi) - lo) / (hi - lo + 1e-8) * 254.0
    mu, sigma = scaled.mean(), scaled.std()
    return ((scaled - mu) / (sigma + 1e-8)).astype(np.float32)


def select_sa_slices(n_total, n_select=SA_N_SLICES):
    mid, half = n_total // 2, (n_select - 1) // 2
    return [max(0, min(n_total - 1, mid + 2 * i)) for i in range(-half, half + 1)]


def process_view(raw_path, seg_path, selected_slices, bbox_labels):
    raw, src_sp = load_nifti(raw_path)
    seg, _ = load_nifti(seg_path)
    raw_r = resample_volume(raw, src_sp, TARGET_SPACING, order=1)
    seg_r = resample_volume(seg, src_sp, TARGET_SPACING, order=0)   # nearest: keep integer labels

    x1, x2, y1, y2, fallback = compute_bbox(seg_r, selected_slices, bbox_labels)
    raw_t = raw_r[:, :, :, ::TEMPORAL_STRIDE]
    n_t = raw_t.shape[3]

    volume = np.zeros((len(selected_slices), n_t, TARGET_HW, TARGET_HW), dtype=np.float32)
    for i, s in enumerate(selected_slices):
        for t in range(n_t):
            volume[i, t] = crop_pad_resize(raw_t[:, :, s, t], x1, x2, y1, y2)
    return normalise(volume), fallback


def process_subject(subject_dir: Path, out_root: Path, labels_cfg, overwrite=False):
    sid = subject_dir.name
    out_dir = out_root / sid
    if not overwrite and (out_dir / "vst_sa.npy").exists():
        return f"[SKIP]  {sid} - already processed", 0

    required = [("la_2ch.nii.gz", "seg_la_2ch.nii.gz"),
                ("la_4ch.nii.gz", "seg_la_4ch.nii.gz"),
                ("sa.nii.gz", "seg_sa.nii.gz")]
    for raw_f, seg_f in required:
        if not (subject_dir / raw_f).exists():
            return f"[SKIP]  {sid} - missing {raw_f}", 0
        if not (subject_dir / seg_f).exists():
            return f"[SKIP]  {sid} - missing {seg_f}", 0

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        sa_slices = select_sa_slices(nib.load(str(subject_dir / "sa.nii.gz")).shape[2])
        views = [
            ("vst_2ch.npy", "la_2ch.nii.gz", "seg_la_2ch.nii.gz", [0], labels_cfg["2ch"]),
            ("vst_4ch.npy", "la_4ch.nii.gz", "seg_la_4ch.nii.gz", [0], labels_cfg["4ch"]),
            ("vst_sa.npy",  "sa.nii.gz",     "seg_sa.nii.gz",     sa_slices, labels_cfg["sa"]),
        ]
        n_fallback = 0
        for out_name, raw_f, seg_f, slices, labels in views:
            vol, fb = process_view(subject_dir / raw_f, subject_dir / seg_f, slices, labels)
            n_fallback += int(fb)
            np.save(str(out_dir / out_name), vol)
        tag = f"[OK]    {sid}"
        if n_fallback:
            tag += f"  (WARNING: {n_fallback}/3 views had an empty segmentation -> centred crop)"
        return tag, n_fallback
    except Exception:                                              # noqa: BLE001
        return f"[ERROR] {sid}\n{traceback.format_exc()}", 0


def _parse_labels(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess cardiac MRI + segmentations into CARDIAC-FM vst_*.npy volumes.")
    ap.add_argument("--raw_dir", type=Path, required=True, help="one directory per subject")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true", help="reprocess subjects already done")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--labels_2ch", default="1", help="segmentation labels for the 2ch bounding box")
    ap.add_argument("--labels_4ch", default="1,2")
    ap.add_argument("--labels_sa", default="1,2,3")
    args = ap.parse_args()

    if not args.raw_dir.is_dir():
        sys.exit(f"--raw_dir not found: {args.raw_dir}")
    labels_cfg = {"2ch": _parse_labels(args.labels_2ch),
                  "4ch": _parse_labels(args.labels_4ch),
                  "sa":  _parse_labels(args.labels_sa)}

    subjects = sorted(d for d in args.raw_dir.iterdir() if d.is_dir())
    if args.limit:
        subjects = subjects[:args.limit]
    if not subjects:
        sys.exit(f"no subject directories under {args.raw_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(subjects)} subject(s); labels 2ch={labels_cfg['2ch']} "
          f"4ch={labels_cfg['4ch']} sa={labels_cfg['sa']}", flush=True)

    ok = skip = err = fallback_views = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_subject, s, args.out_dir, labels_cfg, args.overwrite): s
                for s in subjects}
        for i, fut in enumerate(as_completed(futs), 1):
            msg, nfb = fut.result()
            fallback_views += nfb
            if msg.startswith("[OK]"):
                ok += 1
            elif msg.startswith("[SKIP]"):
                skip += 1
            else:
                err += 1
                print(msg, file=sys.stderr, flush=True)
            if msg.startswith("[OK]") and nfb:
                print(msg, flush=True)
            if i % 50 == 0 or i == len(subjects):
                print(f"  {i}/{len(subjects)}  ok={ok} skip={skip} error={err}", flush=True)

    print(f"\ndone: {ok} processed, {skip} skipped, {err} failed -> {args.out_dir}")
    if fallback_views:
        print(f"WARNING: {fallback_views} view(s) had an empty segmentation and fell back to a "
              f"centred crop. Check that your label numbers match --labels_*; a wrong label set "
              f"silently produces the wrong field of view.")

    # ── self-check against the statistics of the training data ────────────────
    done = [d for d in args.out_dir.iterdir() if (d / "vst_sa.npy").exists()]
    if not done:
        return
    print("\nself-check (expected: shapes exactly as below, per-volume mean 0, std 1):")
    bad = 0
    for name, shape in [("vst_2ch.npy", (1, 25, 224, 224)),
                        ("vst_4ch.npy", (1, 25, 224, 224)),
                        ("vst_sa.npy",  (3, 25, 224, 224))]:
        arrs = [np.load(str(d / name)) for d in done[:20] if (d / name).exists()]
        if not arrs:
            continue
        shapes = {a.shape for a in arrs}
        mean, std = float(np.mean([a.mean() for a in arrs])), float(np.mean([a.std() for a in arrs]))
        shape_ok = shapes == {shape}
        stat_ok = abs(mean) < 1e-3 and abs(std - 1.0) < 1e-2
        bad += (not shape_ok) + (not stat_ok)
        print(f"  {name}: shape={shapes} {'OK' if shape_ok else f'EXPECTED {shape}'} | "
              f"mean={mean:+.4f} std={std:.4f} {'OK' if stat_ok else 'UNEXPECTED'}")
    if bad:
        print("  Something is off -- do not run inference on these volumes until it is understood.")
    else:
        print("  Matches the training-data statistics.")


if __name__ == "__main__":
    main()
