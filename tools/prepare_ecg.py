"""
Convert your 12-lead ECGs into the .mat + .tsv layout that CARDIAC-FM reads.

The model consumes ECGs through fairseq-signals' FileECGDataset, which expects a directory of
per-recording .mat files plus a tab-separated manifest. This script builds both from WFDB records,
CSV files, or NumPy arrays. See docs/DATA_FORMAT.md for the full specification and for the
preprocessing assumptions you are responsible for meeting.

    # WFDB (PhysioNet-style .hea/.dat pairs)
    python tools/prepare_ecg.py --in_dir raw/ --format wfdb --out_root prepared/ --split test

    # one CSV per recording, 12 rows x N columns (or N x 12 with --transpose)
    python tools/prepare_ecg.py --in_dir raw/ --format csv --out_root prepared/ --split test

    # one .npy per recording, shape (12, N)
    python tools/prepare_ecg.py --in_dir raw/ --format npy --out_root prepared/ --split test

Produces:
    prepared/ECG/<id>.mat            feats (12, 5000) float64, plus idx and sample-rate fields
    prepared/ECG_manifest/test.tsv   line 1 = data root, then "<id>.mat\\t<n_samples>"

Then:
    python infer.py --ecg_dir prepared/ECG_manifest --split test \\
      --ckpt af5_ecg.pth --ecg_ckpt ecgfm_mimic_iv_physionet.pt --out predictions.csv

IMPORTANT -- this script does not validate that your signals are physiologically comparable to the
training data. Read the "What you are responsible for" section of docs/DATA_FORMAT.md before
trusting any output.
"""
import argparse
import os
import sys

import numpy as np
from scipy.io import savemat

TARGET_RATE = 500
TARGET_LEN = 5000          # 10 s at 500 Hz
N_LEADS = 12

# The lead order the model was trained on. Reorder your data to match, or pass --lead_order.
STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def resample(sig, src_rate, dst_rate=TARGET_RATE):
    """Linear resample along time. Returns (12, M)."""
    if src_rate == dst_rate:
        return sig
    n_out = int(round(sig.shape[1] * dst_rate / src_rate))
    x_old = np.linspace(0.0, 1.0, sig.shape[1])
    x_new = np.linspace(0.0, 1.0, n_out)
    return np.stack([np.interp(x_new, x_old, ch) for ch in sig])


def fit_length(sig, target=TARGET_LEN):
    """Center-crop or zero-pad the time axis to exactly `target` samples."""
    n = sig.shape[1]
    if n == target:
        return sig
    if n > target:
        start = (n - target) // 2
        return sig[:, start:start + target]
    out = np.zeros((sig.shape[0], target), dtype=sig.dtype)
    start = (target - n) // 2
    out[:, start:start + n] = sig
    return out


def load_wfdb(path):
    try:
        import wfdb
    except ImportError:
        sys.exit("--format wfdb needs the wfdb package:  pip install wfdb")
    rec = wfdb.rdrecord(os.path.splitext(path)[0])
    sig = np.asarray(rec.p_signal, dtype=np.float64).T          # (leads, time)
    names = [str(s) for s in (rec.sig_name or [])]
    return sig, int(rec.fs), names


def load_csv(path, transpose):
    sig = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if transpose:
        sig = sig.T
    return sig, None, []


def load_npy(path, transpose):
    sig = np.load(path).astype(np.float64)
    if transpose:
        sig = sig.T
    return sig, None, []


def reorder(sig, names, wanted):
    """Reorder channels to `wanted` using the record's own lead names."""
    if not names or not wanted:
        return sig, False
    norm = {n.strip().lower(): i for i, n in enumerate(names)}
    idx = []
    for w in wanted:
        j = norm.get(w.strip().lower())
        if j is None:
            return sig, False
        idx.append(j)
    return sig[idx], True


def main():
    ap = argparse.ArgumentParser(description="Convert 12-lead ECGs to the CARDIAC-FM .mat + .tsv layout.")
    ap.add_argument("--in_dir", required=True, help="directory of input recordings")
    ap.add_argument("--format", required=True, choices=["wfdb", "csv", "npy"])
    ap.add_argument("--out_root", required=True, help="output root; ECG/ and ECG_manifest/ go here")
    ap.add_argument("--split", default="test", help="manifest name, e.g. test -> ECG_manifest/test.tsv")
    ap.add_argument("--sample_rate", type=int, default=None,
                    help="source sampling rate in Hz. Read from the header for wfdb; REQUIRED for csv/npy")
    ap.add_argument("--transpose", action="store_true",
                    help="input is (time, leads) rather than (leads, time)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every sample by this. Use it to match the training amplitude "
                         "scale -- see docs/DATA_FORMAT.md")
    ap.add_argument("--lead_order", default=",".join(STANDARD_LEADS),
                    help="target lead order (comma-separated); applied when the input carries lead names")
    ap.add_argument("--limit", type=int, default=None, help="only convert the first N recordings")
    args = ap.parse_args()

    if args.format in ("csv", "npy") and not args.sample_rate:
        ap.error(f"--format {args.format} carries no header, so --sample_rate is required")

    exts = {"wfdb": (".hea",), "csv": (".csv",), "npy": (".npy",)}[args.format]
    files = sorted(f for f in os.listdir(args.in_dir) if f.lower().endswith(exts))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"no {'/'.join(exts)} files found in {args.in_dir}")

    ecg_dir = os.path.join(args.out_root, "ECG")
    man_dir = os.path.join(args.out_root, "ECG_manifest")
    os.makedirs(ecg_dir, exist_ok=True)
    os.makedirs(man_dir, exist_ok=True)

    wanted = [s for s in args.lead_order.split(",") if s.strip()]
    rows, skipped, reordered, unnamed = [], [], 0, 0

    for i, fn in enumerate(files):
        path = os.path.join(args.in_dir, fn)
        rec_id = os.path.splitext(fn)[0]
        try:
            if args.format == "wfdb":
                sig, rate, names = load_wfdb(path)
            elif args.format == "csv":
                sig, rate, names = load_csv(path, args.transpose)
            else:
                sig, rate, names = load_npy(path, args.transpose)
            rate = rate or args.sample_rate

            if sig.ndim != 2:
                raise ValueError(f"expected a 2-D array, got shape {sig.shape}")
            if sig.shape[0] != N_LEADS:
                raise ValueError(f"expected {N_LEADS} leads, got {sig.shape[0]} "
                                 f"(shape {sig.shape}; try --transpose)")

            sig, ok = reorder(sig, names, wanted)
            if names and ok:
                reordered += 1
            elif not names:
                unnamed += 1

            org_len = sig.shape[1]
            sig = resample(sig, rate)
            sig = fit_length(sig)
            sig = np.nan_to_num(sig * args.scale, nan=0.0, posinf=0.0, neginf=0.0)

            savemat(os.path.join(ecg_dir, f"{rec_id}.mat"), {
                "feats": sig.astype(np.float64),
                "idx": np.array([[i]], dtype=np.int64),
                "org_sample_size": np.array([[org_len]], dtype=np.int64),
                "curr_sample_size": np.array([[TARGET_LEN]], dtype=np.int64),
                "org_sample_rate": np.array([[rate]], dtype=np.int64),
                "curr_sample_rate": np.array([[TARGET_RATE]], dtype=np.int64),
            }, do_compression=False)
            rows.append((f"{rec_id}.mat", TARGET_LEN))
        except Exception as e:                                     # noqa: BLE001
            skipped.append((fn, str(e)))

    if not rows:
        sys.exit("every recording failed to convert; see the errors above")

    # Manifest: line 1 is a TAB followed by the absolute data root (trailing slash), then one
    # "<file>\t<n_samples>" per recording. FileECGDataset resolves each file against that root.
    manifest = os.path.join(man_dir, f"{args.split}.tsv")
    root = os.path.abspath(ecg_dir).rstrip("/") + "/"
    with open(manifest, "w") as f:
        f.write(f"\t{root}\n")
        for name, n in rows:
            f.write(f"{name}\t{n}\n")

    print(f"converted {len(rows)}/{len(files)} recordings")
    print(f"  signals  -> {ecg_dir}")
    print(f"  manifest -> {manifest}")
    if reordered:
        print(f"  reordered leads to {','.join(wanted)} for {reordered} record(s)")
    if unnamed:
        print(f"  {unnamed} record(s) carried no lead names -- channel order used as-is. "
              f"Confirm it matches {','.join(wanted)}.")
    if skipped:
        print(f"  SKIPPED {len(skipped)}:")
        for fn, err in skipped[:10]:
            print(f"    {fn}: {err}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

    print("\nNext:")
    print(f"  python infer.py --ecg_dir {man_dir} --split {args.split} \\")
    print( "    --ckpt af5_ecg.pth --ecg_ckpt ecgfm_mimic_iv_physionet.pt --out predictions.csv")
    print("\nBefore trusting the output, read the 'What you are responsible for' section of "
          "docs/DATA_FORMAT.md.")


if __name__ == "__main__":
    main()
