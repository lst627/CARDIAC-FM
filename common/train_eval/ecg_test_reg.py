"""
TEST-set inference for the CMR-feature regression (companion to ecg_finetune_reg.py).

Mirrors ecg_test.py (binary) but for regression:
  * no sigmoid -- raw linear output
  * predictions are un-standardized with the mu/sd saved at train time (target_scaling.json)
Writes result.csv (id, y_true, y_pred) on the ORIGINAL scale, keyed by eid_visit, so it can be
joined to the paper's own predictions and to UKBB_cmr_true_test.csv (which carries `healthy`).

NOTE: reporting must use the TEST set only. `val_r` printed during training is a model-selection
diagnostic, not a result -- the paper reports test-set r / R^2 restricted to healthy participants.

Usage: python ecg_test_reg.py --feature lvm --finetuned_ckpt <best.pth> --scaling <target_scaling.json> ...
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.io import loadmat
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # cardiacfm_new/
sys.path.insert(0, os.path.join(_ROOT, "common", "ecg_encoder"))   # model_ecg.py
sys.path.insert(0, os.path.join(_ROOT, "common", "data"))          # ecg_dataset.py
from model_ecg import ECGFM, CARDIACFM_ECG
from ecg_dataset import ECGDataset


def build_result(pred_vals, args):
    """map predictions back to eid_visit using the manifest + mat idx (same logic as ecg_test.py).

    NO-LABEL MODE (--no_label): external cohorts (CHS/MESA) have no CMR measurements, so there is no
    y.npy to mask against -- we predict for EVERY manifest row and emit y_true = NaN.
    """
    tsv = pd.read_csv(f"{args.ecg_tsv_dir}/{args.split}.tsv", sep="\t")
    if args.no_label:
        tsv["idx"] = np.nan
        tsv["y_true"] = np.nan
        tsv["y_pred"] = pred_vals            # one prediction per manifest row, order preserved
    else:
        label = np.load(f"{args.label_dir}/y.npy").squeeze()
        mat_dir = tsv.columns[1]
        idx_list, y_true = [], []
        for i in range(len(tsv)):
            mat = loadmat(os.path.join(mat_dir, tsv.iloc[i, 0]))
            idx = int(mat["idx"].squeeze())
            idx_list.append(idx); y_true.append(label[idx])
        tsv["idx"] = idx_list
        tsv["y_true"] = y_true
        mask = tsv["y_true"].notna()
        tsv.loc[mask, "y_pred"] = pred_vals
    out = tsv.iloc[:, [0, 3, 4]].copy()
    out.columns = ["id", "y_true", "y_pred"]
    out["id"] = out["id"].str.replace(".mat", "", regex=False)
    return out


@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sc = json.load(open(args.scaling))
    mu, sd = float(sc["mu"]), float(sc["sd"])
    print(f"feature={args.feature}  mu={mu:.3f} sd={sd:.3f}  split={args.split}", flush=True)

    ds = ECGDataset(args.ecg_tsv_dir, f"{args.label_dir}/y.npy", split=args.split)
    # (in --no_label mode the label array is ignored; only the waveforms are used)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
                    collate_fn=ECGDataset.collate_fn)

    if "ECGFM" in args.model_name:
        model = ECGFM(ecgfm_ckpt=args.ecgfm_ckpt)
    else:
        model = CARDIACFM_ECG(ecgfm_ckpt=args.ecgfm_ckpt)
    model.load_state_dict(torch.load(args.finetuned_ckpt, weights_only=False))
    model.to(device).eval()

    preds = []
    for ecgs in dl:
        ecgs["net_input"]["source"] = ecgs["net_input"]["source"].to(device)
        p = model(ecgs).cpu().numpy().ravel()
        if args.no_label:
            preds.extend(p)                          # every row -- no label mask available
        else:
            labels = ecgs["label"].numpy().ravel()
            preds.extend(p[~np.isnan(labels)])       # keep alignment with y_true.notna()
    preds = np.asarray(preds) * sd + mu             # back to the original scale

    res = build_result(preds, args)
    os.makedirs(args.save_dir, exist_ok=True)
    f = os.path.join(args.save_dir, "result.csv")
    res.to_csv(f, index=False)
    v = res.dropna(subset=["y_true", "y_pred"])
    if len(v) > 10:
        r = np.corrcoef(v.y_true, v.y_pred)[0, 1]
        r2 = 1 - np.mean((v.y_true - v.y_pred) ** 2) / np.var(v.y_true)
        print(f"[{args.feature}] TEST (all, unfiltered) n={len(v)}  r={r:.4f}  R2={r2:.4f} -> {f}", flush=True)
    else:
        n = res["y_pred"].notna().sum()
        print(f"[{args.feature}] no-label inference: {n} predictions -> {f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feature", required=True)
    p.add_argument("--model_name", default="CARDIACFM")
    p.add_argument("--ecgfm_ckpt", default="")
    p.add_argument("--finetuned_ckpt", required=True)
    p.add_argument("--scaling", required=True)
    p.add_argument("--ecg_tsv_dir", required=True)
    p.add_argument("--label_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--no_label", action="store_true",
                   help="external cohort with no CMR measurements: predict every manifest row")
    run(p.parse_args())
