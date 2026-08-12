"""
Build UKB survival labels for the time-to-event (Cox/DeepSurv) downstream, aligned EXACTLY like the
binary y.npy (global array indexed by the .mat `idx` field, shape = #all samples).

Encoding (single float per sample, so it rides through the existing ECGDataset/y.npy path):
    label = +tto   if event   (cens==1)
            -tto   if censored (cens==0)
            NaN    if prevalent disease, tto<=0, missing, or eid not in the survival master
Decoded in the Cox loss:  delta = (label > 0);  T = |label|;  NaN -> masked out.

Survival source: csv_HR/UKBB_CMR_AF-HF-IS_WithDem_Analytic.csv (one row per eid; tto in days,
measured from the imaging visit). We join by eid = eid_visit.split('_')[0].

  python build_surv_labels.py            # writes ECG_label_surv/{af,hf}/y.npy
"""
import os
import numpy as np, pandas as pd
from scipy.io import loadmat

UKB = "/gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual"
MAN = f"{UKB}/ECG_manifest_moretest"
SURV = "/gpfs/projects/trend/bojun/CHS_MESA/risk_score/csv_HR/UKBB_CMR_AF-HF-IS_WithDem_Analytic.csv"
OUTROOT = f"{UKB}/ECG_label_surv"
OUTCOMES = {"af": ("ttoaf", "afcens", "prevaf"), "hf": ("ttohf", "hfcens", "prevhf")}


def main():
    surv = pd.read_csv(SURV)
    surv["eid"] = surv["eid"].astype(str)
    surv = surv.drop_duplicates(subset="eid").set_index("eid")

    # size the global arrays from y.npy (same shape / idx convention)
    y = np.load(f"{UKB}/ECG_label/af5/y.npy").squeeze()
    N = len(y)
    labels = {o: np.full(N, np.nan, dtype=np.float64) for o in OUTCOMES}

    n_seen = 0
    for split in ["train", "valid", "test"]:
        t = pd.read_csv(f"{MAN}/{split}.tsv", sep="\t")
        root = t.columns[1]; files = t.iloc[:, 0].tolist()
        for fn in files:
            eid = fn.split("_")[0]
            try:
                idx = int(np.array(loadmat(os.path.join(root, fn), variable_names=["idx"])["idx"]).squeeze())
            except Exception:
                continue
            n_seen += 1
            if eid not in surv.index:
                continue
            row = surv.loc[eid]
            for o, (tv, cv, pv) in OUTCOMES.items():
                tto, cens, prev = row.get(tv), row.get(cv), row.get(pv)
                if pd.isna(tto) or pd.isna(cens) or float(tto) <= 0 or (not pd.isna(prev) and int(prev) == 1):
                    continue
                labels[o][idx] = float(tto) if int(cens) == 1 else -float(tto)
        print(f"  {split}: processed ({n_seen} total files seen)", flush=True)

    for o in OUTCOMES:
        d = f"{OUTROOT}/{o}"; os.makedirs(d, exist_ok=True)
        np.save(f"{d}/y.npy", labels[o].reshape(-1, 1))     # match y.npy's (N,1) shape
        lab = labels[o]
        n_ev = int(np.sum(lab > 0)); n_cen = int(np.sum(lab < 0)); n_use = n_ev + n_cen
        print(f"[{o}] usable={n_use}  events={n_ev}  censored={n_cen}  (NaN/excluded={N - n_use})  -> {d}/y.npy")


if __name__ == "__main__":
    main()
