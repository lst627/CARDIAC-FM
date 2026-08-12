"""
Build CHS/MESA survival labels (signed-T) for the ZERO-SHOT time-to-event downstream, in the exact
format the Cox test scripts expect: a global y.npy of shape (N,1) indexed by each .mat file's `idx`
field (same convention as build_surv_labels.py for UKB).

Encoding per sample:  +tto if event (inc==1);  -tto if censored (inc==0);  NaN if prevalent / tto<=0
/ missing / id not in the survival master. Decoded in the Cox loss: T=|label|, E=(label>0), NaN masked.

Survival source: CHS_split1.csv (id) / MESA_disease.csv (idno_visit). Writes, per cohort,
  <cohort data>/ECG_label_surv/{af,hf}/y.npy

  python build_surv_labels_external.py
"""
import os
import numpy as np, pandas as pd
from scipy.io import loadmat

BASE = "/gpfs/projects/trend/bojun/CHS_MESA"
COHORTS = {
    "CHS":  dict(data=f"{BASE}/data_train_valid_test_individual_CHS",
                 surv=f"{BASE}/risk_score/csv_train_valid_test_individual_id_disease/CHS_split1.csv", idcol="id"),
    "MESA": dict(data=f"{BASE}/data_train_valid_test_individual_MESA",
                 surv=f"{BASE}/MESA/MESA_disease.csv", idcol="idno_visit"),
}
OUTCOMES = {"af": ("ttoaf", "incaf", "prevaf"), "hf": ("ttohf", "inchf", "prevhf")}


def main():
    for coh, cfg in COHORTS.items():
        surv = pd.read_csv(cfg["surv"]); surv["id"] = surv[cfg["idcol"]].astype(str)
        surv = surv.drop_duplicates("id").set_index("id")
        D = cfg["data"]; MAN = f"{D}/ECG_manifest"
        N = len(np.load(f"{D}/ECG_label/af5/y.npy").squeeze())
        labels = {o: np.full(N, np.nan, dtype=np.float64) for o in OUTCOMES}
        seen = matched = 0
        for split in ["train", "valid", "test"]:
            f = f"{MAN}/{split}.tsv"
            if not os.path.exists(f):
                continue
            t = pd.read_csv(f, sep="\t"); root = t.columns[1]; files = t.iloc[:, 0].tolist()
            for fn in files:
                cid = os.path.splitext(fn)[0]          # id = .mat stem (CHS eid / MESA idno_visit)
                try:
                    idx = int(np.array(loadmat(os.path.join(root, fn), variable_names=["idx"])["idx"]).squeeze())
                except Exception:
                    continue
                seen += 1
                if cid not in surv.index:
                    continue
                matched += 1
                row = surv.loc[cid]
                for o, (tv, cv, pv) in OUTCOMES.items():
                    tto, cens, prev = row.get(tv), row.get(cv), row.get(pv)
                    if pd.isna(tto) or pd.isna(cens) or float(tto) <= 0 or (not pd.isna(prev) and int(prev) == 1):
                        continue
                    labels[o][idx] = float(tto) if int(cens) == 1 else -float(tto)
            print(f"  [{coh}/{split}] processed", flush=True)
        print(f"[{coh}] manifest files seen={seen}, matched to survival={matched} ({100*matched/max(seen,1):.1f}%)")
        for o in OUTCOMES:
            d = f"{D}/ECG_label_surv/{o}"; os.makedirs(d, exist_ok=True)
            np.save(f"{d}/y.npy", labels[o].reshape(-1, 1))
            lab = labels[o]
            print(f"  [{coh} {o}] events={int((lab>0).sum())} censored={int((lab<0).sum())} "
                  f"usable={int(np.isfinite(lab).sum())}/{N} -> {d}/y.npy")


if __name__ == "__main__":
    main()
