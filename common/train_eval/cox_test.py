"""
Test-set evaluation for the Cox/DeepSurv fine-tune: load the best checkpoint, infer log-hazard on
the UKB test split, report Harrell's C-index on the (held-out) test survival, and save predictions.

  python cox_test.py --outcome af --ckpt <best.pth> --ecgfm_ckpt <..> \
     --ecg_tsv_dir <ECG_manifest_moretest> --label_dir <ECG_label_surv/af> --save_dir <..>
"""
import os, sys, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # cardiacfm_new/
sys.path.insert(0, os.path.join(_ROOT, "common", "ecg_encoder"))   # model_ecg.py
sys.path.insert(0, os.path.join(_ROOT, "common", "data"))          # ecg_dataset.py
from model_ecg import CARDIACFM_ECG, ECGFM
from ecg_dataset import ECGDataset
import cox_finetune as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outcome", required=True)
    p.add_argument("--model_name", default="CARDIACFM")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ecgfm_ckpt", default="")
    p.add_argument("--ecg_tsv_dir", required=True)
    p.add_argument("--label_dir", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--batch_size", type=int, default=64)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "ECGFM" in a.model_name:
        model = ECGFM(ecgfm_ckpt=a.ecgfm_ckpt).to(dev)
    else:
        model = CARDIACFM_ECG(ecgfm_ckpt=a.ecgfm_ckpt, cardiacfm_pretrained_ckpt=None).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False))
    ds = ECGDataset(a.ecg_tsv_dir, f"{a.label_dir}/y.npy", split=a.split)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=4, collate_fn=ECGDataset.collate_fn)
    c, ev, n = C.val_cindex(dl, model, dev)     # reuse the exact eval used in training
    print(f"[{a.outcome} {a.split}] C-index = {c:.4f}   (events={ev}, n={n})", flush=True)
    # save the arrays for downstream use / the report
    os.makedirs(a.save_dir, exist_ok=True)
    model.eval(); H, TT, EE = [], [], []
    with torch.no_grad():
        for ecgs in dl:
            logh, T, E = C.run_batch(model, ecgs, dev)
            H.append(logh.cpu().numpy()); TT.append(T.cpu().numpy()); EE.append(E.cpu().numpy())
    import pandas as pd
    pd.DataFrame({"logh": np.concatenate(H), "T": np.concatenate(TT), "E": np.concatenate(EE)}) \
        .to_csv(f"{a.save_dir}/result.csv", index=False)
    with open(f"{a.save_dir}/cindex.txt", "w") as f:
        f.write(f"{a.outcome}\t{a.split}\tC-index={c:.4f}\tevents={ev}\tn={n}\n")
    print(f"[saved -> {a.save_dir}/result.csv, cindex.txt]", flush=True)


if __name__ == "__main__":
    main()
