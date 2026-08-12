"""
ECG-only evaluation for CHS/MESA external validation. Eval + build_result + risk fusion
logic is COPIED VERBATIM from CARDIAC-FM/cardiac_fm/ecg_test_binary.py; only the imports
change (local model_ecg, repo dataset via sys.path). Writes result.csv (id,y_true,y_pred)
for the R figure notebooks.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # cardiacfm_new/
sys.path.insert(0, os.path.join(_ROOT, "common", "ecg_encoder"))   # model_ecg.py
sys.path.insert(0, os.path.join(_ROOT, "common", "data"))          # ecg_dataset.py
from model_ecg import (ECGFM, CARDIACFM_ECG, CHARGE_AF, CARDIACFM_ECG_Risk_AF,
                       PREVENT_HF, CARDIACFM_ECG_Risk_HF)
from ecg_dataset import ECGDataset


def build_result(result, args):
    ecg_tsv = pd.read_csv(f"{args.ecg_tsv_dir}/{args.split}.tsv", sep="\t")
    label = np.load(f"{args.label_dir}/y.npy").squeeze()
    mat_dir = ecg_tsv.columns[1]

    idx_list, y_true = [], []
    for i in range(len(ecg_tsv)):
        mat = loadmat(os.path.join(mat_dir, ecg_tsv.iloc[i, 0]))
        idx = int(mat["idx"].squeeze())
        idx_list.append(idx)
        y_true.append(label[idx])

    ecg_tsv["idx"] = idx_list
    ecg_tsv["y_true"] = y_true
    mask = ecg_tsv["y_true"].notna()
    ecg_tsv.loc[mask, "y_pred"] = result["y_pred"].values

    result_df = ecg_tsv.iloc[:, [0, 3, 4]].copy()
    result_df.columns = ["id", "y_true", "y_pred"]
    result_df["id"] = result_df["id"].str.replace(".mat", "", regex=False)
    return result_df


def val_one_epoch(val_data_loader, model, loss_fn, device, args):
    y_true_all, y_pred_all = [], []
    model.eval()
    with torch.no_grad():
        for i, ecgs in enumerate(val_data_loader):
            ecgs["net_input"]["source"] = ecgs["net_input"]["source"].to(device)
            ecg_features = model(ecgs)
            y_pred = torch.sigmoid(ecg_features)
            labels = ecgs["label"].to(device).float()
            if torch.isnan(labels).all():
                continue
            mask = ~torch.isnan(labels)
            y_pred, labels = y_pred[mask], labels[mask]
            _loss = loss_fn(y_pred, labels)
            y_true_all.extend(labels.cpu().numpy())
            y_pred_all.extend(y_pred.cpu().numpy())

    df = pd.DataFrame({'y_true': y_true_all, 'y_pred': y_pred_all})
    save_path = os.path.join(args.save_dir, "result.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = build_result(df, args)

    if args.risk_path != '':
        risk_factors = pd.read_csv(args.risk_path)
        if args.risk_model == "AF":
            df["risk_score"] = CHARGE_AF(risk_factors)
            df = df.rename(columns={"y_pred": "model_score"})
            df["y_pred"] = CARDIACFM_ECG_Risk_AF(df, args.seed)
        else:
            df["risk_score"] = PREVENT_HF(risk_factors)
            df = df.rename(columns={"y_pred": "model_score"})
            df["y_pred"] = CARDIACFM_ECG_Risk_HF(df, args.seed)

    epoch_auc = roc_auc_score(
        df.loc[df["y_true"].notna() & df["y_pred"].notna(), "y_true"],
        df.loc[df["y_true"].notna() & df["y_pred"].notna(), "y_pred"])
    df[["id", "y_true", "y_pred"]].to_csv(save_path, index=False)
    return epoch_auc


def test(batch_size, args):
    labels_dir = f"{args.label_dir}/y.npy"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    testset = ECGDataset(args.ecg_tsv_dir, labels_dir, split=args.split)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=ECGDataset.collate_fn)

    if "ECGFM" in args.model_name:
        print("\nLoading ECGFM model")
        model = ECGFM(ecgfm_ckpt=args.ecgfm_ckpt)
    elif "CARDIACFM" in args.model_name:
        print("\nLoading CARDIACFM model")
        model = CARDIACFM_ECG(ecgfm_ckpt=args.ecgfm_ckpt)

    checkpoint = torch.load(args.finetuned_ckpt, weights_only=False)
    model.load_state_dict(checkpoint)
    model.to(device)
    print("\t Total Params = ", sum(p.numel() for p in model.parameters()))

    loss_fn = nn.BCELoss()
    print("\n\t Started Evaluation on Test Set\n")
    val_auc = val_one_epoch(testloader, model, loss_fn, device, args)
    print("\t Test auc ......", round(val_auc, 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='', help='Path to save test results')
    parser.add_argument('--model_name', type=str, help='Model Name, ECGFM or CARDIACFM')
    parser.add_argument('--ecg_tsv_dir', type=str, default='', help='Path to ecgs tsv dir')
    parser.add_argument('--label_dir', type=str, help='Path to downstream task label')
    parser.add_argument('--ecgfm_ckpt', type=str, default='', help='Path to ecgfm model')
    parser.add_argument('--finetuned_ckpt', type=str, default='', help='Path to fine-tuned model')
    parser.add_argument('--risk_path', type=str, default='', help='Path to risk factor csv')
    parser.add_argument('--risk_model', type=str, default='', help='AF or HF')
    parser.add_argument('--seed', default=1, type=int, help='Seed')
    parser.add_argument('--split', default='test', help='which manifest split to run: test|valid|train')
    args = parser.parse_args()
    test(batch_size=4, args=args)
