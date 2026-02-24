import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
from model import ECGFM, CARDIACFM_ECG, CHARGE_AF, CARDIACFM_ECG_Risk_AF, PREVENT_HF, CARDIACFM_ECG_Risk_HF
from utils import cosine_lr
from dataset import ECGDataset
from sklearn.metrics import roc_auc_score
import argparse
import pandas as pd
from glob import glob
from scipy.io import loadmat
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

def build_result(result):
    
    ecg_tsv_path = f"{args.ecg_tsv_dir}/test.tsv"
    ecg_tsv = pd.read_csv(ecg_tsv_path, sep="\t")
    
    label_path = f"{args.label_dir}/y.npy"
    label = np.load(label_path).squeeze()
    
    mat_dir = ecg_tsv.columns[1]

    idx_list = []
    y_true = []

    for i in range(len(ecg_tsv)):
        mat_path = os.path.join(mat_dir, ecg_tsv.iloc[i, 0])
        mat = loadmat(mat_path)
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
    
def val_one_epoch(val_data_loader, model, loss_fn, device):
    
    ### Local Parameters
    y_true_all = []
    y_pred_all = []

    model.eval()

    with torch.no_grad():
        ### Iterating over data loader
        for i, (ecgs) in enumerate(val_data_loader):
        

            # Forward pass
            ecgs["net_input"]["source"] = ecgs["net_input"]["source"].to(device)
            
            ecg_features = model(ecgs)
    
            y_pred = torch.sigmoid(ecg_features) 

            # Extract labels
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

    df = build_result(df)

    #### Risk Model

    if args.risk_path != '':
        risk_factors = pd.read_csv(args.risk_path)
        if args.risk_model == "AF":
            df["risk_score"] = CHARGE_AF(risk_factors)
            df = df.rename(columns={"y_pred": "model_score"})
            df["y_pred"] = CARDIACFM_ECG_Risk_AF(df, args.seed)
        else:
            risk_factors = pd.read_csv(args.risk_path)
            df["risk_score"] = PREVENT_HF(risk_factors)
            df = df.rename(columns={"y_pred": "model_score"})
            df["y_pred"] = CARDIACFM_ECG_Risk_HF(df, args.seed)

    epoch_auc = roc_auc_score(
    df.loc[df["y_true"].notna() & df["y_pred"].notna(), "y_true"],
    df.loc[df["y_true"].notna() & df["y_pred"].notna(), "y_pred"]
)

    df[["id", "y_true", "y_pred"]].to_csv(save_path, index=False)

    return epoch_auc

def cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def test(batch_size, args):
    """
    DataLoader
    """
    ecgs_dir = args.ecg_tsv_dir
    labels_dir = f"{args.label_dir}/y.npy"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    testset = ECGDataset(ecgs_dir, labels_dir, split="test")
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=ECGDataset.collate_fn)

    """
    Model and Loss
    """
    if "ECGFM" in args.model_name:
        print("\nLoading ECGFM model with checkpoint:", args.model_name)
        model = ECGFM(ecgfm_ckpt = args.ecgfm_ckpt)
    elif "CARDIACFM" in args.model_name:
        print("\nLoading CARDIACFM model with checkpoint:", args.model_name)
        model = CARDIACFM_ECG(ecgfm_ckpt = args.ecgfm_ckpt)

    ckpt_path = args.finetuned_ckpt
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)           
    model.to(device)
    
    print("\n\n\n\n\t Model Loaded")
    print("\t Total Params = ",sum(p.numel() for p in model.parameters()))
    print("\t Trainable Params = ",sum(p.numel() for p in model.parameters() if p.requires_grad))
    loss_fn = nn.BCELoss()
    
    print("\n\t Started Evaluation on Test Set\n")
    val_auc = val_one_epoch(testloader, model, loss_fn, device)
    print("\t Test auc ......", round(val_auc,4))
    #wandb.log({"val_loss": val_loss, "val_auc":val_auc})
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='', help='Path to save test results')
    parser.add_argument('--model_name', type=str, help='Model Name, ECGFM or CARDIACFM')
    parser.add_argument('--ecg_tsv_dir', type=str, default='', help='Path to ecgs tsv dir')
    parser.add_argument('--label_dir', type=str, help='Path to downstream task label')
    parser.add_argument('--ecgfm_ckpt', type=str, default='', help='Path to ecgfm model')
    parser.add_argument('--finetuned_ckpt', type=str, default='', help='Path to cardiacfm model, use this when you want ot finetune based on the model already finetuned on UKB')
    parser.add_argument('--risk_path', type=str, default='', help='Path to risk factor')
    parser.add_argument('--risk_model', type=str, default='', help='Choose the risk model to use, AF or HF')
    parser.add_argument('--seed', default=1, type=int, help='Seed')
    args = parser.parse_args()
    #wandb.init(
        #project="ecg_downstream",
        #name=f"{args.model_name}"
    #)
    test(batch_size=4, args=args)