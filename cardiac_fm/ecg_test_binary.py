import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import time
import numpy as np
from model import ECGFM, CARDIACFM_ECG
from utils import cosine_lr
from dataset import ECGDataset
from sklearn.metrics import roc_auc_score
import argparse
import pandas as pd
from glob import glob
mp.set_sharing_strategy("file_system")

def val_one_epoch(val_data_loader, model, loss_fn, device):
    
    ### Local Parameters
    epoch_loss = []
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
            epoch_loss.append(_loss)
            
            y_true_all.extend(labels.cpu().numpy()) 
            y_pred_all.extend(y_pred.cpu().numpy())  

    epoch_loss = np.mean([l.cpu().item() for l in epoch_loss])
    epoch_auc = roc_auc_score(y_true_all, y_pred_all)
    df = pd.DataFrame({'y_true': y_true_all, 'y_pred': y_pred_all})
    
    save_path = os.path.join(args.save_dir, "result.csv")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return epoch_loss, epoch_auc


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
    val_loss, val_auc = val_one_epoch(testloader, model, loss_fn, device)
    print("\t Test loss ......", round(val_loss,4))
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
    
    args = parser.parse_args()
    #wandb.init(
        #project="ecg_downstream",
        #name=f"{args.model_name}"
    #)
    test(batch_size=4, args=args)