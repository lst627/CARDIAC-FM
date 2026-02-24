import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import time
import numpy as np
from model import CARDIACFM_ECGMRI
from utils import cosine_lr
from dataset import ECGMRIDataset
#import wandb
from sklearn.metrics import roc_auc_score
import argparse
import random
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

def log_metrics(epoch, train_loss, val_loss, lr, val_auc):
    wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": lr, "val_auc":val_auc})

def train_one_epoch(train_data_loader, model, optimizer, loss_fn, device, scheduler, num_of_steps):
    
    epoch_loss = []
   
    model.train()

    begin = time.time()
    ###Iterating over data loader
    for i, (mris, ecgs) in enumerate(train_data_loader): 


        if scheduler != None: scheduler(i + num_of_steps)
        #Loading data and labels to device
        # mris = mris.to(device)
        # ecgs = ecgs.to(device)

        #Reseting Gradients
        optimizer.zero_grad()
        ecgs["net_input"]["source"] = ecgs["net_input"]["source"].to(device)
        #Forward
        outputs = model(ecgs, mris)
        y_pred = torch.sigmoid(outputs) 
 
        labels = ecgs["label"].to(device).float() 

        mask = ~torch.isnan(labels)  
        if labels[mask].numel() == 0:
            print("WARNING: All labels are NaN in this batch!")
        y_pred, labels = y_pred[mask], labels[mask]  
        
        _loss = loss_fn(y_pred, labels)
        epoch_loss.append(_loss) 
        #Backward
        _loss.backward()
        optimizer.step()
    
        if i%10 == 0: 
            print("train_loss = ",_loss.item())

            elapsed_time = time.time() - begin
            estimated_total_time = elapsed_time * (len(train_data_loader) - i - 1) / (i + 1)
            print(f"Elapsed time: {elapsed_time:.2f}s, Estimated remaining time: {estimated_total_time:.2f}s")

    ###Acc and Loss
    epoch_loss = np.mean([l.cpu().item() for l in epoch_loss])

    return epoch_loss

def val_one_epoch(val_data_loader, model, loss_fn, device):
    
    ### Local Parameters
    epoch_loss = []
    y_true_all = []
    y_pred_all = []

    model.eval()

    with torch.no_grad():
        ### Iterating over data loader
        for i, (mris, ecgs) in enumerate(val_data_loader):

            # Forward pass
            ecgs["net_input"]["source"] = ecgs["net_input"]["source"].to(device)
            
            outputs = model(ecgs, mris)
    
            y_pred = torch.sigmoid(outputs) 

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

    

    return epoch_loss, epoch_auc


def cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def train_clip(batch_size, epochs, args):
    """
    DataLoader
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    mris_dir = args.mris_dir
    ecgs_tsv_dir = args.ecg_tsv_dir
    labels_dir = f"{args.label_dir}/y.npy"
    
    mris_train = f"{args.mris_csv_dir}/train.csv"
    mris_valid = f"{args.mris_csv_dir}/valid.csv"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    trainset = ECGMRIDataset(mris_train, ecgs_tsv_dir, mris_dir, labels_dir, split="train")
    validset = ECGMRIDataset(mris_valid, ecgs_tsv_dir, mris_dir, labels_dir, split="valid")
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=ECGMRIDataset.collate_fn)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=ECGMRIDataset.collate_fn)

    """
    Model and Loss
    """
    model = CARDIACFM_ECGMRI(ecgfm_ckpt=args.ecgfm_ckpt, densenet_ckpt=args.densenet_ckpt, cardiacfm_pretrained_ckpt = args.cardiacfm_pretrained_ckpt, device=device)
    model.to(device)

    if args.finetuned_ckpt != '':
        ckpt_path = args.finetuned_ckpt
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint)

        
    print("\n\n\n\n\t Model Loaded")
    print("\t Total Params = ",sum(p.numel() for p in model.parameters()))
    print("\t Trainable Params = ",sum(p.numel() for p in model.parameters() if p.requires_grad))
    """
    Train
    """
    num_batches = math.ceil(len(trainset) // batch_size)
    num_of_steps = 0
    loss_fn = nn.BCELoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.pred.parameters(), "lr": 1e-4},
            {"params": model.ecg_projection_multi.parameters(), "lr": 1e-4},
            {"params": model.ecg_encoder_multi.parameters(), "lr": 1e-5},
            {"params": model.mri_projection_multi.parameters(), "lr": 1e-5},
            {"params": model.mri_encoder_multi.parameters(), "lr": 1e-5},
        ],
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=1e-2
    )

    scheduler = cosine_lr(optimizer, base_lr=5e-6, warmup_length=50, steps=epochs * num_batches)

    print("\n\t Started Training\n")

    best_val_auc = -1
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):

        begin = time.time()

        ###Training
        train_loss = train_one_epoch(trainloader, model, optimizer, loss_fn, device, scheduler, num_of_steps)
        ###Validation
        val_loss,val_auc = val_one_epoch(validloader, model, loss_fn, device)
        num_of_steps += num_batches

        print('\n\n\t Epoch....', epoch + 1)
        print("\t Training loss ......",round(train_loss,4))
        print("\t Val loss ......", round(val_loss,4))
        print("\t Val auc ......", round(val_auc,4))
        print('\t Time per epoch (in mins) = ', round((time.time()-begin)/60,2),'\n\n')

        #log_metrics(epoch+1, train_loss, val_loss, optimizer.param_groups[0]['lr'], val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0

            save_path = os.path.join(args.save_dir, 'epoch_{}.pth'.format(epoch))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}! Best Val AUC = {best_val_auc:.4f}\n")
            break

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--label_dir', type=str, help='Path to downstream task label')
    parser.add_argument('--mris_dir', type=str, default='', help='Path to mris dir')
    parser.add_argument('--mris_csv_dir', type=str, default='', help='Path to mris csv dir')
    parser.add_argument('--ecg_tsv_dir', type=str, default='', help='Path to ecgs tsv dir')
    parser.add_argument('--ecgfm_ckpt', type=str, default='', help='Path to ecgfm model')
    parser.add_argument('--densenet_ckpt', type=str, default='', help='Path to cnnlstm model')
    parser.add_argument('--save_dir', type=str, help='Path to dir used to save model')
    parser.add_argument('--cardiacfm_pretrained_ckpt', type=str, help='contrastive pretrained weight, use this when you want to finetune based on contrastive pretrained weight.')
    parser.add_argument('--finetuned_ckpt', type=str, default='', help='Path to cardiacfm model, use this when you want ot finetune based on the model already finetuned on UKB')
    args = parser.parse_args()
    set_seed(args.seed)
    
    
        
    train_clip(batch_size=4, epochs=args.epochs, args=args) 







