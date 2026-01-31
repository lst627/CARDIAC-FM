import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import time
import numpy as np
from model import MMCL
from loss import InfoNCELoss
from utils import cosine_lr
from dataset import ECGMRIDataset
import wandb
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")
import argparse

def log_metrics(epoch, train_loss, val_loss, lr, temperature):
    wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": lr, "temperature": temperature})

def train_one_epoch(train_data_loader, model, optimizer, loss_fn, scheduler, num_of_steps):
    
    epoch_loss = []
   
    model.train()

    begin = time.time()
    ###Iterating over data loader
    for i, (mris,ecgs) in enumerate(train_data_loader):
        if scheduler != None: scheduler(i + num_of_steps)

        #Reseting Gradients
        optimizer.zero_grad()
        #Forward
        mri_features, ecg_features = model(mris, ecgs)
        #Calculating Loss
        _loss = loss_fn(mri_features, ecg_features)
        epoch_loss.append(_loss.item())      
        #Backward
        _loss.backward()
        optimizer.step()
    
        if i%10 == 0: 
            print("train_loss = ",_loss.item())
            elapsed_time = time.time() - begin
            estimated_total_time = elapsed_time * (len(train_data_loader) - i - 1) / (i + 1)
            print(f"Elapsed time: {elapsed_time:.2f}s, Estimated remaining time: {estimated_total_time:.2f}s")

    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    return epoch_loss

def val_one_epoch(val_data_loader, model, loss_fn):
    
    ### Local Parameters
    epoch_loss = []

    model.eval()

    with torch.no_grad():
        ###Iterating over data loader
        for i, (mris, ecgs) in enumerate(val_data_loader):
            #Forward
            mri_features, ecg_features = model(mris, ecgs)
            #Calculating Loss
            _loss = loss_fn(mri_features, ecg_features)
            epoch_loss.append(_loss.item())

    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    return epoch_loss

def cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def train_clip(args):
    """
    DataLoader
    """
    csv_train_file = os.path.join(args.mri_csv_path, "mri_train.csv") 
    csv_test_file = os.path.join(args.mri_csv_path, "mri_valid.csv") 
    ecgs_dir = args.ecg_tsv_path
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    batch_size = args.batch_size
    epochs = args.epochs
    mris_dir = args.cropped_mri_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    trainset = ECGMRIDataset(csv_train_file, ecgs_dir, mris_dir, split="train")
    testset = ECGMRIDataset(csv_test_file, ecgs_dir, mris_dir, split="valid")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=ECGMRIDataset.collate_fn)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=ECGMRIDataset.collate_fn)


    """
    Model and Loss
    """
    model = MMCL(mri_ckpt=args.pt_mri_path, ecg_ckpt=args.pt_ecg_path)
    model = nn.DataParallel(model,device_ids=range(torch.cuda.device_count()))
    model.to(device)
    print("\n\n\n\n\t Model Loaded")
    print("\t Total Params = ",sum(p.numel() for p in model.parameters()))
    print("\t Trainable Params = ",sum(p.numel() for p in model.parameters() if p.requires_grad))

    """
    Train
    """
    num_batches = math.ceil(len(trainset) // batch_size)
    num_of_steps = 0
    loss_fn = InfoNCELoss(temperature=0.07)
    optimizer = torch.optim.AdamW(
        [{"params": model.parameters(), "lr": args.lr}, 
        {"params": [loss_fn.temperature], "lr": args.lr}],
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.05
    )
    scheduler = cosine_lr(
        optimizer, 
        base_lr=args.lr,         
        warmup_length=10,     
        steps=(epochs * num_batches)
    )

    print("\n\t Started Training\n")
    best_val_loss = None
    for epoch in range(epochs):

        begin = time.time()

        ###Training
        loss = train_one_epoch(trainloader, model, optimizer, loss_fn, scheduler, num_of_steps)
        ###Validation
        val_loss = val_one_epoch(testloader, model, loss_fn)
        if best_val_loss == None: 
            best_val_loss = val_loss
        num_of_steps += num_batches

        print('\n\n\t Epoch....', epoch + 1)
        print("\t Training loss ......",round(loss,4))
        print("\t Val loss ......", round(val_loss,4))
        print('\t Time per epoch (in mins) = ', round((time.time()-begin)/60,2),'\n\n')
        log_metrics(epoch + 1, loss, val_loss, optimizer.param_groups[0]['lr'], loss_fn.temperature.exp().item())

        if val_loss <= best_val_loss:
            torch.save(model.state_dict(),os.path.join(save_path, 'model_epoch_{}.pth'.format(epoch)))
            best_val_loss = val_loss

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4, help='lr')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--mri_csv_path', type=str, default='', help='Path to MRI CSV file')
    parser.add_argument('--cropped_mri_path', type=str, default='', help='Path to cropped MRI file')
    parser.add_argument('--ecg_tsv_path', type=str, default='', help='Path to ECG TSV file')
    parser.add_argument('--save_path', type=str, default='', help='Path to save models')
    parser.add_argument('--pt_mri_path', type=str, default='', help='Path to pretrained MRI model')
    parser.add_argument('--pt_ecg_path', type=str, default='', help='Path to pretrained ECG model')

    args = parser.parse_args()
    wandb.init(
        project="multi-modal-training",
        name=f"Training-bs{args.batch_size}-lr{args.lr:g}-ep{args.epochs}"
    )
    train_clip(args)

# python stage1_CL.py --lr 1e-4 --epochs 20 --batch_size 32 