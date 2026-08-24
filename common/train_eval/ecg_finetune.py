"""
ECG-only fine-tune for CHS/MESA external validation. ECG models and cosine_lr come
from the local model_ecg module; the ECG dataset is loaded by putting the repository
modules on sys.path. This keeps the ECG-only path independent of unused modalities.
"""
import os
import sys
import time
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

# --- local ECG models ---
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
sys.path.insert(0, os.path.join(_ROOT, "common", "ecg_encoder"))   # model_ecg.py
sys.path.insert(0, os.path.join(_ROOT, "common", "data"))          # ecg_dataset.py (vendored)
from model_ecg import ECGFM, CARDIACFM_ECG, cosine_lr
from ecg_dataset import ECGDataset


def train_one_epoch(train_data_loader, model, optimizer, loss_fn, device, scheduler, num_of_steps):
    epoch_loss = []
    model.train()
    begin = time.time()
    for i, ecgs in enumerate(train_data_loader):
        if scheduler is not None:
            scheduler(i + num_of_steps)
        optimizer.zero_grad()
        ecgs["net_input"]["source"] = ecgs["net_input"]["source"].to(device)
        ecg_features = model(ecgs)
        y_pred = torch.sigmoid(ecg_features)
        labels = ecgs["label"].to(device).float()
        if torch.isnan(labels).all():   # skip all-NaN-label batches (else BCELoss(empty)=NaN)
            continue
        mask = ~torch.isnan(labels)
        y_pred, labels = y_pred[mask], labels[mask]
        _loss = loss_fn(y_pred, labels)
        epoch_loss.append(_loss)
        _loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("train_loss = ", _loss.item())
            elapsed_time = time.time() - begin
            estimated_total_time = elapsed_time * (len(train_data_loader) - i - 1) / (i + 1)
            print(f"Elapsed time: {elapsed_time:.2f}s, Estimated remaining time: {estimated_total_time:.2f}s")
    return np.mean([l.cpu().item() for l in epoch_loss])


def val_one_epoch(val_data_loader, model, loss_fn, device):
    epoch_loss, y_true_all, y_pred_all = [], [], []
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
            epoch_loss.append(_loss)
            y_true_all.extend(labels.cpu().numpy())
            y_pred_all.extend(y_pred.cpu().numpy())
    return np.mean([l.cpu().item() for l in epoch_loss]), roc_auc_score(y_true_all, y_pred_all)


def train_clip(batch_size, epochs, args):
    ecgs_tsv_dir = args.ecg_tsv_dir
    label_dir = f"{args.label_dir}/y.npy"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainset = ECGDataset(ecgs_tsv_dir, label_dir, split="train")
    validset = ECGDataset(ecgs_tsv_dir, label_dir, split="valid")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=ECGDataset.collate_fn)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=ECGDataset.collate_fn)

    if "ECGFM" in args.model_name:
        print("\nLoading ECGFM model.....")
        model = ECGFM(ecgfm_ckpt=args.ecgfm_ckpt)
    elif "CARDIACFM" in args.model_name:
        print("\nLoading CARDIACFM model.....")
        model = CARDIACFM_ECG(ecgfm_ckpt=args.ecgfm_ckpt, cardiacfm_pretrained_ckpt=args.cardiacfm_pretrained_ckpt)
    model.to(device)

    if args.finetuned_ckpt != '':
        checkpoint = torch.load(args.finetuned_ckpt, weights_only=False)
        model.load_state_dict(checkpoint)

    # Optional linear-probe: freeze the ECG encoder, train only projection + head.
    if args.freeze_encoder:
        enc = model.ecg_encoder_multi if "CARDIACFM" in args.model_name else model.ecg_encoder
        for p in enc.parameters():
            p.requires_grad = False
        print("\t [freeze_encoder] ECG encoder frozen -> linear probe (projection + head only)")

    print("\n\t Model Loaded")
    print("\t Total Params = ", sum(p.numel() for p in model.parameters()))
    print("\t Trainable Params = ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    num_batches = math.ceil(len(trainset) // batch_size)
    num_of_steps = 0
    loss_fn = nn.BCELoss()

    if "ECGFM" in args.model_name:
        groups = [{"params": model.pred.parameters(), "lr": 1e-4}]
        if not args.freeze_encoder:
            groups.append({"params": model.ecg_encoder.parameters(), "lr": 1e-5})
    elif "CARDIACFM" in args.model_name:
        groups = [{"params": model.pred.parameters(), "lr": 1e-4},
                  {"params": model.ecg_projection_multi.parameters(), "lr": 1e-4}]
        if not args.freeze_encoder:
            groups.append({"params": model.ecg_encoder_multi.parameters(), "lr": 1e-5})
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-6, weight_decay=1e-2)

    # NOTE: cosine_lr overwrites every param group to the SAME base_lr schedule (copied verbatim
    # from CARDIAC-FM). So base_lr is effectively THE lr for all trainable params. For a linear
    # probe use a larger --base_lr (e.g. 1e-3); the default 5e-6 reproduces the original full-FT.
    scheduler = cosine_lr(optimizer, base_lr=args.base_lr, warmup_length=50, steps=epochs * num_batches)

    print("\n\t Started Training\n")
    best_val_auc = -1
    patience = 3
    patience_counter = 0

    for epoch in range(epochs):
        begin = time.time()
        train_loss = train_one_epoch(trainloader, model, optimizer, loss_fn, device, scheduler, num_of_steps)
        val_loss, val_auc = val_one_epoch(validloader, model, loss_fn, device)
        num_of_steps += num_batches
        print('\n\t Epoch....', epoch + 1)
        print("\t Training loss ......", round(train_loss, 4))
        print("\t Val loss ......", round(val_loss, 4))
        print("\t Val auc ......", round(val_auc, 4))
        print('\t Time per epoch (in mins) = ', round((time.time() - begin) / 60, 2), '\n')

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--model_name', type=str, help='Model Name, ECGFM or CARDIACFM')
    parser.add_argument('--label_dir', type=str, help='Path to downstream task label')
    parser.add_argument('--ecg_tsv_dir', type=str, default='', help='Path to ecgs tsv dir')
    parser.add_argument('--ecgfm_ckpt', type=str, default='', help='Path to ecgfm model')
    parser.add_argument('--save_dir', type=str, help='Path to dir used to save model')
    parser.add_argument('--cardiacfm_pretrained_ckpt', type=str, default=None,
                        help='contrastive pretrained weight (init encoder+projection from CL)')
    parser.add_argument('--finetuned_ckpt', type=str, default='',
                        help='continue from a model already fine-tuned on UKB')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='linear probe: freeze the ECG encoder, train only projection + head')
    parser.add_argument('--base_lr', type=float, default=5e-6,
                        help='cosine base_lr (== effective lr for all trainable params); use ~1e-3 for a probe')
    args = parser.parse_args()
    set_seed(args.seed)
    train_clip(batch_size=4, epochs=args.epochs, args=args)
