"""
ECG-only REGRESSION fine-tune for the cardiac-MRI feature task (paper p6 / UKBB_R2_Corr.Rmd):
predict LVEF, LVM, LVEDV, LVESV, LAEF, LAVmin, LAVmax from ECG alone.

Same model/optimizer/schedule as `ecg_finetune.py` (binary) -- the model needs NO change, since
`CARDIACFM_ECG.pred` is already a 1-output linear layer and the sigmoid lives in the training loop.
Only the loop differs:
  * NO sigmoid  -> raw linear output
  * nn.MSELoss  instead of nn.BCELoss
  * model selection on validation Pearson r (higher = better) instead of AUROC
  * TARGET STANDARDIZATION (important): the features live on wildly different scales
    (LVEF 0-100, LVM 0-288, LVEDV 0-481). At the paper's lr=5e-6, raw-scale MSE gives
    mismatched gradient magnitudes per feature and the head barely moves. We z-score the target,
    train on z, and save mu/sd so predictions can be inverted to the original scale at eval.
    (Pearson r is scale-invariant; R^2 = 1 - MSE/Var must be computed on the ORIGINAL scale.)
    mu/sd are computed from the full label vector -- they are normalization constants, not fitted
    parameters, and train-only vs all-sample estimates agree to ~3 decimals at n~57k.

Usage: python ecg_finetune_reg.py --feature lvm --model_name CARDIACFM --cardiacfm_pretrained_ckpt ...
"""
import os, sys, time, math, random, argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # cardiacfm_new/
sys.path.insert(0, os.path.join(_ROOT, "common", "ecg_encoder"))   # model_ecg.py
sys.path.insert(0, os.path.join(_ROOT, "common", "data"))          # ecg_dataset.py
from model_ecg import ECGFM, CARDIACFM_ECG, cosine_lr
from ecg_dataset import ECGDataset


def _pearson(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def train_one_epoch(loader, model, optimizer, loss_fn, device, scheduler, num_of_steps, mu, sd):
    epoch_loss = []
    model.train()
    begin = time.time()
    for i, ecgs in enumerate(loader):
        if scheduler is not None:
            scheduler(i + num_of_steps)
        optimizer.zero_grad()
        ecgs["net_input"]["source"] = ecgs["net_input"]["source"].to(device)
        pred = model(ecgs)                       # raw linear output (NO sigmoid)
        labels = ecgs["label"].to(device).float()
        if torch.isnan(labels).all():
            continue
        mask = ~torch.isnan(labels)
        pred, labels = pred[mask], labels[mask]
        labels_z = (labels - mu) / sd            # standardized target
        _loss = loss_fn(pred, labels_z)
        epoch_loss.append(_loss)
        _loss.backward()
        optimizer.step()
        if i % 50 == 0:
            el = time.time() - begin
            print(f"train_loss = {_loss.item():.4f}  [{i}/{len(loader)}]  "
                  f"elapsed {el:.0f}s est-remaining {el*(len(loader)-i-1)/(i+1):.0f}s", flush=True)
    return float(np.mean([l.item() for l in epoch_loss])) if epoch_loss else np.nan


def val_one_epoch(loader, model, loss_fn, device, mu, sd):
    epoch_loss, yt, yp = [], [], []
    model.eval()
    with torch.no_grad():
        for ecgs in loader:
            ecgs["net_input"]["source"] = ecgs["net_input"]["source"].to(device)
            pred = model(ecgs)
            labels = ecgs["label"].to(device).float()
            if torch.isnan(labels).all():
                continue
            mask = ~torch.isnan(labels)
            pred, labels = pred[mask], labels[mask]
            labels_z = (labels - mu) / sd
            epoch_loss.append(loss_fn(pred, labels_z))
            yt.extend(labels.cpu().numpy().ravel())
            yp.extend((pred.cpu().numpy().ravel() * sd + mu))     # back to original scale
    yt, yp = np.array(yt), np.array(yp)
    mse = float(np.mean((yt - yp) ** 2))
    r2 = 1 - mse / float(np.var(yt))                              # variance-explained, paper's def
    return (float(np.mean([l.item() for l in epoch_loss])) if epoch_loss else np.nan,
            _pearson(yt, yp), r2)


def train(batch_size, epochs, args):
    label_path = f"{args.label_dir}/y.npy"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Y = np.load(label_path).squeeze().astype(float)
    mu, sd = float(np.nanmean(Y)), float(np.nanstd(Y))
    print(f"device: {device}  feature={args.feature}  target mu={mu:.3f} sd={sd:.3f}  "
          f"n_labeled={int((~np.isnan(Y)).sum())}", flush=True)

    trainset = ECGDataset(args.ecg_tsv_dir, label_path, split="train")
    validset = ECGDataset(args.ecg_tsv_dir, label_path, split="valid")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4,
                             collate_fn=ECGDataset.collate_fn)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=4,
                             collate_fn=ECGDataset.collate_fn)

    if "ECGFM" in args.model_name:
        model = ECGFM(ecgfm_ckpt=args.ecgfm_ckpt)
    else:
        model = CARDIACFM_ECG(ecgfm_ckpt=args.ecgfm_ckpt,
                              cardiacfm_pretrained_ckpt=args.cardiacfm_pretrained_ckpt)
    model.to(device)
    print("\t Total Params =", sum(p.numel() for p in model.parameters()), flush=True)

    num_batches = math.ceil(len(trainset) // batch_size)
    loss_fn = nn.MSELoss()
    if "ECGFM" in args.model_name:
        groups = [{"params": model.pred.parameters(), "lr": 1e-4},
                  {"params": model.ecg_encoder.parameters(), "lr": 1e-5}]
    else:
        groups = [{"params": model.pred.parameters(), "lr": 1e-4},
                  {"params": model.ecg_projection_multi.parameters(), "lr": 1e-4},
                  {"params": model.ecg_encoder_multi.parameters(), "lr": 1e-5}]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-6, weight_decay=1e-2)
    scheduler = cosine_lr(optimizer, base_lr=args.base_lr, warmup_length=50,
                          steps=epochs * num_batches)

    os.makedirs(args.save_dir, exist_ok=True)
    json.dump({"feature": args.feature, "mu": mu, "sd": sd},
              open(os.path.join(args.save_dir, "target_scaling.json"), "w"))

    best_r, patience, bad, steps = -np.inf, 3, 0, 0
    for epoch in range(epochs):
        t0 = time.time()
        tr = train_one_epoch(trainloader, model, optimizer, loss_fn, device, scheduler, steps, mu, sd)
        vl, vr, vr2 = val_one_epoch(validloader, model, loss_fn, device, mu, sd)
        steps += num_batches
        print(f"\n\t Epoch {epoch+1}  train_loss={tr:.4f}  val_loss={vl:.4f}  "
              f"val_r={vr:.4f}  val_R2={vr2:.4f}  ({(time.time()-t0)/60:.1f} min)\n", flush=True)
        if np.isfinite(vr) and vr > best_r:
            best_r, bad = vr, 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"epoch_{epoch}.pth"))
        else:
            bad += 1
        if bad >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}. Best val r = {best_r:.4f}\n", flush=True)
            break
    print(f"DONE feature={args.feature} best_val_r={best_r:.4f}", flush=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(s)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feature", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--model_name", default="CARDIACFM")
    p.add_argument("--label_dir", required=True)
    p.add_argument("--ecg_tsv_dir", default="")
    p.add_argument("--ecgfm_ckpt", default="")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--cardiacfm_pretrained_ckpt", default=None)
    p.add_argument("--base_lr", type=float, default=5e-6)
    a = p.parse_args()
    set_seed(a.seed)
    train(batch_size=4, epochs=a.epochs, args=a)
