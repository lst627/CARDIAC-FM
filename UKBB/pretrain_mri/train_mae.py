"""
Pretraining for the multi-view cross-fusion MAE (CineMAE).

Unlike the earlier mri_new script — which called a shared encoder once per view and
averaged the losses externally — here a SINGLE forward consumes all three views,
fuses them in the shared encoder, and returns the averaged reconstruction loss
plus per-view parts for logging.

Usage (single GPU):
  python train_mae.py --data_dir /path/to/cropped_new \
      --train_split_dir ... --val_split_dir ...

Usage (multi-GPU):
  torchrun --nproc_per_node=4 train_mae.py ...
"""

import argparse
import csv
import logging
import math
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# --- self-contained imports: add the in-repo module dirs to sys.path ---
_ROOT = Path(__file__).resolve().parents[2]                 # cardiacfm_new/
sys.path.insert(0, str(_ROOT / "common" / "mri_encoder"))   # cinema_mae.py, utils/misc.py
sys.path.insert(0, str(_ROOT / "common" / "data"))          # mri_dataset.py
from mri_dataset import CardiacMRIDataset
from cinema_mae import CineMAE
from utils.misc import set_seed, save_checkpoint, load_checkpoint, AverageMeter


# ─────────────────────────── distributed helpers ─────────────────────────────

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def is_main():
    return get_rank() == 0

def setup_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); logger.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="w"); fh.setFormatter(fmt); logger.addHandler(fh)

def log_main(msg):
    if is_main():
        logging.info(msg)

def plot_losses(train_losses, val_losses, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train", marker="o", markersize=3)
    ax.plot(val_losses,   label="val",   marker="o", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss")
    ax.set_title("CineMAE Pretraining Loss"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

def _eids_from_split_dir(split_dir):
    eids = set()
    for csv_path in Path(split_dir).glob("*.csv"):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                eids.add(str(row["eid_visit"]))
    if not eids:
        raise RuntimeError(f"No EIDs found in {split_dir}")
    return eids


# ─────────────────────────── train / validate ────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, accum_steps):
    model.train()
    meters = {k: AverageMeter() for k in ("total", "2ch", "4ch", "sa")}
    t0 = time.time()
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        v2ch = batch["v2ch"].to(device)   # (B,1,T,H,W)
        v4ch = batch["v4ch"].to(device)   # (B,1,T,H,W)
        vsa  = batch["vsa"].to(device)    # (B,3,T,H,W)
        B    = v2ch.shape[0]

        with torch.cuda.amp.autocast():
            loss, parts = model(v2ch, v4ch, vsa)
            loss = loss / accum_steps

        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

        meters["total"].update(loss.item() * accum_steps, B)
        for v in ("2ch", "4ch", "sa"):
            meters[v].update(parts[v].item(), B)

        if is_main() and step % 20 == 0:
            log_main(f"  Epoch {epoch:3d} | step {step:4d}/{len(loader)} | "
                     f"loss {meters['total'].avg:.4f} | "
                     f"2ch {meters['2ch'].avg:.4f} | 4ch {meters['4ch'].avg:.4f} | "
                     f"sa {meters['sa'].avg:.4f} | {time.time()-t0:.0f}s")

    return meters["total"].avg


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    meter = AverageMeter()
    for batch in loader:
        v2ch = batch["v2ch"].to(device)
        v4ch = batch["v4ch"].to(device)
        vsa  = batch["vsa"].to(device)
        with torch.cuda.amp.autocast():
            loss, _ = model(v2ch, v4ch, vsa)
        meter.update(loss.item(), v2ch.shape[0])
    return meter.avg


# ─────────────────────────── main ────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",        type=str, required=True)
    p.add_argument("--train_split_dir", type=str, required=True)
    p.add_argument("--val_split_dir",   type=str, required=True)
    p.add_argument("--save_dir",        type=str, default="./checkpoints")
    p.add_argument("--log_dir",         type=str, default="./logs")
    p.add_argument("--log_name",        type=str, default="pretrain_cinema.log")
    # model
    p.add_argument("--img_size",      type=int,   default=112)
    p.add_argument("--n_frames",      type=int,   default=8)
    p.add_argument("--tube_t",        type=int,   default=2)
    p.add_argument("--patch_size",    type=int,   default=16)
    p.add_argument("--encoder_dim",   type=int,   default=384)
    p.add_argument("--encoder_depth", type=int,   default=8)
    p.add_argument("--encoder_heads", type=int,   default=6)
    p.add_argument("--decoder_dim",   type=int,   default=192)
    p.add_argument("--decoder_depth", type=int,   default=4)
    p.add_argument("--decoder_heads", type=int,   default=6)
    p.add_argument("--mask_ratio",    type=float, default=0.9)   # higher for video/tube masking
    p.add_argument("--norm_pix_loss", action="store_true", default=True)
    p.add_argument("--view_encoder",  type=str, default="none", choices=["none", "conv", "vit"],
                   help="per-view stage: none=Conv3d patchify, conv=ConvMAE stem (CineMA), vit=per-view ViT")
    p.add_argument("--view_depth",    type=int, default=2, help="per-view ViT depth (view_encoder=vit)")
    p.add_argument("--n_sa_slices",   type=int, default=3, help="number of short-axis slices")
    # training
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--accum_steps",  type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1.5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs",type=int,   default=5)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--resume",       type=str,   default=None)
    p.add_argument("--patience",     type=int,   default=7,
                   help="early stop if val loss does not improve for this many epochs (<=0 disables)")
    p.add_argument("--seed",         type=int,   default=42)
    args = p.parse_args()

    # ── distributed setup ────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed + get_rank())

    if is_main():
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        setup_logging(os.path.join(args.log_dir, args.log_name))

    log_main(f"Device: {device}  |  World: {get_world_size()}")
    log_main(f"Effective batch: {args.batch_size * args.accum_steps * get_world_size()}")
    log_main(f"Args: {vars(args)}")

    # ── data ───────────────────────────────────────────────────────────────
    train_ds = CardiacMRIDataset(args.data_dir, augment=True,
                                 subject_list=_eids_from_split_dir(args.train_split_dir),
                                 spatial_augment=False, img_size=args.img_size)
    val_ds   = CardiacMRIDataset(args.data_dir, augment=False,
                                 subject_list=_eids_from_split_dir(args.val_split_dir),
                                 spatial_augment=False, img_size=args.img_size)
    log_main(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_sampler = DistributedSampler(train_ds) if is_dist() else None
    train_loader  = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                               shuffle=(train_sampler is None), num_workers=args.num_workers,
                               pin_memory=True, drop_last=True)
    val_loader    = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

    # ── model ───────────────────────────────────────────────────────────────
    model = CineMAE(
        img_size=args.img_size, n_frames=args.n_frames,
        tube_t=args.tube_t, patch_h=args.patch_size, patch_w=args.patch_size,
        encoder_dim=args.encoder_dim, encoder_depth=args.encoder_depth, encoder_heads=args.encoder_heads,
        decoder_dim=args.decoder_dim, decoder_depth=args.decoder_depth, decoder_heads=args.decoder_heads,
        mask_ratio=args.mask_ratio, norm_pix_loss=args.norm_pix_loss,
        view_encoder=args.view_encoder, view_depth=args.view_depth,
        n_sa_slices=args.n_sa_slices,
    ).to(device)

    if is_main():
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        keep = max(1, int(round((model.grid[1] * model.grid[2]) * (1 - args.mask_ratio)))) * model.grid[0]
        log_main(f"Model params: {n_params:.1f}M")
        log_main(f"Tokens/view: {model.N}  visible/view after tube mask: {keep}  "
                 f"(streams: 2ch + 4ch + 3xSA = 5)")

    if is_dist():
        model = DDP(model, device_ids=[local_rank])

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        t = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * t))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.cuda.amp.GradScaler()

    # ── resume ───────────────────────────────────────────────────────────────
    start_epoch, best_loss = 0, float("inf")
    if args.resume and os.path.isfile(args.resume):
        m = model.module if is_dist() else model
        start_epoch, best_loss = load_checkpoint(args.resume, m, optimizer, device)
        log_main(f"Resumed from {args.resume} (epoch {start_epoch})")

    # ── loop ─────────────────────────────────────────────────────────────────
    log_main(f"\nStarting CineMAE pretraining for {args.epochs} epochs ...\n")
    plot_path = os.path.join(args.log_dir, args.log_name.replace(".log", "_loss.png"))
    train_losses, val_losses = [], []
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args.accum_steps)
        val_loss   = validate(model, val_loader, device)
        scheduler.step()

        log_main(f"Epoch {epoch:3d}/{args.epochs} | train={train_loss:.4f} | "
                 f"val={val_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

        # early stopping: val_loader is not sharded, so val_loss is identical on
        # all ranks → every rank decides the same and breaks together.
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if is_main():
            train_losses.append(train_loss); val_losses.append(val_loss)
            plot_losses(train_losses, val_losses, plot_path)
            m = model.module if is_dist() else model
            state = {"epoch": epoch + 1, "model": m.state_dict(),
                     "optimizer": optimizer.state_dict(), "best_loss": best_loss,
                     "args": vars(args)}
            save_checkpoint(state, os.path.join(args.save_dir, "cinema_latest.pth"))
            if is_best:
                save_checkpoint(state, os.path.join(args.save_dir, "cinema_best.pth"))
                log_main(f"  ✓ New best val={best_loss:.4f}")

        if args.patience > 0 and epochs_no_improve >= args.patience:
            log_main(f"Early stopping at epoch {epoch} "
                     f"(no val improvement for {args.patience} epochs; best={best_loss:.4f})")
            break

    log_main(f"CineMAE pretraining complete. Best val={best_loss:.4f}")
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
