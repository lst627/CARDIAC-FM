"""
Stage 1 contrastive pretraining: CineMA (cross-fusion) MRI encoder + ECG-FM encoder.

Same pipeline as stage1_CL_mae.py, but the MRI tower is the new CineMAE, whose
cross-view fusion is INTERNAL — it returns a single fused (B, embed_dim) vector
instead of three per-view embeddings. So the MRI projection takes embed_dim
(not 3*embed_dim) as input. Everything else (ECG-FM encoder, InfoNCE, the paired
ECG/MRI dataset, the training loop) is unchanged.

Usage:
    python stage1_CL_cinema.py --lr 1e-5 \
        --mri_dir /gpfs/projects/trend/data/UKBB/MRI/cropped_new \
        --ecg_dir /gpfs/projects/trend/bojun/mri/outcome/data_train_valid_test_individual/stage1/ecg_tsv \
        --csv_train .../mri_train.csv --csv_val .../mri_valid.csv \
        --mae_ckpt /gpfs/projects/trend/bojun/mri/CineMA/checkpoints/cinema_224_100ep_mg/cinema_best.pth \
        --ecg_ckpt /gpfs/projects/trend/bojun/multimodal/mimic_iv_ecg_physionet_pretrained.pt \
        --out_dir  /gpfs/projects/trend/bojun/mri/CineMA/checkpoints/stage1_cinema
"""

import os
import sys
import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from fairseq_signals.models import build_model_from_checkpoint
from fairseq_signals.data.ecg.raw_ecg_dataset import FileECGDataset

# CineMA cross-fusion MRI encoder (returns a single fused embedding)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # cardiacfm_new/
sys.path.insert(0, os.path.join(_ROOT, "common", "mri_encoder"))
from encoder import CineMAEncoder


# ── lr schedule ──────────────────────────────────────────────────────────────

def cosine_lr(optimizer, warmup_length, steps):
    # Capture each param group's base LR so per-group LRs (e.g. a smaller LR on a
    # pretrained encoder) are preserved — the schedule scales every group by the
    # same cosine factor relative to its OWN base, instead of overwriting them all.
    initial_lrs = [g["lr"] for g in optimizer.param_groups]

    def _lr_adjuster(step):
        if step < warmup_length:
            factor = (step + 1) / warmup_length
        else:
            e = step - warmup_length
            es = steps - warmup_length
            factor = 0.5 * (1 + math.cos(math.pi * e / es))
        for g, lr0 in zip(optimizer.param_groups, initial_lrs):
            g["lr"] = lr0 * factor
    return _lr_adjuster


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(np.log(1 / temperature)))

    def forward(self, z1, z2):
        """z1, z2: (B, D) L2-normalized embeddings."""
        B = z1.shape[0]
        temp = self.temperature.clamp(max=math.log(100)).exp()
        logits = (z1 @ z2.T) * temp          # (B, B)
        labels = torch.arange(B, device=z1.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss


# ── combined model ────────────────────────────────────────────────────────────

class CineMAECGModel(nn.Module):
    def __init__(self, mae_ckpt, ecg_ckpt, embed_dim=384, img_size=224, n_frames=8,
                 mri_tune="partial", unfreeze_blocks=2, ecg_unfreeze=2, pool="mean",
                 view_encoder="none", view_depth=2, n_sa_slices=3,
                 encoder_depth=8, encoder_heads=6):
        """
        mri_tune     : 'full'    = fine-tune the entire MRI encoder;
                       'partial' = freeze all but the last `unfreeze_blocks`
                                   encoder blocks + enc_norm.
        unfreeze_blocks: how many top encoder blocks to unfreeze when mri_tune='partial'.
        ecg_unfreeze : number of ECG transformer layers to unfreeze from the top.
                       -1 = unfreeze all layers; N = unfreeze last N layers only.
        pool         : how the CineMA encoder pools its fused tokens ('mean' or 'cls').
        """
        super().__init__()
        assert mri_tune in ("full", "partial")

        # MRI encoder: CineMA cross-fusion ViT (config must match pretraining)
        mae_cfg = dict(
            img_size=img_size, n_frames=n_frames,
            tube_t=2, patch_h=16, patch_w=16,
            encoder_dim=embed_dim, encoder_depth=encoder_depth, encoder_heads=encoder_heads,
            decoder_dim=192, decoder_depth=4, decoder_heads=6,
            mask_ratio=0.9, norm_pix_loss=True,
            view_encoder=view_encoder, view_depth=view_depth, n_sa_slices=n_sa_slices,
        )
        # full -> nothing frozen; partial -> CineMAEncoder freezes all but enc_norm,
        # then we re-enable the last `unfreeze_blocks` encoder blocks.
        self.mri_encoder = CineMAEncoder(mae_cfg=mae_cfg, ckpt=mae_ckpt,
                                         freeze=(mri_tune == "partial"), pool=pool)
        if mri_tune == "partial":
            n_enc = len(self.mri_encoder.mae.enc_blocks)
            for i, block in enumerate(self.mri_encoder.mae.enc_blocks):
                if i >= n_enc - unfreeze_blocks:
                    for p in block.parameters():
                        p.requires_grad_(True)
            for p in self.mri_encoder.mae.enc_norm.parameters():
                p.requires_grad_(True)

        # ECG encoder: ECG-FM foundation model
        self.ecg_encoder = build_model_from_checkpoint(ecg_ckpt)
        for p in self.ecg_encoder.parameters():
            p.requires_grad_(False)
        n_layers = len(self.ecg_encoder.encoder.layers)
        for i, layer in enumerate(self.ecg_encoder.encoder.layers):
            if ecg_unfreeze == -1 or i >= n_layers - ecg_unfreeze:
                for p in layer.parameters():
                    p.requires_grad_(True)
        for p in self.ecg_encoder.encoder.layer_norm.parameters():
            p.requires_grad_(True)

        # projection heads → 512-dim shared space
        # MRI readout dim depends on pooling: 'mean'/'cls' -> embed_dim, 'per_view' -> 3*embed_dim.
        mri_dim = self.mri_encoder.embed_dim
        self.mri_projection = nn.Sequential(
            nn.LayerNorm(mri_dim),
            nn.Dropout(0.1),
            nn.Linear(mri_dim, 512),
        )
        self.ecg_projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
        )

    def forward_mri(self, v2ch, v4ch, vsa):
        fused = self.mri_encoder(v2ch, v4ch, vsa)        # (B, embed_dim)
        proj  = self.mri_projection(fused)               # (B, 512)
        return F.normalize(proj, dim=-1)

    def forward_ecg(self, ecgs):
        feats = self.ecg_encoder.extract_features(
            source=ecgs["net_input"]["source"],
            padding_mask=ecgs["net_input"]["padding_mask"])
        x = feats["x"]                                   # (B, T, D)
        pad_mask = feats.get("padding_mask", ecgs["net_input"]["padding_mask"])
        if pad_mask is not None:
            non_pad = ~pad_mask
            x = (x * non_pad.unsqueeze(-1).float()).sum(dim=1) / non_pad.float().sum(dim=1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(dim=1)
        proj = self.ecg_projection(x)                    # (B, 512)
        return F.normalize(proj, dim=-1)

    def forward(self, mris, ecgs):
        return (self.forward_mri(mris["v2ch"], mris["v4ch"], mris["vsa"]),
                self.forward_ecg(ecgs))


# ── dataset ───────────────────────────────────────────────────────────────────

class ECGMRIDataset(Dataset):
    """
    Loads paired ECG (fairseq-signals TSV manifest) and MRI (vst_*.npy).
    Only subjects present in the CSV, the ECG manifest, AND with MRI are kept.
    """
    def __init__(self, csv_path, ecg_dir, mri_dir, split="train",
                 img_size=224, n_frames=8, mri_augment=False):
        self.mri_dir     = mri_dir
        self.img_size    = img_size
        self.n_frames    = n_frames
        self.augment     = (split == "train")
        self.mri_augment = mri_augment and (split == "train")

        manifest_path = os.path.join(ecg_dir, f"{split}.tsv")
        self.ecg_data = FileECGDataset(
            manifest_path=manifest_path,
            sample_rate=None, max_sample_size=None, min_sample_size=None,
            pad=True, pad_leads=False, leads_to_load=None,
            label=False, filter=False, normalize=False,
            mean_path=None, std_path=None, num_buckets=0,
            compute_mask_indices=False, leads_bucket=None,
            bucket_selection="uniform",
            training=(split == "train"),
        )

        ecg_eids = set()
        with open(manifest_path) as f:
            f.readline()  # skip root line
            for line in f:
                fname = line.strip().split("\t")[0]
                ecg_eids.add(os.path.splitext(fname)[0])

        df = pd.read_csv(csv_path)
        df["eid_visit"] = df["eid_visit"].astype(str)
        csv_eids = set(df["eid_visit"].tolist())

        mri_eids = {
            d for d in os.listdir(mri_dir)
            if os.path.isdir(os.path.join(mri_dir, d)) and
            all(os.path.exists(os.path.join(mri_dir, d, f))
                for f in ["vst_2ch.npy", "vst_4ch.npy", "vst_sa.npy"])
        }

        valid_eids = sorted(ecg_eids & csv_eids & mri_eids)
        print(f"[{split}] ECG: {len(ecg_eids)}  CSV: {len(csv_eids)}  "
              f"MRI: {len(mri_eids)}  paired: {len(valid_eids)}")

        ecg_eid_to_idx = {}
        with open(manifest_path) as f:
            f.readline()
            for i, line in enumerate(f):
                fname = line.strip().split("\t")[0]
                ecg_eid_to_idx[os.path.splitext(fname)[0]] = i

        self.samples = [(eid, ecg_eid_to_idx[eid]) for eid in valid_eids if eid in ecg_eid_to_idx]

    def __len__(self):
        return len(self.samples)

    def _crop_frames(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        if T <= self.n_frames:
            return x
        start = torch.randint(0, T - self.n_frames + 1, (1,)).item() if self.augment \
                else (T - self.n_frames) // 2
        return x[:, start:start + self.n_frames]

    def _load_mri(self, eid):
        subj = os.path.join(self.mri_dir, eid)
        v2ch = torch.from_numpy(np.load(os.path.join(subj, "vst_2ch.npy")).astype(np.float32))
        v4ch = torch.from_numpy(np.load(os.path.join(subj, "vst_4ch.npy")).astype(np.float32))
        vsa  = torch.from_numpy(np.load(os.path.join(subj, "vst_sa.npy")).astype(np.float32))

        def resize(x):
            S, T, H, W = x.shape
            if H == self.img_size and W == self.img_size:
                return x
            x = x.reshape(S * T, 1, H, W)
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
            return x.reshape(S, T, self.img_size, self.img_size)

        v2ch = self._crop_frames(resize(v2ch))
        v4ch = self._crop_frames(resize(v4ch))
        vsa  = self._crop_frames(resize(vsa))
        if self.mri_augment:
            if torch.rand(1) < 0.5:
                v2ch = v2ch.flip(-1); v4ch = v4ch.flip(-1); vsa = vsa.flip(-1)
            scale = 0.9 + 0.2 * torch.rand(1)
            v2ch, v4ch, vsa = v2ch * scale, v4ch * scale, vsa * scale
        return {"v2ch": v2ch, "v4ch": v4ch, "vsa": vsa}

    def __getitem__(self, idx):
        eid, ecg_idx = self.samples[idx]
        ecg = self.ecg_data[ecg_idx]
        mri = self._load_mri(eid)
        return mri, ecg

    @staticmethod
    def collate_fn(batch):
        mri_batch = {k: torch.stack([b[0][k] for b in batch]) for k in batch[0][0]}
        ecg_list  = [b[1] for b in batch]
        sources   = [s["source"] for s in ecg_list]
        max_len = max(s.shape[-1] for s in sources)
        padded  = torch.zeros(len(sources), sources[0].shape[0], max_len)
        masks   = torch.ones(len(sources), max_len, dtype=torch.bool)
        for i, s in enumerate(sources):
            padded[i, :, :s.shape[-1]] = s
            masks[i, :s.shape[-1]] = False
        ecg_collated = {"net_input": {"source": padded, "padding_mask": masks}}
        return mri_batch, ecg_collated


# ── training ──────────────────────────────────────────────────────────────────

def train_one_epoch(loader, model, optimizer, loss_fn, device, scheduler, step_offset):
    model.train()
    losses = []; t0 = time.time()
    for i, (mris, ecgs) in enumerate(loader):
        if scheduler is not None:
            scheduler(i + step_offset)
        mris = {k: v.to(device) for k, v in mris.items()}
        ecgs["net_input"]["source"]       = ecgs["net_input"]["source"].to(device)
        ecgs["net_input"]["padding_mask"] = ecgs["net_input"]["padding_mask"].to(device)

        optimizer.zero_grad()
        mri_feat, ecg_feat = model(mris, ecgs)   # fp32: bf16 autocast NaN'd on base + full ECG
        loss = loss_fn(mri_feat, ecg_feat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # global-norm clip (CineMA uses 5.0)
        optimizer.step()
        losses.append(loss.item())

        if i % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (len(loader) - i - 1) / (i + 1)
            print(f"  step {i}/{len(loader)}  loss={loss.item():.4f}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)
    return np.mean(losses)


@torch.no_grad()
def val_one_epoch(loader, model, loss_fn, device):
    model.eval()
    losses = []
    for mris, ecgs in loader:
        mris = {k: v.to(device) for k, v in mris.items()}
        ecgs["net_input"]["source"]       = ecgs["net_input"]["source"].to(device)
        ecgs["net_input"]["padding_mask"] = ecgs["net_input"]["padding_mask"].to(device)
        mri_feat, ecg_feat = model(mris, ecgs)
        losses.append(loss_fn(mri_feat, ecg_feat).item())
    return np.mean(losses)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mri_dir",   required=True)
    parser.add_argument("--ecg_dir",   required=True)
    parser.add_argument("--csv_train", required=True)
    parser.add_argument("--csv_val",   required=True)
    parser.add_argument("--mae_ckpt",  required=True)
    parser.add_argument("--ecg_ckpt",  required=True)
    parser.add_argument("--out_dir",   required=True)
    parser.add_argument("--lr",        type=float, default=5e-6,
                        help="base LR; CARDIAC-FM paper uses 5e-6 for full-transformer CL")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="LR warmup length in optimizer steps (paper: 50)")
    parser.add_argument("--ecg_lr_scale", type=float, default=1.0,
                        help="multiplier on --lr for the ECG encoder (1.0 = uniform; <1 preserves FM features)")
    parser.add_argument("--mri_lr_scale", type=float, default=1.0,
                        help="multiplier on --lr for the MRI encoder (1.0 = uniform; <1 = gentler teacher)")
    parser.add_argument("--epochs",    type=int,   default=100)
    parser.add_argument("--patience",  type=int,   default=7,
                        help="early stop if val loss does not improve for this many epochs (<=0 disables)")
    parser.add_argument("--batch_size",type=int,   default=64)
    parser.add_argument("--img_size",  type=int,   default=224)
    parser.add_argument("--n_frames",  type=int,   default=8)
    parser.add_argument("--embed_dim", type=int,   default=384)
    parser.add_argument("--encoder_depth", type=int, default=8, help="must match pretraining (base=12)")
    parser.add_argument("--encoder_heads", type=int, default=6, help="must match pretraining (base=12)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mri_tune",     type=str, default="partial", choices=["full", "partial"],
                        help="'full' = fine-tune whole MRI encoder; 'partial' = last N blocks + enc_norm")
    parser.add_argument("--unfreeze_blocks", type=int, default=2,
                        help="top encoder blocks to unfreeze when --mri_tune partial")
    parser.add_argument("--mri_augment",  action="store_true", default=False)
    parser.add_argument("--pool",         type=str, default="mean", choices=["mean", "cls", "per_view"])
    parser.add_argument("--view_encoder", type=str, default="none", choices=["none", "conv", "vit"],
                        help="must match the pretrained checkpoint's per-view stage")
    parser.add_argument("--view_depth",   type=int, default=2)
    parser.add_argument("--n_sa_slices",  type=int, default=3)
    parser.add_argument("--ecg_unfreeze", type=int, default=-1,
                        help="ECG transformer layers to unfreeze from top (-1 = all)")
    parser.add_argument("--resume",       default=None,
                        help="resume a run: restore weights + optimizer + epoch (continues schedule)")
    parser.add_argument("--init_from",    default=None,
                        help="weights-only warm start from a prior stage: load weights, "
                             "then train fresh (epoch 0, new optimizer/schedule, current freeze config)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = ECGMRIDataset(args.csv_train, args.ecg_dir, args.mri_dir, split="train",
                             img_size=args.img_size, n_frames=args.n_frames,
                             mri_augment=args.mri_augment)
    val_ds   = ECGMRIDataset(args.csv_val, args.ecg_dir, args.mri_dir, split="valid",
                             img_size=args.img_size, n_frames=args.n_frames, mri_augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=ECGMRIDataset.collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=ECGMRIDataset.collate_fn)

    model = CineMAECGModel(args.mae_ckpt, args.ecg_ckpt,
                           embed_dim=args.embed_dim, img_size=args.img_size,
                           n_frames=args.n_frames, mri_tune=args.mri_tune,
                           unfreeze_blocks=args.unfreeze_blocks,
                           ecg_unfreeze=args.ecg_unfreeze, pool=args.pool,
                           view_encoder=args.view_encoder, view_depth=args.view_depth,
                           n_sa_slices=args.n_sa_slices,
                           encoder_depth=args.encoder_depth, encoder_heads=args.encoder_heads)
    model = nn.DataParallel(model)
    model.to(device)
    print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    loss_fn = InfoNCELoss(temperature=0.07).to(device)
    m = model.module if hasattr(model, "module") else model
    # Per-group LRs: encoders can run at a different LR than the random-init
    # projections (scales default to 1.0 = uniform, matching the paper).
    optimizer = torch.optim.AdamW(
        [{"params": m.mri_encoder.parameters(),   "lr": args.lr * args.mri_lr_scale},
         {"params": m.mri_projection.parameters(),"lr": args.lr},
         {"params": m.ecg_encoder.parameters(),   "lr": args.lr * args.ecg_lr_scale},
         {"params": m.ecg_projection.parameters(),"lr": args.lr},
         {"params": [loss_fn.temperature],         "lr": args.lr}],
        betas=(0.9, 0.98), eps=1e-6, weight_decay=0.05)

    n_batches = math.ceil(len(train_ds) / args.batch_size)
    scheduler = cosine_lr(optimizer, warmup_length=args.warmup_steps,
                          steps=args.epochs * n_batches)

    start_epoch, best_val = 0, float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "temperature" in ckpt:
            loss_fn.temperature.data = ckpt["temperature"].to(device)
        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best_val={best_val:.4f}")
    elif args.init_from and os.path.isfile(args.init_from):
        # Weights-only warm start (phase-2 curriculum): load the trained weights from a
        # previous stage, but start FRESH — epoch 0, new optimizer, new LR schedule, and
        # the freeze config set at construction (e.g. frozen MRI teacher + full ECG).
        # Optimizer state and epoch counter are intentionally NOT restored.
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model"])
        if "temperature" in ckpt:
            loss_fn.temperature.data = ckpt["temperature"].to(device)
        print(f"Warm-started weights from {args.init_from} "
              f"(was epoch {ckpt.get('epoch', '?')}, val={ckpt.get('val_loss', float('nan')):.4f}); "
              f"restarting at epoch 0 with a fresh optimizer + LR schedule")

    step_offset = start_epoch * n_batches
    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device, scheduler, step_offset)
        val_loss   = val_one_epoch(val_loader, model, loss_fn, device)
        step_offset += n_batches

        print(f"\nEpoch {epoch+1}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"temp={loss_fn.temperature.exp().item():.4f}  time={(time.time()-t0)/60:.1f}min\n", flush=True)

        m = model.module if hasattr(model, "module") else model
        ckpt = {"epoch": epoch+1, "model": m.state_dict(),
                "optimizer": optimizer.state_dict(),
                "temperature": loss_fn.temperature.data.cpu(),
                "val_loss": val_loss}
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(ckpt, os.path.join(args.out_dir, "stage1_cinema_best.pth"))
            print(f"  saved best checkpoint (val={best_val:.4f})")
        else:
            epochs_no_improve += 1
        torch.save(ckpt, os.path.join(args.out_dir, f"stage1_cinema_ep{epoch+1:03d}.pth"))

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch+1} "
                  f"(no val improvement for {args.patience} epochs; best={best_val:.4f})", flush=True)
            break

    print(f"Stage 1 contrastive training complete. Best val={best_val:.4f}")


if __name__ == "__main__":
    main()
