"""
Time-to-event (Cox / DeepSurv, arXiv:1606.00931) downstream fine-tune for CARDIAC-FM ECG.

Same encoder + dataset as the binary ecg_finetune.py; the ONLY changes are:
  * the head output is a scalar LOG-HAZARD (no sigmoid),
  * the loss is the Cox partial likelihood (Breslow), computed over the risk set WITHIN each batch,
  * the label is the signed time-to-event from build_surv_labels.py:  label>0 => event at |label|,
    label<0 => censored at |label|, NaN => excluded (prevalent / uncovered),
  * model selection is on validation C-index (higher = better).

Because the partial likelihood needs events in the batch and UKB events are rare (~2%), use a LARGE
batch (default 48; the H200 has room). Batches with 0 events contribute no gradient (that's fine).

  python cox_finetune.py --seed 1 --epochs 20 --model_name CARDIACFM \
     --ecgfm_ckpt <..> --cardiacfm_pretrained_ckpt <CL.pth> \
     --ecg_tsv_dir <ECG_manifest_moretest> --label_dir <ECG_label_surv/af> --save_dir <out>
"""
import os, sys, time, math, random, argparse
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # cardiacfm_new/
sys.path.insert(0, os.path.join(_ROOT, "common", "ecg_encoder"))   # model_ecg.py
sys.path.insert(0, os.path.join(_ROOT, "common", "data"))          # ecg_dataset.py
from model_ecg import ECGFM, CARDIACFM_ECG, cosine_lr
from ecg_dataset import ECGDataset


def cox_ph_loss(logh, T, E, eps=1e-7):
    """Breslow negative partial log-likelihood over the batch risk set.
    logh: (B,) log-hazard. T: (B,) time. E: (B,) event indicator (1/0). Returns a scalar."""
    n_events = E.sum()
    if n_events < 1:
        return (logh * 0.0).sum()                     # no events -> no gradient
    order = torch.argsort(T, descending=True)         # descending time => forward cumsum = risk set
    logh, E = logh[order], E[order]
    log_cumsum = torch.logcumsumexp(logh, dim=0)      # log sum_{T_j >= T_i} exp(logh_j)
    pl = (logh - log_cumsum) * E
    return -pl.sum() / (n_events + eps)


def decode(label):
    """signed-T label -> (T, E, mask). label>0 event, label<0 censored, NaN excluded."""
    mask = ~torch.isnan(label)
    E = (label > 0).float()
    T = label.abs()
    return T, E, mask


def run_batch(model, ecgs, device):
    ecgs["net_input"]["source"] = ecgs["net_input"]["source"].to(device)
    logh = model(ecgs).squeeze(-1)                    # scalar log-hazard, NO sigmoid
    label = ecgs["label"].to(device).float().squeeze(-1)
    T, E, mask = decode(label)
    return logh[mask], T[mask], E[mask]


def train_one_epoch(loader, model, optimizer, device, scheduler, num_of_steps):
    model.train(); losses = []; begin = time.time()
    for i, ecgs in enumerate(loader):
        if scheduler is not None:
            scheduler(i + num_of_steps)
        optimizer.zero_grad()
        logh, T, E = run_batch(model, ecgs, device)
        if len(logh) < 2 or E.sum() < 1:              # need >=1 event and >=2 at-risk in the batch
            continue
        loss = cox_ph_loss(logh, T, E)
        loss.backward(); optimizer.step()
        losses.append(loss.item())
        if i % 20 == 0:
            el = time.time() - begin
            print(f"  batch {i}/{len(loader)}  loss={loss.item():.4f}  "
                  f"eta={el*(len(loader)-i-1)/(i+1):.0f}s", flush=True)
    return float(np.mean(losses)) if losses else float("nan")


def val_cindex(loader, model, device):
    model.eval(); H, TT, EE = [], [], []
    with torch.no_grad():
        for ecgs in loader:
            logh, T, E = run_batch(model, ecgs, device)
            H.append(logh.cpu().numpy()); TT.append(T.cpu().numpy()); EE.append(E.cpu().numpy())
    H, TT, EE = np.concatenate(H), np.concatenate(TT), np.concatenate(EE)
    # higher hazard -> shorter survival => negate for lifelines' "higher score = longer survival"
    c = concordance_index(TT, -H, EE) if EE.sum() > 0 else float("nan")
    return c, int(EE.sum()), len(EE)


def train_clip(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    label_dir = f"{args.label_dir}/y.npy"
    trainset = ECGDataset(args.ecg_tsv_dir, label_dir, split="train")
    validset = ECGDataset(args.ecg_tsv_dir, label_dir, split="valid")
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, collate_fn=ECGDataset.collate_fn)
    validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, collate_fn=ECGDataset.collate_fn)

    if "ECGFM" in args.model_name:
        model = ECGFM(ecgfm_ckpt=args.ecgfm_ckpt)
    else:
        model = CARDIACFM_ECG(ecgfm_ckpt=args.ecgfm_ckpt,
                              cardiacfm_pretrained_ckpt=args.cardiacfm_pretrained_ckpt)
    model.to(device)
    if args.finetuned_ckpt:
        model.load_state_dict(torch.load(args.finetuned_ckpt, weights_only=False))
    if args.freeze_encoder:
        enc = model.ecg_encoder_multi if "CARDIACFM" in args.model_name else model.ecg_encoder
        for p in enc.parameters():
            p.requires_grad = False

    num_batches = math.ceil(len(trainset) / args.batch_size)
    if "ECGFM" in args.model_name:
        groups = [{"params": model.pred.parameters(), "lr": 1e-4}]
        if not args.freeze_encoder:
            groups.append({"params": model.ecg_encoder.parameters(), "lr": 1e-5})
    else:
        groups = [{"params": model.pred.parameters(), "lr": 1e-4},
                  {"params": model.ecg_projection_multi.parameters(), "lr": 1e-4}]
        if not args.freeze_encoder:
            groups.append({"params": model.ecg_encoder_multi.parameters(), "lr": 1e-5})
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-6, weight_decay=1e-2)
    scheduler = cosine_lr(optimizer, base_lr=args.base_lr, warmup_length=50, steps=args.epochs * num_batches)

    print(f"\ntrain n={len(trainset)} valid n={len(validset)} batch={args.batch_size} "
          f"batches/epoch={num_batches}\n", flush=True)
    best_c, patience, since, num_of_steps = -1.0, 3, 0, 0
    for epoch in range(args.epochs):
        begin = time.time()
        tl = train_one_epoch(trainloader, model, optimizer, device, scheduler, num_of_steps)
        vc, vev, vn = val_cindex(validloader, model, device)
        num_of_steps += num_batches
        print(f"\n Epoch {epoch+1}: train_loss={tl:.4f}  val_Cindex={vc:.4f} "
              f"(events={vev}/{vn})  {(time.time()-begin)/60:.1f} min\n", flush=True)
        if np.isfinite(vc) and vc > best_c:
            best_c, since = vc, 0
            sp = os.path.join(args.save_dir, f"epoch_{epoch}.pth")
            os.makedirs(os.path.dirname(sp), exist_ok=True)
            torch.save(model.state_dict(), sp)
            print(f"  [saved best C-index {best_c:.4f} -> {sp}]", flush=True)
        else:
            since += 1
        if since >= patience:
            print(f"\nEarly stop @ epoch {epoch+1} (best val C-index {best_c:.4f})\n", flush=True)
            break
    print(f"[DONE] best val C-index = {best_c:.4f}", flush=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(s)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--model_name", type=str, default="CARDIACFM")
    p.add_argument("--label_dir", type=str, required=True, help="dir with the signed-T y.npy (e.g. ECG_label_surv/af)")
    p.add_argument("--ecg_tsv_dir", type=str, default="")
    p.add_argument("--ecgfm_ckpt", type=str, default="")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--cardiacfm_pretrained_ckpt", type=str, default=None)
    p.add_argument("--finetuned_ckpt", type=str, default="")
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--base_lr", type=float, default=5e-6)
    p.add_argument("--batch_size", type=int, default=48, help="large batch => enough events per risk set")
    args = p.parse_args()
    set_seed(args.seed)
    train_clip(args)
