"""
Cox/DeepSurv fine-tune on the ECGFounder backbone (survival-downstream baseline). Reuses
ecgfounder_run.py's ECGFounderDataset + build_model + preprocessing; only the loss (Cox partial
likelihood) and metric (C-index) change. Labels = ECG_label_surv/{af,hf}/y.npy (signed-T encoding).

  train: python cox_ecgfounder.py train --outcome af --ecg_tsv_dir <man> --label_dir <ECG_label_surv/af> \
             --pretrained <12_lead_ECGFounder.pth> --ckpt_out <best.pth>
  test:  python cox_ecgfounder.py test  --outcome af --split test --ckpt_in <best.pth> \
             --ecg_tsv_dir <man> --label_dir <ECG_label_surv/af> --save_dir <dir>
"""
import os, sys, argparse
os.environ.setdefault("ECGFOUNDER_REPO",
                      "/gpfs/projects/trend/bojun/multimodal_rep/ECGFounder_DeepSSL/ECGFounder")
import numpy as np, pandas as pd, torch
from lifelines.utils import concordance_index
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ecgfounder_run as EF          # provides ECGFounderDataset, build_model, loaders, seed_everything


def cox_ph_loss(logh, T, E):
    order = torch.argsort(T, descending=True)
    logh, E = logh[order], E[order].float()
    pl = (logh - torch.logcumsumexp(logh, dim=0)) * E
    return -pl.sum() / E.sum().clamp(min=1.0)


def decode(label):
    m = torch.isfinite(label)
    return label.abs(), (label > 0).float(), m


def evaluate(model, loader, dev):
    model.eval(); H, TT, EE = [], [], []
    with torch.no_grad():
        for x, label, _f, _idx in loader:
            T, E, m = decode(label.float())
            logh = model(x.to(dev)).squeeze(-1).cpu()
            H.append(logh[m].numpy()); TT.append(T[m].numpy()); EE.append(E[m].numpy())
    H, TT, EE = np.concatenate(H), np.concatenate(TT), np.concatenate(EE)
    c = concordance_index(TT, -H, EE) if EE.sum() > 0 else float("nan")
    return c, int(EE.sum()), len(EE), (H, TT, EE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "test"])
    ap.add_argument("--outcome", required=True)
    ap.add_argument("--ecg_tsv_dir", required=True)
    ap.add_argument("--label_dir", required=True)
    ap.add_argument("--pretrained", default=os.path.join(os.environ["ECGFOUNDER_REPO"],
                                                         "checkpoint/12_lead_ECGFounder.pth"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt_out"); ap.add_argument("--ckpt_in"); ap.add_argument("--save_dir")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EF.seed_everything(a.seed)
    model = EF.build_model(1, a.pretrained, dev)
    print("device:", dev, flush=True)

    if a.mode == "train":
        tr = EF.loaders(a.ecg_tsv_dir, a.label_dir, "train", a.batch_size, True, a.seed)
        va = EF.loaders(a.ecg_tsv_dir, a.label_dir, "valid", a.batch_size, False)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        best, best_state, since = -1.0, None, 0
        for ep in range(a.epochs):
            model.train()
            for x, label, _f, _idx in tr:
                T, E, m = decode(label.to(dev).float())
                if m.sum() < 2 or E[m].sum() < 1:
                    continue
                logh = model(x.to(dev)).squeeze(-1)[m]
                loss = cox_ph_loss(logh, T[m], E[m])
                opt.zero_grad(); loss.backward(); opt.step()
            c, ev, n, _ = evaluate(model, va, dev)
            print(f"  epoch {ep}: val C-index = {c:.4f} (events={ev}/{n})", flush=True)
            if np.isfinite(c) and c > best:
                best, best_state, since = c, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                since += 1
                if since >= a.patience:
                    print(f"  early stop @ {ep} (best {best:.4f})", flush=True); break
        if best_state is None:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        os.makedirs(os.path.dirname(a.ckpt_out), exist_ok=True)
        torch.save(best_state, a.ckpt_out)
        print(f"[best val C-index {best:.4f}] -> {a.ckpt_out}", flush=True)
    else:
        model.load_state_dict(torch.load(a.ckpt_in, map_location=dev))
        c, ev, n, (H, TT, EE) = evaluate(model, EF.loaders(a.ecg_tsv_dir, a.label_dir, a.split, a.batch_size, False), dev)
        os.makedirs(a.save_dir, exist_ok=True)
        pd.DataFrame({"logh": H, "T": TT, "E": EE}).to_csv(f"{a.save_dir}/result.csv", index=False)
        open(f"{a.save_dir}/cindex.txt", "w").write(f"ecgfounder\t{a.outcome}\t{a.split}\tC-index={c:.4f}\tevents={ev}\tn={n}\n")
        print(f"[ecgfounder {a.outcome} {a.split}] C-index={c:.4f} (events={ev}, n={n}) -> {a.save_dir}", flush=True)


if __name__ == "__main__":
    main()
