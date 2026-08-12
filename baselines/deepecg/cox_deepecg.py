"""
Cox/DeepSurv fine-tune on the DeepECG-SL and DeepECG-SSL backbones (survival-downstream baselines
for the time-to-event comparison). Reuses each model's DS + Classifier + preprocessing from
deepsl_finetune.py / deepssl_finetune.py; only the loss (Cox partial likelihood) and metric
(Harrell's C-index) change. Labels come from ECG_label_surv/{af,hf}/y.npy (signed-T encoding:
+tto=event, -tto=censored, NaN=excluded), decoded per batch.

  train: python cox_deepecg.py train --model deepsl --outcome af --ecg_tsv_dir <manifest> \
              --label_dir <ECG_label_surv/af> --ckpt_out <best.pth>
  test:  python cox_deepecg.py test  --model deepsl --outcome af --split test --ckpt_in <best.pth> \
              --ecg_tsv_dir <manifest> --label_dir <ECG_label_surv/af> --save_dir <dir>
"""
import os, argparse, importlib.util
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
HERE = os.path.dirname(os.path.abspath(__file__))


def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, f"{HERE}/{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def cox_ph_loss(logh, T, E, eps=1e-7):
    order = torch.argsort(T, descending=True)      # descending time -> forward cumsum = risk set
    logh, E = logh[order], E[order].float()
    log_cumsum = torch.logcumsumexp(logh, dim=0)
    pl = (logh - log_cumsum) * E
    return -pl.sum() / E.sum().clamp(min=1.0)


def decode(label):
    """signed-T label -> (T, E, finite-mask)."""
    m = torch.isfinite(label)
    return label.abs(), (label > 0).float(), m


def evaluate(model, loader, dev):
    model.eval(); H, TT, EE = [], [], []
    with torch.no_grad():
        for x, label, _ in loader:
            T, E, m = decode(label.float())
            logh = model(x.to(dev)).cpu()
            H.append(logh[m].numpy()); TT.append(T[m].numpy()); EE.append(E[m].numpy())
    H, TT, EE = np.concatenate(H), np.concatenate(TT), np.concatenate(EE)
    c = concordance_index(TT, -H, EE) if EE.sum() > 0 else float("nan")
    return c, int(EE.sum()), len(EE), (H, TT, EE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "test"])
    ap.add_argument("--model", required=True, choices=["deepsl", "deepssl"])
    ap.add_argument("--outcome", required=True)
    ap.add_argument("--ecg_tsv_dir", required=True)
    ap.add_argument("--label_dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt_out"); ap.add_argument("--ckpt_in"); ap.add_argument("--save_dir")
    a = ap.parse_args()
    mod = load_mod("deepsl_finetune" if a.model == "deepsl" else "deepssl_finetune")
    mod.seed_all(a.seed)
    Model = mod.DeepSLClassifier if a.model == "deepsl" else mod.DeepSSLClassifier
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = int(os.environ.get("W", "8"))
    print("device:", dev, "model:", a.model, flush=True)

    def loader(split, shuffle):
        g = torch.Generator().manual_seed(a.seed) if shuffle else None
        return DataLoader(mod.DS(a.ecg_tsv_dir, a.label_dir, split), batch_size=a.batch_size,
                          shuffle=shuffle, num_workers=W, generator=g)

    if a.mode == "train":
        tr, va = loader("train", True), loader("valid", False)
        model = Model().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        best, best_state, since = -1.0, None, 0
        for ep in range(a.epochs):
            model.train()
            for x, label, _ in tr:
                T, E, m = decode(label.to(dev).float())
                if m.sum() < 2 or E[m].sum() < 1:
                    continue
                logh = model(x.to(dev))[m]
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
        model = Model().to(dev)
        model.load_state_dict(torch.load(a.ckpt_in, map_location=dev))
        c, ev, n, (H, TT, EE) = evaluate(model, loader(a.split, False), dev)
        os.makedirs(a.save_dir, exist_ok=True)
        pd.DataFrame({"logh": H, "T": TT, "E": EE}).to_csv(f"{a.save_dir}/result.csv", index=False)
        open(f"{a.save_dir}/cindex.txt", "w").write(f"{a.model}\t{a.outcome}\t{a.split}\tC-index={c:.4f}\tevents={ev}\tn={n}\n")
        print(f"[{a.model} {a.outcome} {a.split}] C-index={c:.4f} (events={ev}, n={n}) -> {a.save_dir}", flush=True)


if __name__ == "__main__":
    main()
