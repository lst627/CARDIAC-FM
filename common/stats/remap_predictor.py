"""
Convert an existing downstream (multimodal) ECG checkpoint into a bare CARDIACFM_ECG
state_dict, so we can REUSE the UKB af5/hf5 predictors already trained in the downstream
sweep instead of re-running the UKB fine-tune (01). The downstream ECG head is architecturally
identical to CARDIACFM_ECG; only the key names differ (+ unused MRI params):

    ecg_encoder.*      -> ecg_encoder_multi.*
    ecg_projection.*   -> ecg_projection_multi.*
    head.*             -> pred.*
    mri_encoder.* / mri_projection.*  -> dropped

Output is a plain state_dict loadable by CARDIACFM_ECG.load_state_dict(strict=True).
"""
import argparse
import torch


def remap(in_ckpt):
    c = torch.load(in_ckpt, map_location="cpu", weights_only=False)
    sd = c["model"] if isinstance(c, dict) and "model" in c and isinstance(c["model"], dict) else c
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    out = {}
    for k, v in sd.items():
        if k.startswith("ecg_encoder."):
            out["ecg_encoder_multi." + k[len("ecg_encoder."):]] = v
        elif k.startswith("ecg_projection."):
            out["ecg_projection_multi." + k[len("ecg_projection."):]] = v
        elif k.startswith("head."):
            out["pred." + k[len("head."):]] = v
        # everything else (mri_*) is intentionally dropped
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", required=True, help="downstream_<outcome>_ecg_best.pth")
    ap.add_argument("--out_ckpt", required=True, help="where to write the CARDIACFM_ECG state_dict")
    args = ap.parse_args()
    import os
    out = remap(args.in_ckpt)
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    torch.save(out, args.out_ckpt)
    print(f"wrote {len(out)} keys -> {args.out_ckpt}")
