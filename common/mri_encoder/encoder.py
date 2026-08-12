"""
Downstream encoder backed by a pretrained CineMAE.

Because cross-view fusion now happens INSIDE the model, this wrapper returns a
single fused embedding per subject (B, encoder_dim) — not three per-view vectors.
That fused vector is the natural target for ECG alignment or a prediction head.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))   # cinema_mae.py is a sibling module
from cinema_mae import CineMAE


class CineMAEncoder(nn.Module):
    """
    Wraps a pretrained CineMAE to produce a fused multi-view MRI embedding.

    Parameters
    ----------
    mae_cfg : dict   — kwargs forwarded to CineMAE.__init__()
    ckpt    : str    — path to cinema_best.pth; None to start from scratch
    freeze  : bool   — freeze the backbone (train head / alignment proj only)
    pool    : str    — 'mean' (over fused tokens) or 'cls'
    """

    def __init__(self, mae_cfg: dict, ckpt: str = None,
                 freeze: bool = False, pool: str = "mean"):
        super().__init__()
        self.mae = CineMAE(**mae_cfg)
        enc_dim = mae_cfg.get("encoder_dim", 384)
        self.pool = pool
        # per_view concatenates [2ch | 4ch | sa] -> 3x the encoder dim. This matches
        # the old late-fusion readout capacity while every token has already attended
        # across views inside the shared encoder.
        self.embed_dim = enc_dim * 3 if pool == "per_view" else enc_dim

        if ckpt is not None:
            state = torch.load(ckpt, map_location="cpu")
            sd = state.get("model", state)
            # The encoder is all we use downstream — drop decoder/pixel-head weights so
            # the pretrain decoder config (e.g. a bigger base decoder) can differ freely
            # without shape-clashing on load.
            sd = {k: v for k, v in sd.items()
                  if not (k.startswith("dec_") or k.startswith("decoder") or k == "mask_token")}
            missing, unexpected = self.mae.load_state_dict(sd, strict=False)
            if missing:
                print(f"[CineMA encoder] missing keys: {len(missing)}")
            if unexpected:
                print(f"[CineMA encoder] unexpected keys: {len(unexpected)}")

        if freeze:
            for p in self.mae.parameters():
                p.requires_grad_(False)
            for p in self.mae.enc_norm.parameters():   # keep final norm trainable
                p.requires_grad_(True)

    def forward(self, v2ch, v4ch, vsa):
        """Returns the fused embedding (B, encoder_dim)."""
        return self.mae.encode(v2ch, v4ch, vsa, pool=self.pool)["pooled"]
