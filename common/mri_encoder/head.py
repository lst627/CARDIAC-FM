"""
Finetuning model: pretrained CineMAEncoder + MLP prediction head.

Fusion is internal to the encoder, so the head consumes a single fused embedding
(encoder_dim), unlike the old late-fusion head that took 3*encoder_dim.
"""

import torch.nn as nn


class FinetuneModel(nn.Module):
    def __init__(self, encoder, embed_dim: int = 384,
                 dropout: float = 0.3, n_outputs: int = 1):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_outputs),
        )

    def forward(self, v2ch, v4ch, vsa):
        fused = self.encoder(v2ch, v4ch, vsa)     # (B, embed_dim)
        return self.head(fused).squeeze(-1)       # (B,)
