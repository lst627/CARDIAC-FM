"""
Multi-view cross-fusion Masked Autoencoder for cardiac MRI (CineMA-style).

This is a re-design of the earlier `mri_new` MAE. The key differences, motivated
by the CineMA paper (Fu et al., 2025) and adapted to our setting:

  1. CROSS-VIEW FUSION INSIDE THE ENCODER.
     The earlier model ran one shared encoder on each view *independently* and only
     fused at the very end (concat of pooled vectors + MLP head). Here, the visible
     tokens from all views are concatenated and passed through a single shared
     Transformer encoder, so cross-view attention happens during representation
     learning. This is the headline change.

  2. PER-VIEW ENCODING (specialise early, share late) — SELECTABLE.
     Each view (2ch, 4ch, SA) processes its own pixels before fusion. The stage is
     selectable via `view_encoder`:
       - "none" : a single Conv3d patch-embed (plain ViT tokenization). Cheapest;
                  this is what the first checkpoints were trained with.
       - "conv" : a CineMA/ConvMAE-style convolutional stem (spatial downsampling
                  CNN per view) before tokenization. This is the defining piece of
                  CineMA — local inductive bias + richer per-view features.
       - "vit"  : a small per-view Transformer on the view's visible tokens before
                  cross-view fusion (the "ViT per view" idea).
     In all cases the shared fusion Transformer, decoder, and loss are identical.

  3. CROSS-VIEW RECONSTRUCTION WITH PER-VIEW OUTPUT HEADS.
     A shared decoder Transformer operates over the *full* set of (encoded-visible +
     mask) tokens from all views, so each view's masked patches are reconstructed
     using information from the other views. The final pixel projection is a
     per-view linear head (each view reconstructs into its own statistics).

  4. TEMPORAL (8 frames) WITH TUBE MASKING.
     We keep 8 time frames (spatio-temporal tokens, tube_t over time), unlike
     CineMA's single-frame sampling, because temporal dynamics are exactly the
     signal that aligns with ECG downstream. The per-view encoder is SPATIAL-only
     (it downsamples H,W per frame); time is preserved as a token axis and grouped
     into Gt temporal tokens. Because adjacent frames are highly redundant, we use
     VideoMAE-style TUBE masking (mask the same spatial location across all time
     steps) at a high ratio (default 0.9).

  5. VIEW / SLICE EMBEDDINGS.
     Learned view-type embeddings (2ch / 4ch / sa) and a SA slice embedding are
     added so the shared encoder/decoder can tell tokens apart.

Input (matches the existing dataset):
  v2ch : (B, 1, T, H, W)
  v4ch : (B, 1, T, H, W)
  vsa  : (B, S, T, H, W)   — S short-axis slices (default 3)

Pretraining:  forward(v2ch, v4ch, vsa) -> (loss, parts)
Downstream:   encode(v2ch, v4ch, vsa)  -> {"pooled", "cls", "tokens", "view_pooled"}

Reference: He et al. MAE (CVPR 2022); Bachmann et al. MultiMAE (2022);
Tong et al. VideoMAE (2022); Gao et al. ConvMAE (2022); Fu et al. CineMA (2025).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────── positional embedding ──────────────────────────

def sincos_pos_embed_3d(embed_dim: int, grid: tuple) -> torch.Tensor:
    """Sine-cosine positional embedding for a 3D (Gt, Gh, Gw) grid.
    Returns (N, embed_dim) where N = Gt*Gh*Gw, ordered t-major (t, then h, then w).
    """
    Gt, Gh, Gw = grid
    d = embed_dim // 3

    def axis_embed(n, d_axis):
        if d_axis < 2:
            return torch.zeros(n, max(d_axis, 0))
        half = d_axis // 2
        freq = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float32) / half))
        pos  = torch.arange(n, dtype=torch.float32)
        x    = pos.unsqueeze(1) * freq.unsqueeze(0)
        out  = torch.cat([x.sin(), x.cos()], dim=1)
        if d_axis % 2:
            out = F.pad(out, (0, 1))
        return out

    d_t, d_h = d, d
    d_w = embed_dim - 2 * d

    et = axis_embed(Gt, d_t).view(Gt, 1,  1,  d_t).expand(Gt, Gh, Gw, d_t)
    eh = axis_embed(Gh, d_h).view(1,  Gh, 1,  d_h).expand(Gt, Gh, Gw, d_h)
    ew = axis_embed(Gw, d_w).view(1,  1,  Gw, d_w).expand(Gt, Gh, Gw, d_w)

    return torch.cat([et, eh, ew], dim=-1).reshape(Gt * Gh * Gw, embed_dim)


# ───────────────────────────── transformer blocks ────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv       = nn.Linear(dim, dim * 3, bias=True)
        self.proj      = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # flash / memory-efficient attention: never materialises the N×N matrix,
        # so memory is O(N) not O(N^2). Default scale is 1/sqrt(head_dim) — identical
        # to the explicit scale, so pretrained weights are unchanged.
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mid        = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(nn.Linear(dim, mid), nn.GELU(), nn.Linear(mid, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ───────────────────────────── per-view encoders ─────────────────────────────

class CubeEmbed(nn.Module):
    """Non-overlapping 3D patch embedding via a single Conv3d. (view_encoder='none')"""

    def __init__(self, in_chans=1, embed_dim=384, tube_t=2, patch_h=16, patch_w=16):
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tube_t, patch_h, patch_w),
            stride=(tube_t, patch_h, patch_w),
        )

    def forward(self, x):
        # x: (B, C, T, H, W) -> (B, N, embed_dim), tokens ordered t-major
        return self.proj(x).flatten(2).transpose(1, 2)


def _gn(ch, groups=32):
    return nn.GroupNorm(min(groups, ch), ch)


class _ConvNormAct(nn.Module):
    """2D conv-norm-act. With kernel==stride and valid padding this is a strided
    (downsampling) patch conv, as in ConvMAE's DownsampleEncoder."""

    def __init__(self, in_ch, out_ch, kernel, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
        self.norm = _gn(out_ch)
        self.act  = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class _ResConvBlock(nn.Module):
    """Residual conv block at fixed channels/resolution (the 'conv_n_blocks' refiner)."""

    def __init__(self, ch):
        super().__init__()
        self.norm1 = _gn(ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.norm2 = _gn(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.act   = nn.GELU()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class ConvStem(nn.Module):
    """CineMA/ConvMAE-style per-view convolutional stem. (view_encoder='conv')

    Operates SPATIALLY per frame (folds time into the batch), downsampling H,W by
    `patch_h` total via a strided-conv schedule, then groups `tube_t` frames into a
    single temporal token. Output tokens are ordered t-major over (Gt, Gh, Gw),
    matching the sincos pos-embed and the pixel patchify used for the loss.

    Downsample schedule for n_conv = len(conv_chans) conv stages + final embed:
        strides = [patch_h // 2**n_conv] + [2]*(n_conv-1)   (conv stages)
                  + [2]                                       (final patch-embed)
    whose product is patch_h (e.g. patch_h=16, conv_chans=[64,128] -> [4,2] then 2).
    """

    def __init__(self, embed_dim=384, conv_chans=(64, 128), conv_n_blocks=2,
                 tube_t=2, patch_h=16, patch_w=16):
        super().__init__()
        assert patch_h == patch_w, "ConvStem assumes square spatial patches"
        n_conv = len(conv_chans)
        first = patch_h // (2 ** n_conv)
        assert first >= 1 and first * (2 ** n_conv) == patch_h, \
            f"patch_h={patch_h} not compatible with {n_conv} conv stages"
        strides = [first] + [2] * (n_conv - 1)

        blocks = []
        in_ch = 1
        for stride, ch in zip(strides, conv_chans):
            blocks.append(_ConvNormAct(in_ch, ch, kernel=stride, stride=stride))
            for _ in range(conv_n_blocks):
                blocks.append(_ResConvBlock(ch))
            in_ch = ch
        self.blocks = nn.ModuleList(blocks)

        # final spatial patch-embed (stride 2) -> embed_dim at grid (Gh, Gw)
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=2, stride=2)
        # temporal grouping: pool tube_t adjacent frames into one token
        self.time_pool = nn.Conv3d(embed_dim, embed_dim,
                                   kernel_size=(tube_t, 1, 1), stride=(tube_t, 1, 1))

    def forward(self, x):
        # x: (B, 1, T, H, W) -> tokens (B, Gt*Gh*Gw, embed_dim), t-major
        B, _, T, H, W = x.shape
        x = x.reshape(B * T, 1, H, W)
        for blk in self.blocks:
            x = blk(x)
        x = self.proj(x)                                  # (B*T, D, Gh, Gw)
        _, D, Gh, Gw = x.shape
        x = x.reshape(B, T, D, Gh, Gw).permute(0, 2, 1, 3, 4)   # (B, D, T, Gh, Gw)
        x = self.time_pool(x)                             # (B, D, Gt, Gh, Gw)
        return x.flatten(2).transpose(1, 2)               # (B, Gt*Gh*Gw, D)


# ───────────────────────────── the model ─────────────────────────────────────

VIEWS = ("2ch", "4ch", "sa")        # view types with their own per-view encoder / head


class CineMAE(nn.Module):
    """
    Multi-view cross-fusion MAE.

    forward(v2ch, v4ch, vsa) -> (loss, parts_dict)   [pretraining]
    encode (v2ch, v4ch, vsa) -> dict                 [downstream / ECG alignment]
    """

    def __init__(self,
                 img_size=112, n_frames=8,
                 tube_t=2, patch_h=16, patch_w=16,
                 encoder_dim=384, encoder_depth=8, encoder_heads=6,
                 decoder_dim=192, decoder_depth=4, decoder_heads=6,
                 mask_ratio=0.9, norm_pix_loss=True,
                 view_encoder="none", n_sa_slices=3,
                 view_depth=2, conv_chans=(64, 128), conv_n_blocks=2):
        super().__init__()
        assert view_encoder in ("none", "conv", "vit"), view_encoder
        self.mask_ratio    = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.tube_t, self.patch_h, self.patch_w = tube_t, patch_h, patch_w
        self.patch_vol = tube_t * patch_h * patch_w
        self.view_encoder = view_encoder
        self.n_sa_slices  = n_sa_slices

        Gt = n_frames // tube_t
        Gh = img_size // patch_h
        Gw = img_size // patch_w
        self.grid = (Gt, Gh, Gw)
        self.N    = Gt * Gh * Gw                      # tokens per single-slice view
        self.encoder_dim = encoder_dim

        # ── per-view encoders (separate weights → early specialisation) ────────
        if view_encoder == "conv":
            self.patch_embed = nn.ModuleDict({
                v: ConvStem(encoder_dim, conv_chans, conv_n_blocks, tube_t, patch_h, patch_w)
                for v in VIEWS
            })
        else:  # "none" and "vit" both tokenize with a single Conv3d patch-embed
            self.patch_embed = nn.ModuleDict({
                v: CubeEmbed(1, encoder_dim, tube_t, patch_h, patch_w) for v in VIEWS
            })

        # optional per-view ViT (runs on each view's visible tokens before fusion)
        self.view_blocks = None
        if view_encoder == "vit":
            self.view_blocks = nn.ModuleDict({
                v: nn.ModuleList([Block(encoder_dim, encoder_heads) for _ in range(view_depth)])
                for v in VIEWS
            })

        # ── learned view + SA-slice embeddings (encoder & decoder) ────────────
        self.enc_view_embed  = nn.ParameterDict({
            v: nn.Parameter(torch.zeros(1, 1, encoder_dim)) for v in VIEWS})
        self.dec_view_embed  = nn.ParameterDict({
            v: nn.Parameter(torch.zeros(1, 1, decoder_dim)) for v in VIEWS})
        self.enc_slice_embed = nn.Parameter(torch.zeros(n_sa_slices, encoder_dim))
        self.dec_slice_embed = nn.Parameter(torch.zeros(n_sa_slices, decoder_dim))

        # ── shared fusion encoder ─────────────────────────────────────────────
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        self.enc_blocks = nn.ModuleList([Block(encoder_dim, encoder_heads) for _ in range(encoder_depth)])
        self.enc_norm   = nn.LayerNorm(encoder_dim)

        # ── shared decoder + per-view pixel heads ─────────────────────────────
        self.dec_proj   = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.dec_blocks = nn.ModuleList([Block(decoder_dim, decoder_heads) for _ in range(decoder_depth)])
        self.dec_norm   = nn.LayerNorm(decoder_dim)
        self.dec_pred   = nn.ModuleDict({
            v: nn.Linear(decoder_dim, self.patch_vol, bias=True) for v in VIEWS})

        # ── fixed sincos positional embeddings (all views share the grid) ─────
        self.register_buffer("enc_pos", sincos_pos_embed_3d(encoder_dim, self.grid))
        self.register_buffer("dec_pos", sincos_pos_embed_3d(decoder_dim, self.grid))

        self._init_weights()

    def _init_weights(self):
        for p in (self.cls_token, self.mask_token,
                  *self.enc_view_embed.values(), *self.dec_view_embed.values(),
                  self.enc_slice_embed, self.dec_slice_embed):
            nn.init.trunc_normal_(p, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── tokenization into "streams" ───────────────────────────────────────────
    # A stream is one single-slice token block: 2ch, 4ch, and each SA slice.
    def _stream_specs(self, v2ch, v4ch, vsa):
        """Returns list of dicts: {view, slice, raw (B,1,T,H,W)}."""
        specs = [{"view": "2ch", "slice": None, "raw": v2ch},
                 {"view": "4ch", "slice": None, "raw": v4ch}]
        for s in range(self.n_sa_slices):
            specs.append({"view": "sa", "slice": s, "raw": vsa[:, s:s + 1]})
        return specs

    def _tokenize(self, st, pixel_keep=None):
        """Produce full-grid tokens (B, N, D) for one stream, with embeds added.
        If view_encoder == 'conv' and pixel_keep is given, masked pixels are zeroed
        before the conv stem (SimMIM-style, leak-free masking)."""
        x = st["raw"]
        if self.view_encoder == "conv" and pixel_keep is not None:
            x = x * pixel_keep                       # (B,1,T,H,W) * (B,1,T,H,W)
        tok = self.patch_embed[st["view"]](x)        # (B, N, D)
        tok = tok + self.enc_pos.unsqueeze(0) + self.enc_view_embed[st["view"]]
        if st["slice"] is not None:
            tok = tok + self.enc_slice_embed[st["slice"]].view(1, 1, -1)
        return tok

    def _add_dec_embeds(self, x, st):
        x = x + self.dec_pos.unsqueeze(0) + self.dec_view_embed[st["view"]]
        if st["slice"] is not None:
            x = x + self.dec_slice_embed[st["slice"]].view(1, 1, -1)
        return x

    # ── VideoMAE-style tube masking ───────────────────────────────────────────
    def _make_tube_mask(self, B, device):
        """Sample one tube mask shared by all streams' grids of this batch.

        Returns ids_keep (B, n_keep), ids_restore (B, N), mask (B, N) bool (True=remove),
        n_keep, and spatial_keep (B, Gh*Gw) float (1=keep) for pixel-level masking.
        """
        Gt, Gh, Gw = self.grid
        S = Gh * Gw
        ks = max(1, int(round(S * (1 - self.mask_ratio))))   # spatial tubes to keep

        # one noise value per spatial location, shared across time -> tube structure
        noise_s = torch.rand(B, S, device=device)
        noise   = noise_s.unsqueeze(1).expand(B, Gt, S).reshape(B, self.N)

        ids_shuffle = torch.argsort(noise, dim=1, stable=True)   # equal-noise tubes stay contiguous
        ids_restore = torch.argsort(ids_shuffle, dim=1, stable=True)

        n_keep   = Gt * ks
        ids_keep = ids_shuffle[:, :n_keep]
        mask = torch.ones(B, self.N, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_keep, False)

        # spatial keep pattern (constant across time for a tube): take t=0 slice
        spatial_keep = (~mask).reshape(B, Gt, S)[:, 0].float()   # (B, S)
        return ids_keep, ids_restore, mask, n_keep, spatial_keep

    def _gather(self, tokens, ids_keep):
        D = tokens.shape[-1]
        return torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

    def _pixel_keep(self, spatial_keep, T):
        """Upsample (B, Gh*Gw) spatial keep mask to (B,1,T,H,W) pixel mask."""
        Gt, Gh, Gw = self.grid
        m = spatial_keep.reshape(-1, 1, Gh, Gw)                       # (B,1,Gh,Gw)
        m = m.repeat_interleave(self.patch_h, dim=2).repeat_interleave(self.patch_w, dim=3)
        H, W = Gh * self.patch_h, Gw * self.patch_w
        return m.reshape(-1, 1, 1, H, W).expand(-1, 1, T, H, W)      # (B,1,T,H,W)

    def _run_view_vit(self, x_vis, view):
        if self.view_blocks is not None:
            for blk in self.view_blocks[view]:
                x_vis = blk(x_vis)
        return x_vis

    def _run_encoder(self, vis_tokens):
        B = vis_tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, vis_tokens], dim=1)
        for blk in self.enc_blocks:
            x = blk(x)
        x = self.enc_norm(x)
        return x[:, 0], x[:, 1:]      # cls_out, token_out

    def _patchify(self, x):
        """x: (B, 1, T, H, W) -> (B, N, patch_vol), matching token order."""
        B = x.shape[0]
        Gt, Gh, Gw = self.grid
        tt, ph, pw = self.tube_t, self.patch_h, self.patch_w
        x = x.reshape(B, Gt, tt, Gh, ph, Gw, pw)
        x = x.permute(0, 1, 3, 5, 2, 4, 6)
        return x.reshape(B, Gt * Gh * Gw, tt * ph * pw)

    def _recon_loss(self, pred, raw, mask):
        target = self._patchify(raw)
        if self.norm_pix_loss:
            mean   = target.mean(-1, keepdim=True)
            var    = target.var(-1, keepdim=True, unbiased=False)
            target = (target - mean) / (var + 1e-6).sqrt()
        mse = ((pred - target) ** 2).mean(-1)              # (B, N)
        return (mse * mask.float()).sum() / mask.float().sum().clamp(min=1)

    # ── shared encode path (used by both forward and encode) ───────────────────
    def _encode_streams(self, specs, mask=True):
        """Tokenize + (optionally) mask all streams, run per-view ViT on visible
        tokens, fuse through the shared encoder.

        Returns enc_out (B, total_keep, D), and per-stream masking bookkeeping
        (n_keep / ids_restore / mask) — the latter only meaningful when mask=True.
        """
        B = specs[0]["raw"].shape[0]
        device = specs[0]["raw"].device
        T = specs[0]["raw"].shape[2]

        if mask:
            ids_keep, ids_restore, tube_mask, n_keep, spatial_keep = self._make_tube_mask(B, device)
            pixel_keep = self._pixel_keep(spatial_keep, T) if self.view_encoder == "conv" else None
        else:
            ids_keep = ids_restore = tube_mask = pixel_keep = None
            n_keep = self.N

        vis_list, view_order = [], []
        for st in specs:
            tok = self._tokenize(st, pixel_keep=pixel_keep)      # (B, N, D)
            vis = self._gather(tok, ids_keep) if mask else tok   # (B, n_keep, D)
            vis = self._run_view_vit(vis, st["view"])
            vis_list.append(vis)
            view_order.append(st["view"])
            st["n_keep"] = vis.shape[1]
            st["ids_restore"] = ids_restore
            st["mask"] = tube_mask

        cls_out, enc_out = self._run_encoder(torch.cat(vis_list, dim=1))
        return cls_out, enc_out

    # ── pretraining forward ───────────────────────────────────────────────────
    def forward(self, v2ch, v4ch, vsa):
        specs = self._stream_specs(v2ch, v4ch, vsa)
        _, enc_out = self._encode_streams(specs, mask=True)

        # rebuild full per-stream sequences (visible + mask tokens) for the decoder
        dec_list, off = [], 0
        for st in specs:
            nk = st["n_keep"]
            enc_vis = enc_out[:, off:off + nk]; off += nk
            d = self.dec_proj(enc_vis)                              # (B, nk, dec_dim)
            B, _, Dd = d.shape
            mask_tok = self.mask_token.expand(B, self.N - nk, -1)
            full = torch.cat([d, mask_tok], dim=1)
            full = torch.gather(full, 1, st["ids_restore"].unsqueeze(-1).expand(-1, -1, Dd))
            dec_list.append(self._add_dec_embeds(full, st))

        # shared decoder over ALL views' tokens → cross-view reconstruction
        xd = torch.cat(dec_list, dim=1)
        for blk in self.dec_blocks:
            xd = blk(xd)
        xd = self.dec_norm(xd)

        # per-view pixel heads + per-view masked MSE
        view_losses = {v: [] for v in VIEWS}
        off = 0
        for st in specs:
            seg  = xd[:, off:off + self.N]; off += self.N
            pred = self.dec_pred[st["view"]](seg)
            view_losses[st["view"]].append(self._recon_loss(pred, st["raw"], st["mask"]))

        # average within a view (SA over its slices), then across the 3 views
        per_view = {v: torch.stack(ls).mean() for v, ls in view_losses.items() if ls}
        loss = torch.stack(list(per_view.values())).mean()
        parts = {v: l.detach() for v, l in per_view.items()}
        return loss, parts

    # ── downstream encoding (no masking) ──────────────────────────────────────
    def encode(self, v2ch, v4ch, vsa, pool="mean"):
        """
        Returns dict:
          pooled      : fused MRI embedding for ECG alignment / a prediction head.
                          pool='mean'     -> (B, encoder_dim)     mean over ALL tokens
                          pool='cls'      -> (B, encoder_dim)     CLS-token summary
                          pool='per_view' -> (B, 3*encoder_dim)   [2ch | 4ch | sa] concat
          cls         : (B, encoder_dim)        — CLS-token summary
          tokens      : (B, (2+n_sa)*N, dim)    — full fused token set
          view_pooled : (B, 3*encoder_dim)      — per-view mean-pool, concatenated
                        ([2ch, 4ch, sa]); sa is the mean over its slices. This is
                        the late-fusion-capacity readout but each token has already
                        seen the other views through the shared encoder.

        Token order in `tokens` is [2ch (N), 4ch (N), sa0 (N), ...], so the view
        boundaries below are fixed by self.N. No masking; gradients flow normally.
        """
        specs = self._stream_specs(v2ch, v4ch, vsa)
        cls_out, token_out = self._encode_streams(specs, mask=False)

        # per-view pooling using the fixed stream layout
        N = self.N
        n_sa = self.n_sa_slices
        p2  = token_out[:, 0:N].mean(dim=1)                    # 2ch
        p4  = token_out[:, N:2 * N].mean(dim=1)                # 4ch
        psa = token_out[:, 2 * N:(2 + n_sa) * N].mean(dim=1)   # sa: all slices together
        view_pooled = torch.cat([p2, p4, psa], dim=-1)         # (B, 3*encoder_dim)

        if pool == "cls":
            pooled = cls_out
        elif pool == "per_view":
            pooled = view_pooled
        else:
            pooled = token_out.mean(dim=1)
        return {"pooled": pooled, "cls": cls_out,
                "tokens": token_out, "view_pooled": view_pooled}
