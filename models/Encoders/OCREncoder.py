import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


def createOCREncoder(dim=1024, local_layers=6, global_layers=3, refiner_layers=2,
                     num_output_tokens=256, window_size=7, patch_size=16):
    """Factory mirroring createVisionEncoder/createTextEnc's ``dim=`` convention
    (see VisionEncoder.py / TextEncoder.py). Layer counts are scaled down from
    the reference methodology's 12/6/3 split (see OCR_Encoding_Expert_Methodology.md)
    to something a single-GPU research run can execute -- same scaling
    philosophy already used for HYPERPARAMS["ssl"] in enc_pre.py."""
    return OCREncoder(
        in_channels=3, dim=dim, patch_size=patch_size, window_size=window_size,
        local_layers=local_layers, global_layers=global_layers,
        refiner_layers=refiner_layers, num_output_tokens=num_output_tokens,
    )


# =========================================================================== #
# Stage 1: local, window-attention feature extraction.
#
# Genuine block-local (Swin-style) attention: the patch grid is partitioned
# into non-overlapping window_size x window_size blocks and full attention
# runs independently *within* each block (batched via a reshape, not a
# python loop), giving each layer an O(window_size^2) receptive field instead
# of the full O(grid_h*grid_w) -- appropriate for character-level strokes,
# which are local phenomena, and far cheaper than global attention at the
# patch counts a full document page produces. A 2D sincos position embedding
# is added once, before the window-attention stack, since (unlike
# VisionEncoder's axial RoPE) window attention needs an explicit position
# signal to tell same-relative-offset windows apart.
# =========================================================================== #
class WindowAttention(nn.Module):
    def __init__(self, d_model, nhead, window_size, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.window_size = window_size
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, grid_h, grid_w):
        # x: [B, grid_h*grid_w, D], grid_h/grid_w already padded to a
        # multiple of window_size by the caller (LocalOCREncoder).
        B, N, D = x.shape
        ws = self.window_size
        nh, nw = grid_h // ws, grid_w // ws

        windows = x.view(B, nh, ws, nw, ws, D).permute(0, 1, 3, 2, 4, 5)
        windows = windows.reshape(B * nh * nw, ws * ws, D)      # [B*nWin, ws*ws, D]

        qkv = self.qkv(windows).reshape(windows.size(0), ws * ws, 3, self.nhead, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(windows.size(0), ws * ws, D)
        out = self.proj(out)

        out = out.view(B, nh, nw, ws, ws, D).permute(0, 1, 3, 2, 4, 5)
        return out.reshape(B, N, D)


class WindowEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, window_size, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = WindowAttention(d_model, nhead, window_size, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, grid_h, grid_w):
        x = x + self.attn(self.norm1(x), grid_h, grid_w)
        x = x + self.mlp(self.norm2(x))
        return x


class LocalOCREncoder(nn.Module):
    """Stage 1: patchify + window-attention transformer. Returns the patch
    grid flattened row-major (``tokens[:, i*grid_w + j]`` is patch (i, j)),
    plus (grid_h, grid_w) so callers (GlobalOCREncoder, pretraining losses
    that need spatial structure) can reshape back."""

    def __init__(self, in_channels=3, dim=1024, patch_size=16, window_size=7,
                num_layers=6, nhead=8, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.window_size = window_size
        self.dim = dim

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size,
                                     stride=patch_size)
        self.layers = nn.ModuleList([
            WindowEncoderLayer(dim, nhead, dim * 4, window_size, dropout)
            for _ in range(num_layers)
        ])

    def _pad_to_multiple(self, x, multiple):
        H, W = x.shape[-2:]
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    def _2d_sincos_pos_embed(self, grid_h, grid_w, device):
        dim = self.dim
        assert dim % 4 == 0, "dim must be divisible by 4 for a 2D sincos embed"
        quarter = dim // 4
        omega = torch.exp(-torch.arange(quarter, device=device).float()
                          * (math.log(10000.0) / quarter))
        ys = torch.arange(grid_h, device=device).float()
        xs = torch.arange(grid_w, device=device).float()
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        gy, gx = gy.reshape(-1, 1), gx.reshape(-1, 1)            # [N, 1]
        y_out, x_out = gy * omega[None, :], gx * omega[None, :]   # [N, quarter]
        pe = torch.cat([torch.sin(y_out), torch.cos(y_out),
                        torch.sin(x_out), torch.cos(x_out)], dim=-1)  # [N, dim]
        return pe.unsqueeze(0)

    def forward(self, images):
        # Pad the *pixel* image to a multiple of (patch_size * window_size) so
        # the resulting patch grid tiles evenly into windows with no ragged
        # edge window.
        images = self._pad_to_multiple(images, self.patch_size * self.window_size)
        feats = self.patch_embed(images)                # [B, dim, grid_h, grid_w]
        _, _, grid_h, grid_w = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)        # [B, grid_h*grid_w, dim]
        tokens = tokens + self._2d_sincos_pos_embed(grid_h, grid_w, images.device)
        for layer in self.layers:
            if self.training:
                # Trades recompute for activation memory: at image_size=896
                # the patch grid is 3136 tokens, and storing every layer's
                # activations for backward (x6 layers) is what was blowing
                # out GPU memory -- not batch_size or resolution per se.
                # Recomputing each layer's forward during backward instead
                # of keeping it all resident avoids that with no change to
                # what the model actually computes (only training-time
                # memory/speed, not eval/inference, which skips this).
                tokens = checkpoint.checkpoint(layer, tokens, grid_h, grid_w,
                                               use_reentrant=False)
            else:
                tokens = layer(tokens, grid_h, grid_w)
        return tokens, grid_h, grid_w


# =========================================================================== #
# Stage 2: global context -- full self-attention over every patch token, now
# feasible because Stage 1 already did the expensive local processing at
# window granularity (see module-level docstring in the methodology doc for
# why local-then-global, not global-only, is the point).
# =========================================================================== #
class GlobalOCREncoder(nn.Module):
    def __init__(self, dim=1024, num_layers=3, nhead=8, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        # Full O(n^2) self-attention over every patch token (n=3136 at
        # image_size=896) is the single most memory-hungry stage in this
        # encoder -- gradient checkpointing here (training only; eval/
        # inference has no backward pass to save memory for, so it just
        # uses the plain nn.TransformerEncoder path) is what actually fixes
        # the CUDA OOM this stage was hitting, not a resolution/batch-size
        # cut (see LocalOCREncoder.forward for the same reasoning).
        if not self.training:
            return self.transformer(x)
        for layer in self.transformer.layers:
            x = checkpoint.checkpoint(layer, x, use_reentrant=False)
        return x


# =========================================================================== #
# Stage 3: compression bridge -- a fixed number of learned queries
# cross-attend into the (variable-length) token sequence, so a full document
# page and a small text patch both come out as the same ``num_output_tokens``
# regardless of input resolution (page size varies a lot more for real PDFs
# than for the fixed-crop images the rest of this project's encoders see).
# =========================================================================== #
class CompressionBridge(nn.Module):
    def __init__(self, dim=1024, num_output_tokens=256, nhead=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_output_tokens, dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, batch_first=True)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B = x.size(0)
        q = self.queries.expand(B, -1, -1)
        pooled, _ = self.attn(q, x, x)
        return self.out_proj(pooled)


# =========================================================================== #
# Stage 4: OCR-specific refinement -- a small transformer stack on the
# compressed tokens, the last step before the encoder's output.
# =========================================================================== #
class OCRFeatureRefiner(nn.Module):
    def __init__(self, dim=1024, num_layers=2, nhead=8, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)


class OCREncoder(nn.Module):
    """
    OCR-specialised document encoder (see OCR_Encoding_Expert_Methodology.md):
    a 4-stage pipeline -- local window-attention feature extraction, global
    context processing, a fixed-size compression bridge, and an OCR-specific
    refinement stack -- producing dense, spatially-aware tokens tuned for
    character recognition rather than object classification. Unlike
    VisionEncoder (a general-purpose photo encoder), every stage here is a
    permanent part of the reusable artifact: there is no throwaway decoder,
    since Stage 4 (OCRFeatureRefiner) already *is* the OCR-specific head the
    rest of this project's encoders would otherwise need a separate decoder
    to learn.

    An optional lightweight handwriting classifier (mean-pooled output
    tokens -> typed/handwritten logits) is also permanent, since it is cheap
    and gives any downstream consumer of this encoder's output a free
    typed-vs-handwritten confidence signal without needing its own model.

    ``forward`` accepts [B, 3, H, W] at any resolution (padded internally to
    a multiple of patch_size * window_size); output is always exactly
    ``num_output_tokens`` regardless of input resolution (see
    CompressionBridge) -- real PDF pages vary in size far more than this
    project's other, fixed-crop image inputs, so a resolution-independent
    token count is the point, not an incidental convenience.
    """

    def __init__(self, in_channels=3, dim=1024, patch_size=16, window_size=7,
                local_layers=6, global_layers=3, refiner_layers=2,
                num_output_tokens=256, nhead=8, dropout=0.1):
        super().__init__()
        assert dim % nhead == 0, f"dim ({dim}) must be divisible by nhead ({nhead})"
        self.dim = dim
        self.num_output_tokens = num_output_tokens

        self.local_encoder = LocalOCREncoder(
            in_channels=in_channels, dim=dim, patch_size=patch_size,
            window_size=window_size, num_layers=local_layers, nhead=nhead,
            dropout=dropout,
        )
        self.global_encoder = GlobalOCREncoder(
            dim=dim, num_layers=global_layers, nhead=nhead, dropout=dropout,
        )
        self.compression_bridge = CompressionBridge(
            dim=dim, num_output_tokens=num_output_tokens, nhead=nhead,
        )
        self.refiner = OCRFeatureRefiner(
            dim=dim, num_layers=refiner_layers, nhead=nhead, dropout=dropout,
        )
        self.handwriting_classifier = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1),
        )

    def handwriting_logits(self, refined_tokens):
        """Raw (pre-sigmoid) typed-vs-handwritten logit from mean-pooled
        output tokens -- exposed separately from ``forward`` so pretraining
        can use BCEWithLogits directly instead of re-deriving logits from a
        sigmoid probability."""
        return self.handwriting_classifier(refined_tokens.mean(dim=1)).squeeze(-1)

    def forward(self, images, return_handwriting_score=False):
        local_tokens, grid_h, grid_w = self.local_encoder(images)
        contextualized = self.global_encoder(local_tokens)
        compressed = self.compression_bridge(contextualized)
        refined = self.refiner(compressed)

        if return_handwriting_score:
            score = torch.sigmoid(self.handwriting_logits(refined))
            return refined, score
        return refined

    def load_weights(self, filename, map_location=None, strict=True):
        state_dict = torch.load(filename, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)

    def store_weights(self, path, filename):
        from weight_io import atomic_torch_save

        os.makedirs(path, exist_ok=True)
        atomic_torch_save(self.state_dict(), os.path.join(path, filename))
