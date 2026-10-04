import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.gelu(x + residual)

def createVisionEncoder(dim=1024):
    embed_dim = dim // 2
    return VisionEncoder(in_channels=3, embed_dim=embed_dim, output_dim=dim, nhead=max(1, embed_dim // 32))

# =========================================================================== #
# Axial RoPE self-attention (DINOv3-style).
#
# Each patch token carries a normalized (row, col) coordinate in [-1, 1]; each
# attention head's dimension is split into quarters and rotated independently
# by row/col position (2 axes x sin/cos pairing = 4-way split), so the
# relative position between two patches shows up natively in the q.k dot
# product without any separate learned positional bias. Non-spatial tokens
# (CLS, registers) get coordinate (0, 0) -> zero rotation angle -> identity,
# so they still participate in attention without carrying a spatial meaning.
# =========================================================================== #
class AxialRoPEAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.nhead = nhead
        self.d_head = d_model // nhead
        assert self.d_head % 4 == 0, (
            f"head dim ({self.d_head}) must be divisible by 4 for axial RoPE "
            "(2 spatial axes x sin/cos pairing)"
        )
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def _rope_cos_sin(self, coords):
        # coords: [N, 2] (row, col), already (possibly jittered) in [-box, box]
        quarter = self.d_head // 4
        inv_freq = 1.0 / (100.0 ** (torch.arange(0, quarter, device=coords.device).float() / quarter))
        row_freqs = coords[:, 0:1] * inv_freq[None, :]     # [N, quarter]
        col_freqs = coords[:, 1:2] * inv_freq[None, :]     # [N, quarter]
        freqs = torch.cat([row_freqs, col_freqs], dim=-1)  # [N, d_head/2]
        return torch.cos(freqs), torch.sin(freqs)

    @staticmethod
    def _apply_rope(x, cos, sin):
        # x: [B, H, N, d_head]; cos/sin: [N, d_head/2]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)

    def forward(self, x, coords):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nhead, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = self._rope_cos_sin(coords)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class RoPEEncoderLayer(nn.Module):
    """Pre-norm transformer block using AxialRoPEAttention instead of the
    absolute-position self-attention nn.TransformerEncoderLayer would use."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = AxialRoPEAttention(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, coords):
        x = x + self.attn(self.norm1(x), coords)
        x = x + self.mlp(self.norm2(x))
        return x


class RoPETransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, coords):
        for layer in self.layers:
            x = layer(x, coords)
        return x


class VisionEncoder(nn.Module):
    """
    DINOv3-inspired ViT-style vision encoder.

    Token layout fed to the transformer: [CLS, register_1..register_4,
    patch_1..patch_N]. CLS + registers are learned, non-spatial tokens (see
    Darcet et al., "Vision Transformers Need Registers"); registers exist
    purely to absorb the high-norm "attention sink" activations that would
    otherwise land on patch tokens, keeping patch-level (dense) features
    clean. Position is carried entirely by axial RoPE inside attention (see
    AxialRoPEAttention) rather than an additive positional embedding, so the
    encoder generalises to resolutions/aspect ratios never seen in training.

    Two dedicated final LayerNorms (``local_norm`` / ``global_norm``,
    selected via ``crop_type``) are applied to the backbone output: local and
    global crops have different statistics (different resolutions and, for
    local crops, only ever partial views of the scene), and DINOv3 found a
    dedicated norm per crop type stabilises training and improves dense
    performance over sharing one norm across both.

    Video (multi-frame) input reuses this exact per-frame pass -- same CLS,
    registers, RoPE, masking, and crop-type norm -- run independently on
    every frame, then a 1D sincos embedding over the *time* axis is added on
    top of the (already-projected) per-frame tokens before flattening across
    frames. This deliberately avoids a second, separately-trained temporal
    encoder: single-frame image knowledge is what covers the video, extended
    in time only by a fixed (non-learned) positional signal.
    """

    def __init__(self, in_channels=3, embed_dim=256, output_dim=512, patch_size=16,
                internal_layers=3, nhead=8, num_registers=4):
        super().__init__()
        assert embed_dim % nhead == 0, f"embed_dim ({embed_dim}) must be divisible by nhead ({nhead})"
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_registers = num_registers

        self.deep_stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            ResBlock(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            ResBlock(256),

            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            ResBlock(embed_dim),
            ResBlock(embed_dim)
        )

        # CLS + register tokens (see class docstring); concatenated ahead of
        # the patch tokens before the RoPE transformer.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_registers, embed_dim))

        self.spatial_contextualizer = RoPETransformerEncoder(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            num_layers=internal_layers,
        )

        # Dedicated per-crop-type final norms (see class docstring).
        self.local_norm = nn.LayerNorm(embed_dim)
        self.global_norm = nn.LayerNorm(embed_dim)

        self.output_proj = nn.Linear(embed_dim, output_dim)

    def _pad_to_multiple(self, x):
        H, W = x.shape[-2:]
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)

    def _get_1d_sincos_pos_embed(self, seq_len, device, dim=None):
        dim = dim or self.embed_dim
        pos = torch.arange(seq_len, dtype=torch.float32, device=device)
        omega = torch.exp(-torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (torch.log(torch.tensor(10000.0)) / dim))
        out = pos[:, None] * omega[None, :]
        return torch.stack([torch.sin(out), torch.cos(out)], dim=2).flatten(1)

    # --------------------------------------------------------------------- #
    # Axial RoPE coordinates: CLS + registers get (0, 0) -> identity
    # rotation; patches get a normalized (row, col) in [-box, box]. RoPE-box
    # jittering (training only) rescales the box per forward pass to a
    # random size in [0.5, 2.0], improving robustness to resolution/aspect
    # ratio changes at inference (see DINOv3 Sec. 3.2).
    # --------------------------------------------------------------------- #
    def _rope_coords(self, grid_h, grid_w, device):
        box = 1.0
        if self.training:
            box = float(torch.empty(1).uniform_(0.5, 2.0).item())
        ys = torch.linspace(-box, box, grid_h, device=device)
        xs = torch.linspace(-box, box, grid_w, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        patch_coords = torch.stack([gy.flatten(), gx.flatten()], dim=-1)
        special = torch.zeros(1 + self.num_registers, 2, device=device)
        return torch.cat([special, patch_coords], dim=0)

    # --------------------------------------------------------------------- #
    # Patch masking (iBOT-style): zeroes random patch_size x patch_size
    # pixel blocks *before* the CNN stem patchifies the image, at a ratio
    # sampled uniformly per image in ``mask_ratio_range``. Only ever applied
    # to the student's view -- the teacher must see the unmasked image, so
    # its patch tokens are valid reconstruction targets at the masked
    # positions. Callers toggle this via ``forward(..., mask=...)``.
    # --------------------------------------------------------------------- #
    def mask_patches(self, x, mask_ratio_range=(0.1, 0.5)):
        x = self._pad_to_multiple(x)
        B, _, H, W = x.shape
        ps = self.patch_size
        grid_h, grid_w = H // ps, W // ps
        ratio = torch.empty(B, device=x.device).uniform_(*mask_ratio_range)
        patch_mask = torch.rand(B, grid_h, grid_w, device=x.device) < ratio.view(B, 1, 1)
        mask_expanded = patch_mask.repeat_interleave(ps, dim=1).repeat_interleave(ps, dim=2)
        masked_x = x * (~mask_expanded).unsqueeze(1).float()
        return masked_x, patch_mask

    def forward(self, x, mask=False, crop_type="global", return_all_tokens=False):
        if x.dim() == 5:
            return self._forward_video(x, mask=mask, crop_type=crop_type,
                                       return_all_tokens=return_all_tokens)
        return self._forward_image(x, mask=mask, crop_type=crop_type,
                                   return_all_tokens=return_all_tokens)

    def _forward_image(self, x, mask=False, crop_type="global", return_all_tokens=False):
        B = x.size(0)
        device = x.device

        patch_mask = None
        if mask:
            x, patch_mask = self.mask_patches(x)
        else:
            x = self._pad_to_multiple(x)

        features = self.deep_stem(x)
        _, _, grid_h, grid_w = features.shape
        features = features.flatten(2).transpose(1, 2)          # [B, N, D]

        cls = self.cls_token.expand(B, -1, -1)
        registers = self.register_tokens.expand(B, -1, -1)
        tokens = torch.cat([cls, registers, features], dim=1)    # [B, 1+R+N, D]

        coords = self._rope_coords(grid_h, grid_w, device)
        tokens = self.spatial_contextualizer(tokens, coords)

        norm = self.local_norm if crop_type == "local" else self.global_norm
        tokens = norm(tokens)
        tokens = self.output_proj(tokens)

        if return_all_tokens:
            return (tokens, patch_mask) if mask else tokens

        return tokens[:, 1 + self.num_registers:, :]

    def _forward_video(self, x, mask=False, crop_type="global", return_all_tokens=False):
        """Runs the exact single-frame pass (CLS, registers, RoPE, masking,
        crop-type norm -- same weights, no separate temporal encoder) on
        every frame independently, then extends that per-frame knowledge
        across the video by adding a 1D sincos embedding over time on top of
        the resulting per-frame tokens before flattening across frames."""
        B, T, C, H, W = x.shape
        device = x.device
        frames = x.reshape(B * T, C, H, W)

        result = self._forward_image(frames, mask=mask, crop_type=crop_type,
                                     return_all_tokens=True)
        tokens, patch_mask = result if mask else (result, None)

        S, D = tokens.shape[1], tokens.shape[2]
        tokens = tokens.view(B, T, S, D)

        temporal_pos = self._get_1d_sincos_pos_embed(T, device, dim=D).view(1, T, 1, D)
        tokens = tokens + temporal_pos

        if not return_all_tokens:
            tokens = tokens[:, :, 1 + self.num_registers:, :]
        tokens = tokens.reshape(B, T * tokens.size(2), D)

        if mask:
            if patch_mask is not None:
                patch_mask = patch_mask.view(B, T, patch_mask.size(1), patch_mask.size(2))
            return tokens, patch_mask
        return tokens

    def load_weights(self, filename, map_location=None, strict=True):
        state_dict = torch.load(filename, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)

    def store_weights(self, path, filename):
        from weight_io import atomic_torch_save

        os.makedirs(path, exist_ok=True)
        atomic_torch_save(self.state_dict(), os.path.join(path, filename))
# # Initialize encoder
# encoder = createVisionEncoder()

# # 1. Processing an Image (Not strictly a multiple of 16)
# # Height=250, Width=300 -> Will pad to 256x304
# dummy_image = torch.randn(2, 3, 250, 300) 
# img_embeddings = encoder(dummy_image)
# print(f"Image output shape: {img_embeddings.shape}") 
# # seq_len = (256/16) * (304/16) = 16 * 19 = 304
# # Expected output: [2, 304, 512]

# # 2. Processing a Video (Not strictly a multiple of 16)
# # 10 frames, Height=100, Width=100 -> Will pad to 112x112
# dummy_video = torch.randn(4, 10, 3, 100, 100) 
# vid_embeddings = encoder(dummy_video)
# print(f"Video output shape: {vid_embeddings.shape}") 
# # seq_len = 10 (frames) * (112/16) * (112/16) = 10 * 7 * 7 = 490
# # Expected output: [4, 490, 512]