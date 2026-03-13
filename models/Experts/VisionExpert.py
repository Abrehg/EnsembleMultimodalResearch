import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def createVisionGenExpert(d_model=512, base_res=8, max_res=256, max_frames=16):
    return VisionGenExpert(
        d_model=d_model,
        base_res=base_res,
        max_res=max_res,
        max_frames=max_frames,
    )

class SpatialCrossAttention(nn.Module):
    """
    Learnable spatial query grid that cross-attends into the latent
    state to gather generation instructions.
    """

    def __init__(self, d_model, base_res, nhead=8, dropout=0.1):
        super().__init__()
        self.base_res = base_res
        num_queries = base_res * base_res

        self.spatial_queries = nn.Parameter(
            torch.randn(1, num_queries, d_model) * 0.02
        )

        # Learnable 2D position encoding for the query grid
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_queries, d_model) * 0.02
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, state):
        """
        Args:
            state : [B, K, D]

        Returns:
            features : [B, H*W, D] — spatial features for decoding
        """
        B = state.size(0)
        queries = self.spatial_queries.expand(B, -1, -1) + self.pos_embed

        # Cross-attend into state
        attn_out, _ = self.cross_attn(
            query=queries, key=state, value=state
        )
        features = self.norm1(queries + attn_out)

        # Self-attend for spatial coherence
        self_out, _ = self.self_attn(features, features, features)
        features = self.norm2(features + self_out)

        features = self.norm3(features + self.ffn(features))

        return features


class TemporalExpander(nn.Module):
    """
    Expands a spatial feature grid into multiple frames by cross-
    attending from temporal query tokens into the spatial features.

    Each temporal query learns to represent a different point in time,
    producing frame-specific spatial features.
    """

    def __init__(self, d_model, max_frames, nhead=8, dropout=0.1):
        super().__init__()
        self.max_frames = max_frames

        # One query per possible frame
        self.temporal_queries = nn.Parameter(
            torch.randn(1, max_frames, d_model) * 0.02
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, spatial_features, num_frames):
        """
        Args:
            spatial_features : [B, H*W, D]
            num_frames       : int, how many frames to produce

        Returns:
            temporal_features : [B*T, H*W, D] — ready for the decoder,
                                each frame is a separate batch element
        """
        B, HW, D = spatial_features.shape
        T = min(num_frames, self.max_frames)

        queries = self.temporal_queries[:, :T, :]               # [1, T, D]
        queries = queries.expand(B, -1, -1)                     # [B, T, D]

        # Each temporal query cross-attends into spatial features
        # to produce a frame-specific modulation
        attn_out, _ = self.cross_attn(
            query=queries, key=spatial_features, value=spatial_features
        )
        temporal_mods = self.norm(queries + attn_out)            # [B, T, D]

        # Broadcast: each frame = spatial_features + temporal modulation
        spatial_exp = spatial_features.unsqueeze(1).expand(-1, T, -1, -1)
        temporal_exp = temporal_mods.unsqueeze(2).expand(-1, -1, HW, -1)
        frames = spatial_exp + temporal_exp                      # [B, T, HW, D]

        # Flatten batch and time for the decoder
        frames = frames.reshape(B * T, HW, D)                   # [B*T, HW, D]

        return frames, T


class UpsampleBlock(nn.Module):
    """ConvTranspose2d + GroupNorm + GELU upsampling block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.up(x)))


class PixelDecoder(nn.Module):
    """
    Progressive CNN decoder that upsamples base-resolution feature maps
    to the target resolution.

    Dynamically builds the right number of upsample stages based on
    base_res and max_res (each stage doubles spatial dimensions).
    Produces multi-channel output (1 for masks, 3 for RGB, 4 for RGBA).
    """

    def __init__(self, d_model, base_res, max_res, max_channels=3):
        super().__init__()
        self.base_res = base_res
        self.max_res = max_res

        num_stages = int(math.log2(max_res // base_res))
        channels = [d_model] + [
            max(64, d_model // (2 ** i)) for i in range(1, num_stages + 1)
        ]

        self.input_proj = nn.Linear(d_model, d_model)

        self.upsample_stages = nn.ModuleList([
            UpsampleBlock(channels[i], channels[i + 1])
            for i in range(num_stages)
        ])

        # Separate output heads for different channel counts
        last_ch = channels[-1]
        self.output_heads = nn.ModuleDict({
            "1": nn.Conv2d(last_ch, 1, 3, padding=1),    # mask
            "3": nn.Conv2d(last_ch, 3, 3, padding=1),    # RGB
            "4": nn.Conv2d(last_ch, 4, 3, padding=1),    # RGBA
        })

    def forward(self, features, out_channels=3, out_height=None,
                out_width=None):
        """
        Args:
            features     : [B, H*W, D]
            out_channels : 1 (mask), 3 (RGB), or 4 (RGBA)
            out_height   : target height (defaults to max_res)
            out_width    : target width  (defaults to max_res)

        Returns:
            pixels : [B, C, H_out, W_out]
        """
        B = features.size(0)
        h = w = self.base_res

        x = self.input_proj(features)
        x = x.reshape(B, h, w, -1).permute(0, 3, 1, 2)        # [B, D, h, w]

        for stage in self.upsample_stages:
            x = stage(x)

        # Select the right output head
        head_key = str(out_channels)
        if head_key not in self.output_heads:
            raise ValueError(
                f"Unsupported channel count {out_channels}. "
                f"Supported: 1 (mask), 3 (RGB), 4 (RGBA)."
            )
        pixels = self.output_heads[head_key](x)

        # Resize to exact target if specified
        if out_height is None:
            out_height = self.max_res
        if out_width is None:
            out_width = self.max_res

        if pixels.shape[2] != out_height or pixels.shape[3] != out_width:
            pixels = F.interpolate(
                pixels, size=(out_height, out_width),
                mode="bilinear", align_corners=False,
            )

        return pixels


class ModePredictor(nn.Module):
    """
    Predicts generation parameters from the latent state:
        - mode: mask (1ch), image (3ch), or video (3ch + frames)
        - num_frames: how many frames if video
        - height / width hints (optional, for non-square outputs)
    """

    def __init__(self, d_model, max_frames):
        super().__init__()
        self.max_frames = max_frames

        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4,
            batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(d_model)

        # Predict: [mask_logit, image_logit, video_logit, frame_count]
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 4),
        )

    def forward(self, state):
        """
        Args:
            state : [B, K, D]

        Returns:
            channels   : [B] int — 1 or 3
            num_frames : [B] int — 1 for image/mask, >1 for video
        """
        B = state.size(0)
        query = self.pool_query.expand(B, -1, -1)

        attn_out, _ = self.pool_attn(query, state, state)
        summary = self.pool_norm(query + attn_out).squeeze(1)   # [B, D]

        preds = self.head(summary)                              # [B, 4]

        # Mode selection: softmax over first 3 logits
        mode_logits = preds[:, :3]                              # [B, 3]
        mode_idx = mode_logits.argmax(dim=-1)                   # [B]

        # channels: mask=1, image=3, video=3
        channels = torch.where(mode_idx == 0, 1, 3)

        # frames: mask/image=1, video=predicted count
        raw_frames = torch.sigmoid(preds[:, 3]) * self.max_frames
        frame_count = raw_frames.round().long().clamp(min=2)
        num_frames = torch.where(mode_idx == 2, frame_count, 1)

        return channels, num_frames


class VisionGenExpert(nn.Module):
    """
    Vision generation expert for the routing loop.

    Generates visual output — masks, images, or video — from the
    latent working-memory state.  A mode predictor examines the state
    and decides what to produce, then the generation pipeline builds it:

        state → spatial queries → (optional temporal expansion) →
        progressive pixel decoder → output

    The generation mode is predicted from the state itself, so the
    router just needs to select this expert; the expert figures out
    what kind of visual to produce.

    Compatible with ExpertRegistry:
        registry.add_expert("vision_gen", createVisionGenExpert())

    Forward signature matches non-terminal expert convention:
        output, state = expert(state)

    The output tensor shape depends on mode:
        mask  : [B, 1, H, W]
        image : [B, 3, H, W]
        video : [B, T, 3, H, W]

    Args:
        d_model    : latent dimension (must match state width)
        base_res   : base spatial resolution of query grid
        max_res    : maximum output resolution
        max_frames : maximum video length in frames
        nhead      : attention heads
        dropout    : dropout rate
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        base_res: int = 8,
        max_res: int = 256,
        max_frames: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.base_res = base_res
        self.max_res = max_res
        self.max_frames = max_frames

        # Predict what to generate from state
        self.mode_predictor = ModePredictor(d_model, max_frames)

        # Build spatial features from state
        self.spatial_attn = SpatialCrossAttention(
            d_model, base_res, nhead=nhead, dropout=dropout
        )

        # Expand to video frames if needed
        self.temporal_expander = TemporalExpander(
            d_model, max_frames, nhead=nhead, dropout=dropout
        )

        # Decode features to pixels
        self.pixel_decoder = PixelDecoder(
            d_model, base_res, max_res, max_channels=3
        )

        # State refinement: update state with what was generated
        self.state_update_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.state_update_norm = nn.LayerNorm(d_model)

    def forward(self, state, channels=None, num_frames=None,
                height=None, width=None):
        """
        Args:
            state      : [B, K, d_model]
            channels   : override output channels (1, 3, or 4).
                         If None, predicted from state.
            num_frames : override frame count.
                         If None, predicted from state.
            height     : output height (defaults to max_res)
            width      : output width  (defaults to max_res)

        Returns:
            output : visual tensor — shape depends on mode:
                       [B, C, H, W]       for mask / image
                       [B, T, C, H, W]    for video
            state  : [B, K, d_model] — refined state
        """
        B = state.size(0)

        # --- 1. Decide what to generate ---
        if channels is None or num_frames is None:
            pred_ch, pred_frames = self.mode_predictor(state)

        if channels is None:
            channels_per_sample = pred_ch                       # [B]
        else:
            channels_per_sample = torch.full(
                (B,), channels, dtype=torch.long, device=state.device
            )

        if num_frames is None:
            frames_per_sample = pred_frames                     # [B]
        else:
            frames_per_sample = torch.full(
                (B,), num_frames, dtype=torch.long, device=state.device
            )

        # For batched decoding, use the max across the batch
        out_channels = channels_per_sample.max().item()
        out_frames = frames_per_sample.max().item()

        # --- 2. Build spatial features ---
        spatial_features = self.spatial_attn(state)              # [B, HW, D]

        # --- 3. Generate pixels ---
        if out_frames > 1:
            # Video: expand spatially into temporal frames
            frame_features, T = self.temporal_expander(
                spatial_features, out_frames
            )                                                   # [B*T, HW, D]

            pixels = self.pixel_decoder(
                frame_features, out_channels=out_channels,
                out_height=height, out_width=width,
            )                                                   # [B*T, C, H, W]

            H_out, W_out = pixels.shape[2], pixels.shape[3]
            output = pixels.reshape(B, T, out_channels, H_out, W_out)
        else:
            # Image or mask: decode directly
            output = self.pixel_decoder(
                spatial_features, out_channels=out_channels,
                out_height=height, out_width=width,
            )                                                   # [B, C, H, W]

        # --- 4. Update state with generation context ---
        gen_summary = spatial_features                          # [B, HW, D]
        update_out, _ = self.state_update_attn(
            query=state, key=gen_summary, value=gen_summary
        )
        state = self.state_update_norm(state + update_out)

        return output, state

    # ------------------------------------------------------------------
    # Convenience methods for explicit generation modes
    # ------------------------------------------------------------------

    def generate_mask(self, state, height=None, width=None):
        return self.forward(
            state, channels=1, num_frames=1,
            height=height, width=width,
        )

    def generate_image(self, state, height=None, width=None):
        return self.forward(
            state, channels=3, num_frames=1,
            height=height, width=width,
        )

    def generate_video(self, state, num_frames=8, height=None, width=None):
        return self.forward(
            state, channels=3, num_frames=num_frames,
            height=height, width=width,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def store_weights(self, path, filename="vision_gen_expert.pt"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))
        print(f"Saved VisionGenExpert to {path}/{filename}")

    def load_weights(self, filepath):
        state_dict = torch.load(filepath, map_location="cpu")
        self.load_state_dict(state_dict)
        print(f"Loaded VisionGenExpert from {filepath}")