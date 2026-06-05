import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Structure per group: [LinearAttentionLayer × linear_per_group] + [FullAttentionLayer × 1]
# Repeated for num_groups groups.
class LinearAttentionBlock(nn.Module):
    """
    Bidirectional linear attention with per-head learnable decay and output gating.

    Uses ELU+1 feature maps to ensure positive kernel values, enabling the
    closed-form parallel computation:

        O = Q_φ @ (K_φ^T @ V)  /  (Q_φ @ K_φ.sum(seq))

    where φ(x) = ELU(x) + 1 guarantees positivity.

    Per-head scalar decay λ_h = sigmoid(log_decay_h) ∈ (0,1) scales K before
    accumulation, giving each head independent control over token weighting.
    Initialised at ~0.9 (logit ≈ 2.2) to favour high retention by default.

    A learned sigmoid gate (matching the paper's linear gating mechanism)
    modulates the output pointwise.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead  = nhead
        self.d_head = d_model // nhead

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)
        self.dropout   = nn.Dropout(dropout)

        # Per-head learnable decay: sigmoid(log_decay) ∈ (0, 1)
        # logit(0.9) ≈ 2.2 → initialise near 0.9
        self.log_decay = nn.Parameter(torch.full((nhead,), 2.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, N, D]
        Returns:
            [B, N, D]
        """
        B, N, D = x.shape
        H, d = self.nhead, self.d_head

        # Q, K, V: each [B, H, N, d]
        qkv = self.qkv_proj(x).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # ELU+1 feature map → strictly positive kernel values
        Q = F.elu(Q) + 1.0
        K = F.elu(K) + 1.0

        # Per-head decay applied to K
        decay = torch.sigmoid(self.log_decay).view(1, H, 1, 1)
        K = K * decay

        # Parallel linear attention ─────────────────────────────────────
        # KV : [B, H, d, d]   (outer-product accumulation over N)
        KV = torch.einsum("bhnd,bhnv->bhdv", K, V)
        # O  : [B, H, N, d]
        O  = torch.einsum("bhnd,bhdv->bhnv", Q, KV)

        # Normaliser: Z_i = Q_i · (Σ_j K_j)  →  [B, H, N]
        K_sum = K.sum(dim=2)                                  # [B, H, d]
        Z = torch.einsum("bhnd,bhd->bhn", Q, K_sum)
        O = O / Z.unsqueeze(-1).clamp(min=1e-6)
        # ───────────────────────────────────────────────────────────────

        # [B, H, N, d] → [B, N, D]
        O = O.permute(0, 2, 1, 3).reshape(B, N, D)
        O = self.dropout(self.out_proj(O))

        # Sigmoid gate (paper's "linear gating" mechanism)
        gate = torch.sigmoid(self.gate_proj(x))
        return gate * O


class HybridLinearEncoderLayer(nn.Module):
    """
    One complete encoder layer in the hybrid scheme.

    When use_linear_attn=True  : LinearAttentionBlock as the attention sublayer.
    When use_linear_attn=False : standard nn.MultiheadAttention (the periodic
                                  full-attention "anchor" block).

    Both variants follow pre-norm (LayerNorm before each sublayer) and share
    the same wide FFN: Linear(D, 4D) → GELU → Linear(4D, D).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        use_linear_attn: bool = True,
    ):
        super().__init__()
        self.use_linear_attn = use_linear_attn

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if use_linear_attn:
            self.attn = LinearAttentionBlock(d_model, nhead, dropout)
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        # Attention sublayer (pre-norm + residual)
        h = self.norm1(x)
        if self.use_linear_attn:
            h = self.attn(h)
        else:
            h, _ = self.attn(h, h, h, key_padding_mask=src_key_padding_mask)
        x = x + h

        # FFN sublayer (pre-norm + residual)
        x = x + self.ffn(self.norm2(x))
        return x


class HybridLinearEncoder(nn.Module):
    """
    Full hybrid encoder: interleaves linear and full attention layers in the
    Ring-linear pattern.

    One group = [LinearAttentionLayer * linear_per_group] + [1 FullAttentionLayer]

    Total layers = num_groups * (linear_per_group + 1).
    Default (num_groups=2, linear_per_group=4) gives 10 layers: 8 linear + 2 full.

    Dimension rules:
        d_model         : D
        dim_feedforward : 4 * D  (wide FFN)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_groups: int = 2,
        linear_per_group: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(num_groups):
            for _ in range(linear_per_group):
                layers.append(HybridLinearEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout,
                    use_linear_attn=True,
                ))
            layers.append(HybridLinearEncoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                use_linear_attn=False,
            ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x

class AdaptiveAttentionPooling(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_queries: int = 128,
        min_queries: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        salience_threshold: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_queries = max_queries
        self.min_queries = min_queries
        self.salience_threshold = salience_threshold

        self.query_tokens = nn.Parameter(
            torch.randn(1, max_queries, d_model) * 0.02
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.salience_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, encoder_out, key_padding_mask=None, total_seq_len=None):
        B = encoder_out.size(0)

        effective_max = self.max_queries
        if total_seq_len is not None:
            effective_max = min(self.max_queries, total_seq_len)
        effective_min = min(self.min_queries, effective_max)

        queries = self.query_tokens[:, :effective_max, :].expand(B, -1, -1)

        attn_out, _ = self.cross_attn(
            query=queries,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=key_padding_mask,
        )
        queries = self.norm1(queries + attn_out)
        queries = self.norm2(queries + self.ffn(queries))

        # Per-query salience: [B, Q, 1] ∈ (0, 1)
        salience = torch.sigmoid(self.salience_head(queries))

        # Each cell is gated by its own salience score (soft pruning)
        queries = queries * salience

        # Active cell count = number of queries above threshold, clamped to [min, max]
        num_active_int = (salience.squeeze(-1) > self.salience_threshold).sum(dim=1)
        num_active_int = num_active_int.clamp(min=effective_min, max=effective_max)

        K_trim = int(num_active_int.max().item())
        latents = queries[:, :K_trim, :]

        positions = torch.arange(K_trim, device=encoder_out.device)
        # True = padding (inactive slot)
        active_mask = positions.unsqueeze(0) >= num_active_int.unsqueeze(1)
        return latents, num_active_int, active_mask


# ══════════════════════════════════════════════════════════════════════════════
# Encoding Combination
# ══════════════════════════════════════════════════════════════════════════════

def createEncCombine(dim=512):
    return EncodingCombination(
        d_model=dim,
        dim_feedforward=dim * 4,
        min_latent_tokens=4,
        max_latent_tokens=128,
        num_groups=2,
        linear_per_group=4,
    )


class EncodingCombination(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_latent_tokens: int = 128,
        min_latent_tokens: int = 4,
        num_groups: int = 2,
        linear_per_group: int = 4,
    ):
        super().__init__()
        self.d_model = d_model

        self.encoder = HybridLinearEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_groups=num_groups,
            linear_per_group=linear_per_group,
            dropout=dropout,
        )

        self.pool = AdaptiveAttentionPooling(
            d_model=d_model,
            max_queries=max_latent_tokens,
            min_queries=min_latent_tokens,
            nhead=nhead,
            dropout=dropout,
        )

    def forward(self, input_tensors, padding_masks=None):
        total_seq_len = sum(t.size(1) for t in input_tensors)
        combined_seq = torch.cat(input_tensors, dim=1)
        combined_mask = (
            torch.cat(padding_masks, dim=1)
            if padding_masks is not None
            else None
        )
        memory = self.encoder(combined_seq, src_key_padding_mask=combined_mask)

        return self.pool(
            memory,
            key_padding_mask=combined_mask,
            total_seq_len=total_seq_len,
        )

    def load_weights(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)

    def store_weights(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))
