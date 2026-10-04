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
        # Bounds O's magnitude before it's added to the residual stream (see
        # forward()) -- unlike softmax attention, whose output is a convex
        # combination of V and thus naturally bounded by the input's own
        # scale, this closed-form linear-attention O = Q_phi @ (K_phi^T @ V) / Z
        # has no such guarantee. A real trained checkpoint was found with O
        # reaching norms in the hundreds of millions (qkv_proj's learned
        # weights drifting large during training, with nothing to constrain
        # them), added straight into the residual stream with no
        # normalization at all -- each layer's huge, largely input-independent
        # contribution was overwriting the accumulated residual rather than
        # refining it, washing out genuine per-input direction across the
        # whole encoder stack. This is the standard fix for attention/sublayer
        # formulations that lack softmax's built-in boundedness.
        self.out_norm = nn.LayerNorm(d_model)

        # Per-head learnable decay: sigmoid(log_decay) ∈ (0, 1)
        # logit(0.9) ≈ 2.2 → initialise near 0.9
        self.log_decay = nn.Parameter(torch.full((nhead,), 2.2))

        # ReZero-style residual scale (Bachlechner et al. 2020), initialised
        # to exactly 0. A real trained checkpoint (t2t_stage1, epoch 18),
        # traced layer-by-layer (diagnose_state_collapse.py), showed
        # within/across-source cosine similarity of this encoder's pooled
        # output collapsing from a healthy ~0.18 gap (at the frozen text
        # encoder's own output, genuinely different per input) to ~0.003
        # after just ONE application of this block -- before any of
        # out_norm's bounded-magnitude fix could plausibly be blamed (that
        # fix targets exploding norms; this is a *directional* collapse,
        # where the gated, normalized output converges toward a nearly
        # input-independent vector regardless of magnitude). Since this
        # output gets ADDED into the residual stream (see
        # HybridLinearEncoderLayer.forward), even a "reasonable" (LayerNorm-
        # bounded) but under-differentiated contribution is enough to
        # homogenize the running residual across genuinely different
        # inputs -- and stacked 8 times (this block appears 8 of 10 layers),
        # it compounds fast. This closed-form attention's own structure is a
        # plausible reason: O_i = Q_i . (sum_j K_j (x) V_j) / Z_i pools
        # every position's K/V into ONE shared per-head statistic before
        # each query even sees it, and ELU+1's positive-only feature map
        # gives it far less dynamic range to produce a sharply
        # input-selective combination than softmax attention has. None of
        # this codebase's existing anti-collapse terms (koleo_loss,
        # salience_diversity/margin_loss, decoder_diversity_loss) regularize
        # this block's own output directly -- every one of them acts on
        # quantities several layers downstream of where the collapse
        # actually originates, which is a plausible reason three rounds of
        # those fixes never moved unique_pred_fraction off its collapse
        # floor. Starting residual_scale at 0 means this block contributes
        # NOTHING to the residual at initialization -- 100% of the frozen
        # encoder's real, differentiated signal survives untouched -- and
        # training can only open it up as far as it actually earns via
        # gradient signal, instead of overwriting a diverse residual with an
        # undifferentiated one from the very first step.
        self.residual_scale = nn.Parameter(torch.zeros(1))

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
        O = self.out_norm(O)   # bound O's scale BEFORE gating (see __init__'s comment)

        # Sigmoid gate (paper's "linear gating" mechanism)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.residual_scale * gate * O


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
            # ReZero-style residual scale for the full-attention anchor's
            # OWN attention sublayer -- see ffn_residual_scale's comment
            # below for the fuller rationale. Needed here specifically
            # (LinearAttentionBlock already carries its own internal
            # residual_scale) because a second diagnose_state_collapse.py
            # trace, run AFTER the linear-attention layers were fully
            # neutralized (proven an exact identity function at init: layers
            # 01-04 matched '00_input' to 4 decimal places), showed the
            # within/across-source gap collapsing precisely at the FIRST
            # full-attention anchor (layer 05: gap 0.1792 -> -0.0047, on an
            # input that was STILL healthy going in) -- and layers 06-09
            # (linear-attention, still neutralized) correctly passed that
            # collapsed state through unchanged, confirming layer 05 itself
            # is where it happened. This falsifies the original "linear
            # attention's closed-form math specifically" hypothesis: this is
            # a plain, standard nn.MultiheadAttention layer, and it collapsed
            # just as badly once actually isolated. The more accurate
            # picture: ANY freshly-initialized sublayer in this stack,
            # trained for 18 epochs under the same collapse dynamics, ends
            # up here -- not something unique to the linear-attention
            # mechanism. nn.MultiheadAttention needs its own external gate
            # since, unlike the custom LinearAttentionBlock, it's a built-in
            # module with no internal parameter to add one to.
            self.attn_residual_scale = nn.Parameter(torch.zeros(1))

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # ReZero-style residual scale for THIS layer's FFN sublayer -- now
        # applied uniformly to every layer (not just linear-attention ones,
        # as it originally was): the same trace above showed a full-
        # attention anchor's LAYER-level output collapsing once it received
        # a genuinely diverse input, and per-sublayer attribution wasn't
        # possible from that trace (it only measures once per whole layer),
        # so there's no basis yet to exempt the full-attention layers' own
        # FFN the way there was for the two full-attention layers' larger
        # role (see attn_residual_scale just above). Same conservative
        # "start at zero, only earn a nonzero contribution via gradient
        # signal" treatment applied consistently everywhere in this stack.
        self.ffn_residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        # Attention sublayer (pre-norm + residual)
        h = self.norm1(x)
        if self.use_linear_attn:
            h = self.attn(h)
        else:
            h, _ = self.attn(h, h, h, key_padding_mask=src_key_padding_mask)
            h = self.attn_residual_scale * h
        x = x + h

        # FFN sublayer (pre-norm + residual)
        ffn_out = self.ffn_residual_scale * self.ffn(self.norm2(x))
        x = x + ffn_out
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
        salience_floor: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_queries = max_queries
        self.min_queries = min_queries
        self.salience_threshold = salience_threshold
        # Floors the salience gate below (see forward()) so it can never fully
        # close -- a real trained checkpoint was found with salience_head's
        # logits saturated at ~-27 for every input (sigmoid(-27) ~ 0),
        # multiplying the entire pooled representation to numerically zero
        # regardless of content ("dying gate": once saturated this deep,
        # sigmoid's own vanishing gradient there means there's no path back).
        # The floor bounds salience to [salience_floor, 1.0] instead of
        # (0, 1), keeping a real gradient path open even if the gate drifts
        # toward "mostly closed" again.
        self.salience_floor = salience_floor

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
        # ReZero-style residual scale (init 0) for JUST this FFN sublayer --
        # deliberately NOT applied to the cross-attention step above
        # (queries + attn_out in forward()). At init, `queries` going into
        # THAT residual add is query_tokens -- a fixed, learned parameter
        # identical for every input by construction, carrying zero
        # per-input information -- so scaling attn_out (the only
        # input-dependent term in that sum) toward zero would suppress the
        # one thing making this pooling step input-dependent at all, making
        # collapse WORSE, not better. By the time this FFN runs, `queries`
        # already carries genuine cross-attended, per-input content from
        # that step, so the usual "preserve what's already there, let the
        # FFN only contribute as much as it earns" reasoning (see
        # EncCombine.py's HybridLinearEncoderLayer) applies cleanly here.
        # Diagnosed via diagnose_state_collapse.py: once everything upstream
        # of this module was confirmed clean, this module's own pooled
        # output still showed a real (partial, not total) within/across-
        # source gap drop, 0.1790 -> 0.0516.
        self.ffn_residual_scale = nn.Parameter(torch.zeros(1))

        self.salience_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        # Bias the gate toward "open" at init (sigmoid(2.0) ~ 0.88 pre-floor)
        # rather than PyTorch's default (unbiased toward open or closed) --
        # gives training less room to drift toward the collapsed "always
        # closed" regime early on, when gradients are noisiest. Only the
        # final layer's bias is touched; weights keep their default init.
        nn.init.constant_(self.salience_head[-1].bias, 2.0)

    def forward(self, encoder_out, key_padding_mask=None, total_seq_len=None,
               return_gate_stats=False):
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
        queries = self.norm2(queries + self.ffn_residual_scale * self.ffn(queries))

        # Per-query salience: [B, Q, 1] -- floored to [salience_floor, 1.0]
        # rather than raw sigmoid's (0, 1) so the gate can never fully close
        # (see salience_floor's comment in __init__). raw_logits is kept
        # separately (pre-sigmoid, pre-floor) since the floor only bounds the
        # *output* range -- it does nothing to stop the raw logit itself from
        # saturating deep into either tail, where sigmoid's own vanishing
        # gradient makes the gate un-recoverable ("dead gate", observed on
        # both rails in real training: salience -> 1.0 uniformly before the
        # floor existed, salience -> salience_floor uniformly after). Callers
        # that need a live gradient path to regularize this (see
        # return_gate_stats below) work from raw_logits, not from salience.
        raw_logits = self.salience_head(queries)
        salience = self.salience_floor + (1 - self.salience_floor) * torch.sigmoid(
            raw_logits)

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
        if return_gate_stats:
            gate_stats = {
                "salience_mean": salience.mean().item(),
                "salience_min": salience.min().item(),
                "salience_max": salience.max().item(),
                # Differentiable, unlike the three keys above -- lets a
                # caller (see text_to_text.py's salience_diversity_loss/
                # salience_margin_loss) regularize the gate directly instead
                # of only observing its already-collapsed output.
                "raw_logits": raw_logits,
            }
            return latents, num_active_int, active_mask, gate_stats
        return latents, num_active_int, active_mask


# ══════════════════════════════════════════════════════════════════════════════
# Encoding Combination
# ══════════════════════════════════════════════════════════════════════════════

def createEncCombine(dim=1024):
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

    def forward(self, input_tensors, padding_masks=None, return_gate_stats=False,
               return_memory=False):
        total_seq_len = sum(t.size(1) for t in input_tensors)
        combined_seq = torch.cat(input_tensors, dim=1)
        combined_mask = (
            torch.cat(padding_masks, dim=1)
            if padding_masks is not None
            else None
        )
        memory = self.encoder(combined_seq, src_key_padding_mask=combined_mask)

        pool_out = self.pool(
            memory,
            key_padding_mask=combined_mask,
            total_seq_len=total_seq_len,
            return_gate_stats=return_gate_stats,
        )
        # return_memory: exposes this encoder's own pre-pooling output (see
        # text_to_text.py's encode_to_state/encoder_koleo_loss) -- added once
        # a real harness run (slurm-4800583) showed the collapse repeatedly
        # relapsing well after the ReZero architecture fix landed, with
        # every existing anti-collapse term (koleo_loss, decoder_diversity_loss,
        # salience terms) acting several layers downstream of this encoder's
        # own output rather than on it directly. Appended at the end rather
        # than returned separately so this stays a single call (the encoder
        # forward pass above is the expensive part -- no second pass needed
        # just to also get `memory`).
        if return_memory:
            return (*pool_out, memory)
        return pool_out

    def load_weights(self, filename, map_location=None, strict=True):
        state_dict = torch.load(filename, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)

    def store_weights(self, path, filename):
        from weight_io import atomic_torch_save

        os.makedirs(path, exist_ok=True)
        atomic_torch_save(self.state_dict(), os.path.join(path, filename))
