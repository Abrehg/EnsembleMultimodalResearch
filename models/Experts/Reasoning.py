import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def createReasoningExpert(num_steps=8, d_model=512):
    return ReasoningExpert(
        d_model=d_model,
        dim_feedforward=d_model * 4,
        max_steps=num_steps,
    )


# ── Shared expert primitives (BAR / ACT / MemoryLookback) ────────────────────

class ACTModule(nn.Module):
    """
    Adaptive Computational Time halting (Graves 2016).
    Predicts per-batch halting probability from mean(h). Accumulation is
    handled by the calling loop. Total ponder cost is the RL loss signal.
    """
    def __init__(self, d_model: int, epsilon: float = 0.01):
        super().__init__()
        self.halt_proj = nn.Linear(d_model, 1)
        self.epsilon = epsilon

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, S, D] → halt_prob: [B]"""
        return torch.sigmoid(self.halt_proj(h.mean(dim=1))).squeeze(-1)


class BlockAttentionResidual(nn.Module):
    """
    Block Attention Residuals (inspired by Attention Residuals, arXiv:2603.15031).
    Cross-attends current hidden state to the stack of residuals from prior
    recursive steps. Query: q_k = W_q · mean(h_k), giving hidden-state-dependent
    novel signal at each iteration rather than a fixed positional query.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, residual_stack: list) -> torch.Tensor:
        """
        h             : [B, S, D]
        residual_stack: list of [B, S, D] residuals from steps 0..k-1
        """
        if not residual_stack:
            return h
        kv = torch.cat(residual_stack, dim=1)
        q  = self.query_proj(h.mean(dim=1, keepdim=True))   # [B, 1, D]
        bar_out, _ = self.cross_attn(q, kv, kv)
        return self.norm(h + bar_out)


class MemoryLookback(nn.Module):
    """Cross-attention over 1–3 recent external memory vectors for past-state context."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, memory_vectors: torch.Tensor) -> torch.Tensor:
        """h: [B, S, D],  memory_vectors: [B, M, D]"""
        mem_out, _ = self.cross_attn(h, memory_vectors, memory_vectors)
        return self.norm(h + mem_out)


# ── VAE reasoning block (post-training attachment) ───────────────────────────

class ReasoningVAE(nn.Module):
    """
    Variational reasoning encoder/decoder, inspired by ReGuLaR
    (arXiv:2601.23184, Latent Reasoning Block concept).

    Encodes the mean-pooled latent state into a low-dimensional continuous
    space z, then decodes to a feature vector that serves as a "visual guide"
    in the TRM cross-attention buffer — a continuous analogue of rendered CoT.

    Designed for post-training attachment via ReasoningExpert.set_vae(). During
    initial training the TRMBlock runs without it. Full deployment would replace
    the decoder with an image renderer (→ [B, C, H, W]) and route through the
    shared VisionEncoder for actual rendered chain-of-thought images.
    """
    def __init__(self, d_model: int, vae_dim: int = 256):
        super().__init__()
        self.encoder     = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU())
        self.mu_proj     = nn.Linear(d_model // 2, vae_dim)
        self.logvar_proj = nn.Linear(d_model // 2, vae_dim)
        self.decoder     = nn.Sequential(
            nn.Linear(vae_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, h: torch.Tensor):
        """
        h       : [B, K, D]
        Returns : feature [B, 1, D], kl scalar
        """
        h_pool  = h.mean(dim=1)
        enc     = self.encoder(h_pool)
        mu      = self.mu_proj(enc)
        logvar  = self.logvar_proj(enc).clamp(-10, 10)

        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu

        feature = self.decoder(z).unsqueeze(1)                            # [B, 1, D]
        kl      = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        return feature, kl


# ── TRM core block ────────────────────────────────────────────────────────────

class TRMBlock(nn.Module):
    """
    Two-layer Transformer decoder — the reusable reasoning core (TRM).

    Both layers cross-attend to the same memory tensor at each recursive step.
    Memory grows across steps as the buffer accumulates, mirroring the
    Latent Reasoning Block from ReGuLaR where prior visual guides inform later steps.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()

        def _layer():
            return nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )

        self.layer1 = _layer()
        self.layer2 = _layer()

    def forward(self, h: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        h      : [B, K, D]  hidden state (queries)
        memory : [B, M, D]  cross-attn target (initial state + buffer + visual guide)
        """
        h = self.layer1(tgt=h, memory=memory)
        h = self.layer2(tgt=h, memory=memory)
        return h


# ── Main expert ───────────────────────────────────────────────────────────────

class ReasoningExpert(nn.Module):
    """
    Compact recursive reasoning expert.

    Core architecture: TRMBlock (2-layer decoder) reused for up to max_steps
    recursive passes with per-step ACT halting. Each step:

        memory = [initial_state | accumulated_buffer | VAE_guide (if set)]
        h      = TRM(h, memory)
        h      = MemoryLookback(h, external_memory_vectors)   # past-state context
        h      = BAR(h, residual_stack)                       # inter-iteration context
        buffer.append(VAE_guide or h_summary)
        ACT → halt or continue

    Three context sources per recursive step:
        initial_state + buffer  →  TRM cross-attention  (immediate + accumulated)
        memory_vectors          →  MemoryLookback        (past-state context)
        residual_stack          →  BAR                   (inter-call context)

    Reasoning trace is carried in the updated state (no text artifact).
    Downstream TextOutputExpert decodes the answer from the refined state.

    VAE attachment (post-training):
        expert.set_vae(ReasoningVAE(d_model))
        # VAE renders each step's latent → visual guide; buffer becomes a
        # sequence of compressed reasoning images rather than plain summaries.
        # Full deployment: replace ReasoningVAE.decoder with an image renderer
        # and route through the shared VisionEncoder.

    Confidence signals stored after each call:
        self.last_ponder_cost          [B]  total recursive steps (RL ponder penalty)
        self.last_geometric_confidence [B]  cosine distance of state change
        self.last_kl_loss              scalar  VAE KL divergence (0 if no VAE)
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_steps: int = 8,
    ):
        super().__init__()
        self.d_model   = d_model
        self.max_steps = max_steps

        self.trm             = TRMBlock(d_model, nhead, dim_feedforward, dropout)
        self.act             = ACTModule(d_model)
        self.bar             = BlockAttentionResidual(d_model, nhead, dropout)
        self.memory_lookback = MemoryLookback(d_model, nhead, dropout)

        # None until set_vae() is called; registered as submodule on assignment
        self.vae: Optional[ReasoningVAE] = None

        self.last_ponder_cost: Optional[torch.Tensor]          = None
        self.last_geometric_confidence: Optional[torch.Tensor] = None
        self.last_kl_loss: Optional[torch.Tensor]              = None

    def set_vae(self, vae: ReasoningVAE):
        """Attach the VAE reasoning block (post-training). Registered as a submodule."""
        self.vae = vae

    def forward(
        self,
        state: torch.Tensor,
        memory_vectors: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            state          : [B, K, D]  latent state matrix from router
            memory_vectors : [B, M, D]  1–3 recent memory entries (optional)

        Returns:
            output : None      — no text artifact; reasoning is encoded in the state
            state  : [B, K, D] — updated latent state carrying the reasoning result
        """
        B = state.size(0)
        device = state.device
        state_init = state.detach().clone()

        h = state
        buffer: list         = []
        residual_stack: list = []

        accumulated = torch.zeros(B, device=device)
        remainder   = torch.ones(B, device=device)
        weighted_h  = torch.zeros_like(h)
        halted      = torch.zeros(B, dtype=torch.bool, device=device)
        ponder_cost = torch.zeros(B, device=device)
        total_kl    = torch.tensor(0.0, device=device)

        for step in range(self.max_steps):
            h_prev = h

            # Render current hidden state to a visual guide via VAE (if attached)
            if self.vae is not None:
                v_feat, kl = self.vae(h)              # [B, 1, D], scalar
                total_kl = total_kl + kl
            else:
                v_feat = None

            # Build TRM cross-attention memory: initial state + buffer + visual guide
            memory_parts = [state]
            if buffer:
                memory_parts.append(torch.cat(buffer, dim=1))
            if v_feat is not None:
                memory_parts.append(v_feat)
            memory = torch.cat(memory_parts, dim=1)   # [B, state_K + buf + vis, D]

            # TRM: 2-layer decoder refines h using growing memory
            h = self.trm(h, memory)

            # Past-state context: cross-attend to external memory module vectors
            if memory_vectors is not None:
                h = self.memory_lookback(h, memory_vectors)

            # Inter-iteration context: BAR over accumulated prior residuals
            residual = h - h_prev
            h = self.bar(h, residual_stack)
            residual_stack.append(residual)

            # Buffer entry: visual guide (VAE) if available, else mean-pooled summary
            buffer_entry = v_feat if v_feat is not None else h.mean(dim=1, keepdim=True)
            buffer.append(buffer_entry)               # [B, 1, D]

            # ACT halting with Graves weighted-average output
            halt_prob = self.act(h)                   # [B]
            active    = (~halted).float()
            new_accum = accumulated + halt_prob * active
            is_last   = (step == self.max_steps - 1)

            halts_here = (new_accum >= 1.0 - self.act.epsilon) & ~halted
            weight = torch.where(
                halts_here | (is_last & ~halted),
                remainder,
                halt_prob * active,
            )

            weighted_h  = weighted_h + weight.view(B, 1, 1) * h
            ponder_cost = ponder_cost + active
            accumulated = torch.where(halts_here, torch.ones_like(accumulated), new_accum)
            remainder   = 1.0 - accumulated
            halted      = halted | halts_here

            if halted.all():
                break

        # Geometric confidence: how much did the state change?
        geometric_conf = 1.0 - F.cosine_similarity(
            state_init.mean(dim=1), weighted_h.mean(dim=1), dim=-1
        )  # [B]

        self.last_ponder_cost          = ponder_cost.detach()
        self.last_geometric_confidence = geometric_conf.detach()
        self.last_kl_loss              = total_kl.detach()

        return None, weighted_h

    def store_weights(self, path, filename="reasoning_expert"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))

    def load_weights(self, filepath):
        state_dict = torch.load(filepath, map_location="cpu")
        self.load_state_dict(state_dict)
