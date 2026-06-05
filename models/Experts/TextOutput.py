import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from typing import Optional
from Encoders.TextEncoder import createTokenizer

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3


def createTextOutputExpert(max_seq_len=10000, vocab_size=32000, d_model=512, max_recursive_steps=6):
    return TextOutputExpert(
        vocab_size=vocab_size,
        d_model=d_model,
        dim_feedforward=d_model * 4,
        max_seq_len=max_seq_len,
        max_recursive_steps=max_recursive_steps,
    )


class ACTModule(nn.Module):
    """
    Adaptive Computational Time halting module (Graves 2016).

    Predicts per-batch halting probability from the mean of the current hidden
    state. Accumulation logic is handled by the calling loop. The accumulated
    ponder cost (total steps taken across all token positions) is the loss-based
    confidence signal used by RL methods to incentivize concise computation.
    """
    def __init__(self, d_model: int, epsilon: float = 0.01):
        super().__init__()
        self.halt_proj = nn.Linear(d_model, 1)
        self.epsilon = epsilon  # halt when accumulated >= 1 - epsilon

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, S, D] → halt_prob: [B]"""
        return torch.sigmoid(self.halt_proj(h.mean(dim=1))).squeeze(-1)


class BlockAttentionResidual(nn.Module):
    """
    Block Attention Residuals (inspired by Attention Residuals, arXiv:2603.15031).

    Transfers information across recursive iterations by cross-attending the
    current hidden state to the stack of residuals from prior steps. Departure
    from the paper: uses input-dependent query vectors

        q_k = W_q · mean(h_k)

    so each iteration's query reflects the current hidden state rather than a
    fixed positional query, giving novel signal at every step.
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
        kv = torch.cat(residual_stack, dim=1)               # [B, K·S, D]
        q = self.query_proj(h.mean(dim=1, keepdim=True))     # [B, 1, D]
        bar_out, _ = self.cross_attn(q, kv, kv)             # [B, 1, D]
        return self.norm(h + bar_out)                        # [B, S, D]


class MemoryLookback(nn.Module):
    """
    Cross-attention over 1-3 recent memory vectors for past-state context.

    Allows the expert to retrieve state context from earlier processing steps
    without re-encoding, complementing the current latent state (immediate
    context) and the BAR residuals (inter-iteration context).
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, memory_vectors: torch.Tensor) -> torch.Tensor:
        """
        h              : [B, S, D]
        memory_vectors : [B, M, D]  M ≤ 3
        """
        mem_out, _ = self.cross_attn(h, memory_vectors, memory_vectors)
        return self.norm(h + mem_out)


class TextOutputExpert(nn.Module):
    """
    Autoregressive text generation expert.

    Architecture: Universal Transformer-style recursive decoder — a single
    TransformerDecoderLayer is reused across up to max_recursive_steps steps per
    token position. Each recursive step is enhanced with:
        - Block Attention Residuals (BAR): inter-iteration information flow
        - Memory lookback: cross-attention over 1-3 recent memory vectors
        - ACT halting: per-token adaptive step count

    Three context sources per recursive step:
        latent state matrix  →  decoder cross-attention (immediate state context)
        memory_vectors       →  MemoryLookback          (past state context)
        residual_stack       →  BAR                     (inter-call context)

    Confidence signals are stored as attributes after each forward() call so
    training code can apply RL ponder penalties without changing the return type:
        self.last_ponder_cost          [B]  total recursive steps (RL signal)
        self.last_geometric_confidence [B]  cosine distance of state change
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        max_recursive_steps: int = 6,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.max_recursive_steps = max_recursive_steps

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=PAD_ID,
        )

        # Single decoder layer — reused at every recursive step (Universal Transformer)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.act             = ACTModule(d_model)
        self.bar             = BlockAttentionResidual(d_model, nhead, dropout)
        self.memory_lookback = MemoryLookback(d_model, nhead, dropout)

        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight  # weight tying

        self.state_update_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.state_update_norm = nn.LayerNorm(d_model)

        self.tokenizer = createTokenizer()

        self.last_ponder_cost: Optional[torch.Tensor] = None
        self.last_geometric_confidence: Optional[torch.Tensor] = None

    def _sincos_pos_embed(self, seq_len: int, device) -> torch.Tensor:
        pos   = torch.arange(seq_len, dtype=torch.float32, device=device)
        omega = torch.exp(
            -torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
            * (math.log(10000.0) / self.d_model)
        )
        out = pos[:, None] * omega[None, :]
        emb = torch.stack([torch.sin(out), torch.cos(out)], dim=2).flatten(1)
        return emb.unsqueeze(0)

    @staticmethod
    def _causal_mask(seq_len: int, device) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

    def _recursive_decode(
        self,
        tgt: torch.Tensor,
        state: torch.Tensor,
        memory_vectors: Optional[torch.Tensor],
        tgt_mask: torch.Tensor,
    ):
        """
        ACT-controlled recursive decoder iterations for one token step.

        Returns:
            weighted_h  : [B, T, D]  ACT-weighted sum of hidden states
            ponder_cost : [B]        number of recursive steps taken per item
        """
        B = tgt.size(0)
        device = tgt.device

        h = tgt
        residual_stack = []

        accumulated  = torch.zeros(B, device=device)
        remainder    = torch.ones(B, device=device)
        weighted_h   = torch.zeros_like(h)
        halted       = torch.zeros(B, dtype=torch.bool, device=device)
        ponder_cost  = torch.zeros(B, device=device)

        for step in range(self.max_recursive_steps):
            h_prev = h

            # Immediate state context: causal self-attn + cross-attn over latent state
            h = self.decoder_layer(tgt=h, memory=state, tgt_mask=tgt_mask)

            # Past state context: cross-attend to recent memory vectors
            if memory_vectors is not None:
                h = self.memory_lookback(h, memory_vectors)

            # Inter-iteration context: BAR cross-attention over prior residuals
            residual = h - h_prev
            h = self.bar(h, residual_stack)
            residual_stack.append(residual)

            # ACT halting probability for non-halted batch items
            halt_prob = self.act(h)                           # [B]
            active    = (~halted).float()
            new_accum = accumulated + halt_prob * active
            is_last   = (step == self.max_recursive_steps - 1)

            halts_here = (new_accum >= 1.0 - self.act.epsilon) & ~halted

            # Halting or forced-last items get the remaining probability mass;
            # still-running items accumulate their halt_prob weight
            weight = torch.where(
                halts_here | (is_last & ~halted),
                remainder,
                halt_prob * active,
            )

            weighted_h   = weighted_h + weight.view(B, 1, 1) * h
            ponder_cost  = ponder_cost + active
            accumulated  = torch.where(halts_here, torch.ones_like(accumulated), new_accum)
            remainder    = 1.0 - accumulated
            halted       = halted | halts_here

            if halted.all():
                break

        return weighted_h, ponder_cost

    def forward(
        self,
        state: torch.Tensor,
        memory_vectors: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
    ):
        """
        Args:
            state          : [B, K, D]  latent state matrix from router
            memory_vectors : [B, M, D]  1–3 recent memory entries (optional)
            max_length     : token generation budget (defaults to max_seq_len)
            temperature    : sampling temperature
            top_k          : top-k filtering; 0 disables

        Returns:
            text  : list[str]  decoded output sequences
            state : [B, K, D]  updated latent state
        """
        if max_length is None:
            max_length = self.max_seq_len

        B = state.size(0)
        device = state.device
        state_init = state.detach().clone()

        generated         = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        finished          = torch.zeros(B, dtype=torch.bool, device=device)
        total_ponder_cost = torch.zeros(B, device=device)
        last_decoded      = state  # fallback for zero-length generation

        for _ in range(max_length):
            T        = generated.size(1)
            tgt      = self.token_embedding(generated) * math.sqrt(self.d_model)
            tgt      = tgt + self._sincos_pos_embed(T, device)
            tgt_mask = self._causal_mask(T, device)

            weighted_h, ponder_cost = self._recursive_decode(
                tgt, state, memory_vectors, tgt_mask
            )
            last_decoded = weighted_h

            # Don't accumulate ponder cost for sequences that have already finished
            total_ponder_cost = total_ponder_cost + ponder_cost * (~finished).float()

            logits = self.output_proj(weighted_h[:, -1, :])
            logits = logits / max(temperature, 1e-8)

            if top_k > 0:
                topk_vals, _ = logits.topk(top_k, dim=-1)
                logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = next_token.masked_fill(finished.unsqueeze(-1), PAD_ID)
            generated  = torch.cat([generated, next_token], dim=1)

            finished = finished | (next_token.squeeze(-1) == EOS_ID)
            if finished.all():
                break

        # Update latent state via cross-attention with the final decoder output
        update_out, _ = self.state_update_attn(
            query=state, key=last_decoded, value=last_decoded
        )
        state = self.state_update_norm(state + update_out)

        # Geometric confidence: cosine distance measures how much the state changed.
        # Larger distance → the expert did more meaningful work on the state.
        geometric_conf = 1.0 - F.cosine_similarity(
            state_init.mean(dim=1), state.mean(dim=1), dim=-1
        )  # [B]

        self.last_ponder_cost          = total_ponder_cost.detach()
        self.last_geometric_confidence = geometric_conf.detach()

        text = self.tokenizer.detokenize(generated, strip_special=False)
        return text, state

    def store_weights(self, path, filename="text_output_expert"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))

    def load_weights(self, filepath):
        state_dict = torch.load(filepath, map_location="cpu")
        self.load_state_dict(state_dict)
