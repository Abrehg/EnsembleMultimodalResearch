import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def createReasoningExpert(d_model=512, num_steps=4):
    return ReasoningExpert(d_model=d_model, num_steps=num_steps)

class ReasoningStep(nn.Module):
    """
    A single reasoning step: self-attention over the state followed by
    a cross-attention readback from a scratchpad, then an FFN.

    The scratchpad is a small set of learnable tokens that act as
    persistent working memory across steps — they accumulate intermediate
    conclusions that the state tokens can read from at each step.
    """

    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048,
                 dropout=0.1, num_scratchpad=8):
        super().__init__()

        # Self-attention over state
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention: state reads from scratchpad
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Scratchpad update: scratchpad reads from state
        self.scratch_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.scratch_norm = nn.LayerNorm(d_model)

        # FFN for state
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, state, scratchpad):
        """
        Args:
            state      : [B, K, D]
            scratchpad : [B, P, D]

        Returns:
            state      : [B, K, D]  — refined
            scratchpad : [B, P, D]  — updated with new conclusions
        """
        # 1. State self-attention (reason over current representation)
        attn_out, _ = self.self_attn(state, state, state)
        state = self.norm1(state + attn_out)

        # 2. State reads from scratchpad (recall prior conclusions)
        cross_out, _ = self.cross_attn(
            query=state, key=scratchpad, value=scratchpad
        )
        state = self.norm2(state + cross_out)

        # 3. Scratchpad reads from state (write new conclusions)
        scratch_out, _ = self.scratch_attn(
            query=scratchpad, key=state, value=state
        )
        scratchpad = self.scratch_norm(scratchpad + scratch_out)

        # 4. FFN on state
        state = self.norm3(state + self.ffn(state))

        return state, scratchpad


class ReasoningExpert(nn.Module):
    """
    Train-of-thought reasoning expert for the routing loop.

    Runs multiple iterative reasoning steps over the latent state,
    each step refining the representation through self-attention and
    a shared scratchpad that accumulates intermediate conclusions.

    The scratchpad acts as a chain-of-thought buffer — instead of
    generating explicit text tokens for reasoning (which would require
    decoding and re-encoding), the scratchpad maintains a continuous
    representation of the reasoning chain that persists across steps.

    At the end, the scratchpad's conclusions are folded back into the
    state via a final cross-attention so all reasoning is captured in
    the output state for downstream experts.

    Compatible with ExpertRegistry:
        registry.add_expert("reasoning", createReasoningExpert())

    Forward signature matches non-terminal expert convention:
        output, state = expert(state)

    Args:
        d_model          : latent dimension (must match state width)
        nhead            : attention heads per layer
        num_steps        : number of iterative reasoning passes
        dim_feedforward  : FFN hidden dimension
        dropout          : dropout rate
        num_scratchpad   : number of scratchpad tokens (reasoning buffer)
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_steps: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_scratchpad: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_steps = num_steps

        self.scratchpad_init = nn.Parameter(
            torch.randn(1, num_scratchpad, d_model) * 0.02
        )

        self.steps = nn.ModuleList([
            ReasoningStep(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_scratchpad=num_scratchpad,
            )
            for _ in range(num_steps)
        ])

        self.final_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, state):
        """
        Args:
            state : [B, K, d_model]  — current latent working memory

        Returns:
            output : [B, K, d_model] — same as state (for artifact logging)
            state  : [B, K, d_model] — refined state after reasoning
        """
        B = state.size(0)

        # Initialise scratchpad for this batch
        scratchpad = self.scratchpad_init.expand(B, -1, -1)     # [B, P, D]

        # Iterative reasoning steps
        for step in self.steps:
            state, scratchpad = step(state, scratchpad)

        # Fold final scratchpad conclusions into state
        cross_out, _ = self.final_cross_attn(
            query=state, key=scratchpad, value=scratchpad
        )
        state = self.final_norm(state + cross_out)

        return None, state

    def store_weights(self, path, filename="reasoning_expert"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))
        print(f"Saved ReasoningExpert to {path}/{filename}")

    def load_weights(self, filepath):
        state_dict = torch.load(filepath, map_location="cpu")
        self.load_state_dict(state_dict)
        print(f"Loaded ReasoningExpert from {filepath}")