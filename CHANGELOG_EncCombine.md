# Changelog: models/Encoders/EncCombine.py

Part of the investigation into `text_to_text.py`'s persistent "identical
output regardless of input" collapse (`unique_pred_fraction` pinned at
0.003-0.005 across every real run, surviving three rounds of downstream
loss-term fixes). A multi-stage diagnostic (`diagnose_state_collapse.py`)
traced the collapse to this file specifically: the frozen text encoder's
own output was healthy (genuine cross-input diversity), but everything
downstream of it inside `EncodingCombination` was destroying that
diversity almost immediately.

## Round 1 — `LinearAttentionBlock.residual_scale`

**Evidence**: a layer-by-layer trace through `HybridLinearEncoder`'s 10
layers showed the within/across-source cosine-similarity gap collapsing
from a healthy 0.18 to ~0.003 after just the *first* `LinearAttentionBlock`
layer's application — on a real, 18-epoch-trained checkpoint
(`t2t_stage1_combine_best_epoch_0018...pt`), not a fresh init.

**Hypothesis**: `LinearAttentionBlock`'s closed-form attention
(`O = Q_phi . (K_phi^T @ V) / Z`) pools every position's K/V into one
shared per-head statistic before any query sees it, and ELU+1's
positive-only feature map gives it far less dynamic range than softmax
attention to produce a sharply input-selective combination. Even after
`out_norm`'s existing fix (bounding this block's output *magnitude*,
added for a prior "exploding norm" incident), nothing stopped the block's
output *direction* from converging toward something near input-independent
and overwriting the residual stream.

**Change**: added `self.residual_scale = nn.Parameter(torch.zeros(1))`
(ReZero, Bachlechner et al. 2020), multiplying the block's gated output
before it returns. Starts at exactly 0, so the block contributes nothing
to the residual at init — 100% of the frozen encoder's signal passes
through untouched — and training can only open it as far as it earns via
gradient signal.

**Verified**: locally, with torch directly — `residual_scale=0` gives
exact-zero block output (`torch.allclose` true, max abs diff 0.0);
gradient flows through the parameter.

## Round 2 — `HybridLinearEncoderLayer.ffn_residual_scale` (first pass: linear-attention layers only)

**Evidence**: after Round 1 landed, a re-run of the layer-by-layer trace
showed layer 1's output *still* collapsing (gap 0.179 -> 0.0115) even
though its attention sublayer was now provably zero. The only other
sublayer in that layer is its FFN (`x = x + self.ffn(self.norm2(x))`),
applied unconditionally regardless of attention type — never touched by
Round 1's fix.

**Change**: added `self.ffn_residual_scale = nn.Parameter(torch.zeros(1))`
to `HybridLinearEncoderLayer`, scoped to `use_linear_attn=True` layers
only at this point, gating the FFN sublayer's contribution the same way.

**Verified**: locally — a linear-attention layer with both gates at 0 is
now an *exact* identity function (`max abs diff: 0.0` across the whole
layer, not just the attention branch).

## Round 3 — `attn_residual_scale` + generalizing `ffn_residual_scale` to full-attention layers

**Evidence**: with all 8 linear-attention layers now provably identity at
init, a third trace showed layers 01-04 held flat at the healthy 0.1792
gap exactly as expected — then **layer 05 (the first full-attention
anchor, plain `nn.MultiheadAttention`) collapsed the gap to -0.0047 on an
input that was still healthy going in.** This falsified the original
"linear attention's closed-form math specifically" hypothesis: a
completely standard, well-tested attention mechanism collapsed just as
badly once actually isolated. The real pattern: *any* freshly-initialized
sublayer in this stack, trained under the original collapse dynamics,
ends up here.

**Change**:
- Added `self.attn_residual_scale = nn.Parameter(torch.zeros(1))` to
  `HybridLinearEncoderLayer`, used only for the `use_linear_attn=False`
  branch (gating `nn.MultiheadAttention`'s output externally, since it's a
  built-in module with no internal parameter to add one to).
- Generalized `ffn_residual_scale` from "linear-attention layers only" to
  *every* layer (both types) — no per-sublayer attribution was possible
  from the layer-level trace, so no basis to exempt the full-attention
  layers' own FFN.

**Verified**: locally — a full-attention layer is now also an exact
identity function at init; the *entire* 10-layer `HybridLinearEncoder`
stack is exact identity (max abs diff 0.0 across the whole stack, not
just per-layer).

**Confirmed on the real checkpoint**: a subsequent real-GPU trace
(`diagnose_state_collapse_slurm-4794970.out`) showed all 10 layers
(`00_input` through `10_FULL`) bit-for-bit identical (0.3432/0.1642/0.1790)
— the encoder fully cleared, for the first time in this investigation.

## Round 4 — `AdaptiveAttentionPooling.ffn_residual_scale`

**Evidence**: with the encoder confirmed clean, the same trace gave the
first-ever honest read on `AdaptiveAttentionPooling`: gap dropped from
0.1790 to 0.0516 — real degradation, but partial, not total collapse
(0.78-0.84, nowhere near the ~1.0 seen everywhere else pre-fix).

**Important subtlety — NOT the same treatment everywhere**: this module
has two residual connections (`queries + attn_out` then
`queries + self.ffn(queries)`), but only the FFN one got a gate.
`queries` going into the *first* connection is `self.query_tokens` — a
fixed, learned parameter, identical for every input by construction, with
zero per-input information. Scaling `attn_out` (the only input-dependent
term there) toward zero would suppress the one thing making this pooling
step input-dependent at all, making collapse *worse*. By the time the FFN
runs, `queries` already carries genuine cross-attended per-input content
from that first step, so the usual "preserve what's there, let the FFN
earn its contribution" reasoning applies cleanly there, and only there.

**Change**: added `self.ffn_residual_scale = nn.Parameter(torch.zeros(1))`
to `AdaptiveAttentionPooling`, applied only to
`queries + self.ffn_residual_scale * self.ffn(queries)`. Cross-attention
left untouched.

**Verified**: locally — cross-attention confirmed still live (cosine
similarity 0.076 between two different random inputs, nowhere near
collapsed); `ffn_residual_scale` confirmed 0 at init with gradient flow.

## Round 5 — `EncodingCombination.forward(..., return_memory=False)`

**Why**: a real harness run (`slurm-4800583`, see
`CHANGELOG_text_to_text.md`) showed the collapse repeatedly relapsing well
after all of the above landed, with the new gate values themselves
staying small the whole time — ruling out "the gates reopened" as the
mechanism. The real story: the underlying (still-trainable) layer weights
behind a gate can drift toward a collapse-inducing direction even while
the gate itself stays small, and nothing in the training loop directly
regularized `combine.encoder`'s own output — every existing anti-collapse
term acts several layers downstream of it. Fixing that needed a way to
get at this encoder's raw pre-pooling output from `text_to_text.py`
without a second, redundant forward pass through the whole 10-layer stack.

**Change**: added `return_memory=False` kwarg to
`EncodingCombination.forward`. When `True`, appends the encoder's own
`memory` tensor (pre-pooling) to whatever tuple `self.pool(...)` already
returns (3-tuple -> 4-tuple, or 4-tuple -> 5-tuple with
`return_gate_stats=True` too) — no second encoder pass needed, since the
encoder forward above is the expensive part.

**Verified**: locally, all three flag combinations (`return_memory` alone,
`return_gate_stats` alone, both together) — tuple lengths and `memory`
shape (`[B, N, D]`, matches `combined_seq`) all correct; confirmed no
existing call site (`text_to_text.py`, `models/OverallModel.py`) passes
`return_memory=True`, so this is a fully backward-compatible addition.

## Status as of this writing

All five rounds verified locally (no GPU needed for pure forward-pass
shape/identity checks) and the first four rounds additionally confirmed on
the real production checkpoint via `diagnose_state_collapse.py`. Round 5
is wired into `text_to_text.py`'s `encoder_koleo_loss` term and confirmed
working on a real run (`slurm-4829152`): `encoder_koleo` genuinely
settled into a healthy range for a long stretch after an elevated start,
though with occasional relapse — see `CHANGELOG_text_to_text.md` for the
full result. That same run showed `combine.pool`'s output (`combined`,
the first tuple element `EncodingCombination.forward` already returns
regardless of `return_memory`) still collapsing even while the encoder
stayed healthy — `text_to_text.py`'s `pool_koleo_loss` now also consumes
this value directly from `encode_to_state`'s existing
`return_encoder_memory=True` path, no further change needed in this file
for that. Not yet verified on a real training run.
