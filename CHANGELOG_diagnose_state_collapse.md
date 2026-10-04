# Changelog: diagnose_state_collapse.py

New file — a read-only diagnostic script (no training, no backward pass,
no checkpoints written) that traces where cross-input diversity is lost
across `encode_to_state`'s pipeline. Built specifically because Phase 0's
diagnostics (see `CHANGELOG_text_to_text.md`) could only answer
"upstream-or-downstream," not which of the three separately-initialized
upstream stages (`combine.encoder`, `combine.pool`, `router`) was actually
responsible. Deliberately warm-starts from the real, most-trained
`t2t_stage1_combine`/`router` checkpoints (`checkpoint_prefix=
"t2t_stage1"`) rather than a fresh init — the question this script answers
is "where does collapse happen in the actual production run," not in an
early-step harness restart.

## Round 1 — initial 4-stage trace

Traces `encode_to_state`'s own call sequence exactly, keeping every
intermediate tensor instead of only the final `state`:

1. `model.encode(text, "text")` — frozen TextEncoder's own output
2. `model.combine.encoder(...)` — `HybridLinearEncoder`, pre-pooling
3. `model.combine.pool(...)` — `AdaptiveAttentionPooling`
4. `model.router(...)` — `RoutingLogic`'s `state_encoder`

Reuses `text_to_text.py`'s `_pairwise_state_similarity` at each stage
rather than a second hand-copied version. Sample size and seed
configurable via `DIAG_SAMPLE_SIZE`/`DIAG_SEED` env vars (default 60 rows,
seed 0).

**Result** (`slurm-4789538`, before any architecture fix): stage 1
healthy (0.1801 gap); stages 2-4 all collapsed (~1.0/~1.0) — the first
direct confirmation that `combine.encoder` itself is where the collapse
originates, not just "somewhere downstream of the frozen encoder."

## Round 2 — layer-by-layer trace inside `HybridLinearEncoder`

Added `trace_encoder_layers`, walking `model.combine.encoder.layers` one
at a time (tagging each by kind, `linear` vs. `FULL`) instead of only
measuring before/after the whole 10-layer stack — needed once Round 1
localized the problem to this module but not to a specific layer within
it.

**Result** (`slurm-4789547`, before any architecture fix): the gap
collapsed almost entirely at layer 1 alone (0.1792 -> 0.0033), with layers
2-10 only inheriting an already-collapsed state — directly motivating the
`LinearAttentionBlock.residual_scale` fix (`CHANGELOG_EncCombine.md`
Round 1).

## Operational fix — moved from CPU to GPU

The job was originally submitted as a plain CPU job (`diagnose_state_
collapse.sh`, no `--gres`) on the assumption that a forward-pass-only
workload (no generation, no backward pass) would be cheap regardless of
device. It wasn't: `slurm-4789535` didn't clear 20 of 60 rows within a
30-minute CPU budget (`d_model=1024` plus `HybridLinearEncoder`'s two
full-attention layers, quadratic in sequence length, over some genuinely
long `logi_glue`/`deepmind_math` reasoning-chain inputs, without GPU
parallelism). Switched to `--gres=gpu:1`, matching what
`collapse_fix_harness.py` already establishes runs fast there even with a
full forward+backward+generation pass per step — this script only ever
does a bare forward pass, a small fraction of that cost. Confirmed fast
on GPU afterward (`slurm-4789538` onward all completed well within
budget).

## Status as of this writing

Both traces (4-stage and layer-by-layer) were re-run after every
subsequent architecture-fix round in `CHANGELOG_EncCombine.md`/
`CHANGELOG_Logic.md` and are what confirmed each one worked (identity
function at init, confirmed on the real checkpoint, not just in a local
unit test). Last real run (`slurm-4795025`) confirmed the full pipeline —
encoder, pooling, router — holding `state`'s gap flat at ~0.05 from
`pooled_combined` through `router_state`, with only pooling's own
cross-attention step (deliberately left ungated) doing any real
transformation. No further changes needed to this script right now; it
remains available to re-verify the architecture any time it's suspected
of regressing.
