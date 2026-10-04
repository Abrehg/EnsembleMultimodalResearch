# Changelog: models/Logic.py

Part of the same investigation as `CHANGELOG_EncCombine.md` (see that file
for the full context on why this collapse investigation started and how
it's been diagnosed). This file holds `RoutingLogic`, the last stage of
`text_to_text.py`'s `encode_to_state` pipeline before the decoder sees
`state`.

## Round 1 — `RoutingLogic.state_rezero`

**Evidence**: once `EncCombine.py`'s `HybridLinearEncoder` was confirmed
fully inert (exact identity function at init, see
`CHANGELOG_EncCombine.md` Round 3) and `AdaptiveAttentionPooling`'s own
partial degradation was isolated (Round 4), a `diagnose_state_collapse.py`
trace gave the first-ever honest read on this module:
`state_encoder`'s own output collapsed the within/across-source gap from
a still-meaningful 0.0516 (post-pooling) down to 0.0003 — by far the
largest single remaining drop anywhere in the pipeline.

**Why this wasn't surprising in hindsight**: `state_encoder` is
`nn.TransformerEncoder(nn.TransformerEncoderLayer(...), num_layers=2)` —
the exact same kind of standard pre-LN, self-attention + FFN,
residual-connection transformer stack that `EncCombine.py`'s own
full-attention anchor layers turned out to collapse under (see
`CHANGELOG_EncCombine.md` Round 3), despite looking architecturally
unremarkable. Trained under the same original collapse dynamics, there
was no particular reason to expect this one to have escaped it.

**Change**: added
`self.state_rezero = nn.Parameter(torch.zeros(1))` and changed
`forward`'s `state = self.state_encoder(state)` to
`state = state + self.state_rezero * (self.state_encoder(state) - state)`.
`nn.TransformerEncoder` is a built-in, opaque multi-layer module with no
per-sublayer hook the way `LinearAttentionBlock` has its own internal
gate, so this wraps its *entire* contribution as one unit (gates the whole
2-layer stack's net effect relative to its own input) rather than gating
each of its 2 internal layers separately.

**Verified**: locally — `state` confirmed exactly unchanged at init
(`torch.allclose` true, max abs diff 0.0); gradient flows through
`state_rezero`.

**Confirmed on the real checkpoint**: the next real-GPU trace
(`diagnose_state_collapse_slurm-4795025.out`) showed stage 4
(`router_state`) bit-for-bit identical to stage 3 (`pooled_combined`) —
`0.2892/0.2391/0.0501` both — confirming `state_encoder` now contributes
nothing extra, and that whatever gap survives pooling now survives all
the way to `state` unchanged.

## A checkpoint-safety note (not a change to this file's logic)

Adding `state_rezero` is a new parameter, which breaks `strict=True`
loading of the real `t2t_stage1_router_*.pt` checkpoints (18-31 real
epochs of training). This is handled on the `text_to_text.py` side (see
`CHANGELOG_text_to_text.md`'s `_is_schema_mismatch`/
`_load_weights_tolerant` entry) — `RoutingLogic.load_weights`'s existing
`strict` parameter is what that fallback mechanism uses; nothing in this
file itself needed to change for that to work. Confirmed on a real run
(`slurm-4795025`): the schema-mismatch retry fired correctly for the
router checkpoint specifically (`"Missing key(s) in state_dict:
state_rezero"`), fell back to `strict=False`, and did not delete the file.

## Status as of this writing

Verified locally and confirmed on the real production checkpoint via
`diagnose_state_collapse.py`. Not yet verified on a real training run
(whether `state_rezero` stays well-behaved as it's trained, rather than
reopening into a bad regime the way `EncCombine.py`'s gates did) — that's
part of what `text_to_text.py`'s new `gate_scale_summary` diagnostic
(`CHANGELOG_text_to_text.md`) is tracking going forward.
