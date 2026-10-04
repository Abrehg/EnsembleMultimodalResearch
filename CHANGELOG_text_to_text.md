# Changelog: text_to_text.py

The main training script for the collapse investigation (see
`CHANGELOG_EncCombine.md` for the architectural side of this). Two prior
real-scale fixes (P1.1/P1.2: decoupling anti-collapse coefficient ramps
from `total_steps`, raising `scheduled_sampling_prob`; P2.2: a batch-local
supervised-contrastive term on `decoder_diversity_loss`) both failed to
move `unique_pred_fraction` off its 0.003-0.005 floor before this file's
entries below begin. This file is where every diagnostic that eventually
localized the collapse lives, and where the two concrete fixes it led to
(the `EncCombine.py`/`Logic.py` architecture changes, and this file's own
`encoder_koleo_loss`) get exercised during real training.

## Phase 0 — cheap diagnostics (before any architecture change)

Added three things, all aimed at answering "is this an upstream
(encoder/combine) collapse, or a downstream (decoder ignores a healthy
state) one" before guessing at another fix:

1. **Logged `koleo_loss`'s value** for the first time in the
   `global_step % 50` print/wandb block inside `train_stage` (and mirrored
   in `collapse_fix_harness.py`) — it had been `backward()`'d every
   optimizer step but never actually logged, so "is state itself
   collapsing" was previously unanswerable without an ad-hoc probe.
2. **Added a cross-source `state` similarity diagnostic** inside
   `evaluate_rewards`: pools each sampled row's `state`
   (`state.mean(dim=1).squeeze(0)`), groups by source, and reports mean
   within-source vs. across-source cosine similarity. Refactored the
   computation out into a standalone `_pairwise_state_similarity(pooled,
   sources)` helper so `diagnose_state_collapse.py` could reuse the exact
   same logic later instead of a second hand-copied version drifting from
   it. `evaluate_rewards`'s return signature changed from 3 values to 4
   (`rewards, unique_pred_fraction, precision_recall, state_similarity`)
   — updated both call sites in `train_stage` (the periodic eval-step
   block and the end-of-epoch block) and `collapse_fix_harness.py`'s call
   site.
3. **Fixed `gate_health`'s dropout-mode confound**: it used to run
   whatever mode the caller left the model in, which at its real call site
   (right after `evaluate_rewards`, which itself restores `train()` before
   returning) meant gate stats were measured with dropout *on*, while the
   actual collapsed generations this diagnostic exists to explain come
   from `evaluate_rewards`'s own `model.eval()` pass. Now explicitly forces
   `eval()` before reading gate stats and restores the caller's original
   mode afterward.

**Verified**: the new `state_similarity` logic unit-tested in isolation
(collapsed/clustered/single-row/single-source synthetic cases) before
trusting it on a real run.

**Result** (`slurm-4789538`/`slurm-4789549`/`slurm-4789547`, short
diagnostic harness runs): `state_sim_within` ~= `state_sim_across` ~= 1.0
at every checkpoint — conclusive confirmation of upstream state-collapse,
motivating the architecture investigation in `CHANGELOG_EncCombine.md`.

## Checkpoint-safety fix (`_is_schema_mismatch` / `_load_weights_tolerant`)

**Why**: the architecture fixes in `CHANGELOG_EncCombine.md`/
`CHANGELOG_Logic.md` all add new parameters to `combine`/`router`. Before
this fix, `_warm_start` (inside `build_model`) and
`_load_weights_with_fallback` (used by `train_stage`'s mid-run resume
path) both treated *any* `load_state_dict` failure as file corruption and
deleted the offending checkpoint — which would have silently destroyed
the real `t2t_stage1_combine/router_*.pt` checkpoints (18-31 real epochs
of training) the very first time either function tried to warm-start from
them under the new architecture.

**Change**: added `_is_schema_mismatch(e)` — checks whether a
`RuntimeError` is `load_state_dict`'s own "Missing key(s)"/"Unexpected
key(s)" message (an architecture change) rather than genuine corruption —
and `_load_weights_tolerant(module, path, device)`, which tries a strict
load first and, only on a genuine schema mismatch, retries with
`strict=False` (keeping every matching weight, leaving new parameters at
their fresh-init value) instead of deleting the file. Both
`_load_weights_with_fallback` and `_warm_start` now route through this
helper.

**Verified**: locally, by building a real `EncodingCombination`/
`RoutingLogic`, stripping the new keys from a saved state_dict to simulate
an "old" checkpoint, and confirming `strict=True` raises the expected
message, `_is_schema_mismatch` catches it, and `strict=False` succeeds
without raising.

**Confirmed on real runs**: every subsequent `diagnose_state_collapse.sh`
run against the real `t2t_stage1` checkpoints logged the expected
`"...'s schema doesn't match the current architecture (...); retrying
with strict=False..."` message and successfully warm-started — the
checkpoints were never deleted across any of the architecture-change
rounds in `CHANGELOG_EncCombine.md`/`CHANGELOG_Logic.md`.

## Phase 2 diagnostics support — `encode_to_state(..., return_encoder_memory=False)`

Added alongside the existing `return_gate_logits` option: when
`return_encoder_memory=True`, calls `model.combine(encodings,
return_memory=True)` (see `CHANGELOG_EncCombine.md` Round 5) and also
returns `combine.encoder`'s own pre-pooling output. Used by
`diagnose_state_collapse.py`'s multi-stage trace and, later, by this
file's own `encoder_koleo_loss` term (see below). Default `False` on both
flags keeps every existing call site's plain-tensor return untouched.

## `gate_scale_summary(model)` — tracking the ReZero gates during training

**Why**: a real harness run (`slurm-4795036`) with the full architecture
fix in place showed something genuinely new: `state_sim_within/across`
and `unique_pred_fraction` recovered to real, healthy values early in
training (first time ever, in this whole investigation) but then
*repeatedly relapsed* into collapse over the following ~7000 steps,
despite every anti-collapse coefficient fully ramped. Leading hypothesis
going in: the ReZero gates added in `CHANGELOG_EncCombine.md`/
`CHANGELOG_Logic.md` reopened back into the same collapse-inducing
contribution they were built to silence at init.

**Change**: added `gate_scale_summary(model)` — reads the current
absolute value of every ReZero gate (`LinearAttentionBlock.residual_scale`
x8, `HybridLinearEncoderLayer.ffn_residual_scale` x10,
`HybridLinearEncoderLayer.attn_residual_scale` x2,
`AdaptiveAttentionPooling.ffn_residual_scale` x1, `RoutingLogic.
state_rezero` x1), grouped by role (`linear_attn_mean`, `linear_ffn_mean`,
`full_attn_mean`, `full_ffn_mean`, `pool_ffn`, `router_state`) rather than
all 22 individually. Wired into the same `global_step % 50` print/wandb
block as `koleo`, mirrored in `collapse_fix_harness.py`.

**Verified**: locally, against real `EncodingCombination`/`RoutingLogic`
instances — all-zero at init, correctly reflects each group's mean once
gate values are manually set to simulate "opened" gates.

**Result** (`slurm-4800583`, a 3000-step run with denser 150-step eval):
**ruled out the leading hypothesis.** At the first real collapse episode
(step 900), every gate was still under 0.07 — nowhere near "reopened."
Gate values did jump somewhat around the worst episode (step ~2100:
`gate_full_attn` 0.02->0.04, `gate_pool_ffn` 0.01->0.037) but then
flattened and stayed flat for the rest of the run while the collapse
itself persisted at similar severity. The same small gate magnitude
corresponded to healthy state early in training and collapsed state later
— meaning gate *magnitude* doesn't track collapse severity. Real
conclusion: the underlying, still-trainable layer weights behind a small
gate can themselves drift toward a collapse-inducing direction, since
nothing was directly regularizing them — every existing anti-collapse
term acts on `state`, several layers downstream of `combine.encoder`'s
own output. This directly motivated `encoder_koleo_loss` below.

## `encoder_koleo_loss` (via `HYPERPARAMS["reward"]["encoder_koleo_coef"]`)

**Change**: in `train_stage`'s batch-boundary block (and mirrored in
`collapse_fix_harness.py`), each row's `encode_to_state(model, r["input"],
return_encoder_memory=True)` call now also yields `combine.encoder`'s own
pre-pooling output, pooled the same way as `state`
(`.mean(dim=1).squeeze(0)`). Reuses the existing `koleo_loss` function
directly (it's already generic over any batch of embeddings) rather than
writing a new loss — just a new call site, `koleo_loss(torch.stack(
pooled_memories, dim=0))`, with its own coefficient
(`encoder_koleo_coef`, started at `1.0`, matching `koleo_coef`'s own value
since there's no tuning data yet to justify a different weight) added
into `batch_term_loss` alongside the existing `koleo`/`contrastive` terms.
Also added to the periodic print/wandb log.

This is the direct fix for the gate-logging finding above: a diversity
pressure that acts *directly* on `combine.encoder`'s own output, not
several layers downstream of it, so the underlying layer weights have an
undiluted incentive to stay diverse as training proceeds — something
nothing in this file did before.

**Verified**: locally — confirmed `koleo_loss` (reused, not reimplemented)
gives the expected healthy-negative value on genuinely diverse synthetic
"memory" vectors and the expected large-positive value on near-identical
ones, matching the exact mechanism `EncCombine.py`'s `return_memory` was
built to expose.

**Result** (`slurm-4829152`, the first real run with `encoder_koleo_coef`
active): real, measurable success, with a caveat. `encoder_koleo` itself
ran elevated (0.5-1.8, not healthy but not fully collapsed) for the first
~4000 steps, then settled into the healthy range (-0.1 to -0.3) for a long
stretch (~step 4200-7000) — confirming the term does what it was built
for. It still spiked back up twice near the end (step 7100-7200: +1.0/
+1.1; step 8000: +2.1), so not a complete fix. More importantly: **at
step 5100, `encoder_koleo` was healthy (-0.2254) while `state_sim_within/
across` was fully collapsed (0.985/0.980).** With the encoder confirmed
healthy at that point, the remaining collapse had nowhere left to hide
except `combine.pool`/`router` — directly motivating `pool_koleo_loss`
below. Qualitatively, the run also looked very different from every prior
one: the decoder's dominant failure mode shifted from full-length
repeated-token loops (`"management"x32` etc., the norm in every previous
run) to short, fluent, plausible single words (`"Terry"`, `"Peter"`,
`"neutral"`, ...) — including two genuine `reward=1.000` hits, the first
in this whole investigation. But `unique_pred_fraction` itself barely
moved (still 0/10 final checkpoints above the harness's 0.02 pass bar) —
several checkpoints showed the exact same word (often `"True"`) printed
for `logi_glue`/`trivia_qa`/`nq_open` samples simultaneously, i.e. the
same "same output regardless of input" collapse, just wearing a more
fluent default than before. This motivated `most_common_pred_fraction`
below.

## `pool_koleo_loss` (via `HYPERPARAMS["reward"]["pool_koleo_coef"]`)

**Why**: see the `encoder_koleo_loss` result above — once the encoder is
confirmed healthy, state can still fully collapse, meaning `combine.pool`
(`AdaptiveAttentionPooling`) and/or `router` are still only protected
architecturally (the ReZero gates in `CHANGELOG_EncCombine.md`/
`CHANGELOG_Logic.md`), not with any direct training-time diversity
pressure the way the encoder now has.

**Change**: `encode_to_state(..., return_encoder_memory=True)` now returns
a 3-tuple (`state, memory, combined`) instead of 2 — `combined` (combine
.pool's own output) was already computed by every call path regardless of
this flag, so this is free (no extra forward pass). Updated both call
sites (`train_stage`'s batch-boundary block, `collapse_fix_harness.py`'s
mirror) to collect `pooled_pools` alongside `pooled_states`/
`pooled_memories`, and added `pool_koleo = koleo_loss(torch.stack(
pooled_pools, dim=0))` with its own coefficient (`pool_koleo_coef`,
started at `1.0`, same reasoning as `encoder_koleo_coef`) into
`batch_term_loss`. Also added to the periodic print/wandb log.

**Verified**: locally — confirmed `combined` returned under
`return_memory=True` is bit-identical to a plain call's `combined` once
dropout is disabled (`model.eval()`); the two calls differ under
`model.train()` purely because of dropout's own run-to-run randomness,
not because of anything wrong in the new plumbing.

**Result** (`slurm-4829741`): **regressed**, and the regression is now
traced to this term — see "Regression review" below. Coefficient set to
`0.0` (value still computed and logged).

## `most_common_pred_fraction` (in `evaluate_rewards`)

**Why**: see the `encoder_koleo_loss` result above — `unique_pred_fraction`
alone can't distinguish "32 distinct wrong answers out of 200" from "199
copies of one fluent default word plus 1 different one." Both read as the
same low number, but `slurm-4829152`'s sample log showed the second case
specifically (the same word printed across all three sources at a single
checkpoint) — a real, different-in-kind collapse signature this file had
no way to surface on its own before.

**Change**: `evaluate_rewards` now also returns `most_common_pred_fraction`
(fraction of the sampled predictions equal to the single most frequent
one, via `collections.Counter`) — return signature changed from 4 values
to 5 (`rewards, unique_pred_fraction, most_common_pred_fraction,
precision_recall, state_similarity`). Updated all three call sites (both
in `train_stage`, one in `collapse_fix_harness.py`) and added to both the
periodic print lines and the wandb/JSONL logs.

**Verified**: locally, with synthetic prediction lists — 200 fully
distinct answers gives `most_common_pred_fraction=0.005` (correctly low);
199-copies-of-one-answer gives `0.995` (correctly high); a case
constructed to have a similarly-low `unique_pred_fraction` either way (32
evenly-split clusters vs. one dominant cluster) is cleanly distinguished
by `most_common_pred_fraction` (0.031 vs. 0.995) even when
`unique_pred_fraction` alone is less clear-cut.

**Not yet verified on a real training run.**

## Regression review (`slurm-4829741`) — `pool_koleo` disabled, common-mode diagnostic added

**What regressed**: compared with `slurm-4829152` (same code minus
`pool_koleo_loss`/`most_common_pred_fraction`), early state diversity was
worse at the first validation points (step 300 state koleo +0.52 vs
−0.16; state_sim_within 0.75 vs 0.21), and `most_common_pred_fraction`
sat at exactly 1.000 for most of the run (one answer, `"sesesese..."` then
`""` or `"True"`, across all three sources). Validation CE was lower than
before, so this is a diversity regression, not an overfitting one.

**Root cause (evidence, not guess)**: `pool_koleo` was numerically
duplicating `koleo`. In the log, while the router gate is near zero
(steps ≤ 1000), mean |koleo − pool_koleo| = 0.016 (e.g. step 50: 0.3254 vs
0.3252). Because `state = state_in + state_rezero·(T(state_in) − state_in)`
with `state_rezero ≈ 0`, `state ≈ combined`, so the two terms measured the
same geometry. Enabling `pool_koleo_coef=1.0` therefore doubled the
effective koleo pressure on the pooled representation exactly where the
regression appears. The gap only grows (mean 0.12 after step 1000) once the
router gate opens to ~0.05–0.10, at which point the two terms stop being
redundant — but by then the early dynamics had already been pushed.

**Change**: `HYPERPARAMS["reward"]["pool_koleo_coef"]` 1.0 → 0.0 (with a
comment giving the evidence). The term is still computed and logged, so
its value remains a live diagnostic; only its gradient is removed. No
structural code removed. Re-enable only if it is made genuinely independent
of the state path.

**Caveat**: single-seed comparison. Model init is not seeded in the
harness, so the 4829152-vs-4829741 difference could partly be init
variance. The mechanistic duplication (0.016 mean gap while the router
gate is small) does not depend on that and is the basis for the change.
A short A/B (same seed, pool_koleo on vs. off, ~1000 steps) would settle
the init question.

**Open issue, not changed here — `encoder_koleo` stuck in the collapsed
range.** In the same run `encoder_koleo` sat at +2.2 to +2.8 for ~2000
steps (≈ steps 4100–6000) even with `encoder_koleo_coef=1.0` active, while
the final-state koleo was healthy (≈ −0.2). A plausible mechanism:
`koleo_loss` L2-normalizes its input, so its gradient scales as 1/|x|. If
mean-pooled encoder memory carries a large shared offset, the offset
dominates the norm and damps the gradient on the informative residual.
This is a hypothesis, not a finding, so the change is instrumentation only:

**New diagnostic — `common_mode_ratio(pooled)`**: norm of the batch-mean
vector divided by the mean per-row deviation norm. Verified locally: a
diverse batch gives 0.37; an offset-dominated batch gives 106. Logged per
step as `cm_memory` (encoder) and `cm_state` (final state) in both
`train_stage` and `collapse_fix_harness.py`. If `cm_memory` is large in the
steps where `encoder_koleo` is elevated, the next change is to centre the
batch before `koleo_loss` on the encoder path (the same idea as DINO's
centering). That should be applied only after the diagnostic confirms the
mechanism.

**Still open — decoder-side collapse, unrelated to the koleo terms.** Even
at checkpoints where state is source-differentiated (e.g. step 3900:
within 0.80, across 0.27), the decoder emits one constant answer for every
row. Outputs are mostly `""` (EOS at position 0) or `"True"`, and
`blank_w` had already ramped to 1.0 by step 3050 without eliminating the
empty outputs. This needs its own investigation of `compute_task_loss`'s
blank-answer penalty and the RL term; it is not addressed here.

## A/B result: `pool_koleo` on vs. off (`slurm-4829831` / `slurm-4829832`)

Same seed (0), same init, same data order, two arms. `pool_koleo_coef` was
0.0 in one and 1.0 in the other (set via `HARNESS_POOL_KOLEO_COEF`).

| step | arm | state_sim_within / across | koleo | pool_koleo | output |
|---|---|---|---|---|---|
| 100 | off | 0.48 / 0.33 | +0.22 | +0.23 | `importantly fumble Hutchinson...` |
| 100 | on  | 0.41 / 0.19 | +0.18 | +0.18 | `True` |
| 300 | off | **0.11 / 0.00** | −0.21 | −0.21 | `""` |
| 300 | on  | **0.78 / 0.75** | **+1.71** | **+1.70** | `""` |
| 500 | on  | 0.93 / 0.89 | +1.59 | +1.57 | `""` |

**Reading**: turning the pool term on reproduces the earlier regression at
the same step. `slurm-4829741` (pool on) showed 0.75 at step 300, and this
A/B arm shows 0.78. With pool off, state at step 300 is differentiated
(0.11/0.00). That matches the duplication explanation in the regression
review. Because both arms share a seed, the on/off difference is not init
noise for this seed. That is still one seed, so it does not rule out
init variance across seeds. A second seed would be needed for that.

**Second observation in the on arm**: `gate_salience_mean` sits near 1.0
(0.998 at step 300, 0.9975 at step 400, 0.99 at step 600) against 0.63 and
0.87 in the off arm. The salience gate saturates open with the pool term
on. This is the same saturation failure `salience_diversity_loss` was built
to guard against, so it is a second channel through which the duplicate
pressure does damage. Not addressed separately; it disappears with the
term disabled.

**Both arms** end on a constant output (`""`, `"True"`, or a repeated
token) at most validation points, so the decoder collapse is independent of
the pool term.

## Blank-output penalty widened to PAD/BOS/EOS (`compute_task_loss`)

**Why**: in the A/B, state is differentiated at step 300 in the off arm
(0.11/0.00) yet the decoder emits `""` for every row. The state-side
explanation is therefore ruled out for that output. Reading the code
shows the blank penalty was EOS-only. Verified against the real tokenizer
(`models/Encoders/multimodal_bpe.model`): ids 0 (PAD), 2 (BOS), and 3 (EOS)
all decode to `''`, and 1 (UNK) decodes to `' ⁇ '`. Greedy decoding starts
from `generated=[BOS]` and only stops on EOS. So an argmax of PAD or BOS
yields an empty output and keeps generating, and the old penalty never
touched it.

**Change**: `blank_penalty` now sums teacher-forced probability at position
0 over `[PAD_ID, BOS_ID, EOS_ID]`, gated by `gold_not_empty` (gold first
token not in that set). Same coefficient ramp
(`blank_answer_penalty_coef_*`), no new hyperparameter.

**Verified**: through the real `compute_task_loss` with a small real
`TextOutputExpert`: for gold `"Paris"` the penalty difference (coef 1 minus
coef 0) is 0.9951, equal to the summed PAD+BOS+EOS mass at position 0. For
gold `""` it is 0. The old EOS-only version would have returned roughly
P(EOS) ≈ 0 here, so it would not have registered most of this mass.

**Observation (random init, not trained)**: at init the position-0 mass
sits almost entirely on BOS (0.966). BOS is never a training target, so its
logit only moves through softmax normalization. A never-supervised special
id can therefore dominate the first decode step. Whether the same holds in
a trained model is the question the next run's `pos0` diagnostic answers.

## `pos0_diagnostics` — first-step argmax and special-id mass

**What**: `pos0_diagnostics(expert, states)` runs one teacher-forced step
with `[[BOS]]` for each state and reports the top argmax ids (counted over
the sample) and the mean probability on PAD, BOS, and EOS. It matches the
first iteration of greedy generation exactly, so it shows what generation
will emit before generation runs.

**Wired into**: `collapse_fix_harness.py`'s validation block only (100 rows,
eval mode, no grad). Logged as `pos0_top` / `pos0_mean_p` on the `[VAL]`
line and stored under `"pos0"` in the JSONL. Not yet wired into
`train_stage`'s periodic `[val]` print; that is a follow-up.

**Verified**: runs end to end on a small real expert
(`top_ids`, `mean_p` returned as expected).

**Not yet on a trained model.** The question for the next run is whether
`top_ids` is dominated by 0 or 2 (PAD/BOS). If it is, the widened penalty
is the right fix. If it is dominated by 3 (EOS), the empty outputs are real
EOS stops and the lever is elsewhere.

## Review of `slurm-4829926` (widened blank penalty + `pos0` diagnostic, pool off)

**Setup**: pool off, seed 0, widened penalty (PAD/BOS/EOS), `pos0` logged at
each validation point. Stopped by the 3.5-hour budget at 634 of 1000 steps.

**What `pos0` showed** (first greedy step, 100 rows per checkpoint):

| step | `pos0_top` (id → decoded word) | `pos0` special mass (PAD/BOS/EOS) | state_sim within/across |
|---|---|---|---|
| 100 | 3299 ×100 (`Highway`) | 0.000 | 0.48 / 0.20 |
| 200 | 25404 ×84, 31836 ×16 (`remedy`, `badges`) | EOS 0.007 | 0.40 / 0.00 |
| 300 | **3 (EOS) ×100** | **EOS 0.717** | 0.94 / 0.89 |
| 400 | 31941 ×100 (`I`) | EOS 0.079 | 0.56 / 0.39 |
| 500 | 28630 ×100 (`cardinal`) | 0.000 | 0.58 / 0.44 |
| 600 | 2303 ×38, 7965 ×31, 24277 ×18 | EOS 0.0005 | 0.40 / 0.22 |

The reward samples match these first tokens, so the diagnostic is reading
the right thing. `TextOutput.forward`'s first step is the same computation
as `teacher_forced_forward` with `[[BOS]]` and no memory, so the argmax is
the first emitted token, not an approximation.

**Three findings**:

1. **The EOS-start empty output is real and confirmed at step 300** (EOS
   argmax 100/100, mass 0.72). That matches the earlier empty outputs.
2. **PAD and BOS are gone at trained checkpoints** (mass 0.000 at every
   checkpoint). At random init BOS held 0.97 of the first-step mass, so
   the state is far from random init. Whether the widened penalty caused
   this is not separated out yet, since the same run changed both.
3. **The dominant failure is now a constant first content token that ignores
   state.** At steps 400 and 500 every row's argmax is the same token
   (100/100) even though state is differentiated (across-source 0.39–0.44).
   Each row then degenerates into repetition (`I I I...`, `cardinal ⁇ ⁇...`,
   `ictioniction ×N`). This is decoder-side: the state reaches the decoder,
   and the first step ignores it. The current losses act on the full
   distribution (`decoder_diversity_loss` repulsion on `probs0`), so the
   argmax can stay fixed while the tail moves. Not fixed yet.

**Regression signal**: at step 300 this run's state was collapsed (0.94/
0.89, koleo +1.19, cm_state 3.5), while the EOS-only pool-off run
`ab_off` (`slurm-4829831`, seed 0, same everything else) had 0.11/0.00.
That is the same step and the same seed. The only change is the widened
penalty, which is active from step 1 (blank_w is 0.21 at step 50), so
the divergence starts there. Among pool-off runs, the step-300 state is
healthy (`4829152`, `ab_off`) in two cases and collapsed in one (this).
It is one seed, so it is not proven causal. It may be chaotic sensitivity
to the first gradients rather than the penalty itself. The controlled test
is below.

**Change — switchable blank-penalty id set**: `HYPERPARAMS["reward"]
["blank_penalty_ids"]` (default `(PAD, BOS, EOS)`), read by
`compute_task_loss`. The harness override `HARNESS_BLANK_IDS=eos` restores
the original EOS-only penalty, so the A/B changes exactly one thing. Verified
through the real `compute_task_loss`: `all` gives the special mass 0.9951,
`eos` gives P(EOS) ≈ 0 for the same distribution.

**Next**: a controlled A/B of `eos` vs `all` (pool off, seed 0, ~600 steps,
`pos0` on). If `eos` reproduces `ab_off`'s healthy step 300 (≈0.11/0.00) and
`all` collapses as here, the widened penalty is the cause. The decoder's
constant-first-token issue (finding 3) is a separate problem. Its candidate
fixes are: regularize the first-step argmax or the top-1 margin directly
(not the full distribution), or a state-conditioned contrast on the first
step. Not implemented; needs its own round.

## `pos0_diagnostics` extended: state influence on the first token (`slurm-4830018`)

**Why**: the `det_a` run (`slurm-4830018`) showed the first-step argmax becoming
state-invariant by about step 200. The top-1 margin grows well past the
across-row spread of the top logit (`row_std/margin` falls from 0.50 at step 100
to 0.01–0.2 after). The state reaches the decoder but stops changing the
decision. A diagnostic was needed to tell whether the state is lost inside the
decoder's cross-attention or downstream of it.

**Change**: `pos0_diagnostics` now returns two more fields.
- `margin_mean`: mean top-1 minus top-2 logit at position 0.
- `row_std_top`: across-row std of the top-1 logit (for the modal argmax).
- `xattn_ratio`: norm of the across-row std of the decoder's first-step
  cross-attention output, divided by the norm of the mean query entering it. A
  forward hook on `expert.decoder_layer.multihead_attn` captures the first call
  for each state and is removed immediately after.

**Verified**: runs on a small real expert, returns all fields, and leaves no
hooks attached. The trained-decoder numbers from `bench/` are pending (the
background run is in progress; see `CHANGELOG_collapse_fix_harness.md` Round 12).

**Decision rule**: small `xattn_ratio` together with small `row_std/margin` →
the state is lost inside cross-attention (fix at the decoder input). Large
`xattn_ratio` with small `row_std/margin` → lost downstream of it (fix in the
FFN or final norm).

## Environment note (local testing only)

Importing `models/Experts/TextOutput.py` locally pulls `datasets` through
`Encoders/TextEncoder.py`, which fails here on a pandas/numpy binary
mismatch in the MLenv conda environment. The test used a stub for
`Encoders.TextEncoder` (`createTokenizer` only). This does not affect the
cluster environment (`mulEnv`), which ran these jobs without problems.

## Status as of this writing

Every diagnostic and the checkpoint-safety fix are confirmed working on
real runs. `encoder_koleo_loss` is confirmed working (with occasional
relapse and a long stretch stuck in the elevated range; see the regression
review above). The A/B confirmed that the `pool_koleo` term reproduces the
regression and that disabling it removes the early collapse. The blank
penalty now covers PAD/BOS/EOS (verified locally; not yet on a real run).
`pos0_diagnostics` is in place to test whether PAD/BOS or EOS is the
dominant first-step token on a trained model. `pool_koleo_loss` is confirmed to duplicate `koleo_loss`
early on and is now disabled (coefficient 0.0). `most_common_pred_fraction`
is confirmed working on real logs (it clearly exposes the constant-output
collapse). `common_mode_ratio` is unit-tested only and has not yet run on a
real job. Open threads: the
`encoder_koleo` elevated stretch (diagnostic `cm_memory` now logged, centring
fix pending), and the decoder-side empty/constant-output collapse.
