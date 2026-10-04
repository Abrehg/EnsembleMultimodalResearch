# Changelog: collapse_fix_harness.py

New file — the offline, real-GPU, real-d_model=1024 verification harness
for the collapse investigation (see `CHANGELOG_text_to_text.md`/
`CHANGELOG_EncCombine.md` for the full context). Deliberately reuses
`text_to_text.py`'s real functions directly (`encode_to_state`,
`compute_task_loss`, `t2t_rl_term`, `salience_diversity_loss`/
`salience_margin_loss`, `decoder_diversity_loss`/`_buffer_diversity`,
`koleo_loss`, `gate_scale_summary`, `build_model`, `SpmCodec`, etc.)
rather than reimplementing them, specifically so this harness can never
silently diverge from what a real `text_to_text.sh` run actually does.
Only the per-row loop's outer scaffolding (data loading, epoch/step
bookkeeping, checkpoint-free operation) is new. Always builds `combine`/
`router`/`expert_registry` from a fresh init (`checkpoint_prefix=
"t2t_HARNESS_THROWAWAY_NO_REAL_FILE"`, a prefix no real file will ever
match) — deliberate, so this harness tests "does the architecture train
correctly from scratch" in isolation, not "can old collapse-era weights
recover."

This file's own history is almost entirely "mirror whatever
`text_to_text.py`'s `train_stage` just gained" — see
`CHANGELOG_text_to_text.md` for the *why* behind each of these; this
entry just tracks where the harness picked each one up.

## Round 1 — initial creation (P1.1/P1.2 verification)

Created to verify P1.1 (decoupling anti-collapse coefficient ramps from
`total_steps`) and P1.2 (raised `scheduled_sampling_prob`) at real GPU
scale, since those two fixes were designed and locally reasoned about but
never confirmed against real training dynamics. Fixed a methodology bug
up front: `_schedule_weights` (the supervised/RL blend) and the LR
scheduler both use `total_steps` as their progress-fraction denominator,
so this harness explicitly passes the REAL production `total_steps`
(~68,700) into both, even though it only executes `HARNESS_TARGET_STEPS`
itself — otherwise a short run's own small step count would make both
ramp to full strength almost immediately, measuring something that
doesn't match what the real run experiences at a given `global_step`.

**Result**: P1.1/P1.2 failed at real scale (`unique_pred_fraction` still
pinned at 0.003-0.005 across 8000+ steps) — the finding that motivated
P2.2 and, eventually, the whole architecture investigation.

## Round 2 — P2.2 (`decoder_diversity_batch_contrastive_loss`) mirrored in

Added the same batch-boundary block `text_to_text.py` gained for the new
batch-local supervised-contrastive term, including the shared
single-`encode_to_state`-call-per-row + single-`backward()` pattern (both
`koleo_loss` and the contrastive term need a live graph over the same
batch, and each row's own graph from the main loop is already freed by
that row's own `backward()` call, so this re-encodes the batch once more
rather than twice).

**Result**: also failed at real scale (`slurm-4772416`\slurm-4781717`):
`unique_pred_fraction` still pinned despite the contrastive term's logged
value being genuinely nonzero — the finding that triggered the Phase 0
diagnostics in `CHANGELOG_text_to_text.md`.

## Round 3 — Phase 0 diagnostics mirrored in

Picked up `state_similarity`'s call-site change (3-tuple -> 4-tuple
return from `evaluate_rewards`) and the new `koleo` value in the periodic
log line.

## Round 4 — `gate_scale_summary` mirrored in

Added `gates = t2t.gate_scale_summary(model)` to the same periodic log
block, right after the architecture fixes in `CHANGELOG_EncCombine.md`/
`CHANGELOG_Logic.md` landed — this is the run
(`collapse_fix_harness_slurm-4795036.out`) that produced the "healthy
early, relapses later" finding, and the follow-up `slurm-4800583` run
(denser 150-step eval, `HARNESS_TARGET_STEPS=3000`) that used the newly
logged gate values to rule out "the gates reopened" as the relapse
mechanism (see `CHANGELOG_text_to_text.md`'s `gate_scale_summary` entry
for the full result).

## Round 5 — `encoder_koleo_loss` mirrored in

Updated the batch-boundary block's `encode_to_state` call to
`return_encoder_memory=True`, pooled the returned `memory` tensor the same
way as `state`, and added `encoder_koleo = t2t.koleo_loss(...)` with its
own coefficient into `batch_term_loss`, plus the new value in the periodic
log line.

**Result** (`slurm-4829152`, 8000 steps): `encoder_koleo` itself
genuinely recovered to a healthy range for a long stretch (~step
4200-7000) after an elevated first ~4000 steps — the term works. But
`state_sim`/`unique_pred_fraction` still showed full collapse at some of
those same later checkpoints, and the decoder's dominant failure mode
shifted from full-length repeated-token loops to a single fluent default
word repeated across sources — real progress, not full resolution. Full
analysis in `CHANGELOG_text_to_text.md`'s `encoder_koleo_loss` entry.

## Round 6 — `pool_koleo_loss` + `most_common_pred_fraction` mirrored in

Both added directly in response to Round 5's result (see
`CHANGELOG_text_to_text.md` for the full reasoning):

- Updated the batch-boundary block's `encode_to_state` call to unpack 3
  values (`r_state, r_memory, r_combined`) instead of 2, added
  `pooled_pools`/`pool_koleo` alongside the existing
  `pooled_memories`/`encoder_koleo`, and added `pool_koleo` to the
  periodic log line.
- Updated the `t2t.evaluate_rewards(...)` call site to unpack 5 return
  values instead of 4 (new `most_common_pred_fraction` inserted right
  after `unique_pred_fraction`), added it to the per-checkpoint JSONL
  `result` dict and the `[VAL]` log line.

**Result** (`slurm-4829741`, 8000 steps): regressed. See Round 7.

## Round 7 — regression review: `pool_koleo_coef` → 0.0, common-mode diagnostic mirrored in

**Trigger**: `slurm-4829741` regressed against `slurm-4829152`. Early state
diversity was worse, and `most_common_pred_fraction` was 1.000 at most
validation points (single-answer collapse across all three sources).

**Analysis** (full detail in `CHANGELOG_text_to_text.md`'s "Regression
review" section): in this run's own log, `pool_koleo` tracked `koleo` to
within 0.016 on average while the router gate was small (steps ≤ 1000),
so the Round 6 term was duplicating an existing gradient and doubling
koleo pressure on the pooled representation. Separately,
`encoder_koleo` sat in the collapsed range (~+2.2 to +2.8) for ~2000 steps
despite its active coefficient. A shared-offset mechanism is the likely
cause, but it has not been measured yet.

**Changes**:
- No code change to the harness's loss logic. It reads
  `t2t.HYPERPARAMS["reward"]` (`coefs`), so the `pool_koleo_coef: 0.0`
  change in `text_to_text.py` applies here automatically. The term is
  still computed and logged.
- Added `cm_memory` and `cm_state` to the periodic log line, using
  `t2t.common_mode_ratio(...)` on the stacked `pooled_memories`/
  `pooled_states` already in scope in the batch-boundary block. Measures
  whether a shared offset dominates the vectors that `koleo_loss`
  normalizes.

**Verified**: `py_compile` clean. `common_mode_ratio` unit-tested (diverse
batch 0.37, offset-dominated batch 106, single row NaN). Zero coefficient
confirmed to remove the gradient while the value is still computed.

**Not yet run.** Suggested next run: a short A/B (≈1000 steps, pool_koleo
off, which is now the default) to check whether state diversity recovers
toward `slurm-4829152`'s early numbers. Read `cm_memory` alongside
`encoder_koleo` in that run to test the offset hypothesis.

## Round 8 — pre-dispatch hygiene for an A/B rerun

**Trigger**: asked whether to redispatch the harness after Round 7. Checking
what the next run would actually produce surfaced three problems that would
have made it misleading:

1. **No seeding.** Neither `collapse_fix_harness.py` nor `text_to_text.py`
   seeds torch or the global `random` module, so combine/router/expert init
   differs on every run. A single-seed comparison couldn't separate an
   effect from init variance.
2. **Shared results file.** `RESULTS_PATH` was a fixed name, and
   `collapse_fix_harness_results.jsonl` already held 140 lines from earlier
   runs. Every run appended to it, so new results would mix with old data.
3. **Concurrency unknown.** Two harness jobs appending to the same file would
   interleave their results. Whether another job is still running can only be
   checked with `squeue` on the cluster, not from this environment.

**Changes** (`collapse_fix_harness.py` only; `text_to_text.py` unchanged):

- **Seeding**: `main()` now calls `torch.manual_seed(HARNESS_SEED)` and
  `random.seed(HARNESS_SEED)` before anything builds a model. `HARNESS_SEED`
  defaults to 0 and comes from the environment. Two arms given the same seed
  start from identical init, so their difference is attributable to the code
  change.
- **Run-scoped results file**: `RESULTS_PATH` is now
  `collapse_fix_harness_results_{RUN_TAG}.jsonl`, where `RUN_TAG` is
  `HARNESS_RUN_TAG`, falling back to `SLURM_JOB_ID`. Each run writes its own
  file. The existing 140-line file is left untouched as history.
- **Coefficient override for the A/B**: `main()` takes a copy of
  `HYPERPARAMS["reward"]` and applies `HARNESS_POOL_KOLEO_COEF` to
  `pool_koleo_coef` if set. The shared dict is not mutated. Unset, the
  default (0.0 after Round 7) applies, so the "off" arm needs no env var.
  The "on" arm uses `HARNESS_POOL_KOLEO_COEF=1.0`.
- **Logging**: the seed, run tag, and effective coefficients are printed at
  startup, so each log records what it actually ran.

**Verified**: `py_compile` clean. The override and seed lines were checked by
grep. Not run.

**Dispatch plan** (not yet executed):
```bash
# arm A: pool_koleo off (current default)
HARNESS_TARGET_STEPS=1000 HARNESS_EVAL_EVERY=100 HARNESS_MAX_HOURS=1.5 \
  HARNESS_SEED=0 HARNESS_RUN_TAG=ab_off sbatch --time=02:00:00 collapse_fix_harness.sh
# arm B: pool_koleo on (regression-reproduction arm)
HARNESS_TARGET_STEPS=1000 HARNESS_EVAL_EVERY=100 HARNESS_MAX_HOURS=1.5 \
  HARNESS_SEED=0 HARNESS_RUN_TAG=ab_on HARNESS_POOL_KOLEO_COEF=1.0 \
  sbatch --time=02:00:00 collapse_fix_harness.sh
```
The two arms share a seed, so they start from identical init. Before
submitting, confirm with `squeue -u $USER` that no harness job is still
running.

**What to read from the output**: `state_sim_within/across` and
`most_common_pred_fraction` at each `[VAL]` line, and `cm_memory` alongside
`encoder_koleo` in the step lines. The question is whether arm A's early
state diversity recovers toward `slurm-4829152`'s numbers (step 300
state_sim_within ≈ 0.21) and whether arm B reproduces `slurm-4829741`'s
early collapse (≈ 0.75). If both arms look the same, the regression was
init variance, not the term.

## Round 9 — A/B results, budget miss, blank-penalty fix, pos0 diagnostic, cleanup

**Runs**: `slurm-4829831` (`ab_off`, `pool_koleo_coef=0.0`) and `slurm-4829832`
(`ab_on`, `pool_koleo_coef=1.0`), both seed 0, both 1000-step targets. Full
analysis is in `CHANGELOG_text_to_text.md` ("A/B result"). Summary: pool on
reproduces the earlier early collapse at the same step (state 0.78/0.75 at
step 300, koleo +1.7) and pool off stays differentiated (0.11/0.00). The
pool term is confirmed as the cause.

**My error — budget**: I estimated about 45 minutes per arm. Both arms hit
`HARNESS_MAX_HOURS=1.5` before 1000 steps: `ab_off` stopped at 481 steps and
`ab_on` at 651. Observed pace was about 8–12 s/step (step 50 at 7–13 min),
slower than the earlier 8000-step run's average. The arms also got less
done than the earlier run's early pace suggests. Possibly node contention;
not verified. Recommended budget for 1000 steps with eval every 100:
`HARNESS_MAX_HOURS=3.5`, `--time=04:00:00`.

**Changes**:
- `compute_task_loss`: the blank penalty now covers PAD, BOS, and EOS
  (all decode to `''`), not EOS alone. Same ramp and gating. Verified
  through the real `compute_task_loss` (penalty equals the summed special
  mass at position 0). The `text_to_text.py` side is detailed in its
  changelog.
- `pos0_diagnostics(expert, states)` added to `text_to_text.py`.
  `collapse_fix_harness.py` runs it on 100 validation rows in eval mode with
  no grad at each validation checkpoint. Output goes to the `[VAL]` line
  (`pos0_top`, `pos0_mean_p`) and to the JSONL under `"pos0"`.
- Not wired into `train_stage`'s periodic print. That is a follow-up.

**Cleanup**: the legacy shared results file (140 lines, covering
`slurm-4829152`, `slurm-4829741`, and earlier runs) was moved, not deleted, to
`text_to_text_data/collapse_fix_harness_archive/collapse_fix_harness_results_legacy_shared.jsonl`.
Reversible, and it holds the step-300 numbers the earlier changelogs cite.
The two A/B results (`collapse_fix_harness_results_ab_off.jsonl`,
`collapse_fix_harness_results_ab_on.jsonl`) stay in place. The old `.out`
and `.err` logs are untouched, since they will be cleared separately.

**Known, not changed — harness warning**: each job's `.err` shows a PyTorch
`UserWarning` that a non-bool attention mask is converted to bool on every
call, which slows the fast path. The source is the causal mask built by
`TextOutput._causal_mask`, which is a float `-inf` mask. Fixing it means
changing the mask dtype in `models/Experts/TextOutput.py`, which is outside
this file. It is a performance issue only, and I didn't touch it.

**Verified**: `py_compile` clean for both files. `pos0_diagnostics` run on a
small real expert. Blank penalty checked through the real loss. Not yet run
on the cluster with the widened penalty.

## Round 10 — review of `slurm-4829926`, blank-penalty switch, windowed logging

**Run**: `slurm-4829926` (`pos0` tag, pool off, seed 0, widened penalty).
Stopped at 634 of 1000 steps by the 3.5-hour budget. Findings are in
`CHANGELOG_text_to_text.md` ("Review of `slurm-4829926`"). Short version:
- The EOS-start empty output is confirmed at step 300 (EOS 100/100, mass 0.72).
- PAD and BOS mass is 0.000 at every trained checkpoint.
- The dominant failure is now a constant first content token that ignores
  state (100/100 rows at steps 400 and 500), followed by a repetition loop.
- Step 300 state collapsed (0.94/0.89) where `ab_off` (same seed, EOS-only) did
  not (0.11/0.00). That points at the widened penalty, but one seed is not
  proof.

**Changes** (`collapse_fix_harness.py`):
- `HARNESS_BLANK_IDS=eos|all` (default `all`). Sets
  `coefs["blank_penalty_ids"]`. `eos` restores the original EOS-only penalty.
  Anything else raises. The chosen set is printed at startup as `[coefs]
  blank_penalty_ids=...`.
- Windowed logging. The `step%50` line now reports `ce_avg50` (mean CE over the
  window), `special0` (mean PAD+BOS+EOS mass at position 0), and `eos0`,
  computed from the `probs0` that `compute_task_loss` already returns. The old
  single-row `ce` swung from about 2 to about 185 between log lines, which made
  trends unreadable.
- Startup now also prints the effective `pool_koleo_coef`, `encoder_koleo_coef`,
  `koleo_coef`, and `blank_penalty_ids`.

**Pace problem (open)**: 634 steps in 210 min, about 20 s/step. Earlier, the
8000-step run averaged about 2.6 s/step (343 min total). Nothing in the code
since that run adds meaningful per-step cost (cheap stacks, one `isin`, a few
`.item()` syncs). Most likely node or GPU contention. Not verified. Check
`sacct`/`nvidia-smi` on the node before trusting any timing, and budget
generously until it's understood.

**Recommended controlled A/B** (same seed, pool off, one variable changed):
```bash
HARNESS_TARGET_STEPS=600 HARNESS_EVAL_EVERY=100 HARNESS_MAX_HOURS=4 HARNESS_SEED=0 \
  HARNESS_RUN_TAG=blank_eos HARNESS_BLANK_IDS=eos sbatch --time=05:00:00 collapse_fix_harness.sh
HARNESS_TARGET_STEPS=600 HARNESS_EVAL_EVERY=100 HARNESS_MAX_HOURS=4 HARNESS_SEED=0 \
  HARNESS_RUN_TAG=blank_all HARNESS_BLANK_IDS=all sbatch --time=05:00:00 collapse_fix_harness.sh
```
Read at step 300: `blank_eos` should match `ab_off` (≈0.11/0.00). If
`blank_all` collapses there again (≈0.9/0.9), the widened penalty is the cause.
Also check `pos0_top` at 400–600 for whether the constant-first-token problem
depends on the penalty.

**Verified**: compile clean. The override path was checked through the real
`compute_task_loss`: `eos` gives P(EOS) ≈ 0 and `all` gives 0.9951 for the same
distribution. Not yet run on the cluster.

**Still open** (not changed): the `UserWarning` about the float causal mask
(performance, in `TextOutput._causal_mask`). And the constant-first-token
decoder failure, which needs a loss change on the first decode step, not a
harness change.

## Round 11 — review of `slurm-4829973` (blank_eos) and `slurm-4829974` (blank_all), determinism, margin diagnostics

**Runs**: 600-step controlled A/B, pool off, seed 0, one variable changed
(`HARNESS_BLANK_IDS=eos` vs `all`). Both stopped on the time budget or at
target; both finished.

**Finding 1 — the runs are not reproducible.** Identical configs diverge at
step 50: koleo 0.3055 (`4829926`), 0.3236 (`4829974`), 0.2936 (`4829973`); cm_state
0.992 / 1.047 / 1.008. Seed 0 is set, but no deterministic mode was on. The
step-300 numbers for the same config disagree sharply: `4829926` collapsed
(0.94/0.89), `4829974` did not (0.15/−0.01). So the earlier "widened penalty
causes early collapse" reading is not supported. It is one non-reproducible
run. No single-run A/B result from before this round can be trusted for small
effects.

**Finding 2 — the penalty set is not the bottleneck.** Both arms collapse by
step 400–600. `blank_eos` at 300 (0.13/0.03) matches the earlier pool-off run
(0.11/0.00), but goes collapsed at 500–600 (0.99/0.97, 0.9997/0.9988).
`blank_all` is healthy at 300 and collapses at 400 (0.86/0.81) and 600.

**Finding 3 — EOS takes over again while the penalty is weak.** The blank
penalty's weight is on a ramp, so during these steps it is only 0.2–0.4
(`blank_w` in the log). The per-row CE it competes with is 10–100. At step 400
`blank_eos` has EOS 100/100 (mass 0.71), and at 500 `blank_all` has EOS 100/100
(mass 0.97). The ramp keeps the penalty too weak to hold EOS down through the
first 1000 steps.

**Finding 4 — the constant first token is in every checkpoint of both arms.**
`pos0_top` is 90–100/100 identical argmax at most checkpoints, with state
differentiated across sources. The argmax changes between checkpoints (e.g.
`12267`, `10105`, `207`), but the decoder emits one token for everything at any
given point. The token does decode to a real word, and the repetition loop
follows.

**Changes** (`collapse_fix_harness.py`, `collapse_fix_harness.sh`,
`text_to_text.py`):
- **Deterministic mode.** `torch.backends.cudnn.deterministic=True`,
  `cudnn.benchmark=False`, and `torch.use_deterministic_algorithms(True,
  warn_only=True)` in the harness, plus `export CUBLAS_WORKSPACE_CONFIG=:4096:8`
  in the job script (it must be set before CUDA starts). `warn_only` keeps any
  op without a deterministic kernel from crashing the run. Runs should now
  reproduce step for step on the same node.
- **`HARNESS_BLANK_COEF`.** A fixed blank-penalty weight that replaces the ramp
  when set, so the penalty can be tested at full strength from step 1.
  Unset, the ramp applies, which keeps the current behavior.
- **Margin diagnostics in `pos0_diagnostics`.** Two new fields, logged on the
  `[VAL]` line as `pos0_margin` and `pos0_row_std`, and stored under `"pos0"` in
  the JSONL. `pos0_margin` is the mean top-1 minus top-2 logit gap at position 0.
  `pos0_row_std` is the across-row standard deviation of the top-1 logit. If
  `row_std` is much smaller than `margin`, the state reaches the decoder but
  cannot flip the first token. If it is comparable, the state is changing the
  decision and the bottleneck is elsewhere. This is the decisive test for
  finding 4.

**Analysis on the production checkpoints**: the trained combine, router, and
text-output checkpoints were moved by the user to `bench/`. The first attempt
at the first-step analysis on them was stopped before it finished (a
background-task teardown, not a result), so nothing from it is recorded. It is
re-run in Round 12.

**Verified**: compile clean for both files. Not run on the cluster.

**Recommended next run** (determinism means these should reproduce):
```bash
HARNESS_TARGET_STEPS=600 HARNESS_EVAL_EVERY=100 HARNESS_MAX_HOURS=4 HARNESS_SEED=0 \
  HARNESS_RUN_TAG=det_a HARNESS_BLANK_IDS=all HARNESS_BLANK_COEF=1.0 \
  sbatch --time=05:00:00 collapse_fix_harness.sh
```
Run it twice with the same settings. If the two runs match step for step, the
determinism fix worked and the single-run comparisons can be trusted again. If
they don't, the source is still nondeterministic and the warning output shows
which op.

## Round 12 — review of `slurm-4830018` (det_a), complete determinism, cross-attention diagnostic

**Run**: `slurm-4830018`, tag `det_a`, 600 steps, pool off, `HARNESS_BLANK_COEF=1.0`
(confirmed in the step lines: `blank_w 1.0000` throughout). Only one `det_a` run
exists, so determinism is **not yet verified**; that needs a second identical run.

**Results (VAL)**:

| step | unique | most_common | state within/across | pos0_top | margin | row_std | row_std/margin |
|---|---|---|---|---|---|---|---|
| 100 | 0.095 | 0.67 | 0.47 / 0.27 | 7358 ×99 | 8.06 | 4.04 | 0.50 |
| 200 | 0.010 | 0.94 | 0.42 / 0.19 | 18071 ×93 | 0.50 | 0.104 | 0.21 |
| 300 | 0.005 | 1.00 | 0.35 / 0.18 | 7358 ×100 | 0.57 | 0.083 | 0.15 |
| 400 | 0.035 | 0.84 | 0.31 / 0.09 | 23661 ×100 | 4.47 | 0.045 | 0.01 |
| 500 | 0.010 | 0.995 | 0.49 / 0.13 | 2442 ×100 | 1.43 | 0.064 | 0.045 |

**Finding 1 — the state stops changing the first token early.** At step 100 the
state moves the top logit by about half the margin (row_std/margin 0.50), so it
does influence the decision. From step 200 on, the ratio drops to 0.01–0.2. The
first token is effectively state-invariant from about step 200, and this happens
while state diversity stays healthy (within/across 0.31–0.49 / 0.09–0.19). So the
state reaches the decoder, and the decoder stops using it.

**Finding 2 — the degeneration is partly reduced, not fixed.** Compared with
Round 9–10 runs, unique fraction is higher at 100 and 400, but most_common is
0.995 at step 500, which is worse. Mixed, not a clear improvement.

**Finding 3 — determinism is incomplete.** The stderr shows a memory-efficient
attention kernel and `cumsum_cuda_kernel` flagged nondeterministic. `warn_only`
lets them run, so the same config can still diverge.

**Changes** (`collapse_fix_harness.py`, `text_to_text.py`):
- **Force the math attention kernel.** `torch.backends.cuda.enable_mem_efficient_sdp(False)`
  and `enable_flash_sdp(False)` at harness import. The math path is
  deterministic and has the same semantics, just slower. Guarded with `hasattr`.
  `cumsum` remains a known source of tiny reordering noise in the length term; it
  is noted, not changed.
- **Cross-attention diagnostic in `pos0_diagnostics`.** A forward hook on
  `expert.decoder_layer.multihead_attn` captures, at the first decode step, the
  query entering cross-attention and its output for each state. `xattn_ratio` is
  the norm of the across-row std of that output divided by the norm of the mean
  query. It is logged on `[VAL]` as `xattn_ratio` and stored in the JSONL. Small
  values mean the state barely moves the decoder residual at this step. The hook
  is removed after use, so nothing stays attached (verified: 0 hooks remain).

**Decision rule** (to read from the next run and the trained-decoder check):
- If `xattn_ratio` is small while `pos0_row_std/margin` is also small, the state
  is lost **inside cross-attention**. The fix is at the decoder input (how the
  state enters: its common-mode, its normalization, or the cross-attention's
  learned sharpness).
- If `xattn_ratio` is large but `row_std/margin` stays small, the state is lost
  **downstream** of cross-attention (the FFN or final norm). The fix is there.

**Verified**: compile clean. The hook was checked on a small real expert (ratio
computed, hook removed). Not yet run on the cluster. The trained-decoder check
(`bench/` checkpoints) is running in the background; its numbers will be added
here when it finishes.

**Recommended next runs**:
```bash
# two identical runs to test determinism (must match step for step)
HARNESS_TARGET_STEPS=600 HARNESS_EVAL_EVERY=100 HARNESS_MAX_HOURS=4 HARNESS_SEED=0 \
  HARNESS_RUN_TAG=det_b HARNESS_BLANK_IDS=all HARNESS_BLANK_COEF=1.0 sbatch --time=05:00:00 collapse_fix_harness.sh
HARNESS_TARGET_STEPS=600 HARNESS_EVAL_EVERY=100 HARNESS_MAX_HOURS=4 HARNESS_SEED=0 \
  HARNESS_RUN_TAG=det_c HARNESS_BLANK_IDS=all HARNESS_BLANK_COEF=1.0 sbatch --time=05:00:00 collapse_fix_harness.sh
```

## Operational notes picked up along the way

- `--time`/`--gres` in `collapse_fix_harness.sh` were changed to use
  `${VAR:-default}` overrides (matching `test_datasets.sh`'s existing
  convention) so a short diagnostic-only run can override
  `HARNESS_TARGET_STEPS`/`HARNESS_EVAL_EVERY`/`HARNESS_MAX_HOURS` from the
  `sbatch` command line without editing the script.

## Status as of this writing

Rounds 1-6 all ran on the cluster and produced real results. Round 6's
`pool_koleo_loss` regressed (`slurm-4829741`), and the A/B in Round 9
confirmed it is the cause, so the term stays disabled (Round 7). Round 7's
`cm_memory`/`cm_state` diagnostics have now run in the A/B (`slurm-4829831`/
`slurm-4829832`), and their values are in the logs. Round 8's seeding,
per-run results file, and A/B override are in use. Round 9's blank-penalty
fix (PAD/BOS/EOS) and `pos0` diagnostic are implemented, verified locally,
and not yet run on the cluster. Next run: 1000 steps with the recommended
3.5-hour budget, reading `pos0_top` to decide whether the penalty fix is the
right lever.

Round 10 (`slurm-4829926` review) is done: the EOS-start output is confirmed,
PAD/BOS mass is zero at trained checkpoints, and the dominant failure is a
constant first content token that ignores state. Step 300 collapsed with the
widened penalty but not with EOS-only (one seed, so not yet causal). The
`HARNESS_BLANK_IDS=eos|all` switch and windowed logging are in place, and the
controlled `eos`-vs-`all` A/B is recommended. Not yet run.
