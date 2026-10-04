"""
Text-to-text tuning/training for the ensemble multimodal research model.

Overview
--------
This script builds a text-in / text-out version of the project's
``MultimodalModel`` (models/OverallModel.py), similar in spirit to
dim_tune.py: the text encoder is a *frozen* copy of the encoder trained by
enc_pre.py, and only the trainable pipeline downstream of it (encoder
combination -> router -> text-output expert) is optimised. ``text_only=True``
already strips the vision/memory/reasoning/search branches out of the model,
leaving exactly encode -> combine -> router -> {eos, text_output}.

Pipeline (staged; stops for review after tune and after train_stage1, like
enc_pre.py):

    1. tune         -> Optuna search for training hyperparameters (on a
                       small subset of the compiled data, mixing every
                       source), then exit(0) for review. One shared search
                       used by both training stages below -- not split per
                       stage.
    2. train_stage1 -> full resumable training run restricted to
                       STAGE1_SOURCES (logi_glue/trivia_qa/nq_open --
                       short/closed-form answers), with checkpoints after
                       every optimizer step and after every completed
                       epoch. Stops for review once finished.
    3. train_stage2 -> full resumable training run restricted to
                       STAGE2_SOURCES (asqa/deepmind_math/minerva_math --
                       longer/harder generation), warm-started from
                       train_stage1's finished weights (see build_model's
                       fallback_prefix). Independently resumable and
                       independently epoch-counted from stage 1 (see
                       HYPERPARAMS["epochs_stage1"]/["epochs_stage2"]).

Data pipeline (CSV based, on scratch)
--------------------------------------
Every source task is normalised into rows of
``{input, target, source, aux}`` and written to a single compiled CSV, which
is shuffled once in full (interleaving every source) before being written to
disk -- this is the "inbuilt shuffling of the entire dataset" the rest of the
pipeline relies on. The compiled CSV is then split (stratified per source)
into train.csv / val.csv; the tuning stage further samples small subsets out
of those two files. Downloads are defensive: a source that cannot be
prepared is skipped with a warning, mirroring enc_pre.py.

``source`` drives which reward/metric function scores a prediction at
evaluation time (exact match for most tasks; ROUGE-L + short-answer
containment for ASQA -- see ``compute_reward_metric``).

Training objective ("reward")
------------------------------
Autoregressive generation in ``TextOutputExpert.forward`` samples tokens
(multinomial), which is not differentiable end-to-end. Exactly as dim_tune.py
already decided for this codebase, the trainable signal is therefore a
*teacher-forced* cross-entropy loss (a proxy for the exact-match / ROUGE
reward, which are informative metrics but not usable gradients from
scratch), computed via the expert's own recursive decoder
(``teacher_forced_forward``). Three small auxiliary terms are added on top,
each with a small coefficient (``HYPERPARAMS["reward"]``):

    * length penalty   -- proportional to target-token count. Proxied by the
      *teacher-forced target length* rather than an actual generated length,
      to stay cheap on every step (a free-run generation every step would be
      far too slow); this still shapes the training distribution toward
      concise answers over the course of training.
    * ponder-cost penalty -- the ACT halting module's accumulated recursive
      iterations for the forced pass (``expert._recursive_decode``'s second
      return value), normalised by the per-call recursion budget. This is
      the differentiable "number of iterations" signal.
    * Brier calibration loss -- ``((confidence - correctness) ** 2).mean()``
      computed *per token* (confidence = max softmax prob, correctness =
      detached argmax-vs-target match) rather than per sequence, so it stays
      differentiable and cheap every step. Sequence-level exact-match /
      ROUGE reward is tracked separately (non-differentiably) via periodic
      free-run generation in ``evaluate_rewards``, purely for logging.

Self-correction ("catch its own mistakes")
-------------------------------------------
With probability ``HYPERPARAMS["reward"]["self_correct_prob"]`` per training
row, ``self_correct_step`` lets the *current* model free-run generate its own
candidate answer (no_grad; sampling isn't differentiable), optionally
corrupts it with a single injected error (``corrupt_answer``), then trains
the model -- via a fresh, ordinary differentiable teacher-forced pass -- to
answer "yes"/"no" to "Is the proposed answer correct?". This is added to the
step loss with its own small coefficient.

Frozen encoder / half precision
--------------------------------
The frozen text encoder is loaded from enc_pre.py's best checkpoint (found
via ``find_weights("text_encoder", "best")``) and never receives gradients.
"Half quantized" is implemented as fp16 autocast around the encoder's forward
call (not a literal ``.half()`` weight conversion): TextEncoder's sinusoidal
position embedding is hardcoded fp32, so a real ``.half()`` would crash with
a dtype mismatch the first time it adds fp16 token embeddings to that fp32
tensor. Autocast gets the same reduced-precision compute/memory benefit on
CUDA without touching TextEncoder's internals, and the output is cast back
to fp32 before being handed to the (fp32) trainable pipeline.

Weights & greppability
-----------------------
enc_pre.py and this script share one project-level ``weights/`` directory
(not scratch) so both scripts' artifacts are retrievable from the same
place. Filenames follow enc_pre.py's convention exactly:

    {component}__{tag}__epoch{E}_batch{B}[_valloss{L}].pt
        component in {t2t_stage1_combine, t2t_stage1_router,
                      t2t_stage1_expert_registry, t2t_stage1_text_output,
                      t2t_stage2_combine, t2t_stage2_router,
                      t2t_stage2_expert_registry, t2t_stage2_text_output}
                     -- each training stage gets its own checkpoint
                     namespace (see STAGE1_SOURCES/STAGE2_SOURCES/
                     train_stage); stage 2 warm-starts from stage 1's
                     finished files but writes its own progress separately.
        tag       in {latest, best}

Every trainable component -- ``combine``, ``router``, ``expert_registry``,
``text_output`` -- gets its own standalone file via that module's own
``store_weights``/``load_weights``, directly loadable by another script the
same way ``text_encoder``/``text_output`` already are; there is no separate
resumable-wrapper file. Mid-run resume state (optimizer state, RNG,
epoch/batch/step counters) is instead an optional ``extra`` payload
embedded in ``t2t_expert_registry``'s *'latest' tag only* (see
ExpertRegistry.store_weights) -- 'best' always stays a clean, minimal
weights-only file. ``find_weights(component, tag)`` locates any of them by
glob + regex on the filename (mirrors enc_pre.py). At the end of training
the script prints the located best ``t2t_text_output`` file explicitly.

Both this script and enc_pre.py use the same warmup-then-cosine-decay LR
schedule (``make_lr_lambda``) so later optimizer steps make progressively
smaller changes as training approaches convergence.
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

# Bulk, regenerable data (compiled CSVs, HF cache) lives on scratch; small
# but load-bearing state for a multi-day staged run -- tuned hyperparameters,
# the Optuna study DB, stage progress, trained weights -- lives under the
# project directory on barn instead, since scratch filesystems are
# periodically purged and losing any of these would silently reset an
# in-progress pipeline (weights were already here; STUDY_DIR/PROGRESS_PATH
# moved off scratch for the same reason).
SCRATCH_DIR = "/gpfs/u/home/ENMM/ENMMsrtk/scratch/"
DATA_DIR = os.path.join(SCRATCH_DIR, "text_to_text_data")
STUDY_DIR = os.path.join(PROJECT_DIR, "studies", "text_to_text")
# All weight files live here on barn, never on scratch -- scratch is
# reserved strictly for datasets/HF cache (regenerable, and periodically
# purged). Every trainable component (combine/router/expert_registry/
# text_output) gets its own standalone file here; there is no separate
# resumable-wrapper file to go missing or land in the wrong place (see
# save_checkpoint's docstring for how mid-run resume state is carried
# instead).
WEIGHTS_DIR = os.path.join(PROJECT_DIR, "weights")
HF_CACHE = os.path.join(DATA_DIR, "hf_cache")
PROGRESS_PATH = os.path.join(PROJECT_DIR, "text_to_text_progress.json")

COMPILED_CSV = os.path.join(DATA_DIR, "compiled.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
CSV_FIELDS = ["input", "target", "source", "aux"]

TOKENIZER_PATH = os.path.join(MODELS_DIR, "Encoders", "multimodal_bpe.model")

# SentencePiece special ids (see TextEncoder / spm training).
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


# =========================================================================== #
# HYPERPARAMETERS  (everything tunable lives here, for easy visibility)
# =========================================================================== #
HYPERPARAMS = {
    "global": {
        "d_model": 1024,
        "seed": 42,
        "max_input_len": 96,             # word cap applied when compiling CSV rows
        "max_target_len": 32,            # token cap for teacher-forced targets
        # asqa's target is a full long-form paragraph answer (see load_asqa),
        # not a short factoid string like every other source -- capping it at
        # the same 32 tokens as everything else truncates it mid-sentence
        # *and* (via encode_target always appending EOS_ID right after
        # truncating) actively teaches the model to stop there, i.e. a false
        # early-EOS signal, not just lost information. Other sources' true
        # answers are already well under 32 tokens, so this only raises cost
        # for the one task that actually needs it.
        "max_target_len_overrides": {"asqa": 128},
        "val_fraction": 0.05,            # per-source holdout fraction
        "tune_train_subset": 4000,       # rows sampled from train.csv for tuning
        "tune_val_subset": 800,          # rows sampled from val.csv for tuning
        "checkpoint_every_batches": 1,   # save 'latest' after every optimizer step
        "eval_every_batches": 300,       # run full validation every N optimizer steps
        "half_precision_encoder": True,  # fp16-autocast the frozen text encoder on CUDA
        # Only a running 'latest' and 'best' checkpoint are ever kept per
        # component (see save_checkpoint), bounding disk usage by a small
        # constant regardless of run length.
    },

    "reward": {
        "task_loss_coef": 1.0,
        "length_penalty_coef": 0.01,
        "ponder_cost_coef": 0.01,
        # Scheduled (see scheduled_coef/compute_task_loss's brier_coef
        # param), not flat -- per user direction: calibration pressure
        # should start small (the model is mostly wrong about everything
        # early on; a strong Brier penalty there just punishes necessary
        # uncertainty) and grow substantial by the same training-progress
        # point the RL blend's own share does, once there's something
        # real to be calibrated about. Replaces the old flat "brier_coef"
        # (was 0.05) -- see the 4-candidate offline comparison this was
        # tested against (HYPERPARAMS["reward"]'s composite-complexity
        # postmortem) for why a flat mid-value wasn't kept as-is.
        # Ramp horizon for brier/blank_answer_penalty/repetition_penalty/
        # content_closeness below, decoupled from the full stage's
        # total_steps (~68,700 for a 50-epoch stage-1 run) -- see
        # scheduled_coef's call site in train_stage. Added after a real run
        # (slurm-4752878) showed unique_pred_fraction already pinned at its
        # collapse floor at the very first validation (step 2400), while
        # these coefficients were still ~2-4% of their _end value at that
        # point (CE dominated them 100-2000x). Sharing total_steps meant
        # every prior offline verification of these coefficients also
        # ramped them to _end almost immediately (a short toy run's own
        # total_steps is small), so a toy test's "no permanent lock-in"
        # never actually exercised the real run's near-_start danger
        # window -- this value is a placeholder (no earlier log survived
        # to pin the true lock-in step precisely) and should be revisited
        # if a future run's own early validations narrow it further.
        "anti_collapse_ramp_steps": 3000,
        "brier_coef_start": 0.01,
        "brier_coef_end": 0.3,
        "self_correct_coef": 0.2,
        "self_correct_prob": 0.15,
        # Probability of replacing the teacher-forcing prefix with the
        # model's own no_grad free-run greedy rollout (see
        # _scheduled_sampling_tgt_in/compute_task_loss) -- exposes the
        # supervised loss to the model's own drift, which nothing else in
        # this composite does (everything else is teacher-forced on the
        # true gold prefix). Raised 0.2 -> 0.35 after a real run
        # (slurm-4752878) showed repetition_penalty/content_closeness_term
        # essentially never observing the model's actual free-run failure
        # modes (e.g. a token repeated to max_target_len) at the old rate --
        # the gold prefix almost never contains that pattern, so only this
        # probability's fraction of steps could ever expose it. This is the
        # one lever giving the loss composite real free-run exposure that
        # isn't diluted by anti_collapse_ramp_steps/total_steps at all.
        "scheduled_sampling_prob": 0.35,
        # ---- Degenerate-output penalties (see compute_task_loss) --------- #
        # Discovered via evaluate_rewards' sample logging: free-run greedy
        # generation was collapsing to either an empty answer (immediate
        # EOS) or a single token repeated until the length cap, in both
        # cases scoring 0 reward regardless of how good val_loss looked.
        # Both terms are computed from the same teacher-forced logits
        # already available in compute_task_loss -- no extra forward pass.
        # Scheduled (see scheduled_coef), same reasoning as brier_coef_start/
        # _end above, generalized: the 4-candidate offline comparison found
        # that ramping brier + these three shaping terms all up together
        # from a low start (candidate C) was the only configuration that
        # showed a real, reproducible improvement over every term sitting
        # at full strength from step 0 (candidate A, what slurm-4722503
        # actually ran) -- better final logi_glue precision (0.31 vs ~0.10)
        # despite an identical-looking aggregate reward and identical
        # logged sample predictions, meaning C converged to a policy that
        # guesses less indiscriminately on the *other* rows in the eval
        # sample. Values below are candidate C's tested starting points;
        # a follow-up sweep (per the user's "maybe not starting low
        # enough" hypothesis) is testing lower starts still -- see that
        # sweep's results before assuming these particular numbers are
        # final.
        "blank_answer_penalty_coef_start": 0.2,
        "blank_answer_penalty_coef_end": 1.0,
        # Token ids whose position-0 probability counts as a blank output (see
        # compute_task_loss's blank penalty). All three decode to '' -- verified
        # against models/Encoders/multimodal_bpe.model. Overridable per run
        # (collapse_fix_harness.py's HARNESS_BLANK_IDS=eos) so the widened
        # penalty can be A/B-tested against the original EOS-only version: the
        # widened penalty was added in slurm-4829926's predecessor round, and
        # that run then collapsed at step 300 (state 0.94/0.89) where the
        # EOS-only pool-off run did not (0.11/0.00, same seed). Not yet shown
        # to be causal (one seed), so it stays switchable.
        "blank_penalty_ids": (PAD_ID, BOS_ID, EOS_ID),
        "repetition_penalty_coef_start": 0.1,
        "repetition_penalty_coef_end": 0.6,
        # Small and deliberately so ("the rest of the reward is used for
        # using the correct word") -- see compute_task_loss's comment.
        # text_to_text.py only: not ported to ocr_to_text.py/
        # image_to_text.py/reasoning_to_text.py, which warm-start their own
        # text_output expert from an already-working one and shouldn't have
        # this still-experimental shaping term pulled in underneath them.
        "length_closeness_coef": 0.15,
        # Gates length_closeness_term to rows the model is already mostly
        # getting right under teacher forcing (see compute_task_loss) --
        # without this, a fixed short wrong output (e.g. constant "True")
        # scores this term well on any row with a short gold answer,
        # regardless of content. Diagnosed as one of two mechanisms behind
        # a real training run's collapse to constant "True" output (the
        # other being an uncalibrated shared RL baseline, see
        # rl_baseline_momentum below) -- see slurm-4631350 postmortem.
        "length_closeness_gate_threshold": 0.5,
        # Bounded ([0,1], linear), unlike CE's unbounded log penalty -- see
        # compute_task_loss's comment. text_to_text.py only, same scoping
        # rationale as length_closeness_coef above.
        #
        # Raised 0.2 -> 1.0 (task_loss_coef's own scale) after slurm-4671175's
        # postmortem: even with the salience-gate fix holding (see
        # salience_diversity_coef below) and koleo_coef preventing state
        # collapse, trivia_qa/nq_open still plateaued hard around a ~0.88
        # content-closeness value (near-zero probability on the actual
        # correct token) for the entire real run, while logi_glue kept
        # improving steadily. Offline root-causing (real data, toy encoder,
        # 2000-step runs) tested three candidate fixes against this exact
        # plateau: this coefficient boost (content_closeness_coef +
        # blank_answer_penalty_coef both raised to 1.0) bought a real but
        # modest ~4% improvement; a logi_glue-only curriculum warm-up before
        # introducing trivia_qa/nq_open bought none; and 4x model width
        # (d_model 64->256) also bought only ~4%. The real driver, confirmed
        # via SentencePiece tokenization stats on 500 real rows per source:
        # logi_glue repeats ~46 distinct answers thousands of times each
        # across its rows, while trivia_qa/nq_open's answers are 97%+ unique
        # (486/500 and 492/500 distinct in-sample) -- a single-exposure,
        # closed-book fact-recall problem, not a small-label classification
        # problem, and not something loss weighting/curriculum/toy-scale
        # capacity can manufacture repeated exposure for. This coefficient
        # bump is kept anyway (free, real, no downside observed in any of
        # the above runs) but is not expected to fully resolve
        # trivia_qa/nq_open on its own -- that needs real-scale evidence
        # (real pretrained encoder, full ~34,500-step run) that offline
        # testing can't cheaply substitute for, since the one real run with
        # this scale (slurm-4671175) had its salience gate dead for 46 of
        # its 50 epochs. Watch trivia_qa/nq_open reward specifically over
        # the next real restart, now that the gate fix is in place for the
        # entire run.
        # Scheduled -- see blank_answer_penalty_coef_start's comment just
        # above for the full rationale (this term ramps in sync with it,
        # brier_coef, and repetition_penalty_coef, all sharing scheduled_coef's
        # progress fraction).
        "content_closeness_coef_start": 0.2,
        "content_closeness_coef_end": 1.0,
        # Koleo-style repulsion on the decoder's own first-token output
        # distribution (see decoder_diversity_loss) -- root-cause fix for a
        # third distinct flavor of the same collapse this file keeps
        # finding: slurm-4695121 (gate fix + koleo + the content-closeness/
        # blank-penalty boost above, all active and healthy) still
        # collapsed to unconditionally predicting the single word "The" for
        # every source, starting ~85% through a 50-epoch run. Unlike prior
        # fixes (each patching the specific shortcut just observed), this
        # targets the category directly: emitting the same output
        # regardless of input becomes expensive in general, not just for
        # whichever token currently happens to be cheapest. Starting value,
        # not yet offline-verified at the scale that caught the "The"
        # collapse (see the module's offline-verification methodology).
        "decoder_diversity_coef": 0.5,
        # Rolling-history size decoder_diversity_loss compares each row
        # against (see its docstring for why a rolling buffer instead of a
        # live batch) -- bigger than batch_size so it accumulates broader
        # cross-source coverage than a single optimizer step's batch would.
        "decoder_diversity_buffer_size": 32,
        # P2.2 (see /Users/adityaasuratkal/.claude/plans/dazzling-toasting-moler.md):
        # decoder_diversity_batch_contrastive_loss's weight -- the missing
        # "positive pull" counterpart to decoder_diversity_coef's pure
        # history-repulsion. Added after slurm-4772416 (P1.1/P1.2 alone, a
        # real 8,000-step GPU run) showed unique_pred_fraction pinned at its
        # 0.005 collapse floor across all 26 validation checkpoints,
        # including the entire second half of the run where every scheduled
        # anti-collapse coefficient was already at full strength -- direct
        # confirmation that repulsion-only wasn't escaping the collapse, only
        # relocating it (the catalog of distinct degenerate output strings
        # grew over that run instead of shrinking). Starting value, not yet
        # offline-verified -- this needs its own harness run before a real
        # cluster commit, same as every prior anti-collapse mechanism in
        # this file.
        "decoder_diversity_contrastive_coef": 0.5,
        # Standard supervised-contrastive temperature (Khosla et al. 2020) --
        # lower sharpens the softmax over cosine similarities (stronger
        # gradient near-ties, more aggressive push/pull), higher softens it.
        # 0.1 is the commonly-used default in that literature; not yet tuned
        # for this specific setting.
        "decoder_diversity_contrastive_temperature": 0.1,
        # ---- Scheduled supervised/RL blend (see t2t_rl_term/_schedule_weights) -- #
        # RL's share ramps from 0 up to this ceiling over training progress,
        # supervised always the strict complement (1 - rl_w) -- deliberately
        # capped well below reasoning_to_text.py's 100% (this stage should
        # never fully hand over to RL), per explicit instruction: 25-50% range.
        # Lowered from 0.35 after the slurm-4631350 collapse postmortem --
        # RL's uncorrected share climbing that high, combined with a
        # per-source-uncalibrated baseline (now fixed, see rl_baselines in
        # train()) and the ungated length-closeness term above, is part of
        # why the collapse locked in and never recovered.
        "rl_ceiling": 0.20,
        # Per-source moving-average baselines (see train()'s rl_baselines
        # dict) share this same momentum -- previously a single global
        # scalar blended 5 near-always-0-reward tasks with logi_glue's
        # real signal, letting logi_glue's REINFORCE gradient dominate the
        # one shared generation policy. Now tracked independently per
        # source so each task's advantage is calibrated against its own
        # reward distribution.
        "rl_baseline_momentum": 0.9,
        # Entropy bonus on the RL term (see t2t_rl_term/TextOutput.forward's
        # return_entropy) -- discourages the sampled generation policy from
        # collapsing to a near-deterministic output distribution, the
        # proximate mechanism of the slurm-4631350 "always True" collapse.
        # Same scale as this file's other auxiliary coefficients.
        "entropy_coef": 0.02,
        # Anti-collapse regularizer on `state` (see koleo_loss) -- root-cause
        # fix for the same "always True" collapse, found after the salience-gate
        # and LinearAttentionBlock fixes (both real, both necessary) still
        # weren't sufficient: state's pooled representation was collapsing
        # toward a near-identical direction across every input during real
        # training, regardless of any decoder-side mechanism tried (masking,
        # self-attention blocking, cosine attention, boosted cross-attention
        # LR, direct injection -- nine variants tested offline, none worked).
        # koleo_loss (Sablayrolles et al. 2018, already used/validated in
        # enc_pre.py's vision SSL pretraining) pushes a batch of state's pooled
        # representations apart directly, addressing the collapse at its
        # source instead of trying to extract a shrinking signal downstream.
        # Verified offline over 4800 real rows with the real, unmodified loss
        # composite: state cosine stayed bounded (never approached the ~0.99+
        # collapse), free-run predictions showed sustained differentiation
        # (avg 2.45/3 unique vs. a flat 1.0/3 without this term), and task loss
        # dropped steadily throughout -- this coefficient matches what was
        # verified.
        "koleo_coef": 1.0,
        # Direct anti-collapse regularizer on combine.encoder's own
        # pre-pooling output (see encoder_koleo_loss/encode_to_state's
        # return_encoder_memory) -- added after a real harness run
        # (slurm-4800583) showed state collapse repeatedly relapsing well
        # after the ReZero architecture fix landed, even while the new
        # gates stayed small throughout (gate_scale_summary), pointing at
        # the underlying layer weights drifting toward a collapse-inducing
        # direction with nothing acting directly on them -- koleo_coef
        # above only ever regularizes the FINAL pooled state, several
        # layers downstream. Matches koleo_coef's own value as a starting
        # point (same loss shape, no existing data yet to justify a
        # different weight) -- revisit once a verification run's own
        # encoder_koleo metric establishes healthy vs. collapsed ranges for
        # this specific tensor the way koleo_loss's docstring already does
        # for the final state.
        "encoder_koleo_coef": 1.0,
        # Direct anti-collapse regularizer on combine.pool's own output
        # (AdaptiveAttentionPooling's `combined`/`latents`, pre-router) --
        # added after a real harness run (slurm-4829152, the first real run
        # with encoder_koleo_coef active) showed encoder_koleo itself
        # genuinely settle into a healthy range (-0.1 to -0.3) for a long
        # stretch (~step 4200-7000) -- confirming that term works -- while
        # state_sim_within/across STILL showed full collapse at some of
        # those exact same checkpoints (e.g. step 5100). With the encoder
        # itself confirmed healthy, the remaining collapse has nowhere left
        # to hide except combine.pool or router -- exactly the two modules
        # that got the architectural ReZero fix but never got their own
        # direct training-time diversity pressure, same gap encoder_koleo_coef
        # already closed for the encoder specifically.
        #
        # DISABLED (0.0) after slurm-4829741 regressed relative to slurm-4829152.
        # While the router's gate is small (state_rezero ~0, steps <= ~1000),
        # state == combined up to a scale, so pool_koleo was numerically the
        # same loss as koleo_coef's term: mean |koleo - pool_koleo| was 0.016
        # over steps <= 1000 vs 0.12 after. Enabling it doubled the effective
        # koleo pressure on the pooled representation right where the
        # regression appeared (state koleo at step 300 went +0.52 vs -0.16 in
        # the prior run). The term is kept computed and logged as a
        # diagnostic (its value still reports pooled geometry), but its
        # gradient is zeroed. Re-enable only if it's made genuinely
        # independent of the state path (e.g. a pooling-specific objective).
        "pool_koleo_coef": 0.0,
        # Anti-collapse regularizers on combine's raw salience logits (see
        # salience_diversity_loss/salience_margin_loss) -- koleo_coef alone
        # wasn't sufficient either: the real slurm-4671175 restart (koleo
        # active the whole run) still collapsed, with gate_salience_mean
        # pinned at exactly 1.0 for ~4 epochs then exactly salience_floor
        # (0.1) for the remaining 46, never recovering. koleo operates on
        # L2-normalized pooled state, so a salience gate collapsed to the
        # *same* value for every token/example is invisible to it (a shared
        # scale factor cancels under normalization) while still crushing the
        # actual signal reaching the decoder. These two terms regularize the
        # gate directly instead. Small/secondary, same scale as entropy_coef
        # -- exact values from offline verification, not guessed.
        "salience_diversity_coef": 0.05,
        "salience_margin_coef": 0.01,
    },

    # Split into two per-stage constants (was a single "epochs": 50, bumped
    # from 15 after the slurm-4631350 collapse postmortem -- see
    # train_stage/run_stages) once training itself was split into two
    # independently-stoppable stages (STAGE1_SOURCES/STAGE2_SOURCES): stage 1
    # (logi_glue/trivia_qa/nq_open, short/closed-form) and stage 2
    # (asqa/deepmind_math/minerva_math, "more complicated and longer"
    # generation) each get their own hand-set epoch count rather than sharing
    # one value, since they cover different row counts and different
    # difficulty. Both default to the same 50 the combined run used; not
    # necessarily the same wall-clock/step budget per stage even though the
    # number matches (stage 1's logi_glue is uncapped, unlike every stage-2
    # source). Wall-clock confirmed not a constraint by the user.
    "epochs_stage1": 50,
    "epochs_stage2": 50,
    # Bumped alongside the widened learning_rate range below -- a 3x wider
    # log-uniform range needs proportionally more trials to actually get
    # sampled/explored, not just a wider nominal bound (see search_space's
    # comment for the full "why widen" reasoning).
    "tune_trials": 25,
    # Bumped from 300 -- a short single-process probe is one thing Optuna's
    # tuning can't avoid being (see search_space's comment), but making it
    # longer at least makes each trial's val_loss a slightly more reliable
    # signal to rank candidate learning rates against, rather than a very
    # short window's outcome.
    "tune_budget_batches": 600,

    "defaults": {
        "batch_size": 16,
        "learning_rate": 2e-4,
        "weight_decay": 1e-2,
        "warmup_steps": 200,
        "lr_min_scale": 0.1,             # cosine decay floor (fraction of peak LR)
        "label_smoothing": 0.0,
    },
    "search_space": {
        "batch_size": [8, 16, 32],
        # History: originally (5e-5, 1e-3) -- a run at that range picked
        # 9.49e-4 (right at the old ceiling) and trained cleanly (healthy
        # val_loss, healthy logi_glue reward throughout). Widened once to
        # (1e-5, 3e-3) suspecting the ceiling itself was the problem; that
        # was wrong -- the re-tune picked an even more aggressive LR
        # (~2.72e-3), and a real distributed run on it visibly destabilized
        # partway through training (val_loss spiked from its epoch-1 best
        # and never recovered; logi_glue reward collapsed to 0.0 and stayed
        # there even as val_loss partially recovered afterward). That's
        # direct evidence the higher end of the widened range is actively
        # harmful, not just unexplored -- pulled the ceiling back down to
        # (at/below) the range that's actually been proven to work, rather
        # than leaving the widened one in place. Note this project's
        # sync_gradients() *averages* (all_reduce SUM then /= world_size)
        # rather than summing gradients across ranks, so the usual "scale
        # LR up with batch size" large-batch-training rule of thumb doesn't
        # clearly apply here to begin with -- consistent with going higher
        # having hurt rather than helped. Lower bound left extended (no
        # evidence lower values are harmful, and it costs nothing to leave
        # Optuna room there).
        "learning_rate": (1e-5, 1e-3),      # log-uniform
        "weight_decay": (1e-5, 1e-1),       # log-uniform
        "warmup_steps": [0, 100, 300],
        "label_smoothing": [0.0, 0.1],
    },

    # Per-source download caps. `None` means "use the entire split". There is
    # no logi_glue cap: every entry from every split of its four sub-datasets
    # is always pulled (see load_logi_glue).
    "data": {
        "minerva_cap": None,
        "deepmind_easy_cap": 15_000,
        "deepmind_medium_cap": 10_000,
        "trivia_qa_cap": None,
        "nq_open_cap": None,
        "asqa_cap": None,
    },
}

STAGES = ["tune", "train"]


# --------------------------------------------------------------------------- #
# Dataset identifiers
# --------------------------------------------------------------------------- #
MINERVA_HF = "math-ai/minervamath"

DEEPMIND_MATH_HF = "math_dataset"
DEEPMIND_EASY_CONFIGS = [
    "arithmetic__add_or_sub", "arithmetic__add_sub_multiple", "arithmetic__mul",
    "arithmetic__div", "numbers__place_value", "numbers__round_number",
    "comparison__closest", "comparison__pair", "algebra__linear_1d",
]
DEEPMIND_MEDIUM_CONFIGS = [
    "algebra__linear_2d", "algebra__polynomial_roots", "polynomials__evaluate",
    "polynomials__expand", "calculus__differentiate", "numbers__gcd",
    "numbers__lcm", "comparison__sort",
]

LOGI_GLUE_HF = "logicreasoning/logi_glue"
# Exact HF subset (config) names, confirmed against the dataset viewer.
LOGI_GLUE_CONFIGS = ["proofwriter", "logiQA", "logicNLI", "babi_task_16"]

TRIVIAQA_HF = "mandarjoshi/trivia_qa"
TRIVIAQA_CONFIG = "rc.nocontext"

NQ_OPEN_HF = "google-research-datasets/nq_open"
ASQA_HF = "din0s/asqa"


# --------------------------------------------------------------------------- #
# Weights & Biases (optional; enabled when an API key is provided)
# --------------------------------------------------------------------------- #
_WANDB = {"enabled": False, "run": None}


def init_wandb(api_key, stage):
    if not api_key or api_key == "_":
        return
    try:
        import wandb

        wandb.login(key=api_key)
        _WANDB["run"] = wandb.init(
            project="text_to_text", name=stage, reinit=True, config=HYPERPARAMS,
        )
        _WANDB["enabled"] = True
        print(f"[wandb] logging enabled for stage '{stage}'")
    except Exception as e:
        print(f"[wandb] disabled ({e})")


def wandb_log(payload):
    if _WANDB["enabled"]:
        try:
            import wandb
            wandb.log(payload)
        except Exception:
            pass


def finish_wandb():
    if _WANDB["enabled"]:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
    _WANDB["enabled"] = False
    _WANDB["run"] = None


# --------------------------------------------------------------------------- #
# Stage-progress state machine
# --------------------------------------------------------------------------- #
def load_progress():
    if os.path.isfile(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"completed": []}


def save_progress(progress):
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def mark_stage_done(stage):
    progress = load_progress()
    if stage not in progress["completed"]:
        progress["completed"].append(stage)
    save_progress(progress)


# --------------------------------------------------------------------------- #
# Greppable weight naming + lookup (shared weights/ directory with enc_pre.py)
#
# Filename scheme: sectionLabel_epoch_epochNum_batch_batchNum[_valLoss...]
#   sectionLabel = {component}_{tag}
#   at the end of a fully-finished epoch, "batch_batchNum" is replaced
#   outright by the single word "complete" (not "batch_complete") -- see
#   _is_epoch_complete, which specifically looks for that literal marker so
#   resume logic never mistakes "some batch inside this epoch got
#   checkpointed" for "this epoch is actually finished".
# --------------------------------------------------------------------------- #
def weight_path(component, tag, epoch, batch, val_loss=None, epoch_complete=False):
    name = f"{component}_{tag}_epoch_{epoch:04d}_"
    name += "complete" if epoch_complete else f"batch_{batch:07d}"
    if val_loss is not None:
        name += f"_valLoss{val_loss:.4f}"
    return os.path.join(WEIGHTS_DIR, name + ".pt")


_EPOCH_RE = re.compile(r"_epoch_(\d+)_")
_BATCH_RE = re.compile(r"_batch_(\d+)")
_COMPLETE_RE = re.compile(r"_epoch_\d+_complete(?:_|\.pt$)")
# \d+\.\d+ (not a greedy [0-9.]+): a greedy character class would also match
# the literal "." that starts the ".pt" extension (e.g. "2.4034." from
# "..._valLoss2.4034.pt"), which float() can't parse -- this crashed every
# completed training run the moment a 'best' checkpoint existed, since
# find_weights(component, "best") calls _valloss on every candidate file via
# min(..., key=_valloss). See enc_pre.py for the traceback that surfaced this.
_VALLOSS_RE = re.compile(r"valLoss(\d+\.\d+)")


def _is_epoch_complete(path):
    """True only if this filename carries the literal 'complete' marker --
    never inferred from a batch number happening to equal an epoch's total
    batch count (see weight_path / train's resume logic)."""
    return bool(_COMPLETE_RE.search(os.path.basename(path)))


def _epoch_batch(path):
    name = os.path.basename(path)
    m_epoch = _EPOCH_RE.search(name)
    epoch = int(m_epoch.group(1)) if m_epoch else -1
    if _is_epoch_complete(path):
        # Sorts after every in-epoch batch checkpoint for the same epoch.
        return (epoch, float("inf"))
    m_batch = _BATCH_RE.search(name)
    batch = int(m_batch.group(1)) if m_batch else -1
    return (epoch, batch)


def _valloss(path):
    m = _VALLOSS_RE.search(os.path.basename(path))
    return float(m.group(1)) if m else float("inf")


def find_weights(component, tag):
    """Return the path to the requested weights, or None. Any tag other than
    "best" picks the highest (epoch, batch) -- a fully-finished epoch's
    'complete' checkpoint sorts after every in-epoch batch checkpoint for
    that epoch; "best" picks the lowest embedded valLoss."""
    files = glob.glob(os.path.join(WEIGHTS_DIR, f"{component}_{tag}_epoch_*.pt"))
    if not files:
        return None
    key = _valloss if tag == "best" else _epoch_batch
    return min(files, key=key) if tag == "best" else max(files, key=key)


def _is_schema_mismatch(e):
    """True if ``e`` is load_state_dict's own strict-mode key-mismatch error
    (an architecture change added/removed a parameter -- e.g.
    LinearAttentionBlock's residual_scale, added for the encoder-collapse
    fix) rather than genuine file corruption (a bad pickle, truncated
    write). _load_weights_with_fallback/_warm_start both otherwise delete
    whatever they fail to load, on the assumption that any load failure
    means a corrupt file -- true for a truncated GPFS write, but not for a
    schema mismatch, which is a normal, recoverable event on every future
    architecture change to a model class with a persisted checkpoint.
    Conflating the two would silently delete real, expensive-to-reproduce
    trained weights just because their schema predates a code change, not
    because anything is actually wrong with the file."""
    msg = str(e)
    return "Missing key(s) in state_dict" in msg or "Unexpected key(s) in state_dict" in msg


def _load_weights_tolerant(module, path, device):
    """Shared by _load_weights_with_fallback/_warm_start: a strict load
    first (catches genuine corruption at its strictest), and on a pure
    schema mismatch (see _is_schema_mismatch) a retry with strict=False --
    keeping every matching weight and leaving new parameters at their
    fresh-init value -- instead of treating a legitimate architecture
    change as corruption worth deleting the file over."""
    try:
        return module.load_weights(path, map_location=device)
    except RuntimeError as e:
        if not _is_schema_mismatch(e):
            raise
        print(f"[t2t] {os.path.basename(path)}'s schema doesn't match the "
              f"current architecture ({e}); retrying with strict=False "
              f"instead of treating this as corruption")
        return module.load_weights(path, map_location=device, strict=False)


def _load_weights_with_fallback(module, component, device):
    """Try 'latest' then 'best' for this component, loading into ``module``
    via its own ``load_weights`` (see _load_weights_tolerant for the
    schema-mismatch-vs-corruption distinction). Corrupt files are deleted as
    they're found rather than crashing the run outright -- specifically
    matters for resuming a mid-run checkpoint, where a corrupted 'latest'
    file would otherwise crash every subsequent resubmission the same way
    instead of falling back to the last good ('best') snapshot. Returns
    (path, load_weights' return value), or (None, None) if neither tag
    loads -- for ExpertRegistry that return value is the embedded ``extra``
    training-state payload (see ExpertRegistry.store_weights), None for
    everything else."""
    for tag in ("latest", "best"):
        path = find_weights(component, tag)
        if not path:
            continue
        try:
            return path, _load_weights_tolerant(module, path, device)
        except Exception as e:
            print(f"[resume] {os.path.basename(path)} is unreadable ({e}); "
                  f"deleting it and trying an older checkpoint")
            try:
                os.remove(path)
            except OSError:
                pass
    return None, None


def _prune_other_tagged_files(component, tag, keep_path):
    """Delete every other file previously saved under this exact
    (component, tag) pair, now that ``keep_path`` has landed successfully --
    called only *after* a new file is written, so a failed write never
    leaves zero valid checkpoints behind.

    Refuses to delete a file representing *more* progress (higher
    (epoch, batch)) than keep_path. Without this, a run whose resume missed
    an existing, further-along checkpoint -- a stale filesystem view on a
    different node, or two jobs for the same stage accidentally running at
    once -- would have its very first save irreversibly destroy the
    more-advanced checkpoint it should have resumed from instead. Loudly
    warns and keeps both files when that happens rather than silently
    losing the better one."""
    keep_progress = _epoch_batch(keep_path)
    for f in glob.glob(os.path.join(WEIGHTS_DIR, f"{component}_{tag}_epoch_*.pt")):
        if f == keep_path:
            continue
        if _epoch_batch(f) > keep_progress:
            print(f"[checkpoint] WARNING: not deleting {os.path.basename(f)} -- it "
                  f"represents MORE progress than the checkpoint just written "
                  f"({os.path.basename(keep_path)}). This usually means resume "
                  f"missed it (stale filesystem view, or two jobs running the "
                  f"same stage at once) -- investigate before continuing; both "
                  f"files are being kept for now.")
            continue
        try:
            os.remove(f)
        except OSError:
            pass


def _strip_training_state(component):
    """Re-save the 'latest' (and 'best', if present) tagged file for this
    component with its embedded training-state payload removed (see
    save_checkpoint / ExpertRegistry.store_weights' ``extra`` param) --
    training state (optimizer momentum/variance buffers, epoch/batch/step
    counters) is only ever needed to resume a run that's still in progress,
    and can be sizable (AdamW's buffers are roughly 2x the trainable-
    parameter size), so it's worth dropping once a stage genuinely
    finishes. Unlike the old '_wrapper' scheme this doesn't delete the
    file itself -- the weights it carries are the permanent, reusable
    artifact now, not a throwaway."""
    for tag in ("latest", "best"):
        f = find_weights(component, tag)
        if not f:
            continue
        checkpoint = torch.load(f, map_location="cpu")
        if "extra" not in checkpoint:
            continue
        del checkpoint["extra"]
        _atomic_torch_save(checkpoint, f)


def _atomic_torch_save(obj, path, retries=3, backoff_seconds=5.0):
    """torch.save is not atomic and not resilient to transient filesystem
    hiccups -- a write failure partway through (seen in practice as
    PytorchStreamWriter "file write failed" on a flaky GPFS mount) can leave
    the process stuck rather than cleanly crashing, hanging the whole job
    until the SLURM time limit kills it (see enc_pre.py for the traceback
    that surfaced this). Writing to a temp file and renaming into place
    means a failed attempt never produces a half-written checkpoint at
    ``path``, and retrying with a short backoff rides out a transient
    glitch instead of taking the whole run down with it."""
    tmp_path = f"{path}.tmp{os.getpid()}"
    last_err = None
    for attempt in range(retries):
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
            return
        except (RuntimeError, OSError) as e:
            last_err = e
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if attempt < retries - 1:
                print(f"[checkpoint] write to {path} failed ({e}); "
                      f"retrying ({attempt + 1}/{retries})...")
                time.sleep(backoff_seconds)
    raise last_err


def save_checkpoint(model, optimizer, epoch, batch, global_step, best_val_loss,
                    val_loss=None, epoch_complete=False, rl_baselines=None,
                    checkpoint_prefix="t2t"):
    """Persist ``combine``/``router``/``expert_registry``/``text_output`` as
    four standalone, directly-loadable files (each via that module's own
    store_weights -- the same pattern text_encoder/text_output already use)
    under the 'latest' tag, always, and also under 'best' when val_loss
    improves on best_val_loss. There is no separate resumable-wrapper file:
    the optimizer state and epoch/batch/step counters are stored as an
    ``extra`` payload embedded in ``{checkpoint_prefix}_expert_registry``'s
    *'latest' tag only* (see ExpertRegistry.store_weights) -- 'best' always stays a
    clean, minimal weights-only file, since another script warm-starting
    from it never wants stale optimizer state, while 'latest' doubles as
    this run's own resume point. Only ever one 'latest' and one 'best' file
    exist per component at a time -- the previous file for that exact tag
    is deleted right after the new one lands successfully (see
    _prune_other_tagged_files) -- rather than keeping a growing 'step'
    history, so disk usage stays bounded by a small constant regardless of
    run length. ``epoch_complete=True`` marks this as the checkpoint
    written once ``epoch`` has genuinely finished; only that literal marker
    -- never a batch number alone -- makes train's resume logic treat an
    epoch as done.

    A save that still fails after atomic_torch_save's own retries (a
    sustained GPFS write outage, not a transient blip) is logged and
    swallowed rather than raised: this is a rank-0-only periodic save, and
    letting it propagate kills the entire distributed job -- every other
    rank loses its next collective op and dies with it -- over what's
    recoverable by just trying again at the next checkpoint interval. The
    previous 'latest'/'best' files on disk are untouched by a failed
    attempt (each component's own store_weights only replaces its target
    file after a fully successful write)."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    training_state = {
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "batch": batch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "rng": torch.get_rng_state(),
        # Scheduled RL term's per-source moving-average baselines (see
        # t2t_rl_term / HYPERPARAMS["reward"]'s schedule comment) -- one
        # independent baseline per task source (not a single shared
        # scalar -- see the rl_ceiling/rl_baseline_momentum comments in
        # HYPERPARAMS for why), resumed so a resubmitted job doesn't lose
        # its variance-reduction warm-up.
        "rl_baselines": rl_baselines or {},
    }
    text_output = model.expert_registry.get_expert("text_output")

    tags = ["latest"]
    if val_loss is not None and val_loss <= best_val_loss:
        tags.append("best")

    combine_c = f"{checkpoint_prefix}_combine"
    router_c = f"{checkpoint_prefix}_router"
    registry_c = f"{checkpoint_prefix}_expert_registry"
    text_output_c = f"{checkpoint_prefix}_text_output"

    try:
        for tag in tags:
            cp = weight_path(combine_c, tag, epoch, batch, val_loss, epoch_complete=epoch_complete)
            rp = weight_path(router_c, tag, epoch, batch, val_loss, epoch_complete=epoch_complete)
            ep = weight_path(registry_c, tag, epoch, batch, val_loss, epoch_complete=epoch_complete)
            tp = weight_path(text_output_c, tag, epoch, batch, val_loss, epoch_complete=epoch_complete)

            model.combine.store_weights(WEIGHTS_DIR, os.path.basename(cp))
            model.router.store_weights(WEIGHTS_DIR, os.path.basename(rp))
            model.expert_registry.store_weights(
                WEIGHTS_DIR, os.path.basename(ep),
                extra=training_state if tag == "latest" else None,
            )
            text_output.store_weights(WEIGHTS_DIR, os.path.basename(tp))

            _prune_other_tagged_files(combine_c, tag, cp)
            _prune_other_tagged_files(router_c, tag, rp)
            _prune_other_tagged_files(registry_c, tag, ep)
            _prune_other_tagged_files(text_output_c, tag, tp)
    except (RuntimeError, OSError) as e:
        print(f"[checkpoint] save_checkpoint failed at epoch {epoch} batch {batch} "
              f"step {global_step} even after retries ({e}); skipping this "
              f"checkpoint and continuing training -- will try again at the "
              f"next interval.")


# --------------------------------------------------------------------------- #
# Incremental LR decay (shared design with enc_pre.py): linear warmup, then
# cosine decay down to a floor fraction of the peak LR, so later optimizer
# steps make progressively smaller changes as training converges.
# --------------------------------------------------------------------------- #
def make_lr_lambda(warmup_steps, total_steps, min_scale=0.1):
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if total_steps <= warmup_steps:
            return 1.0
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_scale + (1.0 - min_scale) * cosine
    return lr_lambda


# --------------------------------------------------------------------------- #
# Answer normalisation, exact match, ROUGE-L (self-contained, no extra deps)
# --------------------------------------------------------------------------- #
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction, gold_answers):
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(a) for a in gold_answers if a)


def _lcs_len(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] \
                else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def _rouge_l_prf(pred_text, ref_text):
    """Precision/recall/F1 over the longest common (sub)sequence of
    whitespace tokens -- rouge_l_f1 (below) already computed precision/
    recall internally before this was split out; exposed separately so
    evaluate_rewards' precision/recall diagnostic (see
    HYPERPARAMS["reward"]'s postmortem on the loss composite possibly
    overwhelming the model) can report them without a second LCS pass."""
    p_toks, r_toks = pred_text.split(), ref_text.split()
    if not p_toks or not r_toks:
        return 0.0, 0.0, 0.0
    lcs = _lcs_len(p_toks, r_toks)
    if lcs == 0:
        return 0.0, 0.0, 0.0
    prec, rec = lcs / len(p_toks), lcs / len(r_toks)
    return prec, rec, 2 * prec * rec / (prec + rec)


def rouge_l_f1(pred_text, ref_text):
    return _rouge_l_prf(pred_text, ref_text)[2]


def compute_reward_metric(source, pred_text, target_text, aux):
    """Task-specific reward/metric, dispatched off the CSV 'source' column.
    Itself non-differentiable (plain string/number comparisons), but IS
    used as a training signal: t2t_rl_term's REINFORCE term scores a
    sampled generation with it directly -- the reward value itself carries
    no gradient (it's a python float), only the sampled generation's
    log-probability does, per the standard REINFORCE/score-function
    estimator (see reasoning_to_text.py's identical reasoning_rl_term for
    the same pattern). Also used for pure logging/validation
    (evaluate_rewards).

    Every non-ASQA source used to be flat exact-match-or-nothing (1.0/0.0)
    -- graduated to exact-match-first, ROUGE-L partial-credit fallback
    (same pattern reasoning_to_text.py already uses for numinamath), since
    a REINFORCE term driven by an almost-always-0 reward is exactly the
    classic RL sparse-reward problem, on top of CE's own already-harsh
    penalty for hard-to-know tokens."""
    if source == "asqa":
        short_answers = [a for a in (aux or "").split("|") if a]
        r = rouge_l_f1(pred_text, target_text)
        contains = 1.0 if any(
            normalize_answer(a) in normalize_answer(pred_text) for a in short_answers
        ) else 0.0
        return 0.5 * r + 0.5 * contains
    aliases = [a for a in (aux or "").split("|") if a]
    gold = aliases if aliases else [target_text]
    if exact_match(pred_text, gold):
        return 1.0
    return max((rouge_l_f1(pred_text, g) for g in gold), default=0.0)


# --------------------------------------------------------------------------- #
# Target tokenisation (reuse the project's shared SentencePiece model)
# --------------------------------------------------------------------------- #
class SpmCodec:
    def __init__(self, model_path=TOKENIZER_PATH, max_len=32):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.max_len = max_len

    def encode_target(self, text, device, max_len=None):
        cap = max_len if max_len is not None else self.max_len
        ids = self.sp.encode_as_ids(text)[: cap - 2]
        ids = [BOS_ID] + ids + [EOS_ID]
        return torch.tensor([ids], dtype=torch.long, device=device)


# --------------------------------------------------------------------------- #
# CSV row helpers
# --------------------------------------------------------------------------- #
def _truncate_words(text, max_words):
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def _mkrow(inp, tgt, source, aux=""):
    inp = _truncate_words(str(inp).strip(), HYPERPARAMS["global"]["max_input_len"])
    tgt = str(tgt).strip()
    if not inp or not tgt:
        return None
    return {"input": inp, "target": tgt, "source": source, "aux": aux}


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample_rows(rows, n, seed):
    rng = random.Random(seed)
    if n >= len(rows):
        return list(rows)
    return rng.sample(rows, n)


# --------------------------------------------------------------------------- #
# Per-source downloaders. Each raises on failure; the caller in
# compile_dataset() catches and skips (data loads are defensive, matching
# enc_pre.py's philosophy: a source that cannot be fetched is skipped with a
# warning so the rest of the compile proceeds). Dataset schemas on Hugging
# Face vary and drift over time, so field lookups are best-effort with
# fallbacks, same spirit as enc_pre.py's VRDTask._extract_relationship.
# --------------------------------------------------------------------------- #
def load_minerva_math(cap):
    from datasets import load_dataset

    rows = []
    for split in ("test", "train", "validation"):
        try:
            ds = load_dataset(MINERVA_HF, split=split, cache_dir=HF_CACHE)
        except Exception:
            continue
        for r in ds:
            q = r.get("problem") or r.get("question")
            a = r.get("answer") or r.get("solution")
            if q and a:
                row = _mkrow(q, a, "minerva_math")
                if row:
                    rows.append(row)
            if cap is not None and len(rows) >= cap:
                break
        if cap is not None and len(rows) >= cap:
            break
    if not rows:
        raise RuntimeError("minerva_math yielded no usable rows")
    return rows


def _load_deepmind_configs(configs, cap, source_tag):
    from datasets import load_dataset

    rows = []
    per_cfg = max(1, cap // max(1, len(configs)))
    for cfg in configs:
        if len(rows) >= cap:
            break
        try:
            ds = load_dataset(DEEPMIND_MATH_HF, cfg, split="train", streaming=True,
                              cache_dir=HF_CACHE, trust_remote_code=True)
        except Exception as e:
            print(f"[data] SKIP deepmind_math config '{cfg}' ({e})")
            continue
        n = 0
        for r in ds:
            q = r.get("question") or r.get("problem")
            a = r.get("answer") or r.get("solution")
            if q and a:
                row = _mkrow(q, a, source_tag)
                if row:
                    rows.append(row)
                    n += 1
            if n >= per_cfg or len(rows) >= cap:
                break
    return rows


def load_deepmind_math(easy_cap, medium_cap):
    # Questions are generated on the fly by the upstream module (no static
    # dataset), starting with mechanically-simple configs, then a smaller
    # slice of harder ones -- see module docstring / DEEPMIND_*_CONFIGS.
    rows = _load_deepmind_configs(DEEPMIND_EASY_CONFIGS, easy_cap, "deepmind_math")
    rows += _load_deepmind_configs(DEEPMIND_MEDIUM_CONFIGS, medium_cap, "deepmind_math")
    if not rows:
        raise RuntimeError("deepmind_math yielded no usable rows")
    return rows


def load_logi_glue():
    """Pull every entry from every available split of the four Logi-Glue
    sub-datasets the project uses (ProofWriter, LogiQA, LogicNLI, bAbI). No
    cap and no held-out split here -- the compile+split pipeline
    (compile_dataset/split_dataset) re-partitions the whole merged pool into
    train.csv/val.csv downstream, same as every other source, so nothing is
    held back at this stage.

    Real schema (verified against the dataset viewer -- an earlier version of
    this loader guessed a generic {premise, hypothesis, label} shape and was
    wrong): each row has {context, question, choices, answer_choice,
    answer_text, proof, original_dataset, input, ...}; ``input`` is already a
    ready-made pre-formatted prompt, used directly when present. Not every
    config has both splits -- ``babi_task_16`` only ships a ``test`` split --
    so splits are discovered per config rather than assumed."""
    from datasets import load_dataset, get_dataset_split_names

    rows = []
    for cfg in LOGI_GLUE_CONFIGS:
        try:
            splits = get_dataset_split_names(LOGI_GLUE_HF, cfg)
        except Exception as e:
            print(f"[data] SKIP logi_glue config '{cfg}' ({e})")
            continue
        for split in splits:
            try:
                ds = load_dataset(LOGI_GLUE_HF, cfg, split=split, cache_dir=HF_CACHE)
            except Exception as e:
                print(f"[data] SKIP logi_glue config '{cfg}' split '{split}' ({e})")
                continue
            n = 0
            for r in ds:
                answer = r.get("answer_text")
                inp = r.get("input") or (
                    f"{r.get('context', '')}\nQuestion: {r.get('question', '')}"
                )
                if not answer or not inp.strip():
                    continue
                row = _mkrow(inp, answer, "logi_glue")
                if row:
                    rows.append(row)
                    n += 1
            print(f"[data]   logi_glue/{cfg}[{split}]: {n} rows")
    if not rows:
        raise RuntimeError("logi_glue yielded no usable rows")
    return rows


def load_trivia_qa(cap):
    from datasets import load_dataset

    ds = load_dataset(TRIVIAQA_HF, TRIVIAQA_CONFIG, split="train", streaming=True,
                      cache_dir=HF_CACHE)
    rows = []
    for r in ds:
        answer = r.get("answer") or {}
        value = answer.get("value") or ""
        if not value:
            continue
        aliases = list(dict.fromkeys(
            (answer.get("aliases") or []) + (answer.get("normalized_aliases") or [])
            + [value]
        ))
        row = _mkrow(r.get("question", ""), value, "trivia_qa", aux="|".join(aliases))
        if row:
            rows.append(row)
        if cap is not None and len(rows) >= cap:
            break
    if not rows:
        raise RuntimeError("trivia_qa yielded no usable rows")
    return rows


def load_nq_open(cap):
    from datasets import load_dataset

    ds = load_dataset(NQ_OPEN_HF, split="train", streaming=True, cache_dir=HF_CACHE)
    rows = []
    for r in ds:
        answers = r.get("answer") or []
        if not answers:
            continue
        row = _mkrow(r.get("question", ""), answers[0], "nq_open",
                     aux="|".join(answers))
        if row:
            rows.append(row)
        if cap is not None and len(rows) >= cap:
            break
    if not rows:
        raise RuntimeError("nq_open yielded no usable rows")
    return rows


def load_asqa(cap):
    from datasets import load_dataset

    ds = load_dataset(ASQA_HF, split="train", cache_dir=HF_CACHE)
    rows = []
    for r in ds:
        question = r.get("ambiguous_question") or r.get("question")
        annotations = r.get("annotations") or []
        long_answer, short_answers = None, []
        for ann in annotations:
            long_answer = long_answer or ann.get("long_answer")
            for qa in ann.get("qa_pairs") or []:
                short_answers.extend(qa.get("short_answers") or [])
        if not question or not long_answer:
            continue
        row = _mkrow(question, long_answer, "asqa",
                     aux="|".join(dict.fromkeys(short_answers)))
        if row:
            rows.append(row)
        if cap is not None and len(rows) >= cap:
            break
    if not rows:
        raise RuntimeError("asqa yielded no usable rows")
    return rows


SOURCE_LOADERS = {
    "minerva_math": lambda cfg: load_minerva_math(cfg["data"]["minerva_cap"]),
    "deepmind_math": lambda cfg: load_deepmind_math(
        cfg["data"]["deepmind_easy_cap"], cfg["data"]["deepmind_medium_cap"]),
    "logi_glue": lambda cfg: load_logi_glue(),
    "trivia_qa": lambda cfg: load_trivia_qa(cfg["data"]["trivia_qa_cap"]),
    "nq_open": lambda cfg: load_nq_open(cfg["data"]["nq_open_cap"]),
    "asqa": lambda cfg: load_asqa(cfg["data"]["asqa_cap"]),
}

# Two independently-stoppable training stages (see train_stage/run_stages):
# stage 1 covers the short/closed-form answer sources, stage 2 the harder,
# longer-generation ones -- the same split considered (and deferred) as a
# curriculum-staging option during the collapse-fix work, reused here purely
# for deployment/checkpoint-boundary reasons, not to address that collapse.
STAGE1_SOURCES = {"logi_glue", "trivia_qa", "nq_open"}
STAGE2_SOURCES = {"asqa", "deepmind_math", "minerva_math"}


# --------------------------------------------------------------------------- #
# Compile (download + normalise + shuffle) and split (train/val, stratified)
# --------------------------------------------------------------------------- #
def compile_dataset():
    if os.path.isfile(COMPILED_CSV):
        print(f"[data] compiled dataset already present at {COMPILED_CSV}")
        return
    os.makedirs(DATA_DIR, exist_ok=True)

    all_rows = []
    for name, loader in SOURCE_LOADERS.items():
        try:
            rows = loader(HYPERPARAMS)
            print(f"[data] {name}: {len(rows)} rows")
            all_rows.extend(rows)
        except Exception as e:
            print(f"[data] SKIP source '{name}' — could not prepare ({e})")

    if not all_rows:
        raise RuntimeError("No text_to_text data sources could be prepared.")

    # Inbuilt shuffling of the entire compiled dataset (interleaves sources).
    random.Random(HYPERPARAMS["global"]["seed"]).shuffle(all_rows)

    _write_csv(COMPILED_CSV, all_rows)
    print(f"[data] wrote {len(all_rows)} compiled rows -> {COMPILED_CSV}")


def split_dataset():
    if os.path.isfile(TRAIN_CSV) and os.path.isfile(VAL_CSV):
        print("[data] train/val split already present")
        return

    rows = load_csv_rows(COMPILED_CSV)
    by_source = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    rng = random.Random(HYPERPARAMS["global"]["seed"] + 1)
    val_frac = HYPERPARAMS["global"]["val_fraction"]
    train_rows, val_rows = [], []
    print("[data] per-source train/val split:")
    for source, srows in sorted(by_source.items()):
        rng.shuffle(srows)
        n_val = max(1, int(len(srows) * val_frac))
        val_rows.extend(srows[:n_val])
        train_rows.extend(srows[n_val:])
        print(f"[data]   {source}: {len(srows) - n_val} train / {n_val} val "
              f"({len(srows)} total)")

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    _write_csv(TRAIN_CSV, train_rows)
    _write_csv(VAL_CSV, val_rows)
    print(f"[data] split -> train={len(train_rows)} val={len(val_rows)}")


def ensure_dataset_ready():
    compile_dataset()
    split_dataset()


# --------------------------------------------------------------------------- #
# Frozen text encoder (loaded from enc_pre.py's best checkpoint)
# --------------------------------------------------------------------------- #
class FrozenTextEncoderWrapper(nn.Module):
    """Wraps the frozen text encoder. Runs its forward pass under fp16
    autocast on CUDA (instead of a literal .half() conversion -- see module
    docstring) when half precision is enabled, and always stays in eval mode
    regardless of the outer model's train()/eval() state."""

    def __init__(self, encoder, half_precision):
        super().__init__()
        self.encoder = encoder
        self.half_precision = half_precision

    def forward(self, x):
        device = next(self.encoder.parameters()).device
        if self.half_precision and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.encoder(x)
            return out.float()
        return self.encoder(x)

    def forward_tokens(self, *args, **kwargs):
        return self.encoder.forward_tokens(*args, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()          # frozen encoder never leaves eval mode
        return self


def find_frozen_encoder_weights():
    return find_weights("text_encoder", "best") or find_weights("text_encoder", "latest")


def build_model(device, d_model, half_precision, checkpoint_prefix="t2t", fallback_prefix=None):
    from models.OverallModel import MultimodalModel
    from models.Encoders.TextEncoder import createTextEnc

    model = MultimodalModel(dim=d_model, text_only=True).to(device)

    raw_encoder = createTextEnc(dim=d_model).to(device)
    best = find_frozen_encoder_weights()
    if best:
        raw_encoder.load_weights(best, map_location=device)
        print(f"[t2t] frozen text encoder loaded from {os.path.basename(best)}")
    else:
        print("[t2t] WARNING: no trained text encoder found via optimal grep in "
              f"{WEIGHTS_DIR}; using a randomly initialised (untrained) text encoder")
    raw_encoder.eval()
    for p in raw_encoder.parameters():
        p.requires_grad_(False)

    frozen = FrozenTextEncoderWrapper(raw_encoder, half_precision).to(device)
    model.encoder_registry.remove_encoder("text")
    model.encoder_registry.add_encoder("text", frozen)
    model.expert_registry.set_pipeline(
        text_encoder=frozen, combiner=model.combine, router=model.router,
    )

    # ---- Warm-start combine/router/expert_registry from this stage's own --- #
    # ---- prior progress, if any -- falling back to another stage's finished  #
    # ---- weights only when this stage has never started at all ------------ #
    # "best" first, "latest" as a fallback (e.g. early in a run before any
    # val_loss improvement has landed yet, only "latest" exists) -- this is
    # the generic "give me the best available starting point" lookup used
    # everywhere else in this project (frozen encoders, text_output in the
    # other *_to_text.py scripts). A run resuming its own exact mid-epoch
    # progress needs "latest" specifically and overrides this in train_stage()
    # (see train_stage()'s resume logic) -- build_model() itself doesn't need
    # to know or care whether it's being called for a fresh stage, a resume,
    # or another script reusing these weights entirely.
    #
    # checkpoint_prefix/fallback_prefix (see STAGE1_SOURCES/STAGE2_SOURCES):
    # tries checkpoint_prefix's own files first, then fallback_prefix's if
    # given and nothing was found under checkpoint_prefix -- covers both
    # "this stage hasn't started yet, inherit the prior stage's finished
    # weights" (fallback_prefix hits) and "this stage has its own progress to
    # resume" (checkpoint_prefix hits first, fallback_prefix never touched)
    # with the same lookup, no separate branching needed. Stage 1 passes
    # fallback_prefix=None (starts from scratch, nothing to inherit).
    def _warm_start(module, component_suffix, label):
        prefixes = (checkpoint_prefix, fallback_prefix) if fallback_prefix else (checkpoint_prefix,)
        for prefix in prefixes:
            # Same corrupt-file tolerance as _load_weights_with_fallback (a
            # truncated write from a flaky GPFS save shouldn't crash every
            # subsequent run that warm-starts from it) -- order is "best"
            # then "latest" here rather than _load_weights_with_fallback's
            # "latest"-then-"best", since build_model() wants the generic
            # best available starting point, not this exact run's precise
            # progress.
            for tag in ("best", "latest"):
                path = find_weights(f"{prefix}_{component_suffix}", tag)
                if not path:
                    continue
                try:
                    _load_weights_tolerant(module, path, device)
                    print(f"[t2t] warm-started {label} from {os.path.basename(path)}")
                    return
                except Exception as e:
                    print(f"[t2t] {os.path.basename(path)} is unreadable ({e}); "
                          f"deleting it and trying an older checkpoint")
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        print(f"[t2t] no {label} checkpoint found; using a fresh (randomly initialised) one")

    _warm_start(model.combine, "combine", "combine")
    _warm_start(model.router, "router", "router")
    _warm_start(model.expert_registry, "expert_registry", "expert_registry")

    # Belt-and-suspenders: add_encoder() now matches the registered module's
    # own device (see EncRegistry.py), but re-syncing the whole model here
    # too costs nothing and catches this whole bug class regardless of how
    # a future encoder swap is implemented.
    model = model.to(device)
    return model


# --------------------------------------------------------------------------- #
# Core forward helpers (mirrors dim_tune.py's encode_to_state / teacher
# forcing, extended to also surface the ACT ponder cost)
# --------------------------------------------------------------------------- #
def encode_to_state(model, text, return_gate_logits=False, return_encoder_memory=False):
    """``return_gate_logits=True`` additionally returns combine's raw (pre-
    sigmoid, pre-floor) salience logits -- see AdaptiveAttentionPooling's
    return_gate_stats and salience_diversity_loss/salience_margin_loss below.

    ``return_encoder_memory=True`` additionally returns combine.encoder's own
    pre-pooling output AND combine.pool's own output (see
    encoder_koleo_loss/pool_koleo_loss below) -- added once a real harness
    run (slurm-4800583) showed state_sim_within/across and
    unique_pred_fraction repeatedly relapsing into collapse well after the
    ReZero architecture fix (EncCombine.py's LinearAttentionBlock/
    HybridLinearEncoderLayer/AdaptiveAttentionPooling, Logic.py's
    RoutingLogic) landed, with the newly-added gate values themselves
    staying small throughout (so "the gates reopened" wasn't the mechanism
    -- see gate_scale_summary) -- nothing in this file had ever directly
    regularized combine.encoder's/combine.pool's own output; every existing
    anti-collapse term (koleo_loss, decoder_diversity_loss, salience terms)
    acts several layers downstream of them instead, which is a plausible
    reason the underlying (still-trainable) layer weights behind those
    gates could drift toward a collapse-inducing direction with nothing to
    stop them. ``combined`` (combine.pool's output) was already computed by
    every call path below regardless of this flag -- this just also
    returns it, no extra forward pass needed.

    Default False on both keeps every existing call site's plain-tensor
    return untouched; only train_stage's main loop / batch-boundary block
    opt in, since only they need a live gradient path."""
    encodings = model.encode(text, "text")
    if return_gate_logits:
        combined, _, _, gate_stats = model.combine(encodings, return_gate_stats=True)
        _, state = model.router(combined)
        return state, gate_stats["raw_logits"]
    if return_encoder_memory:
        combined, _, _, memory = model.combine(encodings, return_memory=True)
        _, state = model.router(combined)
        return state, memory, combined
    combined, _, _ = model.combine(encodings)
    _, state = model.router(combined)
    return state


def koleo_loss(embeddings, chunk_size=16, eps=1e-8):
    """Sablayrolles et al. 2018: encourages embeddings within a (small) batch to
    spread out uniformly via -log(distance to nearest neighbour). Computed in
    chunks of ``chunk_size`` (DINOv3's "small batches of 16"). Identical to
    enc_pre.py's koleo_loss (already validated there for the vision SSL
    pretraining pipeline) -- duplicated here rather than imported, matching this
    codebase's convention of each *_to_text.py script staying self-contained
    and independently launchable.

    Root-cause fix for the "always True" free-run collapse (see
    HYPERPARAMS["reward"]["koleo_coef"]'s comment and train_stage's call site):
    applied directly to a batch of `state`'s pooled representations, this is
    what actually stopped state's bulk direction from collapsing toward
    near-identical across every input during real training -- something none
    of nine different decoder-side interventions (masking, self-attention
    blocking, cosine attention, boosted cross-attention LR, direct injection)
    managed to do, since none of them addressed the collapse at its source."""
    x = F.normalize(embeddings, dim=-1, eps=eps)
    n = x.size(0)
    if n < 2:
        return torch.zeros((), device=x.device)
    losses = []
    for i in range(0, n, chunk_size):
        chunk = x[i:i + chunk_size]
        if chunk.size(0) < 2:
            continue
        dots = chunk @ chunk.t()
        dots.fill_diagonal_(-2.0)
        nn_idx = dots.argmax(dim=1)
        dist = (chunk - chunk[nn_idx]).pow(2).sum(dim=-1).add(eps).sqrt()
        losses.append(-torch.log(dist).mean())
    if not losses:
        return torch.zeros((), device=x.device)
    return torch.stack(losses).mean()


def salience_diversity_loss(raw_logits, target_std=2.0):
    """Penalizes AdaptiveAttentionPooling's salience_head degenerating into an
    input-independent constant (see EncCombine.py's salience_floor comment and
    this module's collapse postmortem, part 2): koleo_loss alone wasn't
    enough because it operates on L2-normalized pooled state, so a salience
    gate that collapses to the *same* value for every token and every example
    is invisible to it (a shared scale factor cancels out under
    normalization) -- yet real training hit exactly that: gate_salience_mean
    pinned at a single saturated value (1.0, then salience_floor) for dozens
    of epochs straight. This is a hinge on raw_logits' per-example std across
    the query dimension, active only below target_std, so genuinely
    low-entropy inputs (a handful of clearly-salient tokens, rest clearly
    not) aren't penalized -- only the degenerate case where every query slot
    converges to the exact same gate value is."""
    std = raw_logits.squeeze(-1).std(dim=1)   # [B]
    return F.relu(target_std - std).mean()


def salience_margin_loss(raw_logits, margin=6.0):
    """Cheap structural safety net alongside salience_diversity_loss: a hinge
    on |raw_logits| itself, active only beyond `margin` (well past where
    sigmoid's gradient is already vanishing), discouraging the logit from
    saturating deep enough into either tail to become a dead gate in the
    first place. Same defensive spirit as salience_floor, addressing the
    logit rather than only the sigmoid's output range."""
    overflow = (raw_logits.abs() - margin).clamp(min=0)
    return overflow.pow(2).mean()


def teacher_forced_forward(expert, state, tgt_in):
    device = tgt_in.device
    T = tgt_in.size(1)
    emb = expert.token_embedding(tgt_in) * math.sqrt(expert.d_model)
    emb = emb + expert._sincos_pos_embed(T, device)
    tgt_mask = expert._causal_mask(T, device)
    weighted_h, ponder_cost = expert._recursive_decode(emb, state, None, tgt_mask)
    logits = expert.output_proj(weighted_h)
    return logits, ponder_cost


def _scheduled_sampling_tgt_in(expert, state, codec, device, T_in):
    """Free-run (no_grad, greedy) rollout from the model's OWN current policy,
    re-tokenized and padded/truncated to T_in, to condition the (still fully
    differentiable) teacher-forced pass on -- while tgt_out stays the true
    gold continuation. Exposes the supervised loss to the model's own drift,
    which nothing else in compute_task_loss does (every other term
    conditions on the true gold prefix). Greedy (top_k=1), matching
    evaluate_rewards' rationale -- targets the deterministic collapse mode
    actually observed ("True" every time), not stochastic sampling noise."""
    with torch.no_grad():
        gen_text, _ = expert.forward(state, max_length=T_in, top_k=1)
    ids = codec.sp.encode_as_ids(gen_text[0] if gen_text else "")
    ids = [BOS_ID] + ids[: T_in - 1]
    ids = ids + [PAD_ID] * (T_in - len(ids))
    return torch.tensor([ids], dtype=torch.long, device=device)


def compute_task_loss(expert, state, target_text, codec, device, coefs, label_smoothing,
                      source=None, scheduled_sampling_prob=0.0, rng=None,
                      return_probs0=False, brier_coef=None, blank_answer_penalty_coef=None,
                      repetition_penalty_coef=None, content_closeness_coef=None):
    """Composite training objective: teacher-forced CE (the differentiable
    proxy for the exact-match / ROUGE-L reward) plus small length, ponder-cost
    and Brier-calibration terms. See module docstring for the reasoning.
    ``brier_coef``/``blank_answer_penalty_coef``/``repetition_penalty_coef``/
    ``content_closeness_coef``: the resolved scalar weights for their
    respective scheduled terms (see scheduled_coef/HYPERPARAMS["reward"]'s
    "_start"/"_end" pairs) -- train_stage's main loop resolves each itself
    (it has global_step/total_steps in scope) and passes them in, keeping
    this function a pure function of its inputs rather than recomputing
    training progress itself. Callers with no meaningful training-progress
    concept of their own (validate/self_correct_step/tune) leave these at
    their defaults, which each fall back to that term's own "_end" value --
    the "mature model" weight, a reasonable flat constant for a
    non-training-loop call.
    ``source`` selects a longer target cap for tasks whose true answer
    doesn't fit the default (see HYPERPARAMS["global"]["max_target_len_overrides"]);
    left None for callers whose target is never one of those sources (e.g.
    self_correct_step's synthetic 'yes'/'no' verification target).
    ``scheduled_sampling_prob``/``rng``: with this probability, condition on
    the model's own free-run rollout instead of the true gold prefix (see
    _scheduled_sampling_tgt_in) -- left at their defaults (off) by every
    caller except train()'s main loop, so val_loss/tune/self_correct_step
    stay pure teacher-forced measurements.
    ``return_probs0=True`` additionally returns the row's first-position
    output distribution and whether that position is already correct (see
    decoder_diversity_loss/train_stage's call site) -- both reuse tensors
    already computed below for the brier term, no extra forward-pass work.
    Default False keeps every other call site's 2-tuple return untouched."""
    max_len = HYPERPARAMS["global"]["max_target_len_overrides"].get(source) if source else None
    tgt = codec.encode_target(target_text, device, max_len=max_len)
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    if scheduled_sampling_prob > 0 and rng is not None and rng.random() < scheduled_sampling_prob:
        tgt_in = _scheduled_sampling_tgt_in(expert, state, codec, device, tgt_in.size(1))
    logits, ponder_cost = teacher_forced_forward(expert, state, tgt_in)

    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
        ignore_index=PAD_ID, label_smoothing=label_smoothing,
    )

    probs = F.softmax(logits, dim=-1)
    conf, pred_tok = probs.max(dim=-1)
    valid = (tgt_out != PAD_ID).float()
    correct = (pred_tok.detach() == tgt_out).float()
    brier = (((conf - correct) ** 2) * valid).sum() / valid.sum().clamp(min=1)

    length_term = valid.sum() / codec.max_len
    ponder_term = ponder_cost.mean() / expert.max_recursive_steps

    # Blank-answer penalty: directly penalizes P(EOS) at the very first
    # generated position when the gold answer isn't actually empty there.
    # Ordinary CE already penalizes any wrong token at position 0, but
    # treats "predicted EOS" the same as any other wrong token -- this adds
    # an *extra* cost specifically for the "gave up immediately" collapse
    # free-run generation was observed doing (see HYPERPARAMS["reward"]).
    # Covers every id that decodes to '' (PAD, BOS, EOS -- verified against the
    # real tokenizer in multimodal_bpe.model), not EOS alone. Greedy decoding
    # can emit PAD or BOS as its first token, which yields an empty output
    # without stopping generation, and the EOS-only version never penalized
    # that. Found by reading the code after slurm-4829832: the ab_off and ab_on
    # arms both sat on '' outputs with state diverse, so the blank-output
    # collapse was not coming from state.
    empty_decoding_ids = list(coefs["blank_penalty_ids"])
    empty_prob_at_start = probs[:, 0, empty_decoding_ids].sum(dim=-1)
    gold_not_empty = (~torch.isin(tgt_out[:, 0], torch.tensor(empty_decoding_ids, device=device))).float()
    blank_penalty = (empty_prob_at_start * gold_not_empty).mean()

    # Repetition penalty: penalizes the model for assigning high probability
    # to immediately repeating the token it was just fed, at every position
    # -- computed from the same teacher-forced logits, no extra generation
    # pass needed. Skips positions where the gold answer legitimately does
    # repeat the previous token, so this doesn't fight genuinely-repetitive
    # correct answers, only the excess probability mass beyond what's
    # actually correct there. Targets the "yes yes yes..."/"sesese..."
    # collapse free-run generation was observed doing.
    repeat_prob = probs.gather(-1, tgt_in.unsqueeze(-1)).squeeze(-1)
    gold_is_repeat = (tgt_out == tgt_in).float()
    repetition_penalty = ((repeat_prob * (1 - gold_is_repeat)) * valid).sum() / valid.sum().clamp(min=1)

    # Length-closeness bonus: unlike the two penalties above (only active
    # when generation goes badly wrong), this is a small *positive* signal,
    # active at every step, that continuously pulls the model's own
    # predicted output length toward the gold answer's actual length --
    # independent of whether the words themselves are right yet. An easier,
    # content-independent target to satisfy early on (get the length right
    # first, the words later) than waiting for content accuracy alone to
    # improve, and a positive complement to the penalties rather than only
    # ever punishing the worst failure modes.
    #
    # Computed from the model's own per-position P(EOS) under teacher
    # forcing, treated as a per-step hazard rate: P(stops exactly at
    # position t) = P(EOS at t) * P(hasn't already stopped before t) --
    # giving a fully differentiable "expected stopping length" to compare
    # against the gold length (any probability mass that never resolves to
    # EOS within this window is attributed to the window's own length, the
    # natural censored-case convention).
    #
    # Gated by correctness (see HYPERPARAMS["reward"]["length_closeness_gate_threshold"]):
    # ungated, this term rewards ANY short output on rows with a short gold
    # answer regardless of whether the words are right -- diagnosed as one
    # of two mechanisms (the other being an uncalibrated shared RL
    # baseline) behind a real training run's collapse to a fixed,
    # short-but-usually-wrong output ("True", see slurm-4631350
    # postmortem). Only rewards length-shaping on rows the model is
    # already mostly getting right under teacher forcing, using the
    # `correct` tensor already computed above for brier -- no extra
    # forward pass.
    eos_probs = probs[..., EOS_ID]                                       # [B, T]
    survival = torch.cumprod(1.0 - eos_probs, dim=1)                     # P(hasn't stopped by/at t)
    survived_before = torch.cat(
        [torch.ones_like(survival[:, :1]), survival[:, :-1]], dim=1)     # P(hasn't stopped before t)
    p_stop_at = survived_before * eos_probs                              # P(stops exactly at t)
    seq_len = logits.size(1)
    positions = torch.arange(1, seq_len + 1, device=device, dtype=torch.float32).unsqueeze(0)
    expected_length = (p_stop_at * positions).sum(dim=1) + survival[:, -1] * seq_len
    gold_length = valid.sum(dim=1)
    per_row_length_gap = (expected_length - gold_length).abs()           # [B], kept per-row for gating
    row_correct_frac = (correct * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    length_gate = (row_correct_frac >= coefs["length_closeness_gate_threshold"]).float()
    length_closeness_term = (
        (per_row_length_gap * length_gate).sum() / length_gate.sum().clamp(min=1)
    ) / codec.max_len

    # Content-closeness term: bounded ([0, 1]) and linear, unlike CE's
    # unbounded log penalty -- rewards probability mass placed on the
    # correct token at each position, but doesn't let a single
    # genuinely-hard-to-know token (e.g. an obscure trivia_qa/nq_open
    # proper noun a from-scratch model has no realistic path to knowing)
    # produce the same runaway loss magnitude CE alone would. CE remains
    # the dominant, sharper signal for tokens the model actually can
    # learn; this softens the "avalanche" that otherwise makes giving up
    # on content (in favor of the learnable auxiliary terms above) the
    # path of least resistance during optimization.
    p_correct = probs.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)       # [B, T]: P(correct token) at each position
    content_closeness_term = (
        1.0 - (p_correct * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    ).mean()

    resolved_brier_coef = brier_coef if brier_coef is not None else coefs["brier_coef_end"]
    resolved_blank_penalty_coef = (blank_answer_penalty_coef if blank_answer_penalty_coef is not None
                                   else coefs["blank_answer_penalty_coef_end"])
    resolved_repetition_penalty_coef = (repetition_penalty_coef if repetition_penalty_coef is not None
                                        else coefs["repetition_penalty_coef_end"])
    resolved_content_closeness_coef = (content_closeness_coef if content_closeness_coef is not None
                                       else coefs["content_closeness_coef_end"])
    loss = (
        coefs["task_loss_coef"] * ce
        + coefs["length_penalty_coef"] * length_term
        + coefs["ponder_cost_coef"] * ponder_term
        + resolved_brier_coef * brier
        + resolved_blank_penalty_coef * blank_penalty
        + resolved_repetition_penalty_coef * repetition_penalty
        + coefs["length_closeness_coef"] * length_closeness_term
        + resolved_content_closeness_coef * content_closeness_term
    )
    if return_probs0:
        probs0 = probs[:, 0, :].squeeze(0)
        pos0_correct = bool(correct[0, 0].item())
        return loss, ce.detach().item(), probs0, pos0_correct
    return loss, ce.detach().item()


def decoder_diversity_loss(probs0, recent_buffer, eps=1e-8):
    """Same nearest-neighbour-repulsion idea as koleo_loss (see its
    docstring), applied to the decoder's own first-token output
    distribution instead of the encoder's pooled state -- targets the
    "always predict the same word regardless of input" collapse directly,
    rather than whichever specific token that collapse currently happens to
    favor. Root-cause context: even with koleo_loss (state collapse) and
    the salience-gate diversity terms (gate collapse) both active and
    healthy, a real run (slurm-4695121) still collapsed to unconditionally
    predicting the single word 'The' for every source starting ~85% through
    a 50-epoch run -- the third distinct flavor of the same underlying
    mechanism this file keeps finding (previously 'True', then blank/EOS).
    Every prior fix patched the specific shortcut just observed; this one
    instead makes emitting the same output regardless of input expensive in
    general.

    recent_buffer holds detached probs[:, 0, :] vectors from recently
    processed rows (any source -- shared across the whole stage-1 mix,
    matching how the collapse itself was shared across all three sources).
    Compared against this rolling history instead of a live re-encoded
    batch (unlike koleo_loss) since this row's own probs0 must stay part of
    *this* row's graph for its own backward() call (train_stage does true
    per-row backward accumulation, so nothing later could re-use it) --
    this also means, unlike koleo_loss, no second forward pass is needed at
    all: probs0 is already computed by compute_task_loss's normal forward
    pass. Only ever applied on rows the model's position-0 prediction is
    still wrong on (see train_stage's call site) -- logi_glue legitimately
    has ~46 answers repeated constantly, so many of its rows should cluster
    near "True"/"False" in probability space, and this gate keeps the term
    from fighting that legitimate convergence (see
    length_closeness_gate_threshold's identical gating rationale)."""
    if len(recent_buffer) < 2:
        return torch.zeros((), device=probs0.device)
    x = F.normalize(probs0, dim=-1)
    bank = F.normalize(torch.stack(recent_buffer), dim=-1)
    dots = bank @ x
    nn_idx = dots.argmax()
    dist = (probs0 - recent_buffer[nn_idx]).pow(2).sum().add(eps).sqrt()
    return -torch.log(dist)


def decoder_diversity_batch_contrastive_loss(probs0_batch, gold_first_tokens, temperature, eps=1e-8):
    """P2.2 (see decoder_diversity_contrastive_coef's HYPERPARAMS comment
    for the full motivation): the "positive pull" decoder_diversity_loss
    structurally lacks. That term only ever repels probs0 from a rolling
    history buffer -- it has no notion of which rows' outputs SHOULD
    legitimately agree (e.g. logi_glue's real, frequent True/False repeats)
    vs. which merely happen to collide because the policy is
    content-blind. This term supplies that signal directly, within one
    optimizer-step batch: rows sharing the same gold first token are
    pulled together, rows with a different gold first token are pushed
    apart -- computed on the same L2-normalized probability-simplex
    representation decoder_diversity_loss already uses (not a separate
    embedding).

    Two branches, chosen per-row by whether that row has a same-gold
    partner in this batch:

    - Has a partner: standard batch-local supervised-contrastive loss
      (Khosla et al. 2020) -- pulls toward the positive(s), pushes apart
      from the negatives, via one softmax-normalised ratio.
    - No partner in this batch (the common case for trivia_qa/nq_open,
      whose answers are 97%+ unique -- a typical mixed-source batch often
      has ZERO shared gold tokens among them): falls back to plain mean
      cosine-similarity repulsion against every other row. Without this
      fallback, verified via a synthetic unit test, a row with no positive
      partner contributes EXACTLY ZERO regardless of whether its output
      has collapsed to match an unrelated row's -- precisely the failure
      mode this term exists to catch, and the more common batch
      composition than the "has a partner" branch given this project's
      source mix.

    probs0_batch: [B, V] stacked first-token output distributions, live
    graph -- requires a fresh batch-local forward pass to obtain (unlike
    decoder_diversity_loss's single row, probs0 for every row in the batch
    must be simultaneously live; the main per-row loop's own probs0 tensors
    are already freed by each row's own backward() by the time a batch
    boundary is reached -- see train_stage's call site, mirrors
    koleo_loss's identical "re-encode this batch once more, cheaply, for a
    batch-level term" pattern).
    gold_first_tokens: length-B list/tensor of each row's true first
    target-vocabulary id (tgt_out[:, 0] in compute_task_loss's terms)."""
    B = probs0_batch.size(0)
    if B < 2:
        return torch.zeros((), device=probs0_batch.device)
    x = F.normalize(probs0_batch, dim=-1)
    raw_sim = x @ x.t()                                                   # [B, B], cosine, [-1, 1]
    gold = torch.as_tensor(gold_first_tokens, device=probs0_batch.device)
    self_mask = torch.eye(B, dtype=torch.bool, device=probs0_batch.device)
    pos_mask = (gold.unsqueeze(0) == gold.unsqueeze(1)) & ~self_mask
    has_positive = pos_mask.any(dim=1)                                    # [B]

    losses = torch.zeros(B, device=probs0_batch.device, dtype=probs0_batch.dtype)
    if has_positive.any():
        sim = raw_sim / temperature
        exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values.detach())
        exp_sim = exp_sim.masked_fill(self_mask, 0.0)
        denom = exp_sim.sum(dim=1).clamp(min=eps)
        pos_sum = (exp_sim * pos_mask).sum(dim=1).clamp(min=eps)
        log_prob = torch.log(pos_sum) - torch.log(denom)
        losses = torch.where(has_positive, -log_prob, losses)
    no_positive = ~has_positive
    if no_positive.any():
        mean_sim_to_others = raw_sim.masked_fill(self_mask, 0.0).sum(dim=1) / (B - 1)
        losses = torch.where(no_positive, mean_sim_to_others, losses)
    return losses.mean()


def _buffer_diversity(buffer):
    """Diagnostic only (not a loss, no gradient) -- mean pairwise L2
    distance among decoder_diversity_loss's own rolling buffer entries.
    decoder_div's value alone can't distinguish "the buffer holds a
    genuinely diverse set of outputs and this row's nearest neighbor just
    happens to be close" from "the buffer itself has collapsed to a small,
    fixed set of shortcuts" (repulsion relocating the collapse rather than
    escaping it -- see decoder_diversity_loss's docstring for the collapse
    history this term fights). Logged alongside unique_pred_fraction so a
    future run can tell these two cases apart directly instead of
    re-deriving it from grepping sample-output logs after the fact."""
    if len(buffer) < 2:
        return 0.0
    bank = torch.stack(buffer)
    dists = torch.cdist(bank, bank, p=2)
    mask = ~torch.eye(dists.size(0), dtype=torch.bool, device=dists.device)
    return dists[mask].mean().item()


@torch.no_grad()
def gate_scale_summary(model):
    """Diagnostic only (no gradient) -- mean |value| of the ReZero-style
    residual gates added to EncCombine.py's LinearAttentionBlock/
    HybridLinearEncoderLayer/AdaptiveAttentionPooling and Logic.py's
    RoutingLogic (all init 0, see their own docstrings for the collapse
    investigation this fixed). A real harness run (slurm-4795036) showed
    unique_pred_fraction/state_sim_within/across recovering to genuinely
    healthy values early in training (something no run before these fixes
    ever did), then repeatedly relapsing back toward collapse over the
    following ~7000 steps despite every anti-collapse loss term fully
    ramped -- this logs the gates' own current magnitude so that relapse
    can be directly checked against "did these gates open back up into the
    same regime they started at exactly 0 to avoid" instead of guessing.
    Grouped by role rather than logging all 22 individual scalars: linear-
    attention layers' own attn/ffn gates (8 each), the 2 full-attention
    anchors' attn/ffn gates, pooling's single ffn gate, and the router's
    single state gate."""
    linear_attn, linear_ffn, full_attn, full_ffn = [], [], [], []
    for layer in model.combine.encoder.layers:
        if layer.use_linear_attn:
            linear_attn.append(layer.attn.residual_scale.abs().item())
            linear_ffn.append(layer.ffn_residual_scale.abs().item())
        else:
            full_attn.append(layer.attn_residual_scale.abs().item())
            full_ffn.append(layer.ffn_residual_scale.abs().item())
    return {
        "linear_attn_mean": sum(linear_attn) / len(linear_attn) if linear_attn else float("nan"),
        "linear_ffn_mean": sum(linear_ffn) / len(linear_ffn) if linear_ffn else float("nan"),
        "full_attn_mean": sum(full_attn) / len(full_attn) if full_attn else float("nan"),
        "full_ffn_mean": sum(full_ffn) / len(full_ffn) if full_ffn else float("nan"),
        "pool_ffn": model.combine.pool.ffn_residual_scale.abs().item(),
        "router_state": model.router.state_rezero.abs().item(),
    }


# --------------------------------------------------------------------------- #
# Scheduled supervised/RL blend (see HYPERPARAMS["reward"]'s schedule
# comment). Mirrors reasoning_to_text.py's reasoning_rl_term/_schedule_weights
# -- t2t_rl_term reuses the exact same single-sample REINFORCE mechanism
# (sampled generation + compute_reward_metric + a moving-average baseline),
# just for text_to_text.py specifically. Unlike reasoning_to_text.py, whose
# schedule lets RL reach 100% by the end, this one caps RL's share at
# HYPERPARAMS["reward"]["rl_ceiling"] (25-50%, tunable) so supervised
# learning always stays dominant -- this stage never hands the wheel fully
# to RL the way reasoning_to_text.py's does.
# --------------------------------------------------------------------------- #
def t2t_rl_term(expert, state, row):
    """Single-sample REINFORCE term for one training row: sample a
    generation from the current policy (gradients flow through its
    log-probability, not through the discrete sampling choice itself --
    the standard score-function estimator), score it with
    compute_reward_metric, and return (reward, logprob, entropy) for the
    caller to combine with a per-source baseline. Reward is a plain float
    (no grad needed on it) -- logprob and entropy both need to stay
    tensors. entropy is the sampled generation's summed per-step sampling
    entropy (see TextOutputExpert.forward's return_entropy) -- used as an
    anti-collapse bonus, see HYPERPARAMS["reward"]["entropy_coef"]. Uses
    the same max_target_len_overrides lookup evaluate_rewards already
    does, so a long-target source like asqa isn't unfairly truncated
    mid-answer."""
    max_len = HYPERPARAMS["global"]["max_target_len_overrides"].get(
        row["source"], HYPERPARAMS["global"]["max_target_len"])
    gen, _, logprob, entropy = expert.forward(
        state, max_length=max_len, return_logprob=True, return_entropy=True)
    pred = gen[0] if gen else ""
    reward = compute_reward_metric(row["source"], pred, row["target"], row.get("aux", ""))
    return reward, logprob.mean(), entropy.mean()


def _schedule_weights(global_step, total_steps, coefs):
    """Linearly ramp the RL share from 0 up to coefs["rl_ceiling"] by
    training progress (global_step / total_steps, the same progress
    fraction the LR scheduler already uses), with the supervised share
    always the strict complement (1 - rl_w) -- unlike
    reasoning_to_text.py's four independently-configured start/end values,
    this is deliberately a single tunable knob (the ceiling) since RL is
    meant to stay a minority contributor here, never fully take over."""
    p = min(1.0, global_step / max(1, total_steps))
    rl_w = p * coefs["rl_ceiling"]
    return 1.0 - rl_w, rl_w


def scheduled_coef(global_step, total_steps, start, end):
    """Generic linear ramp from ``start`` (step 0) to ``end`` (global_step
    == total_steps), sharing _schedule_weights' exact progress fraction --
    lets any HYPERPARAMS["reward"] coefficient become a schedule (an
    "X_start"/"X_end" pair) instead of a flat constant, moving in sync with
    the RL blend's own ramp. Added for brier_coef specifically (see its
    comment: calibration pressure should stay light while the model is
    still mostly wrong about everything, then grow substantial by the same
    point the RL share does), generalized so other reward-shaping terms
    can be tested as schedules too without each reimplementing this math."""
    p = min(1.0, global_step / max(1, total_steps))
    return start + (end - start) * p


# --------------------------------------------------------------------------- #
# Self-correction: generate, intentionally corrupt, then train the model to
# catch its own mistake (see module docstring).
# --------------------------------------------------------------------------- #
_DECOY_WORDS = ["wrong", "different", "unrelated", "incorrect", "mismatched"]


def corrupt_answer(text, rng):
    """Heuristic single-error corruption: flips a digit for numeric-looking
    answers, otherwise swaps in one decoy word."""
    digit_positions = [i for i, c in enumerate(text) if c.isdigit()]
    if digit_positions:
        i = rng.choice(digit_positions)
        old = text[i]
        choices = [d for d in "0123456789" if d != old]
        return text[:i] + rng.choice(choices) + text[i + 1:]
    words = text.split()
    if not words:
        return text
    i = rng.randrange(len(words))
    words[i] = rng.choice(_DECOY_WORDS)
    return " ".join(words)


def self_correct_step(model, expert, codec, device, question, gold_answer, coefs, rng):
    with torch.no_grad():
        state = encode_to_state(model, question)
        gen_text, _ = expert.forward(state, max_length=HYPERPARAMS["global"]["max_target_len"])
        candidate = (gen_text[0] if gen_text else "").strip() or gold_answer

    corrupted = rng.random() < 0.5
    shown = corrupt_answer(candidate, rng) if corrupted else candidate
    verify_input = (
        f"Question: {question}\nProposed answer: {shown}\n"
        f"Is the proposed answer correct? Reply with exactly 'yes' or 'no'."
    )
    verify_target = "no" if corrupted else "yes"

    v_state = encode_to_state(model, verify_input)
    loss, _ = compute_task_loss(
        expert, v_state, verify_target, codec, device, coefs, label_smoothing=0.0,
    )
    return loss


# --------------------------------------------------------------------------- #
# Validation (teacher-forced CE, drives checkpoint/valloss) + reward metrics
# (free-run generation, logging only)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, expert, codec, device, val_rows, coefs, sample_size):
    model.eval()
    rows = sample_rows(val_rows, min(sample_size, len(val_rows)), seed=0)
    total_ce, n = 0.0, 0
    for r in rows:
        state = encode_to_state(model, r["input"])
        _, ce = compute_task_loss(expert, state, r["target"], codec, device, coefs,
                                  label_smoothing=0.0, source=r["source"])
        total_ce += ce
        n += 1
    model.train()
    return total_ce / max(1, n)


@torch.no_grad()
def gate_health(model, val_rows):
    """Periodic diagnostic (see EncCombine.py's salience_floor/return_gate_stats
    comments): a real trained checkpoint was found with combine's salience gate
    saturated at ~0 for every input, silently zeroing out state regardless of
    content -- this surfaces that failure mode directly in the log (a collapsing
    salience_mean trending toward the floor) instead of requiring a full manual
    investigation to discover, the way evaluate_rewards' sample logging
    originally surfaced the free-run generation collapse. One cheap sample row,
    same encode() call encode_to_state itself makes, just requesting the extra
    diagnostic return value.

    Explicitly toggles eval()/train() itself (rather than relying on
    whatever mode the caller left the model in) since the call site sits
    right after evaluate_rewards, which already restores train() before
    returning -- without this, gate stats were measured with dropout ON
    while the actual collapsed generations this diagnostic is meant to
    explain come from evaluate_rewards' model.eval() pass, a methodology
    mismatch that could make salience look healthier (more variance from
    dropout noise) than it really is."""
    if not val_rows:
        return None
    was_training = model.training
    model.eval()
    with torch.no_grad():
        encodings = model.encode(val_rows[0]["input"], "text")
        _, _, _, stats = model.combine(encodings, return_gate_stats=True)
    if was_training:
        model.train()
    # raw_logits is a differentiable tensor meant for salience_diversity_loss/
    # salience_margin_loss's live gradient path (see train_stage), not for
    # this read-only diagnostic -- drop it so callers that dump gate_health's
    # dict wholesale (the periodic print's gate_str, wandb_log) don't choke
    # on a non-scalar value.
    stats.pop("raw_logits", None)
    return stats


def _source_precision_recall(source, pairs):
    """Macro-averaged classification precision/recall for logi_glue's small,
    constantly-repeated closed label set (see HYPERPARAMS["reward"]'s
    "overwhelming loss composite" postmortem -- precision/recall requested
    as a clearer signal than reward alone for whether the model is
    genuinely discriminating between labels or just favoring one); mean
    ROUGE-L precision/recall (best-matching alias, same lookup
    compute_reward_metric already uses) for open-ended generation sources,
    where a fixed label set doesn't apply. ``pairs`` is a list of
    (pred, target, aux) tuples for one source from evaluate_rewards."""
    if source == "logi_glue":
        norm_pairs = [(normalize_answer(p), normalize_answer(t)) for p, t, _ in pairs]
        labels = sorted(set(t for _, t in norm_pairs))
        precisions, recalls = [], []
        for label in labels:
            tp = sum(1 for p, t in norm_pairs if p == label and t == label)
            pred_pos = sum(1 for p, _ in norm_pairs if p == label)
            actual_pos = sum(1 for _, t in norm_pairs if t == label)
            precisions.append(tp / pred_pos if pred_pos else 0.0)
            recalls.append(tp / actual_pos if actual_pos else 0.0)
        return ((sum(precisions) / len(precisions) if precisions else 0.0),
                (sum(recalls) / len(recalls) if recalls else 0.0))
    precs, recs = [], []
    for pred, target, aux in pairs:
        aliases = [a for a in (aux or "").split("|") if a]
        gold = aliases if aliases else [target]
        best = max((_rouge_l_prf(pred, g) for g in gold),
                  key=lambda prf: prf[2], default=(0.0, 0.0, 0.0))
        precs.append(best[0])
        recs.append(best[1])
    return ((sum(precs) / len(precs) if precs else 0.0),
            (sum(recs) / len(recs) if recs else 0.0))


def _pairwise_state_similarity(pooled_states, pooled_sources):
    """Mean cosine similarity between pooled vectors sharing a source
    ("within_source") vs. not ("across_source") -- see evaluate_rewards'
    docstring for the diagnostic rationale. Standalone (not inlined in
    evaluate_rewards) so diagnose_state_collapse.py's per-stage trace (see
    its own module docstring -- same question, asked at intermediate points
    in the encode_to_state pipeline, not just the final pooled state) can
    reuse the identical computation instead of a second hand-copied version
    silently drifting from this one."""
    state_similarity = {"within_source": float("nan"), "across_source": float("nan")}
    if len(pooled_states) >= 2:
        normed = F.normalize(torch.stack(pooled_states, dim=0), dim=-1)
        sim = normed @ normed.t()
        uniq_sources = sorted(set(pooled_sources))
        label_map = {s: i for i, s in enumerate(uniq_sources)}
        labels = torch.tensor([label_map[s] for s in pooled_sources], device=sim.device)
        same_source = labels.unsqueeze(0) == labels.unsqueeze(1)
        not_self = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        within_mask = same_source & not_self
        across_mask = ~same_source
        if within_mask.any():
            state_similarity["within_source"] = sim[within_mask].mean().item()
        if across_mask.any():
            state_similarity["across_source"] = sim[across_mask].mean().item()
    return state_similarity


@torch.no_grad()
def common_mode_ratio(pooled):
    """Diagnostic only. For a [B, D] stack of pooled representations: norm of
    the batch-mean vector divided by the mean norm of each row's deviation
    from that mean. Large values mean one shared offset dominates every row.
    That matters for koleo_loss because it L2-normalizes its inputs, whose
    gradient scales as 1/|x|: a big common offset shrinks the gradient on the
    informative residual. Added to test why encoder_koleo sat in the
    collapsed range (~+2.2 to +2.8) for ~2000 steps in slurm-4829741 despite
    its coefficient being active."""
    if pooled.size(0) < 2:
        return float("nan")
    mean = pooled.mean(dim=0)
    dev = (pooled - mean).norm(dim=-1).mean()
    return (mean.norm() / dev.clamp(min=1e-8)).item()


@torch.no_grad()
def evaluate_rewards(model, expert, val_rows, sample_size, seed):
    """Returns (per-source avg reward, unique_pred_fraction,
    most_common_pred_fraction, per-source (precision, recall),
    state_similarity).

    unique_pred_fraction is a direct early-warning tripwire for the "same
    output regardless of input" collapse family (see decoder_diversity_loss
    /HYPERPARAMS["reward"]["decoder_diversity_coef"]'s postmortem) -- a real
    run collapsed to unconditionally predicting the single word "The" for
    ~15% of a 50-epoch run before anyone noticed, since gate_salience_*
    looked fine (that collapse isn't a gate problem) and per-source reward
    looked fine or even *better* (ROUGE-L rewarding a lucky "The ..."
    prefix match on trivia_qa). Counting distinct predictions across this
    same sample (already generated below, just not deduplicated) surfaces
    that regardless of which specific token a future collapse favors or
    whether it happens to score well by coincidence.

    most_common_pred_fraction (fraction of this sample sharing the single
    most frequent prediction string) exists alongside unique_pred_fraction
    because that alone can't tell "32 distinct wrong answers out of 200"
    apart from "199 copies of one fluent default word plus 1 different
    one" -- both read as the same low unique_pred_fraction, but they're
    very different severities. Added after a real run (slurm-4829152)
    showed the decoder had stopped its earlier degenerate repeated-token
    looping (a real improvement) while still printing the exact same
    plausible-looking word -- e.g. "True" -- across logi_glue/trivia_qa/
    nq_open samples at a single checkpoint: the same collapse, now wearing
    a fluent default instead of gibberish, which unique_pred_fraction
    alone reports identically either way.

    precision/recall (see _source_precision_recall) give a second,
    independent read on the same question -- whether the model is
    genuinely discriminating per input or converging on a favorite.

    state_similarity is {"within_source": mean cosine sim between pooled
    `state` of two rows sharing a source, "across_source": same but for two
    rows from different sources} -- a direct test of whether a flat
    unique_pred_fraction floor traces to upstream state-collapse (encoder/
    combine producing near-identical state regardless of input, so
    within/across similarity are both ~1.0) vs. a downstream decoder that
    ignores a genuinely-varying state (across_source clearly lower than
    within_source, since same-source rows are textually more alike to
    begin with). Reuses the same mean-pool-over-sequence-dim state already
    computed for koleo_loss (train_stage's batch-boundary block) rather
    than inventing a new pooling convention."""
    model.eval()
    rows = sample_rows(val_rows, min(sample_size, len(val_rows)), seed=seed)
    by_source = {}
    preds_by_source = {}
    all_preds = []
    pooled_states = []
    pooled_sources = []
    # Print one real (prediction, target) pair per source the first time
    # that source comes up this call -- rewards have been sitting at ~0.0
    # across every source for many epochs despite steadily falling
    # teacher-forced CE, and the only way to tell "the model's free-run
    # generations are genuinely empty/degenerate" apart from "they're just
    # not exact-match-correct yet" is to actually look at what's being
    # generated, not just the aggregate score.
    logged_sources = set()
    for r in rows:
        state = encode_to_state(model, r["input"])
        pooled_states.append(state.mean(dim=1).squeeze(0))
        pooled_sources.append(r["source"])
        # Greedy, not the default temperature=1.0/top_k=50 sampling used for
        # training-time exploration (see self_correct_step) -- reward is an
        # exact-match-style metric measuring whether the model's *best*
        # answer is right, so it must decode the argmax token every step
        # (top_k=1 collapses the post-filter softmax to one candidate,
        # making the subsequent multinomial draw deterministic). Sampling
        # here would silently understate accuracy on any answer the model
        # already gets right most, but not all, of the time.
        max_len = HYPERPARAMS["global"]["max_target_len_overrides"].get(
            r["source"], HYPERPARAMS["global"]["max_target_len"])
        gen, _ = expert.forward(state, max_length=max_len, top_k=1)
        pred = gen[0] if gen else ""
        all_preds.append(pred)
        reward = compute_reward_metric(r["source"], pred, r["target"], r.get("aux", ""))
        by_source.setdefault(r["source"], []).append(reward)
        preds_by_source.setdefault(r["source"], []).append((pred, r["target"], r.get("aux", "")))
        if r["source"] not in logged_sources:
            logged_sources.add(r["source"])
            print(f"  [reward-sample] {r['source']}: pred={pred!r} target={r['target']!r} reward={reward:.3f}")
    model.train()
    unique_pred_fraction = len(set(all_preds)) / len(all_preds) if all_preds else 0.0
    most_common_count = max(Counter(all_preds).values()) if all_preds else 0
    most_common_pred_fraction = most_common_count / len(all_preds) if all_preds else 0.0
    precision_recall = {s: _source_precision_recall(s, pairs) for s, pairs in preds_by_source.items()}

    state_similarity = _pairwise_state_similarity(pooled_states, pooled_sources)

    return ({s: sum(v) / len(v) for s, v in by_source.items()}, unique_pred_fraction,
            most_common_pred_fraction, precision_recall, state_similarity)


@torch.no_grad()
def pos0_diagnostics(expert, states):
    """Diagnostic only. The first greedy decode step over a sample of rows: which
    token id the argmax lands on, and the mean probability mass on each id that
    decodes to ''. Free-run generation starts from exactly this distribution
    (generated = [BOS]), so this shows what it will emit before generation runs.
    Added to tell apart the empty outputs seen in slurm-4829831/4829832 -- an
    EOS stop, or PAD/BOS emitted without stopping (see compute_task_loss's blank
    penalty, which previously covered EOS alone)."""
    counts = Counter()
    mass = {PAD_ID: 0.0, BOS_ID: 0.0, EOS_ID: 0.0}
    all_logits, margins = [], []
    xattn_queries, xattn_outs = [], []
    captured = {}

    def _capture(module, inputs, output):
        if "q" not in captured:
            captured["q"] = inputs[0][0, 0].detach().clone()
            captured["out"] = output[0][0, 0].detach().clone()

    handle = expert.decoder_layer.multihead_attn.register_forward_hook(_capture)
    for state in states:
        captured.clear()
        logits, _ = teacher_forced_forward(expert, state, torch.tensor([[BOS_ID]], device=state.device))
        xattn_queries.append(captured["q"])
        xattn_outs.append(captured["out"])
        raw = logits[0, 0, :]
        p = F.softmax(raw, dim=-1)
        counts[int(p.argmax().item())] += 1
        for k in mass:
            mass[k] += p[k].item()
        top2 = raw.topk(2).values
        margins.append((top2[0] - top2[1]).item())
        all_logits.append(raw)
    n = max(1, len(states))
    # State can only flip the argmax if its effect on the top logit (row_std)
    # is comparable to the top-1 margin. If row_std << margin_mean, the state
    # reaches the decoder but not enough to change the first token.
    handle.remove()
    mode_id = counts.most_common(1)[0][0]
    row_std = (torch.stack(all_logits)[:, mode_id].std().item() if len(all_logits) > 1 else float("nan"))
    # Share of the decoder's first-step cross-attention OUTPUT that varies across
    # rows, relative to the magnitude of its query (the residual entering that
    # sublayer). Small means the state barely moves the decoder's residual at
    # this step, so the decision is made by the state-independent path. Splits
    # "state lost inside cross-attention" from "state lost downstream of it".
    if len(xattn_outs) > 1:
        out_stack, q_stack = torch.stack(xattn_outs), torch.stack(xattn_queries)
        xattn_row_spread = out_stack.std(dim=0).norm().item()
        query_scale = q_stack.mean(dim=0).norm().item()
        xattn_ratio = xattn_row_spread / max(query_scale, 1e-8)
    else:
        xattn_ratio = float("nan")
    return {"top_ids": counts.most_common(3),
            "mean_p": {"pad": round(mass[PAD_ID] / n, 4),
                       "bos": round(mass[BOS_ID] / n, 4),
                       "eos": round(mass[EOS_ID] / n, 4)},
            "margin_mean": round(sum(margins) / n, 3),
            "row_std_top": round(row_std, 3),
            "xattn_ratio": round(xattn_ratio, 4)}


# --------------------------------------------------------------------------- #
# Optuna hyperparameter tuning
# --------------------------------------------------------------------------- #
def _is_oom_error(e):
    """torch.cuda.OutOfMemoryError (a RuntimeError subclass) exists on
    recent torch; fall back to a message match so this still catches OOM on
    older versions that only ever raised a plain RuntimeError for it."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(e, oom_cls):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def tune(device):
    import optuna

    print("\n===== TUNING: text_to_text =====")
    ensure_dataset_ready()
    train_rows = load_csv_rows(TRAIN_CSV)
    val_rows = load_csv_rows(VAL_CSV)
    gcfg = HYPERPARAMS["global"]
    tune_train = sample_rows(train_rows, gcfg["tune_train_subset"], seed=1)
    tune_val = sample_rows(val_rows, gcfg["tune_val_subset"], seed=2)
    print(f"[data] tuning on {len(tune_train)} train / {len(tune_val)} val rows")

    codec = SpmCodec(max_len=gcfg["max_target_len"])
    space = HYPERPARAMS["search_space"]
    budget = HYPERPARAMS["tune_budget_batches"]
    d_model = gcfg["d_model"]
    coefs = HYPERPARAMS["reward"]

    os.makedirs(STUDY_DIR, exist_ok=True)
    study = optuna.create_study(
        study_name="text_to_text",
        storage=f"sqlite:///{os.path.join(STUDY_DIR, 'text_to_text.db')}",
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=0),
        load_if_exists=True,
    )

    # Seeded from this study's own pruned-trial history (persists across
    # resubmissions) -- see enc_pre.py's tune() for the full reasoning
    # (identical scheme): every pruned trial here is an OOM, so the
    # smallest batch_size that ever OOM'd is a real observed data point,
    # not a prediction, and nothing else in this search space meaningfully
    # changes memory footprint.
    prior_oom = [
        t.params["batch_size"] for t in
        study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.PRUNED,))
        if "batch_size" in t.params
    ]
    known_oom_batch_size = min(prior_oom) if prior_oom else None
    if known_oom_batch_size is not None:
        print(f"  [tune] batch_size >= {known_oom_batch_size} previously OOM'd; "
              f"skipping those without attempting them")

    def objective(trial):
        nonlocal known_oom_batch_size
        hp = {
            "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            "weight_decay": trial.suggest_float("weight_decay", *space["weight_decay"], log=True),
            "warmup_steps": trial.suggest_categorical("warmup_steps", space["warmup_steps"]),
            "label_smoothing": trial.suggest_categorical("label_smoothing", space["label_smoothing"]),
        }

        if known_oom_batch_size is not None and hp["batch_size"] >= known_oom_batch_size:
            print(f"  trial {trial.number}: SKIPPED -- batch_size={hp['batch_size']} "
                  f">= known OOM threshold {known_oom_batch_size}")
            raise optuna.TrialPruned()

        model = build_model(device, d_model, gcfg["half_precision_encoder"])
        expert = model.expert_registry.get_expert("text_output")
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=hp["learning_rate"], weight_decay=hp["weight_decay"],
        )

        try:
            model.train()
            accum = 0
            optimizer.zero_grad()
            for step in range(budget):
                row = tune_train[step % len(tune_train)]
                state = encode_to_state(model, row["input"])
                loss, _ = compute_task_loss(expert, state, row["target"], codec, device,
                                            coefs, hp["label_smoothing"], source=row["source"])
                (loss / hp["batch_size"]).backward()
                accum += 1
                if accum == hp["batch_size"]:
                    optimizer.step()
                    optimizer.zero_grad()
                    accum = 0

            val_loss = validate(model, expert, codec, device, tune_val, coefs,
                                sample_size=len(tune_val))
        except Exception as e:
            if not _is_oom_error(e):
                raise
            known_oom_batch_size = min(known_oom_batch_size or hp["batch_size"], hp["batch_size"])
            print(f"  trial {trial.number}: OOM at batch_size={hp['batch_size']} "
                  f"({e}); pruning this trial and continuing with the next one.")
            raise optuna.TrialPruned()
        finally:
            # Always runs, including on a CUDA OOM mid-trial -- otherwise the
            # failed model/optimizer (and torch's fragmented allocator
            # cache) stay resident and the *next* trial OOMs too.
            del model, optimizer
            if device == "cuda":
                torch.cuda.empty_cache()

        print(f"  trial {trial.number}: val_loss={val_loss:.4f}  hp={hp}")
        wandb_log({"tune/val_loss": val_loss, **{f"tune/{k}": v for k, v in hp.items()}})
        return val_loss

    # tune_trials is the *target total* completed trials for this study, not
    # "how many more to run" -- otherwise resubmitting after a crash keeps
    # adding tune_trials more forever instead of the search ever finishing.
    # Only COMPLETE trials count (a trial cut short by a crash isn't a
    # completed data point).
    target = HYPERPARAMS["tune_trials"]
    n_done = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
    n_remaining = max(0, target - n_done)
    if n_done:
        print(f"[resume] {n_done} trial(s) already completed")
    if n_remaining == 0:
        print(f"[resume] study already has {n_done} completed trial(s) "
              f">= target tune_trials={target}; nothing to do.")
    else:
        print(f"[resume] running {n_remaining} more trial(s) to reach "
              f"target tune_trials={target}")
        study.optimize(objective, n_trials=n_remaining)

    best_hp = study.best_params
    out = os.path.join(STUDY_DIR, "text_to_text_best_hparams.json")
    with open(out, "w") as f:
        json.dump(best_hp, f, indent=2)
    print(f"\nBest text_to_text hyperparameters: {best_hp}")
    print(f"Saved to {out}")
    return best_hp


def load_best_hparams():
    path = os.path.join(STUDY_DIR, "text_to_text_best_hparams.json")
    if os.path.isfile(path):
        with open(path) as f:
            hp = json.load(f)
        hp.setdefault("lr_min_scale", HYPERPARAMS["defaults"]["lr_min_scale"])
        return hp
    print("[train] no tuned hyperparameters found; using defaults")
    return HYPERPARAMS["defaults"]


# --------------------------------------------------------------------------- #
# Full (resumable) training over the entire compiled dataset
# --------------------------------------------------------------------------- #
def train_stage(device, stage_sources, epochs_key, checkpoint_prefix,
                warm_start_prefix=None, rank=0, world_size=1):
    """One of text_to_text's two independently-stoppable training stages (see
    STAGE1_SOURCES/STAGE2_SOURCES/run_stages) -- ``stage_sources`` restricts
    both train_rows/val_rows to that stage's task subset, ``epochs_key``
    selects that stage's own hand-set epoch count out of HYPERPARAMS, and
    ``checkpoint_prefix``/``warm_start_prefix`` give this stage its own
    checkpoint namespace while letting it inherit a prior stage's finished
    weights as its starting point (see build_model's fallback_prefix)."""
    from dist_utils import barrier, broadcast_parameters, is_main_process, rank0_first, sync_gradients

    is_distributed = world_size > 1
    main = is_main_process(rank)
    if main:
        print(f"\n===== TRAINING: text_to_text ({checkpoint_prefix}) =====")
    rank0_first(rank, world_size, ensure_dataset_ready)
    train_rows = [r for r in load_csv_rows(TRAIN_CSV) if r["source"] in stage_sources]
    val_rows = [r for r in load_csv_rows(VAL_CSV) if r["source"] in stage_sources]
    if main:
        print(f"[data] train rows: {len(train_rows)} | val rows: {len(val_rows)}")

    hp = load_best_hparams()
    gcfg = HYPERPARAMS["global"]
    coefs = HYPERPARAMS["reward"]
    d_model = gcfg["d_model"]

    model = build_model(device, d_model, gcfg["half_precision_encoder"],
                        checkpoint_prefix=checkpoint_prefix, fallback_prefix=warm_start_prefix)
    expert = model.expert_registry.get_expert("text_output")
    codec = SpmCodec(max_len=gcfg["max_target_len"])

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=hp["learning_rate"], weight_decay=hp["weight_decay"],
    )

    epochs = HYPERPARAMS[epochs_key]
    batch_size = hp["batch_size"]
    # Per-rank row count: truncated to a multiple of (batch_size * world_size)
    # so every rank's shard is itself an exact multiple of batch_size --
    # every rank then calls optimizer.step()/sync_gradients() the same
    # number of times per epoch (required: a rank calling an all_reduce one
    # more time than the others would hang the whole job waiting on it).
    usable_rows = len(train_rows) - (len(train_rows) % (batch_size * world_size))
    rows_per_rank = usable_rows // world_size
    n_opt_steps_per_epoch = max(1, rows_per_rank // batch_size)
    total_steps = n_opt_steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(hp["warmup_steps"], total_steps, hp["lr_min_scale"]),
    )

    # ---- Resume from the latest checkpoint if present ---------------------- #
    # build_model() already warm-started combine/router/expert_registry from
    # whatever's on disk (preferring 'best'). If t2t_expert_registry's own
    # 'latest' tag carries an embedded training-state payload (see
    # save_checkpoint / ExpertRegistry.store_weights' ``extra`` param), that
    # means *this exact stage* has prior progress to resume -- re-load
    # combine/router/expert_registry from their own 'latest' tag specifically
    # (not whatever build_model() picked, which could be an earlier 'best'
    # checkpoint than the one this optimizer state and step counters are
    # actually consistent with), then restore the optimizer/RNG/counters.
    # Every rank reads the same shared-storage files independently, so no
    # broadcast is needed for anything that was actually loaded from disk;
    # broadcasting unconditionally afterward is still correct and cheap, and
    # covers the one case that does need it (fresh init with nothing on disk
    # at all yet, which would otherwise differ per rank and never receive
    # gradients to converge back together, since frozen params get no
    # updates at all).
    start_epoch, start_row, global_step = 0, 0, 0
    best_val = float("inf")
    # Moving-average baselines for the scheduled RL term's variance
    # reduction (see t2t_rl_term) -- one independent baseline per task
    # source, not a single shared scalar (see
    # HYPERPARAMS["reward"]["rl_baseline_momentum"]'s comment for why: a
    # single global baseline blended 5 near-always-0-reward tasks with
    # logi_glue's real signal, letting logi_glue's REINFORCE gradient
    # dominate the shared generation policy -- see slurm-4631350
    # postmortem). Per-rank-local, never synced across ranks, matching
    # reasoning_to_text.py's identical RL baselines (a small,
    # noise-reduction heuristic, not something requiring exact cross-rank
    # consistency).
    rl_baselines = {src: 0.0 for src in stage_sources}
    er_path, training_state = _load_weights_with_fallback(
        model.expert_registry, f"{checkpoint_prefix}_expert_registry", device)
    if training_state:
        for module, name in ((model.combine, f"{checkpoint_prefix}_combine"),
                             (model.router, f"{checkpoint_prefix}_router")):
            _load_weights_with_fallback(module, name, device)
        optimizer.load_state_dict(training_state["optimizer_state"])
        # training_state was loaded via ExpertRegistry.load_weights(...,
        # map_location=device) -- on a GPU rank that remaps every tensor in
        # the checkpoint, including this one, onto the GPU. set_rng_state()
        # rejects anything but a plain CPU torch.ByteTensor, so it must be
        # moved back explicitly regardless of where map_location put it.
        torch.set_rng_state(training_state["rng"].cpu())
        global_step, best_val = training_state["global_step"], training_state["best_val_loss"]
        # .get(...) default: backward-compatible with checkpoints saved
        # before this field existed (or saved under the old single-scalar
        # "rl_baseline" key, from before the per-source baseline change).
        rl_baselines = training_state.get("rl_baselines", {})
        for src in stage_sources:
            rl_baselines.setdefault(src, 0.0)
        for _ in range(global_step):          # fast-forward the LR decay schedule
            scheduler.step()
        if _is_epoch_complete(er_path):
            # Only trust that an epoch is truly finished if we're resuming
            # from its literal 'complete' marker checkpoint -- a mid-epoch
            # row checkpoint never implies completion, even if its row
            # number happens to equal that epoch's total row count.
            start_epoch, start_row = training_state["epoch"] + 1, 0
        else:
            start_epoch, start_row = training_state["epoch"], training_state["batch"]
        if main:
            print(f"[resume] {os.path.basename(er_path)} -> epoch {start_epoch}, "
                  f"row {start_row}, step {global_step}")
    if is_distributed:
        broadcast_parameters(model.parameters(), src=0)
    barrier(is_distributed)

    ckpt_every = gcfg["checkpoint_every_batches"]
    eval_every = gcfg["eval_every_batches"]
    self_correct_p = coefs["self_correct_prob"]
    val_sample = min(len(val_rows), 500)

    # Rolling history for decoder_diversity_loss (see its docstring) --
    # persists across epochs within this run, same as rl_baselines, but
    # intentionally *not* checkpointed/restored across a resume: it's a
    # short-memory (buffer_size rows) regularizer that self-corrects within
    # a fraction of one epoch regardless of how it starts.
    recent_probs0 = []

    model.train()
    for epoch in range(start_epoch, epochs):
        # Deterministic per-epoch shuffle interleaves every source task --
        # identical on every rank (same seed), then stride-sliced into
        # disjoint per-rank shards (see dist_utils.shard_indices).
        order = list(range(len(train_rows)))
        random.Random(gcfg["seed"] + epoch).shuffle(order)
        order = order[:usable_rows]
        if is_distributed:
            from dist_utils import shard_indices
            order = shard_indices(order, rank, world_size)
        rng = random.Random(gcfg["seed"] * 1000 + epoch + rank)

        skip_to = start_row if epoch == start_epoch else 0
        accum, batch_idx = 0, skip_to // batch_size
        optimizer.zero_grad()

        for i in range(skip_to, len(order)):
            row = train_rows[order[i]]
            state, salience_logits = encode_to_state(model, row["input"], return_gate_logits=True)
            # See anti_collapse_ramp_steps' comment: these four ramp together
            # over a short, dedicated horizon (NOT total_steps/the RL blend's
            # own progress fraction) so they reach real strength well before
            # the early-training collapse window, not just by the time the
            # full stage finishes.
            ramp_steps = coefs["anti_collapse_ramp_steps"]
            brier_w = scheduled_coef(global_step, ramp_steps,
                                     coefs["brier_coef_start"], coefs["brier_coef_end"])
            blank_penalty_w = scheduled_coef(global_step, ramp_steps,
                                             coefs["blank_answer_penalty_coef_start"],
                                             coefs["blank_answer_penalty_coef_end"])
            repetition_penalty_w = scheduled_coef(global_step, ramp_steps,
                                                  coefs["repetition_penalty_coef_start"],
                                                  coefs["repetition_penalty_coef_end"])
            content_closeness_w = scheduled_coef(global_step, ramp_steps,
                                                 coefs["content_closeness_coef_start"],
                                                 coefs["content_closeness_coef_end"])
            supervised_loss, ce, probs0, pos0_correct = compute_task_loss(
                expert, state, row["target"], codec, device, coefs, hp["label_smoothing"],
                source=row["source"], scheduled_sampling_prob=coefs["scheduled_sampling_prob"],
                rng=rng, return_probs0=True, brier_coef=brier_w,
                blank_answer_penalty_coef=blank_penalty_w,
                repetition_penalty_coef=repetition_penalty_w,
                content_closeness_coef=content_closeness_w,
            )

            if rng.random() < self_correct_p:
                supervised_loss = supervised_loss + coefs["self_correct_coef"] * self_correct_step(
                    model, expert, codec, device, row["input"], row["target"], coefs, rng,
                )

            # Scheduled supervised/RL blend (see HYPERPARAMS["reward"]'s
            # schedule comment). advantage uses this source's baseline
            # *before* this row's update (matching reasoning_to_text.py's
            # identical ordering) -- reward and rl_baselines are plain
            # floats, so none of this touches the autograd graph; only
            # logprob/entropy do. entropy_coef discourages the sampled
            # policy from collapsing to a near-deterministic distribution
            # (see t2t_rl_term/TextOutput.forward's return_entropy).
            sup_w, rl_w = _schedule_weights(global_step, total_steps, coefs)
            reward, logprob, entropy = t2t_rl_term(expert, state, row)
            source = row["source"]
            pg_loss = -(reward - rl_baselines[source]) * logprob - coefs["entropy_coef"] * entropy
            momentum = coefs["rl_baseline_momentum"]
            rl_baselines[source] = momentum * rl_baselines[source] + (1 - momentum) * reward

            # Salience-gate anti-collapse terms (see salience_diversity_loss/
            # salience_margin_loss and the HYPERPARAMS["reward"] comment) --
            # computed on the same forward pass that already produced state,
            # no re-encode needed (unlike koleo_loss below, which does need
            # one, since this loop backward()s per row and salience_logits'
            # own graph is still live right here).
            gate_diversity = salience_diversity_loss(salience_logits)
            gate_margin = salience_margin_loss(salience_logits)

            # Decoder-output anti-collapse term (see decoder_diversity_loss
            # and the module docstring's "The" collapse postmortem) -- only
            # on rows the model isn't already getting right at position 0
            # (see decoder_diversity_loss's gating rationale), same pattern
            # as length_closeness_term's correctness gate above.
            decoder_div = (torch.zeros((), device=device) if pos0_correct
                          else decoder_diversity_loss(probs0, recent_probs0))

            loss = (sup_w * supervised_loss + rl_w * pg_loss
                    + coefs["salience_diversity_coef"] * gate_diversity
                    + coefs["salience_margin_coef"] * gate_margin
                    + coefs["decoder_diversity_coef"] * decoder_div)

            (loss / batch_size).backward()
            recent_probs0.append(probs0.detach())
            if len(recent_probs0) > coefs["decoder_diversity_buffer_size"]:
                recent_probs0.pop(0)
            accum += 1

            if accum == batch_size:
                # Anti-collapse regularizer (see koleo_loss/HYPERPARAMS["reward"]
                # ["koleo_coef"] and the module docstring's collapse postmortem):
                # pushes this batch's pooled state representations apart. Each
                # row's own state graph was already freed by its own backward()
                # call above (this loop does true per-row backward accumulation,
                # not a single batched forward/backward), so this re-encodes the
                # same batch_size rows just processed a second time -- a cheap
                # encode_to_state-only pass, not the full decoder -- purely to
                # get a live graph for this term's gradient, which joins the
                # same optimizer step as everything already accumulated above.
                # koleo_loss (state diversity) and P2.2's contrastive term
                # (decoder_diversity_contrastive_coef, see its HYPERPARAMS
                # comment) both need a fresh, live-graph forward pass over
                # this batch's rows -- each row's own graph from the main
                # loop above is already freed by that row's own backward()
                # call. Shares ONE encode_to_state call per row between both
                # terms (koleo only needs the pooled state; the contrastive
                # term needs that same state fed through the decoder for
                # probs0) rather than re-encoding independently, and combines
                # both into a SINGLE backward() call (not two separate ones)
                # since r_state is a shared upstream tensor for both branches
                # -- backward()ing one first would free graph buffers the
                # other still needs.
                # encoder_koleo/pool_koleo (see HYPERPARAMS["reward"]
                # ["encoder_koleo_coef"]/["pool_koleo_coef"] and
                # encode_to_state's return_encoder_memory comment): same
                # repulsion idea as koleo above, applied to combine.encoder's
                # own pre-pooling output and combine.pool's own output
                # instead of only the final pooled state -- encoder_koleo
                # was the direct fix for the relapse-after-small-gates
                # failure mode (slurm-4800583); pool_koleo is the same fix
                # one stage further downstream, added once a real run
                # (slurm-4829152) confirmed encoder_koleo itself recovers
                # (settles healthy for a long stretch) while state_sim still
                # fully collapses at some of those same checkpoints -- with
                # the encoder confirmed healthy, the remaining collapse has
                # nowhere left to hide except pooling/router. pool_koleo's
                # coefficient is currently 0.0 (its value is still computed
                # and logged; see HYPERPARAMS["reward"]["pool_koleo_coef"]
                # for why it duplicates koleo while the router gate is
                # small). Shares the SAME encode_to_state call per row as
                # koleo/contrastive below (two extra return values, not two
                # extra forward passes).
                koleo_rows = [train_rows[order[j]] for j in range(i - batch_size + 1, i + 1)]
                pooled_states, pooled_memories, pooled_pools, batch_probs0, batch_gold0 = [], [], [], [], []
                for r in koleo_rows:
                    r_state, r_memory, r_combined = encode_to_state(
                        model, r["input"], return_encoder_memory=True)
                    pooled_states.append(r_state.mean(dim=1).squeeze(0))
                    pooled_memories.append(r_memory.mean(dim=1).squeeze(0))
                    pooled_pools.append(r_combined.mean(dim=1).squeeze(0))
                    r_max_len = gcfg["max_target_len_overrides"].get(r["source"])
                    r_tgt = codec.encode_target(r["target"], device, max_len=r_max_len)
                    r_logits, _ = teacher_forced_forward(expert, r_state, r_tgt[:, :-1])
                    batch_probs0.append(F.softmax(r_logits, dim=-1)[:, 0, :].squeeze(0))
                    batch_gold0.append(r_tgt[0, 1].item())
                contrastive = decoder_diversity_batch_contrastive_loss(
                    torch.stack(batch_probs0, dim=0), batch_gold0,
                    coefs["decoder_diversity_contrastive_temperature"])
                koleo = koleo_loss(torch.stack(pooled_states, dim=0))
                encoder_koleo = koleo_loss(torch.stack(pooled_memories, dim=0))
                pool_koleo = koleo_loss(torch.stack(pooled_pools, dim=0))
                batch_term_loss = (
                    coefs["koleo_coef"] * koleo
                    + coefs["encoder_koleo_coef"] * encoder_koleo
                    + coefs["pool_koleo_coef"] * pool_koleo
                    + coefs["decoder_diversity_contrastive_coef"] * contrastive
                )
                batch_term_loss.backward()

                sync_gradients(model.parameters(), world_size)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum = 0
                batch_idx += 1
                global_step += 1

                if main and global_step % 50 == 0:
                    # decoder_div/buffer_div: see _buffer_diversity's docstring --
                    # distinguishes "repulsion inactive" (both near 0) from
                    # "repulsion active but just relocating the collapse among a
                    # small fixed set of shortcuts" (decoder_div healthy-looking
                    # while buffer_div stays low) without re-deriving it from
                    # sample-output logs after the fact.
                    buffer_div = _buffer_diversity(recent_probs0)
                    # koleo: see koleo_loss's docstring for expected ranges --
                    # healthy pooled-state diversity is roughly -0.3 to -0.7,
                    # collapsed (near-identical state across the batch) is
                    # roughly +3 to +18. Never logged before despite being
                    # backward()'d every optimizer step, which made "is state
                    # itself collapsing" unanswerable without an ad-hoc probe.
                    gates = gate_scale_summary(model)
                    cm_memory = common_mode_ratio(torch.stack(pooled_memories, dim=0))
                    cm_state = common_mode_ratio(torch.stack(pooled_states, dim=0))
                    print(f"  e{epoch} row{i} step{global_step} "
                          f"source={row['source']} ce={ce:.4f} brier_w={brier_w:.4f} "
                          f"blank_w={blank_penalty_w:.4f} rep_w={repetition_penalty_w:.4f} "
                          f"cc_w={content_closeness_w:.4f} decoder_div={decoder_div.item():.4f} "
                          f"buffer_div={buffer_div:.4f} contrastive={contrastive.item():.4f} "
                          f"koleo={koleo.item():.4f} "
                          f"encoder_koleo={encoder_koleo.item():.4f} "
                          f"pool_koleo={pool_koleo.item():.4f} "
                          f"cm_memory={cm_memory:.3f} cm_state={cm_state:.3f} "
                          f"gate_lin_attn={gates['linear_attn_mean']:.4f} "
                          f"gate_lin_ffn={gates['linear_ffn_mean']:.4f} "
                          f"gate_full_attn={gates['full_attn_mean']:.4f} "
                          f"gate_full_ffn={gates['full_ffn_mean']:.4f} "
                          f"gate_pool_ffn={gates['pool_ffn']:.4f} "
                          f"gate_router={gates['router_state']:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")
                    wandb_log({"train/ce_loss": ce, "train/brier_w": brier_w,
                              "train/blank_penalty_w": blank_penalty_w,
                              "train/repetition_penalty_w": repetition_penalty_w,
                              "train/content_closeness_w": content_closeness_w,
                              "train/decoder_div": decoder_div.item(),
                              "train/buffer_div": buffer_div,
                              "train/contrastive": contrastive.item(),
                              "train/koleo": koleo.item(),
                              "train/encoder_koleo": encoder_koleo.item(),
                              "train/pool_koleo": pool_koleo.item(),
                              "train/cm_memory": cm_memory, "train/cm_state": cm_state,
                              **{f"train/gate_{k}": v for k, v in gates.items()},
                              "train/lr": scheduler.get_last_lr()[0],
                              "epoch": epoch, "row": i})

                # Intermediate weight saving after each batch (optimizer step).
                # Eval-step checkpoints already embed val_loss, which strictly
                # subsumes a plain ckpt_every save at the same global_step, so
                # only fall back to the plain save when this step isn't also
                # an eval step (avoids two near-duplicate files for one step).
                # Rank-0-only: no collective involved, so other ranks just
                # move on to their next row instead of waiting.
                if main and global_step % eval_every == 0:
                    val = validate(model, expert, codec, device, val_rows, coefs, val_sample)
                    best_val = min(best_val, val)
                    (rewards, unique_pred_fraction, most_common_pred_fraction, precision_recall,
                     state_similarity) = evaluate_rewards(
                        model, expert, val_rows, sample_size=200, seed=global_step)
                    gate = gate_health(model, val_rows)
                    gate_str = (f" gate_salience_mean={gate['salience_mean']:.4f}"
                                f" gate_salience_min={gate['salience_min']:.4f}"
                                f" gate_salience_max={gate['salience_max']:.4f}") if gate else ""
                    pr_str = {k: (round(p, 3), round(r, 3)) for k, (p, r) in precision_recall.items()}
                    # state_similarity: see evaluate_rewards' docstring -- both
                    # numbers near 1.0 (within_source ~= across_source) points
                    # at upstream state-collapse; across_source clearly lower
                    # than within_source points at a downstream decoder
                    # ignoring a genuinely-varying state instead.
                    # most_common_pred_fraction: see evaluate_rewards' docstring
                    # -- distinguishes "many distinct wrong answers" from "one
                    # fluent default word standing in for almost everything,"
                    # which unique_pred_fraction alone can't.
                    print(f"  [val] step {global_step}: val_loss={val:.4f} "
                          f"(best={best_val:.4f}) rewards="
                          f"{ {k: round(v, 3) for k, v in rewards.items()} }{gate_str} "
                          f"unique_pred_fraction={unique_pred_fraction:.3f} "
                          f"most_common_pred_fraction={most_common_pred_fraction:.3f} "
                          f"state_sim_within={state_similarity['within_source']:.4f} "
                          f"state_sim_across={state_similarity['across_source']:.4f} "
                          f"precision_recall={pr_str}")
                    wandb_log({"val/loss": val, "val/unique_pred_fraction": unique_pred_fraction,
                              "val/most_common_pred_fraction": most_common_pred_fraction,
                              "val/state_sim_within": state_similarity["within_source"],
                              "val/state_sim_across": state_similarity["across_source"],
                              **{f"val/reward_{k}": v for k, v in rewards.items()},
                              **{f"val/precision_{k}": p for k, (p, _) in precision_recall.items()},
                              **{f"val/recall_{k}": r for k, (_, r) in precision_recall.items()},
                              **({f"val/gate_{k}": v for k, v in gate.items()} if gate else {})})
                    save_checkpoint(model, optimizer, epoch, i + 1, global_step, best_val,
                                    val_loss=val, rl_baselines=rl_baselines,
                                    checkpoint_prefix=checkpoint_prefix)
                elif main and global_step % ckpt_every == 0:
                    save_checkpoint(model, optimizer, epoch, i + 1, global_step, best_val,
                                    rl_baselines=rl_baselines, checkpoint_prefix=checkpoint_prefix)

        # ---- End-of-epoch validation + checkpoint --------------------------- #
        # All ranks must reach this point before any starts the next epoch's
        # loop (rank 0's extra validation/checkpoint work makes it slower per
        # epoch than the others).
        barrier(is_distributed)
        start_row = 0
        if not main:
            continue
        val = validate(model, expert, codec, device, val_rows, coefs, val_sample)
        best_val = min(best_val, val)
        (rewards, unique_pred_fraction, most_common_pred_fraction, precision_recall,
         state_similarity) = evaluate_rewards(
            model, expert, val_rows, sample_size=300, seed=epoch)
        gate = gate_health(model, val_rows)
        gate_str = (f" gate_salience_mean={gate['salience_mean']:.4f}"
                    f" gate_salience_min={gate['salience_min']:.4f}"
                    f" gate_salience_max={gate['salience_max']:.4f}") if gate else ""
        pr_str = {k: (round(p, 3), round(r, 3)) for k, (p, r) in precision_recall.items()}
        print(f"[epoch {epoch}] val_loss={val:.4f} (best={best_val:.4f}) rewards={rewards}{gate_str} "
              f"unique_pred_fraction={unique_pred_fraction:.3f} "
              f"most_common_pred_fraction={most_common_pred_fraction:.3f} "
              f"state_sim_within={state_similarity['within_source']:.4f} "
              f"state_sim_across={state_similarity['across_source']:.4f} "
              f"precision_recall={pr_str}")
        wandb_log({"val/epoch_loss": val, "epoch": epoch,
                  "val/unique_pred_fraction": unique_pred_fraction,
                  "val/most_common_pred_fraction": most_common_pred_fraction,
                  "val/state_sim_within": state_similarity["within_source"],
                  "val/state_sim_across": state_similarity["across_source"],
                  **{f"val/precision_{k}": p for k, (p, _) in precision_recall.items()},
                  **{f"val/recall_{k}": r for k, (_, r) in precision_recall.items()},
                  **({f"val/gate_{k}": v for k, v in gate.items()} if gate else {})})
        save_checkpoint(model, optimizer, epoch, len(order), global_step, best_val,
                        val_loss=val, epoch_complete=True, rl_baselines=rl_baselines,
                        checkpoint_prefix=checkpoint_prefix)

    barrier(is_distributed)
    if not main:
        return
    # 'best'/'latest' are maintained continuously by save_checkpoint (see its
    # docstring), so no finalize step is needed here -- just read them. The
    # embedded training-state payload on this stage's expert_registry tags is
    # no longer needed now that training has genuinely finished, so it's
    # stripped out (the file itself stays -- it's a permanent artifact now,
    # not a throwaway).
    _strip_training_state(f"{checkpoint_prefix}_expert_registry")
    best_text_output = find_weights(f"{checkpoint_prefix}_text_output", "best")
    latest_text_output = find_weights(f"{checkpoint_prefix}_text_output", "latest")
    print(f"\n[done] text_to_text {checkpoint_prefix} complete.")
    print(f"[optimal weights] text_output -> {best_text_output or latest_text_output}")
    print(f"[latest weights] text_output -> {latest_text_output}")


# --------------------------------------------------------------------------- #
# Stage driver
# --------------------------------------------------------------------------- #
def run_stages(args, device, rank=0, world_size=1):
    """The tune stage stays single-process (rank 0 only, other ranks idle at
    a barrier) -- many short, independent trials is a different, unrelated
    axis of parallelism from data-parallel training (see dist_utils.py's
    module docstring). Both train stages run on every rank via
    train_stage()'s own distributed handling. train_stage1/train_stage2 each
    get their own stop-and-resubmit boundary, exactly like tune does --
    stage 2 warm-starts from stage 1's finished weights (see
    build_model's fallback_prefix) once resubmitted."""
    from dist_utils import barrier, is_main_process

    is_distributed = world_size > 1
    main = is_main_process(rank)
    progress = load_progress()

    if "tune" not in progress["completed"]:
        if main:
            init_wandb(args.wandb_key, "tune")
            tune(device)
            mark_stage_done("tune")
            finish_wandb()
            print("\n*** Stage 'tune' complete. Stopping for review. "
                  "Re-submit the job to continue with training. ***")
        barrier(is_distributed)
        return

    if "train_stage1" not in progress["completed"]:
        if main:
            init_wandb(args.wandb_key, "train_stage1")
        train_stage(device, stage_sources=STAGE1_SOURCES, epochs_key="epochs_stage1",
                   checkpoint_prefix="t2t_stage1", rank=rank, world_size=world_size)
        if main:
            mark_stage_done("train_stage1")
            finish_wandb()
            print("\n*** Stage 'train_stage1' complete. Stopping for review. "
                  "Re-submit the job to continue with stage 2. ***")
        barrier(is_distributed)
        return

    if "train_stage2" not in progress["completed"]:
        if main:
            init_wandb(args.wandb_key, "train_stage2")
        train_stage(device, stage_sources=STAGE2_SOURCES, epochs_key="epochs_stage2",
                   checkpoint_prefix="t2t_stage2", warm_start_prefix="t2t_stage1",
                   rank=rank, world_size=world_size)
        if main:
            mark_stage_done("train_stage2")
            finish_wandb()

    if main:
        print("\nAll text_to_text stages complete.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Text-to-text tuning/training.")
    p.add_argument("--token", default="_", help="Hugging Face token.")
    p.add_argument("--wandb_key", default="_", help="Weights & Biases API key.")
    p.add_argument("--prepare_data_only", action="store_true",
                   help="Compile + split the CSV dataset, then exit.")
    p.add_argument("--reset_progress", action="store_true",
                   help="Clear stage progress (does not delete weights/data).")
    return p.parse_args()


def main():
    # Degrades to single-process (rank 0, world_size 1) automatically when
    # not launched via torchrun -- see dist_utils.py's module docstring.
    from dist_utils import cleanup_distributed, init_distributed, is_main_process, rank0_first

    args = parse_args()
    rank, world_size, local_rank, is_distributed, device = init_distributed()
    main_process = is_main_process(rank)
    print(f"[device] {device}" + (f" (rank {rank}/{world_size})" if is_distributed else ""))

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    if args.token and args.token != "_":
        try:
            from huggingface_hub import login
            login(token=args.token)
        except Exception as e:
            if main_process:
                print(f"[hf] login skipped: {e}")

    if args.reset_progress:
        if main_process:
            save_progress({"completed": []})
            print("[progress] reset.")
        cleanup_distributed(is_distributed)
        return

    if args.prepare_data_only:
        rank0_first(rank, world_size, ensure_dataset_ready)
        if main_process:
            print("[data] Preparation complete.")
        cleanup_distributed(is_distributed)
        return

    run_stages(args, device, rank=rank, world_size=world_size)
    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    sys.exit(main())
