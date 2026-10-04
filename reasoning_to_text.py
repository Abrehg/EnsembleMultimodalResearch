"""
Reasoning-expert training for the ensemble multimodal research model.

Overview
--------
Unlike text_to_text.py / ocr_to_text.py / image_to_text.py (which all use
``MultimodalModel(text_only=True)`` and never touch the reasoning expert at
all -- confirmed by direct inspection before writing this file), this script
is the *first* place in the repo that trains models/Experts/Reasoning.py's
``ReasoningExpert`` and the router/expert-registry selection mechanism it
depends on. Two things the reasoning expert already had, verified by reading
Reasoning.py/Logic.py/ExpRegistry.py directly before writing any of this:

    - A genuine TRM ("Two-layer Transformer decoder" core, ``TRMBlock``)
      with ACT halting, reused recursively.
    - A VAE bolt-on (``ReasoningVAE`` + ``ReasoningExpert.set_vae()``),
      inert until attached -- exactly the "post-training attachment"
      described in Reasoning.py's own docstring.

Two things it did *not* have, and that this script had to build:

    - Any loss function at all on ReasoningExpert (it only ever returned
      ``(None, state)`` with no gradient-usable ponder/KL access -- see the
      small addition in Reasoning.py: ``last_ponder_cost_raw`` /
      ``last_kl_loss_raw``, non-detached siblings of the existing detached
      logging attributes, added because forward()'s return signature is a
      fixed contract OverallModel.py depends on and couldn't be changed).
    - Any expert-selection *loss*. ``ExpRegistry.score()``/``compare()``
      already compute a route-vector-vs-expert-embedding similarity
      (confirmed genuinely present), but nothing in the repo ever turned
      that into a training signal -- see ``selection_loss`` below, a
      standard cross-entropy over ``expert_registry.score(route_vec)``
      against a per-row "which expert *should* this have routed to" label
      that this script constructs from each row's source (reasoning tasks
      -> "reasoning", non-reasoning tasks -> "text_output"), since no such
      ground-truth label exists anywhere in the model itself.
    - A reasoning-chain-text-vs-hidden-state comparison (did not exist
      anywhere in the repo at all) -- see ``trajectory_imitation_loss``.

Two-phase curriculum (VAE bolts onto the already-trained TRM, not from
scratch)
------------------------------------------------------------------------
    1. traj_gen              -> offline: Qwen2.5-VL-7B generates a
                                 reasoning trajectory (chain-of-thought text
                                 + final answer) for every row that needs
                                 the reasoning expert, cached to a resumable
                                 JSONL sidecar (see generate_trajectories()).
    2. tune                  -> ONE Optuna search, reused for both the TRM
                                 and VAE training stages -- the VAE stage is
                                 a light structural bolt-on at a near-frozen
                                 base LR (see vae_lr_scale), not an
                                 independent architecture needing its own
                                 hyperparameter search.
    3. reasoning_train_trm   -> full resumable training, TRM only
                                 (expert.vae is None the whole stage).
    4. reasoning_train_vae   -> resumes from stage 3's finished checkpoint
                                 (warm-starting combine/router/expert_registry
                                 too, not just the reasoning expert -- see
                                 that stage's train() call for the VAE
                                 cold-start path, which reads the TRM stage's
                                 own per-component tagged files directly),
                                 attaches
                                 ReasoningVAE, and trains with two LR groups:
                                 the base TRM/ACT/BAR/MemoryLookback params at
                                 ``learning_rate * vae_lr_scale`` (near-frozen,
                                 per the task spec), the new VAE params at the
                                 full tuned learning_rate.
    5. reasoning_rl          -> small REINFORCE fine-tune with no expert
                                 trajectories at all (no Qwen dependency),
                                 optimizing the same task reward metrics
                                 directly. Needs TextOutputExpert.forward()
                                 to expose a sampled sequence's log-prob,
                                 which it didn't -- see the small,
                                 backward-compatible ``return_logprob=False``
                                 addition in TextOutput.py (existing 2-tuple
                                 return unchanged unless a caller opts in).

Tasks (entire datasets, interleaved, no per-source caps -- same
"think-on-its-feet" reasoning as text_to_text.py/image_to_text.py):

    - logi_glue    (FOLIO + PrOntoQA configs of logicreasoning/logi_glue --
                    NOT the four configs text_to_text.py's load_logi_glue
                    uses; same already-verified {input, answer_text} schema
                    reused directly from that function's own docstring, see
                    LOGI_GLUE_HF/LOGI_GLUE_CONFIGS below). Reasoning expert
                    used. Reward: binary accuracy (exact_match).
    - numinamath   (AI-MO/NuminaMath-CoT). Reasoning expert used. Reward:
                    "close match or ROUGE-L" -- close_match (numeric
                    tolerance / exact) first, else a continuous ROUGE-L
                    partial-credit score (see compute_reward_metric).
    - geoqa        (best-effort HF mirror id, unverified -- see the
                    GEOQA_HF comment, same caveat as image_to_text.py).
                    Reasoning expert used, needs the vision encoder too.
                    Reward: exact match.
    - truthfulqa   (truthfulqa/truthful_qa, "generation" config). Reasoning
                    expert *skipped* -- routed straight to text_output, per
                    the task spec. Reward: exact match.
    - commonsenseqa (tau/commonsense_qa). Reasoning expert *skipped*, MCQ
                    accuracy (exact_match on the gold option letter, same
                    scheme as image_to_text.py's ai2d/scienceqa).

Which sources skip the reasoning expert is exactly the same signal used to
build the expert-selection loss's ground-truth label (REASONING_SOURCES /
NO_REASONING_SOURCES below) -- one flag serves both purposes, since "does
this row need reasoning" and "what's the correct expert for this row" are
the same fact.

Reward composition (see compute_task_loss): task CE + length penalty +
text_output's own ponder cost (identical to the other three scripts) +
brier calibration -- coefficient intentionally *larger* here (0.15 vs.
0.02-0.05 elsewhere, per explicit instruction: this stage's answers involve
genuine uncertainty, unlike the mostly-short/low-ambiguity answers the other
stages calibrate against) -- plus, only when the reasoning expert was used:
the reasoning expert's own ACT ponder cost ("number of iterations"), a flat
"expert was called" cost ("requiring experts called"), the VAE's KL term
(zero when no VAE is attached, so this term is always safe to include
unconditionally), the expert-selection loss, and (when a Qwen trajectory is
available and passes its ROUGE-L quality gate) the SRL imitation loss.

Self-correction training (generate an answer, corrupt it half the time,
train the model to say whether the shown answer is right) is the same
pattern as the other three *_to_text.py scripts, routed through the
reasoning expert first when the row needs it -- see self_correct_step.

Supervised/RL loss schedule (TRM + VAE stages only -- see
HYPERPARAMS["reward"]'s schedule comment and reasoning_rl_term):
every training row's full supervised composite above is additionally
blended, every step, with a single-sample REINFORCE term reusing the same
sampled-generation + compute_reward_metric mechanism reasoning_rl's own
stage uses -- both objectives train simultaneously rather than in disjoint
stages, with each side's weight linearly scheduled by training progress
(supervised dominant early, RL dominant late) so neither an
over-constraining supervised signal nor an abruptly-introduced RL signal
(gradient collapse) wins by default. reasoning_rl (stage 5) is deliberately
left untouched by this and stays pure REINFORCE.

Checkpoint naming, resumability, distributed training, the Optuna
target-total/OOM-soft-fail machinery, and _prune_other_tagged_files' resume-
miss safety check are all identical in spirit and code to the other three
*_to_text.py scripts (duplicated, not imported -- established project
convention).
"""

import argparse
import glob
import json
import math
import os
import random
import re
import string
import sys
import time

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

SCRATCH_DIR = "/gpfs/u/home/ENMM/ENMMsrtk/scratch/"
DATA_DIR = os.path.join(SCRATCH_DIR, "reasoning_to_text_data")
STUDY_DIR = os.path.join(PROJECT_DIR, "studies", "reasoning_to_text")
# All weight files -- including the resumable "_wrapper" checkpoints --
# live here on barn, never on scratch. scratch is reserved strictly for
# datasets/HF cache. Shared with every other *_to_text.py / enc_pre.py.
WEIGHTS_DIR = os.path.join(PROJECT_DIR, "weights")
HF_CACHE = os.path.join(DATA_DIR, "hf_cache")
PROGRESS_PATH = os.path.join(PROJECT_DIR, "reasoning_to_text_progress.json")

COMPILED_DIR = os.path.join(DATA_DIR, "compiled")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
# Resumable JSONL sidecar for Qwen-generated trajectories, keyed by row_id
# (assigned once at compile time -- see compile_dataset). Not an arrow
# dataset: arrow datasets aren't efficiently appendable/updatable in place,
# and this needs to survive a killed/resubmitted job mid-generation the same
# way every other long-running stage in this project does.
TRAJECTORY_JSONL = os.path.join(DATA_DIR, "trajectories.jsonl")

TOKENIZER_PATH = os.path.join(MODELS_DIR, "Encoders", "multimodal_bpe.model")
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

# Same {input, answer_text} schema as text_to_text.py's load_logi_glue --
# see that function's docstring for how it was verified. Different configs
# here (FOLIO + PrOntoQA specifically, per this stage's task spec) from
# text_to_text.py's four (proofwriter/logiQA/logicNLI/babi_task_16).
LOGI_GLUE_HF = "logicreasoning/logi_glue"
# Config names are lowercase on the hub ('folio'/'prontoqa'), not 'FOLIO'/
# 'PrOntoQA' -- confirmed against the dataset's own BuilderConfig list.
LOGI_GLUE_CONFIGS = ["folio", "prontoqa"]
NUMINAMATH_HF = "AI-MO/NuminaMath-CoT"
# geoqa moved off HF entirely (MathLLMs/GeoQA doesn't exist on the Hub) to
# Geometry3K, its original official academic source -- see
# GEOMETRY3K_GDRIVE_FOLDER_ID/load_geoqa's comments for the full rationale
# (same swap as image_to_text.py's identical GeoQA -> Geometry3K change;
# duplicated here rather than imported per this project's established
# self-contained-scripts convention).
GEOMETRY3K_GDRIVE_FOLDER_ID = "1d05WYXtlgKIoaPpK1v94LYph_heiXM7Z"
TRUTHFULQA_HF = "truthfulqa/truthful_qa"
TRUTHFULQA_CONFIG = "generation"
COMMONSENSEQA_HF = "tau/commonsense_qa"

# Which sources route through the reasoning expert -- also doubles as the
# ground-truth "correct expert" label for the selection loss (see module
# docstring).
REASONING_SOURCES = {"logi_glue", "numinamath", "geoqa"}
NO_REASONING_SOURCES = {"truthfulqa", "commonsenseqa"}


# =========================================================================== #
# Hyperparameters
# =========================================================================== #
HYPERPARAMS = {
    "global": {
        "d_model": 1024,
        "seed": 42,
        "max_input_len": 96,
        "max_target_len": 32,
        # Token cap when encoding a Qwen-generated reasoning-chain string
        # through the frozen text encoder for the SRL imitation loss --
        # separate from max_target_len since chains are much longer than a
        # final answer.
        "max_trajectory_len": 256,
        "image_size": 224,           # for geoqa (see image_to_text.py's identical reasoning)
        "val_fraction": 0.05,
        "tune_train_subset": 2000,
        "tune_val_subset": 400,
        "checkpoint_every_batches": 1,
        "eval_every_batches": 300,
        "half_precision_encoder": True,
    },

    "reward": {
        "task_loss_coef": 1.0,
        "length_penalty_coef": 0.01,
        "ponder_cost_coef": 0.01,          # text_output's own ponder cost, same as the other 3 scripts
        "reasoning_ponder_coef": 0.02,     # reasoning expert's ACT ponder cost -- "number of iterations"
        "expert_call_coef": 0.01,          # flat cost for invoking the reasoning expert -- "requiring experts called"
        "kl_coef": 0.01,                   # VAE KL term; always 0 contribution when expert.vae is None
        # LARGER than the other three *_to_text.py scripts (0.02-0.05) --
        # requested explicitly for this stage: reasoning/math/logic answers
        # carry real uncertainty, unlike the short/low-ambiguity answers the
        # other stages calibrate confidence against.
        "brier_coef": 0.15,
        "selection_coef": 0.3,             # expert-selection contrastive (cross-entropy) loss weight
        "self_correct_coef": 0.2,
        "self_correct_prob": 0.15,
        "trajectory_coef": 0.5,            # SRL imitation loss weight
        # "Error checking for plain text is a straightforward comparison
        # with ROUGE-L" -- gates whether a Qwen trajectory is trusted for
        # imitation at all; a low-ROUGE trajectory (Qwen got the final
        # answer wrong) is skipped rather than actively taught.
        "trajectory_quality_threshold": 0.3,

        # ---- Degenerate-output penalties (see compute_task_loss) --------- #
        # See text_to_text.py's identical addition: discovered via
        # evaluate_rewards' sample logging that free-run greedy generation
        # can collapse to either an empty answer (immediate EOS) or a
        # single token repeated until the length cap, both scoring 0
        # reward regardless of how good val_loss looks. Both terms are
        # computed from the same teacher-forced logits already available.
        "blank_answer_penalty_coef": 0.3,
        "repetition_penalty_coef": 0.3,

        # ---- Supervised/RL loss schedule (TRM + VAE stages only; the -- #
        # ---- reasoning_rl stage stays pure REINFORCE, unchanged) -------- #
        # Every training row already gets the full supervised composite
        # above (task CE, selection, ponder, KL, trajectory-imitation,
        # self-correct); this schedule *additionally* blends in a small,
        # single-sample REINFORCE term (see reasoning_rl_term) reusing the
        # same sampled-generation + compute_reward_metric machinery
        # reasoning_rl's own stage uses, so both objectives train the
        # model simultaneously instead of in disjoint stages. The relative
        # weight of each linearly interpolates by training progress
        # (global_step / total_steps, the same progress fraction the LR
        # scheduler already uses): supervised dominates early -- grounding
        # the model in valid, well-formed outputs before it can usefully
        # explore -- and RL's share grows across training so the model
        # isn't purely supervised-constrained by the end ("stifling
        # growth"/creativity, per explicit task requirement). Ramping
        # continuously rather than hard-switching between stages is
        # deliberate: introducing a strong RL signal abruptly, with no
        # supervised grounding at all, is exactly what risks the gradient
        # collapse this was designed to avoid. Tune these 5 directly --
        # they're intentionally plain constants here rather than part of
        # HYPERPARAMS["search_space"]/Optuna's tuned trials, matching every
        # other loss-composition coefficient above (only the *training
        # regime* knobs -- batch size/LR/weight decay/warmup/label
        # smoothing -- are Optuna-tuned; loss *composition* is hand-set).
        # Optuna's short tune-stage probe (tune_budget_batches, far fewer
        # steps than a real epoch) still calibrates correctly against this
        # schedule without needing its own change: it necessarily samples
        # only the early, supervised-heavy part of the schedule, which is
        # exactly the LR-sensitive regime that matters most for picking a
        # stable learning rate (see text_to_text.py's LR-instability
        # postmortem this session for why that regime specifically is the
        # one to calibrate against).
        "supervised_coef_start": 1.0,
        "supervised_coef_end": 0.3,
        "rl_coef_start": 0.0,
        "rl_coef_end": 1.0,
        "rl_baseline_momentum": 0.9,        # same smoothing as reasoning_rl's own baseline

        # Anti-collapse regularizer on combine/router's raw pooled output (see
        # koleo_loss and train()'s call site) -- ported from text_to_text.py
        # after an extended investigation there found the "always True"
        # free-run collapse traced back to state's pooled representation
        # collapsing toward a near-identical direction during real training,
        # regardless of any decoder-side mechanism tried (nine variants,
        # including masking, self-attention blocking, cosine attention, a
        # boosted cross-attention LR, and a direct injection -- none worked).
        # koleo_loss, applied directly to state, was the one thing that fixed
        # it, verified over thousands of real training rows. Same architecture
        # (EncCombine.py's EncodingCombination + Logic.py's RoutingLogic) is
        # shared by every *_to_text.py script, so the same risk applies here.
        "koleo_coef": 1.0,
    },

    # Bumped from 3: each training row now does an extra sampled-generation
    # forward pass (for the RL term above) alongside the existing
    # teacher-forced one, and the schedule above needs enough total steps
    # for its late-training, RL-dominant regime to actually get meaningful
    # exposure rather than being cut off just as it ramps up.
    "epochs": 5,
    # One tune stage covers both the TRM and VAE training stages -- see
    # module docstring for why (VAE is a light bolt-on, not an independent
    # architecture needing its own search).
    "tune_trials": 12,
    "tune_budget_batches": 250,
    # "Near frozen" LR multiplier applied to the TRM/ACT/BAR/MemoryLookback
    # params specifically during the VAE stage (the VAE's own params train
    # at the full tuned learning_rate) -- see reasoning_train's optimizer
    # param groups.
    "vae_lr_scale": 0.01,

    "rl": {
        # "A small stage of RL" per the task spec -- a lightweight, standard
        # single-sample REINFORCE with a moving-average baseline, not
        # PPO/GAE or any heavier RL infrastructure, proportionate to "small".
        "iters": 1500,
        "batch_size": 8,
        "learning_rate": 5e-6,
        "baseline_momentum": 0.9,
        # Same anti-collapse regularizer as HYPERPARAMS["reward"]["koleo_coef"]
        # (see koleo_loss) -- own entry here since reasoning_rl reads from this
        # dict, not HYPERPARAMS["reward"]. reasoning_rl still trains
        # combine/router (loaded/saved as reasoning_rl_combine/
        # reasoning_rl_router), so the same collapse risk applies.
        "koleo_coef": 1.0,
    },

    "defaults": {
        "batch_size": 8,
        "learning_rate": 2e-4,
        "weight_decay": 1e-2,
        "warmup_steps": 200,
        "lr_min_scale": 0.1,
        "label_smoothing": 0.0,
    },
    "search_space": {
        "batch_size": [4, 8, 16],
        "learning_rate": (5e-5, 1e-3),
        "weight_decay": (1e-5, 1e-1),
        "warmup_steps": [0, 100, 300],
        "label_smoothing": [0.0, 0.1],
    },

    "data": {
        # None = use the entire available split for every source, per the
        # task spec (entire datasets, interleaved).
        "logi_glue_cap": None,
        "numinamath_cap": None,
        "geoqa_cap": None,
        "truthfulqa_cap": None,
        "commonsenseqa_cap": None,
    },

    "trajectory": {
        "qwen_model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "max_new_tokens": 512,
    },
}

STAGES = ["traj_gen", "tune", "reasoning_train_trm", "reasoning_train_vae", "reasoning_rl"]


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
            project="reasoning_to_text", name=stage, reinit=True, config=HYPERPARAMS,
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
# Greppable weight naming + lookup (shared weights/ directory -- identical
# scheme to every other script in this project, see any of their module
# docstrings for the full rationale).
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
_VALLOSS_RE = re.compile(r"valLoss(\d+\.\d+)")


def _is_epoch_complete(path):
    return bool(_COMPLETE_RE.search(os.path.basename(path)))


def _epoch_batch(path):
    name = os.path.basename(path)
    m_epoch = _EPOCH_RE.search(name)
    epoch = int(m_epoch.group(1)) if m_epoch else -1
    if _is_epoch_complete(path):
        return (epoch, float("inf"))
    m_batch = _BATCH_RE.search(name)
    batch = int(m_batch.group(1)) if m_batch else -1
    return (epoch, batch)


def _valloss(path):
    m = _VALLOSS_RE.search(os.path.basename(path))
    return float(m.group(1)) if m else float("inf")


def find_weights(component, tag):
    files = glob.glob(os.path.join(WEIGHTS_DIR, f"{component}_{tag}_epoch_*.pt"))
    if not files:
        return None
    key = _valloss if tag == "best" else _epoch_batch
    return min(files, key=key) if tag == "best" else max(files, key=key)


def _load_weights_with_fallback(module, component, device):
    """Try 'latest' then 'best' for this component, loading into ``module``
    via its own ``load_weights``. Corrupt files are deleted as they're
    found rather than crashing the run outright. Returns (path,
    load_weights' return value), or (None, None) if neither tag loads --
    for ExpertRegistry that return value is the embedded ``extra``
    training-state payload (see ExpertRegistry.store_weights), None for
    everything else. See any other script's identical function for the
    full rationale."""
    for tag in ("latest", "best"):
        path = find_weights(component, tag)
        if not path:
            continue
        try:
            return path, module.load_weights(path, map_location=device)
        except Exception as e:
            print(f"[resume] {os.path.basename(path)} is unreadable ({e}); "
                  f"deleting it and trying an older checkpoint")
            try:
                os.remove(path)
            except OSError:
                pass
    return None, None


def _peek_extra(component, device):
    """Read-only peek at whether this component's checkpoint carries an
    embedded training-state payload, without applying anything to a module.
    Used only to decide *how* to load a stage (e.g. whether a VAE needs
    attaching first, for a VAE-shaped checkpoint) before the real,
    module-applying load via _load_weights_with_fallback happens -- calling
    that eagerly here would apply state into a not-yet-VAE-attached
    reasoning expert and fail on a genuine VAE-stage resume."""
    for tag in ("latest", "best"):
        path = find_weights(component, tag)
        if not path:
            continue
        try:
            return path, torch.load(path, map_location=device).get("extra")
        except Exception:
            continue
    return None, None


def _prune_other_tagged_files(component, tag, keep_path):
    """See enc_pre.py / text_to_text.py / ocr_to_text.py / image_to_text.py
    for the full rationale (identical): refuses to delete a file
    representing *more* progress than keep_path, loudly warning instead of
    silently destroying it, since that's exactly the failure mode a missed
    resume (stale filesystem view, two jobs on the same stage) can trigger."""
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
    component with its embedded training-state payload removed -- called
    once a stage has genuinely finished, since the optimizer/RNG/counters
    are meaningless past that point but the weights themselves are the
    permanent, reusable artifact and must stay (unlike the old wrapper
    file, which was simply deleted outright). See text_to_text.py's
    identical helper for the full rationale."""
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
    """See any other script's identical function for the traceback that
    motivated this."""
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


def save_checkpoint(prefix, model, optimizer, epoch, batch, global_step, best_val_loss,
                    val_loss=None, epoch_complete=False, rl_baseline=0.0):
    """Persist ``combine``/``router``/``expert_registry`` (the *whole*
    registry -- since ExpertRegistry.experts is an nn.ModuleDict, this
    already recursively includes the reasoning expert's own parameters, VAE
    included once attached) as standalone, directly-loadable files under
    the 'latest' tag, always, and also under 'best' when val_loss improves.
    There is no separate resumable-wrapper file: the optimizer state and
    epoch/batch/step counters are stored as an ``extra`` payload embedded
    in ``{prefix}_expert_registry``'s *'latest' tag only* (see
    ExpertRegistry.store_weights) -- 'best' always stays a clean, minimal
    weights-only file. Also persists two more standalone, permanently-
    reusable artifacts (mirroring how encoders/text_output get their own
    file in every other script) so a later script/stage can load just the
    trained reasoning core or just the text-output head without needing
    the whole registry: ``{prefix}_expert`` (reasoning_expert.store_weights)
    and ``{prefix}_text_output`` (text_output.store_weights) -- these two
    predate this refactor and already had their own store_weights/
    load_weights; nothing else changes about them here. Same for
    ``{prefix}_combine``/``{prefix}_router``, added specifically so the
    reasoning_rl stage can warm-start combine/router from the VAE stage's
    *finished* state instead of restarting them from scratch.

    Stage-prefixed (not a shared "reasoning_expert_registry" name across
    stages) deliberately: the TRM and VAE stages each restart their own
    epoch counter at 0, and _prune_other_tagged_files' resume-miss safety
    check compares (epoch, batch) to decide what's "more progress" -- a
    shared name across two independently-numbered stages would make that
    check misfire (an early VAE-stage checkpoint looking "less advanced"
    than the TRM stage's final one, even though the VAE stage is the one
    that should now be treated as current).

    A save that still fails after atomic_torch_save's own retries (a
    sustained GPFS write outage, not a transient blip) is logged and
    swallowed rather than raised: this is a rank-0-only periodic save, and
    letting it propagate kills the entire distributed job over what's
    recoverable by just trying again at the next checkpoint interval."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    training_state = {
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "batch": batch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "rng": torch.get_rng_state(),
        # Scheduled RL term's moving-average baseline (see
        # HYPERPARAMS["reward"]'s schedule comment / reasoning_rl_term) --
        # resumed so a resubmitted job doesn't lose its variance-reduction
        # warm-up every time.
        "rl_baseline": rl_baseline,
    }
    text_output = model.expert_registry.get_expert("text_output")
    reasoning_expert = model.expert_registry.get_expert("reasoning")

    tags = ["latest"]
    if val_loss is not None and val_loss <= best_val_loss:
        tags.append("best")

    try:
        for tag in tags:
            rp = weight_path(f"{prefix}_expert", tag, epoch, batch, val_loss, epoch_complete=epoch_complete)
            tp = weight_path(f"{prefix}_text_output", tag, epoch, batch, val_loss, epoch_complete=epoch_complete)
            cp = weight_path(f"{prefix}_combine", tag, epoch, batch, val_loss, epoch_complete=epoch_complete)
            rtp = weight_path(f"{prefix}_router", tag, epoch, batch, val_loss, epoch_complete=epoch_complete)
            erp = weight_path(f"{prefix}_expert_registry", tag, epoch, batch, val_loss, epoch_complete=epoch_complete)

            reasoning_expert.store_weights(WEIGHTS_DIR, os.path.basename(rp))
            text_output.store_weights(WEIGHTS_DIR, os.path.basename(tp))
            model.combine.store_weights(WEIGHTS_DIR, os.path.basename(cp))
            model.router.store_weights(WEIGHTS_DIR, os.path.basename(rtp))
            model.expert_registry.store_weights(
                WEIGHTS_DIR, os.path.basename(erp),
                extra=training_state if tag == "latest" else None,
            )

            _prune_other_tagged_files(f"{prefix}_expert", tag, rp)
            _prune_other_tagged_files(f"{prefix}_text_output", tag, tp)
            _prune_other_tagged_files(f"{prefix}_combine", tag, cp)
            _prune_other_tagged_files(f"{prefix}_router", tag, rtp)
            _prune_other_tagged_files(f"{prefix}_expert_registry", tag, erp)
    except (RuntimeError, OSError) as e:
        print(f"[checkpoint] save_checkpoint failed at epoch {epoch} batch {batch} "
              f"step {global_step} even after retries ({e}); skipping this "
              f"checkpoint and continuing training -- will try again at the "
              f"next interval.")


# --------------------------------------------------------------------------- #
# Incremental LR decay (shared design with every other script in this
# project)
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
# Answer normalisation, exact match, close match, ROUGE-L (self-contained,
# no extra deps -- identical to text_to_text.py / image_to_text.py)
# --------------------------------------------------------------------------- #
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction, gold_answers):
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(g) for g in gold_answers if g)


def _try_float(s):
    try:
        return float(str(s).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def close_match(prediction, gold_answers, tol=0.05):
    """See image_to_text.py's close_match for the full rationale (ChartQA's
    relaxed-accuracy scheme, reused here for numinamath's numeric answers)."""
    pred_num = _try_float(prediction)
    if pred_num is not None:
        for g in gold_answers:
            gold_num = _try_float(g)
            if gold_num is None:
                continue
            if gold_num == 0:
                if pred_num == 0:
                    return True
            elif abs(pred_num - gold_num) / abs(gold_num) <= tol:
                return True
    return exact_match(prediction, gold_answers)


def _lcs_len(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_f1(pred_text, ref_text):
    pred, ref = normalize_answer(pred_text).split(), normalize_answer(ref_text).split()
    if not pred or not ref:
        return 0.0
    lcs = _lcs_len(pred, ref)
    if lcs == 0:
        return 0.0
    p, r = lcs / len(pred), lcs / len(ref)
    return 2 * p * r / (p + r)


def compute_reward_metric(source, pred_text, target_text, aux):
    """Task-specific reward/metric, dispatched off the 'source' column.
    Itself non-differentiable (plain string/number comparisons), but IS
    used as a training signal: reasoning_rl's own REINFORCE stage scores a
    sampled generation with it directly, and the TRM/VAE stages' scheduled
    RL term (see reasoning_rl_term / HYPERPARAMS["reward"]'s schedule
    comment) reuses it the same way -- the reward value itself carries no
    gradient (it's a python float), only the sampled generation's
    log-probability does, per the standard REINFORCE/score-function
    estimator. Also used for pure logging/validation (evaluate_rewards)."""
    aliases = [a for a in (aux or "").split("|") if a]
    gold = aliases if aliases else [target_text]
    if source == "numinamath":
        # "Close match or ROUGE-L": try the numeric-tolerance/exact check
        # first; fall back to a continuous ROUGE-L partial-credit score
        # rather than a binary 0/1 for anything close-but-not-matching, so
        # this reward isn't all-or-nothing on free-form math answers.
        if close_match(pred_text, gold):
            return 1.0
        return max((rouge_l_f1(pred_text, g) for g in gold), default=0.0)
    # logi_glue (binary accuracy), geoqa/truthfulqa (exact match), and
    # commonsenseqa (MCQ accuracy -- target is already just the gold
    # option's letter, see _format_mcq/_mcq_answer_letter in
    # image_to_text.py's identical scheme) all collapse to exact_match.
    return 1.0 if exact_match(pred_text, gold) else 0.0


# --------------------------------------------------------------------------- #
# Target tokenisation (reuse the project's shared SentencePiece model)
# --------------------------------------------------------------------------- #
class SpmCodec:
    def __init__(self, model_path=TOKENIZER_PATH, max_len=32):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.max_len = max_len

    def encode_target(self, text, device):
        ids = self.sp.encode_as_ids(text)[: self.max_len - 2]
        ids = [BOS_ID] + ids + [EOS_ID]
        return torch.tensor([ids], dtype=torch.long, device=device)


# --------------------------------------------------------------------------- #
# Row helpers + image loading + MCQ formatting (mirrors image_to_text.py)
# --------------------------------------------------------------------------- #
def _truncate_words(text, max_words):
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def _mkrow(question, answers, source, image=None, row_id=None):
    question = _truncate_words(str(question).strip(), HYPERPARAMS["global"]["max_input_len"])
    answers = [str(a).strip() for a in answers if a is not None and str(a).strip()]
    if not question or not answers:
        return None
    return {"input": question, "target": answers[0], "source": source,
           "aux": "|".join(dict.fromkeys(answers)), "image": image,
           "row_id": row_id, "trajectory": "", "traj_answer": ""}


_MCQ_LETTERS = "ABCDEFGH"


def _format_mcq(question, choices):
    opts = "\n".join(f"{_MCQ_LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return f"{question}\n{opts}\nAnswer with the option's letter."


def _mcq_answer_letter(answer, choices):
    """See image_to_text.py's identical function for the full rationale."""
    n = len(choices)
    if isinstance(answer, str):
        stripped = answer.strip()
        if len(stripped) == 1 and stripped.upper() in _MCQ_LETTERS[:n]:
            return stripped.upper()
        for i, c in enumerate(choices):
            if str(c).strip().lower() == stripped.lower():
                return _MCQ_LETTERS[i]
        return None
    try:
        idx = int(answer)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < n:
        return _MCQ_LETTERS[idx]
    if 1 <= idx <= n:
        return _MCQ_LETTERS[idx - 1]
    return None


def _load_image_tensor(img, size):
    """See image_to_text.py's identical function for the full rationale."""
    from PIL import Image
    import numpy as np

    img = img.convert("RGB").resize((size, size))
    arr = torch.from_numpy(np.asarray(img, dtype="float32") / 255.0)
    return arr.permute(2, 0, 1).unsqueeze(0)


def sample_rows(rows, n, seed):
    rng = random.Random(seed)
    if n >= len(rows):
        return list(rows)
    return rng.sample(rows, n)


# --------------------------------------------------------------------------- #
# Per-source loaders. Each is wrapped in compile_dataset()'s per-source
# try/except (see SOURCE_LOADERS) so one bad/renamed dataset id or schema
# guess doesn't take the other sources down with it.
# --------------------------------------------------------------------------- #
def load_logi_glue(cap):
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
                row = _mkrow(inp, [answer], "logi_glue")
                if row:
                    rows.append(row)
                    n += 1
                if cap is not None and len(rows) >= cap:
                    break
            print(f"[data]   logi_glue/{cfg}[{split}]: {n} rows")
            if cap is not None and len(rows) >= cap:
                break
        if cap is not None and len(rows) >= cap:
            break
    if not rows:
        raise RuntimeError("logi_glue yielded no usable rows")
    return rows


def load_numinamath(cap):
    from datasets import load_dataset

    ds = load_dataset(NUMINAMATH_HF, split="train", streaming=True, cache_dir=HF_CACHE)
    rows = []
    for r in ds:
        question = r.get("problem") or r.get("question")
        answer = r.get("solution")
        if not question or not answer:
            continue
        row = _mkrow(question, [answer], "numinamath")
        if row:
            rows.append(row)
        if cap is not None and len(rows) >= cap:
            break
    if not rows:
        raise RuntimeError("numinamath yielded no usable rows")
    return rows


def _download_geometry3k():
    """See image_to_text.py's identical function for the full rationale
    (including the proxy-filtering fallback -- scp train.zip/val.zip/
    test.zip directly to DATA_DIR/geometry3k_raw/ if gdown can't get
    through a restrictive cluster proxy) -- duplicated here (own DATA_DIR)
    per this project's self-contained-scripts convention."""
    raw_dir = os.path.join(DATA_DIR, "geometry3k_raw")
    os.makedirs(raw_dir, exist_ok=True)
    already_extracted = any(os.path.isdir(os.path.join(raw_dir, s))
                            for s in ("train", "val", "test"))
    if not already_extracted:
        missing = [s for s in ("train", "val", "test")
                  if not os.path.isfile(os.path.join(raw_dir, f"{s}.zip"))]
        if missing:
            import gdown
            gdown.download_folder(id=GEOMETRY3K_GDRIVE_FOLDER_ID, output=raw_dir,
                                  quiet=False, use_cookies=False)
        import zipfile
        for split in ("train", "val", "test"):
            zpath = os.path.join(raw_dir, f"{split}.zip")
            if os.path.isfile(zpath):
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(raw_dir)
    return raw_dir


def load_geoqa(cap):
    """Geometry3K -- see image_to_text.py's identical load_geoqa for the
    full schema rationale (confirmed against the InterGPS repo's own
    annotation-tool example data.json). Formatted as an MCQ (_format_mcq/
    _mcq_answer_letter, same as this file's truthfulqa/commonsenseqa use
    elsewhere) since Geometry3K's answer is always one of 4 lettered
    choices. image=None isn't possible here (unlike this function's other
    callers) -- every problem folder has an image, rows without one are
    skipped outright rather than falling back to text-only."""
    from PIL import Image as PILImage

    raw_dir = _download_geometry3k()
    rows = []
    for split in ("train", "val", "test"):
        split_dir = os.path.join(raw_dir, split)
        if not os.path.isdir(split_dir):
            continue
        n = 0
        for problem_id in sorted(os.listdir(split_dir)):
            problem_dir = os.path.join(split_dir, problem_id)
            data_path = os.path.join(problem_dir, "data.json")
            if not os.path.isdir(problem_dir) or not os.path.isfile(data_path):
                continue
            with open(data_path) as f:
                d = json.load(f)
            question, choices, answer = d.get("problem_text"), d.get("choices"), d.get("answer")
            if not question or not choices or not answer:
                continue
            img_path = next((p for name in ("img_problem.png", "img_diagram.png")
                            if os.path.isfile(p := os.path.join(problem_dir, name))), None)
            if img_path is None:
                continue
            letter = _mcq_answer_letter(answer, choices)
            if letter is None:
                continue
            img = PILImage.open(img_path).convert("RGB")
            row = _mkrow(_format_mcq(question, choices), [letter], "geoqa", image=img)
            if row:
                rows.append(row)
                n += 1
            if cap is not None and len(rows) >= cap:
                break
        print(f"[data]   geoqa (Geometry3K)/{split}: {n} rows")
        if cap is not None and len(rows) >= cap:
            break
    if not rows:
        raise RuntimeError("geoqa yielded no usable rows")
    return rows


def load_truthfulqa(cap):
    from datasets import load_dataset

    ds = load_dataset(TRUTHFULQA_HF, TRUTHFULQA_CONFIG, split="validation", cache_dir=HF_CACHE)
    rows = []
    for r in ds:
        question = r.get("question")
        best = r.get("best_answer")
        aliases = r.get("correct_answers") or []
        if not question or not best:
            continue
        row = _mkrow(question, [best] + list(aliases), "truthfulqa")
        if row:
            rows.append(row)
        if cap is not None and len(rows) >= cap:
            break
    if not rows:
        raise RuntimeError("truthfulqa yielded no usable rows")
    return rows


def load_commonsenseqa(cap):
    from datasets import load_dataset

    rows = []
    for split in ("train", "validation"):
        try:
            ds = load_dataset(COMMONSENSEQA_HF, split=split, streaming=True, cache_dir=HF_CACHE)
        except Exception as e:
            print(f"[data] SKIP commonsenseqa split '{split}' ({e})")
            continue
        n = 0
        for r in ds:
            question = r.get("question")
            choices = r.get("choices") or {}
            # tau/commonsense_qa's "choices" is {"label": [...], "text": [...]}
            # (parallel lists), not a plain list like image_to_text.py's
            # ai2d/scienceqa -- handled specifically here.
            labels, texts = choices.get("label") or [], choices.get("text") or []
            answer_key = r.get("answerKey")
            if not question or not texts or not labels or not answer_key:
                continue
            if answer_key not in labels:
                continue
            gold_idx = labels.index(answer_key)
            letter = _MCQ_LETTERS[gold_idx] if gold_idx < len(_MCQ_LETTERS) else None
            if letter is None:
                continue
            row = _mkrow(_format_mcq(question, texts), [letter], "commonsenseqa")
            if row:
                rows.append(row)
                n += 1
            if cap is not None and len(rows) >= cap:
                break
        print(f"[data]   commonsenseqa[{split}]: {n} rows")
        if cap is not None and len(rows) >= cap:
            break
    if not rows:
        raise RuntimeError("commonsenseqa yielded no usable rows")
    return rows


SOURCE_LOADERS = {
    "logi_glue": lambda cfg: load_logi_glue(cfg["data"]["logi_glue_cap"]),
    "numinamath": lambda cfg: load_numinamath(cfg["data"]["numinamath_cap"]),
    "geoqa": lambda cfg: load_geoqa(cfg["data"]["geoqa_cap"]),
    "truthfulqa": lambda cfg: load_truthfulqa(cfg["data"]["truthfulqa_cap"]),
    "commonsenseqa": lambda cfg: load_commonsenseqa(cfg["data"]["commonsenseqa_cap"]),
}


# --------------------------------------------------------------------------- #
# Compile (download + normalise + shuffle) and split (train/val, stratified
# per source -- like text_to_text.py/image_to_text.py)
# --------------------------------------------------------------------------- #
def compile_dataset():
    if os.path.isdir(COMPILED_DIR) and os.listdir(COMPILED_DIR):
        print(f"[data] compiled dataset already present at {COMPILED_DIR}")
        return
    os.makedirs(DATA_DIR, exist_ok=True)

    all_rows = []
    for name, loader in SOURCE_LOADERS.items():
        try:
            rows = loader(HYPERPARAMS)
            print(f"[data] {name}: {len(rows)} rows")
            all_rows.extend(rows)
        except Exception as e:
            print(f"[data] SKIP source '{name}' -- could not prepare ({e})")

    if not all_rows:
        raise RuntimeError("No reasoning_to_text data sources could be prepared.")

    # Every source must be classified as reasoning-expert-needing or not --
    # an unclassified source would silently get needs_reasoning=False
    # everywhere (REASONING_SOURCES membership is the only check used at
    # training time), which is a much quieter failure than this explicit
    # check catching it now, at compile time.
    unclassified = {r["source"] for r in all_rows} - REASONING_SOURCES - NO_REASONING_SOURCES
    if unclassified:
        raise RuntimeError(
            f"Source(s) {sorted(unclassified)} aren't in REASONING_SOURCES or "
            f"NO_REASONING_SOURCES -- classify them before training (see module "
            f"docstring's task list).")

    # Inbuilt shuffling of the entire compiled dataset (interleaves sources).
    random.Random(HYPERPARAMS["global"]["seed"]).shuffle(all_rows)

    # row_id assigned once, after the final shuffle, and never reassigned --
    # traj_gen's resumable JSONL sidecar keys on this, so it must stay
    # stable for as long as the compiled dataset itself does (it's only
    # ever regenerated if COMPILED_DIR is deleted).
    for i, row in enumerate(all_rows):
        row["row_id"] = i

    from datasets import Dataset
    Dataset.from_list(all_rows).save_to_disk(COMPILED_DIR)
    print(f"[data] wrote {len(all_rows)} compiled rows -> {COMPILED_DIR}")


def split_dataset():
    if os.path.isdir(TRAIN_DIR) and os.path.isdir(VAL_DIR):
        print("[data] train/val split already present")
        return

    from datasets import Dataset, load_from_disk

    rows = list(load_from_disk(COMPILED_DIR))
    rows = _merge_trajectories(rows)
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
    Dataset.from_list(train_rows).save_to_disk(TRAIN_DIR)
    Dataset.from_list(val_rows).save_to_disk(VAL_DIR)
    print(f"[data] split -> train={len(train_rows)} val={len(val_rows)}")


def load_rows(path):
    from datasets import load_from_disk
    return list(load_from_disk(path))


def ensure_dataset_ready():
    compile_dataset()
    split_dataset()


# --------------------------------------------------------------------------- #
# Trajectory generation (Qwen2.5-VL-7B) -- offline, resumable via a JSONL
# sidecar keyed by row_id. Runs as its own stage (traj_gen), fully separate
# from the training model: Qwen is loaded, used, and released (del +
# empty_cache) before this stage returns, so it never coexists in memory
# with the training model in any later stage.
# --------------------------------------------------------------------------- #
def _load_completed_trajectory_ids():
    completed = {}
    if os.path.isfile(TRAJECTORY_JSONL):
        with open(TRAJECTORY_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue    # a partially-flushed last line from a killed job -- skip it
                completed[rec["row_id"]] = rec
    return completed


def _append_trajectory(row_id, trajectory, traj_answer):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TRAJECTORY_JSONL, "a") as f:
        f.write(json.dumps({"row_id": row_id, "trajectory": trajectory,
                           "traj_answer": traj_answer}) + "\n")


def _merge_trajectories(rows):
    """Join the JSONL sidecar's generated trajectories back onto compiled
    rows by row_id -- called by split_dataset() so train/val rows carry
    whatever trajectories exist at split time (a resubmitted traj_gen job
    that generates more afterward has no effect on an already-written
    split; re-run split_dataset by clearing TRAIN_DIR/VAL_DIR if that
    matters for a given run)."""
    completed = _load_completed_trajectory_ids()
    if not completed:
        return rows
    for row in rows:
        rec = completed.get(row["row_id"])
        if rec:
            row["trajectory"] = rec["trajectory"]
            row["traj_answer"] = rec["traj_answer"]
    return rows


def _load_qwen(model_id):
    """Requires a transformers version with Qwen2.5-VL support
    (Qwen2_5_VLForConditionalGeneration) -- if this import fails, traj_gen
    reports it clearly rather than crashing on some other cryptic path."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def _qwen_generate_trajectory(model, processor, question, image, max_new_tokens):
    content = []
    if image is not None:
        content.append({"type": "image"})
    content.append({
        "type": "text",
        "text": (f"{question}\n\nThink step by step, then give your final answer "
                 f"on its own last line as 'Final Answer: <answer>'."),
    })
    messages = [{"role": "user", "content": content}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [image.convert("RGB")] if image is not None else None
    inputs = processor(text=[text_prompt], images=images, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_only = generated[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(gen_only, skip_special_tokens=True)[0]

    if "Final Answer:" in output_text:
        chain, _, answer = output_text.rpartition("Final Answer:")
    else:
        chain, answer = output_text, output_text
    return chain.strip(), answer.strip()


def generate_trajectories(device):
    print("\n===== TRAJ_GEN: reasoning trajectories (Qwen2.5-VL-7B) =====")
    ensure_dataset_ready()
    from datasets import load_from_disk

    rows = list(load_from_disk(COMPILED_DIR))
    reasoning_rows = [r for r in rows if r["source"] in REASONING_SOURCES]
    print(f"[traj_gen] {len(reasoning_rows)} row(s) need a trajectory "
          f"(sources: {sorted(REASONING_SOURCES)})")

    already_done = _load_completed_trajectory_ids()
    remaining = [r for r in reasoning_rows if r["row_id"] not in already_done]
    print(f"[traj_gen] {len(already_done)} already generated (resuming); "
          f"{len(remaining)} remaining")
    if not remaining:
        print("[traj_gen] nothing left to generate.")
        return

    tcfg = HYPERPARAMS["trajectory"]
    model, processor = _load_qwen(tcfg["qwen_model_id"])
    try:
        for i, row in enumerate(remaining):
            chain, answer = _qwen_generate_trajectory(
                model, processor, row["input"], row["image"], tcfg["max_new_tokens"],
            )
            _append_trajectory(row["row_id"], chain, answer)
            if (i + 1) % 50 == 0:
                print(f"  [traj_gen] {i + 1}/{len(remaining)} generated "
                      f"(row_id={row['row_id']}, source={row['source']})")
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"[traj_gen] done -- {len(remaining)} trajectory(ies) written to "
          f"{TRAJECTORY_JSONL}")


# --------------------------------------------------------------------------- #
# Frozen encoders (loaded from enc_pre.py's best checkpoints -- identical to
# image_to_text.py's build_model, just also instantiating the reasoning
# expert, which needs text_only=False)
# --------------------------------------------------------------------------- #
class FrozenEncoderWrapper(nn.Module):
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

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self


class FrozenTextEncoderWrapper(FrozenEncoderWrapper):
    def forward_tokens(self, *args, **kwargs):
        return self.encoder.forward_tokens(*args, **kwargs)


def build_model(device, d_model, half_precision):
    from models.OverallModel import MultimodalModel
    from models.Encoders.TextEncoder import createTextEnc
    from models.Encoders.VisionEncoder import createVisionEncoder

    # text_only=False -- unlike every other *_to_text.py script, this one
    # needs the reasoning expert (and, harmlessly, gets local_mem/
    # internet_search too -- both are lightweight nn.Module skeletons with
    # no eager external-model loading in their constructors, verified by
    # reading LocalMem.py/InternetSearch.py directly before relying on this;
    # their "Qwen 1.5B" mentions are docstring-only future-work notes, not
    # instantiated code).
    model = MultimodalModel(dim=d_model, text_only=False).to(device)

    # ---- Frozen text encoder (enc_pre.py's text_train stage) --------------- #
    raw_text = createTextEnc(dim=d_model).to(device)
    best_text = find_weights("text_encoder", "best") or find_weights("text_encoder", "latest")
    if best_text:
        raw_text.load_weights(best_text, map_location=device)
        print(f"[reasoning] frozen text encoder loaded from {os.path.basename(best_text)}")
    else:
        print("[reasoning] WARNING: no trained text encoder found via optimal grep in "
              f"{WEIGHTS_DIR}; using a randomly initialised (untrained) text encoder")
    raw_text.eval()
    for p in raw_text.parameters():
        p.requires_grad_(False)
    frozen_text = FrozenTextEncoderWrapper(raw_text, half_precision).to(device)
    model.encoder_registry.remove_encoder("text")
    model.encoder_registry.add_encoder("text", frozen_text)

    # ---- Frozen vision encoder (enc_pre.py's image_train stage) ------------ #
    # text_only=False already auto-created a *trainable* vision encoder
    # (OverallModel.py); swap it for the frozen pretrained one, same
    # remove-then-add pattern as the text encoder above.
    raw_vision = createVisionEncoder(dim=d_model).to(device)
    best_vision = find_weights("image_encoder", "best") or find_weights("image_encoder", "latest")
    if best_vision:
        raw_vision.load_weights(best_vision, map_location=device)
        print(f"[reasoning] frozen vision encoder loaded from {os.path.basename(best_vision)}")
    else:
        print("[reasoning] WARNING: no trained vision encoder found via optimal grep in "
              f"{WEIGHTS_DIR}; using a randomly initialised (untrained) vision encoder")
    raw_vision.eval()
    for p in raw_vision.parameters():
        p.requires_grad_(False)
    frozen_vision = FrozenEncoderWrapper(raw_vision, half_precision).to(device)
    model.encoder_registry.remove_encoder("vision")
    model.encoder_registry.add_encoder("vision", frozen_vision)

    # ---- Warm-start text_output/combine/router from this script's own ------ #
    # ---- furthest-completed stage (or *_to_text.py, for text_output) ------- #
    # Priority order for each: the furthest-along stage's own saved
    # artifact first (reasoning_rl > reasoning_vae > reasoning_trm, each
    # stage's save only exists once that stage has actually run).
    # text_output additionally falls back to the original
    # text_to_text.py/image_to_text.py/ocr_to_text.py artifact when none of
    # this script's own stages have saved one yet; combine/router have no
    # such fallback (see save_checkpoint's docstring: no other script in
    # this project ever gave them a standalone artifact, this one is the
    # first). reasoning_expert is never warm-started here -- it's brand new
    # everywhere in this repo for the TRM stage's fresh start, and the VAE/
    # RL stages each warm-start it explicitly themselves (see train() /
    # reasoning_rl()) since which checkpoint is correct depends on whether
    # a VAE needs attaching first.
    def _warm_start(module, component_names, label):
        # Same corrupt-file tolerance as _load_weights_with_fallback (a
        # truncated write from a flaky GPFS save shouldn't crash every
        # subsequent run that warm-starts from it) -- "best" then "latest"
        # per candidate name, falling through to the next name in priority
        # order only once both tags for this one are exhausted.
        for name in component_names:
            for tag in ("best", "latest"):
                path = find_weights(name, tag)
                if not path:
                    continue
                try:
                    module.load_weights(path, map_location=device)
                    print(f"[reasoning] warm-started {label} from {os.path.basename(path)}")
                    return
                except Exception as e:
                    print(f"[reasoning] {os.path.basename(path)} is unreadable ({e}); "
                          f"deleting it and trying an older checkpoint")
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        print(f"[reasoning] no {label} checkpoint found to warm-start from; "
              "using a fresh (randomly initialised) one")

    text_output = model.expert_registry.get_expert("text_output")
    _warm_start(text_output, [
        "reasoning_rl_text_output", "reasoning_vae_text_output", "reasoning_trm_text_output",
        "t2t_text_output", "img2t_text_output", "ocr2t_text_output",
    ], "text_output")
    _warm_start(model.combine, [
        "reasoning_rl_combine", "reasoning_vae_combine", "reasoning_trm_combine",
    ], "combine")
    _warm_start(model.router, [
        "reasoning_rl_router", "reasoning_vae_router", "reasoning_trm_router",
    ], "router")

    model.expert_registry.set_pipeline(
        text_encoder=frozen_text, combiner=model.combine, router=model.router,
    )
    model = model.to(device)
    return model


# --------------------------------------------------------------------------- #
# Core forward helpers
# --------------------------------------------------------------------------- #
def encode_to_state(model, question, image_tensor=None):
    """Unlike the other *_to_text.py scripts' encode_to_state, this one
    also returns route_vec (needed for the selection loss) and supports an
    optional image (most sources here are text-only; only geoqa carries an
    image -- see REASONING_SOURCES/load_geoqa)."""
    encodings = model.encode(image_tensor, "vision") if image_tensor is not None else []
    encodings = model.encode(question, "text", encodings)
    combined, _, _ = model.combine(encodings)
    route_vec, routed_state = model.router(combined)
    return route_vec, routed_state


def koleo_loss(embeddings, chunk_size=16, eps=1e-8):
    """Sablayrolles et al. 2018: encourages embeddings within a (small) batch to
    spread out uniformly via -log(distance to nearest neighbour). Computed in
    chunks of ``chunk_size`` (DINOv3's "small batches of 16"). Identical to
    enc_pre.py's koleo_loss (already validated there for the vision SSL
    pretraining pipeline) and text_to_text.py's (validated there against the
    real "always True" collapse -- see HYPERPARAMS["reward"]["koleo_coef"]'s
    comment) -- duplicated here rather than imported, matching this codebase's
    convention of each *_to_text.py script staying self-contained and
    independently launchable.

    Applied to combine/router's raw pooled output specifically (encode_to_state's
    own return value, before any reasoning-expert transformation) -- the exact
    component proven in text_to_text.py to collapse toward a near-identical
    direction across every input during real training, regardless of any
    decoder-side mechanism, unless directly regularized at the source."""
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


def teacher_forced_forward(expert, state, tgt_in):
    device = tgt_in.device
    T = tgt_in.size(1)
    emb = expert.token_embedding(tgt_in) * math.sqrt(expert.d_model)
    emb = emb + expert._sincos_pos_embed(T, device)
    tgt_mask = expert._causal_mask(T, device)
    weighted_h, ponder_cost = expert._recursive_decode(emb, state, None, tgt_mask)
    logits = expert.output_proj(weighted_h)
    return logits, ponder_cost


def selection_loss(model, route_vec, needs_reasoning):
    """Pulls route_vec closer to the correct expert's embedding, and (via
    the softmax's normalisation over every OTHER expert) further from every
    incorrect one -- exactly "closer if correct, further if incorrect" from
    the task spec, implemented as a standard cross-entropy over
    expert_registry.score(route_vec) (the similarity mechanism that already
    existed, see module docstring) against the row's known-correct expert
    index. Bypasses the non-differentiable hard-argmax select()/fuse_route()
    path entirely -- score() alone is what needs to be differentiable here."""
    registry = model.expert_registry
    scores = registry.score(route_vec)                      # [B, N]
    target_name = "reasoning" if needs_reasoning else "text_output"
    target_idx = registry.expert_names.index(target_name)
    target = torch.full((route_vec.size(0),), target_idx, dtype=torch.long, device=route_vec.device)
    return F.cross_entropy(scores, target)


def trajectory_imitation_loss(model, reasoning_state, trajectory_text, traj_answer,
                              gold_target, coefs, max_trajectory_len):
    """SRL supervised term. "Error checking for plain text is a
    straightforward comparison with ROUGE-L": traj_answer (Qwen's extracted
    final answer) vs. the gold target gates whether this trajectory is
    trusted at all -- a trajectory whose own final answer is wrong isn't
    something the model should imitate. "The given reasoning chain is
    processed through model's text encoder and compared with cosine
    similarity to internal hidden state": the chain text goes through the
    same frozen text encoder used everywhere else in this pipeline (not the
    full encode-and-route path -- just the raw encoder, to avoid mixing in
    the learned "text" modality embedding, which isn't part of what's being
    compared here), mean-pooled, and cosine-compared against the
    reasoning expert's own output state (also mean-pooled). Returns None
    when there's no trajectory to imitate, or it fails the quality gate --
    callers should skip this term entirely in that case, not treat None as
    zero (a genuinely absent term, not a zero-loss one)."""
    if not trajectory_text:
        return None
    quality = rouge_l_f1(traj_answer or "", gold_target)
    if quality < coefs["trajectory_quality_threshold"]:
        return None
    text_encoder = model.encoder_registry.get_encoder("text")
    chain_input = trajectory_text
    # Word-level truncation only (cheap, no tokenizer round-trip needed just
    # to bound the frozen encoder's input length) -- the encoder itself
    # handles arbitrary token counts, this just keeps the imitation pass
    # cheap on very long Qwen chains.
    chain_input = " ".join(chain_input.split()[:max_trajectory_len])
    chain_tokens = text_encoder(chain_input)         # [1, T, D]
    chain_repr = chain_tokens.mean(dim=1)             # [1, D]
    state_repr = reasoning_state.mean(dim=1)          # [1, D]
    sim = F.cosine_similarity(chain_repr, state_repr, dim=-1)
    return (1.0 - sim).mean()


def compute_task_loss(model, expert_registry, state, route_vec, target_text, codec, device,
                      coefs, label_smoothing, needs_reasoning, trajectory_text=None,
                      traj_answer=None, max_trajectory_len=256):
    """Composite training objective. Unlike the other three *_to_text.py
    scripts' compute_task_loss, this one also folds in: the expert-selection
    loss (always), and -- only when the reasoning expert was actually used
    -- its ACT ponder cost ("number of iterations"), a flat "expert called"
    cost ("requiring experts called"), its VAE KL term (harmlessly 0 when no
    VAE is attached), and the SRL trajectory-imitation term (when a
    trajectory is available and passes its quality gate)."""
    expert = expert_registry.get_expert("text_output")
    tgt = codec.encode_target(target_text, device)
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
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

    # Blank-answer / repetition penalties -- see text_to_text.py's
    # compute_task_loss for the full rationale (identical logic): both
    # computed from the same teacher-forced logits, no extra forward pass.
    eos_prob_at_start = probs[:, 0, EOS_ID]
    gold_not_empty = (tgt_out[:, 0] != EOS_ID).float()
    blank_penalty = (eos_prob_at_start * gold_not_empty).mean()

    repeat_prob = probs.gather(-1, tgt_in.unsqueeze(-1)).squeeze(-1)
    gold_is_repeat = (tgt_out == tgt_in).float()
    repetition_penalty = ((repeat_prob * (1 - gold_is_repeat)) * valid).sum() / valid.sum().clamp(min=1)

    loss = (
        coefs["task_loss_coef"] * ce
        + coefs["length_penalty_coef"] * length_term
        + coefs["ponder_cost_coef"] * ponder_term
        + coefs["brier_coef"] * brier
        + coefs["blank_answer_penalty_coef"] * blank_penalty
        + coefs["repetition_penalty_coef"] * repetition_penalty
        + coefs["selection_coef"] * selection_loss(model, route_vec, needs_reasoning)
    )

    if needs_reasoning:
        reasoning_expert = expert_registry.get_expert("reasoning")
        reasoning_ponder = reasoning_expert.last_ponder_cost_raw.mean() / reasoning_expert.max_steps
        loss = loss + coefs["reasoning_ponder_coef"] * reasoning_ponder
        loss = loss + coefs["expert_call_coef"]
        if reasoning_expert.last_kl_loss_raw is not None:
            loss = loss + coefs["kl_coef"] * reasoning_expert.last_kl_loss_raw

    return loss, ce.detach().item()


# --------------------------------------------------------------------------- #
# Supervised/RL loss schedule (TRM + VAE stages only -- see
# HYPERPARAMS["reward"]'s "Supervised/RL loss schedule" comment for the full
# rationale). reasoning_rl_term reuses reasoning_rl's exact single-sample
# REINFORCE mechanism (sampled generation + compute_reward_metric + a
# moving-average baseline), just computed inline per training row instead of
# in its own separate stage.
# --------------------------------------------------------------------------- #
def _schedule_weights(global_step, total_steps, coefs):
    """Linearly interpolate (supervised_weight, rl_weight) by training
    progress -- p=0 at the very first step, p=1 once global_step reaches
    total_steps (and clamped there for any step beyond, in case of
    off-by-one rounding in total_steps' own estimate)."""
    p = min(1.0, global_step / max(1, total_steps))
    sup_w = coefs["supervised_coef_start"] + p * (coefs["supervised_coef_end"] - coefs["supervised_coef_start"])
    rl_w = coefs["rl_coef_start"] + p * (coefs["rl_coef_end"] - coefs["rl_coef_start"])
    return sup_w, rl_w


def reasoning_rl_term(expert, state, row):
    """Single-sample REINFORCE term for one training row: sample a
    generation from the current policy (gradients flow through its
    log-probability, not through the discrete sampling choice itself --
    the standard score-function estimator, identical to reasoning_rl's own
    usage), score it with the same task reward metric used everywhere else
    in this file, and return (reward, logprob) for the caller to combine
    with a baseline. Reward is a plain float (no grad needed on it) --
    only logprob needs to stay a tensor."""
    gen, _, logprob = expert.forward(
        state, max_length=HYPERPARAMS["global"]["max_target_len"], return_logprob=True,
    )
    pred = gen[0] if gen else ""
    reward = compute_reward_metric(row["source"], pred, row["target"], row.get("aux", ""))
    return reward, logprob.mean()


# --------------------------------------------------------------------------- #
# Self-correction: generate, intentionally corrupt, then train the model to
# catch its own mistake (see text_to_text.py's module docstring). Routes
# through the reasoning expert first when the row needs it, matching real
# inference behaviour.
# --------------------------------------------------------------------------- #
_DECOY_WORDS = ["wrong", "different", "unrelated", "incorrect", "mismatched"]


def corrupt_answer(text, rng):
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


def _reasoned_state(model, question, image_tensor, needs_reasoning):
    route_vec, state = encode_to_state(model, question, image_tensor)
    if needs_reasoning:
        _, state = model.expert_registry.get_expert("reasoning")(state)
    return route_vec, state


def self_correct_step(model, codec, device, question, image_tensor, gold_answer,
                      needs_reasoning, coefs, rng):
    expert = model.expert_registry.get_expert("text_output")
    with torch.no_grad():
        _, state = _reasoned_state(model, question, image_tensor, needs_reasoning)
        gen_text, _ = expert.forward(state, max_length=HYPERPARAMS["global"]["max_target_len"])
        candidate = (gen_text[0] if gen_text else "").strip() or gold_answer

    corrupted = rng.random() < 0.5
    shown = corrupt_answer(candidate, rng) if corrupted else candidate
    verify_input = (
        f"Question: {question}\nProposed answer: {shown}\n"
        f"Is the proposed answer correct? Reply with exactly 'yes' or 'no'."
    )
    verify_target = "no" if corrupted else "yes"

    _, v_state = _reasoned_state(model, verify_input, image_tensor, needs_reasoning)
    route_vec, _ = encode_to_state(model, verify_input, image_tensor)
    loss, _ = compute_task_loss(
        model, model.expert_registry, v_state, route_vec, verify_target, codec, device,
        coefs, label_smoothing=0.0, needs_reasoning=needs_reasoning,
    )
    return loss


# --------------------------------------------------------------------------- #
# Validation (teacher-forced CE, drives checkpoint/valloss) + reward metrics
# (free-run generation, logging only)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, codec, device, val_rows, coefs, sample_size, image_size):
    model.eval()
    rows = sample_rows(val_rows, min(sample_size, len(val_rows)), seed=0)
    total_ce, n = 0.0, 0
    for r in rows:
        needs_reasoning = r["source"] in REASONING_SOURCES
        image_tensor = (_load_image_tensor(r["image"], image_size).to(device)
                        if r.get("image") is not None else None)
        route_vec, state = _reasoned_state(model, r["input"], image_tensor, needs_reasoning)
        _, ce = compute_task_loss(model, model.expert_registry, state, route_vec, r["target"],
                                  codec, device, coefs, label_smoothing=0.0,
                                  needs_reasoning=needs_reasoning)
        total_ce += ce
        n += 1
    model.train()
    return total_ce / max(1, n)


@torch.no_grad()
def evaluate_rewards(model, val_rows, sample_size, seed, image_size, device):
    model.eval()
    expert = model.expert_registry.get_expert("text_output")
    rows = sample_rows(val_rows, min(sample_size, len(val_rows)), seed=seed)
    by_source = {}
    # Print one real (prediction, target) pair per source the first time
    # that source comes up this call -- see text_to_text.py's identical
    # addition for why: an aggregate reward stuck near 0 is ambiguous
    # (genuinely-hard task vs. degenerate generation) without actually
    # looking at what's being generated.
    logged_sources = set()
    for r in rows:
        needs_reasoning = r["source"] in REASONING_SOURCES
        image_tensor = (_load_image_tensor(r["image"], image_size).to(device)
                        if r.get("image") is not None else None)
        _, state = _reasoned_state(model, r["input"], image_tensor, needs_reasoning)
        # Greedy (top_k=1), not the default sampling used for training-time
        # exploration -- see text_to_text.py's evaluate_rewards for the full
        # rationale.
        gen, _ = expert.forward(state, max_length=HYPERPARAMS["global"]["max_target_len"],
                                top_k=1)
        pred = gen[0] if gen else ""
        reward = compute_reward_metric(r["source"], pred, r["target"], r.get("aux", ""))
        by_source.setdefault(r["source"], []).append(reward)
        if r["source"] not in logged_sources:
            logged_sources.add(r["source"])
            print(f"  [reward-sample] {r['source']}: pred={pred!r} target={r['target']!r} reward={reward:.3f}")
    model.train()
    return {s: sum(v) / len(v) for s, v in by_source.items()}


# --------------------------------------------------------------------------- #
# Optuna hyperparameter tuning -- ONE study, reused for both the TRM and VAE
# training stages (see module docstring).
# --------------------------------------------------------------------------- #
def _is_oom_error(e):
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(e, oom_cls):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def tune(device):
    import optuna

    print("\n===== TUNING: reasoning_to_text =====")
    ensure_dataset_ready()
    train_rows = load_rows(TRAIN_DIR)
    val_rows = load_rows(VAL_DIR)
    gcfg = HYPERPARAMS["global"]
    tune_train = sample_rows(train_rows, gcfg["tune_train_subset"], seed=1)
    tune_val = sample_rows(val_rows, gcfg["tune_val_subset"], seed=2)
    print(f"[data] tuning on {len(tune_train)} train / {len(tune_val)} val rows")

    codec = SpmCodec(max_len=gcfg["max_target_len"])
    space = HYPERPARAMS["search_space"]
    budget = HYPERPARAMS["tune_budget_batches"]
    d_model = gcfg["d_model"]
    image_size = gcfg["image_size"]
    coefs = HYPERPARAMS["reward"]

    os.makedirs(STUDY_DIR, exist_ok=True)
    study = optuna.create_study(
        study_name="reasoning_to_text",
        storage=f"sqlite:///{os.path.join(STUDY_DIR, 'reasoning_to_text.db')}",
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=0),
        load_if_exists=True,
    )

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
                needs_reasoning = row["source"] in REASONING_SOURCES
                image_tensor = (_load_image_tensor(row["image"], image_size).to(device)
                                if row.get("image") is not None else None)
                route_vec, state = _reasoned_state(model, row["input"], image_tensor, needs_reasoning)
                loss, _ = compute_task_loss(model, model.expert_registry, state, route_vec,
                                            row["target"], codec, device, coefs,
                                            hp["label_smoothing"], needs_reasoning)
                (loss / hp["batch_size"]).backward()
                accum += 1
                if accum == hp["batch_size"]:
                    optimizer.step()
                    optimizer.zero_grad()
                    accum = 0

            val_loss = validate(model, codec, device, tune_val, coefs,
                                sample_size=len(tune_val), image_size=image_size)
        except Exception as e:
            if not _is_oom_error(e):
                raise
            known_oom_batch_size = min(known_oom_batch_size or hp["batch_size"], hp["batch_size"])
            print(f"  trial {trial.number}: OOM at batch_size={hp['batch_size']} "
                  f"({e}); pruning this trial and continuing with the next one.")
            raise optuna.TrialPruned()
        finally:
            del model, optimizer
            if device == "cuda":
                torch.cuda.empty_cache()

        print(f"  trial {trial.number}: val_loss={val_loss:.4f}  hp={hp}")
        wandb_log({"tune/val_loss": val_loss, **{f"tune/{k}": v for k, v in hp.items()}})
        return val_loss

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
    out = os.path.join(STUDY_DIR, "reasoning_to_text_best_hparams.json")
    with open(out, "w") as f:
        json.dump(best_hp, f, indent=2)
    print(f"\nBest reasoning_to_text hyperparameters: {best_hp}")
    print(f"Saved to {out}")
    return best_hp


def load_best_hparams():
    path = os.path.join(STUDY_DIR, "reasoning_to_text_best_hparams.json")
    if os.path.isfile(path):
        with open(path) as f:
            hp = json.load(f)
        hp.setdefault("lr_min_scale", HYPERPARAMS["defaults"]["lr_min_scale"])
        return hp
    print("[train] no tuned hyperparameters found; using defaults")
    return HYPERPARAMS["defaults"]


# --------------------------------------------------------------------------- #
# Full (resumable) training over the entire compiled dataset -- shared
# between the TRM stage (vae_enabled=False) and the VAE stage
# (vae_enabled=True); see module docstring for how the VAE stage warm-starts
# from the TRM stage's full checkpoint rather than just the reasoning
# expert's standalone weights.
# --------------------------------------------------------------------------- #
def train(device, vae_enabled, rank=0, world_size=1):
    from dist_utils import barrier, broadcast_parameters, is_main_process, rank0_first, sync_gradients

    is_distributed = world_size > 1
    main = is_main_process(rank)
    prefix = "reasoning_vae" if vae_enabled else "reasoning_trm"
    if main:
        print(f"\n===== TRAINING: reasoning_to_text ({prefix}) =====")
    rank0_first(rank, world_size, ensure_dataset_ready)
    train_rows = load_rows(TRAIN_DIR)
    val_rows = load_rows(VAL_DIR)
    if main:
        print(f"[data] train rows: {len(train_rows)} | val rows: {len(val_rows)}")

    hp = load_best_hparams()
    gcfg = HYPERPARAMS["global"]
    coefs = HYPERPARAMS["reward"]
    d_model = gcfg["d_model"]
    image_size = gcfg["image_size"]

    model = build_model(device, d_model, gcfg["half_precision_encoder"])
    reasoning_expert = model.expert_registry.get_expert("reasoning")
    text_output = model.expert_registry.get_expert("text_output")
    codec = SpmCodec(max_len=gcfg["max_target_len"])

    start_epoch, start_row, global_step = 0, 0, 0
    best_val = float("inf")
    # Moving-average baseline for the RL term's variance reduction (see
    # reasoning_rl_term) -- deliberately per-rank-local, never synced across
    # ranks, matching reasoning_rl's own stage-5 baseline (a small,
    # noise-reduction heuristic, not something requiring exact
    # cross-rank consistency).
    rl_baseline = 0.0
    # Read-only peek (see _peek_extra) -- just enough to decide whether this
    # stage already has progress to resume, *before* touching the model, so
    # the VAE-attach-before-load ordering constraint below can be honored.
    _, pending_state = _peek_extra(f"{prefix}_expert_registry", device)

    if vae_enabled and not pending_state:
        # First time entering the VAE stage: warm-start combine/router/
        # expert_registry (which recursively includes the TRM's own
        # trained params, since ExpertRegistry.experts is an nn.ModuleDict)
        # from the *finished* TRM stage's own per-component files, not just
        # reasoning_expert's standalone weights -- "bolts onto original
        # structure" means continuing the same trained instance, not just
        # the reasoning core in isolation. This is a fresh-stage cold start,
        # not a resume, so TRM's optimizer state is correctly never loaded
        # here (only params) -- _load_weights_with_fallback's returned
        # ``extra`` (if any) is simply discarded.
        trm_er_path, _ = _load_weights_with_fallback(
            model.expert_registry, "reasoning_trm_expert_registry", device)
        if trm_er_path:
            for module, name in ((model.combine, "reasoning_trm_combine"),
                                 (model.router, "reasoning_trm_router")):
                _load_weights_with_fallback(module, name, device)
            if main:
                print(f"[resume] VAE stage warm-started from TRM stage checkpoint "
                      f"{os.path.basename(trm_er_path)}")
        elif main:
            print("[reasoning] WARNING: no finished reasoning_trm checkpoint found to "
                  "warm-start the VAE stage from -- starting the VAE stage from a "
                  "fresh (untrained) reasoning core instead.")
        reasoning_expert.set_vae(ReasoningVAEForStage(d_model).to(device))

    elif vae_enabled and pending_state:
        # Resuming a VAE stage that's already in progress -- the VAE must
        # already be attached (it was set on the previous save), so
        # reattach it before loading state into it.
        reasoning_expert.set_vae(ReasoningVAEForStage(d_model).to(device))

    optimizer = _build_optimizer(model, reasoning_expert, hp, vae_enabled)

    epochs = HYPERPARAMS["epochs"]
    batch_size = hp["batch_size"]
    usable_rows = len(train_rows) - (len(train_rows) % (batch_size * world_size))
    rows_per_rank = usable_rows // world_size
    n_opt_steps_per_epoch = max(1, rows_per_rank // batch_size)
    total_steps = n_opt_steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(hp["warmup_steps"], total_steps, hp["lr_min_scale"]),
    )

    # ---- Resume from the latest checkpoint if present ---------------------- #
    # build_model() already warm-started combine/router/expert_registry from
    # whatever's on disk (preferring 'best'); the VAE-cold-start branch above
    # may have additionally warm-started them from the TRM stage. If this
    # stage's own expert_registry 'latest' tag carries an embedded
    # training-state payload, that means *this exact stage* has prior
    # progress to resume -- re-load combine/router/expert_registry from
    # their own 'latest' tag specifically (not whatever was picked above),
    # then restore the optimizer/RNG/counters.
    er_path, training_state = _load_weights_with_fallback(
        model.expert_registry, f"{prefix}_expert_registry", device)
    if training_state:
        for module, name in ((model.combine, f"{prefix}_combine"), (model.router, f"{prefix}_router")):
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
        # before this field existed.
        rl_baseline = training_state.get("rl_baseline", 0.0)
        for _ in range(global_step):
            scheduler.step()
        if _is_epoch_complete(er_path):
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
    max_traj_len = gcfg["max_trajectory_len"]

    model.train()
    for epoch in range(start_epoch, epochs):
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
            needs_reasoning = row["source"] in REASONING_SOURCES
            image_tensor = (_load_image_tensor(row["image"], image_size).to(device)
                            if row.get("image") is not None else None)

            route_vec, state = encode_to_state(model, row["input"], image_tensor)
            if needs_reasoning:
                _, state = reasoning_expert(state)

            supervised_loss, ce = compute_task_loss(model, model.expert_registry, state, route_vec,
                                                    row["target"], codec, device, coefs,
                                                    hp["label_smoothing"], needs_reasoning)

            if needs_reasoning and row.get("trajectory"):
                traj_loss = trajectory_imitation_loss(
                    model, state, row["trajectory"], row.get("traj_answer", ""),
                    row["target"], coefs, max_traj_len,
                )
                if traj_loss is not None:
                    supervised_loss = supervised_loss + coefs["trajectory_coef"] * traj_loss

            if rng.random() < self_correct_p:
                supervised_loss = supervised_loss + coefs["self_correct_coef"] * self_correct_step(
                    model, codec, device, row["input"], image_tensor,
                    row["target"], needs_reasoning, coefs, rng,
                )

            # Scheduled supervised/RL blend (see HYPERPARAMS["reward"]'s
            # schedule comment). advantage uses the baseline *before* this
            # row's update (matching reasoning_rl's own ordering) -- reward
            # and rl_baseline are plain floats, so none of this touches the
            # autograd graph; only logprob does.
            sup_w, rl_w = _schedule_weights(global_step, total_steps, coefs)
            reward, logprob = reasoning_rl_term(text_output, state, row)
            pg_loss = -(reward - rl_baseline) * logprob
            momentum = coefs["rl_baseline_momentum"]
            rl_baseline = momentum * rl_baseline + (1 - momentum) * reward

            loss = sup_w * supervised_loss + rl_w * pg_loss

            (loss / batch_size).backward()
            accum += 1

            if accum == batch_size:
                # Anti-collapse regularizer (see koleo_loss/HYPERPARAMS["reward"]
                # ["koleo_coef"] and text_to_text.py's identical mechanism) --
                # pushes this batch's raw combine/router pooled output apart.
                # Each row's own state graph was already freed by its own
                # backward() call above (this loop does true per-row backward
                # accumulation, not a single batched forward/backward), so this
                # re-encodes the same batch_size rows just processed a second
                # time -- a cheap encode_to_state-only pass, not the full
                # reasoning-expert/decoder pipeline -- purely to get a live
                # graph for this term's gradient, which joins the same
                # optimizer step as everything already accumulated above.
                # Deliberately uses encode_to_state's raw output, not the
                # post-reasoning-expert state used above -- this targets
                # combine/router specifically, the exact component proven in
                # text_to_text.py to need it.
                koleo_rows = [train_rows[order[j]] for j in range(i - batch_size + 1, i + 1)]
                koleo_states = []
                for r in koleo_rows:
                    r_image = (_load_image_tensor(r["image"], image_size).to(device)
                              if r.get("image") is not None else None)
                    _, r_state = encode_to_state(model, r["input"], r_image)
                    koleo_states.append(r_state.mean(dim=1).squeeze(0))
                koleo_states = torch.stack(koleo_states, dim=0)
                koleo_val = koleo_loss(koleo_states)
                (coefs["koleo_coef"] * koleo_val).backward()

                sync_gradients(model.parameters(), world_size)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum = 0
                batch_idx += 1
                global_step += 1

                if main and global_step % 50 == 0:
                    print(f"  e{epoch} row{i} step{global_step} "
                          f"source={row['source']} ce={ce:.4f} koleo={koleo_val.item():.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")
                    wandb_log({"train/ce_loss": ce, "train/lr": scheduler.get_last_lr()[0],
                              "epoch": epoch, "row": i})

                if main and global_step % eval_every == 0:
                    val = validate(model, codec, device, val_rows, coefs, val_sample, image_size)
                    best_val = min(best_val, val)
                    rewards = evaluate_rewards(model, val_rows, sample_size=200,
                                               seed=global_step, image_size=image_size,
                                               device=device)
                    print(f"  [val] step {global_step}: val_loss={val:.4f} "
                          f"(best={best_val:.4f}) rewards="
                          f"{ {k: round(v, 3) for k, v in rewards.items()} }")
                    wandb_log({"val/loss": val,
                              **{f"val/reward_{k}": v for k, v in rewards.items()}})
                    save_checkpoint(prefix, model, optimizer, epoch, i + 1, global_step,
                                    best_val, val_loss=val, rl_baseline=rl_baseline)
                elif main and global_step % ckpt_every == 0:
                    save_checkpoint(prefix, model, optimizer, epoch, i + 1, global_step, best_val,
                                    rl_baseline=rl_baseline)

        barrier(is_distributed)
        start_row = 0
        if not main:
            continue
        val = validate(model, codec, device, val_rows, coefs, val_sample, image_size)
        best_val = min(best_val, val)
        rewards = evaluate_rewards(model, val_rows, sample_size=300, seed=epoch,
                                   image_size=image_size, device=device)
        print(f"[epoch {epoch}] val_loss={val:.4f} (best={best_val:.4f}) rewards={rewards}")
        wandb_log({"val/epoch_loss": val, "epoch": epoch})
        save_checkpoint(prefix, model, optimizer, epoch, len(order), global_step, best_val,
                        val_loss=val, epoch_complete=True, rl_baseline=rl_baseline)

    barrier(is_distributed)
    if not main:
        return
    # The embedded training-state payload is no longer needed now that this
    # stage has genuinely finished, so it's stripped from this stage's own
    # expert_registry tagged files (keeping the weights themselves). Unlike
    # the old wrapper scheme, the TRM stage's files never need to be kept
    # alive for the VAE stage's sake -- VAE cold-start reads TRM's plain
    # per-component files directly (see above), which persist regardless of
    # this stripping, so this can run immediately after either stage
    # finishes rather than being deferred until the VAE stage completes too.
    _strip_training_state(f"{prefix}_expert_registry")
    best_reasoning = find_weights(f"{prefix}_expert", "best")
    latest_reasoning = find_weights(f"{prefix}_expert", "latest")
    print(f"\n[done] reasoning_to_text ({prefix}) training complete.")
    print(f"[optimal weights] reasoning_expert -> {best_reasoning or latest_reasoning}")
    print(f"[latest weights] reasoning_expert -> {latest_reasoning}")


class ReasoningVAEForStage(nn.Module):
    """Thin re-export so train() doesn't need to reach into
    models.Experts.Reasoning's private-ish internals more than once --
    just constructs the real ReasoningVAE."""
    def __new__(cls, d_model):
        from models.Experts.Reasoning import ReasoningVAE
        return ReasoningVAE(d_model)


def _build_optimizer(model, reasoning_expert, hp, vae_enabled):
    if not vae_enabled or reasoning_expert.vae is None:
        return torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=hp["learning_rate"], weight_decay=hp["weight_decay"],
        )
    # VAE stage: the reasoning core (trm/act/bar/memory_lookback) trains at
    # a "near frozen" fraction of the tuned LR -- the VAE bolts on and
    # learns at the full rate. See HYPERPARAMS["vae_lr_scale"].
    vae_param_ids = {id(p) for p in reasoning_expert.vae.parameters()}
    reasoning_core_params = [
        p for p in reasoning_expert.parameters()
        if p.requires_grad and id(p) not in vae_param_ids
    ]
    vae_params = [p for p in reasoning_expert.vae.parameters() if p.requires_grad]
    # model.parameters() (unlike a manual per-child sum via named_children())
    # deduplicates shared parameters via its own internal memo set -- needed
    # here because model.memory and model.encoder_registry.encoders["memory"]
    # are literally the same object (see OverallModel.__init__), so a naive
    # per-child walk would double-count that submodule's params and the
    # optimizer would reject the resulting param groups as overlapping.
    reasoning_ids = {id(p) for p in reasoning_expert.parameters()}
    other_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in reasoning_ids]

    return torch.optim.AdamW([
        {"params": other_params, "lr": hp["learning_rate"]},
        {"params": reasoning_core_params, "lr": hp["learning_rate"] * HYPERPARAMS["vae_lr_scale"]},
        {"params": vae_params, "lr": hp["learning_rate"]},
    ], weight_decay=hp["weight_decay"])


# --------------------------------------------------------------------------- #
# Stage 5: small RL fine-tune, no expert trajectories. Single-sample
# REINFORCE with a moving-average baseline -- "a small stage of RL", not a
# full PPO/GAE build-out. No Qwen dependency at all.
# --------------------------------------------------------------------------- #
def reasoning_rl(device, rank=0, world_size=1):
    from dist_utils import barrier, broadcast_parameters, is_main_process, sync_gradients

    is_distributed = world_size > 1
    main = is_main_process(rank)
    if main:
        print("\n===== TRAINING: reasoning_to_text (rl) =====")

    train_rows = load_rows(TRAIN_DIR)
    gcfg = HYPERPARAMS["global"]
    rl_cfg = HYPERPARAMS["rl"]
    image_size = gcfg["image_size"]

    model = build_model(device, gcfg["d_model"], gcfg["half_precision_encoder"])
    reasoning_expert = model.expert_registry.get_expert("reasoning")
    expert = model.expert_registry.get_expert("text_output")

    # Continue from the VAE stage's finished reasoning core (falling back to
    # the TRM stage's if the VAE stage hasn't produced one yet for some
    # reason); text_output already gets warm-started inside build_model via
    # its own permanent artifact lookup.
    best_reasoning = (
        find_weights("reasoning_vae_expert", "best") or find_weights("reasoning_vae_expert", "latest")
        or find_weights("reasoning_trm_expert", "best") or find_weights("reasoning_trm_expert", "latest")
    )
    if best_reasoning:
        try:
            reasoning_expert.set_vae(ReasoningVAEForStage(gcfg["d_model"]).to(device))
            reasoning_expert.load_weights(best_reasoning, map_location=device)
            if main:
                print(f"[reasoning_rl] resumed reasoning core from {os.path.basename(best_reasoning)}")
        except Exception as e:
            if main:
                print(f"[reasoning_rl] WARNING: could not load {best_reasoning} with VAE "
                      f"attached ({e}); retrying without VAE")
            reasoning_expert.vae = None
            reasoning_expert.load_weights(best_reasoning, map_location=device)
    elif main:
        print("[reasoning_rl] WARNING: no trained reasoning_expert checkpoint found; "
              "starting the RL stage from an untrained reasoning core.")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=rl_cfg["learning_rate"],
    )

    er_path, training_state = _load_weights_with_fallback(
        model.expert_registry, "reasoning_rl_expert_registry", device)
    start_iter, baseline = 0, 0.0
    if training_state:
        for module, name in ((model.combine, "reasoning_rl_combine"), (model.router, "reasoning_rl_router")):
            _load_weights_with_fallback(module, name, device)
        optimizer.load_state_dict(training_state["optimizer_state"])
        torch.set_rng_state(training_state["rng"].cpu())
        start_iter = training_state["epoch"]     # reuses the "epoch" field as the RL iteration counter
        baseline = training_state.get("best_val_loss", 0.0)
        if main:
            print(f"[resume] {os.path.basename(er_path)} -> RL iter {start_iter}, "
                  f"baseline={baseline:.3f}")
    if is_distributed:
        broadcast_parameters(model.parameters(), src=0)
    from dist_utils import barrier as _barrier
    _barrier(is_distributed)

    codec = SpmCodec(max_len=gcfg["max_target_len"])
    rng = random.Random(gcfg["seed"] + rank)
    momentum = rl_cfg["baseline_momentum"]

    model.train()
    for it in range(start_iter, rl_cfg["iters"]):
        optimizer.zero_grad()
        batch_reward, batch_logprob, batch_pooled_state = [], [], []
        for _ in range(rl_cfg["batch_size"]):
            row = train_rows[rng.randrange(len(train_rows))]
            needs_reasoning = row["source"] in REASONING_SOURCES
            image_tensor = (_load_image_tensor(row["image"], image_size).to(device)
                            if row.get("image") is not None else None)
            # Inlines _reasoned_state's logic (rather than calling it directly)
            # to keep a handle on encode_to_state's raw output specifically --
            # see koleo_loss's docstring for why this targets combine/router's
            # output rather than the post-reasoning-expert state used below.
            # No extra forward pass: this loop's own graphs stay live until the
            # single pg_loss.backward() call below (unlike train()'s per-row
            # immediate-backward loop, which needs a separate re-encode for
            # the same regularizer).
            _, raw_state = encode_to_state(model, row["input"], image_tensor)
            batch_pooled_state.append(raw_state.mean(dim=1).squeeze(0))
            state = raw_state
            if needs_reasoning:
                _, state = reasoning_expert(state)
            gen, _, logprob = expert.forward(
                state, max_length=gcfg["max_target_len"], return_logprob=True,
            )
            pred = gen[0] if gen else ""
            reward = compute_reward_metric(row["source"], pred, row["target"], row.get("aux", ""))
            batch_reward.append(reward)
            batch_logprob.append(logprob)

        rewards_t = torch.tensor(batch_reward, device=device, dtype=torch.float32)
        logprob_t = torch.cat(batch_logprob)
        advantage = rewards_t - baseline
        # REINFORCE: maximise E[advantage * logprob] == minimise -E[...]
        pg_loss = -(advantage.detach() * logprob_t).mean()
        # Anti-collapse regularizer (see koleo_loss/HYPERPARAMS["rl"]["koleo_coef"]).
        koleo = koleo_loss(torch.stack(batch_pooled_state, dim=0))
        total_loss = pg_loss + rl_cfg["koleo_coef"] * koleo
        total_loss.backward()
        sync_gradients(model.parameters(), world_size)
        optimizer.step()

        baseline = momentum * baseline + (1 - momentum) * rewards_t.mean().item()

        if main and (it + 1) % 50 == 0:
            print(f"  [rl] iter {it + 1}/{rl_cfg['iters']} "
                  f"mean_reward={rewards_t.mean().item():.3f} baseline={baseline:.3f} "
                  f"pg_loss={pg_loss.item():.4f} koleo={koleo.item():.4f}")
            # Same visibility as evaluate_rewards' sample logging -- `row`/
            # `pred` here are just whatever the inner loop last processed
            # this iter, cheap to reuse rather than a dedicated extra pass.
            print(f"  [reward-sample] {row['source']}: pred={pred!r} target={row['target']!r} "
                 f"reward={reward:.3f}")
            wandb_log({"rl/mean_reward": rewards_t.mean().item(), "rl/baseline": baseline,
                      "rl/pg_loss": pg_loss.item(), "rl/iter": it})

        if main and (it + 1) % 50 == 0:
            # No val_loss metric exists for the RL stage, so only ever a
            # "latest" tag gets written -- see save_checkpoint's docstring
            # for why the training-state payload only ever belongs on
            # 'latest' anyway (there's no meaningful "best optimizer
            # state"). combine/router now get their own standalone files
            # here too (previously missing -- build_model()'s _warm_start
            # already looked for reasoning_rl_combine/reasoning_rl_router
            # first and was silently falling through to the VAE stage's
            # files every time).
            training_state = {
                "optimizer_state": optimizer.state_dict(),
                "epoch": it + 1, "batch": 0, "global_step": it + 1,
                "best_val_loss": baseline, "rng": torch.get_rng_state(),
            }
            os.makedirs(WEIGHTS_DIR, exist_ok=True)
            try:
                rp = weight_path("reasoning_rl_expert", "latest", it + 1, 0, epoch_complete=False)
                tp = weight_path("reasoning_rl_text_output", "latest", it + 1, 0, epoch_complete=False)
                cp = weight_path("reasoning_rl_combine", "latest", it + 1, 0, epoch_complete=False)
                rtp = weight_path("reasoning_rl_router", "latest", it + 1, 0, epoch_complete=False)
                erp = weight_path("reasoning_rl_expert_registry", "latest", it + 1, 0, epoch_complete=False)

                reasoning_expert.store_weights(WEIGHTS_DIR, os.path.basename(rp))
                expert.store_weights(WEIGHTS_DIR, os.path.basename(tp))
                model.combine.store_weights(WEIGHTS_DIR, os.path.basename(cp))
                model.router.store_weights(WEIGHTS_DIR, os.path.basename(rtp))
                model.expert_registry.store_weights(
                    WEIGHTS_DIR, os.path.basename(erp), extra=training_state,
                )

                _prune_other_tagged_files("reasoning_rl_expert", "latest", rp)
                _prune_other_tagged_files("reasoning_rl_text_output", "latest", tp)
                _prune_other_tagged_files("reasoning_rl_combine", "latest", cp)
                _prune_other_tagged_files("reasoning_rl_router", "latest", rtp)
                _prune_other_tagged_files("reasoning_rl_expert_registry", "latest", erp)
            except (RuntimeError, OSError) as e:
                print(f"[checkpoint] RL checkpoint save failed at iter {it + 1} even "
                      f"after retries ({e}); skipping this checkpoint and continuing "
                      f"training -- will try again at the next interval.")

    _barrier(is_distributed)
    if not main:
        return
    _strip_training_state("reasoning_rl_expert_registry")
    print("\n[done] reasoning_to_text (rl) training complete.")
    latest_reasoning = find_weights("reasoning_rl_expert", "latest")
    print(f"[final weights] reasoning_expert -> {latest_reasoning}")


# --------------------------------------------------------------------------- #
# Stage driver
# --------------------------------------------------------------------------- #
def run_stages(args, device, rank=0, world_size=1):
    from dist_utils import barrier, is_main_process

    is_distributed = world_size > 1
    main = is_main_process(rank)
    progress = load_progress()

    if "traj_gen" not in progress["completed"]:
        if main:
            generate_trajectories(device)
            mark_stage_done("traj_gen")
        barrier(is_distributed)
        if main:
            print("\n*** Stage 'traj_gen' complete. Stopping for review. "
                  "Re-submit the job to continue with tuning. ***")
        return

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

    if "reasoning_train_trm" not in progress["completed"]:
        if main:
            init_wandb(args.wandb_key, "reasoning_train_trm")
        train(device, vae_enabled=False, rank=rank, world_size=world_size)
        if main:
            mark_stage_done("reasoning_train_trm")
            finish_wandb()
        barrier(is_distributed)
        return

    if "reasoning_train_vae" not in progress["completed"]:
        if main:
            init_wandb(args.wandb_key, "reasoning_train_vae")
        train(device, vae_enabled=True, rank=rank, world_size=world_size)
        if main:
            mark_stage_done("reasoning_train_vae")
            finish_wandb()
        barrier(is_distributed)
        return

    if "reasoning_rl" not in progress["completed"]:
        if main:
            init_wandb(args.wandb_key, "reasoning_rl")
        reasoning_rl(device, rank=rank, world_size=world_size)
        if main:
            mark_stage_done("reasoning_rl")
            finish_wandb()

    if main:
        print("\nAll reasoning_to_text stages complete.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Reasoning-expert fine-tuning.")
    p.add_argument("--token", default="_", help="Hugging Face token.")
    p.add_argument("--wandb_key", default="_", help="Weights & Biases API key.")
    p.add_argument("--prepare_data_only", action="store_true",
                   help="Compile + split the dataset, then exit.")
    p.add_argument("--reset_progress", action="store_true",
                   help="Clear stage progress (does not delete weights/data).")
    return p.parse_args()


def main():
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
