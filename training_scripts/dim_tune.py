"""
Dimension tuning stage for the ensemble multimodal research model.

Goal
----
Find the optimal model ``dimension`` (d_model) for the *overall* model,
searched across the *same* multi-task mixture text_to_text.py trains on
(Minerva Math, DeepMind Mathematics, Logi-Glue, TriviaQA, NQ Open, ASQA) --
not a single narrow QA task. We use Optuna to search over the candidate
dimensions and optimise *two* objectives simultaneously:

    1. Validation loss across the task mixture (held-out)     (minimise)
    2. FLOPs consumed per example                              (minimise)

This is a multi-objective study, so Optuna returns a Pareto front of
(loss, flops) trade-offs rather than a single "best" trial. We then apply a
simple loss-per-FLOP heuristic to recommend one dimension.

Why the full task mixture, not just TriviaQA
----------------------------------------------
An earlier version of this script tuned dimension against TriviaQA alone.
With a fixed, modest optimizer-step budget, that consistently favoured the
smallest dimension: larger dimensions need more steps/data to reach a
comparable loss on any *one* task, so at a fixed step count they mostly just
looked undertrained while also paying ~10-15x the FLOPs -- a false signal that
would undersell the capacity actually needed once the model has to cover
math, logic and long-form QA together. Reusing text_to_text.py's compiled,
already-shuffled multi-task CSVs (via ``ensure_dataset_ready`` /
``load_csv_rows``) fixes the *task* mismatch; ``_opt_steps_for_dimension``
below (quadratic step scaling with dimension) fixes the *undertraining*
mismatch, so bigger dimensions aren't penalised purely for having had fewer
effective optimizer steps than they need.

Auxiliary ("dummy") learnable hyperparameters
---------------------------------------------
The quantity we care about is ``dimension``. But the accuracy a given dimension
can reach depends on how it is trained -- a learning rate that suits dim=512
may cripple dim=2048. To make each dimension's accuracy *trustworthy* we let
Optuna also search a set of auxiliary training hyperparameters (learning rate,
batch size, weight decay, warmup, label smoothing). Each dimension is therefore
scored at (approximately) its own best training configuration, so the final
dimension comparison is not confounded by a single arbitrary LR/batch choice.
Run more trials (``--n_trials``) to explore these auxiliaries more thoroughly.

The model under test is the project's own ``MultimodalModel`` (models/), built
at the candidate ``dimension`` and threaded through every stage:

    text encoder -> encoder registry -> combination -> router -> text expert

Only the text encoder and the text-output expert are active for this stage
(the non-text encoders/experts are commented out in the model's __init__);
the registries, combination and router stages are kept intact. Unlike
text_to_text.py, the text encoder here is trained from scratch at each
candidate dimension (not the frozen enc_pre.py encoder), since that encoder
is itself trained at one fixed dimension -- this stage is what decides which
dimension to train it at in the first place.

The multi-task CSVs are compiled once (by text_to_text.py's own data
pipeline) and cached on scratch, so a run here after a text_to_text.py
``--prepare_data_only`` pass needs no network access. The Optuna study is
persisted to a small SQLite file on scratch (``STUDY_DB``), so a job that is
killed by the 48-hour SLURM wall clock can simply be resubmitted and will
resume from the last completed trial rather than restarting the search from
scratch (see the comment above ``STUDY_DB`` for the resumability/lightness
trade-off this makes).

The tuning objective is validation loss / perplexity on held-out examples
(minimized), not exact match: a from-scratch model scores ~0 EM on these
tasks, which is uninformative, whereas loss is continuous and sensitive to
model capacity -- so the dimension actually shows up in the signal.

Task mixture: see text_to_text.py's module docstring / SOURCE_LOADERS.
"""

import argparse
import json
import math
import os
import re
import string
import sys

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# Everything data-related lives on scratch (not barn) this time.
SCRATCH_DIR = "/gpfs/u/home/ENMM/ENMMsrtk/scratch/"
DATA_DIR = os.path.join(SCRATCH_DIR, "dim_tune")

# The model package imports its submodules as top-level (e.g. `from Logic
# import ...`), so the models/ directory must be importable directly.
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

# text_to_text.py owns the multi-task data pipeline (download, normalise,
# shuffle, compile to CSV, stratified train/val split); reused here rather
# than duplicated so there is one source of truth for the task mixture.
import text_to_text as t2t

# The Optuna study is persisted to a small SQLite file on scratch (same
# pattern as enc_pre.py / text_to_text.py's own tuning stages), rather than
# held in memory: with the multi-task mixture some trials can be slow enough
# that a full sweep may not fit in one 48-hour SLURM allocation, so a
# resubmitted job needs to pick up completed trials instead of restarting the
# whole search. This only checkpoints at *trial* granularity (a trial killed
# mid-training is lost, not resumed mid-way) -- each trial's cost is already
# bounded by --max_opt_steps, so that's a small, bounded amount of lost work,
# and it keeps this change as light as the SQLite file itself. There's no
# reset flag; delete STUDY_DB manually to start a fresh search.
STUDY_DB = os.path.join(DATA_DIR, "dim_tune_study.db")
RESULTS_JSON = os.path.join(DATA_DIR, "dim_tune_results.json")

# Reuse the project's SentencePiece tokenizer so the tuning stage speaks the
# same vocabulary as the rest of the model.
TOKENIZER_PATH = os.path.join(MODELS_DIR, "Encoders", "multimodal_bpe.model")

# --- The primary search dimension ------------------------------------------ #
DIMENSION_CHOICES = [768, 1024]#[512, 768, 1024, 1536, 2048]

# --- Final-recommendation safety margins ------------------------------------ #
# This study only ever instantiates text_only=True: the text encoder + the
# text-output expert. It never sees the vision encoder, memory, reasoning or
# search experts -- and it certainly never sees the coding/audio/tooling
# experts planned for later -- yet ``dim`` is shared as one hard,
# retroactively-unchangeable architectural width across all of them (every
# future expert is built with ``d_model=dim``). Undersizing costs a full
# re-train of everything downstream of enc_pre.py; oversizing just costs more
# compute now. Two knobs compensate for that blind spot:
#
#   RECOMMENDATION_*_WEIGHT -- the Pareto recommendation normally trades loss
#   off against FLOPs at parity; weighting loss higher stops the heuristic
#   from mechanically picking the cheapest dimension whenever losses happen
#   to be close (exactly what produced the dim=512 pick from a 15-trial run).
#
#   DIMENSION_SAFETY_MARGIN_TIERS -- on top of that, bump the final pick up
#   this many steps in the DIMENSION_CHOICES ladder, so the recommendation
#   deliberately overshoots what text-only tasks alone justify. One tier
#   (e.g. 512 -> 768) roughly matches the working dimensions of common
#   single-modality baselines outside this repo (audio encoders ~512-1024,
#   code-model hidden sizes ~768+, tool/function-calling schemas benefiting
#   from >=768) without jumping straight to a much more expensive tier.
RECOMMENDATION_LOSS_WEIGHT = 2.0
RECOMMENDATION_FLOPS_WEIGHT = 0.5
DIMENSION_SAFETY_MARGIN_TIERS = 1

# --- Auxiliary ("dummy") learnable hyperparameters -------------------------- #
# Searched jointly with ``dimension`` so each dimension is scored near its own
# best training configuration. These do not change the *architecture width*;
# they only govern how each candidate is optimised.
# Per-dimension log-uniform LR search range: larger dimensions get a
# narrower, lower range (they're more prone to instability at the LRs a
# small dimension tolerates fine). Only entries for dimensions actually in
# DIMENSION_CHOICES are ever looked up -- one not under test is simply
# never used (see _lr_range_for_dimension).
LR_RANGES = {
    512:  (1e-4, 1e-3),
    768:  (2e-4, 2e-3),
    1024: (5e-5, 5e-4),
    1536: (1e-5, 2e-4),
    2048: (1e-5, 1e-4),
}
BATCH_SIZE_CHOICES = [32]#[8, 16, 32, 64]        # realised via gradient accumulation
WEIGHT_DECAY_RANGE = (1e-5, 1e-1)           # log-uniform
WARMUP_CHOICES = [50]#[0, 10, 50]                # optimizer-step linear warmup
LABEL_SMOOTHING_CHOICES = [0.1]#[0.0, 0.1]

MAX_SRC_LEN = 48       # question length cap (tokens; FLOPs-estimate only)
MAX_TGT_LEN = 32       # answer length cap (tokens) -- bumped from 24 so
                       # longer-form targets (ASQA, logic explanations) in
                       # the shared task mixture aren't over-truncated

# Larger dimensions need more optimizer steps to reach a comparable loss than
# smaller ones do; scaling the per-trial step budget with dimension keeps
# big-vs-small comparisons fairer (see docstring above). Quadratic, uncapped:
# a dimension 4x the base (e.g. 2048 vs. 512) gets 16x the step budget.
STEP_SCALE_BASE_DIM = 512


def _opt_steps_for_dimension(base_steps, dimension):
    mult = max(1.0, (dimension / STEP_SCALE_BASE_DIM) ** 2)
    return int(base_steps * mult)


def _lr_range_for_dimension(dimension):
    if dimension in LR_RANGES:
        return LR_RANGES[dimension]
    print(f"[warn] no LR range configured for dimension={dimension} in "
          f"LR_RANGES; falling back to the broadest configured range")
    lows = [r[0] for r in LR_RANGES.values()]
    highs = [r[1] for r in LR_RANGES.values()]
    return (min(lows), max(highs))


# SentencePiece special ids (see TextEncoder.createTokenizer / spm training).
PAD_ID = 0
BOS_ID = 2
EOS_ID = 3


# --------------------------------------------------------------------------- #
# Answer normalisation & exact match (SQuAD / TriviaQA style)
# --------------------------------------------------------------------------- #
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(text: str) -> str:
    """Lower-case, remove punctuation, articles and extra whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES.sub(" ", text)
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, gold_aliases) -> bool:
    """True if the (normalised) prediction matches any (normalised) alias."""
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(a) for a in gold_aliases if a)


# --------------------------------------------------------------------------- #
# Dataset preparation: delegate entirely to text_to_text.py's data pipeline
# (download, normalise, shuffle, compile to CSV, stratified train/val split),
# then adapt its {input, target, source, aux} rows into the (question, answer,
# aliases) tuples the rest of this file already expects.
# --------------------------------------------------------------------------- #
def prepare_task_mixture():
    """Compile + split the shared multi-task CSVs (no-op if already cached)."""
    t2t.ensure_dataset_ready()


def _rows_to_examples(rows):
    examples = []
    for r in rows:
        aliases = [a for a in (r.get("aux") or "").split("|") if a] or [r["target"]]
        examples.append((r["input"], r["target"], aliases))
    return examples


def load_examples(split, limit=None):
    """Load a (question, answer, aliases) sample from text_to_text.py's
    compiled train/val CSVs. ``limit`` caps how many rows are sampled."""
    path = t2t.TRAIN_CSV if split == "train" else t2t.VAL_CSV
    rows = t2t.load_csv_rows(path)
    if limit is not None:
        rows = t2t.sample_rows(rows, limit, seed=0 if split == "train" else 1)
    return _rows_to_examples(rows)


# --------------------------------------------------------------------------- #
# Tokenisation (reuse the project's SentencePiece model)
# --------------------------------------------------------------------------- #
class SpmCodec:
    """Thin wrapper over the shared SentencePiece processor for building the
    gold answer token ids used by the training loss."""

    def __init__(self, model_path=TOKENIZER_PATH):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode_target(self, text, device, max_len=MAX_TGT_LEN):
        ids = self.sp.encode_as_ids(text)[: max_len - 2]
        ids = [BOS_ID] + ids + [EOS_ID]
        return torch.tensor([ids], dtype=torch.long, device=device)


# --------------------------------------------------------------------------- #
# Training / evaluation helpers (loss lives in the script, not the model)
# --------------------------------------------------------------------------- #
def teacher_forced_logits(expert, state, tgt_in):
    """Drive the text-output expert's own sub-modules with teacher forcing to
    obtain next-token logits [B, T, vocab]. The cross-entropy loss is applied
    by the caller -- we deliberately keep the loss in the tuning script rather
    than baking it into the expert."""
    device = tgt_in.device
    T = tgt_in.size(1)
    emb = expert.token_embedding(tgt_in) * math.sqrt(expert.d_model)
    emb = emb + expert._sincos_pos_embed(T, device)
    tgt_mask = expert._causal_mask(T, device)
    # The expert uses a Universal-Transformer-style recursive decoder; drive it
    # with teacher forcing (no memory vectors) to get the ACT-weighted hidden
    # states, then project to vocab logits.
    weighted_h, _ = expert._recursive_decode(emb, state, None, tgt_mask)
    return expert.output_proj(weighted_h)


def encode_to_state(model, question):
    """Run the shared front-end: text encode -> combine -> router, returning
    the routed ``state`` (memory) that the text expert consumes."""
    encodings = model.encode(question, "text")
    combined, _, _ = model.combine(encodings)  # (latents, num_active, active_mask)
    _, state = model.router(combined)
    return state


# --------------------------------------------------------------------------- #
# FLOPs measurement (per example forward pass through the whole pipeline)
# --------------------------------------------------------------------------- #
def measure_flops(model, expert, question, tgt_in, dimension):
    """Measured FLOPs for one example's forward pass. Depends on ``dimension``
    (not on the auxiliary training hyperparameters). Falls back to an analytic
    estimate if torch's FlopCounterMode is unavailable."""
    try:
        from torch.utils.flop_counter import FlopCounterMode

        was_training = model.training
        model.eval()
        counter = FlopCounterMode(display=False)
        with torch.no_grad(), counter:
            state = encode_to_state(model, question)
            teacher_forced_logits(expert, state, tgt_in)
        if was_training:
            model.train()
        total = counter.get_total_flops()
        if total > 0:
            return float(total)  # pipeline runs one example at a time (B=1)
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"[flops] FlopCounterMode unavailable ({e}); using estimate.")

    # Analytic fallback: transformer FLOPs are dominated by the linear layers,
    # which scale ~ O(seq_len * dimension^2 * num_layers). Layer counts below
    # mirror the project's components (text enc 3 + combine 6 + router 2 +
    # expert decoder 4).
    layers = 3 + 6 + 2 + 4
    s = MAX_SRC_LEN + MAX_TGT_LEN
    return float(2 * layers * s * dimension * dimension * 6)


# --------------------------------------------------------------------------- #
# Train + evaluate a single (dimension + hyperparameter) configuration
# --------------------------------------------------------------------------- #
def train_and_evaluate(dimension, hp, train_examples, eval_examples, codec,
                       device, max_opt_steps, eval_samples):
    """Build the overall model at ``dimension``, train it with the auxiliary
    hyperparameters ``hp``, and return (val_loss, flops_per_example)."""
    from models.OverallModel import MultimodalModel

    # Bigger dimensions get quadratically more optimizer steps, so they
    # aren't undersold purely for being undertrained at a fixed step budget
    # (see module docstring).
    effective_steps = _opt_steps_for_dimension(max_opt_steps, dimension)

    print(
        f"\n[trial] dimension={dimension} | lr={hp['learning_rate']:.2e} "
        f"| batch={hp['batch_size']} | wd={hp['weight_decay']:.2e} "
        f"| warmup={hp['warmup_steps']} | ls={hp['label_smoothing']} "
        f"| opt_steps={effective_steps} (base={max_opt_steps}) | device={device}"
    )
    torch.manual_seed(0)

    # text_only=True keeps just the text encoder + text-output expert active
    # (registries, combination and router stages preserved), per the
    # dimension-tuning rule.
    model = MultimodalModel(dim=dimension, text_only=True)
    model = model.to(device)
    expert = model.expert_registry.get_expert("text_output")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
    )

    warmup = hp["warmup_steps"]

    def lr_scale(step):  # linear warmup then constant
        return (step + 1) / warmup if warmup > 0 and step < warmup else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    batch_size = hp["batch_size"]
    label_smoothing = hp["label_smoothing"]

    # ---- Training: teacher forcing + gradient accumulation ----------------- #
    model.train()
    n = len(train_examples)
    optimizer.zero_grad()
    accum, opt_step, ex_idx = 0, 0, 0
    running_loss = 0.0
    while opt_step < effective_steps:
        question, answer, _ = train_examples[ex_idx % n]
        ex_idx += 1

        state = encode_to_state(model, question)
        tgt = codec.encode_target(answer, device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

        logits = teacher_forced_logits(expert, state, tgt_in)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=PAD_ID,
            label_smoothing=label_smoothing,
        )
        (loss / batch_size).backward()
        running_loss += loss.item()
        accum += 1

        if accum == batch_size:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accum = 0
            opt_step += 1
            if opt_step % max(1, effective_steps // 5) == 0:
                avg = running_loss / batch_size
                print(f"  opt-step {opt_step:4d}/{effective_steps}  loss={avg:.4f}")
            running_loss = 0.0

    # ---- Evaluation: validation loss (teacher-forced CE) on held-out answers #
    # We use loss, not exact match: a from-scratch model scores ~0 EM on
    # closed-book TriviaQA (uninformative), whereas per-token loss is continuous
    # and capacity-sensitive, so the dimension actually shows up in the signal.
    # It is also cheap (no autoregressive generation).
    model.eval()
    eval_subset = eval_examples[:eval_samples]
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for question, answer, _ in eval_subset:
            state = encode_to_state(model, question)
            tgt = codec.encode_target(answer, device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = teacher_forced_logits(expert, state, tgt_in)
            l = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
                ignore_index=PAD_ID,
                reduction="sum",
            )
            total_loss += l.item()
            total_tokens += int((tgt_out != PAD_ID).sum().item())
    val_loss = total_loss / max(1, total_tokens)

    # ---- FLOPs (per example) ---------------------------------------------- #
    q0, a0, _ = eval_subset[0]
    tgt0 = codec.encode_target(a0, device)[:, :-1]
    flops = measure_flops(model, expert, q0, tgt0, dimension)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[trial] dimension={dimension} -> val_loss={val_loss:.4f} "
        f"(ppl={math.exp(min(val_loss, 20.0)):.2f}) | "
        f"FLOPs/ex={flops:.3e} | params={n_params:,}"
    )

    del model, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return val_loss, flops


# --------------------------------------------------------------------------- #
# Optuna study
# --------------------------------------------------------------------------- #
def run_study(args):
    import optuna

    device = "cuda" if torch.cuda.is_available() else "cpu"

    prepare_task_mixture()
    train_examples = load_examples("train", limit=args.train_examples)
    eval_examples = load_examples("validation", limit=args.eval_examples)
    print(
        f"[data] train examples: {len(train_examples)} | "
        f"eval examples: {len(eval_examples)}"
    )

    codec = SpmCodec()

    def objective(trial):
        dimension = trial.suggest_categorical("dimension", DIMENSION_CHOICES)
        # Auxiliary "dummy" learnable hyperparameters.
        hp = {
            "learning_rate": trial.suggest_float(
                "learning_rate", *_lr_range_for_dimension(dimension), log=True
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", BATCH_SIZE_CHOICES
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", *WEIGHT_DECAY_RANGE, log=True
            ),
            "warmup_steps": trial.suggest_categorical(
                "warmup_steps", WARMUP_CHOICES
            ),
            "label_smoothing": trial.suggest_categorical(
                "label_smoothing", LABEL_SMOOTHING_CHOICES
            ),
        }
        val_loss, flops = train_and_evaluate(
            dimension, hp, train_examples, eval_examples, codec,
            device, args.max_opt_steps, args.eval_samples,
        )
        trial.set_user_attr("flops_per_example", flops)
        trial.set_user_attr("val_loss", val_loss)
        trial.set_user_attr("perplexity", math.exp(min(val_loss, 20.0)))
        return val_loss, flops

    # SQLite-backed study (see STUDY_DB comment above) so a resubmitted job
    # resumes from the last completed trial instead of restarting the whole
    # search. NSGA-II explores dimension + auxiliary hyperparameters and
    # converges on the (loss, FLOPs) Pareto front; it reconstructs its
    # generation state from the trials already in storage on reload.
    os.makedirs(DATA_DIR, exist_ok=True)
    sampler = optuna.samplers.NSGAIISampler(seed=0)
    study = optuna.create_study(
        study_name="dim_tune",
        storage=f"sqlite:///{STUDY_DB}",
        directions=["minimize", "minimize"],  # val loss down, FLOPs down
        sampler=sampler,
        load_if_exists=True,
    )
    # --n_trials is the *target total* completed trials for the whole study,
    # not "how many more to run this invocation" -- otherwise every
    # resubmission after a SLURM timeout keeps adding n_trials more forever
    # instead of the search ever finishing. Only COMPLETE trials count
    # towards that total; a trial killed mid-training by the wall clock is
    # left behind in the DB as RUNNING/FAIL and is simply retried, not
    # counted as done.
    n_done = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
    n_remaining = max(0, args.n_trials - n_done)
    if n_done:
        print(f"[resume] loaded existing study from {STUDY_DB} "
              f"({n_done} trial(s) already completed)")
    if n_remaining == 0:
        print(f"[resume] study already has {n_done} completed trial(s) "
              f">= target --n_trials={args.n_trials}; nothing to do.")
    else:
        print(f"[resume] running {n_remaining} more trial(s) to reach "
              f"target --n_trials={args.n_trials}")
        study.optimize(objective, n_trials=n_remaining)

    report_results(study)


def _bump_dimension(dimension, tiers):
    """Shift ``dimension`` up by ``tiers`` steps in the fixed DIMENSION_CHOICES
    ladder, clamped at the top. Applies the safety margin described above
    DIMENSION_SAFETY_MARGIN_TIERS on top of the raw score-based pick."""
    sorted_dims = sorted(DIMENSION_CHOICES)
    idx = sorted_dims.index(dimension)
    return sorted_dims[min(idx + tiers, len(sorted_dims) - 1)]


def _pareto_front(points):
    """Return the Pareto-optimal subset minimising both val_loss and flops."""
    front = []
    for p in points:
        dominated = any(
            q["val_loss"] <= p["val_loss"]
            and q["flops_per_example"] <= p["flops_per_example"]
            and (q["val_loss"] < p["val_loss"]
                 or q["flops_per_example"] < p["flops_per_example"])
            for q in points
        )
        if not dominated:
            front.append(p)
    return front


def report_results(study):
    print("\n" + "=" * 70)
    print("DIMENSION TUNING RESULTS")
    print("=" * 70)

    # Aggregate by dimension: each dimension keeps the LOWEST validation loss
    # found across its (LR/batch/...) trials -- i.e. that dimension evaluated at
    # its own best training configuration. FLOPs are essentially constant per
    # dimension, so we just record them.
    by_dim = {}
    for t in study.trials:
        if t.values is None:
            continue
        dim = t.params.get("dimension")
        val_loss, flops = t.values
        entry = by_dim.setdefault(
            dim, {"dimension": dim, "val_loss": float("inf"),
                  "flops_per_example": flops, "best_hparams": None,
                  "n_trials": 0}
        )
        entry["n_trials"] += 1
        entry["flops_per_example"] = flops
        if val_loss < entry["val_loss"]:
            entry["val_loss"] = val_loss
            entry["best_hparams"] = {
                k: v for k, v in t.params.items() if k != "dimension"
            }

    per_dim = sorted(by_dim.values(), key=lambda r: r["dimension"])
    for r in per_dim:
        r["perplexity"] = math.exp(min(r["val_loss"], 20.0))

    print("\nBest result per dimension (at its own best hyperparameters):")
    for r in per_dim:
        print(
            f"  dim={r['dimension']:<5} val_loss={r['val_loss']:.4f} "
            f"(ppl={r['perplexity']:.2f})  FLOPs/ex={r['flops_per_example']:.3e}  "
            f"(trials={r['n_trials']}, hp={r['best_hparams']})"
        )

    pareto = sorted(_pareto_front(per_dim), key=lambda r: r["dimension"])
    print("\nPareto-optimal dimensions (loss vs FLOPs):")
    for r in pareto:
        print(
            f"  dim={r['dimension']:<5} val_loss={r['val_loss']:.4f}  "
            f"FLOPs/ex={r['flops_per_example']:.3e}"
        )

    # Score-based pick: among Pareto-optimal dimensions, the best balance of
    # accuracy (low loss) and efficiency (low FLOPs), scored on min-max
    # normalized [0,1] loss + FLOPs -- weighted so loss dominates FLOPs (see
    # RECOMMENDATION_*_WEIGHT above), since this study can't see the future
    # experts that will also have to live inside this dimension.
    scored_pick = None
    if pareto:
        losses = [r["val_loss"] for r in pareto]
        flps = [r["flops_per_example"] for r in pareto]
        lo_l, hi_l = min(losses), max(losses)
        lo_f, hi_f = min(flps), max(flps)

        def _score(r):
            nl = (r["val_loss"] - lo_l) / (hi_l - lo_l) if hi_l > lo_l else 0.0
            nf = (r["flops_per_example"] - lo_f) / (hi_f - lo_f) if hi_f > lo_f else 0.0
            return RECOMMENDATION_LOSS_WEIGHT * nl + RECOMMENDATION_FLOPS_WEIGHT * nf

        scored_pick = min(pareto, key=_score)
        print(
            f"\nScore-based pick (loss weighted {RECOMMENDATION_LOSS_WEIGHT}x vs "
            f"FLOPs {RECOMMENDATION_FLOPS_WEIGHT}x): dimension={scored_pick['dimension']} "
            f"(val_loss={scored_pick['val_loss']:.4f}, "
            f"ppl={scored_pick['perplexity']:.2f}, "
            f"FLOPs/ex={scored_pick['flops_per_example']:.3e}, "
            f"hp={scored_pick['best_hparams']})"
        )

    # Final recommendation: the score-based pick, bumped up by a fixed safety
    # margin in the DIMENSION_CHOICES ladder so the answer stays realistic
    # for the untested future experts (vision/memory/reasoning/search now,
    # coding/audio/tooling later) that will share this same dimension.
    recommended_dimension = None
    if scored_pick:
        recommended_dimension = _bump_dimension(
            scored_pick["dimension"], DIMENSION_SAFETY_MARGIN_TIERS
        )
        if recommended_dimension == scored_pick["dimension"]:
            print(f"Recommended dimension: {recommended_dimension} "
                  f"(already at the top of DIMENSION_CHOICES; no margin to add)")
        else:
            print(f"Recommended dimension: {recommended_dimension} "
                  f"(+{DIMENSION_SAFETY_MARGIN_TIERS} tier safety margin over the "
                  f"score-based pick of {scored_pick['dimension']}, for future experts "
                  f"this study doesn't test)")

    results = {
        "dimension_choices": DIMENSION_CHOICES,
        "recommendation_weights": {
            "loss_weight": RECOMMENDATION_LOSS_WEIGHT,
            "flops_weight": RECOMMENDATION_FLOPS_WEIGHT,
            "safety_margin_tiers": DIMENSION_SAFETY_MARGIN_TIERS,
        },
        "auxiliary_search_space": {
            "learning_rate_by_dimension": LR_RANGES,
            "batch_size": BATCH_SIZE_CHOICES,
            "weight_decay": WEIGHT_DECAY_RANGE,
            "warmup_steps": WARMUP_CHOICES,
            "label_smoothing": LABEL_SMOOTHING_CHOICES,
        },
        "per_dimension": per_dim,
        "pareto_front": pareto,
        "scored_pick": scored_pick,
        "recommended_dimension": recommended_dimension,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_JSON}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Dimension tuning with Optuna.")
    # Accepted for compatibility with the shared SLURM launch template.
    p.add_argument("--token", default="_", help="Hugging Face token.")
    p.add_argument("--n_trials", type=int, default=15,
                   help="Target TOTAL completed trials for the persisted "
                        "study (STUDY_DB), not 'more trials this run' -- a "
                        "resubmitted job only runs however many are still "
                        "needed to reach this count, then stops.")
    p.add_argument("--train_examples", type=int, default=20000,
                   help="Cap on training examples sampled from the compiled "
                        "multi-task CSV (see text_to_text.py).")
    p.add_argument("--eval_examples", type=int, default=2000,
                   help="Cap on eval examples sampled from the compiled "
                        "multi-task CSV (see text_to_text.py).")
    p.add_argument("--max_opt_steps", type=int, default=3000,
                   help="Base optimizer-step budget per trial at "
                        "STEP_SCALE_BASE_DIM; larger dimensions get a "
                        "quadratically scaled-up budget (see docstring) -- "
                        "e.g. 4x the base dimension gets 16x the steps.")
    p.add_argument("--eval_samples", type=int, default=500,
                   help="Number of eval examples scored for validation loss.")
    p.add_argument("--prepare_data_only", action="store_true",
                   help="Compile + split the shared multi-task CSVs "
                        "(text_to_text.py's data pipeline), then exit.")
    return p.parse_args()


def main():
    args = parse_args()

    # Log into HF only if a real token was supplied (dataset is public, so this
    # is optional, but keeps parity with the launch template).
    if args.token and args.token != "_":
        try:
            from huggingface_hub import login

            login(token=args.token)
        except Exception as e:
            print(f"[hf] login skipped: {e}")

    if args.prepare_data_only:
        prepare_task_mixture()
        print("[data] Preparation complete.")
        return

    run_study(args)


if __name__ == "__main__":
    sys.exit(main())
