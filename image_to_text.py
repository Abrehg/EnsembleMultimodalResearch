"""
Image-to-text fine-tuning for the ensemble multimodal research model.

Overview
--------
This script pairs two encoders pretrained *in isolation* by enc_pre.py --
the frozen vision encoder (``image_train`` stage) and the frozen text
encoder (``text_train`` stage) -- into a basic image-understanding
fine-tuning task: "vision" modality (a photo, chart, diagram, or figure)
plus "text" modality (a question about it) are encoded, combined, routed,
and decoded through the project's ``TextOutput`` expert, mirroring
ocr_to_text.py's reward/loss machinery almost exactly, just swapping the
frozen OCR encoder for the frozen vision encoder and (unlike ocr_to_text.py,
which has a single source) interleaving five independent task sources the
way text_to_text.py does. This is deliberately a *separate* script from
enc_pre.py, text_to_text.py, and ocr_to_text.py -- run this one after
text_to_text.py, once both frozen encoders it depends on exist (enc_pre.py's
``text_train`` and ``image_train`` stages).

Pipeline (staged; stops for review after the tuning stage, like the other
two *_to_text.py scripts):

    1. tune  -> Optuna search for training hyperparameters (on a small
                subset of the compiled data), then exit(0) for review.
    2. train -> full resumable training run over the entire compiled
                dataset, with checkpoints after every optimizer step and
                after every completed epoch.

Tasks (all five entire datasets are used and interleaved -- no source-level
caps -- so the model has to "think on its feet" across task types rather
than specialising batch-by-batch; see HYPERPARAMS["data"], every *_cap is
None):

    - chartqa    -- chart reading (HuggingFaceM4/ChartQA). Reward: "close
                    match" -- ChartQA's standard relaxed-accuracy metric
                    (numeric answers within 5% relative tolerance, else
                    exact match -- see close_match).
    - geoqa      -- geometry diagram QA. Source: Geometry3K (Lu et al., ACL
                    2021, github.com/lupantech/InterGPS), NOT chen-judge/
                    GeoQA (that original is Chinese-language/pre-tokenized/
                    formal-DSL-answer, not a natural-language Q/A pair --
                    Geometry3K is a different original English benchmark
                    filling the same role, per explicit user direction; see
                    GEOMETRY3K_GDRIVE_FOLDER_ID's comment). Reward: MCQ
                    accuracy (_format_mcq/_mcq_answer_letter, same scheme
                    as ai2d/scienceqa below). Google-Drive-hosted -- needs
                    gdown (see image_to_text.sh's setup step).
    - clevr      -- synthetic compositional visual QA. Source: the original
                    CLEVR v1.0 release (Johnson et al., CVPR 2017,
                    cs.stanford.edu/people/jcjohns/clevr), direct HTTPS
                    download from Meta/FAIR's own CDN (see CLEVR_URL).
                    Reward: exact match. 18GB, one-time download -- see
                    test_datasets.py's LARGE_DOWNLOAD_SOURCES.
    - ai2d       -- diagram understanding, MCQ (lmms-lab/ai2d). Reward: MCQ
                    accuracy -- collapses to exact_match since the training
                    target is already just the gold option's letter (see
                    _format_mcq / _mcq_answer_letter).
    - scienceqa  -- general science MCQ (derek-thomas/ScienceQA). Reward:
                    MCQ accuracy, same scheme as ai2d. Text-only rows (no
                    attached image) are skipped -- this stage is
                    specifically "text + image in", not text-only QA.

Every loader's exact column names are a best-effort guess at each dataset's
schema (`.get()` with fallback key names) since this was written without
network access to inspect them directly -- if a schema guess is wrong for
some field, the defensive try/except around each source in compile_dataset()
means that source is skipped with a clear log line rather than crashing the
whole compile.

Data pipeline (arrow dataset on scratch, not CSV)
---------------------------------------------------
Same reasoning as ocr_to_text.py: every row carries an embedded image, so
the compiled dataset is a HuggingFace ``datasets.Dataset`` (arrow format)
saved via ``save_to_disk``/loaded via ``load_from_disk`` instead of a CSV.
Unlike ocr_to_text.py's single-source split, but exactly like
text_to_text.py, the train/val split is stratified *per source* (see
split_dataset) so a small source can't be starved out of val by pure chance.
Each row is ``{input, target, source, aux, image}``: ``input`` is the
question (MCQ sources have the formatted options baked in, see
_format_mcq), ``target`` the primary/gold answer, ``aux`` the pipe-joined
alternate answers where a source has more than one valid phrasing, ``image``
the picture/chart/diagram.

Reward/loss machinery, checkpoint naming and finalize-best-and-latest
cleanup, resumability, and the incremental warmup-then-cosine LR schedule
are all identical in spirit to ocr_to_text.py / text_to_text.py (see either
file's module docstring for the full reasoning) -- duplicated here rather
than imported, matching this project's established "self-contained scripts"
convention. Brier-loss coefficient is intentionally smaller than the other
two *_to_text.py scripts (0.02 vs 0.05) -- these five tasks are mostly
short, low-ambiguity answers (MCQ letters, chart numbers, single words), so
a confident-and-correct answer shouldn't be penalised as heavily as it would
be for the more open-ended text_to_text.py tasks.

Self-correction training (generate an answer, intentionally corrupt it half
the time, then train the model to say whether the shown answer is right) is
identical to the other two *_to_text.py scripts -- see self_correct_step.

At the end of training the script prints the located best ``img2t_text_output``
file explicitly.
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

# Bulk, regenerable data (compiled dataset, HF cache) lives on scratch;
# small but load-bearing state -- tuned hyperparameters, the Optuna study
# DB, stage progress, trained weights -- lives under the project directory
# on barn instead, since scratch filesystems are periodically purged and
# losing any of these would silently reset an in-progress pipeline.
SCRATCH_DIR = "/gpfs/u/home/ENMM/ENMMsrtk/scratch/"
DATA_DIR = os.path.join(SCRATCH_DIR, "image_to_text_data")
STUDY_DIR = os.path.join(PROJECT_DIR, "studies", "image_to_text")
# All weight files -- including the resumable "_wrapper" checkpoints --
# live here on barn, never on scratch. scratch is reserved strictly for
# datasets/HF cache. Shared with enc_pre.py / text_to_text.py / ocr_to_text.py.
WEIGHTS_DIR = os.path.join(PROJECT_DIR, "weights")
HF_CACHE = os.path.join(DATA_DIR, "hf_cache")
PROGRESS_PATH = os.path.join(PROJECT_DIR, "image_to_text_progress.json")

COMPILED_DIR = os.path.join(DATA_DIR, "compiled")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

TOKENIZER_PATH = os.path.join(MODELS_DIR, "Encoders", "multimodal_bpe.model")
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

CHARTQA_HF = "HuggingFaceM4/ChartQA"
# geoqa/clevr both moved off HF entirely (MathLLMs/GeoQA doesn't exist on
# the Hub; HuggingFaceM4/CLEVR is script-based with no supported data
# files -- see test_datasets.py's KNOWN_PENDING_REPLACEMENT for how these
# were diagnosed) to their ORIGINAL, official academic sources, for
# publication provenance -- see _download_geometry3k/load_geoqa and
# _download_clevr/load_clevr below.
AI2D_HF = "lmms-lab/ai2d"
SCIENCEQA_HF = "derek-thomas/ScienceQA"

# Geometry3K (Lu et al., ACL 2021 -- github.com/lupantech/InterGPS), MIT
# licensed -- replaces GeoQA in this project's "geometry diagram QA" role.
# NOT the original chen-judge/GeoQA (Chinese-language, pre-tokenized,
# formal-DSL-program answers, no natural-language Q/A pair -- see this
# investigation's conversation history for the full diagnosis) -- this is
# a *different*, English-language original geometry benchmark filling the
# same functional role, per explicit user direction. Hosted on Google
# Drive (not a HF-mirror convenience -- an inherent property of this class
# of small academic geometry benchmark; GeoQA's own official release has
# the same characteristic), so needs `pip install gdown` (not in mulEnv by
# default -- one-time setup) rather than a plain HTTP download.
GEOMETRY3K_GDRIVE_FOLDER_ID = "1d05WYXtlgKIoaPpK1v94LYph_heiXM7Z"

# CLEVR (Johnson et al., CVPR 2017 -- cs.stanford.edu/people/jcjohns/clevr),
# CC BY 4.0 licensed, official release via Meta/FAIR's own CDN -- replaces
# the unverified HuggingFaceM4/CLEVR HF id. Direct HTTPS download, no auth/
# quota concerns (unlike Geometry3K's Google Drive hosting).
CLEVR_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"


# =========================================================================== #
# Hyperparameters
# =========================================================================== #
HYPERPARAMS = {
    "global": {
        "d_model": 1024,
        "seed": 42,
        "max_input_len": 96,             # word cap applied when compiling rows
        "max_target_len": 32,            # token cap for teacher-forced targets
        # Chart/diagram/MCQ imagery needs more legible detail than the
        # generic 128px resolution enc_pre.py's image_train stage tuned the
        # vision encoder at (HYPERPARAMS["global"]["image_size"] there, see
        # enc_pre.py) -- but the encoder's axial-RoPE spatial encoding
        # generalises across resolutions (no learned absolute position
        # embedding to outgrow) and it pads internally to a patch-size
        # multiple regardless, so a moderate bump is safe. Same reasoning
        # ocr_to_text.py applies to the OCR encoder's image_size.
        "image_size": 224,
        "val_fraction": 0.05,
        "tune_train_subset": 2000,
        "tune_val_subset": 400,
        "checkpoint_every_batches": 1,
        "eval_every_batches": 300,
        "half_precision_encoder": True,  # fp16-autocast both frozen encoders on CUDA
        # Only a running 'latest' and 'best' checkpoint are ever kept per
        # component (see save_checkpoint), bounding disk usage by a small
        # constant regardless of run length.
    },

    "reward": {
        "task_loss_coef": 1.0,
        "length_penalty_coef": 0.01,
        "ponder_cost_coef": 0.01,
        # Smaller than text_to_text.py / ocr_to_text.py's 0.05 -- see module
        # docstring for why (short, low-ambiguity answers here).
        "brier_coef": 0.02,
        "self_correct_coef": 0.2,
        "self_correct_prob": 0.15,
        # ---- Degenerate-output penalties (see compute_task_loss) --------- #
        # See text_to_text.py's identical addition: discovered via
        # evaluate_rewards' sample logging that free-run greedy generation
        # can collapse to either an empty answer (immediate EOS) or a
        # single token repeated until the length cap, both scoring 0
        # reward regardless of how good val_loss looks. Both terms are
        # computed from the same teacher-forced logits already available.
        "blank_answer_penalty_coef": 0.3,
        "repetition_penalty_coef": 0.3,
    },

    "epochs": 3,
    "tune_trials": 12,
    "tune_budget_batches": 250,

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
        # None = use the entire available split for every source -- the
        # whole point of this stage (per the task spec) is training on the
        # complete, interleaved datasets, not a subsample.
        "chartqa_cap": None,
        "geoqa_cap": None,
        "clevr_cap": None,
        "ai2d_cap": None,
        "scienceqa_cap": None,
    },
}

STAGES = ["tune", "train"]


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
            project="image_to_text", name=stage, reinit=True, config=HYPERPARAMS,
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
# Greppable weight naming + lookup (shared weights/ directory with
# enc_pre.py / text_to_text.py / ocr_to_text.py -- identical scheme, see any
# of those files' module docstrings for the full rationale).
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
    found rather than crashing the run outright -- specifically matters for
    resuming a mid-run checkpoint, where a corrupted 'latest' file would
    otherwise crash every subsequent resubmission the same way instead of
    falling back to the last good ('best') snapshot. Returns
    (path, load_weights' return value), or (None, None) if neither tag
    loads -- for ExpertRegistry that return value is the embedded ``extra``
    training-state payload (see ExpertRegistry.store_weights), None for
    everything else."""
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
    component with its embedded training-state payload removed -- called
    once training has genuinely finished, since the optimizer/RNG/counters
    are meaningless past that point but the weights themselves are the
    permanent, reusable artifact and must stay (unlike the old wrapper
    file, which was simply deleted outright)."""
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
    """See text_to_text.py / enc_pre.py for the traceback that motivated
    this (a torch.save write failure on a flaky network filesystem left a
    job hung for its entire wall-clock allocation instead of crashing
    cleanly). Write to a temp file and rename into place, retrying on
    transient failure."""
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
                    val_loss=None, epoch_complete=False):
    """Persist ``combine``/``router``/``expert_registry``/``text_output`` as
    four standalone, directly-loadable files (each via that module's own
    store_weights -- the same pattern text_encoder/image_encoder already
    use) under the 'latest' tag, always, and also under 'best' when
    val_loss improves on best_val_loss. There is no separate
    resumable-wrapper file: the optimizer state and epoch/batch/step
    counters are stored as an ``extra`` payload embedded in
    ``img2t_expert_registry``'s *'latest' tag only* (see
    ExpertRegistry.store_weights) -- 'best' always stays a clean, minimal
    weights-only file, since another script warm-starting from it never
    wants stale optimizer state, while 'latest' doubles as this run's own
    resume point. Only ever one 'latest' and one 'best' file exist per
    component at a time -- the previous file for that exact tag is deleted
    right after the new one lands successfully (see
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
    recoverable by just trying again at the next checkpoint interval."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    training_state = {
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "batch": batch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "rng": torch.get_rng_state(),
    }
    text_output = model.expert_registry.get_expert("text_output")

    tags = ["latest"]
    if val_loss is not None and val_loss <= best_val_loss:
        tags.append("best")

    try:
        for tag in tags:
            cp = weight_path("img2t_combine", tag, epoch, batch, val_loss, epoch_complete=epoch_complete)
            rp = weight_path("img2t_router", tag, epoch, batch, val_loss, epoch_complete=epoch_complete)
            ep = weight_path("img2t_expert_registry", tag, epoch, batch, val_loss, epoch_complete=epoch_complete)
            tp = weight_path("img2t_text_output", tag, epoch, batch, val_loss, epoch_complete=epoch_complete)

            model.combine.store_weights(WEIGHTS_DIR, os.path.basename(cp))
            model.router.store_weights(WEIGHTS_DIR, os.path.basename(rp))
            model.expert_registry.store_weights(
                WEIGHTS_DIR, os.path.basename(ep),
                extra=training_state if tag == "latest" else None,
            )
            text_output.store_weights(WEIGHTS_DIR, os.path.basename(tp))

            _prune_other_tagged_files("img2t_combine", tag, cp)
            _prune_other_tagged_files("img2t_router", tag, rp)
            _prune_other_tagged_files("img2t_expert_registry", tag, ep)
            _prune_other_tagged_files("img2t_text_output", tag, tp)
    except (RuntimeError, OSError) as e:
        print(f"[checkpoint] save_checkpoint failed at epoch {epoch} batch {batch} "
              f"step {global_step} even after retries ({e}); skipping this "
              f"checkpoint and continuing training -- will try again at the "
              f"next interval.")


# --------------------------------------------------------------------------- #
# Incremental LR decay (shared design with enc_pre.py / text_to_text.py /
# ocr_to_text.py)
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
# Answer normalisation, exact match, "close match" (self-contained, no extra
# deps)
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
    """ChartQA's standard "relaxed accuracy" metric: a numeric answer counts
    as correct within a 5% relative tolerance of the gold value (reading a
    bar chart as "42" vs. the annotated "41.9" shouldn't count as wrong the
    way a literal string mismatch would); falls back to exact_match for
    non-numeric answers (chart labels, categories, yes/no)."""
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


def compute_reward_metric(source, pred_text, target_text, aux):
    """Task-specific reward/metric, dispatched off the 'source' column.
    Non-differentiable; used only for logging/validation, never for the
    training gradient (see module docstring)."""
    aliases = [a for a in (aux or "").split("|") if a]
    gold = aliases if aliases else [target_text]
    if source == "chartqa":
        return 1.0 if close_match(pred_text, gold) else 0.0
    # geoqa, clevr: exact match on the answer string. ai2d, scienceqa: "MCQ
    # accuracy" collapses to the same exact_match check, since their
    # training target is already just the gold option's letter (see
    # _format_mcq / _mcq_answer_letter) -- a separate MCQ-specific metric
    # would be redundant.
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
# Row helpers + image loading + MCQ formatting
# --------------------------------------------------------------------------- #
def _truncate_words(text, max_words):
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def _mkrow(question, answers, source, image):
    question = _truncate_words(str(question).strip(), HYPERPARAMS["global"]["max_input_len"])
    answers = [str(a).strip() for a in answers if a is not None and str(a).strip()]
    if not question or not answers or image is None:
        return None
    return {"input": question, "target": answers[0], "source": source,
           "aux": "|".join(dict.fromkeys(answers)), "image": image}


_MCQ_LETTERS = "ABCDEFGH"


def _format_mcq(question, choices):
    opts = "\n".join(f"{_MCQ_LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return f"{question}\n{opts}\nAnswer with the option's letter."


def _mcq_answer_letter(answer, choices):
    """Normalises an MCQ gold answer -- already a letter, a 0-based or
    1-based choice index, or (some benchmark packagings) the literal text of
    the correct choice -- to a single uppercase letter matching
    _format_mcq's lettering. Returns None if it can't be resolved."""
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
    if 1 <= idx <= n:          # 1-based fallback
        return _MCQ_LETTERS[idx - 1]
    return None


def _load_image_tensor(img, size):
    """PIL image (any mode, any aspect ratio) -> float tensor [1, 3, size,
    size] in [0, 1] -- batch-of-one, ready for model.encode(..., "vision").
    Plain resize (not aspect-preserving-pad like ocr_to_text.py's document
    pages) -- matches enc_pre.py's own _load_image_tensor convention, which
    is what the frozen vision encoder was actually trained on."""
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
# guess doesn't take the other four sources down with it.
# --------------------------------------------------------------------------- #
def load_chartqa(cap):
    from datasets import load_dataset

    rows = []
    for split in ("train", "val"):
        try:
            ds = load_dataset(CHARTQA_HF, split=split, streaming=True, cache_dir=HF_CACHE)
        except Exception as e:
            print(f"[data] SKIP chartqa split '{split}' ({e})")
            continue
        n = 0
        for r in ds:
            img = r.get("image")
            question = r.get("query") or r.get("question")
            label = r.get("label")
            if label is None:
                label = r.get("answer")
            answers = label if isinstance(label, list) else ([label] if label else [])
            row = _mkrow(question, answers, "chartqa", img)
            if row:
                rows.append(row)
                n += 1
            if cap is not None and len(rows) >= cap:
                break
        print(f"[data]   chartqa[{split}]: {n} rows")
        if cap is not None and len(rows) >= cap:
            break
    if not rows:
        raise RuntimeError("chartqa yielded no usable rows")
    return rows


def _download_geometry3k():
    """Downloads + extracts the original Geometry3K dataset once, cached
    under DATA_DIR/geometry3k_raw (see _materialize_image_pool's identical
    "download once, reuse" idiom in enc_pre.py for the rationale). Needs
    gdown (``pip install gdown``) since the release is Google-Drive-hosted,
    not a direct HTTP link -- see GEOMETRY3K_GDRIVE_FOLDER_ID's comment.

    Some clusters' outbound proxy filters Google Drive's large-file-download
    host (doc-*-docs.googleusercontent.com) -- confirmed via a real run
    (slurm-4781715): ProxyError/"Tunnel connection failed: 403 Filtered",
    not a Drive-side access denial (CLEVR's direct-HTTPS download through
    the same proxy, a different host, worked fine in that same run). If
    gdown can't get through, download train.zip/val.zip/test.zip from
    GEOMETRY3K_GDRIVE_FOLDER_ID's Drive folder on a machine that isn't
    behind that proxy and scp them directly to
    DATA_DIR/geometry3k_raw/<split>.zip -- this function checks for
    already-downloaded zips before ever calling gdown, and extracts
    whichever split zips it finds regardless of where they came from."""
    raw_dir = os.path.join(DATA_DIR, "geometry3k_raw")
    os.makedirs(raw_dir, exist_ok=True)
    already_extracted = any(os.path.isdir(os.path.join(raw_dir, s))
                            for s in ("train", "val", "test"))
    if not already_extracted:
        # The Drive folder holds train.zip/val.zip/test.zip plus
        # logic_forms.zip/symbols.zip (InterGPS's own formal-predicate
        # annotations for its symbolic solver) -- only the three split zips
        # are needed here, this project only consumes each problem's plain
        # problem_text/choices/answer. Only calls gdown if at least one
        # split zip is actually missing -- a manual scp of all three skips
        # the network call entirely.
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
    """Geometry3K -- see _download_geometry3k and GEOMETRY3K_GDRIVE_FOLDER_ID's
    comment for why this (not the original chen-judge/GeoQA) is the source
    here. Schema confirmed directly against the InterGPS repo's own
    annotation-tool example set (annotation_tool/data_collection/
    data_examples/11/data.json): each problem is a numbered folder holding
    data.json ({"problem_text", "choices": [4 strings], "answer": "A"-"D"})
    plus img_problem.png/img_diagram.png -- a plain 4-choice MCQ, same shape
    as ai2d/scienceqa below (_format_mcq/_mcq_answer_letter). Deliberately
    ignores each folder's logic_form.json (InterGPS's own formal-predicate
    annotations for its symbolic solver) -- irrelevant here, this project
    only needs the natural-language question/answer pair. Source tag stays
    "geoqa" (not renamed to "geometry3k") since it's a drop-in replacement
    for the same task role -- avoids touching every other "geoqa"-keyed
    reference across this file/HYPERPARAMS/REASONING_SOURCES-equivalents."""
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
            row = _mkrow(_format_mcq(question, choices), [letter], "geoqa", img)
            if row:
                rows.append(row)
                n += 1
            if cap is not None and len(rows) >= cap:
                break
        print(f"[data]   geoqa (Geometry3K)/{split}: {n} rows")
        if cap is not None and len(rows) >= cap:
            break
    if not rows:
        raise RuntimeError("geoqa (Geometry3K) yielded no usable rows")
    return rows


def _download_clevr():
    """Downloads + extracts the original CLEVR v1.0 release once, cached
    under DATA_DIR/clevr_raw. Direct HTTPS from Meta/FAIR's own CDN (no
    auth/quota concerns, unlike Geometry3K's Google Drive hosting) -- but
    large (18GB with images), so this is a real one-time cost; see
    test_datasets.py's precompute-cost estimator pattern for how to budget
    for a source like this before a real training run."""
    import zipfile
    import time
    import requests

    raw_dir = os.path.join(DATA_DIR, "clevr_raw")
    if os.path.isdir(raw_dir) and os.path.isdir(os.path.join(raw_dir, "CLEVR_v1.0")):
        return os.path.join(raw_dir, "CLEVR_v1.0")
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "CLEVR_v1.0.zip")
    if not os.path.isfile(zip_path):
        print(f"[data] downloading CLEVR v1.0 from {CLEVR_URL} (18GB, one-time)...", flush=True)
        t0 = time.time()
        with requests.get(CLEVR_URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        size_gb = os.path.getsize(zip_path) / 1e9
        elapsed = time.time() - t0
        print(f"[data] CLEVR download done: {size_gb:.1f}GB in {elapsed/60:.1f} min "
              f"({size_gb*1000/elapsed:.1f} MB/s) -- this is a ONE-TIME cost, "
              f"see test_datasets.py's precompute-cost estimator to decide "
              f"whether this needs to run as a separate job.", flush=True)
    print("[data] extracting CLEVR_v1.0.zip...", flush=True)
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    print(f"[data] CLEVR extraction done in {(time.time()-t0)/60:.1f} min", flush=True)
    return os.path.join(raw_dir, "CLEVR_v1.0")


def load_clevr(cap):
    """Original CLEVR (Johnson et al., CVPR 2017) -- see _download_clevr
    and CLEVR_URL's comment for why this (not the unverified
    HuggingFaceM4/CLEVR HF id) is the source here. Schema per the official
    project page: questions/CLEVR_<split>_questions.json holds a top-level
    "questions" list, each {"image_filename", "question", "answer", ...};
    the matching image lives at images/<split>/<image_filename>. Test
    split has no released answers, so only train/val are usable here."""
    from PIL import Image as PILImage

    clevr_dir = _download_clevr()
    rows = []
    for split in ("train", "val"):
        q_path = os.path.join(clevr_dir, "questions", f"CLEVR_{split}_questions.json")
        if not os.path.isfile(q_path):
            print(f"[data] SKIP clevr split '{split}' (questions file not found)")
            continue
        with open(q_path) as f:
            questions = json.load(f)["questions"]
        n = 0
        for q in questions:
            question, answer = q.get("question"), q.get("answer")
            img_filename = q.get("image_filename")
            if not question or answer is None or not img_filename:
                continue
            img_path = os.path.join(clevr_dir, "images", split, img_filename)
            if not os.path.isfile(img_path):
                continue
            img = PILImage.open(img_path).convert("RGB")
            row = _mkrow(question, [str(answer)], "clevr", img)
            if row:
                rows.append(row)
                n += 1
            if cap is not None and len(rows) >= cap:
                break
        print(f"[data]   clevr/{split}: {n} rows")
        if cap is not None and len(rows) >= cap:
            break
    if not rows:
        raise RuntimeError("clevr yielded no usable rows")
    return rows


def load_ai2d(cap):
    from datasets import load_dataset

    try:
        ds = load_dataset(AI2D_HF, split="test", streaming=True, cache_dir=HF_CACHE)
    except Exception as e:
        raise RuntimeError(f"could not load '{AI2D_HF}' ({e})") from e
    rows = []
    for r in ds:
        img = r.get("image")
        question = r.get("question")
        choices = r.get("options") or r.get("choices")
        answer = r.get("answer")
        if answer is None:
            answer = r.get("correct_answer")
        if not question or not choices or answer is None or img is None:
            continue
        letter = _mcq_answer_letter(answer, choices)
        if letter is None:
            continue
        row = _mkrow(_format_mcq(question, choices), [letter], "ai2d", img)
        if row:
            rows.append(row)
        if cap is not None and len(rows) >= cap:
            break
    if not rows:
        raise RuntimeError("ai2d yielded no usable rows")
    return rows


def load_scienceqa(cap):
    from datasets import load_dataset

    rows = []
    for split in ("train", "validation"):
        try:
            ds = load_dataset(SCIENCEQA_HF, split=split, streaming=True, cache_dir=HF_CACHE)
        except Exception as e:
            print(f"[data] SKIP scienceqa split '{split}' ({e})")
            continue
        n = 0
        for r in ds:
            img = r.get("image")
            if img is None:
                continue     # text-only ScienceQA rows aren't usable here (image required)
            question = r.get("question")
            choices = r.get("choices")
            answer = r.get("answer")
            if not question or not choices or answer is None:
                continue
            letter = _mcq_answer_letter(answer, choices)
            if letter is None:
                continue
            hint = r.get("hint")
            q_text = f"{question}\n{hint}" if hint else question
            row = _mkrow(_format_mcq(q_text, choices), [letter], "scienceqa", img)
            if row:
                rows.append(row)
                n += 1
            if cap is not None and len(rows) >= cap:
                break
        print(f"[data]   scienceqa[{split}]: {n} rows")
        if cap is not None and len(rows) >= cap:
            break
    if not rows:
        raise RuntimeError("scienceqa yielded no usable rows")
    return rows


SOURCE_LOADERS = {
    "chartqa": lambda cfg: load_chartqa(cfg["data"]["chartqa_cap"]),
    "geoqa": lambda cfg: load_geoqa(cfg["data"]["geoqa_cap"]),
    "clevr": lambda cfg: load_clevr(cfg["data"]["clevr_cap"]),
    "ai2d": lambda cfg: load_ai2d(cfg["data"]["ai2d_cap"]),
    "scienceqa": lambda cfg: load_scienceqa(cfg["data"]["scienceqa_cap"]),
}


# --------------------------------------------------------------------------- #
# Compile (download + normalise + shuffle) and split (train/val, stratified
# per source -- like text_to_text.py, unlike ocr_to_text.py's single source)
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
        raise RuntimeError("No image_to_text data sources could be prepared.")

    # Inbuilt shuffling of the entire compiled dataset (interleaves sources).
    random.Random(HYPERPARAMS["global"]["seed"]).shuffle(all_rows)

    from datasets import Dataset
    Dataset.from_list(all_rows).save_to_disk(COMPILED_DIR)
    print(f"[data] wrote {len(all_rows)} compiled rows -> {COMPILED_DIR}")


def split_dataset():
    if os.path.isdir(TRAIN_DIR) and os.path.isdir(VAL_DIR):
        print("[data] train/val split already present")
        return

    from datasets import Dataset, load_from_disk

    rows = list(load_from_disk(COMPILED_DIR))
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
# Frozen encoders (loaded from enc_pre.py's best checkpoints)
# --------------------------------------------------------------------------- #
class FrozenEncoderWrapper(nn.Module):
    """Wraps a frozen encoder. Runs its forward pass under fp16 autocast on
    CUDA (instead of a literal .half() conversion) when half precision is
    enabled, and always stays in eval mode regardless of the outer model's
    train()/eval() state (mirrors ocr_to_text.py's FrozenEncoderWrapper,
    identical for the same reason -- this script also freezes two different
    encoder kinds)."""

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
        self.encoder.eval()          # frozen encoder never leaves eval mode
        return self


class FrozenTextEncoderWrapper(FrozenEncoderWrapper):
    def forward_tokens(self, *args, **kwargs):
        return self.encoder.forward_tokens(*args, **kwargs)


def build_model(device, d_model, half_precision):
    from models.OverallModel import MultimodalModel
    from models.Encoders.TextEncoder import createTextEnc
    from models.Encoders.VisionEncoder import createVisionEncoder

    model = MultimodalModel(dim=d_model, text_only=True).to(device)

    # ---- Frozen text encoder (enc_pre.py's text_train stage) --------------- #
    raw_text = createTextEnc(dim=d_model).to(device)
    best_text = find_weights("text_encoder", "best") or find_weights("text_encoder", "latest")
    if best_text:
        raw_text.load_weights(best_text, map_location=device)
        print(f"[img2t] frozen text encoder loaded from {os.path.basename(best_text)}")
    else:
        print("[img2t] WARNING: no trained text encoder found via optimal grep in "
              f"{WEIGHTS_DIR}; using a randomly initialised (untrained) text encoder")
    raw_text.eval()
    for p in raw_text.parameters():
        p.requires_grad_(False)
    frozen_text = FrozenTextEncoderWrapper(raw_text, half_precision).to(device)
    model.encoder_registry.remove_encoder("text")
    model.encoder_registry.add_encoder("text", frozen_text)

    # ---- Frozen vision encoder (enc_pre.py's image_train stage) ------------ #
    # createVisionEncoder's architecture defaults must match what enc_pre.py
    # used when it trained this checkpoint -- both files rely on the same
    # factory defaults (see models/Encoders/VisionEncoder.py) and neither
    # overrides them, so this stays in sync automatically. Registered under
    # the "vision" modality key (see models/OverallModel.py's "Modality
    # keys" docstring) -- NOT "image", which is only enc_pre.py's internal
    # HYPERPARAMS/task-kind name for this encoder.
    raw_vision = createVisionEncoder(dim=d_model).to(device)
    best_vision = find_weights("image_encoder", "best") or find_weights("image_encoder", "latest")
    if best_vision:
        raw_vision.load_weights(best_vision, map_location=device)
        print(f"[img2t] frozen vision encoder loaded from {os.path.basename(best_vision)}")
    else:
        print("[img2t] WARNING: no trained vision encoder found via optimal grep in "
              f"{WEIGHTS_DIR}; using a randomly initialised (untrained) vision encoder")
    raw_vision.eval()
    for p in raw_vision.parameters():
        p.requires_grad_(False)
    frozen_vision = FrozenEncoderWrapper(raw_vision, half_precision).to(device)
    model.encoder_registry.add_encoder("vision", frozen_vision)

    model.expert_registry.set_pipeline(
        text_encoder=frozen_text, combiner=model.combine, router=model.router,
    )

    # ---- Warm-start combine/router/expert_registry from this script's own -- #
    # ---- prior progress, if any (fresh stage start otherwise) -------------- #
    # "best" first, "latest" as a fallback (e.g. early in a run before any
    # val_loss improvement has landed yet, only "latest" exists) -- this is
    # the generic "give me the best available starting point" lookup used
    # everywhere else in this project (frozen encoders, text_output). A run
    # resuming its own exact mid-epoch progress needs "latest" specifically
    # and overrides this in train() (see train()'s resume logic) --
    # build_model() itself doesn't need to know or care whether it's being
    # called for a fresh stage, a resume, or another script reusing these
    # weights entirely.
    def _warm_start(module, component_name, label):
        # Same corrupt-file tolerance as _load_weights_with_fallback (a
        # truncated write from a flaky GPFS save shouldn't crash every
        # subsequent run that warm-starts from it) -- order is "best" then
        # "latest" here rather than _load_weights_with_fallback's
        # "latest"-then-"best", since build_model() wants the generic best
        # available starting point, not this exact run's precise progress.
        for tag in ("best", "latest"):
            path = find_weights(component_name, tag)
            if not path:
                continue
            try:
                module.load_weights(path, map_location=device)
                print(f"[img2t] warm-started {label} from {os.path.basename(path)}")
                return
            except Exception as e:
                print(f"[img2t] {os.path.basename(path)} is unreadable ({e}); "
                      f"deleting it and trying an older checkpoint")
                try:
                    os.remove(path)
                except OSError:
                    pass
        print(f"[img2t] no {label} checkpoint found; using a fresh (randomly initialised) one")

    _warm_start(model.combine, "img2t_combine", "combine")
    _warm_start(model.router, "img2t_router", "router")
    _warm_start(model.expert_registry, "img2t_expert_registry", "expert_registry")

    # Belt-and-suspenders: add_encoder() now matches the registered module's
    # own device (see EncRegistry.py), but re-syncing the whole model here
    # too costs nothing and catches this whole bug class regardless of how
    # a future encoder swap is implemented.
    model = model.to(device)
    return model


# --------------------------------------------------------------------------- #
# Core forward helpers (mirrors ocr_to_text.py's encode_to_state / teacher
# forcing, just encoding the image under the "vision" modality instead of
# "ocr")
# --------------------------------------------------------------------------- #
def encode_to_state(model, image_tensor, question):
    encodings = model.encode(image_tensor, "vision")
    encodings = model.encode(question, "text", encodings)
    combined, _, _ = model.combine(encodings)
    _, state = model.router(combined)
    return state


def teacher_forced_forward(expert, state, tgt_in):
    device = tgt_in.device
    T = tgt_in.size(1)
    emb = expert.token_embedding(tgt_in) * math.sqrt(expert.d_model)
    emb = emb + expert._sincos_pos_embed(T, device)
    tgt_mask = expert._causal_mask(T, device)
    weighted_h, ponder_cost = expert._recursive_decode(emb, state, None, tgt_mask)
    logits = expert.output_proj(weighted_h)
    return logits, ponder_cost


def compute_task_loss(expert, state, target_text, codec, device, coefs, label_smoothing):
    """Composite training objective: teacher-forced CE (the differentiable
    proxy for the exact-match / close-match reward) plus small length,
    ponder-cost and Brier-calibration terms -- identical to
    text_to_text.py's / ocr_to_text.py's version, just with a smaller
    brier_coef (see HYPERPARAMS["reward"])."""
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
    )
    return loss, ce.detach().item()


# --------------------------------------------------------------------------- #
# Self-correction: generate, intentionally corrupt, then train the model to
# catch its own mistake (see text_to_text.py's module docstring).
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


def self_correct_step(model, expert, codec, device, image_tensor, question, gold_answer, coefs, rng):
    with torch.no_grad():
        state = encode_to_state(model, image_tensor, question)
        gen_text, _ = expert.forward(state, max_length=HYPERPARAMS["global"]["max_target_len"])
        candidate = (gen_text[0] if gen_text else "").strip() or gold_answer

    corrupted = rng.random() < 0.5
    shown = corrupt_answer(candidate, rng) if corrupted else candidate
    verify_input = (
        f"Question: {question}\nProposed answer: {shown}\n"
        f"Is the proposed answer correct? Reply with exactly 'yes' or 'no'."
    )
    verify_target = "no" if corrupted else "yes"

    v_state = encode_to_state(model, image_tensor, verify_input)
    loss, _ = compute_task_loss(
        expert, v_state, verify_target, codec, device, coefs, label_smoothing=0.0,
    )
    return loss


# --------------------------------------------------------------------------- #
# Validation (teacher-forced CE, drives checkpoint/valloss) + reward metrics
# (free-run generation, logging only)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, expert, codec, device, val_rows, coefs, sample_size, image_size):
    model.eval()
    rows = sample_rows(val_rows, min(sample_size, len(val_rows)), seed=0)
    total_ce, n = 0.0, 0
    for r in rows:
        image_tensor = _load_image_tensor(r["image"], image_size).to(device)
        state = encode_to_state(model, image_tensor, r["input"])
        _, ce = compute_task_loss(expert, state, r["target"], codec, device, coefs,
                                  label_smoothing=0.0)
        total_ce += ce
        n += 1
    model.train()
    return total_ce / max(1, n)


@torch.no_grad()
def evaluate_rewards(model, expert, val_rows, sample_size, seed, image_size, device):
    model.eval()
    rows = sample_rows(val_rows, min(sample_size, len(val_rows)), seed=seed)
    by_source = {}
    # Print one real (prediction, target) pair per source the first time
    # that source comes up this call -- see text_to_text.py's identical
    # addition for why: an aggregate reward stuck near 0 is ambiguous
    # (genuinely-hard task vs. degenerate generation) without actually
    # looking at what's being generated.
    logged_sources = set()
    for r in rows:
        image_tensor = _load_image_tensor(r["image"], image_size).to(device)
        state = encode_to_state(model, image_tensor, r["input"])
        # Greedy, not the default temperature=1.0/top_k=50 sampling used for
        # training-time exploration (see self_correct_step) -- reward is an
        # accuracy-style metric measuring whether the model's *best* answer
        # is right, so it must decode the argmax token every step (top_k=1
        # collapses the post-filter softmax to one candidate, making the
        # subsequent multinomial draw deterministic). See text_to_text.py's
        # evaluate_rewards for the full rationale.
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

    print("\n===== TUNING: image_to_text =====")
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
        study_name="image_to_text",
        storage=f"sqlite:///{os.path.join(STUDY_DIR, 'image_to_text.db')}",
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
                image_tensor = _load_image_tensor(row["image"], image_size).to(device)
                state = encode_to_state(model, image_tensor, row["input"])
                loss, _ = compute_task_loss(expert, state, row["target"], codec, device,
                                            coefs, hp["label_smoothing"])
                (loss / hp["batch_size"]).backward()
                accum += 1
                if accum == hp["batch_size"]:
                    optimizer.step()
                    optimizer.zero_grad()
                    accum = 0

            val_loss = validate(model, expert, codec, device, tune_val, coefs,
                                sample_size=len(tune_val), image_size=image_size)
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
    out = os.path.join(STUDY_DIR, "image_to_text_best_hparams.json")
    with open(out, "w") as f:
        json.dump(best_hp, f, indent=2)
    print(f"\nBest image_to_text hyperparameters: {best_hp}")
    print(f"Saved to {out}")
    return best_hp


def load_best_hparams():
    path = os.path.join(STUDY_DIR, "image_to_text_best_hparams.json")
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
def train(device, rank=0, world_size=1):
    from dist_utils import barrier, broadcast_parameters, is_main_process, rank0_first, sync_gradients

    is_distributed = world_size > 1
    main = is_main_process(rank)
    if main:
        print("\n===== TRAINING: image_to_text =====")
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
    expert = model.expert_registry.get_expert("text_output")
    codec = SpmCodec(max_len=gcfg["max_target_len"])

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=hp["learning_rate"], weight_decay=hp["weight_decay"],
    )

    epochs = HYPERPARAMS["epochs"]
    batch_size = hp["batch_size"]
    # Per-rank row count: truncated to a multiple of (batch_size * world_size)
    # so every rank's shard is itself an exact multiple of batch_size -- see
    # text_to_text.py's train() for the full reasoning (identical scheme).
    usable_rows = len(train_rows) - (len(train_rows) % (batch_size * world_size))
    rows_per_rank = usable_rows // world_size
    n_opt_steps_per_epoch = max(1, rows_per_rank // batch_size)
    total_steps = n_opt_steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(hp["warmup_steps"], total_steps, hp["lr_min_scale"]),
    )

    # ---- Resume from the latest checkpoint if present ---------------------- #
    # build_model() already warm-started combine/router/expert_registry from
    # whatever's on disk (preferring 'best'). If img2t_expert_registry's own
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
    er_path, training_state = _load_weights_with_fallback(model.expert_registry, "img2t_expert_registry", device)
    if training_state:
        for module, name in ((model.combine, "img2t_combine"), (model.router, "img2t_router")):
            _load_weights_with_fallback(module, name, device)
        optimizer.load_state_dict(training_state["optimizer_state"])
        # training_state was loaded via ExpertRegistry.load_weights(...,
        # map_location=device) -- on a GPU rank that remaps every tensor in
        # the checkpoint, including this one, onto the GPU. set_rng_state()
        # rejects anything but a plain CPU torch.ByteTensor, so it must be
        # moved back explicitly regardless of where map_location put it.
        torch.set_rng_state(training_state["rng"].cpu())
        global_step, best_val = training_state["global_step"], training_state["best_val_loss"]
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
            image_tensor = _load_image_tensor(row["image"], image_size).to(device)
            state = encode_to_state(model, image_tensor, row["input"])
            loss, ce = compute_task_loss(expert, state, row["target"], codec, device,
                                         coefs, hp["label_smoothing"])

            if rng.random() < self_correct_p:
                loss = loss + coefs["self_correct_coef"] * self_correct_step(
                    model, expert, codec, device, image_tensor, row["input"],
                    row["target"], coefs, rng,
                )

            (loss / batch_size).backward()
            accum += 1

            if accum == batch_size:
                sync_gradients(model.parameters(), world_size)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum = 0
                batch_idx += 1
                global_step += 1

                if main and global_step % 50 == 0:
                    print(f"  e{epoch} row{i} step{global_step} "
                          f"source={row['source']} ce={ce:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")
                    wandb_log({"train/ce_loss": ce, "train/lr": scheduler.get_last_lr()[0],
                              "epoch": epoch, "row": i})

                # Rank-0-only: no collective involved, so other ranks just
                # move on to their next row instead of waiting.
                if main and global_step % eval_every == 0:
                    val = validate(model, expert, codec, device, val_rows, coefs,
                                   val_sample, image_size)
                    best_val = min(best_val, val)
                    rewards = evaluate_rewards(model, expert, val_rows, sample_size=200,
                                               seed=global_step, image_size=image_size,
                                               device=device)
                    print(f"  [val] step {global_step}: val_loss={val:.4f} "
                          f"(best={best_val:.4f}) rewards="
                          f"{ {k: round(v, 3) for k, v in rewards.items()} }")
                    wandb_log({"val/loss": val,
                              **{f"val/reward_{k}": v for k, v in rewards.items()}})
                    save_checkpoint(model, optimizer, epoch, i + 1, global_step, best_val,
                                    val_loss=val)
                elif main and global_step % ckpt_every == 0:
                    save_checkpoint(model, optimizer, epoch, i + 1, global_step, best_val)

        # ---- End-of-epoch validation + checkpoint --------------------------- #
        barrier(is_distributed)
        start_row = 0
        if not main:
            continue
        val = validate(model, expert, codec, device, val_rows, coefs, val_sample, image_size)
        best_val = min(best_val, val)
        rewards = evaluate_rewards(model, expert, val_rows, sample_size=300, seed=epoch,
                                   image_size=image_size, device=device)
        print(f"[epoch {epoch}] val_loss={val:.4f} (best={best_val:.4f}) rewards={rewards}")
        wandb_log({"val/epoch_loss": val, "epoch": epoch})
        save_checkpoint(model, optimizer, epoch, len(order), global_step, best_val,
                        val_loss=val, epoch_complete=True)

    barrier(is_distributed)
    if not main:
        return
    # 'best'/'latest' are maintained continuously by save_checkpoint (see its
    # docstring), so no finalize step is needed here -- just read them. The
    # embedded training-state payload is no longer needed now that training
    # has genuinely finished, so it's stripped from img2t_expert_registry's
    # tagged files (keeping the weights themselves, unlike the old wrapper
    # file, which was dropped outright).
    _strip_training_state("img2t_expert_registry")
    best_text_output = find_weights("img2t_text_output", "best")
    latest_text_output = find_weights("img2t_text_output", "latest")
    print("\n[done] image_to_text training complete.")
    print(f"[optimal weights] text_output -> {best_text_output or latest_text_output}")
    print(f"[latest weights] text_output -> {latest_text_output}")


# --------------------------------------------------------------------------- #
# Stage driver
# --------------------------------------------------------------------------- #
def run_stages(args, device, rank=0, world_size=1):
    """The tune stage stays single-process (rank 0 only, other ranks idle at
    a barrier) -- see dist_utils.py's module docstring. The train stage runs
    on every rank via train()'s own distributed handling."""
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

    if "train" not in progress["completed"]:
        if main:
            init_wandb(args.wandb_key, "train")
        train(device, rank=rank, world_size=world_size)
        if main:
            mark_stage_done("train")
            finish_wandb()

    if main:
        print("\nAll image_to_text stages complete.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Image-to-text fine-tuning.")
    p.add_argument("--token", default="_", help="Hugging Face token.")
    p.add_argument("--wandb_key", default="_", help="Weights & Biases API key.")
    p.add_argument("--prepare_data_only", action="store_true",
                   help="Compile + split the dataset, then exit.")
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
