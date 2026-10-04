"""
OCR-to-text fine-tuning for the ensemble multimodal research model.

Overview
--------
This script pairs two encoders pretrained *in isolation* by enc_pre.py --
the frozen OCR encoder (``ocr_train`` stage) and the frozen text encoder
(``text_train`` stage) -- into a document-QA fine-tuning task: "ocr" modality
(a rendered document page) plus "text" modality (a question about it) are
encoded, combined, routed, and decoded through the project's ``TextOutput``
expert, mirroring text_to_text.py's reward/loss machinery exactly, just with
a second frozen encoder feeding the combine stage. This is deliberately a
*separate* script from both enc_pre.py (which never pairs the OCR encoder
into anything -- see its module docstring) and text_to_text.py -- run this
one after text_to_text.py, once both frozen encoders it depends on exist.

Pipeline (staged; stops for review after the tuning stage, like text_to_text.py):

    1. tune  -> Optuna search for training hyperparameters (on a small
                subset of the compiled data), then exit(0) for review.
    2. train -> full resumable training run over the entire compiled
                dataset, with checkpoints after every optimizer step and
                after every completed epoch.

Data pipeline (arrow dataset on scratch, not CSV)
---------------------------------------------------
Unlike text_to_text.py's rows (pure text -> a CSV is the natural format),
every row here also carries a document *image*, so the compiled dataset is
a HuggingFace ``datasets.Dataset`` (arrow format, handles embedded images
natively) saved via ``save_to_disk``/loaded via ``load_from_disk`` instead
of a CSV -- same "compile once, shuffle, then split into train/val" shape
as text_to_text.py, just the right storage format for this data type. Each
row is ``{input, target, source, aux, image}``: ``input`` is the question,
``target`` the primary answer, ``aux`` the pipe-joined alternate answers
(DocVQA questions commonly have several valid phrasings -- see
``compute_reward_metric``), ``image`` the document page.

Source: DocVQA (``DOCVQA_HF``) -- real document images with question/answer
pairs (see OCR_Encoding_Expert_Methodology.md's Dataset Composition section,
which references the same corpus). Loaded defensively: if it cannot be
fetched, compilation fails loudly rather than silently training on nothing,
since (unlike text_to_text.py's six independent sources) there is only one
source here.

Reward/loss machinery, checkpoint naming and finalize-best-and-latest
cleanup, resumability, and the incremental warmup-then-cosine LR schedule
are all identical in spirit to text_to_text.py (see that file's module
docstring for the full reasoning) -- duplicated here rather than imported,
matching this project's established "self-contained scripts" convention
(enc_pre.py and text_to_text.py already duplicate this same machinery
between themselves rather than sharing a common module).

At the end of training the script prints the located best ``ocr2t_text_output``
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
DATA_DIR = os.path.join(SCRATCH_DIR, "ocr_to_text_data")
STUDY_DIR = os.path.join(PROJECT_DIR, "studies", "ocr_to_text")
# All weight files -- including the resumable "_wrapper" checkpoints --
# live here on barn, never on scratch. scratch is reserved strictly for
# datasets/HF cache. Shared with enc_pre.py / text_to_text.py.
WEIGHTS_DIR = os.path.join(PROJECT_DIR, "weights")
HF_CACHE = os.path.join(DATA_DIR, "hf_cache")
PROGRESS_PATH = os.path.join(PROJECT_DIR, "ocr_to_text_progress.json")

COMPILED_DIR = os.path.join(DATA_DIR, "compiled")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

TOKENIZER_PATH = os.path.join(MODELS_DIR, "Encoders", "multimodal_bpe.model")
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

DOCVQA_HF = "lmms-lab/DocVQA"
DOCVQA_CONFIG = "DocVQA"


# =========================================================================== #
# Hyperparameters
# =========================================================================== #
HYPERPARAMS = {
    "global": {
        "d_model": 1024,
        "seed": 42,
        "max_input_len": 96,             # word cap applied when compiling rows
        "max_target_len": 32,            # token cap for teacher-forced targets
        # Document page render resolution fed to the frozen OCR encoder --
        # must roughly match what enc_pre.py's ocr_train stage pretrained it
        # on (see enc_pre.py HYPERPARAMS["ocr"]["image_size"]); the encoder
        # itself tolerates other resolutions (it pads internally) but
        # performs best close to its training distribution.
        "image_size": 896,
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
        "brier_coef": 0.05,
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
        "docvqa_cap": None,   # None = use the entire available split
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
            project="ocr_to_text", name=stage, reinit=True, config=HYPERPARAMS,
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
# enc_pre.py / text_to_text.py -- identical scheme, see either file's
# module docstring for the full rationale).
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
    store_weights -- the same pattern text_encoder/ocr_encoder already use)
    under the 'latest' tag, always, and also under 'best' when val_loss
    improves on best_val_loss. There is no separate resumable-wrapper file:
    the optimizer state and epoch/batch/step counters are stored as an
    ``extra`` payload embedded in ``ocr2t_expert_registry``'s *'latest' tag
    only* (see ExpertRegistry.store_weights) -- 'best' always stays a
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
            cp = weight_path("ocr2t_combine", tag, epoch, batch, val_loss, epoch_complete=epoch_complete)
            rp = weight_path("ocr2t_router", tag, epoch, batch, val_loss, epoch_complete=epoch_complete)
            ep = weight_path("ocr2t_expert_registry", tag, epoch, batch, val_loss, epoch_complete=epoch_complete)
            tp = weight_path("ocr2t_text_output", tag, epoch, batch, val_loss, epoch_complete=epoch_complete)

            model.combine.store_weights(WEIGHTS_DIR, os.path.basename(cp))
            model.router.store_weights(WEIGHTS_DIR, os.path.basename(rp))
            model.expert_registry.store_weights(
                WEIGHTS_DIR, os.path.basename(ep),
                extra=training_state if tag == "latest" else None,
            )
            text_output.store_weights(WEIGHTS_DIR, os.path.basename(tp))

            _prune_other_tagged_files("ocr2t_combine", tag, cp)
            _prune_other_tagged_files("ocr2t_router", tag, rp)
            _prune_other_tagged_files("ocr2t_expert_registry", tag, ep)
            _prune_other_tagged_files("ocr2t_text_output", tag, tp)
    except (RuntimeError, OSError) as e:
        print(f"[checkpoint] save_checkpoint failed at epoch {epoch} batch {batch} "
              f"step {global_step} even after retries ({e}); skipping this "
              f"checkpoint and continuing training -- will try again at the "
              f"next interval.")


# --------------------------------------------------------------------------- #
# Incremental LR decay (shared design with enc_pre.py / text_to_text.py)
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
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction, gold_answers):
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(g) for g in gold_answers)


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


def compute_reward_metric(pred_text, target_text, aux):
    """Non-differentiable reward/metric used only for logging/validation,
    never for the training gradient (see module docstring). DocVQA
    questions commonly have several valid phrasings (aux = pipe-joined
    alternates, see load_docvqa) -- exact_match checks the prediction
    against every one of them."""
    aliases = [a for a in (aux or "").split("|") if a]
    gold = aliases if aliases else [target_text]
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
# Row helpers + document-image loading
# --------------------------------------------------------------------------- #
def _truncate_words(text, max_words):
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def _mkrow(question, answers, source, image):
    question = _truncate_words(str(question).strip(), HYPERPARAMS["global"]["max_input_len"])
    answers = [str(a).strip() for a in answers if str(a).strip()]
    if not question or not answers or image is None:
        return None
    return {"input": question, "target": answers[0], "source": source,
           "aux": "|".join(dict.fromkeys(answers)), "image": image}


def _load_doc_image_tensor(img, size):
    """PIL image (any mode, any aspect ratio) -> float tensor [1, 3, size,
    size] in [0, 1] -- batch-of-one, ready for model.encode(..., "ocr").
    Resized to fit within size x size preserving aspect ratio and padded
    with white, matching enc_pre.py's _load_ocr_image_tensor exactly (the
    frozen OCR encoder was pretrained on images built that same way)."""
    from PIL import Image
    import numpy as np

    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((new_w, new_h))
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, (0, 0))
    arr = torch.from_numpy(np.asarray(canvas, dtype="float32") / 255.0)
    return arr.permute(2, 0, 1).unsqueeze(0)


def sample_rows(rows, n, seed):
    rng = random.Random(seed)
    if n >= len(rows):
        return list(rows)
    return rng.sample(rows, n)


# --------------------------------------------------------------------------- #
# DocVQA loader. Raises on failure; the caller in compile_dataset() lets it
# propagate (unlike text_to_text.py's six independent sources, there is only
# one source here, so there is nothing left to fall back to).
# --------------------------------------------------------------------------- #
def load_docvqa(cap):
    from datasets import load_dataset

    rows = []
    for split in ("validation", "train"):
        try:
            ds = load_dataset(DOCVQA_HF, DOCVQA_CONFIG, split=split,
                              streaming=True, cache_dir=HF_CACHE)
        except Exception as e:
            print(f"[data] SKIP docvqa split '{split}' ({e})")
            continue
        n = 0
        for r in ds:
            img = r.get("image")
            question = r.get("question")
            answers = r.get("answers") or ([r["answer"]] if r.get("answer") else [])
            if img is None or not question or not answers:
                continue
            row = _mkrow(question, answers, "docvqa", img)
            if row:
                rows.append(row)
                n += 1
            if cap is not None and len(rows) >= cap:
                break
        print(f"[data]   docvqa[{split}]: {n} rows")
        if cap is not None and len(rows) >= cap:
            break
    if not rows:
        raise RuntimeError("docvqa yielded no usable rows")
    return rows


# --------------------------------------------------------------------------- #
# Compile (download + normalise + shuffle) and split (train/val)
# --------------------------------------------------------------------------- #
def compile_dataset():
    if os.path.isdir(COMPILED_DIR) and os.listdir(COMPILED_DIR):
        print(f"[data] compiled dataset already present at {COMPILED_DIR}")
        return
    os.makedirs(DATA_DIR, exist_ok=True)

    rows = load_docvqa(HYPERPARAMS["data"]["docvqa_cap"])
    print(f"[data] docvqa: {len(rows)} rows")

    # Inbuilt shuffling of the entire compiled dataset.
    random.Random(HYPERPARAMS["global"]["seed"]).shuffle(rows)

    from datasets import Dataset
    Dataset.from_list(rows).save_to_disk(COMPILED_DIR)
    print(f"[data] wrote {len(rows)} compiled rows -> {COMPILED_DIR}")


def split_dataset():
    if os.path.isdir(TRAIN_DIR) and os.path.isdir(VAL_DIR):
        print("[data] train/val split already present")
        return

    from datasets import Dataset, load_from_disk

    rows = list(load_from_disk(COMPILED_DIR))
    rng = random.Random(HYPERPARAMS["global"]["seed"] + 1)
    rng.shuffle(rows)
    val_frac = HYPERPARAMS["global"]["val_fraction"]
    n_val = max(1, int(len(rows) * val_frac))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"[data] docvqa: {len(train_rows)} train / {len(val_rows)} val "
          f"({len(rows)} total)")

    Dataset.from_list(train_rows).save_to_disk(TRAIN_DIR)
    Dataset.from_list(val_rows).save_to_disk(VAL_DIR)


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
    train()/eval() state (mirrors text_to_text.py's FrozenTextEncoderWrapper,
    generalised since this script freezes two different encoder kinds)."""

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
    from models.Encoders.OCREncoder import createOCREncoder

    model = MultimodalModel(dim=d_model, text_only=True).to(device)

    # ---- Frozen text encoder (enc_pre.py's text_train stage) --------------- #
    raw_text = createTextEnc(dim=d_model).to(device)
    best_text = find_weights("text_encoder", "best") or find_weights("text_encoder", "latest")
    if best_text:
        raw_text.load_weights(best_text, map_location=device)
        print(f"[ocr2t] frozen text encoder loaded from {os.path.basename(best_text)}")
    else:
        print("[ocr2t] WARNING: no trained text encoder found via optimal grep in "
              f"{WEIGHTS_DIR}; using a randomly initialised (untrained) text encoder")
    raw_text.eval()
    for p in raw_text.parameters():
        p.requires_grad_(False)
    frozen_text = FrozenTextEncoderWrapper(raw_text, half_precision).to(device)
    model.encoder_registry.remove_encoder("text")
    model.encoder_registry.add_encoder("text", frozen_text)

    # ---- Frozen OCR encoder (enc_pre.py's ocr_train stage) ------------------ #
    # createOCREncoder's architecture defaults (local/global/refiner layer
    # counts, window_size, patch_size, num_output_tokens) must match what
    # enc_pre.py used when it trained this checkpoint -- both files rely on
    # the same factory defaults (see models/Encoders/OCREncoder.py) and
    # neither overrides them, so this stays in sync automatically.
    raw_ocr = createOCREncoder(dim=d_model).to(device)
    best_ocr = find_weights("ocr_encoder", "best") or find_weights("ocr_encoder", "latest")
    if best_ocr:
        raw_ocr.load_weights(best_ocr, map_location=device)
        print(f"[ocr2t] frozen OCR encoder loaded from {os.path.basename(best_ocr)}")
    else:
        print("[ocr2t] WARNING: no trained OCR encoder found via optimal grep in "
              f"{WEIGHTS_DIR}; using a randomly initialised (untrained) OCR encoder")
    raw_ocr.eval()
    for p in raw_ocr.parameters():
        p.requires_grad_(False)
    frozen_ocr = FrozenEncoderWrapper(raw_ocr, half_precision).to(device)
    model.encoder_registry.add_encoder("ocr", frozen_ocr)

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
                print(f"[ocr2t] warm-started {label} from {os.path.basename(path)}")
                return
            except Exception as e:
                print(f"[ocr2t] {os.path.basename(path)} is unreadable ({e}); "
                      f"deleting it and trying an older checkpoint")
                try:
                    os.remove(path)
                except OSError:
                    pass
        print(f"[ocr2t] no {label} checkpoint found; using a fresh (randomly initialised) one")

    _warm_start(model.combine, "ocr2t_combine", "combine")
    _warm_start(model.router, "ocr2t_router", "router")
    _warm_start(model.expert_registry, "ocr2t_expert_registry", "expert_registry")

    # Belt-and-suspenders: add_encoder() now matches the registered module's
    # own device (see EncRegistry.py), but re-syncing the whole model here
    # too costs nothing and catches this whole bug class regardless of how
    # a future encoder swap is implemented.
    model = model.to(device)
    return model


# --------------------------------------------------------------------------- #
# Core forward helpers (mirrors text_to_text.py's encode_to_state / teacher
# forcing, extended to also encode the document image)
# --------------------------------------------------------------------------- #
def encode_to_state(model, image_tensor, question):
    encodings = model.encode(image_tensor, "ocr")
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
    proxy for the exact-match / ROUGE-L reward) plus small length, ponder-cost
    and Brier-calibration terms -- identical to text_to_text.py's version."""
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
        image_tensor = _load_doc_image_tensor(r["image"], image_size).to(device)
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
    rewards = []
    # Print a couple of real (prediction, target) pairs every call -- see
    # text_to_text.py's identical addition for why: an aggregate reward
    # stuck near 0 is ambiguous (genuinely-hard task vs. degenerate
    # generation) without actually looking at what's being generated.
    for i, r in enumerate(rows):
        image_tensor = _load_doc_image_tensor(r["image"], image_size).to(device)
        state = encode_to_state(model, image_tensor, r["input"])
        # Greedy, not the default temperature=1.0/top_k=50 sampling used for
        # training-time exploration (see self_correct_step) -- see
        # text_to_text.py's evaluate_rewards for the full rationale.
        gen, _ = expert.forward(state, max_length=HYPERPARAMS["global"]["max_target_len"],
                                top_k=1)
        pred = gen[0] if gen else ""
        reward = compute_reward_metric(pred, r["target"], r.get("aux", ""))
        rewards.append(reward)
        if i < 2:
            print(f"  [reward-sample] docvqa: pred={pred!r} target={r['target']!r} reward={reward:.3f}")
    model.train()
    return sum(rewards) / max(1, len(rewards))


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

    print("\n===== TUNING: ocr_to_text =====")
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
        study_name="ocr_to_text",
        storage=f"sqlite:///{os.path.join(STUDY_DIR, 'ocr_to_text.db')}",
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
                image_tensor = _load_doc_image_tensor(row["image"], image_size).to(device)
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
    out = os.path.join(STUDY_DIR, "ocr_to_text_best_hparams.json")
    with open(out, "w") as f:
        json.dump(best_hp, f, indent=2)
    print(f"\nBest ocr_to_text hyperparameters: {best_hp}")
    print(f"Saved to {out}")
    return best_hp


def load_best_hparams():
    path = os.path.join(STUDY_DIR, "ocr_to_text_best_hparams.json")
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
        print("\n===== TRAINING: ocr_to_text =====")
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
    # whatever's on disk (preferring 'best'). If ocr2t_expert_registry's own
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
    er_path, training_state = _load_weights_with_fallback(model.expert_registry, "ocr2t_expert_registry", device)
    if training_state:
        for module, name in ((model.combine, "ocr2t_combine"), (model.router, "ocr2t_router")):
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
            image_tensor = _load_doc_image_tensor(row["image"], image_size).to(device)
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
                    reward = evaluate_rewards(model, expert, val_rows, sample_size=200,
                                              seed=global_step, image_size=image_size,
                                              device=device)
                    print(f"  [val] step {global_step}: val_loss={val:.4f} "
                          f"(best={best_val:.4f}) reward={reward:.3f}")
                    wandb_log({"val/loss": val, "val/reward": reward})
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
        reward = evaluate_rewards(model, expert, val_rows, sample_size=300, seed=epoch,
                                  image_size=image_size, device=device)
        print(f"[epoch {epoch}] val_loss={val:.4f} (best={best_val:.4f}) reward={reward:.3f}")
        wandb_log({"val/epoch_loss": val, "epoch": epoch})
        save_checkpoint(model, optimizer, epoch, len(order), global_step, best_val,
                        val_loss=val, epoch_complete=True)

    barrier(is_distributed)
    if not main:
        return
    # 'best'/'latest' are maintained continuously by save_checkpoint (see its
    # docstring), so no finalize step is needed here -- just read them. The
    # embedded training-state payload is no longer needed now that training
    # has genuinely finished, so it's stripped from ocr2t_expert_registry's
    # tagged files (keeping the weights themselves, unlike the old wrapper
    # file, which was dropped outright).
    _strip_training_state("ocr2t_expert_registry")
    best_text_output = find_weights("ocr2t_text_output", "best")
    latest_text_output = find_weights("ocr2t_text_output", "latest")
    print("\n[done] ocr_to_text training complete.")
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
        print("\nAll ocr_to_text stages complete.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="OCR-to-text fine-tuning.")
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
