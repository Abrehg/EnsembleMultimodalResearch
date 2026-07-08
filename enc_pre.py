"""
Encoder pretraining (text encoder + image encoder, in a vacuum).

Overview
--------
This script pretrains the project's two encoders -- ``TextEncoder`` and
``VisionEncoder`` -- *independently of the multimodal model*. Each encoder is
temporarily wrapped with its own small decoder transformer (2-4 layers) and a
set of task-specific heads, trained on a spread of tasks, and then the decoder
is thrown away: only the encoder weights are kept as the reusable artifact.

Pipeline (staged; the program stops for review after every *tuning* stage):

    1. text_tune   -> Optuna search for the text encoder's hyperparameters,
                      then exit(0) for review.
    2. text_train  -> train the text encoder wrapper (resumable), save encoder.
    3. image_tune  -> Optuna search for the image encoder's hyperparameters
                      (using the frozen best text encoder), then exit(0).
    4. image_train -> train the image encoder wrapper (resumable) with the
                      frozen text encoder, save encoder.

Progress is persisted, so a resubmitted job skips finished stages. Training is
itself resumable mid-epoch: checkpoints are labelled with epoch + batch number
and, after each validation, the validation loss is embedded in the filename.
Weights follow a fixed, greppable naming scheme (see ``weight_path``):

    {component}__{tag}__epoch{E}_batch{B}[_valloss{L}].pt
        component in {text_encoder, text_wrapper, image_encoder, image_wrapper}
        tag       in {latest, best}

``find_weights(component, tag)`` locates them programmatically.

All entries from all of an encoder's tasks are merged into a *single static
dataloader*; every entry carries a task label, and a task-homogeneous batch
sampler ensures each batch belongs to exactly one task, so only that task's
head is activated (in both tuning and training).

Datasets are downloaded + cached under SCRATCH (see below). Text datasets are
populated up front; the image datasets (all on HuggingFace) are pulled lazily
when the image-tuning stage begins. GAP coreference is not on HuggingFace -- it
is fetched from its GitHub raw TSVs (no auth needed; the cluster proxy handles
egress). Data loads are defensive: a task whose dataset cannot be fetched is
skipped with a warning so the rest of the run proceeds.

Task -> dataset map:
    Text : NER (CoNLL2003), POS + dependency relations (Universal Dependencies),
           coreference (GAP/GitHub), extractive QA (SQuAD 2.0),
           NLI (MultiNLI), discourse (SWAG), MLM (Wikipedia).
    Image: reconstruction + instance discrimination (imagenet-1k),
           depth (NYU Depth V2), visual relationships (GQA scene graph),
           semantic segmentation (ADE20K / scene_parse_150).

Only a slice of each dataset is used (exposure, not saturation): ~500k entries
per encoder overall, capped so the on-disk footprint stays well under 20GB.
"""

import argparse
import glob
import json
import os
import re
import sys

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

# Everything heavy (datasets, weights, optuna studies) lives on scratch.
SCRATCH_DIR = "/gpfs/u/home/ENMM/ENMMsrtk/scratch/"
DATA_DIR = os.path.join(SCRATCH_DIR, "enc_pre_data")
WEIGHTS_DIR = os.path.join(SCRATCH_DIR, "enc_pre_weights")
STUDY_DIR = os.path.join(SCRATCH_DIR, "enc_pre_studies")
PROGRESS_PATH = os.path.join(WEIGHTS_DIR, "progress.json")
HF_CACHE = os.path.join(DATA_DIR, "hf_cache")

TOKENIZER_PATH = os.path.join(MODELS_DIR, "Encoders", "multimodal_bpe.model")

# SentencePiece special ids (see TextEncoder / spm training).
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
VOCAB_SIZE = 32000
MASK_ID = UNK_ID  # spm has no dedicated mask token; reuse <unk> as the MLM mask


# =========================================================================== #
# HYPERPARAMETERS  (everything tunable lives here, for easy visibility)
# =========================================================================== #
HYPERPARAMS = {
    "global": {
        "d_model": 512,               # encoder width (fixed here; tuned in dim_tune)
        "val_holdout": 1000,          # entries held out for validation (variable)
        "eval_every_batches": 500,    # run validation every N optimizer steps
        "checkpoint_every_batches": 200,  # write a resumable checkpoint every N
        "seed": 42,
        "max_text_len": 64,           # token cap for text tasks
        "image_size": 128,            # images resized to this square
    },

    "text": {
        "epochs": 3,
        "per_task_cap": 80000,       # caps large tasks; ~500k total / 8 tasks
        "tune_trials": 15,
        "tune_budget_batches": 300,   # short training budget per Optuna trial
        # Fixed starting point / fallback if no tuned config is found.
        "defaults": {
            "batch_size": 16,
            "learning_rate": 3e-4,
            "weight_decay": 1e-2,
            "warmup_steps": 100,
            "decoder_layers": 3,
            "decoder_heads": 8,
            "dropout": 0.1,
        },
        # Optuna search space (consumed by suggest_* in tune()).
        "search_space": {
            "batch_size": [8, 16, 32, 64],
            "learning_rate": (1e-4, 1e-3),      # log-uniform
            "weight_decay": (1e-5, 1e-1),       # log-uniform
            "warmup_steps": [0, 50, 200],
            "decoder_layers": [2, 3, 4],
            "decoder_heads": [4, 8],
            "dropout": [0.0, 0.1, 0.2],
        },
    },

    "image": {
        "epochs": 3,
        "per_task_cap": 30_000,       # full-res images: keeps total under ~20GB
        "tune_trials": 12,
        "tune_budget_batches": 250,
        "defaults": {
            "batch_size": 16,
            "learning_rate": 3e-4,
            "weight_decay": 1e-2,
            "warmup_steps": 100,
            "decoder_layers": 3,
            "decoder_heads": 8,
            "dropout": 0.1,
        },
        "search_space": {
            "batch_size": [8, 16, 32],
            "learning_rate": (1e-4, 1e-3),
            "weight_decay": (1e-5, 1e-1),
            "warmup_steps": [0, 50, 200],
            "decoder_layers": [2, 3, 4],
            "decoder_heads": [4, 8],
            "dropout": [0.0, 0.1, 0.2],
        },
    },
}

STAGES = ["text_tune", "text_train", "image_tune", "image_train"]

# Universal Dependencies relation label set (fixed so head size is known up
# front, independent of which UD treebank slice we happen to load).
UD_DEPRELS = [
    "acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp",
    "clf", "compound", "conj", "cop", "csubj", "dep", "det", "discourse",
    "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark",
    "nmod", "nsubj", "nummod", "obj", "obl", "orphan", "parataxis", "punct",
    "reparandum", "root", "vocative", "xcomp",
]
DEPREL_TO_ID = {r: i for i, r in enumerate(UD_DEPRELS)}

# Universal POS tag set (fixed head size; the UD ``upos`` field is a ClassLabel
# so entries may arrive as ints -- used directly -- or strings -- mapped here).
UPOS_TAGS = [
    "NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", "DET",
    "CCONJ", "PROPN", "PRON", "X", "_", "ADV", "INTJ", "VERB", "AUX",
]
UPOS_TO_ID = {t: i for i, t in enumerate(UPOS_TAGS)}
NUM_UPOS = len(UPOS_TAGS)
NUM_NER = 9                      # CoNLL2003 IOB tags (O + B/I x PER/ORG/LOC/MISC)
NUM_COREF = 3                    # GAP: pronoun corefers with A, B, or neither

NUM_INSTANCE_CLASSES = 1000      # imagenet-1k
NUM_SEG_CLASSES = 151            # ADE20K (150 classes + background/unlabeled=0)
NUM_GQA_ANSWERS = 500            # top-K relation/answer vocab, built from data

# Dataset identifiers.
CONLL_HF = "eriktks/conll2003"
UD_HF = "universal-dependencies/universal_dependencies"
UD_CONFIG = "en_ewt"
IMAGENET_HF = "ILSVRC/imagenet-1k"
NYU_DEPTH_HF = "sayakpaul/nyu_depth_v2"
GQA_HF = "Voxel51/GQA-Scene-Graph"
ADE20K_HF = "merve/scene_parse_150"

# GAP coreference lives on GitHub (not HuggingFace); we fetch the raw TSVs.
GAP_BASE = ("https://raw.githubusercontent.com/google-research-datasets/"
            "gap-coreference/master")
GAP_FILES = ["gap-development.tsv", "gap-validation.tsv", "gap-test.tsv"]


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
            project="encoder_pretraining", name=stage, reinit=True,
            config=HYPERPARAMS,
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
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def mark_stage_done(stage):
    progress = load_progress()
    if stage not in progress["completed"]:
        progress["completed"].append(stage)
    save_progress(progress)


# --------------------------------------------------------------------------- #
# Greppable weight naming + lookup
# --------------------------------------------------------------------------- #
def weight_path(component, tag, epoch, batch, val_loss=None):
    name = f"{component}__{tag}__epoch{epoch:04d}_batch{batch:07d}"
    if val_loss is not None:
        name += f"_valloss{val_loss:.4f}"
    return os.path.join(WEIGHTS_DIR, name + ".pt")


_EPOCH_BATCH_RE = re.compile(r"epoch(\d+)_batch(\d+)")
_VALLOSS_RE = re.compile(r"valloss([0-9.]+)")


def _epoch_batch(path):
    m = _EPOCH_BATCH_RE.search(os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def _valloss(path):
    m = _VALLOSS_RE.search(os.path.basename(path))
    return float(m.group(1)) if m else float("inf")


def find_weights(component, tag):
    """Return the path to the requested weights, or None. ``latest`` picks the
    highest (epoch, batch); ``best`` picks the lowest embedded valloss."""
    files = glob.glob(os.path.join(WEIGHTS_DIR, f"{component}__{tag}__*.pt"))
    if not files:
        return None
    key = _valloss if tag == "best" else _epoch_batch
    return min(files, key=key) if tag == "best" else max(files, key=key)


def _cleanup(component, tag, keep_path):
    """Keep only ``keep_path`` for this (component, tag); delete older ones."""
    for f in glob.glob(os.path.join(WEIGHTS_DIR, f"{component}__{tag}__*.pt")):
        if os.path.abspath(f) != os.path.abspath(keep_path):
            try:
                os.remove(f)
            except OSError:
                pass


def save_checkpoint(prefix, wrapper, optimizer, epoch, batch, global_step,
                    best_val_loss, val_loss=None, is_best=False):
    """Persist both the reusable encoder weights and a full resumable wrapper
    checkpoint, under the ``latest`` tag (always) and ``best`` tag (when
    ``is_best``). ``prefix`` is 'text' or 'image'."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    enc, wrp = f"{prefix}_encoder", f"{prefix}_wrapper"

    resume_blob = {
        "wrapper_state": wrapper.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "batch": batch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "config": wrapper.config,
        "rng": torch.get_rng_state(),
    }

    def _write(tag, vloss):
        ep = weight_path(enc, tag, epoch, batch, vloss)
        wp = weight_path(wrp, tag, epoch, batch, vloss)
        torch.save(wrapper.encoder.state_dict(), ep)
        torch.save(resume_blob, wp)
        _cleanup(enc, tag, ep)
        _cleanup(wrp, tag, wp)

    _write("latest", val_loss)
    if is_best:
        _write("best", val_loss)


# --------------------------------------------------------------------------- #
# Tokenisation (shared SentencePiece model)
# --------------------------------------------------------------------------- #
class TextCodec:
    def __init__(self, model_path=TOKENIZER_PATH):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def ids(self, text):
        return self.sp.encode_as_ids(text)

    def pieces_per_word(self, word):
        return self.sp.encode_as_ids(word)

    def single(self, text, max_len):
        return [BOS_ID] + self.ids(text)[: max_len - 2] + [EOS_ID]

    def pair(self, a, b, max_len):
        a_ids = self.ids(a)[: (max_len - 3) // 2]
        b_ids = self.ids(b)[: (max_len - 3) - len(a_ids)]
        return [BOS_ID] + a_ids + [EOS_ID] + b_ids + [EOS_ID]

    def pad_batch(self, seqs, device):
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), PAD_ID, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask = ids == PAD_ID
        return ids.to(device), mask.to(device)


# =========================================================================== #
# Encoder wrappers:  encoder -> (shared) decoder transformer -> task heads
# =========================================================================== #
class TextPretrainWrapper(nn.Module):
    """Text encoder + a discardable decoder stack + one head per active task."""

    def __init__(self, config, active_tasks):
        super().__init__()
        from models.Encoders.TextEncoder import createTextEnc

        self.config = config
        d = config["d_model"]
        self.encoder = createTextEnc(dim=d)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config["decoder_heads"],
            dim_feedforward=d * 4, dropout=config["dropout"],
            activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, config["decoder_layers"])

        self.heads = nn.ModuleDict()
        if "ner" in active_tasks:                          # token-level
            self.heads["ner"] = nn.Linear(d, NUM_NER)
        if "pos" in active_tasks:                          # token-level
            self.heads["pos"] = nn.Linear(d, NUM_UPOS)
        if "ud_deprel" in active_tasks:                    # token-level
            self.heads["ud_deprel"] = nn.Linear(d, len(UD_DEPRELS))
        if "gap_coref" in active_tasks:                    # span-level
            self.heads["gap_coref"] = nn.Linear(3 * d, NUM_COREF)
        if "squad" in active_tasks:                        # span-level
            self.heads["squad"] = nn.Linear(d, 2)          # start / end logits
        if "mnli" in active_tasks:                         # sentence-level
            self.heads["mnli"] = nn.Linear(d, 3)
        if "swag" in active_tasks:                         # sentence-level
            self.heads["swag"] = nn.Linear(d, 1)           # per-candidate score
        if "mlm" in active_tasks:                          # sentence-level
            self.heads["mlm"] = nn.Linear(d, VOCAB_SIZE)

    def features(self, input_ids, pad_mask):
        enc = self.encoder.forward_tokens(input_ids, padding_mask=pad_mask)
        return self.decoder(enc, src_key_padding_mask=pad_mask)

    @staticmethod
    def masked_mean(feats, pad_mask):
        valid = (~pad_mask).unsqueeze(-1).float()
        return (feats * valid).sum(1) / valid.sum(1).clamp(min=1.0)


class ImagePretrainWrapper(nn.Module):
    """Vision encoder + discardable decoder + one head per active task. Tasks
    marked as needing text use the (frozen) text encoder passed in."""

    def __init__(self, config, active_tasks, frozen_text=None, text_codec=None,
                 seg_class_embed=None, patch_pixels=768):
        super().__init__()
        from models.Encoders.VisionEncoder import createVisionEncoder

        self.config = config
        d = config["d_model"]
        self.encoder = createVisionEncoder()
        self.frozen_text = frozen_text          # nn.Module (eval, no grad) or None
        self.text_codec = text_codec

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config["decoder_heads"],
            dim_feedforward=d * 4, dropout=config["dropout"],
            activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, config["decoder_layers"])

        self.heads = nn.ModuleDict()
        if "recon" in active_tasks:
            self.heads["recon"] = nn.Linear(d, patch_pixels)
        if "depth" in active_tasks:
            self.heads["depth"] = nn.Linear(d, 1)
        if "instance" in active_tasks:
            self.heads["instance"] = nn.Linear(d, NUM_INSTANCE_CLASSES)
        if "vrd" in active_tasks:
            self.heads["vrd"] = nn.Sequential(
                nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, NUM_GQA_ANSWERS)
            )
        if "semseg" in active_tasks:
            # Per-patch features are compared against frozen text class-name
            # embeddings (CLIP-style), so no learnable classifier weights.
            self.register_buffer(
                "seg_class_embed",
                seg_class_embed if seg_class_embed is not None
                else torch.zeros(NUM_SEG_CLASSES, d),
            )

    def features(self, images):
        enc = self.encoder(images)              # [B, N, D]
        return self.decoder(enc)

    @torch.no_grad()
    def text_embed(self, texts):
        """Mean-pooled frozen text-encoder embedding for a list of strings."""
        self.frozen_text.eval()   # stay in eval even after wrapper.train()
        out = []
        for t in texts:
            feats = self.frozen_text(t)         # [1, L, D]
            out.append(feats.mean(dim=1).squeeze(0))
        return torch.stack(out, dim=0)


# =========================================================================== #
# Task definitions.  Each task: prepares (and caches) data, collates a batch,
# and computes (loss, accuracy).  One head receives loss at a time.
# =========================================================================== #
class Task:
    name = "base"
    encoder = "text"        # 'text' | 'image'
    needs_text = False      # image tasks that require the frozen text encoder

    def download(self, cap):
        """Return a datasets.Dataset (single split) capped to ``cap`` rows."""
        raise NotImplementedError

    def build(self, cap, val_holdout, n_tasks):
        """Load (cache) data and split off a validation holdout."""
        from datasets import load_from_disk

        path = os.path.join(DATA_DIR, self.name)
        if os.path.isdir(path) and os.listdir(path):
            ds = load_from_disk(path)
        else:
            ds = self.download(cap)
            os.makedirs(DATA_DIR, exist_ok=True)
            ds.save_to_disk(path)
        val_n = max(1, val_holdout // n_tasks)
        val_n = min(val_n, len(ds) // 5)
        self.val_ds = ds.select(range(val_n))
        self.train_ds = ds.select(range(val_n, len(ds)))
        print(f"[data] {self.name}: {len(self.train_ds)} train / "
              f"{len(self.val_ds)} val")

    def collate(self, rows, ctx):
        raise NotImplementedError

    def loss_acc(self, wrapper, batch):
        raise NotImplementedError


def tokenize_with_offsets(codec, text, max_len):
    """Tokenize ``text`` word-by-word, returning (ids, word_char_starts,
    word_first_tok) so callers can map a character offset to the index of the
    subword token where the containing word begins."""
    ids, word_starts, word_first_tok, pos = [BOS_ID], [], [], 0
    for w in text.split(" "):
        word_starts.append(pos)
        word_first_tok.append(len(ids))
        for s in (codec.pieces_per_word(w) or [UNK_ID]):
            ids.append(s)
        pos += len(w) + 1
    ids = ids[: max_len - 1] + [EOS_ID]
    return ids, word_starts, word_first_tok


def offset_to_token(char_off, word_starts, word_first_tok, max_pos):
    wi = 0
    for k, ws in enumerate(word_starts):
        if ws <= char_off:
            wi = k
        else:
            break
    return min(word_first_tok[wi] if word_first_tok else 0, max_pos)


# ---- Text task: Named entity recognition (CoNLL2003) ----------------------- #
class NERTask(Task):
    name, encoder = "ner", "text"

    def download(self, cap):
        from datasets import load_dataset

        ds = load_dataset(CONLL_HF, split="train", cache_dir=HF_CACHE,
                          trust_remote_code=True)
        ds = ds.select(range(min(cap, len(ds))))
        return ds.remove_columns(
            [c for c in ds.column_names if c not in ("tokens", "ner_tags")]
        )

    def collate(self, rows, ctx):
        codec, device, cfg = ctx["codec"], ctx["device"], ctx["cfg"]
        max_len = cfg["max_text_len"]
        seqs, labels = [], []
        for r in rows:
            ids, labs = [BOS_ID], [-100]
            for word, tag in zip(r["tokens"], r["ner_tags"]):
                sub = codec.pieces_per_word(word) or [UNK_ID]
                lid = int(tag) if int(tag) < NUM_NER else 0
                for j, s in enumerate(sub):
                    ids.append(s)
                    labs.append(lid if j == 0 else -100)
            ids, labs = ids[: max_len - 1] + [EOS_ID], labs[: max_len - 1] + [-100]
            seqs.append(ids)
            labels.append(labs)
        ids, mask = codec.pad_batch(seqs, device)
        width = ids.size(1)
        lab = torch.full((len(labels), width), -100, dtype=torch.long)
        for i, l in enumerate(labels):
            lab[i, : len(l)] = torch.tensor(l, dtype=torch.long)
        return {"ids": ids, "mask": mask, "labels": lab.to(device)}

    def loss_acc(self, wrapper, batch):
        return _token_loss_acc(wrapper, "ner", batch)


# ---- Text task: Part-of-speech tagging (Universal Dependencies) ------------ #
class POSTask(Task):
    name, encoder = "pos", "text"

    def download(self, cap):
        from datasets import load_dataset

        ds = load_dataset(UD_HF, UD_CONFIG, split="train", cache_dir=HF_CACHE,
                          trust_remote_code=True)
        ds = ds.select(range(min(cap, len(ds))))
        return ds.remove_columns(
            [c for c in ds.column_names if c not in ("tokens", "upos")]
        )

    def collate(self, rows, ctx):
        codec, device, cfg = ctx["codec"], ctx["device"], ctx["cfg"]
        max_len = cfg["max_text_len"]
        seqs, labels = [], []
        for r in rows:
            ids, labs = [BOS_ID], [-100]
            for word, tag in zip(r["tokens"], r["upos"]):
                sub = codec.pieces_per_word(word) or [UNK_ID]
                lid = tag if isinstance(tag, int) else UPOS_TO_ID.get(tag, 12)
                lid = lid if lid < NUM_UPOS else 12          # 12 == "X"
                for j, s in enumerate(sub):
                    ids.append(s)
                    labs.append(lid if j == 0 else -100)
            ids, labs = ids[: max_len - 1] + [EOS_ID], labs[: max_len - 1] + [-100]
            seqs.append(ids)
            labels.append(labs)
        ids, mask = codec.pad_batch(seqs, device)
        width = ids.size(1)
        lab = torch.full((len(labels), width), -100, dtype=torch.long)
        for i, l in enumerate(labels):
            lab[i, : len(l)] = torch.tensor(l, dtype=torch.long)
        return {"ids": ids, "mask": mask, "labels": lab.to(device)}

    def loss_acc(self, wrapper, batch):
        return _token_loss_acc(wrapper, "pos", batch)


def _token_loss_acc(wrapper, head_name, batch):
    """Shared token-classification loss/accuracy (NER, POS, dependency rel)."""
    feats = wrapper.features(batch["ids"], batch["mask"])
    logits = wrapper.heads[head_name](feats)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1),
        ignore_index=-100,
    )
    valid = batch["labels"] != -100
    acc = ((logits.argmax(-1) == batch["labels"]) & valid).sum() / valid.sum().clamp(min=1)
    return loss, acc.item(), batch["ids"].size(0)


# ---- Text task: Coreference resolution (GAP, from GitHub TSVs) ------------- #
class GAPCorefTask(Task):
    name, encoder = "gap_coref", "text"

    def download(self, cap):
        import csv
        import urllib.request
        from datasets import Dataset

        os.makedirs(DATA_DIR, exist_ok=True)
        rows = []
        for fname in GAP_FILES:
            local = os.path.join(DATA_DIR, fname)
            if not os.path.isfile(local):
                url = f"{GAP_BASE}/{fname}"
                print(f"[data] downloading GAP file {url}")
                urllib.request.urlretrieve(url, local)   # honours http(s)_proxy env
            with open(local, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    a_coref = r["A-coref"].strip().upper() == "TRUE"
                    b_coref = r["B-coref"].strip().upper() == "TRUE"
                    label = 0 if a_coref else (1 if b_coref else 2)
                    rows.append({
                        "text": r["Text"],
                        "pronoun_offset": int(r["Pronoun-offset"]),
                        "a_offset": int(r["A-offset"]),
                        "b_offset": int(r["B-offset"]),
                        "label": label,
                    })
                    if len(rows) >= cap:
                        break
            if len(rows) >= cap:
                break
        if not rows:
            raise RuntimeError("GAP download produced no rows")
        return Dataset.from_list(rows)

    def collate(self, rows, ctx):
        codec, device, cfg = ctx["codec"], ctx["device"], ctx["cfg"]
        max_len = cfg["max_text_len"]
        seqs, p_pos, a_pos, b_pos, labels = [], [], [], [], []
        for r in rows:
            ids, ws, wf = tokenize_with_offsets(codec, r["text"], max_len)
            m = len(ids) - 1
            seqs.append(ids)
            p_pos.append(offset_to_token(r["pronoun_offset"], ws, wf, m))
            a_pos.append(offset_to_token(r["a_offset"], ws, wf, m))
            b_pos.append(offset_to_token(r["b_offset"], ws, wf, m))
            labels.append(r["label"])
        ids, mask = codec.pad_batch(seqs, device)
        return {
            "ids": ids, "mask": mask,
            "pos": torch.tensor([p_pos, a_pos, b_pos], device=device).t(),  # [B,3]
            "labels": torch.tensor(labels, device=device),
        }

    def loss_acc(self, wrapper, batch):
        feats = wrapper.features(batch["ids"], batch["mask"])     # [B, L, D]
        d = feats.size(-1)
        idx = batch["pos"].unsqueeze(-1).expand(-1, -1, d)        # [B, 3, D]
        gathered = feats.gather(1, idx).reshape(feats.size(0), 3 * d)
        logits = wrapper.heads["gap_coref"](gathered)
        loss = F.cross_entropy(logits, batch["labels"])
        acc = (logits.argmax(-1) == batch["labels"]).float().mean()
        return loss, acc.item(), feats.size(0)


# ---- Text task: Dependency relation prediction (Universal Dependencies) ---- #
class UDDepRelTask(Task):
    name, encoder = "ud_deprel", "text"

    def download(self, cap):
        from datasets import load_dataset

        ds = load_dataset(UD_HF, UD_CONFIG, split="train",
                          cache_dir=HF_CACHE, trust_remote_code=True)
        ds = ds.select(range(min(cap, len(ds))))
        return ds.remove_columns(
            [c for c in ds.column_names if c not in ("tokens", "deprel")]
        )

    def collate(self, rows, ctx):
        codec, device, cfg = ctx["codec"], ctx["device"], ctx["cfg"]
        max_len = cfg["max_text_len"]
        seqs, labels = [], []
        for r in rows:
            ids, labs = [BOS_ID], [-100]
            for word, rel in zip(r["tokens"], r["deprel"]):
                sub = codec.pieces_per_word(word) or [UNK_ID]
                base = rel.split(":")[0]
                lid = DEPREL_TO_ID.get(base, DEPREL_TO_ID["dep"])
                for j, s in enumerate(sub):
                    ids.append(s)
                    labs.append(lid if j == 0 else -100)
            ids, labs = ids[: max_len - 1] + [EOS_ID], labs[: max_len - 1] + [-100]
            seqs.append(ids)
            labels.append(labs)
        ids, mask = codec.pad_batch(seqs, device)
        width = ids.size(1)
        lab = torch.full((len(labels), width), -100, dtype=torch.long)
        for i, l in enumerate(labels):
            lab[i, : len(l)] = torch.tensor(l, dtype=torch.long)
        return {"ids": ids, "mask": mask, "labels": lab.to(device)}

    def loss_acc(self, wrapper, batch):
        feats = wrapper.features(batch["ids"], batch["mask"])
        logits = wrapper.heads["ud_deprel"](feats)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1),
            ignore_index=-100,
        )
        valid = batch["labels"] != -100
        acc = ((logits.argmax(-1) == batch["labels"]) & valid).sum() / valid.sum().clamp(min=1)
        return loss, acc.item(), batch["ids"].size(0)


# ---- Text task: Extractive QA (SQuAD 2.0) ---------------------------------- #
class SquadTask(Task):
    name, encoder = "squad", "text"

    def download(self, cap):
        from datasets import load_dataset

        ds = load_dataset("squad_v2", split="train", cache_dir=HF_CACHE)
        ds = ds.select(range(min(cap, len(ds))))
        return ds.remove_columns(
            [c for c in ds.column_names
             if c not in ("question", "context", "answers")]
        )

    def collate(self, rows, ctx):
        codec, device, cfg = ctx["codec"], ctx["device"], ctx["cfg"]
        max_len = cfg["max_text_len"]
        seqs, starts, ends = [], [], []
        for r in rows:
            q_ids = codec.ids(r["question"])[: max_len // 3]
            ids = [BOS_ID] + q_ids + [EOS_ID]
            # Build context subwords while tracking each word's first-token index.
            words, char_pos, pos = [], [], 0
            for w in r["context"].split(" "):
                words.append(w)
                char_pos.append(pos)
                pos += len(w) + 1
            first_tok = {}
            for wi, w in enumerate(words):
                first_tok[wi] = len(ids)
                for s in (codec.pieces_per_word(w) or [UNK_ID]):
                    ids.append(s)
            ids = ids[: max_len - 1] + [EOS_ID]

            s_pos = e_pos = 0                       # default: no-answer -> CLS(0)
            ans = r["answers"]["text"]
            if ans:
                a_start = r["answers"]["answer_start"][0]
                a_text = ans[0]
                sw = max((wi for wi, cp in enumerate(char_pos) if cp <= a_start),
                         default=None)
                a_end_char = a_start + len(a_text)
                ew = max((wi for wi, cp in enumerate(char_pos) if cp < a_end_char),
                         default=sw)
                if sw is not None and first_tok.get(sw, 0) < len(ids) - 1:
                    s_pos = min(first_tok[sw], len(ids) - 1)
                    e_pos = min(first_tok.get(ew, s_pos), len(ids) - 1)
            seqs.append(ids)
            starts.append(s_pos)
            ends.append(e_pos)
        ids, mask = codec.pad_batch(seqs, device)
        return {
            "ids": ids, "mask": mask,
            "start": torch.tensor(starts, device=device),
            "end": torch.tensor(ends, device=device),
        }

    def loss_acc(self, wrapper, batch):
        feats = wrapper.features(batch["ids"], batch["mask"])
        logits = wrapper.heads["squad"](feats)        # [B, L, 2]
        start_logits, end_logits = logits[..., 0], logits[..., 1]
        loss = 0.5 * (
            F.cross_entropy(start_logits, batch["start"])
            + F.cross_entropy(end_logits, batch["end"])
        )
        acc = 0.5 * (
            (start_logits.argmax(-1) == batch["start"]).float().mean()
            + (end_logits.argmax(-1) == batch["end"]).float().mean()
        )
        return loss, acc.item(), batch["ids"].size(0)


# ---- Text task: Natural language inference (MultiNLI) ---------------------- #
class MNLITask(Task):
    name, encoder = "mnli", "text"

    def download(self, cap):
        from datasets import load_dataset

        ds = load_dataset("multi_nli", split="train", cache_dir=HF_CACHE)
        ds = ds.filter(lambda r: r["label"] in (0, 1, 2))
        ds = ds.select(range(min(cap, len(ds))))
        return ds.remove_columns(
            [c for c in ds.column_names
             if c not in ("premise", "hypothesis", "label")]
        )

    def collate(self, rows, ctx):
        codec, device, cfg = ctx["codec"], ctx["device"], ctx["cfg"]
        seqs = [codec.pair(r["premise"], r["hypothesis"], cfg["max_text_len"])
                for r in rows]
        ids, mask = codec.pad_batch(seqs, device)
        labels = torch.tensor([r["label"] for r in rows], device=device)
        return {"ids": ids, "mask": mask, "labels": labels}

    def loss_acc(self, wrapper, batch):
        feats = wrapper.features(batch["ids"], batch["mask"])
        pooled = wrapper.masked_mean(feats, batch["mask"])
        logits = wrapper.heads["mnli"](pooled)
        loss = F.cross_entropy(logits, batch["labels"])
        acc = (logits.argmax(-1) == batch["labels"]).float().mean()
        return loss, acc.item(), batch["ids"].size(0)


# ---- Text task: Discourse coherence / grounded commonsense (SWAG) ---------- #
class SwagTask(Task):
    name, encoder = "swag", "text"

    def download(self, cap):
        from datasets import load_dataset

        ds = load_dataset("swag", "regular", split="train", cache_dir=HF_CACHE)
        ds = ds.select(range(min(cap, len(ds))))
        keep = ("sent1", "sent2", "ending0", "ending1", "ending2", "ending3",
                "label")
        return ds.remove_columns([c for c in ds.column_names if c not in keep])

    def collate(self, rows, ctx):
        codec, device, cfg = ctx["codec"], ctx["device"], ctx["cfg"]
        # 4 candidate sequences per example, flattened into the batch.
        seqs, labels = [], []
        for r in rows:
            for k in range(4):
                ending = r[f"ending{k}"]
                seqs.append(codec.pair(f"{r['sent1']} {r['sent2']}", ending,
                                       cfg["max_text_len"]))
            labels.append(r["label"])
        ids, mask = codec.pad_batch(seqs, device)
        return {"ids": ids, "mask": mask,
                "labels": torch.tensor(labels, device=device), "n": len(rows)}

    def loss_acc(self, wrapper, batch):
        feats = wrapper.features(batch["ids"], batch["mask"])
        pooled = wrapper.masked_mean(feats, batch["mask"])
        scores = wrapper.heads["swag"](pooled).view(batch["n"], 4)
        loss = F.cross_entropy(scores, batch["labels"])
        acc = (scores.argmax(-1) == batch["labels"]).float().mean()
        return loss, acc.item(), batch["n"]


# ---- Text task: Masked language modeling (Wikipedia) ----------------------- #
class MLMTask(Task):
    name, encoder = "mlm", "text"

    def download(self, cap):
        from datasets import load_dataset

        stream = load_dataset("wikimedia/wikipedia", "20231101.en",
                              split="train", streaming=True,
                              cache_dir=HF_CACHE)
        chunks, words_per = [], 48
        for art in stream:
            words = art["text"].split()
            for i in range(0, len(words), words_per):
                chunk = " ".join(words[i:i + words_per])
                if len(chunk) > 40:
                    chunks.append(chunk)
                if len(chunks) >= cap:
                    break
            if len(chunks) >= cap:
                break
        from datasets import Dataset
        return Dataset.from_dict({"text": chunks})

    def collate(self, rows, ctx):
        codec, device, cfg = ctx["codec"], ctx["device"], ctx["cfg"]
        seqs, labels = [], []
        for r in rows:
            ids = codec.single(r["text"], cfg["max_text_len"])
            lab = [-100] * len(ids)
            for i in range(1, len(ids) - 1):            # never mask BOS/EOS
                if torch.rand(1).item() < 0.15:
                    lab[i] = ids[i]
                    ids[i] = MASK_ID
            seqs.append(ids)
            labels.append(lab)
        ids, mask = codec.pad_batch(seqs, device)
        width = ids.size(1)
        lab = torch.full((len(labels), width), -100, dtype=torch.long)
        for i, l in enumerate(labels):
            lab[i, : len(l)] = torch.tensor(l, dtype=torch.long)
        return {"ids": ids, "mask": mask, "labels": lab.to(device)}

    def loss_acc(self, wrapper, batch):
        feats = wrapper.features(batch["ids"], batch["mask"])
        logits = wrapper.heads["mlm"](feats)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1),
            ignore_index=-100,
        )
        valid = batch["labels"] != -100
        acc = ((logits.argmax(-1) == batch["labels"]) & valid).sum() / valid.sum().clamp(min=1)
        return loss, acc.item(), batch["ids"].size(0)


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def _load_image_tensor(img, size):
    """PIL image (any mode) -> float tensor [3, size, size] in [0, 1]."""
    import numpy as np

    img = img.convert("RGB").resize((size, size))
    arr = torch.from_numpy(np.asarray(img, dtype="float32") / 255.0)
    return arr.permute(2, 0, 1)


# ---- Image task: masked image reconstruction (imagenet-1k) ----------------- #
class ReconTask(Task):
    name, encoder = "recon", "image"

    def download(self, cap):
        from datasets import load_dataset, Dataset

        stream = load_dataset(IMAGENET_HF, split="train", streaming=True,
                              cache_dir=HF_CACHE)
        rows = []
        for ex in stream:
            rows.append({"image": ex["image"]})
            if len(rows) >= cap:
                break
        return Dataset.from_list(rows)

    def collate(self, rows, ctx):
        size, ps = ctx["cfg"]["image_size"], 16
        device = ctx["device"]
        imgs = torch.stack([_load_image_tensor(r["image"], size) for r in rows])
        # Zero out a random 25% of patches as the reconstruction mask.
        g = size // ps
        masked = imgs.clone()
        patch_mask = (torch.rand(imgs.size(0), g, g) < 0.25)
        for b in range(imgs.size(0)):
            for i in range(g):
                for j in range(g):
                    if patch_mask[b, i, j]:
                        masked[b, :, i * ps:(i + 1) * ps, j * ps:(j + 1) * ps] = 0.0
        return {"masked": masked.to(device), "target": imgs.to(device),
                "patch_mask": patch_mask.to(device), "g": g, "ps": ps}

    def loss_acc(self, wrapper, batch):
        feats = wrapper.features(batch["masked"])          # [B, g*g, D]
        pred = wrapper.heads["recon"](feats)               # [B, g*g, ps*ps*3]
        B, g, ps = batch["masked"].size(0), batch["g"], batch["ps"]
        target = batch["target"].unfold(2, ps, ps).unfold(3, ps, ps)  # B,3,g,g,ps,ps
        target = target.permute(0, 2, 3, 1, 4, 5).reshape(B, g * g, -1)
        m = batch["patch_mask"].reshape(B, g * g, 1).float()
        loss = ((pred - target) ** 2 * m).sum() / m.sum().clamp(min=1) / target.size(-1)
        acc = torch.exp(-loss)                             # bounded pseudo-accuracy
        return loss, acc.item(), B


# ---- Image task: depth estimation (NYU Depth V2) --------------------------- #
class DepthTask(Task):
    name, encoder = "depth", "image"

    def download(self, cap):
        from datasets import load_dataset, Dataset

        ds = load_dataset(NYU_DEPTH_HF, split="train",
                          streaming=True, cache_dir=HF_CACHE)
        rows = []
        for ex in ds:
            rows.append({"image": ex["image"], "depth_map": ex["depth_map"]})
            if len(rows) >= cap:
                break
        return Dataset.from_list(rows)

    def collate(self, rows, ctx):
        size, ps = ctx["cfg"]["image_size"], 16
        device, g = ctx["device"], ctx["cfg"]["image_size"] // 16
        imgs = torch.stack([_load_image_tensor(r["image"], size) for r in rows])
        depths = []
        for r in rows:
            d = _load_image_tensor(r["depth_map"], size)[0:1]     # single channel
            d = F.adaptive_avg_pool2d(d.unsqueeze(0), (g, g)).reshape(-1)
            depths.append(d)
        return {"images": imgs.to(device),
                "depth": torch.stack(depths).to(device)}

    def loss_acc(self, wrapper, batch):
        feats = wrapper.features(batch["images"])
        pred = wrapper.heads["depth"](feats).squeeze(-1)          # [B, g*g]
        loss = F.mse_loss(pred, batch["depth"])
        return loss, torch.exp(-loss).item(), batch["images"].size(0)


# ---- Image task: instance discrimination (imagenet-1k, label proxy) -------- #
class InstanceTask(Task):
    name, encoder = "instance", "image"

    def download(self, cap):
        from datasets import load_dataset, Dataset

        stream = load_dataset(IMAGENET_HF, split="train", streaming=True,
                              cache_dir=HF_CACHE)
        rows = []
        for ex in stream:
            rows.append({"image": ex["image"], "label": ex["label"]})
            if len(rows) >= cap:
                break
        return Dataset.from_list(rows)

    def collate(self, rows, ctx):
        size, device = ctx["cfg"]["image_size"], ctx["device"]
        imgs = torch.stack([_load_image_tensor(r["image"], size) for r in rows])
        labels = torch.tensor([r["label"] for r in rows], device=device)
        return {"images": imgs.to(device), "labels": labels}

    def loss_acc(self, wrapper, batch):
        feats = wrapper.features(batch["images"]).mean(dim=1)     # global pool
        logits = wrapper.heads["instance"](feats)
        loss = F.cross_entropy(logits, batch["labels"])
        acc = (logits.argmax(-1) == batch["labels"]).float().mean()
        return loss, acc.item(), batch["images"].size(0)


# ---- Image task: visual relationship detection (GQA; needs frozen text) ---- #
class VRDTask(Task):
    name, encoder, needs_text = "vrd", "image", True
    answer_vocab = None

    @staticmethod
    def _extract_relationship(ex):
        """Pull (subject, predicate, object) from a scene-graph row. Schemas
        vary; try the common shapes and return None if none match."""
        rels = ex.get("relationships") or ex.get("relations")
        objs = ex.get("objects")
        if not rels:
            return None
        rel = rels[0] if isinstance(rels, list) else rels
        if isinstance(rel, dict):
            pred = rel.get("predicate") or rel.get("name") or rel.get("label")
            subj = rel.get("subject") or rel.get("subject_name") or ""
            obj = rel.get("object") or rel.get("object_name") or ""
            # Some schemas store subject/object as indices into ``objects``.
            if isinstance(subj, int) and objs:
                subj = (objs[subj].get("name", "") if subj < len(objs) else "")
            if isinstance(obj, int) and objs:
                obj = (objs[obj].get("name", "") if obj < len(objs) else "")
            if pred:
                return str(subj), str(pred), str(obj)
        return None

    def download(self, cap):
        from datasets import load_dataset, Dataset

        ds = load_dataset(GQA_HF, split="train", streaming=True,
                          cache_dir=HF_CACHE, trust_remote_code=True)
        rows = []
        for ex in ds:
            img = ex.get("image")
            triple = self._extract_relationship(ex)
            if img is None or triple is None:
                continue
            subj, pred, obj = triple
            rows.append({"image": img, "subject": subj, "predicate": pred,
                         "object": obj})
            if len(rows) >= cap:
                break
        if not rows:
            raise RuntimeError("GQA-Scene-Graph yielded no usable relationships")
        # Build a top-K predicate vocabulary from the sampled data.
        from collections import Counter
        counts = Counter(r["predicate"] for r in rows)
        vocab = [p for p, _ in counts.most_common(NUM_GQA_ANSWERS)]
        self.answer_vocab = {p: i for i, p in enumerate(vocab)}
        with open(os.path.join(DATA_DIR, "gqa_predicates.json"), "w") as f:
            json.dump(vocab, f)
        for r in rows:
            r["pred_id"] = self.answer_vocab.get(r["predicate"], 0)
        return Dataset.from_list(rows)

    def collate(self, rows, ctx):
        size, device = ctx["cfg"]["image_size"], ctx["device"]
        imgs = torch.stack([_load_image_tensor(r["image"], size) for r in rows])
        # The subject/object phrase is what the frozen text encoder grounds on.
        phrases = [f"{r['subject']} and {r['object']}" for r in rows]
        return {"images": imgs.to(device), "phrases": phrases,
                "labels": torch.tensor([r["pred_id"] for r in rows], device=device)}

    def loss_acc(self, wrapper, batch):
        img_feats = wrapper.features(batch["images"]).mean(dim=1)     # [B, D]
        txt_feats = wrapper.text_embed(batch["phrases"]).to(img_feats.device)
        logits = wrapper.heads["vrd"](torch.cat([img_feats, txt_feats], dim=-1))
        loss = F.cross_entropy(logits, batch["labels"])
        acc = (logits.argmax(-1) == batch["labels"]).float().mean()
        return loss, acc.item(), batch["images"].size(0)


# ---- Image task: semantic segmentation (ADE20K; needs frozen text) --------- #
class SemSegTask(Task):
    name, encoder, needs_text = "semseg", "image", True

    def download(self, cap):
        from datasets import load_dataset, Dataset

        ds = load_dataset(ADE20K_HF, split="train", streaming=True,
                          cache_dir=HF_CACHE, trust_remote_code=True)
        rows = []
        for ex in ds:
            rows.append({"image": ex["image"], "annotation": ex["annotation"]})
            if len(rows) >= cap:
                break
        return Dataset.from_list(rows)

    def collate(self, rows, ctx):
        size, device = ctx["cfg"]["image_size"], ctx["device"]
        g = size // 16
        imgs = torch.stack([_load_image_tensor(r["image"], size) for r in rows])
        segs = []
        for r in rows:
            import numpy as np
            a = r["annotation"].convert("L").resize((g, g))
            segs.append(torch.from_numpy(np.asarray(a, dtype="int64")).reshape(-1))
        return {"images": imgs.to(device), "seg": torch.stack(segs).to(device)}

    def loss_acc(self, wrapper, batch):
        feats = wrapper.features(batch["images"])                 # [B, g*g, D]
        cls = wrapper.seg_class_embed.to(feats.device)            # [C, D]
        logits = feats @ cls.t()                                  # [B, g*g, C]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), batch["seg"].reshape(-1),
            ignore_index=0,
        )
        valid = batch["seg"] != 0
        acc = ((logits.argmax(-1) == batch["seg"]) & valid).sum() / valid.sum().clamp(min=1)
        return loss, acc.item(), batch["images"].size(0)


TEXT_TASKS = [NERTask, POSTask, UDDepRelTask, GAPCorefTask, SquadTask,
              MNLITask, SwagTask, MLMTask]
IMAGE_TASKS = [ReconTask, DepthTask, InstanceTask, VRDTask, SemSegTask]


# --------------------------------------------------------------------------- #
# Build + prepare the active task list for an encoder (skipping failures)
# --------------------------------------------------------------------------- #
def build_tasks(encoder_kind, cfg):
    classes = TEXT_TASKS if encoder_kind == "text" else IMAGE_TASKS
    cap = HYPERPARAMS[encoder_kind]["per_task_cap"]
    val_holdout = cfg["val_holdout"]
    tasks = []
    for cls in classes:
        task = cls()
        try:
            task.build(cap, val_holdout, len(classes))
            tasks.append(task)
        except Exception as e:
            print(f"[data] SKIP task '{task.name}' — could not prepare ({e})")
    if not tasks:
        raise RuntimeError(f"No {encoder_kind} tasks could be prepared.")
    return tasks


# --------------------------------------------------------------------------- #
# Single static dataloader over every task, one task-label per entry.
#
# All entries from all of an encoder's tasks are merged into one Dataset; each
# entry is tagged with the task it belongs to. A task-homogeneous batch sampler
# guarantees every batch contains entries from a single task, so exactly one
# head is activated per step (in both tuning and training).
# --------------------------------------------------------------------------- #
class UnifiedTaskDataset(torch.utils.data.Dataset):
    def __init__(self, tasks, split):
        self.tasks = tasks
        self.split = split
        self.index = []                      # [(task_idx, local_idx), ...]
        for ti, t in enumerate(tasks):
            ds = t.train_ds if split == "train" else t.val_ds
            self.index.extend((ti, li) for li in range(len(ds)))

    def _split_ds(self, ti):
        t = self.tasks[ti]
        return t.train_ds if self.split == "train" else t.val_ds

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ti, li = self.index[i]
        return ti, self._split_ds(ti)[li]    # (task label, raw entry)


class TaskHomogeneousBatchSampler(torch.utils.data.Sampler):
    """Batches whose members all share one task. Deterministic per ``seed`` so
    training can resume mid-epoch by skipping already-finished batches."""

    def __init__(self, dataset, batch_size, seed, shuffle=True):
        from collections import defaultdict

        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.by_task = defaultdict(list)
        for gi, (ti, _) in enumerate(dataset.index):
            self.by_task[ti].append(gi)
        self._len = sum((len(v) + batch_size - 1) // batch_size
                        for v in self.by_task.values())

    def __len__(self):
        return self._len

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        batches = []
        for gis in self.by_task.values():
            if self.shuffle:
                order = torch.randperm(len(gis), generator=g).tolist()
                gis = [gis[k] for k in order]
            for i in range(0, len(gis), self.batch_size):
                batches.append(gis[i:i + self.batch_size])
        if self.shuffle:                      # interleave tasks across the epoch
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[k] for k in perm]
        yield from batches


def make_collate(tasks, ctx):
    def _collate(samples):
        ti = samples[0][0]                    # sampler guarantees one task/batch
        task = tasks[ti]
        batch = task.collate([row for _, row in samples], ctx)
        batch["task_idx"] = ti
        batch["task_name"] = task.name        # drives which head is activated
        return batch
    return _collate


def make_loader(dataset, tasks, ctx, batch_size, seed, shuffle=True):
    sampler = TaskHomogeneousBatchSampler(dataset, batch_size, seed, shuffle)
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=make_collate(tasks, ctx),
        num_workers=0,
    )


# --------------------------------------------------------------------------- #
# Validation: average loss across the unified holdout set
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(wrapper, tasks, ctx, batch_size):
    wrapper.eval()
    task_map = {t.name: t for t in tasks}
    val_ds = UnifiedTaskDataset(tasks, "val")
    loader = make_loader(val_ds, tasks, ctx, batch_size, seed=0, shuffle=False)
    total_loss, total_n = 0.0, 0
    for batch in loader:
        t = task_map[batch["task_name"]]      # only this task's head fires
        loss, _, bn = t.loss_acc(wrapper, batch)
        total_loss += loss.item() * bn
        total_n += bn
    wrapper.train()
    return total_loss / max(1, total_n)


# --------------------------------------------------------------------------- #
# Frozen text encoder (for image tasks that require it)
# --------------------------------------------------------------------------- #
def load_frozen_text_encoder(cfg, device):
    from models.Encoders.TextEncoder import createTextEnc

    enc = createTextEnc(dim=cfg["d_model"]).to(device)
    best = find_weights("text_encoder", "best") or find_weights("text_encoder", "latest")
    if best:
        enc.load_state_dict(torch.load(best, map_location=device))
        print(f"[image] loaded frozen text encoder from {os.path.basename(best)}")
    else:
        print("[image] WARNING: no trained text encoder found; using random init")
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def build_seg_class_embed(frozen_text, device, d_model):
    """Embed ADE20K class names with the frozen text encoder (CLIP-style)."""
    names = ["background"] + [f"class {i}" for i in range(1, NUM_SEG_CLASSES)]
    with torch.no_grad():
        embs = [frozen_text(n).mean(dim=1).squeeze(0) for n in names]
    return torch.stack(embs, dim=0).to(device)


def make_image_wrapper(config, active_names, device):
    frozen_text = load_frozen_text_encoder(config, device)
    codec = TextCodec()
    seg_embed = None
    if "semseg" in active_names:
        seg_embed = build_seg_class_embed(frozen_text, device, config["d_model"])
    return ImagePretrainWrapper(
        config, active_names, frozen_text=frozen_text, text_codec=codec,
        seg_class_embed=seg_embed,
    ).to(device)


# --------------------------------------------------------------------------- #
# Optuna hyperparameter tuning (per encoder)
# --------------------------------------------------------------------------- #
def make_config(encoder_kind, hp):
    cfg = dict(HYPERPARAMS["global"])
    cfg.update(hp)
    return cfg


def build_wrapper(encoder_kind, config, active_names, device):
    if encoder_kind == "text":
        return TextPretrainWrapper(config, active_names).to(device)
    return make_image_wrapper(config, active_names, device)


def short_train_and_eval(encoder_kind, config, tasks, ctx, device, budget):
    active = [t.name for t in tasks]
    wrapper = build_wrapper(encoder_kind, config, active, device)
    optimizer = torch.optim.AdamW(
        (p for p in wrapper.parameters() if p.requires_grad),
        lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    wrapper.train()
    task_map = {t.name: t for t in tasks}
    train_ds = UnifiedTaskDataset(tasks, "train")
    loader = make_loader(train_ds, tasks, ctx, config["batch_size"], seed=0)
    for step, batch in enumerate(loader):
        if step >= budget:
            break
        t = task_map[batch["task_name"]]      # only this task's head fires
        loss, _, _ = t.loss_acc(wrapper, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    val = validate(wrapper, tasks, ctx, config["batch_size"])
    del wrapper, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return val


def tune(encoder_kind, device):
    import optuna

    print(f"\n===== TUNING: {encoder_kind} encoder =====")
    cfg0 = make_config(encoder_kind, HYPERPARAMS[encoder_kind]["defaults"])
    tasks = build_tasks(encoder_kind, cfg0)
    codec = TextCodec()
    ctx = {"codec": codec, "device": device, "cfg": cfg0}
    space = HYPERPARAMS[encoder_kind]["search_space"]
    budget = HYPERPARAMS[encoder_kind]["tune_budget_batches"]

    def objective(trial):
        hp = {
            "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            "weight_decay": trial.suggest_float("weight_decay", *space["weight_decay"], log=True),
            "warmup_steps": trial.suggest_categorical("warmup_steps", space["warmup_steps"]),
            "decoder_layers": trial.suggest_categorical("decoder_layers", space["decoder_layers"]),
            "decoder_heads": trial.suggest_categorical("decoder_heads", space["decoder_heads"]),
            "dropout": trial.suggest_categorical("dropout", space["dropout"]),
        }
        cfg = make_config(encoder_kind, hp)
        ctx["cfg"] = cfg
        val = short_train_and_eval(encoder_kind, cfg, tasks, ctx, device, budget)
        print(f"  trial {trial.number}: val_loss={val:.4f}  hp={hp}")
        wandb_log({f"tune/{encoder_kind}/val_loss": val, **{
            f"tune/{encoder_kind}/{k}": v for k, v in hp.items()}})
        return val

    os.makedirs(STUDY_DIR, exist_ok=True)
    study = optuna.create_study(
        study_name=f"{encoder_kind}_encoder",
        storage=f"sqlite:///{os.path.join(STUDY_DIR, encoder_kind + '.db')}",
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=0),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=HYPERPARAMS[encoder_kind]["tune_trials"])

    best_hp = study.best_params
    out = os.path.join(STUDY_DIR, f"{encoder_kind}_best_hparams.json")
    with open(out, "w") as f:
        json.dump(best_hp, f, indent=2)
    print(f"\nBest {encoder_kind} hyperparameters: {best_hp}")
    print(f"Saved to {out}")
    return best_hp


def load_best_hparams(encoder_kind):
    path = os.path.join(STUDY_DIR, f"{encoder_kind}_best_hparams.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    print(f"[train] no tuned hparams for {encoder_kind}; using defaults")
    return HYPERPARAMS[encoder_kind]["defaults"]


# --------------------------------------------------------------------------- #
# Full (resumable) training for one encoder
# --------------------------------------------------------------------------- #
def train_encoder(encoder_kind, device):
    print(f"\n===== TRAINING: {encoder_kind} encoder =====")
    hp = load_best_hparams(encoder_kind)
    config = make_config(encoder_kind, hp)
    tasks = build_tasks(encoder_kind, config)
    active = [t.name for t in tasks]
    codec = TextCodec()
    ctx = {"codec": codec, "device": device, "cfg": config}
    task_map = {t.name: t for t in tasks}

    wrapper = build_wrapper(encoder_kind, config, active, device)
    optimizer = torch.optim.AdamW(
        (p for p in wrapper.parameters() if p.requires_grad),
        lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )

    def lr_scale(step):
        w = config["warmup_steps"]
        return (step + 1) / w if w > 0 and step < w else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    # ---- Resume from the latest checkpoint if present --------------------- #
    start_epoch, start_batch, global_step = 0, 0, 0
    best_val = float("inf")
    resume = find_weights(f"{encoder_kind}_wrapper", "latest")
    if resume:
        blob = torch.load(resume, map_location=device)
        wrapper.load_state_dict(blob["wrapper_state"])
        optimizer.load_state_dict(blob["optimizer_state"])
        start_epoch, start_batch = blob["epoch"], blob["batch"]
        global_step, best_val = blob["global_step"], blob["best_val_loss"]
        print(f"[resume] {os.path.basename(resume)} -> epoch {start_epoch}, "
              f"batch {start_batch}")

    eval_every = config["eval_every_batches"]
    ckpt_every = config["checkpoint_every_batches"]
    epochs = HYPERPARAMS[encoder_kind]["epochs"]

    train_ds = UnifiedTaskDataset(tasks, "train")

    wrapper.train()
    for epoch in range(start_epoch, epochs):
        # One static dataloader per epoch, deterministically seeded so a resumed
        # run reproduces the exact batch order and can skip finished batches.
        loader = make_loader(train_ds, tasks, ctx, config["batch_size"],
                             seed=config["seed"] + epoch)
        n_batches = len(loader)
        skip_to = start_batch if epoch == start_epoch else 0
        for batch_idx, batch in enumerate(loader):
            if batch_idx < skip_to:          # fast-forward over finished batches
                continue
            tname = batch["task_name"]
            t = task_map[tname]              # activate only this task's head
            loss, acc, _ = t.loss_acc(wrapper, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % 50 == 0:
                print(f"  e{epoch} b{batch_idx} step{global_step} "
                      f"task={tname} loss={loss.item():.4f} acc={acc:.3f}")
                wandb_log({f"train/{encoder_kind}/loss": loss.item(),
                           f"train/{encoder_kind}/acc": acc,
                           f"train/{encoder_kind}/task": tname,
                           "epoch": epoch, "batch": batch_idx})

            if global_step % ckpt_every == 0:
                save_checkpoint(encoder_kind, wrapper, optimizer, epoch,
                                batch_idx, global_step, best_val)

            if global_step % eval_every == 0:
                val = validate(wrapper, tasks, ctx, config["batch_size"])
                is_best = val < best_val
                best_val = min(best_val, val)
                print(f"  [val] step {global_step}: val_loss={val:.4f} "
                      f"(best={best_val:.4f})")
                wandb_log({f"val/{encoder_kind}/loss": val})
                save_checkpoint(encoder_kind, wrapper, optimizer, epoch,
                                batch_idx, global_step, best_val,
                                val_loss=val, is_best=is_best)

        # ---- End-of-epoch validation + checkpoint ------------------------- #
        val = validate(wrapper, tasks, ctx, config["batch_size"])
        is_best = val < best_val
        best_val = min(best_val, val)
        print(f"[epoch {epoch}] val_loss={val:.4f} (best={best_val:.4f})")
        wandb_log({f"val/{encoder_kind}/epoch_loss": val, "epoch": epoch})
        save_checkpoint(encoder_kind, wrapper, optimizer, epoch,
                        n_batches, global_step, best_val,
                        val_loss=val, is_best=is_best)
        start_batch = 0

    print(f"[done] {encoder_kind} encoder trained. "
          f"best weights: {find_weights(f'{encoder_kind}_encoder', 'best')}")


# --------------------------------------------------------------------------- #
# Stage driver
# --------------------------------------------------------------------------- #
def run_stages(args, device):
    progress = load_progress()
    for stage in STAGES:
        if stage in progress["completed"]:
            continue

        init_wandb(args.wandb_key, stage)
        kind = "text" if stage.startswith("text") else "image"

        if stage.endswith("tune"):
            tune(kind, device)
            mark_stage_done(stage)
            finish_wandb()
            print(f"\n*** Stage '{stage}' complete. Stopping for review. "
                  f"Re-submit the job to continue with training. ***")
            return                       # internal stop for review
        else:
            train_encoder(kind, device)
            mark_stage_done(stage)
            finish_wandb()
            # training stages auto-advance into the next (tuning) stage

    print("\nAll encoder-pretraining stages complete.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Encoder pretraining.")
    p.add_argument("--token", default="_", help="Hugging Face token.")
    p.add_argument("--wandb_key", default="_", help="Weights & Biases API key.")
    p.add_argument("--prepare_data_only", action="store_true",
                   help="Download + cache every dataset, then exit.")
    p.add_argument("--reset_progress", action="store_true",
                   help="Clear stage progress (does not delete weights/data).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    if args.token and args.token != "_":
        try:
            from huggingface_hub import login
            login(token=args.token)
        except Exception as e:
            print(f"[hf] login skipped: {e}")

    if args.reset_progress:
        save_progress({"completed": []})
        print("[progress] reset.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    if args.prepare_data_only:
        # Only the text datasets are populated up front. The image datasets are
        # all on HuggingFace and are pulled lazily when the image-tuning stage
        # begins, so we deliberately do not download them here.
        gcfg = dict(HYPERPARAMS["global"])
        print("\n[data] preparing text datasets (image datasets are pulled "
              "later, at the image-tuning stage)...")
        try:
            build_tasks("text", gcfg)
        except Exception as e:
            print(f"[data] text: {e}")
        print("[data] text preparation complete.")
        return

    run_stages(args, device)


if __name__ == "__main__":
    sys.exit(main())
