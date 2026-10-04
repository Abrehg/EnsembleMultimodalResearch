"""
Encoder pretraining (text encoder + OCR encoder + image encoder, in a vacuum).

Overview
--------
This script pretrains the project's encoders -- ``TextEncoder``, ``OCREncoder``,
and ``VisionEncoder`` -- *independently of the multimodal model*. The text and
image encoders are each temporarily wrapped with their own small decoder
transformer (2-4 layers) and a set of task-specific heads, trained on a spread
of tasks, and then the decoder is thrown away: only the encoder weights are
kept as the reusable artifact. The OCR encoder has no throwaway decoder --
every stage of its 4-stage architecture (see models/Encoders/OCREncoder.py) is
permanent, so its pretraining wrapper only adds a small auxiliary head used
purely to shape the pretraining loss (see "OCR encoder pretraining" below).

Unlike the image encoder, the OCR encoder is *not* paired into the rest of
the model here -- it is trained in isolation, the same as the text and
vision encoders are, and its pairing with a frozen text encoder for
document-QA fine-tuning happens in a separate script, ``ocr_to_text.py``,
which runs after ``text_to_text.py`` (mirroring how the image encoder's own
downstream pairing happens outside this file too, just staged differently:
image_tune/image_train below fold the frozen text encoder in directly,
because the image encoder needs it during pretraining itself for
CLIP-style semantic segmentation supervision, whereas OCR pretraining needs
no text encoder at all).

Pipeline (staged; the program stops for review after every *tuning* stage,
and after Phase 0b-1a/b -- see below):

    1. text_tune          -> Optuna search for the text encoder's
                             hyperparameters, then exit(0) for review.
    2. text_train          -> train the text encoder wrapper (resumable),
                             save encoder.
    3. ocr_tune            -> Optuna search for the OCR encoder's
                             pretraining hyperparameters, then exit(0) for
                             review.
    4. ocr_train           -> train the OCR encoder wrapper (resumable) in
                             isolation (no other encoder involved), save
                             encoder. See "OCR encoder pretraining" below.
    5. vision_ssl_pretrain -> Phase 0b-1a/b: DINOv3-style self-supervised
                             pretraining of the vision encoder (DINO + iBOT +
                             KoLeo, EMA teacher). Stops for manual review
                             every time it reaches its target iteration count
                             (see HYPERPARAMS["ssl"]["proceed_to_gram_anchoring"]).
    6. vision_gram_anchor  -> Phase 0b-1c: adds Gram anchoring against a
                             frozen teacher snapshot to fix dense-feature
                             degradation from extended training.
    7. vision_hires_adapt  -> Phase 0b-1d: short high-resolution adaptation
                             run (512/768 global, 112-336 local crops).
    8. image_tune          -> Phase 0b-2 (supervised task-head fine-tuning):
                             Optuna search for the image encoder's
                             hyperparameters, using the frozen best text
                             encoder and the vision_hires_adapt checkpoint as
                             the encoder's starting point.
    9. image_train         -> train the image encoder wrapper (resumable)
                             with the frozen text encoder, save encoder.

See the "Vision SSL pretraining" section further down for the full DINOv3-
inspired sub-pipeline (Phases 0b-1a through 0b-1d): what each stage's loss
is, how the EMA/Gram teachers work, and why the encoder architecture (CLS +
register tokens, axial RoPE, dual crop-type norms, patch masking) changed to
support it -- see also models/Encoders/VisionEncoder.py's docstring.

OCR encoder pretraining
------------------------
Runs immediately after text_train, before any vision stage (see module
docstring's pipeline list) -- entirely separate data, architecture, and
losses from both the text/image Task machinery and the vision SSL
sub-pipeline. Based on OCR_Encoding_Expert_Methodology.md: a 4-stage
architecture (local window-attention feature extraction -> global context
processing -> fixed-size compression bridge -> OCR-specific refinement, plus
an optional handwriting classifier -- see models/Encoders/OCREncoder.py's
docstring) trained with a composite loss:

    - content: per-Stage-1-patch binary classifier (does this patch contain
      OCR-relevant ink?), supervised directly from pixel statistics -- see
      ``_ocr_content_mask``. Deliberately *not* supervised from external
      word/character bounding-box annotations: those schemas vary and drift
      across datasets, whereas ink-density works identically for a real
      rendered PDF page, a handwriting scan, or anything else, with no
      annotation dependency at all.
    - diversity: reuses this file's existing ``koleo_loss`` (already used by
      the vision SSL pipeline) over every output token in the batch, instead
      of the covariance-determinant formulation in the reference methodology
      -- that formulation is degenerate whenever batch_size < hidden_dim
      (the covariance matrix is rank-deficient, so its determinant is
      exactly zero, not just small), which is the common case here.
    - spatial coherence: L2 distance between spatially-consecutive Stage-1
      patch tokens (methodology-faithful, numerically fine as specified).
    - line coherence: tokens within the same "line band" -- a run of
      content-bearing patch-grid rows, again pixel-derived rather than
      annotation-derived (see ``_ocr_line_bands``) -- pulled toward their
      band's mean via cosine similarity.
    - handwriting: binary cross-entropy against the source dataset's known
      typed/handwritten label (PDF source = typed, IAM source = handwritten).

Data: real PDF documents (``PDFA_HF``, streamed as raw PDF bytes and
rendered to page images with pypdfium2 -- this is the concrete difference from
a generic image-based OCR pipeline: the encoder trains on genuinely-rendered
PDF pages, not pre-rasterised document-image datasets) mixed with real
handwriting-line images (``IAM_LINES_HF``), at
``HYPERPARAMS["ocr"]["handwritten_fraction"]``. Both sources are tried
defensively, same "skip and continue" philosophy as every other dataset in
this file.

Progress is persisted, so a resubmitted job skips finished stages. Training is
itself resumable mid-epoch (or, for the SSL phases, mid-iteration-count):
checkpoints are labelled with epoch + batch number and, after each validation,
the validation loss is embedded in the filename. Weights follow a fixed,
greppable naming scheme (see ``weight_path``):

    {component}__{tag}__epoch{E}_batch{B}[_valloss{L}].pt
        component in {text_encoder, text_wrapper, ocr_encoder, ocr_wrapper,
                      image_encoder, image_wrapper,
                      vision_ssl_pretrain_encoder, vision_ssl_pretrain_wrapper,
                      vision_gram_anchor_encoder, vision_gram_anchor_wrapper,
                      vision_hires_adapt_encoder, vision_hires_adapt_wrapper}
        tag       in {latest, best}

``find_weights(component, tag)`` locates them programmatically. For the
vision_* SSL components, the "*_encoder" artifact is always the EMA
*teacher's* encoder (the model downstream stages should build on), not the
noisier, more frequently updated, masked student.

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

Vision SSL pretraining (Phases 0b-1a/b/c/d)
--------------------------------------------
Before the supervised task-head fine-tuning above (now Phase 0b-2), the
vision encoder goes through a DINOv3-inspired self-supervised pretraining
sub-pipeline, entirely separate from the text/image Task machinery described
above (different data, different losses, no task heads or decoder).

Architecture changes this required (models/Encoders/VisionEncoder.py):
    - a learned CLS token + 4 learned register tokens, concatenated ahead of
      the patch tokens (registers absorb high-norm "attention sink"
      activations so patch-level features stay clean -- Darcet et al.,
      "Vision Transformers Need Registers");
    - axial RoPE replacing the additive 2D sincos position embedding, so the
      encoder generalises to resolutions/aspect ratios never seen in
      training (used by the 0b-1d high-res adaptation phase);
    - two dedicated final LayerNorms, selected by ``crop_type``, since local
      and global crops have different statistics;
    - patch masking (``VisionEncoder.mask_patches``), toggled per call via
      ``forward(..., mask=...)`` -- the student sees masked global crops (for
      the iBOT loss), the teacher never does.
The video (multi-frame) path reuses this exact per-frame pass and just adds
a fixed 1D sincos embedding over time on top -- no separate temporal encoder.

Stage breakdown:
    vision_ssl_pretrain (0b-1a/b) -- L = L_DINO + L_iBOT + 0.1*L_KoLeo.
        DINO: image-level (CLS token) cross-entropy between the teacher's
        Sinkhorn-Knopp-centered soft assignment (SwAV-style, replacing DINO's
        original running-mean centering, per DINOv3) and the student's
        softmax, over every (teacher global crop, student crop) pair except a
        crop against itself. iBOT: the same mechanic at the patch level,
        restricted to masked positions. KoLeo: encourages the batch's CLS
        embeddings to spread out (see ``koleo_loss``). Constant LR (after a
        10k-step warmup) and constant EMA momentum -- no schedule at all, so
        training can be extended indefinitely (bump
        HYPERPARAMS["ssl"]["target_iters"] and resubmit) rather than needing
        to know the optimization horizon up front. A Gram-teacher candidate
        snapshot is saved automatically once iterations fall in
        [gram_candidate_min_iter, gram_candidate_max_iter]. This stage always
        stops for manual review once it reaches target_iters (inspect the
        patch-similarity PNG it writes to DATA_DIR): flip
        HYPERPARAMS["ssl"]["proceed_to_gram_anchoring"] to True and resubmit
        once you're ready to move on, whether because the maps are
        degrading or you're simply satisfied with the training budget.

    vision_gram_anchor (0b-1c) -- adds L_Gram = ||X_S X_S^T - X_G X_G^T||^2
        (L2-normalised patch features, global crops only) against a frozen
        "Gram teacher" -- an earlier, dense-feature-healthy snapshot -- to
        repair the patch-level consistency that degrades over long training
        schedules (see DINOv3 Sec. 4). The Gram teacher itself is refreshed
        from the live EMA teacher every gram_teacher_update_every steps.

    vision_hires_adapt (0b-1d) -- short run at larger, mixed crop
        resolutions (512/768 global, 112-336 local), Gram teacher = the
        current EMA teacher (frozen at the start of this phase, not the
        0b-1a/b candidate). Teaches the encoder to generalise to inference
        resolutions larger than anything used in training; RoPE (rather than
        an absolute position embedding) is what makes this possible.

Data: every batch streams from IMAGENET_HF (see ``SSLImageStream``).
ImageNet-21K and OpenImages were both tried as additional "background pool"
sources for diversity, but both showed real, repeated connectivity failures
against this cluster's proxy in practice -- see the comment above
IMAGENET_HF's definition for the full removal rationale -- and were removed
rather than hardened further, leaving IMAGENET_HF as the sole source.

Once Phase 0b-1d completes, only the encoder's weights carry forward: the
DINO/iBOT heads, the Gram teacher, and (implicitly) the local-crop norm are
all training scaffolding that Phase 0b-2 never touches (``ImagePretrainWrapper``
always calls the encoder with the default ``crop_type="global"``).
"""

import argparse
import glob
import json
import math
import os
import random
import re
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

# Bulk, regenerable data (raw/cached datasets) lives on scratch; anything
# that's small but load-bearing for a multi-day staged run -- tuned
# hyperparameters, the Optuna study DB, stage progress, trained weights --
# lives under the project directory on barn instead, since scratch
# filesystems are periodically purged and losing any of these would
# silently reset or corrupt an in-progress pipeline (weights were already
# here; STUDY_DIR/PROGRESS_PATH moved off scratch for the same reason).
SCRATCH_DIR = "/gpfs/u/home/ENMM/ENMMsrtk/scratch/"
DATA_DIR = os.path.join(SCRATCH_DIR, "enc_pre_data")
# All weight files -- including the resumable "_wrapper" checkpoints --
# live here on barn, never on scratch. scratch is reserved strictly for
# datasets/HF cache.
WEIGHTS_DIR = os.path.join(PROJECT_DIR, "weights")
STUDY_DIR = os.path.join(PROJECT_DIR, "studies", "enc_pre")
PROGRESS_PATH = os.path.join(PROJECT_DIR, "enc_pre_progress.json")
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
        "d_model": 1024,               # encoder width (fixed here; tuned in dim_tune)
        "val_holdout": 1000,          # entries held out for validation (variable)
        "eval_every_batches": 500,    # run validation every N optimizer steps
        "checkpoint_every_batches": 200,  # write a resumable checkpoint every N
        # Only a running 'latest' and 'best' checkpoint are ever kept per
        # component (see save_checkpoint) -- a single text_wrapper
        # checkpoint at d_model=1024 is already ~700MB+ once optimizer
        # state is included, so this bounds disk usage by a small constant
        # regardless of run length instead of accumulating a growing
        # history of intermediate checkpoints.
        "seed": 42,
        "max_text_len": 64,           # token cap for text tasks
        # SQuAD contexts run ~150-200 BPE subword tokens, far past the
        # shared max_text_len (64) budget most other text tasks are fine
        # with. SquadTask.collate computes each answer's token position
        # against the *untruncated* context, then checks it against the
        # truncated length -- so when max_text_len is this small, the
        # overwhelming majority of genuinely-answerable questions have
        # their answer fall outside the surviving window and get silently
        # relabelled as "no answer" (position 0), poisoning the training
        # signal with mostly-fake unanswerable labels. Giving SQuAD its own
        # larger budget (not raising max_text_len globally, which would
        # inflate compute for every other, much lighter text task) fixes
        # this without touching the rest of the tasks.
        "squad_max_text_len": 256,
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

    # OCR encoder pretraining (see module docstring's "OCR encoder
    # pretraining" section and models/Encoders/OCREncoder.py). Trained in
    # isolation, immediately after text_train and before any vision stage --
    # no other encoder is involved.
    "ocr": {
        "epochs": 9,
        "per_task_cap": 40_000,        # total OCR documents (typed + handwritten)
        "handwritten_fraction": 0.3,   # fraction of per_task_cap drawn from IAM
        # Page render resolution (long side, then padded to a square canvas
        # -- see _load_ocr_image_tensor). Deliberately *not* read from
        # HYPERPARAMS["global"]["image_size"] (128, tuned for the much
        # smaller generic image tasks) -- OCR needs far more resolution to
        # keep individual characters legible. Must stay a multiple of
        # patch_size * window_size (16 * 7 = 112) so the rendered canvas
        # already tiles evenly into the encoder's attention windows with no
        # further padding; OCREncoder.py pads defensively if this is ever
        # changed to a non-multiple, but pick a multiple of 112 here.
        "image_size": 896,
        "content_ink_threshold": 0.08,  # per-patch non-background pixel fraction
                                        # that counts as "this patch has ink"
        "min_line_content_patches": 2,  # patch-row needs >= this many content
                                        # patches to count as a text-line band
        "output_tokens": 256,           # OCREncoder's fixed output token count
        "pdf_render_dpi": 150,
        "tune_trials": 10,
        "tune_budget_batches": 200,
        "defaults": {
            "batch_size": 8,
            "learning_rate": 2e-4,
            "weight_decay": 1e-2,
            "warmup_steps": 200,
            "content_weight": 1.0,
            "diversity_weight": 0.1,
            "spatial_weight": 0.05,
            "line_weight": 0.1,
            "handwriting_weight": 0.2,
        },
        "search_space": {
            "batch_size": [4, 8, 16],
            "learning_rate": (1e-4, 1e-3),
            "weight_decay": (1e-5, 1e-1),
            "warmup_steps": [0, 100, 300],
            "content_weight": [0.5, 1.0, 2.0],
            "diversity_weight": [0.05, 0.1, 0.2],
            "spatial_weight": [0.02, 0.05, 0.1],
            "line_weight": [0.05, 0.1, 0.2],
            "handwriting_weight": [0.1, 0.2, 0.4],
        },
    },

    # DINOv3-style self-supervised pretraining for the vision encoder, run
    # before the (existing) image_tune/image_train supervised-task stages.
    # Values are scaled down from the paper's (7B model, 256-GPU) recipe to
    # something a single-GPU job can actually execute; every knob here is
    # still the same knob DINOv3 exposes, just smaller magnitudes. See the
    # module docstring's "Vision SSL pretraining" section for the phase
    # breakdown (vision_ssl_pretrain -> vision_gram_anchor -> vision_hires_adapt).
    "ssl": {
        "batch_size": 64,
        "lr": 0.0004,
        "warmup_steps": 10_000,
        "weight_decay": 0.04,
        "teacher_momentum": 0.999,          # constant EMA momentum, no schedule
        "dino_out_dim": 8192,               # scaled down from the paper's 256k
        "ibot_out_dim": 8192,               # scaled down from the paper's 96k
        "head_hidden_dim": 2048,
        "head_bottleneck_dim": 256,
        "sinkhorn_iters": 3,
        "sinkhorn_epsilon": 0.05,
        "student_temp": 0.1,
        "koleo_weight": 0.1,
        # dino_sharpness_loss (see its own docstring): fights a DIFFERENT
        # collapse mode than koleo_weight above -- intra-sample softmax
        # flatness, not inter-sample direction collapse -- diagnosed from a
        # real run (enc_pre_slurm-4795027) where `dino` sat pinned at
        # log(dino_out_dim)=9.011 from ~iteration 25,000 onward despite
        # koleo staying healthy throughout. Entropy is on the SAME 0..
        # log(K) scale as dino_loss/ibot_loss themselves (unlike koleo,
        # which operates on a different scale and needed the 0.1
        # down-weight), so this starts at a moderate 0.5 -- enough to give
        # it real pull without letting it dominate the main cross-entropy
        # objectives. No prior tuning data for this exact term yet; revisit
        # once a verification run's own "sharpness" metric shows whether
        # dino actually breaks away from the log(K) ceiling at this weight.
        "sharpness_weight": 0.5,
        "gram_weight": 2.0,
        "mask_ratio_range": (0.1, 0.5),
        "n_global_crops": 2,
        "n_local_crops": 8,
        "global_crop_size": 224,
        "local_crop_size": 96,
        "global_crop_scale": (0.32, 1.0),
        "local_crop_scale": (0.05, 0.32),

        # ---- Image source pools (see SSLImageStream / _materialize_image_pool) -- #
        # Each source is pre-downloaded to local disk ONCE (rank0_first-gated,
        # same idiom as every Task.download(cap) pool below) and cycled from
        # that local copy for the entire (very long) SSL run -- no network
        # dependency during actual training. `None` = materialize the whole
        # split; a number bounds the one-time download (IMAGENET21K_HF is
        # 281GB total, PASS/iNaturalist are 1.4M/2.7M images -- capping all
        # three is essential regardless of network reliability, for local
        # disk footprint alone).
        "imagenet1k_pool_cap": None,
        "imagenet21k_pool_cap": 500_000,
        "pass_pool_cap": 500_000,
        "inaturalist_pool_cap": 500_000,
        # Each training batch's per-source mix -- renormalized at runtime
        # over whichever sources actually materialize (see SSLImageStream),
        # so a source that fails to download doesn't silently vanish a
        # fraction of every batch, it just redistributes proportionally
        # among the survivors. imagenet21k is the primary corpus; pass/
        # inaturalist are diversity supplements (replacing OpenImages/
        # CC12M respectively -- see IMAGENET21K_HF's neighboring comments
        # for why those two were dropped and why re-adding *these* two
        # doesn't reintroduce that risk); imagenet1k is the "10%
        # homogeneous batches" baseline. Must sum to 1.0.
        "ssl_source_fractions": {
            "imagenet1k": 0.10,
            "imagenet21k": 0.70,
            "pass": 0.10,
            "inaturalist": 0.10,
        },

        # Phase 0b-1a/b: initial + continued SSL pretraining.
        "target_iters": 200_000,             # bump + resubmit to extend 0b-1b
        "gram_candidate_min_iter": 100_000,   # window for saving a Gram-teacher snapshot
        "gram_candidate_max_iter": 200_000,
        "eval_every": 20_000,                # patch-similarity visualization cadence
        "checkpoint_every": 2_000,
        # Manual gate: after reaching target_iters, the stage always stops for
        # review (see docstring) until this is flipped to True, at which point
        # the *next* run marks the stage complete and moves on to Gram
        # anchoring -- this is the "clean maps -> keep going, degrading maps
        # -> proceed" decision from the phase spec, made explicit instead of
        # automated (patch-map quality isn't something to threshold blindly).
        "proceed_to_gram_anchoring": False,

        # Phase 0b-1c: Gram anchoring refinement.
        "gram_anchor_iters": 50_000,
        "gram_teacher_update_every": 10_000,

        # Phase 0b-1d: high-resolution adaptation.
        "hires_iters": 10_000,
        "hires_global_crop_sizes": [512, 768],
        "hires_local_crop_sizes": [112, 168, 224, 336],
    },
}

STAGES = [
    "text_tune", "text_train",
    "ocr_tune", "ocr_train",
    "vision_ssl_pretrain", "vision_gram_anchor", "vision_hires_adapt",
    "image_tune", "image_train",
]

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
NUM_REGISTERS = 4                # VisionEncoder register tokens (see VisionEncoder.py)

# Dataset identifiers.
CONLL_HF = "eriktks/conll2003"
UD_HF = "universal-dependencies/universal_dependencies"
UD_CONFIG = "en_ewt"
IMAGENET_HF = "ILSVRC/imagenet-1k"
# Gated (accept terms + a logged-in HF token/--token, same prerequisite
# ImageNet-1k already has). Ships in webdataset format -- confirmed via its
# HF Data Studio viewer: "train" split (~221k rows; a validation split also
# exists but is unused here), columns "__key__"/"__url__" (webdataset
# housekeeping) + "jpeg" (the image, auto-decoded to a PIL Image by
# `datasets`' webdataset loader, same as IMAGENET_HF's "image" column just
# under a different key -- see _materialize_image_pool's decode logic,
# which checks "jpeg" first for exactly this reason). Re-added (see
# SSLImageStream/HYPERPARAMS["ssl"]'s pool comment) after an earlier mirror
# (gmongaras/Imagenet21K) was removed for HF-tunnel failures -- this is a
# different mirror, and more importantly the local-pool design below
# eliminates the actual failure mode (a live network dependency spanning
# the entire SSL run) regardless of which mirror is used, so reintroducing
# it no longer carries the same risk.
IMAGENET21K_HF = "zeyuanyin/imagenet-21k-p"
# Diversity supplements (see HYPERPARAMS["ssl"]'s pool comment). Both go
# through the same local-pool materialization as everything else here, so
# neither reintroduces the live-network-dependency risk OpenImages/CC12M
# were originally dropped for -- see IMAGENET21K_HF's comment above.
#
# PASS: confirmed real via its HF Data Studio page -- standard decoded
# "image" column (same shape as IMAGENET_HF's), ~1.44M images, single
# "train" split, not gated. Official author upload, replaces OpenImages
# (same "diversity pool" role OpenImages was originally chosen for).
PASS_HF = "yukimasano/pass"
# iNaturalist 2021 mini: no working HuggingFace mirror exists (the
# previously-tried "mikehemberger/inaturalist-2021" id 401'd on both its
# dataset page and the HF datasets API, and doesn't turn up in a search of
# that uploader's actual public datasets -- almost certainly not a real,
# public dataset). Downloaded instead via direct wget from iNaturalist's
# own AWS S3 host (a different network endpoint than HF/browser CDNs) and
# extracted locally -- confirmed present: 9,712 species folders under
# INATURALIST_ROOT, ~50 images each (~485K total), matching the official
# "train_mini" split's documented shape. Loaded by walking this local
# directory directly with PIL (see _materialize_inaturalist_pool) rather
# than torchvision.datasets.INaturalist, since torchvision isn't a
# dependency this project otherwise uses (see _random_resized_crop's
# comment for the same reasoning applied elsewhere in this file).
INATURALIST_ROOT = os.path.join(SCRATCH_DIR, "inaturalist", "train_mini")
NYU_DEPTH_HF = "sayakpaul/nyu_depth_v2"
# vrd moved off the HF mirror (Voxel51/GQA-Scene-Graph is a FiftyOne export
# whose samples.json relationship annotations the generic image-folder
# auto-loader never ingests -- see test_datasets.py's KNOWN_PENDING_REPLACEMENT
# for the full diagnosis) to GQA's own official release, for publication
# provenance -- see VRDTask.download's comment. Direct HTTPS from Stanford's
# own CDN, CC BY 4.0 licensed, no auth/quota concerns.
GQA_SCENEGRAPHS_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip"
GQA_IMAGES_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
ADE20K_HF = "merve/scene_parse_150"
# OCR encoder pretraining sources (see "OCR encoder pretraining" above):
# PDFA_HF streams raw PDF bytes (rendered to page images ourselves with
# pypdfium2, not pre-rasterised); IAM_LINES_HF is real handwriting-line images.
PDFA_HF = "pixparse/pdfa-eng-wds"
IAM_LINES_HF = "Teklia/IAM-line"
# SSL pretraining's image sources are IMAGENET_HF + IMAGENET21K_HF (see
# HYPERPARAMS["ssl"] and SSLImageStream). An earlier IMAGENET21K_HF mirror
# (gmongaras/Imagenet21K) and OpenImages (direct per-image HTTPS GETs
# against a raw S3 bucket, a fundamentally different traffic pattern from
# every other source here) were both removed after real, repeated
# connectivity failures against this cluster's proxy -- but that failure
# mode was specifically a *live, ongoing* network dependency across the
# entire, very long SSL run (every one of target_iters batches, across
# every one of world_size ranks independently -- see SSLImageStream's
# docstring). SSLImageStream no longer works that way: every source is
# pre-downloaded to local disk ONCE, then cycled from disk for the rest of
# the run, so reintroducing a second source (a different mirror, this time)
# no longer reintroduces that risk -- the one remaining network dependency
# is a single bounded, one-time download, not something the whole
# multi-day run depends on staying up.

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
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def mark_stage_done(stage):
    progress = load_progress()
    if stage not in progress["completed"]:
        progress["completed"].append(stage)
    save_progress(progress)


# --------------------------------------------------------------------------- #
# Greppable weight naming + lookup
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
# min(..., key=_valloss).
_VALLOSS_RE = re.compile(r"valLoss(\d+\.\d+)")


def _is_epoch_complete(path):
    """True only if this filename carries the literal 'complete' marker --
    never inferred from a batch number happening to equal an epoch's total
    batch count (see weight_path / train_encoder's resume logic)."""
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


def _find_valid_resume(component, device):
    """Try the 'latest' checkpoint for this component, falling back to
    'best' if 'latest' is missing or unreadable. Training maintains only a
    running 'latest' and 'best' per component (see save_checkpoint), not a
    growing history of 'step' files, so 'best' is the only other candidate
    left to fall back to -- this still protects against exactly what
    motivated it: a stale, truncated checkpoint from an old disk-write
    failure sitting on disk and crashing every subsequent resume attempt
    immediately in torch.load, before the run ever gets a chance to train.
    Corrupt files are deleted as they're found. Returns (path, blob), or
    (None, None) if neither loads."""
    for tag in ("latest", "best"):
        f = find_weights(component, tag)
        if not f:
            continue
        try:
            return f, torch.load(f, map_location=device)
        except Exception as e:
            print(f"[resume] {os.path.basename(f)} is unreadable ({e}); "
                  f"deleting it and trying an older checkpoint")
            try:
                os.remove(f)
            except OSError:
                pass
    return None, None


def _prune_other_tagged_files(component, tag, keep_path):
    """Delete every other file previously saved under this exact
    (component, tag) pair, now that ``keep_path`` has landed successfully --
    called only *after* a new file is written, so a failed write never
    leaves zero valid checkpoints behind. Combined with save_checkpoint only
    ever writing 'latest' (always) and 'best' (only on improvement), this
    keeps disk usage bounded by two files per component regardless of run
    length, instead of accumulating a 'step' file per checkpoint event --
    the direct cause of the scratch-quota exhaustion and repeated torch.save
    write failures seen on long runs previously.

    Refuses to delete a file representing *more* progress (higher
    (epoch, batch)) than keep_path. Without this, a run whose resume missed
    an existing, further-along checkpoint -- a stale filesystem view on a
    different node, or two jobs for the same stage accidentally running at
    once -- would have its very first save irreversibly destroy the
    more-advanced checkpoint it should have resumed from instead, exactly
    the failure mode that motivated this check. Loudly warns and keeps both
    files when that happens rather than silently losing the better one."""
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


def _drop_wrapper_files(component):
    """Delete the resumable 'wrapper' checkpoint (both 'latest' and 'best'
    tags) for this component. Its only purpose is enabling mid-run resume
    (optimizer state, RNG, epoch/batch counters) -- moot once a stage has
    genuinely finished -- so calling this once a stage completes leaves just
    the 2 reusable-artifact files (encoder 'best'/'latest') behind instead
    of 4. Never call this for a stage that might still be resumed later
    (e.g. vision_ssl_pretrain paused for manual review, not yet proceeding)."""
    for tag in ("latest", "best"):
        f = find_weights(component, tag)
        if f:
            try:
                os.remove(f)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# Incremental LR decay: linear warmup, then cosine decay down to a floor
# fraction of the peak LR, so later optimizer steps make progressively
# smaller changes as training approaches convergence. Shared design with
# text_to_text.py's ``make_lr_lambda``.
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


def _atomic_torch_save(obj, path, retries=3, backoff_seconds=5.0):
    """torch.save is not atomic and not resilient to transient filesystem
    hiccups -- a write failure partway through (seen in practice as
    PytorchStreamWriter "file write failed" on a flaky GPFS mount) leaves
    the exception propagating out of the middle of the save, and has been
    observed to leave the process stuck on the follow-up close() instead of
    cleanly crashing, hanging the whole job until the SLURM time limit kills
    it. Writing to a temp file and renaming into place means a failed
    attempt never produces a half-written checkpoint at ``path``, and
    retrying with a short backoff rides out a transient glitch instead of
    taking the whole run down with it."""
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


def _urlretrieve_with_timeout(url, path, timeout=30, retries=3, backoff_seconds=5.0):
    """urllib.request.urlretrieve has no way to pass a per-call timeout (it
    silently falls back to the process-wide socket default, which is None
    -- i.e. blocks forever) -- a stalled connection on a flaky proxy (seen
    in practice: this exact class of hang, on the Open Images image-id CSV
    download, sitting completely silent for hours with no error and no
    further job output, right as wandb was independently hitting 429
    rate-limit retries against the same network path) leaves the whole job
    deadlocked until SLURM's wall-clock limit kills it -- every other rank
    stuck waiting at rank0_first's barrier for a call that was never going
    to return. Streams via urlopen(..., timeout=timeout) instead (which DOES
    accept one), writing to a temp file and renaming into place (same
    corruption-avoidance reasoning as _atomic_torch_save), retrying with a
    short backoff on any timeout/network error."""
    import urllib.request

    tmp_path = f"{path}.tmp{os.getpid()}"
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp_path, path)
            return
        except Exception as e:
            last_err = e
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if attempt < retries - 1:
                print(f"[download] {url} failed ({e}); retrying ({attempt + 1}/{retries})...")
                time.sleep(backoff_seconds)
    raise last_err


def save_checkpoint(prefix, wrapper, optimizer, epoch, batch, global_step,
                    best_val_loss, val_loss=None, epoch_complete=False):
    """Persist the reusable encoder weights -- via the encoder's own
    store_weights, its canonical save method (see models/Encoders/*.py) --
    and a full resumable wrapper checkpoint (optimizer/RNG/epoch state; not
    itself an nn.Module, so it has no component-level store_weights of its
    own) under the 'latest' tag, always, and also under 'best' when
    val_loss improves on best_val_loss. Only ever one 'latest' and one
    'best' file exist per component at a time -- the previous file for that
    exact tag is deleted right after the new one lands successfully (see
    _prune_other_tagged_files) -- rather than keeping a growing 'step'
    history, so disk usage stays bounded by a small constant regardless of
    run length. ``epoch_complete=True`` marks this as the checkpoint
    written once ``epoch`` has genuinely finished (see weight_path /
    _is_epoch_complete); only that literal marker -- never a batch number
    alone -- makes train_encoder's resume logic treat an epoch as done."""
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

    tags = ["latest"]
    if val_loss is not None and val_loss <= best_val_loss:
        tags.append("best")

    for tag in tags:
        ep = weight_path(enc, tag, epoch, batch, val_loss, epoch_complete=epoch_complete)
        wp = weight_path(wrp, tag, epoch, batch, val_loss, epoch_complete=epoch_complete)
        wrapper.encoder.store_weights(WEIGHTS_DIR, os.path.basename(ep))
        _atomic_torch_save(resume_blob, wp)
        _prune_other_tagged_files(enc, tag, ep)
        _prune_other_tagged_files(wrp, tag, wp)


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
        self.encoder = createVisionEncoder(dim=d)   # was missing dim=d; also
                                                     # required for the SSL
                                                     # checkpoint below to load
        # Prefer the most-refined SSL phase's output (Phase 0b-1d high-res
        # adaptation), falling back to earlier phases if a later one hasn't
        # been run yet (see the module docstring's Vision SSL section).
        ssl_ckpt = None
        for ssl_prefix in ("vision_hires_adapt", "vision_gram_anchor", "vision_ssl_pretrain"):
            ssl_ckpt = (find_weights(f"{ssl_prefix}_encoder", "best")
                       or find_weights(f"{ssl_prefix}_encoder", "latest"))
            if ssl_ckpt:
                break
        if ssl_ckpt:
            self.encoder.load_weights(ssl_ckpt, map_location="cpu")
            print(f"[image] encoder initialised from SSL pretraining "
                  f"checkpoint {os.path.basename(ssl_ckpt)}")
        else:
            print("[image] WARNING: no SSL pretraining checkpoint found "
                  "(vision_hires_adapt/vision_gram_anchor/vision_ssl_pretrain); "
                  "encoder starts from random init")
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


class OCRPretrainWrapper(nn.Module):
    """OCR encoder pretraining wrapper (see module docstring's "OCR encoder
    pretraining" section). Unlike Text/ImagePretrainWrapper, every
    architectural stage -- local window-attention extraction, global
    context, compression bridge, OCR refinement, handwriting classifier --
    lives inside OCREncoder itself and is kept as the reusable artifact;
    there is no throwaway decoder here. The only pretraining-only scaffolding
    is a per-patch content-presence head (does this Stage-1 patch contain
    OCR-relevant ink? -- see ``_ocr_content_mask`` / ``ocr_content_loss``),
    used purely to shape the encoder's dense features during pretraining."""

    def __init__(self, config, active_tasks=None):
        super().__init__()
        from models.Encoders.OCREncoder import createOCREncoder

        self.config = config
        d = config["d_model"]
        self.encoder = createOCREncoder(
            dim=d, num_output_tokens=HYPERPARAMS["ocr"]["output_tokens"],
        )
        self.content_head = nn.Linear(d, 1)     # per-patch logit, Stage-1 grid

    def forward_pretrain(self, images):
        """Runs every OCREncoder stage manually (rather than calling
        ``encoder.forward``) so pretraining losses can reach Stage 1's patch
        grid (content/spatial/line losses) as well as the final refined
        output (diversity/handwriting losses) from a single forward pass."""
        patch_tokens, grid_h, grid_w = self.encoder.local_encoder(images)
        contextualized = self.encoder.global_encoder(patch_tokens)
        compressed = self.encoder.compression_bridge(contextualized)
        refined = self.encoder.refiner(compressed)
        content_logits = self.content_head(patch_tokens).squeeze(-1)   # [B, grid_h*grid_w]
        hw_logits = self.encoder.handwriting_logits(refined)
        return {
            "patch_tokens": patch_tokens, "grid_h": grid_h, "grid_w": grid_w,
            "refined": refined, "content_logits": content_logits, "hw_logits": hw_logits,
        }


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
            # See _materialize_image_pool's identical fix (enc_pre_slurm-4795027
            # postmortem) -- this branch used to print the exact same line as
            # the fresh-download branch below, so there was no way to tell
            # from the log alone whether a source (e.g. depth, precomputed
            # externally and manually placed at DATA_DIR/depth) was actually
            # using that precomputed cache or silently re-downloading anyway.
            print(f"[data] {self.name}: found existing local cache -> {path}, "
                  f"skipping download()", flush=True)
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
        from datasets import Dataset

        os.makedirs(DATA_DIR, exist_ok=True)
        rows = []
        for fname in GAP_FILES:
            local = os.path.join(DATA_DIR, fname)
            if not os.path.isfile(local):
                url = f"{GAP_BASE}/{fname}"
                print(f"[data] downloading GAP file {url}")
                _urlretrieve_with_timeout(url, local)
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
        # Own (larger) budget, not the shared max_text_len -- see
        # HYPERPARAMS["global"]["squad_max_text_len"] for why.
        max_len = cfg.get("squad_max_text_len", cfg["max_text_len"])
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

        # name="default" -- see _materialize_image_pool's docstring for why
        # this is needed for a script-based source: without an explicit
        # config name, datasets 2.19.2's dataset_module_factory() probes the
        # script file via HfFileSystem before downloading it properly, and
        # that probe fails against NYU_DEPTH_HF with a UnicodeDecodeError
        # (it reads the hub CDN's gzip-encoded response as raw UTF-8).
        # nyu_depth_v2.py defines no BuilderConfig subclasses, so "default"
        # is its only/implicit config. Also requires h5py to be installed
        # (the script reads HDF5 files) -- confirmed missing from mulEnv as
        # of this fix; install it before this source will actually work.
        ds = load_dataset(NYU_DEPTH_HF, name="default", split="train",
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
def _download_gqa_scene_graphs():
    """Downloads + extracts GQA's official scene graph annotations once,
    cached under DATA_DIR/gqa_scenegraphs_raw (42.7MB zip -- see
    GQA_SCENEGRAPHS_URL's comment). Small enough to always fetch alongside
    images (unlike GQA_IMAGES_URL's 20GB, see _download_gqa_images).

    Some clusters' outbound proxy filters downloads.cs.stanford.edu --
    confirmed via a real run (slurm-4781715): ProxyError/"Tunnel connection
    failed: 403 Filtered", not a Stanford-side denial (CLEVR's
    direct-HTTPS download through the same proxy, a different host, worked
    fine in that same run). If this function can't get through, download
    sceneGraphs.zip from GQA_SCENEGRAPHS_URL on a machine that isn't behind
    that proxy and scp it directly to
    DATA_DIR/gqa_scenegraphs_raw/sceneGraphs.zip -- this function checks
    for an already-downloaded zip before ever making the request, and
    extracts it regardless of where it came from."""
    import zipfile
    import time
    import requests

    raw_dir = os.path.join(DATA_DIR, "gqa_scenegraphs_raw")
    os.makedirs(raw_dir, exist_ok=True)
    already_extracted = any(f.lower().endswith(".json") for f in os.listdir(raw_dir))
    if not already_extracted:
        zip_path = os.path.join(raw_dir, "sceneGraphs.zip")
        if not os.path.isfile(zip_path):
            t0 = time.time()
            with requests.get(GQA_SCENEGRAPHS_URL, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            print(f"[data] GQA scene graphs download: "
                 f"{os.path.getsize(zip_path)/1e6:.1f}MB in {time.time()-t0:.1f}s", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
    return raw_dir


def _download_gqa_images():
    """Downloads + extracts GQA's official base images once, cached under
    DATA_DIR/gqa_images_raw (~20GB -- see GQA_IMAGES_URL's comment; this is
    a real, one-time cost, see test_datasets.py's precompute-cost estimator
    pattern for how to budget for a source like this before a real training
    run). Deliberately only the base images.zip, not the optional
    spatial/object precomputed-feature files GQA also offers (~53GB more,
    meant for older non-end-to-end VQA models -- irrelevant here, VRDTask
    encodes images itself via the frozen vision encoder).

    Some clusters' outbound proxy filters downloads.cs.stanford.edu --
    confirmed via a real run (slurm-4781715): ProxyError/"Tunnel connection
    failed: 403 Filtered", not a Stanford-side denial (CLEVR's
    direct-HTTPS download through the same proxy, a different host, worked
    fine in that same run). If this function can't get through, download
    images.zip from GQA_IMAGES_URL on a machine that isn't behind that
    proxy and scp it directly to DATA_DIR/gqa_images_raw/images.zip -- this
    function checks for an already-downloaded zip before ever making the
    request, and extracts it regardless of where it came from."""
    import zipfile
    import time
    import requests

    raw_dir = os.path.join(DATA_DIR, "gqa_images_raw")
    os.makedirs(raw_dir, exist_ok=True)
    already_extracted = any(
        f.lower().endswith((".jpg", ".jpeg")) or
        (f == "images" and os.path.isdir(os.path.join(raw_dir, f))
         and os.listdir(os.path.join(raw_dir, f)))
        for f in os.listdir(raw_dir)
    )
    if already_extracted:
        return raw_dir
    zip_path = os.path.join(raw_dir, "images.zip")
    if not os.path.isfile(zip_path):
        print(f"[data] downloading GQA images from {GQA_IMAGES_URL} (~20GB, one-time)...", flush=True)
        t0 = time.time()
        with requests.get(GQA_IMAGES_URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        size_gb = os.path.getsize(zip_path) / 1e9
        elapsed = time.time() - t0
        print(f"[data] GQA images download done: {size_gb:.1f}GB in {elapsed/60:.1f} min "
             f"({size_gb*1000/elapsed:.1f} MB/s) -- this is a ONE-TIME cost, "
             f"see test_datasets.py's precompute-cost estimator to decide "
             f"whether this needs to run as a separate job.", flush=True)
    print("[data] extracting GQA images.zip...", flush=True)
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    print(f"[data] GQA images extraction done in {(time.time()-t0)/60:.1f} min", flush=True)
    return raw_dir


class VRDTask(Task):
    name, encoder, needs_text = "vrd", "image", True
    answer_vocab = None

    @staticmethod
    def _extract_relationship(ex):
        """Pull (subject, predicate, object) from a scene-graph row. Schemas
        vary; try the common shapes and return None if none match. Matches
        GQA's own official schema (objects keyed by id, each with "name"
        and a "relations" list of {"name": <predicate>, "object": <target
        object id>} -- see download()'s comment) as well as the
        flatter/older shapes this function originally handled, so it stays
        usable as a general-purpose extractor regardless of exact source."""
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
        """GQA's official release (see GQA_SCENEGRAPHS_URL/GQA_IMAGES_URL) --
        replaces the earlier Voxel51/GQA-Scene-Graph HF mirror (a FiftyOne
        export whose relationship annotations the generic HF image-folder
        loader never actually ingested -- confirmed root cause, see
        test_datasets.py's KNOWN_PENDING_REPLACEMENT history).

        Scene graph JSON structure per the official page (downloads.cs.
        stanford.edu/nlp/data/gqa -- confirmed field names, not independently
        verified byte-for-byte here since the 73GB+ full release is too
        large to fetch just for inspection): a dict keyed by image_id, each
        value {"width", "height", "objects": {object_id: {"name", "x", "y",
        "w", "h", "attributes": [...], "relations": [{"name": <predicate>,
        "object": <target object_id>}, ...]}}}. train_sceneGraphs.json/
        val_sceneGraphs.json is the standard GQA filename convention (used
        consistently across the GQA literature/tooling); this function
        tries both those names and a couple of case variants defensively,
        raising a clear error naming what it found instead if none match,
        rather than silently misparsing. Images are expected at
        images/<image_id>.jpg under the extracted images.zip (GQA's
        standard flat naming) -- same defensive fallback if not found.

        Converts each object's own "name" + its first "relations" entry
        into the (subject, predicate, object) triple _extract_relationship
        already knows how to consume for the flatter schemas, so downstream
        code (collate/loss_acc) is completely unchanged."""
        from datasets import Dataset
        from PIL import Image as PILImage

        sg_dir = _download_gqa_scene_graphs()
        entries = os.listdir(sg_dir)
        by_lower = {f.lower(): f for f in entries}
        actual_name = by_lower.get("train_scenegraphs.json")
        if actual_name is None:
            raise RuntimeError(
                f"GQA scene graphs: expected 'train_sceneGraphs.json' in {sg_dir}, "
                f"found {entries} instead -- GQA's official release may "
                f"have changed its internal filename; update download()'s lookup.")
        sg_file = os.path.join(sg_dir, actual_name)
        with open(sg_file) as f:
            scene_graphs = json.load(f)

        images_dir = _download_gqa_images()
        # images.zip may extract into a nested "images/" subfolder or flat --
        # tolerate either rather than assume.
        if os.path.isdir(os.path.join(images_dir, "images")):
            images_dir = os.path.join(images_dir, "images")

        rows = []
        for image_id, sg in scene_graphs.items():
            objects = sg.get("objects") or {}
            triple = None
            for obj in objects.values():
                if obj.get("relations"):
                    rel = obj["relations"][0]
                    target = objects.get(rel.get("object"), {})
                    triple = (obj.get("name", ""), rel.get("name", ""), target.get("name", ""))
                    break
            if triple is None or not triple[1]:
                continue
            img_path = os.path.join(images_dir, f"{image_id}.jpg")
            if not os.path.isfile(img_path):
                continue
            img = PILImage.open(img_path).convert("RGB")
            subj, pred, obj_name = triple
            rows.append({"image": img, "subject": subj, "predicate": pred,
                         "object": obj_name})
            if len(rows) >= cap:
                break
        if not rows:
            raise RuntimeError("GQA yielded no usable relationships")
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


# --------------------------------------------------------------------------- #
# OCR task: real PDF pages (rendered via pypdfium2) + real handwriting-line
# images, trained with the composite loss described in the module docstring's
# "OCR encoder pretraining" section. A single Task (unlike text/image, which
# spread across several) since every loss component here applies uniformly
# to every row regardless of source -- there is no separate per-task head to
# select between.
# --------------------------------------------------------------------------- #
def _load_ocr_image_tensor(img, size):
    """PIL image (any mode, any aspect ratio) -> float tensor [3, size, size]
    in [0, 1], resized to fit within size x size preserving aspect ratio and
    padded with white -- not black/zero -- since these are document pages:
    zero-padding would read as black ink to the pixel-derived content mask
    (see _ocr_content_mask), spuriously marking the padding as OCR content."""
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
    return arr.permute(2, 0, 1)


def _ocr_content_mask(images, patch_size, ink_threshold):
    """Per-patch 'does this patch contain OCR-relevant ink' mask, derived
    directly from pixel statistics rather than external word/character
    bounding-box annotations -- works uniformly across every source (real
    PDFs, handwriting scans, anything) with no annotation-schema dependency
    (see module docstring). A patch counts as content if more than
    ``ink_threshold`` of its pixels are darker than a near-white cutoff --
    document pages are overwhelmingly white background with dark ink.
    Returns bool [B, grid_h, grid_w], the same grid OCREncoder's Stage 1
    produces for an image of this (already patch/window-multiple) size."""
    B = images.size(0)
    gray = images.mean(dim=1)                          # [B, H, W], in [0, 1]
    ink = (gray < 0.85).float()
    ink = ink.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    ink_frac = ink.reshape(B, ink.size(1), ink.size(2), -1).mean(dim=-1)
    return ink_frac > ink_threshold


def _ocr_line_bands(content_mask_hw, min_content_patches):
    """content_mask_hw: bool [grid_h, grid_w] for one image. Returns a list
    of (row_start, row_end) exclusive ranges: runs of adjacent patch-grid
    rows that each have >= min_content_patches content patches -- an
    annotation-free stand-in for "text line" (see _ocr_content_mask)."""
    row_has_content = (content_mask_hw.sum(dim=-1) >= min_content_patches).tolist()
    bands, start = [], None
    for i, has in enumerate(row_has_content + [False]):
        if has and start is None:
            start = i
        elif not has and start is not None:
            bands.append((start, i))
            start = None
    return bands


def ocr_content_loss(content_logits, content_mask):
    target = content_mask.reshape(content_logits.shape).float()
    return F.binary_cross_entropy_with_logits(content_logits, target)


def ocr_spatial_coherence_loss(patch_tokens):
    """L2 distance between spatially-consecutive Stage-1 patch tokens
    (row-major flattened, see LocalOCREncoder) -- penalising large jumps
    encourages smooth spatial transitions across the document page."""
    diffs = torch.diff(patch_tokens, dim=1)
    return torch.norm(diffs, p=2, dim=-1).mean()


def ocr_line_coherence_loss(patch_tokens, grid_h, grid_w, content_mask, min_content_patches):
    """Pulls tokens within the same pixel-derived "line band" (see
    _ocr_line_bands) toward that band's mean via cosine similarity."""
    B, D = patch_tokens.size(0), patch_tokens.size(-1)
    grid = patch_tokens.view(B, grid_h, grid_w, D)
    losses = []
    for b in range(B):
        for s, e in _ocr_line_bands(content_mask[b], min_content_patches):
            band_tokens = grid[b, s:e].reshape(-1, D)
            if band_tokens.size(0) < 2:
                continue
            band_mean = band_tokens.mean(dim=0, keepdim=True)
            sim = F.cosine_similarity(band_tokens, band_mean, dim=-1)
            losses.append((1 - sim).mean())
    if not losses:
        return torch.zeros((), device=patch_tokens.device)
    return torch.stack(losses).mean()


def ocr_handwriting_loss(hw_logits, is_handwritten):
    return F.binary_cross_entropy_with_logits(hw_logits, is_handwritten)


class OCRTask(Task):
    name, encoder = "ocr", "ocr"

    def download(self, cap):
        from datasets import load_dataset, Dataset

        hw_frac = HYPERPARAMS["ocr"]["handwritten_fraction"]
        n_hw = int(cap * hw_frac)
        n_typed = cap - n_hw
        rows = []

        # ---- Typed: real PDF pages, rendered ourselves via pypdfium2 (a
        # PDFium binding -- switched from PyMuPDF, which has no prebuilt
        # ppc64le wheel and whose source build needs a mupdf.com download
        # this cluster's proxy blocks) -- see module docstring's "OCR
        # encoder pretraining" section for why this (not a pre-rasterised
        # document-image dataset) is the point. ----
        try:
            import pypdfium2 as pdfium

            stream = load_dataset(PDFA_HF, split="train", streaming=True, cache_dir=HF_CACHE)
            scale = HYPERPARAMS["ocr"]["pdf_render_dpi"] / 72.0
            n = 0
            for ex in stream:
                pdf_bytes = ex.get("pdf")
                if not pdf_bytes:
                    continue
                try:
                    doc = pdfium.PdfDocument(pdf_bytes)
                    img = doc[0].render(scale=scale).to_pil().convert("RGB")
                    doc.close()
                except Exception:
                    continue
                rows.append({"image": img, "is_handwritten": False})
                n += 1
                if n >= n_typed:
                    break
        except Exception as e:
            print(f"[data] ocr: typed/PDF source unavailable ({e})")

        # ---- Handwritten: real handwriting-line images (IAM). ----
        try:
            stream = load_dataset(IAM_LINES_HF, split="train", streaming=True, cache_dir=HF_CACHE)
            n = 0
            for ex in stream:
                img = ex.get("image")
                if img is None:
                    continue
                rows.append({"image": img, "is_handwritten": True})
                n += 1
                if n >= n_hw:
                    break
        except Exception as e:
            print(f"[data] ocr: handwritten source unavailable ({e})")

        if not rows:
            raise RuntimeError("both PDF and handwriting sources failed")
        return Dataset.from_list(rows)

    def collate(self, rows, ctx):
        # Deliberately reads HYPERPARAMS["ocr"]["image_size"] directly rather
        # than ctx["cfg"]["image_size"] -- the latter is the *global* config
        # key shared by the generic image tasks (128px), far too small for a
        # legible document page (see HYPERPARAMS["ocr"]["image_size"]).
        size = HYPERPARAMS["ocr"]["image_size"]
        device = ctx["device"]
        imgs = torch.stack([_load_ocr_image_tensor(r["image"], size) for r in rows])
        hw = torch.tensor([1.0 if r["is_handwritten"] else 0.0 for r in rows])
        return {"images": imgs.to(device), "is_handwritten": hw.to(device)}

    def loss_acc(self, wrapper, batch):
        out = wrapper.forward_pretrain(batch["images"])
        patch_size = wrapper.encoder.local_encoder.patch_size
        ocr_hp = HYPERPARAMS["ocr"]

        content_mask = _ocr_content_mask(batch["images"], patch_size, ocr_hp["content_ink_threshold"])
        loss_content = ocr_content_loss(out["content_logits"], content_mask)
        loss_diversity = koleo_loss(out["refined"].reshape(-1, out["refined"].size(-1)))
        loss_spatial = ocr_spatial_coherence_loss(out["patch_tokens"])
        loss_line = ocr_line_coherence_loss(
            out["patch_tokens"], out["grid_h"], out["grid_w"], content_mask,
            ocr_hp["min_line_content_patches"],
        )
        loss_hw = ocr_handwriting_loss(out["hw_logits"], batch["is_handwritten"])

        hp = wrapper.config
        loss = (
            hp["content_weight"] * loss_content
            + hp["diversity_weight"] * loss_diversity
            + hp["spatial_weight"] * loss_spatial
            + hp["line_weight"] * loss_line
            + hp["handwriting_weight"] * loss_hw
        )
        pred = torch.sigmoid(out["content_logits"]) > 0.5
        acc = (pred == content_mask.reshape(pred.shape)).float().mean().item()
        return loss, acc, batch["images"].size(0)


TEXT_TASKS = [NERTask, POSTask, UDDepRelTask, GAPCorefTask, SquadTask,
              MNLITask, SwagTask, MLMTask]
IMAGE_TASKS = [ReconTask, DepthTask, InstanceTask, VRDTask, SemSegTask]
OCR_TASKS = [OCRTask]
TASKS_BY_KIND = {"text": TEXT_TASKS, "image": IMAGE_TASKS, "ocr": OCR_TASKS}


# --------------------------------------------------------------------------- #
# Build + prepare the active task list for an encoder (skipping failures)
# --------------------------------------------------------------------------- #
def build_tasks(encoder_kind, cfg):
    classes = TASKS_BY_KIND[encoder_kind]
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
    training can resume mid-epoch by skipping already-finished batches.

    With world_size > 1, the full deterministic batch list -- identical on
    every rank, since every rank uses the same seed -- is stride-sliced per
    rank (see dist_utils.shard_indices): each rank trains on a disjoint
    subset of batches, and gradients are averaged back together via manual
    all-reduce (see train_encoder / dist_utils.sync_gradients)."""

    def __init__(self, dataset, batch_size, seed, shuffle=True, rank=0, world_size=1):
        from collections import defaultdict

        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.by_task = defaultdict(list)
        for gi, (ti, _) in enumerate(dataset.index):
            self.by_task[ti].append(gi)
        total = sum((len(v) + batch_size - 1) // batch_size
                    for v in self.by_task.values())
        if world_size > 1:
            self._len = (total - (total % world_size)) // world_size
        else:
            self._len = total

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
        if self.world_size > 1:
            from dist_utils import shard_indices
            batches = shard_indices(batches, self.rank, self.world_size)
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


def make_loader(dataset, tasks, ctx, batch_size, seed, shuffle=True, rank=0, world_size=1):
    sampler = TaskHomogeneousBatchSampler(dataset, batch_size, seed, shuffle,
                                          rank=rank, world_size=world_size)
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
        enc.load_weights(best, map_location=device)
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


# =========================================================================== #
# Vision SSL pretraining (DINOv3-style) -- Phases 0b-1a/b/c/d.
#
# Runs immediately after text_train, before image_tune/image_train (which
# become Phase 0b-2: the existing supervised task-head fine-tuning, now
# initialised from this stage's output instead of a random encoder). Three
# sub-stages, all driven by the shared ``run_ssl_training`` engine below:
#
#   vision_ssl_pretrain  (0b-1a/b) -- DINO + iBOT + KoLeo, constant LR/EMA,
#                                     resumable toward a target iteration
#                                     count; stops for manual review each
#                                     time it reaches that target (see
#                                     HYPERPARAMS["ssl"]["proceed_to_gram_anchoring"]).
#   vision_gram_anchor   (0b-1c)    -- adds the Gram-anchoring loss against a
#                                     frozen "Gram teacher" snapshot taken
#                                     during 0b-1a/b.
#   vision_hires_adapt   (0b-1d)    -- short run at larger (512/768 global,
#                                     112-336 local) crop sizes, Gram teacher
#                                     = the current EMA teacher.
#
# Only the encoder is carried forward past this section (the DINO/iBOT heads,
# the local/global crop-norm distinction, and the Gram teacher are all
# training scaffolding, discarded once Phase 0b-2 begins -- see
# ImagePretrainWrapper, which now always calls the encoder with the default
# crop_type="global").
# =========================================================================== #
SSL_PROGRESS_PATH = os.path.join(PROJECT_DIR, "vision_ssl_progress.json")


def load_ssl_state():
    if os.path.isfile(SSL_PROGRESS_PATH):
        with open(SSL_PROGRESS_PATH) as f:
            return json.load(f)
    return {"gram_candidate_path": None}


def save_ssl_state(state):
    os.makedirs(os.path.dirname(SSL_PROGRESS_PATH), exist_ok=True)
    with open(SSL_PROGRESS_PATH, "w") as f:
        json.dump(state, f, indent=2)


# --------------------------------------------------------------------------- #
# Data: SSL pretraining draws from local, pre-downloaded image pools --
# IMAGENET_HF always, IMAGENET21K_HF too when it materializes successfully
# (see HYPERPARAMS["ssl"]'s pool comment and the module comment above
# IMAGENET21K_HF's definition for why reintroducing a second source no
# longer carries its old risk). Each pool is downloaded ONCE, capped, and
# cached to local disk via _materialize_image_pool -- SSLImageStream itself
# never touches the network. This is what makes rank0_first's "populate a
# shared cache once, every other rank gets a cheap local read" contract
# (see its docstring) actually true here: the old design called
# rank0_first around a function that opened a fresh *live* stream on every
# call regardless of rank, which is not that contract at all -- every one
# of world_size ranks was independently hitting huggingface.co on every
# single one of target_iters batches, for the entire (very long) run.
# --------------------------------------------------------------------------- #
def _materialize_image_pool(hf_id, cap, pool_name, retries=3, backoff_seconds=60.0,
                             config_name=None):
    """Pre-download (bounded to `cap` rows, or the whole split if `cap` is
    None) and cache a single image source to local disk ONCE -- same idiom
    as every Task.download(cap)/build() pool below. A later call (any rank,
    any future resubmission) hits the cached copy directly, no network
    involved. Handles a standard decoded HF ``Image`` feature under an
    "image" key (e.g. IMAGENET_HF), and -- confirmed against IMAGENET21K_HF's
    actual HF Data Studio viewer -- a webdataset-format source whose image
    column is named after its file extension (here: "jpeg", auto-decoded to
    a PIL Image by `datasets`' webdataset loader already, same as the
    "image" case, just under a different key) alongside "__key__"/"__url__"
    housekeeping columns. Also tolerates a raw-bytes column instead (some
    webdataset configs don't auto-decode), for robustness against a
    differently-configured source than the one actually checked.

    `config_name`, if given, is passed as `load_dataset`'s `name=` argument.
    PASS_HF needs this: `datasets` 2.19.2's dataset_module_factory() only
    probes a script-based dataset's raw file (via HfFileSystem, to check for
    a DEFAULT_CONFIG_NAME marker) when no config name is given up front --
    and that probe fails against PASS_HF specifically with a
    UnicodeDecodeError (it reads the hub CDN's gzip-encoded response as raw
    UTF-8 without decompressing it first; confirmed this is a `datasets`-
    internal bug, not a schema/network issue -- both plain `requests` and
    `huggingface_hub.hf_hub_download` fetch and decode the same file
    correctly). Passing name="default" (pass.py defines no BuilderConfig
    subclasses, so "default" is its only/implicit config) sets
    _require_default_config_name=False, which skips that broken probe
    entirely and loads normally. IMAGENET_HF/IMAGENET21K_HF don't hit this
    (modern parquet-format sources, no dataset script involved), so they
    pass config_name=None and are unaffected.

    Retries the whole attempt (not just the single HTTP request
    `huggingface_hub`'s own internal retry already covers) on failure, since
    a sustained proxy/network blip against huggingface.co -- a recurring,
    real failure mode on this cluster -- can easily outlast that internal
    budget (observed: ~23s across 5 attempts) without outlasting a few
    minutes of our own backoff. All prints use flush=True so a later crash
    (e.g. every source failing) doesn't swallow this source's actual
    failure reason in stdout's buffer -- see SSLImageStream's docstring."""
    from datasets import Dataset, load_dataset, load_from_disk
    from PIL import Image as PILImage

    path = os.path.join(DATA_DIR, pool_name)
    if os.path.isdir(path) and os.listdir(path):
        # Silent before this fix -- a source found here (the common case for
        # any pool prepared ahead of time, e.g. via a manual precompute
        # transfer) printed nothing at all, while a source that needed a
        # live materialization printed several lines. On a run mixing both
        # (some sources pre-cached, some not, see SSLImageStream's own
        # per-source loop), the log made it look like every source was
        # hitting the network, even though only the ones that actually
        # needed to were -- confirmed as the cause of exactly this
        # confusion on a real run (enc_pre_slurm-4795027): imagenet1k had
        # loaded silently from its pre-extracted local pool, but with
        # nothing printed for it, the log's first visible SSL-data line was
        # imagenet21k's (genuinely, by design) live download, making it
        # look like nothing local was being used at all.
        print(f"[ssl-data] {pool_name}: found existing local pool -> {path} "
              f"(skipping {hf_id})", flush=True)
        return load_from_disk(path)

    last_err = None
    for attempt in range(retries):
        try:
            print(f"[ssl-data] {pool_name}: no local pool yet -- materializing from "
                 f"{hf_id} now (attempt {attempt + 1}/{retries}, up to "
                 f"{cap if cap is not None else 'the entire split'} images; this can "
                 f"take a while the first time).", flush=True)
            ds = load_dataset(hf_id, name=config_name, split="train", streaming=True,
                              cache_dir=HF_CACHE)
            rows = []
            for ex in ds:
                img = ex.get("image")
                if img is None:
                    for key in ("jpeg", "jpg", "png"):
                        raw = ex.get(key)
                        if raw is None:
                            continue
                        if isinstance(raw, PILImage.Image):
                            img = raw
                        elif isinstance(raw, (bytes, bytearray)):
                            from io import BytesIO
                            img = PILImage.open(BytesIO(raw))
                        else:
                            img = PILImage.open(raw)   # file path or file-like object
                        break
                if img is None:
                    continue
                rows.append({"image": img})
                if cap is not None and len(rows) >= cap:
                    break
            if not rows:
                raise RuntimeError(f"{hf_id} yielded no usable images -- check its column "
                                   f"schema against _materialize_image_pool's decode logic")
            os.makedirs(DATA_DIR, exist_ok=True)
            Dataset.from_list(rows).save_to_disk(path)
            print(f"[ssl-data] {pool_name}: cached {len(rows)} images -> {path}", flush=True)
            return load_from_disk(path)
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                print(f"[ssl-data] {pool_name}: attempt {attempt + 1}/{retries} failed "
                     f"({e}); retrying in {backoff_seconds:.0f}s...", flush=True)
                time.sleep(backoff_seconds)
    raise last_err


def _materialize_inaturalist_pool(cap, pool_name, retries=3, backoff_seconds=60.0):
    """iNaturalist-2021-mini counterpart to _materialize_image_pool -- same
    cache-once contract and output shape (a local datasets.Dataset with an
    "image" column), but reads from INATURALIST_ROOT's already-extracted
    local directory tree instead of a HuggingFace load_dataset() stream (no
    working HF mirror exists -- see INATURALIST_ROOT's definition). Species
    folders are only used to enumerate image files; the species label
    itself is discarded, same as every other source's labels here -- this
    is unlabeled SSL pretraining data.

    The directory-existence check is deliberately outside the retry loop
    (retrying a fundamentally-missing/misconfigured directory wastes
    minutes for no benefit -- that's a setup problem, not a blip); the
    directory walk + image open + save is inside it, since this session
    has directly observed real transient slowness/hiccups on this same
    mounted filesystem. All prints use flush=True -- see
    _materialize_image_pool's identical note."""
    from datasets import Dataset, load_from_disk
    from PIL import Image as PILImage

    path = os.path.join(DATA_DIR, pool_name)
    if os.path.isdir(path) and os.listdir(path):
        # See _materialize_image_pool's identical fix -- same silent-success
        # gap that made a mixed cached/live-download run's log look like
        # everything was going online.
        print(f"[ssl-data] {pool_name}: found existing local pool -> {path} "
              f"(skipping {INATURALIST_ROOT})", flush=True)
        return load_from_disk(path)

    if not os.path.isdir(INATURALIST_ROOT):
        raise RuntimeError(
            f"INATURALIST_ROOT ({INATURALIST_ROOT}) doesn't exist -- expected the "
            f"already-extracted train_mini/ directory from the AWS S3 download "
            f"(see INATURALIST_ROOT's module comment)."
        )

    last_err = None
    for attempt in range(retries):
        try:
            print(f"[ssl-data] {pool_name}: no local pool yet -- materializing from "
                 f"{INATURALIST_ROOT} now (attempt {attempt + 1}/{retries}, up to "
                 f"{cap if cap is not None else 'every'} images found; this can take a "
                 f"while the first time).", flush=True)
            image_paths = []
            for species_dir in os.scandir(INATURALIST_ROOT):
                if not species_dir.is_dir():
                    continue
                for entry in os.scandir(species_dir.path):
                    if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_paths.append(entry.path)
            if not image_paths:
                raise RuntimeError(f"no image files found under {INATURALIST_ROOT}")

            random.shuffle(image_paths)   # so capping doesn't just take the first few species alphabetically
            if cap is not None:
                image_paths = image_paths[:cap]

            rows = [{"image": PILImage.open(p)} for p in image_paths]
            os.makedirs(DATA_DIR, exist_ok=True)
            Dataset.from_list(rows).save_to_disk(path)
            print(f"[ssl-data] {pool_name}: cached {len(rows)} images -> {path}", flush=True)
            return load_from_disk(path)
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                print(f"[ssl-data] {pool_name}: attempt {attempt + 1}/{retries} failed "
                     f"({e}); retrying in {backoff_seconds:.0f}s...", flush=True)
                time.sleep(backoff_seconds)
    raise last_err


class SSLImageStream:
    """Cycles batches from local image pools (IMAGENET_HF, IMAGENET21K_HF,
    PASS_HF -- all three via HuggingFace, see _materialize_image_pool; and
    INATURALIST_ROOT, a local directory tree via _materialize_inaturalist_pool,
    since no working HF mirror exists for it), mixed per
    HYPERPARAMS["ssl"]["ssl_source_fractions"]. No network access happens
    after __init__ returns (INATURALIST_ROOT never needed any to begin with
    -- it's already-downloaded local data).

    Every source is best-effort: a source that fails to materialize (bad
    id, gated/access error, schema mismatch, missing local directory) is
    dropped with a warning rather than crashing the run, and the
    configured fractions are renormalized over whichever sources actually
    succeeded -- see each source's own definition (IMAGENET21K_HF, PASS_HF,
    INATURALIST_ROOT) for what's actually been verified vs. not. Only
    raises if literally every source fails."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._iters = {}
        self.fraction = {}
        self.counts = {}

        sources = [
            ("imagenet1k", IMAGENET_HF,
             lambda: _materialize_image_pool(IMAGENET_HF, cfg["imagenet1k_pool_cap"], "ssl_pool_imagenet1k")),
            ("imagenet21k", IMAGENET21K_HF,
             lambda: _materialize_image_pool(IMAGENET21K_HF, cfg["imagenet21k_pool_cap"], "ssl_pool_imagenet21k")),
            ("pass", PASS_HF,
             lambda: _materialize_image_pool(PASS_HF, cfg["pass_pool_cap"], "ssl_pool_pass",
                                              config_name="default")),
            ("inaturalist", INATURALIST_ROOT,
             lambda: _materialize_inaturalist_pool(cfg["inaturalist_pool_cap"], "ssl_pool_inaturalist")),
        ]
        configured = cfg["ssl_source_fractions"]
        succeeded = []
        for label, source_id, materialize in sources:
            try:
                ds = materialize()
                self._iters[source_id] = self._cycle_local(ds)
                self.counts[source_id] = 0
                succeeded.append((label, source_id))
            except Exception as e:
                print(f"[ssl-data] WARNING: could not materialize {label} ({source_id}) ({e}); "
                     f"continuing without this source.", flush=True)

        if not succeeded:
            raise RuntimeError(
                "SSLImageStream: every configured image source failed to materialize -- "
                "check network/proxy status and each source's gating requirements."
            )

        total_weight = sum(configured[label] for label, _ in succeeded)
        for label, source_id in succeeded:
            self.fraction[source_id] = configured[label] / total_weight
        print(f"[ssl-data] active sources: " +
             ", ".join(f"{label}={self.fraction[source_id]:.2f}" for label, source_id in succeeded),
             flush=True)

    @staticmethod
    def _cycle_local(ds):
        """Repeatedly reshuffle and iterate a fully-local datasets.Dataset --
        no network access at all, unlike the old design's live re-streaming
        every pass."""
        n = len(ds)
        while True:
            order = list(range(n))
            random.shuffle(order)
            for i in order:
                yield ds[i]["image"]

    def next_batch(self, batch_size):
        hf_ids = list(self._iters.keys())
        weights = [self.fraction[h] for h in hf_ids]
        batch = []
        for _ in range(batch_size):
            hf_id = random.choices(hf_ids, weights=weights, k=1)[0]
            batch.append(next(self._iters[hf_id]))
            self.counts[hf_id] += 1
        return batch

    def summary(self):
        """Cumulative image count drawn so far this run, per source."""
        total = sum(self.counts.values())
        return dict(self.counts), total


def _random_resized_crop(img, out_size, scale_range):
    """Random-area/aspect-ratio crop then resize to a square, matching
    torchvision's RandomResizedCrop (not a listed dependency here, so
    reimplemented directly on PIL)."""
    img = img.convert("RGB")
    W, H = img.size
    area = W * H
    for _ in range(10):
        target_area = random.uniform(*scale_range) * area
        aspect = random.uniform(3 / 4, 4 / 3)
        w = int(round((target_area * aspect) ** 0.5))
        h = int(round((target_area / aspect) ** 0.5))
        if 0 < w <= W and 0 < h <= H:
            x0 = random.randint(0, W - w)
            y0 = random.randint(0, H - h)
            return img.crop((x0, y0, x0 + w, y0 + h)).resize((out_size, out_size))
    m = min(W, H)
    x0, y0 = (W - m) // 2, (H - m) // 2
    return img.crop((x0, y0, x0 + m, y0 + m)).resize((out_size, out_size))


def make_multicrop_batch(images, cfg, global_size_choices=None, local_size_choices=None,
                         device="cpu"):
    """images: list of PIL images (one training batch). Returns
    (global_crops, local_crops): lists of [B, 3, size, size] tensors, one
    entry per crop slot (``n_global_crops`` + ``n_local_crops``). Each crop
    slot's size is independently sampled from *_size_choices (a single-size
    list for phases 0b-1a/b/c; the mixed-resolution candidate lists from
    HYPERPARAMS["ssl"] for the 0b-1d high-res adaptation phase)."""
    g_choices = global_size_choices or [cfg["global_crop_size"]]
    l_choices = local_size_choices or [cfg["local_crop_size"]]

    global_crops = []
    for _ in range(cfg["n_global_crops"]):
        size = random.choice(g_choices)
        crops = [_random_resized_crop(im, size, cfg["global_crop_scale"]) for im in images]
        global_crops.append(torch.stack([_load_image_tensor(c, size) for c in crops]).to(device))

    local_crops = []
    for _ in range(cfg["n_local_crops"]):
        size = random.choice(l_choices)
        crops = [_random_resized_crop(im, size, cfg["local_crop_scale"]) for im in images]
        local_crops.append(torch.stack([_load_image_tensor(c, size) for c in crops]).to(device))

    return global_crops, local_crops


# --------------------------------------------------------------------------- #
# Heads: identical MLP + L2-normalize + prototype-projection architecture
# for both DINO (CLS token) and iBOT (patch tokens) -- applied per-token, so
# the same class serves both (broadcasts over any leading dims).
# --------------------------------------------------------------------------- #
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, bottleneck_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class VisionSSLWrapper(nn.Module):
    """Student: encoder + DINO head + iBOT head, bundled so the optimizer and
    checkpointing can treat 'the student' as one module. The teacher is a
    separate EMATeacher, never optimised directly."""

    def __init__(self, d_model, ssl_cfg):
        super().__init__()
        from models.Encoders.VisionEncoder import createVisionEncoder

        self.encoder = createVisionEncoder(dim=d_model)
        self.dino_head = ProjectionHead(
            d_model, ssl_cfg["head_hidden_dim"], ssl_cfg["head_bottleneck_dim"],
            ssl_cfg["dino_out_dim"],
        )
        self.ibot_head = ProjectionHead(
            d_model, ssl_cfg["head_hidden_dim"], ssl_cfg["head_bottleneck_dim"],
            ssl_cfg["ibot_out_dim"],
        )


class EMATeacher:
    """Frozen, EMA-updated copy of (encoder, dino_head, ibot_head). Never
    receives gradients directly; call update() after every optimizer step."""

    def __init__(self, encoder, dino_head, ibot_head, momentum):
        import copy

        self.encoder = copy.deepcopy(encoder)
        self.dino_head = copy.deepcopy(dino_head)
        self.ibot_head = copy.deepcopy(ibot_head)
        for m in (self.encoder, self.dino_head, self.ibot_head):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        self.momentum = momentum

    @torch.no_grad()
    def update(self, encoder, dino_head, ibot_head):
        m = self.momentum
        for teacher_m, student_m in (
            (self.encoder, encoder), (self.dino_head, dino_head), (self.ibot_head, ibot_head),
        ):
            for tp, sp in zip(teacher_m.parameters(), student_m.parameters()):
                tp.mul_(m).add_(sp.detach(), alpha=1 - m)

    def to(self, device):
        self.encoder.to(device)
        self.dino_head.to(device)
        self.ibot_head.to(device)
        return self

    def state_dicts(self):
        return {
            "encoder": self.encoder.state_dict(),
            "dino_head": self.dino_head.state_dict(),
            "ibot_head": self.ibot_head.state_dict(),
        }

    def load_state_dicts(self, blob):
        self.encoder.load_state_dict(blob["encoder"])
        self.dino_head.load_state_dict(blob["dino_head"])
        self.ibot_head.load_state_dict(blob["ibot_head"])


class GramTeacher:
    """Frozen encoder-only snapshot used for Gram anchoring (only patch
    features are needed). Refreshed periodically from the live EMA teacher
    during Phase 0b-1c (see gram_teacher_update_every)."""

    def __init__(self, encoder_state_dict, d_model, device):
        from models.Encoders.VisionEncoder import createVisionEncoder

        self.encoder = createVisionEncoder(dim=d_model).to(device)
        self.encoder.load_state_dict(encoder_state_dict)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def refresh_from(self, teacher):
        self.encoder.load_state_dict(teacher.encoder.state_dict())


# --------------------------------------------------------------------------- #
# Losses: Sinkhorn-Knopp centering (SwAV-style, replacing DINO's running-mean
# centering per DINOv3), DINO (image-level), iBOT (masked patch-level),
# KoLeo (batch-spread regulariser), Gram (patch-Gram-matrix anchoring).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sinkhorn_knopp(logits, n_iters=3, epsilon=0.05):
    shape = logits.shape
    K = shape[-1]
    Q = torch.exp(logits.reshape(-1, K) / epsilon).t()   # [K, B]
    B = Q.shape[1]
    Q /= Q.sum()
    for _ in range(n_iters):
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= K
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= B
    Q *= B
    return Q.t().reshape(shape)


def dino_loss(student_logits, teacher_logits, cfg):
    """student_logits: list of [B, K] (every crop); teacher_logits: list of
    [B, K] (global crops only). Cross-entropy between the teacher's SK-soft
    assignment and the student's softmax, averaged over every (teacher,
    student) pair except a crop against itself."""
    total, n_terms = 0.0, 0
    teacher_probs = [sinkhorn_knopp(t, cfg["sinkhorn_iters"], cfg["sinkhorn_epsilon"])
                     for t in teacher_logits]
    for ti, tp in enumerate(teacher_probs):
        for si, sl in enumerate(student_logits):
            if ti == si:
                continue
            log_p = F.log_softmax(sl / cfg["student_temp"], dim=-1)
            total = total + (-(tp.detach() * log_p).sum(dim=-1)).mean()
            n_terms += 1
    return total / max(1, n_terms)


def dino_sharpness_loss(student_logits, cfg):
    """Mean per-sample entropy of softmax(logits / student_temp), averaged
    over every crop -- a direct penalty on near-uniform individual-sample
    student distributions, the specific failure mode dino_loss's own
    cross-entropy and koleo_loss (both already active) can't catch. A real
    run (enc_pre_slurm-4795027) showed `dino` permanently pinned at
    log(dino_out_dim)=9.011 -- the exact value a PERFECTLY UNIFORM softmax
    produces against ANY teacher target, since -(tp * log_p).sum() reduces
    to -log(1/K) regardless of tp once log_p itself is uniform -- from
    roughly iteration 25,000 onward, still pinned at iteration 89,100 (out
    of a 200,000-iteration target) when last checked, while koleo (computed
    on this same
    student_cls_logits tensor, see ssl_train_step) stayed healthy the whole
    time. That isn't a contradiction: koleo only measures whether DIFFERENT
    samples' logit vectors point in different directions (inter-sample
    diversity); it says nothing about whether ONE sample's own K=8192
    logits have enough spread ACROSS THEMSELVES to produce a non-uniform
    softmax (intra-sample sharpness) -- the two are independent, and only
    the former was previously instrumented or regularized. ibot_loss (same
    ProjectionHead architecture, same cfg) never showed this plateau,
    consistent with its many masked-patch predictions per image giving it
    much richer gradient signal than dino's single global CLS prediction --
    this term is scoped to the CLS/dino path specifically since nothing
    points at ibot needing it too. Minimizing entropy (not maximizing) --
    the two existing counter-pressures against this overshooting into a
    DIFFERENT degenerate "always confidently predict one fixed class"
    collapse are dino_loss's own cross-entropy against a per-sample-varying
    Sinkhorn-balanced teacher target (incompatible with one fixed answer
    for every input) and koleo_loss's inter-sample repulsion (already
    active, unaffected by this term)."""
    total, n = 0.0, 0
    for sl in student_logits:
        p = F.softmax(sl / cfg["student_temp"], dim=-1)
        entropy = -(p * (p + 1e-8).log()).sum(dim=-1).mean()
        total = total + entropy
        n += 1
    return total / max(1, n)


def ibot_loss(student_patch_logits, teacher_patch_logits, patch_masks, cfg):
    """Lists of [B, N, K] (one per global crop) + matching [B, N] bool masks
    (True = masked). Only masked positions contribute."""
    total, n_terms = 0.0, 0
    for s, t, m in zip(student_patch_logits, teacher_patch_logits, patch_masks):
        mask_flat = m.reshape(-1)
        if mask_flat.sum() == 0:
            continue
        sf = s.reshape(-1, s.size(-1))[mask_flat]
        tf = t.reshape(-1, t.size(-1))[mask_flat]
        tp = sinkhorn_knopp(tf, cfg["sinkhorn_iters"], cfg["sinkhorn_epsilon"])
        log_p = F.log_softmax(sf / cfg["student_temp"], dim=-1)
        total = total + (-(tp.detach() * log_p).sum(dim=-1)).mean()
        n_terms += 1
    if n_terms == 0:
        return torch.zeros((), device=student_patch_logits[0].device)
    return total / n_terms


def koleo_loss(embeddings, chunk_size=16, eps=1e-8):
    """Sablayrolles et al. 2018: encourages embeddings within a (small)
    batch to spread out uniformly via -log(distance to nearest neighbour).
    Computed in chunks of ``chunk_size`` (DINOv3's "small batches of 16")."""
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
        dots.fill_diagonal_(-2.0)          # exclude self as nearest neighbour
        nn_idx = dots.argmax(dim=1)
        dist = (chunk - chunk[nn_idx]).pow(2).sum(dim=-1).add(eps).sqrt()
        losses.append(-torch.log(dist).mean())
    if not losses:
        return torch.zeros((), device=x.device)
    return torch.stack(losses).mean()


def gram_loss(student_patch_feats, teacher_patch_feats):
    """||X_S X_S^T - X_G X_G^T||^2 (mean-reduced), L2-normalised patch
    features. Both [B, N, D] for one global crop."""
    xs = F.normalize(student_patch_feats, dim=-1)
    xg = F.normalize(teacher_patch_feats, dim=-1)
    gs = torch.bmm(xs, xs.transpose(1, 2))
    gg = torch.bmm(xg, xg.transpose(1, 2))
    return ((gs - gg) ** 2).mean()


def visualize_patch_similarity(teacher, images, device, out_path):
    """Cosine similarity between a roughly-central patch and every other
    patch, one tile per sample image, saved as a grayscale PNG strip -- the
    manual diagnostic the phase spec calls for ("if maps are clean ->
    continue 0b-1b, if degrading -> proceed to 0b-1c"). No new plotting
    dependency: rendered directly with PIL, matching this file's existing
    pillow/numpy-only image handling."""
    from PIL import Image as PILImage
    import numpy as np

    was_training = teacher.encoder.training
    teacher.encoder.eval()
    tiles = []
    with torch.no_grad():
        for img in images[:4]:
            size = 224
            crop = _random_resized_crop(img, size, (0.8, 1.0))
            tensor = _load_image_tensor(crop, size).unsqueeze(0).to(device)
            tokens = teacher.encoder(tensor, crop_type="global", return_all_tokens=True)
            patches = F.normalize(tokens[0, 1 + NUM_REGISTERS:], dim=-1)   # [N, D]
            n = patches.size(0)
            g = int(round(n ** 0.5))
            if g * g != n:
                continue
            ref = patches[n // 2 + g // 2]
            sim = (patches @ ref).reshape(g, g)
            sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
            tile = (sim.cpu().numpy() * 255).astype("uint8")
            tiles.append(np.array(PILImage.fromarray(tile).resize((128, 128), PILImage.NEAREST)))
    teacher.encoder.train(was_training)
    if not tiles:
        return
    strip = np.concatenate(tiles, axis=1)
    PILImage.fromarray(strip).save(out_path)
    print(f"[ssl] wrote patch-similarity visualization to {out_path}")


# --------------------------------------------------------------------------- #
# Greppable checkpointing for the SSL phases (same weight_path/find_weights
# naming convention as the rest of the file; the reusable "*_encoder"
# artifact is always the TEACHER's encoder -- the EMA teacher is the "good"
# model downstream stages should build on, not the noisier, more frequently
# updated, masked student).
# --------------------------------------------------------------------------- #
def save_ssl_checkpoint(prefix, student, teacher, optimizer, iteration, best_val,
                        val_loss=None):
    """Persist the teacher's encoder (via its own store_weights, see
    models/Encoders/VisionEncoder.py) and a full resumable wrapper
    checkpoint under 'latest', always, and also under 'best' when val_loss
    improves on best_val -- see save_checkpoint for the full rationale
    (identical scheme; SSL phases never actually pass val_loss today, since
    they have no validation-loss metric, so only 'latest' ever gets
    written in practice -- entirely correct, not a special case)."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    resume_blob = {
        "student_state": student.state_dict(),
        "teacher_state": teacher.state_dicts(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
        "best_val": best_val,
        "rng": torch.get_rng_state(),
    }

    wrp, enc = f"{prefix}_wrapper", f"{prefix}_encoder"
    tags = ["latest"]
    if val_loss is not None and val_loss <= best_val:
        tags.append("best")

    for tag in tags:
        wp = weight_path(wrp, tag, 0, iteration, val_loss)
        ep = weight_path(enc, tag, 0, iteration, val_loss)
        _atomic_torch_save(resume_blob, wp)
        teacher.encoder.store_weights(WEIGHTS_DIR, os.path.basename(ep))
        _prune_other_tagged_files(wrp, tag, wp)
        _prune_other_tagged_files(enc, tag, ep)


# --------------------------------------------------------------------------- #
# Shared training engine for Phases 0b-1a/b/c/d -- differences between
# phases are just which knobs get passed in (see the three phase-runner
# functions at the bottom of this section).
# --------------------------------------------------------------------------- #
def ssl_train_step(student, teacher, images, cfg, device, gram_teacher=None):
    global_crops, local_crops = make_multicrop_batch(images, cfg, device=device)

    with torch.no_grad():
        teacher.encoder.eval()
        teacher.dino_head.eval()
        teacher.ibot_head.eval()
        teacher_cls_logits, teacher_patch_logits = [], []
        for gc in global_crops:
            tokens = teacher.encoder(gc, crop_type="global", return_all_tokens=True)
            teacher_cls_logits.append(teacher.dino_head(tokens[:, 0]))
            teacher_patch_logits.append(teacher.ibot_head(tokens[:, 1 + NUM_REGISTERS:]))

    student_cls_logits, student_patch_logits, patch_masks = [], [], []
    for gc in global_crops:
        tokens, patch_mask = student.encoder(gc, mask=True, crop_type="global",
                                             return_all_tokens=True)
        student_cls_logits.append(student.dino_head(tokens[:, 0]))
        student_patch_logits.append(student.ibot_head(tokens[:, 1 + NUM_REGISTERS:]))
        patch_masks.append(patch_mask)
    for lc in local_crops:
        tokens = student.encoder(lc, mask=False, crop_type="local", return_all_tokens=True)
        student_cls_logits.append(student.dino_head(tokens[:, 0]))

    l_dino = dino_loss(student_cls_logits, teacher_cls_logits, cfg)
    l_ibot = ibot_loss(student_patch_logits, teacher_patch_logits, patch_masks, cfg)
    l_koleo = koleo_loss(student_cls_logits[0])
    # dino_sharpness_loss: see its own docstring -- the direct fix for `dino`
    # permanently pinned at log(dino_out_dim) (enc_pre_slurm-4795027), a
    # failure mode koleo_loss above can't catch since it only measures
    # inter-sample (not intra-sample) diversity. Scoped to the dino/CLS path
    # only -- ibot never showed this plateau, so no reason to add it there.
    l_sharpness = dino_sharpness_loss(student_cls_logits, cfg)
    loss = (l_dino + l_ibot + cfg["koleo_weight"] * l_koleo
            + cfg["sharpness_weight"] * l_sharpness)
    metrics = {"dino": l_dino.item(), "ibot": l_ibot.item(), "koleo": l_koleo.item(),
              "sharpness": l_sharpness.item()}

    if gram_teacher is not None:
        gram_terms = []
        for gc in global_crops:
            with torch.no_grad():
                g_tokens = gram_teacher.encoder(gc, crop_type="global", return_all_tokens=True)
                g_patch = g_tokens[:, 1 + NUM_REGISTERS:]
            s_tokens = student.encoder(gc, mask=False, crop_type="global", return_all_tokens=True)
            s_patch = s_tokens[:, 1 + NUM_REGISTERS:]
            gram_terms.append(gram_loss(s_patch, g_patch))
        l_gram = torch.stack(gram_terms).mean()
        loss = loss + cfg["gram_weight"] * l_gram
        metrics["gram"] = l_gram.item()

    return loss, metrics


def run_ssl_training(device, phase_name, target_iters, use_gram=False,
                     global_size_choices=None, local_size_choices=None,
                     gram_update_every=None, rank=0, world_size=1):
    from dist_utils import (barrier, broadcast_parameters, is_main_process,
                            rank0_first, sync_gradients)

    is_distributed = world_size > 1
    main = is_main_process(rank)
    ssl_cfg = HYPERPARAMS["ssl"]
    d_model = HYPERPARAMS["global"]["d_model"]

    # SSLImageStream's __init__ materializes each image source to a local,
    # shared on-disk pool on first use (see _materialize_image_pool) --
    # rank0_first avoids every rank racing to populate that cache
    # concurrently (see its docstring); every rank after the first gets a
    # cheap local load_from_disk instead of its own redundant download.
    # Unlike train_encoder's epoch-bounded dataset, there's no exact
    # per-rank shard of the pool to compute: each rank's own independent
    # process-level randomness already makes their draws differ, which is
    # all data parallelism needs here (some overlap across ranks is
    # harmless for continuous SSL pretraining, unlike the exact
    # non-overlapping epoch coverage train_encoder needs).
    stream = rank0_first(rank, world_size, SSLImageStream, ssl_cfg)

    student = VisionSSLWrapper(d_model, ssl_cfg).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=ssl_cfg["lr"], weight_decay=ssl_cfg["weight_decay"],
    )
    teacher = EMATeacher(student.encoder, student.dino_head, student.ibot_head,
                         ssl_cfg["teacher_momentum"]).to(device)

    def lr_scale(step):
        w = ssl_cfg["warmup_steps"]
        return (step + 1) / w if w > 0 and step < w else 1.0

    # Constant LR after warmup (no decay schedule at all): DINOv3 drops all
    # parameter scheduling for the SSL phases specifically so training can be
    # extended indefinitely without knowing the optimization horizon a priori.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    # 'latest' is continuously updated by save_ssl_checkpoint (see its
    # docstring), so resuming is just loading it back. Every rank reads the
    # same shared-storage file independently (a safe, read-only op), so
    # resuming -- or initialising from a previous phase's checkpoint --
    # needs no broadcast; only a genuinely fresh start does, since every
    # rank's random init otherwise differs.
    iteration, best_val = 0, float("inf")
    resume, blob = _find_valid_resume(f"{phase_name}_wrapper", device)
    if resume:
        student.load_state_dict(blob["student_state"])
        teacher.load_state_dicts(blob["teacher_state"])
        optimizer.load_state_dict(blob["optimizer_state"])
        iteration, best_val = blob["iteration"], blob["best_val"]
        for _ in range(iteration):
            scheduler.step()
        if main:
            print(f"[resume] {phase_name}: {os.path.basename(resume)} -> iteration {iteration}")
    else:
        prev_phase = {"vision_gram_anchor": "vision_ssl_pretrain",
                      "vision_hires_adapt": "vision_gram_anchor"}.get(phase_name)
        prev_loaded = False
        if prev_phase:
            prev_ckpt = find_weights(f"{prev_phase}_wrapper", "latest")
            if prev_ckpt:
                prev_blob = torch.load(prev_ckpt, map_location=device)
                student.load_state_dict(prev_blob["student_state"])
                teacher.load_state_dicts(prev_blob["teacher_state"])
                prev_loaded = True
                if main:
                    print(f"[ssl] {phase_name} initialised from '{prev_phase}' "
                          f"checkpoint {os.path.basename(prev_ckpt)}")
            elif main:
                print(f"[ssl] WARNING: no '{prev_phase}' checkpoint found; "
                      f"{phase_name} starting from a random-init student/teacher")
        if not prev_loaded and is_distributed:
            broadcast_parameters(student.parameters(), src=0)
            broadcast_parameters(teacher.encoder.parameters(), src=0)
            broadcast_parameters(teacher.dino_head.parameters(), src=0)
            broadcast_parameters(teacher.ibot_head.parameters(), src=0)
    barrier(is_distributed)

    gram_teacher = None
    if use_gram:
        if phase_name == "vision_gram_anchor":
            state = load_ssl_state()
            candidate = state.get("gram_candidate_path")
            if candidate and os.path.isfile(candidate):
                gram_teacher = GramTeacher(torch.load(candidate, map_location=device),
                                          d_model, device)
                if main:
                    print(f"[ssl] Gram teacher loaded from {os.path.basename(candidate)}")
        if gram_teacher is None:
            gram_teacher = GramTeacher(teacher.encoder.state_dict(), d_model, device)
            if main:
                print("[ssl] Gram teacher seeded from the current EMA teacher")

    ckpt_every = ssl_cfg["checkpoint_every"]
    eval_every = ssl_cfg["eval_every"]

    student.train()
    while iteration < target_iters:
        images = stream.next_batch(ssl_cfg["batch_size"])
        loss, metrics = ssl_train_step(student, teacher, images, ssl_cfg, device,
                                       gram_teacher=gram_teacher)
        optimizer.zero_grad()
        loss.backward()
        sync_gradients(student.parameters(), world_size)
        optimizer.step()
        scheduler.step()
        teacher.update(student.encoder, student.dino_head, student.ibot_head)
        iteration += 1

        if gram_teacher is not None and gram_update_every and iteration % gram_update_every == 0:
            gram_teacher.refresh_from(teacher)

        if main and iteration % 50 == 0:
            msg = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  [{phase_name}] iter {iteration}/{target_iters} {msg} "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
            wandb_log({f"{phase_name}/{k}": v for k, v in metrics.items()})

        if main and iteration % ckpt_every == 0:
            save_ssl_checkpoint(phase_name, student, teacher, optimizer, iteration, best_val)

        # Save a one-time Gram-teacher candidate once inside the configured
        # window (only meaningful during the initial pretraining phase).
        if main and phase_name == "vision_ssl_pretrain":
            state = load_ssl_state()
            if (ssl_cfg["gram_candidate_min_iter"] <= iteration <= ssl_cfg["gram_candidate_max_iter"]
                    and not state.get("gram_candidate_path")):
                cand_path = weight_path("vision_ssl_gram_candidate", "snapshot", 0, iteration)
                teacher.encoder.store_weights(WEIGHTS_DIR, os.path.basename(cand_path))
                state["gram_candidate_path"] = cand_path
                save_ssl_state(state)
                print(f"[ssl] saved Gram-teacher candidate at iteration {iteration} -> {cand_path}")

        if main and (iteration % eval_every == 0 or iteration == target_iters):
            out_path = os.path.join(DATA_DIR, f"{phase_name}_patch_similarity_iter{iteration}.png")
            visualize_patch_similarity(teacher, images, device, out_path)
            counts, total = stream.summary()
            print(f"[ssl-data] {phase_name}: {total} images drawn so far this run")
            for hf_id, n in counts.items():
                pct = 100.0 * n / max(1, total)
                print(f"[ssl-data]   {hf_id}: {n} ({pct:.1f}%)")

    barrier(is_distributed)
    if not main:
        return
    save_ssl_checkpoint(phase_name, student, teacher, optimizer, iteration, best_val)
    print(f"[done] {phase_name} reached {iteration} iterations.")


def run_ssl_pretrain_phase(device, rank=0, world_size=1):
    """Phases 0b-1a/b. Returns True if the pipeline should mark this stage
    complete and auto-advance to Gram anchoring, False if it should stop for
    manual review (see HYPERPARAMS["ssl"]["proceed_to_gram_anchoring"]).
    'latest'/'best' are maintained continuously by save_ssl_checkpoint (see
    its docstring), so unlike the old 'step'-history scheme there is no
    finalize step needed here -- the print below only fires when actually
    proceeding, since printing it mid-review (before the phase is done)
    would be premature."""
    from dist_utils import is_main_process

    ssl_cfg = HYPERPARAMS["ssl"]
    run_ssl_training(device, "vision_ssl_pretrain", ssl_cfg["target_iters"],
                     rank=rank, world_size=world_size)
    proceed = ssl_cfg["proceed_to_gram_anchoring"]
    if proceed and is_main_process(rank):
        _drop_wrapper_files("vision_ssl_pretrain_wrapper")
        best = find_weights("vision_ssl_pretrain_encoder", "best")
        latest = find_weights("vision_ssl_pretrain_encoder", "latest")
        print(f"[optimal weights] vision_ssl_pretrain_encoder -> {best or latest}")
        print(f"[latest weights] vision_ssl_pretrain_encoder -> {latest}")
    return proceed


def run_gram_anchoring_phase(device, rank=0, world_size=1):
    """Phase 0b-1c. Always runs to completion in one call (no review pause)."""
    from dist_utils import is_main_process

    ssl_cfg = HYPERPARAMS["ssl"]
    run_ssl_training(
        device, "vision_gram_anchor", ssl_cfg["gram_anchor_iters"],
        use_gram=True, gram_update_every=ssl_cfg["gram_teacher_update_every"],
        rank=rank, world_size=world_size,
    )
    if not is_main_process(rank):
        return
    _drop_wrapper_files("vision_gram_anchor_wrapper")
    best = find_weights("vision_gram_anchor_encoder", "best")
    latest = find_weights("vision_gram_anchor_encoder", "latest")
    print(f"[optimal weights] vision_gram_anchor_encoder -> {best or latest}")
    print(f"[latest weights] vision_gram_anchor_encoder -> {latest}")
    for f in glob.glob(os.path.join(WEIGHTS_DIR, "vision_ssl_gram_candidate__snapshot__*.pt")):
        try:
            os.remove(f)
            print(f"[cleanup] removed {os.path.basename(f)} (consumed by Gram anchoring)")
        except OSError:
            pass


def run_hires_adaptation_phase(device, rank=0, world_size=1):
    """Phase 0b-1d. Gram teacher = the current EMA teacher (frozen once at
    the start of this phase), not the 0b-1a/b candidate snapshot. Always
    runs to completion in one call."""
    from dist_utils import is_main_process

    ssl_cfg = HYPERPARAMS["ssl"]
    run_ssl_training(
        device, "vision_hires_adapt", ssl_cfg["hires_iters"],
        use_gram=True, gram_update_every=None,
        global_size_choices=ssl_cfg["hires_global_crop_sizes"],
        local_size_choices=ssl_cfg["hires_local_crop_sizes"],
        rank=rank, world_size=world_size,
    )
    if not is_main_process(rank):
        return
    _drop_wrapper_files("vision_hires_adapt_wrapper")
    best = find_weights("vision_hires_adapt_encoder", "best")
    latest = find_weights("vision_hires_adapt_encoder", "latest")
    print(f"[optimal weights] vision_hires_adapt_encoder -> {best or latest}")
    print(f"[latest weights] vision_hires_adapt_encoder -> {latest}")


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
    if encoder_kind == "ocr":
        return OCRPretrainWrapper(config, active_names).to(device)
    return make_image_wrapper(config, active_names, device)


def _is_oom_error(e):
    """torch.cuda.OutOfMemoryError (a RuntimeError subclass) exists on
    recent torch; fall back to a message match so this still catches OOM on
    older versions that only ever raised a plain RuntimeError for it."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(e, oom_cls):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def short_train_and_eval(encoder_kind, config, tasks, ctx, device, budget):
    active = [t.name for t in tasks]
    wrapper = build_wrapper(encoder_kind, config, active, device)
    optimizer = torch.optim.AdamW(
        (p for p in wrapper.parameters() if p.requires_grad),
        lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    try:
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
        return validate(wrapper, tasks, ctx, config["batch_size"])
    finally:
        # Always runs, including on a CUDA OOM mid-trial -- otherwise the
        # failed wrapper/optimizer (and torch's fragmented allocator cache)
        # stay resident and the *next* trial OOMs too, even at a smaller,
        # genuinely-fine batch_size.
        del wrapper, optimizer
        if device == "cuda":
            torch.cuda.empty_cache()


def tune(encoder_kind, device):
    import optuna

    print(f"\n===== TUNING: {encoder_kind} encoder =====")
    cfg0 = make_config(encoder_kind, HYPERPARAMS[encoder_kind]["defaults"])
    tasks = build_tasks(encoder_kind, cfg0)
    codec = TextCodec()
    ctx = {"codec": codec, "device": device, "cfg": cfg0}
    space = HYPERPARAMS[encoder_kind]["search_space"]
    budget = HYPERPARAMS[encoder_kind]["tune_budget_batches"]

    os.makedirs(STUDY_DIR, exist_ok=True)
    study = optuna.create_study(
        study_name=f"{encoder_kind}_encoder",
        storage=f"sqlite:///{os.path.join(STUDY_DIR, encoder_kind + '.db')}",
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=0),
        load_if_exists=True,
    )

    # Seeded from this study's own pruned-trial history, which persists
    # across resubmissions (SQLite-backed) -- every pruned trial here is an
    # OOM (see objective() below, the only place this file prunes a trial),
    # so the smallest batch_size that ever OOM'd is a real, observed
    # data point, not a guess. Nothing else in this search space
    # (learning_rate/weight_decay/warmup_steps/loss weights) meaningfully
    # changes memory footprint, so "OOM'd at this batch_size" safely implies
    # "will OOM at this batch_size or larger" -- there's no reliable way to
    # *predict* the OOM threshold analytically ahead of time (actual memory
    # use depends on fragmentation, the CUDA allocator's reserved-vs-
    # allocated behavior, and whatever else is sharing the GPU -- see the
    # "28.38 GiB reserved... 26.16 GiB already allocated" gap in the crash
    # this is responding to), but *learning* it from a real failure and
    # skipping everything at least that large is both simpler and more
    # trustworthy than a formula.
    prior_oom = [
        t.params["batch_size"] for t in
        study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.PRUNED,))
        if "batch_size" in t.params
    ]
    known_oom_batch_size = min(prior_oom) if prior_oom else None
    if known_oom_batch_size is not None:
        print(f"  [tune] batch_size >= {known_oom_batch_size} previously OOM'd "
              f"for {encoder_kind}; skipping those without attempting them")

    # Built generically from HYPERPARAMS[encoder_kind]["search_space"] rather
    # than a hardcoded key list, since encoder kinds don't all tune the same
    # hyperparameters (e.g. OCR tunes loss-component weights instead of
    # decoder_layers/decoder_heads/dropout) -- "learning_rate"/"weight_decay"
    # are always the log-uniform (lo, hi) ranges, everything else is a
    # discrete choice list, same convention every encoder_kind's
    # search_space already follows.
    def objective(trial):
        nonlocal known_oom_batch_size
        hp = {}
        for key, choices in space.items():
            if key in ("learning_rate", "weight_decay"):
                hp[key] = trial.suggest_float(key, *choices, log=True)
            else:
                hp[key] = trial.suggest_categorical(key, choices)

        if (known_oom_batch_size is not None and "batch_size" in hp
                and hp["batch_size"] >= known_oom_batch_size):
            print(f"  trial {trial.number}: SKIPPED -- batch_size={hp['batch_size']} "
                  f">= known OOM threshold {known_oom_batch_size}")
            raise optuna.TrialPruned()

        cfg = make_config(encoder_kind, hp)
        ctx["cfg"] = cfg
        try:
            val = short_train_and_eval(encoder_kind, cfg, tasks, ctx, device, budget)
        except Exception as e:
            if not _is_oom_error(e):
                raise
            if "batch_size" in hp:
                known_oom_batch_size = min(known_oom_batch_size or hp["batch_size"],
                                           hp["batch_size"])
            print(f"  trial {trial.number}: OOM at batch_size={hp.get('batch_size')} "
                  f"({e}); pruning this trial and continuing with the next one.")
            raise optuna.TrialPruned()

        print(f"  trial {trial.number}: val_loss={val:.4f}  hp={hp}")
        wandb_log({f"tune/{encoder_kind}/val_loss": val, **{
            f"tune/{encoder_kind}/{k}": v for k, v in hp.items()}})
        return val

    # tune_trials is the *target total* completed trials for this study, not
    # "how many more to run" -- otherwise resubmitting after a crash (e.g.
    # the CUDA OOM this stage can hit) keeps adding tune_trials more forever
    # instead of the search ever finishing. Only COMPLETE trials count
    # (a trial cut short by a crash isn't a completed data point).
    target = HYPERPARAMS[encoder_kind]["tune_trials"]
    n_done = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
    n_remaining = max(0, target - n_done)
    if n_done:
        print(f"[resume] {encoder_kind}: {n_done} trial(s) already completed")
    if n_remaining == 0:
        print(f"[resume] {encoder_kind}: study already has {n_done} completed "
              f"trial(s) >= target tune_trials={target}; nothing to do.")
    else:
        print(f"[resume] {encoder_kind}: running {n_remaining} more trial(s) "
              f"to reach target tune_trials={target}")
        study.optimize(objective, n_trials=n_remaining)

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
def train_encoder(encoder_kind, device, rank=0, world_size=1):
    from dist_utils import (barrier, broadcast_parameters, is_main_process,
                            rank0_first, sync_gradients)

    is_distributed = world_size > 1
    main = is_main_process(rank)
    if main:
        print(f"\n===== TRAINING: {encoder_kind} encoder =====")
    hp = load_best_hparams(encoder_kind)
    config = make_config(encoder_kind, hp)
    tasks = rank0_first(rank, world_size, build_tasks, encoder_kind, config)
    active = [t.name for t in tasks]
    codec = TextCodec()
    ctx = {"codec": codec, "device": device, "cfg": config}
    task_map = {t.name: t for t in tasks}

    wrapper = build_wrapper(encoder_kind, config, active, device)
    optimizer = torch.optim.AdamW(
        (p for p in wrapper.parameters() if p.requires_grad),
        lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )

    # Only one task's head (out of wrapper.heads -- ner/pos/squad/mnli/...)
    # is active per batch, so most steps leave most heads' parameters with
    # grad=None. Since different ranks can be processing *different* tasks
    # at the same step (batches are interleaved across tasks, then sharded
    # across ranks), each rank's set of non-None grads -- and therefore
    # which parameters sync_gradients() calls all_reduce on -- can differ
    # from rank to rank. NCCL requires every rank to issue the exact same
    # sequence of collective calls; a mismatch there doesn't error, it just
    # hangs forever once some rank waits on an all_reduce no other rank will
    # ever issue (this is exactly what produced the "Watchdog caught
    # collective operation timeout" crash after 30 minutes). Pre-populating
    # every trainable parameter's .grad with zeros up front, and never
    # letting zero_grad() null them back out (set_to_none=False below),
    # guarantees every rank always has a real (if sometimes all-zero)
    # gradient tensor for every parameter, so sync_gradients() always
    # issues the identical set of all_reduce calls regardless of which
    # task's head happened to be active this step.
    if is_distributed:
        for p in wrapper.parameters():
            if p.requires_grad:
                p.grad = torch.zeros_like(p)

    epochs = HYPERPARAMS[encoder_kind]["epochs"]
    train_ds = UnifiedTaskDataset(tasks, "train")
    n_batches_per_epoch = len(
        TaskHomogeneousBatchSampler(train_ds, config["batch_size"], config["seed"],
                                    rank=rank, world_size=world_size)
    )
    total_steps = n_batches_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(config["warmup_steps"], total_steps)
    )

    # ---- Resume from the latest checkpoint if present --------------------- #
    # 'latest' is continuously updated by save_checkpoint (see its
    # docstring); _find_valid_resume falls back to 'best' if 'latest' is
    # corrupted, instead of crashing the whole run immediately on resume.
    # Every rank reads the same shared-storage file independently (a safe,
    # read-only op), so resuming needs no broadcast; a *fresh* start does,
    # since every rank's random init otherwise differs (see below).
    start_epoch, start_batch, global_step = 0, 0, 0
    best_val = float("inf")
    resume, blob = _find_valid_resume(f"{encoder_kind}_wrapper", device)
    if resume:
        wrapper.load_state_dict(blob["wrapper_state"])
        optimizer.load_state_dict(blob["optimizer_state"])
        global_step, best_val = blob["global_step"], blob["best_val_loss"]
        for _ in range(global_step):          # fast-forward the LR decay schedule
            scheduler.step()
        if _is_epoch_complete(resume):
            # Only trust that an epoch is truly finished if we're resuming
            # from its literal 'complete' marker checkpoint -- a mid-epoch
            # batch checkpoint never implies completion, even if its batch
            # number happens to equal that epoch's total batch count.
            start_epoch, start_batch = blob["epoch"] + 1, 0
        else:
            start_epoch, start_batch = blob["epoch"], blob["batch"]
        if main:
            print(f"[resume] {os.path.basename(resume)} -> epoch {start_epoch}, "
                  f"batch {start_batch}")
    elif is_distributed:
        broadcast_parameters(wrapper.parameters(), src=0)
    barrier(is_distributed)

    eval_every = config["eval_every_batches"]
    ckpt_every = config["checkpoint_every_batches"]

    wrapper.train()
    for epoch in range(start_epoch, epochs):
        # One static dataloader per epoch, deterministically seeded so a resumed
        # run reproduces the exact batch order and can skip finished batches.
        loader = make_loader(train_ds, tasks, ctx, config["batch_size"],
                             seed=config["seed"] + epoch, rank=rank, world_size=world_size)
        n_batches = len(loader)
        skip_to = start_batch if epoch == start_epoch else 0
        for batch_idx, batch in enumerate(loader):
            if batch_idx < skip_to:          # fast-forward over finished batches
                continue
            tname = batch["task_name"]
            t = task_map[tname]              # activate only this task's head
            loss, acc, _ = t.loss_acc(wrapper, batch)
            # set_to_none=False (distributed only) keeps every parameter's
            # .grad a real zero tensor rather than nulling it back out --
            # see the comment above the pre-populate loop earlier in this
            # function for why that matters here.
            optimizer.zero_grad(set_to_none=not is_distributed)
            loss.backward()
            sync_gradients(wrapper.parameters(), world_size)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if main and global_step % 50 == 0:
                print(f"  e{epoch} b{batch_idx} step{global_step} "
                      f"task={tname} loss={loss.item():.4f} acc={acc:.3f}")
                wandb_log({f"train/{encoder_kind}/loss": loss.item(),
                           f"train/{encoder_kind}/acc": acc,
                           f"train/{encoder_kind}/task": tname,
                           "epoch": epoch, "batch": batch_idx})

            # Eval-step checkpoints already embed val_loss, which strictly
            # subsumes a plain ckpt_every save at the same global_step, so
            # only fall back to the plain save when this step isn't also an
            # eval step (avoids two near-duplicate files for one step).
            # Rank-0-only: validation/checkpointing don't need every rank's
            # participation (no collective involved), so other ranks just
            # move on to their next batch instead of waiting.
            if main and global_step % eval_every == 0:
                val = validate(wrapper, tasks, ctx, config["batch_size"])
                best_val = min(best_val, val)
                print(f"  [val] step {global_step}: val_loss={val:.4f} "
                      f"(best={best_val:.4f})")
                wandb_log({f"val/{encoder_kind}/loss": val})
                save_checkpoint(encoder_kind, wrapper, optimizer, epoch,
                                batch_idx, global_step, best_val, val_loss=val)
            elif main and global_step % ckpt_every == 0:
                save_checkpoint(encoder_kind, wrapper, optimizer, epoch,
                                batch_idx, global_step, best_val)

        # ---- End-of-epoch validation + checkpoint ------------------------- #
        # All ranks must reach this point before any of them starts the next
        # epoch's dataloader (rank 0's extra validation/checkpoint work makes
        # it slower per epoch than the others) -- otherwise a faster rank
        # could race ahead into gradient sync calls rank 0 hasn't reached yet.
        barrier(is_distributed)
        if not main:
            start_batch = 0
            continue
        val = validate(wrapper, tasks, ctx, config["batch_size"])
        best_val = min(best_val, val)
        print(f"[epoch {epoch}] val_loss={val:.4f} (best={best_val:.4f})")
        wandb_log({f"val/{encoder_kind}/epoch_loss": val, "epoch": epoch})
        save_checkpoint(encoder_kind, wrapper, optimizer, epoch,
                        n_batches, global_step, best_val, val_loss=val,
                        epoch_complete=True)
        start_batch = 0

    barrier(is_distributed)
    if not main:
        return
    # 'best'/'latest' are maintained continuously by save_checkpoint (see its
    # docstring), so no finalize step is needed here -- just read them. The
    # resumable 'wrapper' files (optimizer/RNG state) are no longer needed
    # now that training has genuinely finished, so they're dropped, leaving
    # only the 2 reusable encoder files (best + latest) on disk.
    _drop_wrapper_files(f"{encoder_kind}_wrapper")
    best_encoder = find_weights(f"{encoder_kind}_encoder", "best")
    latest_encoder = find_weights(f"{encoder_kind}_encoder", "latest")
    print(f"[done] {encoder_kind} encoder trained.")
    print(f"[optimal weights] {encoder_kind}_encoder -> {best_encoder or latest_encoder}")
    print(f"[latest weights] {encoder_kind}_encoder -> {latest_encoder}")


# --------------------------------------------------------------------------- #
# Stage driver
# --------------------------------------------------------------------------- #
def run_stages(args, device, rank=0, world_size=1):
    """Optuna tuning stages stay single-process (rank 0 only, other ranks
    idle at a barrier) -- many short, independent trials is a different,
    unrelated axis of parallelism from data-parallel training (see
    dist_utils.py's module docstring). Training/SSL stages run on every
    rank via train_encoder / run_ssl_training's own distributed handling."""
    from dist_utils import barrier, is_main_process

    is_distributed = world_size > 1
    main = is_main_process(rank)
    progress = load_progress()
    for stage in STAGES:
        if stage in progress["completed"]:
            continue

        if main:
            init_wandb(args.wandb_key, stage)

        if stage == "vision_ssl_pretrain":
            proceed = run_ssl_pretrain_phase(device, rank=rank, world_size=world_size)
            if main:
                finish_wandb()
            if not proceed:
                if main:
                    print("\n*** Stage 'vision_ssl_pretrain' paused for review. Inspect the "
                          "patch cosine-similarity visualization just written to DATA_DIR. "
                          "If the maps still look clean, bump HYPERPARAMS['ssl']['target_iters'] "
                          "and resubmit to keep pretraining (0b-1b); if they're degrading (or "
                          "you're satisfied with this budget), flip "
                          "HYPERPARAMS['ssl']['proceed_to_gram_anchoring'] to True and resubmit "
                          "to move on to Gram anchoring (0b-1c). ***")
                return
            if main:
                mark_stage_done(stage)
            barrier(is_distributed)
            continue

        if stage == "vision_gram_anchor":
            run_gram_anchoring_phase(device, rank=rank, world_size=world_size)
            if main:
                mark_stage_done(stage)
                finish_wandb()
            barrier(is_distributed)
            continue

        if stage == "vision_hires_adapt":
            run_hires_adaptation_phase(device, rank=rank, world_size=world_size)
            if main:
                mark_stage_done(stage)
                finish_wandb()
            barrier(is_distributed)
            continue

        if stage.startswith("text"):
            kind = "text"
        elif stage.startswith("ocr"):
            kind = "ocr"
        else:
            kind = "image"

        if stage.endswith("tune"):
            if main:
                tune(kind, device)
                mark_stage_done(stage)
                finish_wandb()
                print(f"\n*** Stage '{stage}' complete. Stopping for review. "
                      f"Re-submit the job to continue with training. ***")
            barrier(is_distributed)
            return                       # internal stop for review
        else:
            train_encoder(kind, device, rank=rank, world_size=world_size)
            if main:
                mark_stage_done(stage)
                finish_wandb()
            barrier(is_distributed)
            # training stages auto-advance into the next (tuning) stage

    if main:
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
        # Only the text datasets are populated up front. The OCR and image
        # datasets are pulled lazily when their own tuning stage begins (OCR
        # documents and image-task datasets are both heavier per-sample than
        # the text tasks), so we deliberately do not download them here.
        # Nothing past text_train ever needs text data prepared again, so
        # skip re-touching it (re-verifying/reloading every cached dataset
        # from disk on every resubmission, even once both text stages are
        # long done) once both are already marked complete.
        text_done = {"text_tune", "text_train"} <= set(load_progress()["completed"])
        if text_done:
            if main_process:
                print("\n[data] text_tune/text_train already complete; "
                      "skipping text dataset preparation.")
            cleanup_distributed(is_distributed)
            return
        gcfg = dict(HYPERPARAMS["global"])
        if main_process:
            print("\n[data] preparing text datasets (OCR and image datasets are "
                  "pulled later, at their own tuning stages)...")
        try:
            rank0_first(rank, world_size, build_tasks, "text", gcfg)
        except Exception as e:
            if main_process:
                print(f"[data] text: {e}")
        if main_process:
            print("[data] text preparation complete.")
        cleanup_distributed(is_distributed)
        return

    run_stages(args, device, rank=rank, world_size=world_size)
    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    sys.exit(main())
