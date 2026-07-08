"""
Dimension tuning stage for the ensemble multimodal research model.

Goal
----
Find the optimal model ``dimension`` (d_model) for the *overall* model on a
basic factual QA task (TriviaQA). We use Optuna to search over the candidate
dimensions and optimise *two* objectives simultaneously:

    1. Exact-match accuracy on held-out TriviaQA questions   (maximise)
    2. FLOPs consumed per example                            (minimise)

This is a multi-objective study, so Optuna returns a Pareto front of
(accuracy, flops) trade-offs rather than a single "best" trial. We then apply a
simple accuracy-per-FLOP heuristic to recommend one dimension.

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
the registries, combination and router stages are kept intact.

TriviaQA is downloaded once from Hugging Face into a plain JSONL file on
scratch and iterated linearly (no Arrow/DB store), so subsequent runs read it
straight off disk with no network access. The Optuna study is held in memory
(no SQLite), so each run starts from scratch with nothing to clear.

The tuning objective is validation loss / perplexity on the held-out answers
(minimized), not exact match: a from-scratch model scores ~0 EM on closed-book
TriviaQA, which is uninformative, whereas loss is continuous and sensitive to
model capacity -- so the dimension actually shows up in the signal.

Dataset: https://huggingface.co/datasets/mandarjoshi/trivia_qa
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

# TriviaQA is cached as plain JSONL (one example per line) and iterated
# linearly -- no Arrow/DB store. Each line: {"question", "answer", "aliases"}.
TRIVIAQA_JSONL = {
    "train": os.path.join(DATA_DIR, "trivia_qa_train.jsonl"),
    "validation": os.path.join(DATA_DIR, "trivia_qa_validation.jsonl"),
}
# How many rows to persist per split on first download (loaders cap further).
PREP_TRAIN_CAP = 50_000
PREP_VAL_CAP = 5_000

# The Optuna study is kept in-memory (no SQLite) so every run starts fresh --
# there is no DB to clear. Only the human-readable results summary is written.
RESULTS_JSON = os.path.join(DATA_DIR, "dim_tune_results.json")

# Reuse the project's SentencePiece tokenizer so the tuning stage speaks the
# same vocabulary as the rest of the model.
TOKENIZER_PATH = os.path.join(MODELS_DIR, "Encoders", "multimodal_bpe.model")

# TriviaQA config: the "no context" reading-comprehension split gives clean
# (question -> answer) pairs without the heavy Wikipedia/web evidence docs,
# which is exactly what we want for a basic factual-QA tuning signal.
TRIVIAQA_HF_NAME = "mandarjoshi/trivia_qa"
TRIVIAQA_CONFIG = "rc.nocontext"

# --- The primary search dimension ------------------------------------------ #
DIMENSION_CHOICES = [512, 768, 1024, 1536, 2048]

# --- Auxiliary ("dummy") learnable hyperparameters -------------------------- #
# Searched jointly with ``dimension`` so each dimension is scored near its own
# best training configuration. These do not change the *architecture width*;
# they only govern how each candidate is optimised.
LR_RANGE = (5e-5, 5e-3)                     # log-uniform
BATCH_SIZE_CHOICES = [8, 16, 32, 64]        # realised via gradient accumulation
WEIGHT_DECAY_RANGE = (1e-5, 1e-1)           # log-uniform
WARMUP_CHOICES = [0, 10, 50]                # optimizer-step linear warmup
LABEL_SMOOTHING_CHOICES = [0.0, 0.1]

MAX_SRC_LEN = 48       # question length cap (tokens)
MAX_TGT_LEN = 24       # answer length cap (tokens)

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
# Dataset preparation: download once into plain JSONL on scratch, then iterate
# it linearly. No Arrow / DB store.
# --------------------------------------------------------------------------- #
def _row_to_record(row):
    """Extract {question, answer, aliases} from a raw TriviaQA row, or None."""
    answer = row["answer"]
    value = answer.get("value") or ""
    aliases = list(answer.get("aliases") or [])
    aliases += list(answer.get("normalized_aliases") or [])
    if value:
        aliases.append(value)
    aliases = [a for a in dict.fromkeys(aliases) if a]   # de-dup, keep order
    if not value or not aliases:
        return None
    return {"question": row["question"], "answer": value, "aliases": aliases}


def prepare_triviaqa():
    """Stream TriviaQA from Hugging Face and write each split to a JSONL file on
    scratch (one JSON record per line). No-op if the files already exist."""
    from datasets import load_dataset

    os.makedirs(DATA_DIR, exist_ok=True)

    if all(os.path.isfile(p) for p in TRIVIAQA_JSONL.values()):
        print("[data] TriviaQA JSONL already present on scratch.")
        return

    caps = {"train": PREP_TRAIN_CAP, "validation": PREP_VAL_CAP}
    for split, path in TRIVIAQA_JSONL.items():
        print(f"[data] Downloading {TRIVIAQA_HF_NAME} ({TRIVIAQA_CONFIG}) "
              f"split='{split}' -> {path}")
        stream = load_dataset(
            TRIVIAQA_HF_NAME, TRIVIAQA_CONFIG, split=split, streaming=True,
        )
        written = 0
        with open(path, "w") as f:
            for row in stream:
                rec = _row_to_record(row)
                if rec is None:
                    continue
                f.write(json.dumps(rec) + "\n")
                written += 1
                if written >= caps[split]:
                    break
        print(f"[data]   wrote {written} records")


def load_examples(split, limit=None):
    """Read a JSONL split linearly into a list of (question, answer, aliases)."""
    path = TRIVIAQA_JSONL[split]
    examples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            obj = json.loads(line)
            examples.append((obj["question"], obj["answer"], obj["aliases"]))
    return examples


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

    print(
        f"\n[trial] dimension={dimension} | lr={hp['learning_rate']:.2e} "
        f"| batch={hp['batch_size']} | wd={hp['weight_decay']:.2e} "
        f"| warmup={hp['warmup_steps']} | ls={hp['label_smoothing']} "
        f"| device={device}"
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
    while opt_step < max_opt_steps:
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
            if opt_step % max(1, max_opt_steps // 5) == 0:
                avg = running_loss / batch_size
                print(f"  opt-step {opt_step:4d}/{max_opt_steps}  loss={avg:.4f}")
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

    prepare_triviaqa()
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
                "learning_rate", *LR_RANGE, log=True
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

    # In-memory study (no SQLite storage) -> every run starts fresh; there is
    # nothing to clear between runs. NSGA-II explores dimension + auxiliary
    # hyperparameters and converges on the (loss, FLOPs) Pareto front.
    sampler = optuna.samplers.NSGAIISampler(seed=0)
    study = optuna.create_study(
        directions=["minimize", "minimize"],  # val loss down, FLOPs down
        sampler=sampler,
    )
    study.optimize(objective, n_trials=args.n_trials)

    report_results(study)


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

    # Single recommendation: among Pareto-optimal dimensions, the best balance
    # of accuracy (low loss) and efficiency (low FLOPs), scored on min-max
    # normalized [0,1] loss + FLOPs (lower is better on both).
    recommended = None
    if pareto:
        losses = [r["val_loss"] for r in pareto]
        flps = [r["flops_per_example"] for r in pareto]
        lo_l, hi_l = min(losses), max(losses)
        lo_f, hi_f = min(flps), max(flps)

        def _score(r):
            nl = (r["val_loss"] - lo_l) / (hi_l - lo_l) if hi_l > lo_l else 0.0
            nf = (r["flops_per_example"] - lo_f) / (hi_f - lo_f) if hi_f > lo_f else 0.0
            return nl + nf                      # lower is better on both axes

        recommended = min(pareto, key=_score)
        print(
            f"\nRecommended dimension: {recommended['dimension']} "
            f"(val_loss={recommended['val_loss']:.4f}, "
            f"ppl={recommended['perplexity']:.2f}, "
            f"FLOPs/ex={recommended['flops_per_example']:.3e}, "
            f"hp={recommended['best_hparams']})"
        )

    results = {
        "dimension_choices": DIMENSION_CHOICES,
        "auxiliary_search_space": {
            "learning_rate": LR_RANGE,
            "batch_size": BATCH_SIZE_CHOICES,
            "weight_decay": WEIGHT_DECAY_RANGE,
            "warmup_steps": WARMUP_CHOICES,
            "label_smoothing": LABEL_SMOOTHING_CHOICES,
        },
        "per_dimension": per_dim,
        "pareto_front": pareto,
        "recommended": recommended,
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
                   help="Optuna trials this run (in-memory study, always fresh). "
                        "Dimension is the variable of interest, so a small "
                        "number suffices.")
    p.add_argument("--train_examples", type=int, default=20000,
                   help="Cap on TriviaQA training examples read from JSONL.")
    p.add_argument("--eval_examples", type=int, default=2000,
                   help="Cap on TriviaQA eval examples read from JSONL.")
    p.add_argument("--max_opt_steps", type=int, default=3000,
                   help="Optimizer steps per trial "
                        "(each step accumulates that trial's batch size).")
    p.add_argument("--eval_samples", type=int, default=500,
                   help="Number of eval examples scored for validation loss.")
    p.add_argument("--prepare_data_only", action="store_true",
                   help="Download TriviaQA to JSONL on scratch, then exit.")
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
        prepare_triviaqa()
        print("[data] Preparation complete.")
        return

    run_study(args)


if __name__ == "__main__":
    sys.exit(main())
