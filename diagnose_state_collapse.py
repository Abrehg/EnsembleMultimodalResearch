"""
Phase 2 diagnostic (see /Users/adityaasuratkal/.claude/plans/dazzling-toasting-moler.md):
Phase 0's within/across-source cosine-similarity probe (evaluate_rewards'
state_similarity, see text_to_text.py) confirmed the collapse is upstream --
pooled `state` is ~indistinguishable (cosine sim ~0.98-1.0) across rows from
entirely different sources (logi_glue/trivia_qa/nq_open), at every checkpoint
of a real short verification run. That only answers upstream-vs-downstream,
though -- `encode_to_state`'s pipeline is actually three separately-
initialized stages stacked on top of the frozen text encoder:

    1. model.encode(text, "text")          -> frozen TextEncoder's own output
    2. model.combine.encoder(...)          -> HybridLinearEncoder (10 layers:
                                               8 LinearAttentionBlock + 2 full
                                               attention), pre-pooling
    3. model.combine.pool(...)             -> AdaptiveAttentionPooling
                                               (fixed learned query cross-
                                               attention + salience gating)
    4. model.router(...)                   -> RoutingLogic's state_encoder
                                               (a further 2-layer
                                               TransformerEncoder over the
                                               pooled latents)

Any of stages 2-4 (all freshly initialized every run, unlike the frozen
stage-1 text encoder) could be where cross-input diversity actually gets
crushed -- this script re-runs _pairwise_state_similarity (text_to_text.py)
at each stage's output to localize exactly where the within/across gap
disappears, instead of guessing from architecture alone.

Uses the REAL, most-trained checkpoints (t2t_stage1_combine/router, latest
epoch 0031 as of this writing) via build_model(checkpoint_prefix="t2t_stage1"),
not a fresh/throwaway init like collapse_fix_harness.py -- the question here
is where collapse happens in the actual production run, not in an early-step
harness restart.

Read-only: no training, no backward pass, no checkpoints written. Cheap
enough (one forward pass per sampled row, no generation/RL rollout) to run
on CPU if no GPU is available.
"""
import sys, os, random

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "models"))
os.chdir(PROJECT_DIR)

import torch
import text_to_text as t2t

SAMPLE_SIZE = int(os.environ.get("DIAG_SAMPLE_SIZE", 60))
SEED = int(os.environ.get("DIAG_SEED", 0))

STAGES = ["1_frozen_encoder", "2_hybrid_memory", "3_pooled_combined", "4_router_state"]


def log(msg):
    print(msg, flush=True)


@torch.no_grad()
def trace_encoder_layers(model, text):
    """Zooms into model.combine.encoder (HybridLinearEncoder) layer-by-layer,
    once the 4-macro-stage trace (trace_stages) has already localized the
    collapse to somewhere inside it (see this module's docstring's update --
    stage 1, the frozen text encoder, was healthy; stage 2, this encoder's
    full-stack output, was already fully collapsed). 10 layers (2 groups of
    [4 LinearAttentionBlock + 1 full-attention anchor], see EncCombine.py's
    HybridLinearEncoder), each tagged by kind so a collapse concentrated in
    the linear-attention layers specifically (its closed-form O = Q_phi @
    (K_phi^T @ V) pools every position's K,V into one shared per-head
    statistic -- a structural reason different inputs could converge toward
    a similar output) shows up distinctly from one spread evenly across
    both layer kinds, or concentrated in the standard full-attention
    anchors instead."""
    encodings = model.encode(text, "text")
    x = torch.cat(encodings, dim=1)
    per_layer = {"00_input": x.mean(dim=1).squeeze(0)}
    for i, layer in enumerate(model.combine.encoder.layers):
        x = layer(x)
        kind = "linear" if layer.use_linear_attn else "FULL"
        per_layer[f"{i + 1:02d}_{kind}"] = x.mean(dim=1).squeeze(0)
    return per_layer


@torch.no_grad()
def trace_stages(model, text):
    """Mirrors encode_to_state's own call sequence (text_to_text.py) exactly,
    just keeping every intermediate tensor instead of only the final state.
    Pooling convention (mean over sequence dim, then squeeze batch) matches
    koleo_loss/evaluate_rewards' existing pooling so results are directly
    comparable to Phase 0's already-collected state_similarity numbers."""
    encodings = model.encode(text, "text")
    stage1 = encodings[0].mean(dim=1).squeeze(0)

    combined_seq = torch.cat(encodings, dim=1)
    memory = model.combine.encoder(combined_seq)
    stage2 = memory.mean(dim=1).squeeze(0)

    combined, _, _ = model.combine.pool(memory, total_seq_len=combined_seq.size(1))
    stage3 = combined.mean(dim=1).squeeze(0)

    _, state = model.router(combined)
    stage4 = state.mean(dim=1).squeeze(0)

    return {"1_frozen_encoder": stage1, "2_hybrid_memory": stage2,
            "3_pooled_combined": stage3, "4_router_state": stage4}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[device] {device}")

    gcfg = t2t.HYPERPARAMS["global"]
    d_model = gcfg["d_model"]
    log(f"[model] building at real d_model={d_model}, warm-starting combine/router "
        f"from the real t2t_stage1 checkpoints (not a fresh/throwaway init)...")
    model = t2t.build_model(device, d_model, gcfg["half_precision_encoder"],
                            checkpoint_prefix="t2t_stage1")
    model.eval()

    t2t.ensure_dataset_ready()
    val_rows = t2t.load_csv_rows(t2t.VAL_CSV)
    rows = t2t.sample_rows(val_rows, min(SAMPLE_SIZE, len(val_rows)), seed=SEED)
    by_source = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    log(f"[data] {len(rows)} rows sampled: {by_source}")

    pooled_by_stage = {s: [] for s in STAGES}
    pooled_by_layer = {}
    sources = []
    for i, r in enumerate(rows):
        stages = trace_stages(model, r["input"])
        for s in STAGES:
            pooled_by_stage[s].append(stages[s])
        layers = trace_encoder_layers(model, r["input"])
        for k, v in layers.items():
            pooled_by_layer.setdefault(k, []).append(v)
        sources.append(r["source"])
        if (i + 1) % 20 == 0:
            log(f"[progress] {i + 1}/{len(rows)} rows traced")

    log("\n===== Per-stage within/across-source cosine similarity =====")
    log(f"{'stage':<20} {'within_source':>14} {'across_source':>14} {'gap':>10}")
    for s in STAGES:
        sim = t2t._pairwise_state_similarity(pooled_by_stage[s], sources)
        gap = sim["within_source"] - sim["across_source"]
        log(f"{s:<20} {sim['within_source']:>14.4f} {sim['across_source']:>14.4f} {gap:>10.4f}")

    log("\nInterpretation: a stage where within~=across (gap near 0, both near "
        "1.0) has already lost cross-input diversity by that point in the "
        "pipeline. The FIRST stage (in order 1->4) where this happens is "
        "where diversity actually gets crushed -- stages after it may look "
        "just as collapsed, but that's inherited, not newly caused there.")

    log("\n===== Layer-by-layer trace inside HybridLinearEncoder (model.combine.encoder) =====")
    log(f"{'layer':<20} {'within_source':>14} {'across_source':>14} {'gap':>10}")
    for k in sorted(pooled_by_layer.keys()):
        sim = t2t._pairwise_state_similarity(pooled_by_layer[k], sources)
        gap = sim["within_source"] - sim["across_source"]
        log(f"{k:<20} {sim['within_source']:>14.4f} {sim['across_source']:>14.4f} {gap:>10.4f}")

    log("\nInterpretation: '00_input' is this encoder's own input (should "
        "match '1_frozen_encoder' above almost exactly -- a sanity check on "
        "the trace itself). Each subsequent row is the output after that "
        "one additional layer (kind tagged 'linear' for LinearAttentionBlock, "
        "'FULL' for the full-attention anchor). Whichever layer's row is the "
        "first to show gap near 0 (both similarities near 1.0) is where "
        "collapse actually happens -- if it's concentrated in 'linear' rows, "
        "that implicates LinearAttentionBlock's closed-form global KV "
        "statistic specifically, not depth or the full-attention anchors.")


if __name__ == "__main__":
    main()
