"""
Offline verification harness for the P1.1/P1.2 anti-collapse fixes (see
/Users/adityaasuratkal/.claude/plans/dazzling-toasting-moler.md -- the
"content-independent shortcut" collapse investigation triggered by
slurm-4752878, where unique_pred_fraction stayed pinned at 0.003-0.005 for
an entire 43,300+-step real run). Meant to run on the cluster (see
collapse_fix_harness.sh) as a single-GPU job -- CPU-only local testing
measured ~44s/step (t2t_rl_term's full 32-token autoregressive generation
dominates), making the originally-planned 5,000-8,000-step verification
take 2+ days on a laptop; a real GPU should make this comfortably feasible
within the 48-hour QOS window instead.

Reuses text_to_text.py's REAL functions directly (encode_to_state,
compute_task_loss, t2t_rl_term, salience_diversity_loss/salience_margin_loss,
decoder_diversity_loss/_buffer_diversity, koleo_loss, scheduled_coef,
_schedule_weights, evaluate_rewards, gate_health, build_model, SpmCodec,
load_best_hparams, SOURCE_LOADERS) rather than reimplementing them, to
minimize the risk of the harness silently diverging from what a real run
actually does. Only the per-row loop's outer scaffolding (data loading,
epoch/step bookkeeping, checkpoint-free operation) is new.

Fixes the methodology bug the plan identified: `_schedule_weights` (the
supervised/RL blend) and the LR scheduler both use `total_steps` as their
progress-fraction denominator. A short run's OWN small total_steps would
make both ramp to full strength almost immediately -- so this harness
explicitly passes the REAL production `total_steps` (~68,700, cross-checked
against slurm-4752878's logged coefficient values) into both, even though
this harness itself only executes HARNESS_TARGET_STEPS. This makes RL's
share and the LR schedule match what the real run experiences at the same
global_step, not what a short run's own recomputed schedule would give.
(The four anti-collapse coefficients -- brier/blank/repetition/
content_closeness -- don't need this treatment anymore: P1.1 already made
them ramp against the fixed HYPERPARAMS["reward"]["anti_collapse_ramp_steps"]
=3000, not total_steps, so they're correct by construction regardless of
harness scale.)

Real d_model=1024 is used throughout (not a smaller toy width) because the
one real frozen text encoder checkpoint on disk
(text_encoder_best_epoch_0001_complete_valLoss2.4342.pt) is pinned at
d_model=1024 (verified: token_embedding [32000,512], output_proj [1024,512])
-- there is no smaller real pretrained encoder to substitute, and the plan
explicitly prioritizes a real frozen encoder over a smaller width.

combine/router/expert_registry/text_output start FRESH (random init) --
checkpoint_prefix is a throwaway value no real file matches -- matching a
real stage-1 run's actual starting point (only the frozen text encoder is
pretrained). No checkpoints are saved to the real weights/ directory at all;
this harness is read-only with respect to the real project's persistent
state. Results are appended incrementally to RESULTS_PATH (DATA_DIR, not
weights/) as JSONL, so a SLURM time-limit kill still leaves partial results
readable.

Data: a real, proportioned SUBSET of stage-1's three sources (logi_glue
~36.6%, trivia_qa/nq_open ~31.7% each, matching the real run's
120,899/329,849 and an even split of the remainder -- see slurm-4753091's
dataset counts), loaded via the real SOURCE_LOADERS functions, not
synthetic data.

Stopping: whichever comes first of HARNESS_TARGET_STEPS or
HARNESS_MAX_HOURS (a safety margin under the SLURM job's own --time limit,
since real per-step GPU throughput is unmeasured going in) -- see
collapse_fix_harness.sh for both values.

Pass criterion (from the plan): unique_pred_fraction recovers above its
proven collapse floor (0.003-0.005) and STAYS recovered across 5-10
consecutive validation checkpoints at the real eval_every_batches=300
cadence.
"""
import sys, os, time, random, json, copy

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "models"))
os.chdir(PROJECT_DIR)

import torch
import text_to_text as t2t
# Set before the first CUDA kernel runs. Identical configs with the same seed
# were diverging at step 50 (slurm-4829926 vs 4829974), so no single-run A/B
# was interpretable until this was pinned.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
# slurm-4830018's stderr showed the memory-efficient SDPA kernel is nondeterministic
# even with the flag above, so force the math kernel for attention.
for _disable in ("enable_mem_efficient_sdp", "enable_flash_sdp"):
    if hasattr(torch.backends.cuda, _disable):
        getattr(torch.backends.cuda, _disable)(False)

# Real cluster paths (t2t.HF_CACHE/DATA_DIR) are genuine and writable here --
# unlike the local-Mac testing earlier in this investigation, no override
# needed. DATA_DIR should already exist from real training runs, but
# create it defensively rather than assume.
os.makedirs(t2t.DATA_DIR, exist_ok=True)
RUN_TAG = os.environ.get("HARNESS_RUN_TAG") or os.environ.get("SLURM_JOB_ID", "local")
RESULTS_PATH = os.path.join(t2t.DATA_DIR, f"collapse_fix_harness_results_{RUN_TAG}.jsonl")
HARNESS_SEED = int(os.environ.get("HARNESS_SEED", 0))
POOL_KOLEO_COEF_OVERRIDE = os.environ.get("HARNESS_POOL_KOLEO_COEF")
BLANK_IDS_OVERRIDE = os.environ.get("HARNESS_BLANK_IDS")  # "all" (default) or "eos" (original EOS-only penalty)
BLANK_COEF_OVERRIDE = os.environ.get("HARNESS_BLANK_COEF")  # fixed blank-penalty weight; unset = use the ramp

REAL_TOTAL_STEPS = 68700           # cross-checked real total_steps (epochs_stage1=50 * n_opt_steps_per_epoch)
HARNESS_TARGET_STEPS = int(os.environ.get("HARNESS_TARGET_STEPS", 8000))
HARNESS_MAX_HOURS = float(os.environ.get("HARNESS_MAX_HOURS", 40))  # safety margin under --time=48:00:00
EVAL_EVERY = int(os.environ.get("HARNESS_EVAL_EVERY", 300))          # matches real eval_every_batches
POOL_SIZE = int(os.environ.get("HARNESS_POOL_SIZE", 4000))           # real-proportioned subset size
LOGI_GLUE_FRAC = 120899 / 329849   # real stage-1 proportion (~36.6%)


def log(msg):
    print(msg, flush=True)


def load_real_proportioned_pool():
    log("[data] loading real stage-1 sources (capped, real proportions)...")
    n_logi = int(POOL_SIZE * LOGI_GLUE_FRAC)
    n_other = (POOL_SIZE - n_logi) // 2

    # text_to_text's own SOURCE_LOADERS: logi_glue ignores cap (loads everything,
    # by design -- see its own comment), so cap it ourselves after loading;
    # trivia_qa/nq_open respect cfg["data"]["<source>_cap"] directly.
    cfg = copy.deepcopy(t2t.HYPERPARAMS)
    cfg["data"]["trivia_qa_cap"] = n_other
    cfg["data"]["nq_open_cap"] = n_other

    logi_rows = t2t.SOURCE_LOADERS["logi_glue"](cfg)
    random.Random(0).shuffle(logi_rows)
    logi_rows = logi_rows[:n_logi]

    trivia_rows = t2t.SOURCE_LOADERS["trivia_qa"](cfg)
    nq_rows = t2t.SOURCE_LOADERS["nq_open"](cfg)

    pool = logi_rows + trivia_rows + nq_rows
    random.Random(1).shuffle(pool)
    log(f"[data] pool: {len(logi_rows)} logi_glue + {len(trivia_rows)} trivia_qa + "
        f"{len(nq_rows)} nq_open = {len(pool)} total")

    val_n = max(50, len(pool) // 10)
    val_rows, train_rows = pool[:val_n], pool[val_n:]
    log(f"[data] train={len(train_rows)} val={len(val_rows)}")
    return train_rows, val_rows


def main():
    t_start = time.time()
    torch.manual_seed(HARNESS_SEED)
    random.seed(HARNESS_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[device] {device} seed={HARNESS_SEED} run_tag={RUN_TAG}")
    if device == "cpu":
        log("[device] WARNING: no GPU detected -- this harness was sized "
            "assuming GPU throughput (see HARNESS_MAX_HOURS); on CPU it may "
            "not reach a useful step count within the time budget.")

    train_rows, val_rows = load_real_proportioned_pool()

    gcfg = t2t.HYPERPARAMS["global"]
    coefs = dict(t2t.HYPERPARAMS["reward"])
    if POOL_KOLEO_COEF_OVERRIDE is not None:
        coefs["pool_koleo_coef"] = float(POOL_KOLEO_COEF_OVERRIDE)
    if BLANK_IDS_OVERRIDE == "eos":
        coefs["blank_penalty_ids"] = (t2t.EOS_ID,)
    elif BLANK_IDS_OVERRIDE not in (None, "", "all"):
        raise ValueError(f"HARNESS_BLANK_IDS must be 'all' or 'eos', got {BLANK_IDS_OVERRIDE!r}")
    log(f"[coefs] blank_penalty_ids={tuple(coefs['blank_penalty_ids'])}")
    log(f"[coefs] pool_koleo_coef={coefs['pool_koleo_coef']} "
        f"encoder_koleo_coef={coefs['encoder_koleo_coef']} koleo_coef={coefs['koleo_coef']}")
    d_model = gcfg["d_model"]

    log(f"[model] building at real d_model={d_model} with real frozen encoder...")
    model = t2t.build_model(device, d_model, gcfg["half_precision_encoder"],
                            checkpoint_prefix="t2t_HARNESS_THROWAWAY_NO_REAL_FILE")
    expert = model.expert_registry.get_expert("text_output")
    codec = t2t.SpmCodec(max_len=gcfg["max_target_len"])

    hp = t2t.load_best_hparams()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=hp["learning_rate"], weight_decay=hp["weight_decay"],
    )
    batch_size = hp["batch_size"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, t2t.make_lr_lambda(hp["warmup_steps"], REAL_TOTAL_STEPS, hp["lr_min_scale"]),
    )
    log(f"[model] hp: batch_size={batch_size} lr={hp['learning_rate']} "
        f"warmup_steps={hp['warmup_steps']} (LR/RL schedules pinned to REAL "
        f"total_steps={REAL_TOTAL_STEPS}, not this harness's own step count)")

    usable_rows = len(train_rows) - (len(train_rows) % batch_size)
    n_opt_steps_per_epoch = max(1, usable_rows // batch_size)
    n_epochs = max(1, -(-HARNESS_TARGET_STEPS // n_opt_steps_per_epoch))  # ceil
    log(f"[model] {n_opt_steps_per_epoch} opt steps/epoch -> running up to "
        f"{n_epochs} epochs to reach {HARNESS_TARGET_STEPS} steps "
        f"(or {HARNESS_MAX_HOURS}h, whichever comes first)")

    rl_baselines = {src: 0.0 for src in t2t.STAGE1_SOURCES}
    recent_probs0 = []
    global_step = 0
    self_correct_p = coefs["self_correct_prob"]
    val_sample = min(len(val_rows), int(os.environ.get("HARNESS_VAL_SAMPLE", 200)))

    checkpoints = []  # (global_step, unique_pred_fraction)
    # Windowed means over each 50-step log interval. A single row's CE (the
    # previous logging) swung between ~2 and ~185 from one step to the next,
    # which made trends unreadable.
    win_ce, win_special0, win_eos0, win_n = 0.0, 0.0, 0.0, 0
    max_seconds = HARNESS_MAX_HOURS * 3600

    def time_up():
        return (time.time() - t_start) >= max_seconds

    model.train()
    for epoch in range(n_epochs):
        if global_step >= HARNESS_TARGET_STEPS or time_up():
            break
        order = list(range(len(train_rows)))
        random.Random(42 + epoch).shuffle(order)
        order = order[:usable_rows]
        rng = random.Random(42 * 1000 + epoch)

        accum = 0
        optimizer.zero_grad()

        for i in range(len(order)):
            if global_step >= HARNESS_TARGET_STEPS or time_up():
                break
            row = train_rows[order[i]]
            state, salience_logits = t2t.encode_to_state(model, row["input"], return_gate_logits=True)

            ramp_steps = coefs["anti_collapse_ramp_steps"]
            brier_w = t2t.scheduled_coef(global_step, ramp_steps,
                                         coefs["brier_coef_start"], coefs["brier_coef_end"])
            blank_penalty_w = t2t.scheduled_coef(global_step, ramp_steps,
                                                 coefs["blank_answer_penalty_coef_start"],
                                                 coefs["blank_answer_penalty_coef_end"])
            if BLANK_COEF_OVERRIDE is not None:
                blank_penalty_w = float(BLANK_COEF_OVERRIDE)
            repetition_penalty_w = t2t.scheduled_coef(global_step, ramp_steps,
                                                       coefs["repetition_penalty_coef_start"],
                                                       coefs["repetition_penalty_coef_end"])
            content_closeness_w = t2t.scheduled_coef(global_step, ramp_steps,
                                                      coefs["content_closeness_coef_start"],
                                                      coefs["content_closeness_coef_end"])
            supervised_loss, ce, probs0, pos0_correct = t2t.compute_task_loss(
                expert, state, row["target"], codec, device, coefs, hp["label_smoothing"],
                source=row["source"], scheduled_sampling_prob=coefs["scheduled_sampling_prob"],
                rng=rng, return_probs0=True, brier_coef=brier_w,
                blank_answer_penalty_coef=blank_penalty_w,
                repetition_penalty_coef=repetition_penalty_w,
                content_closeness_coef=content_closeness_w,
            )
            win_ce += ce
            win_special0 += probs0[[t2t.PAD_ID, t2t.BOS_ID, t2t.EOS_ID]].sum().item()
            win_eos0 += probs0[t2t.EOS_ID].item()
            win_n += 1

            if rng.random() < self_correct_p:
                supervised_loss = supervised_loss + coefs["self_correct_coef"] * t2t.self_correct_step(
                    model, expert, codec, device, row["input"], row["target"], coefs, rng,
                )

            # REAL_TOTAL_STEPS, not this harness's own total -- see module docstring.
            sup_w, rl_w = t2t._schedule_weights(global_step, REAL_TOTAL_STEPS, coefs)
            reward, logprob, entropy = t2t.t2t_rl_term(expert, state, row)
            source = row["source"]
            pg_loss = -(reward - rl_baselines[source]) * logprob - coefs["entropy_coef"] * entropy
            momentum = coefs["rl_baseline_momentum"]
            rl_baselines[source] = momentum * rl_baselines[source] + (1 - momentum) * reward

            gate_diversity = t2t.salience_diversity_loss(salience_logits)
            gate_margin = t2t.salience_margin_loss(salience_logits)
            decoder_div = (torch.zeros((), device=device) if pos0_correct
                          else t2t.decoder_diversity_loss(probs0, recent_probs0))

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
                # P2.2 (see text_to_text.py's decoder_diversity_contrastive_coef
                # comment) -- mirrors train_stage's own updated batch-boundary
                # block exactly: shares one encode_to_state call per row
                # between koleo_loss (pooled state) and the new contrastive
                # term (needs that same state fed through the decoder for
                # probs0), combined into a single backward() since r_state is
                # a shared upstream tensor for both branches.
                koleo_rows = [train_rows[order[j]] for j in range(i - batch_size + 1, i + 1)]
                pooled_states, pooled_memories, pooled_pools, batch_probs0, batch_gold0 = [], [], [], [], []
                for r in koleo_rows:
                    r_state, r_memory, r_combined = t2t.encode_to_state(
                        model, r["input"], return_encoder_memory=True)
                    pooled_states.append(r_state.mean(dim=1).squeeze(0))
                    pooled_memories.append(r_memory.mean(dim=1).squeeze(0))
                    pooled_pools.append(r_combined.mean(dim=1).squeeze(0))
                    r_max_len = gcfg["max_target_len_overrides"].get(r["source"])
                    r_tgt = codec.encode_target(r["target"], device, max_len=r_max_len)
                    r_logits, _ = t2t.teacher_forced_forward(expert, r_state, r_tgt[:, :-1])
                    batch_probs0.append(t2t.F.softmax(r_logits, dim=-1)[:, 0, :].squeeze(0))
                    batch_gold0.append(r_tgt[0, 1].item())
                contrastive = t2t.decoder_diversity_batch_contrastive_loss(
                    torch.stack(batch_probs0, dim=0), batch_gold0,
                    coefs["decoder_diversity_contrastive_temperature"])
                koleo = t2t.koleo_loss(torch.stack(pooled_states, dim=0))
                encoder_koleo = t2t.koleo_loss(torch.stack(pooled_memories, dim=0))
                pool_koleo = t2t.koleo_loss(torch.stack(pooled_pools, dim=0))
                batch_term_loss = (
                    coefs["koleo_coef"] * koleo
                    + coefs["encoder_koleo_coef"] * encoder_koleo
                    + coefs["pool_koleo_coef"] * pool_koleo
                    + coefs["decoder_diversity_contrastive_coef"] * contrastive
                )
                batch_term_loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum = 0
                global_step += 1

                if global_step % 50 == 0:
                    buffer_div = t2t._buffer_diversity(recent_probs0)
                    gates = t2t.gate_scale_summary(model)
                    cm_memory = t2t.common_mode_ratio(torch.stack(pooled_memories, dim=0))
                    cm_state = t2t.common_mode_ratio(torch.stack(pooled_states, dim=0))
                    elapsed_min = (time.time() - t_start) / 60
                    ce_avg, special0_avg, eos0_avg = win_ce / win_n, win_special0 / win_n, win_eos0 / win_n
                    win_ce, win_special0, win_eos0, win_n = 0.0, 0.0, 0.0, 0
                    log(f"  e{epoch} step{global_step}/{HARNESS_TARGET_STEPS} "
                        f"({elapsed_min:.1f}min elapsed) source={row['source']} ce_avg50={ce_avg:.4f} "
                        f"special0={special0_avg:.4f} eos0={eos0_avg:.4f} "
                        f"brier_w={brier_w:.4f} blank_w={blank_penalty_w:.4f} "
                        f"rep_w={repetition_penalty_w:.4f} cc_w={content_closeness_w:.4f} "
                        f"decoder_div={decoder_div.item():.4f} buffer_div={buffer_div:.4f} "
                        f"contrastive={contrastive.item():.4f} koleo={koleo.item():.4f} "
                        f"encoder_koleo={encoder_koleo.item():.4f} "
                        f"pool_koleo={pool_koleo.item():.4f} "
                        f"cm_memory={cm_memory:.3f} cm_state={cm_state:.3f} "
                        f"gate_lin_attn={gates['linear_attn_mean']:.4f} "
                        f"gate_lin_ffn={gates['linear_ffn_mean']:.4f} "
                        f"gate_full_attn={gates['full_attn_mean']:.4f} "
                        f"gate_full_ffn={gates['full_ffn_mean']:.4f} "
                        f"gate_pool_ffn={gates['pool_ffn']:.4f} "
                        f"gate_router={gates['router_state']:.4f}")

                if global_step % EVAL_EVERY == 0:
                    val = t2t.validate(model, expert, codec, device, val_rows, coefs, val_sample)
                    (rewards, unique_pred_fraction, most_common_pred_fraction, precision_recall,
                     state_similarity) = t2t.evaluate_rewards(
                        model, expert, val_rows, sample_size=val_sample, seed=global_step)
                    gate = t2t.gate_health(model, val_rows)
                    model.eval()
                    with torch.no_grad():
                        pos0_rows = t2t.sample_rows(val_rows, min(100, len(val_rows)), seed=global_step)
                        pos0_states = [t2t.encode_to_state(model, r["input"]) for r in pos0_rows]
                        pos0 = t2t.pos0_diagnostics(expert, pos0_states)
                    model.train()
                    checkpoints.append((global_step, unique_pred_fraction))
                    result = {
                        "global_step": global_step, "val_loss": val,
                        "unique_pred_fraction": unique_pred_fraction,
                        "most_common_pred_fraction": most_common_pred_fraction,
                        "rewards": rewards, "precision_recall": precision_recall,
                        "gate": gate, "state_similarity": state_similarity,
                        "pos0": pos0,
                        "elapsed_min": (time.time() - t_start) / 60,
                    }
                    with open(RESULTS_PATH, "a") as f:
                        f.write(json.dumps(result) + "\n")
                    log(f"  [VAL] step={global_step} val_loss={val:.4f} "
                        f"unique_pred_fraction={unique_pred_fraction:.3f} "
                        f"most_common_pred_fraction={most_common_pred_fraction:.3f} rewards={rewards} "
                        f"gate_salience_mean={gate.get('salience_mean', float('nan')):.4f} "
                        f"state_sim_within={state_similarity['within_source']:.4f} "
                        f"state_sim_across={state_similarity['across_source']:.4f} "
                        f"pos0_top={pos0['top_ids']} pos0_mean_p={pos0['mean_p']} "
                        f"pos0_margin={pos0['margin_mean']} pos0_row_std={pos0['row_std_top']} "
                        f"xattn_ratio={pos0['xattn_ratio']}")

                    if len(checkpoints) >= 5:
                        last5 = checkpoints[-5:]
                        if all(c[1] > 0.02 for c in last5):  # well above the 0.003-0.005 collapse floor
                            log(f"  >>> PASS SIGNAL: unique_pred_fraction > 0.02 for the last "
                                f"{len(last5)} consecutive checkpoints (steps "
                                f"{last5[0][0]}-{last5[-1][0]})")

    stop_reason = "step target reached" if global_step >= HARNESS_TARGET_STEPS else (
        "time budget reached" if time_up() else "data exhausted")
    log(f"\n[DONE] ({stop_reason}) {global_step} steps in "
        f"{(time.time()-t_start)/60:.1f} min. {len(checkpoints)} validation checkpoints recorded.")
    log(f"unique_pred_fraction trajectory: {[round(c[1], 3) for c in checkpoints]}")
    last10 = checkpoints[-10:]
    n_recovered = sum(1 for c in last10 if c[1] > 0.02)
    log(f"Final {len(last10)} checkpoints: {n_recovered}/{len(last10)} above 0.02 "
        f"(pass criterion: consecutive recovery, see log above for exact streak)")
    log(f"Full results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
