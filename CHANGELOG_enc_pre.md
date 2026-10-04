# Changelog: enc_pre.py

Covers the vision SSL pretraining (`vision_ssl_pretrain` stage) and
dataset-loading logging fixes made during this investigation. This is a
separate thread from the `text_to_text.py` collapse investigation (see
`CHANGELOG_text_to_text.md`/`CHANGELOG_EncCombine.md`/
`CHANGELOG_Logic.md`) — unrelated code path, unrelated failure mode, just
discovered in the same conversation while reviewing a concurrently-running
`enc_pre.sh` job. Dataset-provenance work on this file (GeoQA/CLEVR/GQA
source replacements, proxy-blocking workarounds) predates this changelog
and is already documented in `DATASET_LOADERS.md`/`GQA_IMAGES_SETUP.md` —
not duplicated here.

## Round 1 — logging clarity: silent cache hits looked like network fallback

**Why**: reviewing `enc_pre_slurm-4795027.out` (mid `vision_ssl_pretrain`),
the log's first visible SSL-data line was `imagenet21k`'s "no local pool
yet -- materializing from zeyuanyin/imagenet-21k-p now" — with *nothing*
printed for `imagenet1k` first, even though `SSLImageStream` processes
sources in order (`imagenet1k` before `imagenet21k`) and `imagenet1k`'s
large pool had deliberately been pre-extracted onto `scratch` ahead of
time specifically to avoid a live download. The actual data routing was
correct the whole time; the "already cached" fast path in
`_materialize_image_pool`/`_materialize_inaturalist_pool`/`Task.build()`
simply had zero print statements, so a cache hit and "did nothing" looked
identical in the log, while a live download printed several lines —
making a run that mixed both (some sources pre-cached, some intentionally
left to materialize live, e.g. `imagenet21k`'s own ~15-minute cost was
always meant to be paid inline) look like *everything* was hitting the
network.

**Change**: added a `[ssl-data] <pool_name>: found existing local pool ->
<path> (skipping <source>)` print to `_materialize_image_pool`'s and
`_materialize_inaturalist_pool`'s cache-hit fast paths, and a `[data]
<name>: found existing local cache -> <path>, skipping download()` print
to `Task.build()`'s equivalent (used later by `image_tune`/`image_train`'s
`recon`/`depth`/`instance`/`vrd`/`semseg` tasks — same ambiguity existed
there, just not yet visibly triggered since that stage hadn't run yet).

**Verified**: compiled clean; no behavior change (purely additive
logging) confirmed by inspection — the cache-hit `return` paths themselves
are untouched, just preceded by a `print`.

## Round 2 — `dino_sharpness_loss`: DINO loss stuck at the uniform-distribution ceiling

**Evidence**: the same `enc_pre_slurm-4795027.out` review showed `dino`
pinned at 9.0107-9.0110 from roughly iteration 25,000 onward, still pinned
at iteration 89,100 (out of a 200,000-iteration target) when checked.
`log(dino_out_dim) = log(8192) = 9.0109` — an exact match. That's not a
coincidence: `dino_loss`'s cross-entropy
`-(tp.detach() * log_p).sum(dim=-1)` reduces to exactly `log(K)`
*regardless of the teacher's target `tp`* once the student's own
`log_p = log_softmax(sl / student_temp)` is uniform — so a value pinned
at exactly that number is a direct signature of the student's distribution
having gone uniform, for every sample, independent of what the teacher
says. `ibot_loss` — same `ProjectionHead` architecture, same `cfg`
(`student_temp`/`sinkhorn_iters`/`sinkhorn_epsilon` all shared) — was
*not* stuck (dropped to 4.22 at one point, later drifted back up to
~4.8-4.9), ruling out a shared hyperparameter typo and pointing
specifically at the DINO/CLS-token path.

**The subtlety that made this non-obvious**: `koleo_loss` is computed on
this exact same tensor (`student_cls_logits[0]`, the DINO head's raw
output) and stayed healthy (-0.28ish) the whole time — which looks like
it should rule out "the head's output is degenerate," but doesn't.
`koleo_loss` only measures whether *different samples'* logit vectors
point in different directions after L2 normalization (inter-sample
diversity). It says nothing about whether *one sample's own* 8192 logit
values have enough spread across themselves to produce a non-uniform
softmax (intra-sample sharpness) — an entirely independent property, and
one nothing in this file was previously instrumenting or regularizing.
Working hypothesis for the mechanism: `ProjectionHead.last_layer` (a
`Linear(256, 8192, bias=False)` applied to an L2-normalized bottleneck)
has likely collapsed to low effective rank — its 8192 output rows close
enough together that any input produces nearly the same value across all
8192 dimensions, even though different inputs still land in different
enough *directions* to pass koleo's check.

**Change**: added `dino_sharpness_loss(student_logits, cfg)` — mean
per-sample entropy of `softmax(logits / student_temp)` across every crop,
the direct within-sample counterpart to koleo's between-sample repulsion.
Minimizing (not maximizing) entropy, scoped to the DINO/CLS path only
(ibot never showed this plateau, so nothing points at it needing the same
treatment). Wired into `ssl_train_step` via a new
`HYPERPARAMS["ssl"]["sharpness_weight"] = 0.5` (entropy is on the same
0..log(K) scale as `dino_loss`/`ibot_loss` themselves, unlike `koleo_weight`
which needed its 0.1 down-weight for a different-scale term — no prior
tuning data for this exact term, revisit once a verification run's own
`sharpness` metric comes back). Added to `ssl_train_step`'s `metrics`
dict, which automatically flows into the existing generic per-iteration
print/wandb_log loop (`msg = " ".join(f"{k}={v:.4f}" for k, v in
metrics.items())`) with no further wiring needed there.

**Verified**: locally, with a synthetic test (no project dependencies
needed, just torch) — near-uniform logits (std 0.001, simulating the
hypothesized collapsed `last_layer` rows) give entropy 9.0108, matching
the observed plateau almost exactly; genuinely spread-out logits give
0.22; a near-uniform input still has a small but nonzero gradient
(0.00055 norm), confirming the degenerate point isn't an exact
zero-gradient trap this term can't escape.

**Not yet verified on a real training run** (would need a real GPU +
real ImageNet-scale data; no cheap CPU-only harness equivalent exists yet
for this pipeline the way `collapse_fix_harness.py` exists for
`text_to_text.py`). The `vision_ssl_pretrain` job keeps running toward its
200k-iteration auto-review point regardless of this fix landing or not;
`ibot`/`koleo`/`gram` likely still contribute useful learning even with
`dino` stuck, so the run isn't pure waste either way.

## Status as of this writing

Round 1 (logging) is low-risk and confirmed correct by inspection. Round 2
(`dino_sharpness_loss`) is a reasoned, evidence-based hypothesis fix,
verified only via synthetic unit test — the real test is whether `dino`
actually breaks away from the `log(8192)` ceiling on the next real
`vision_ssl_pretrain` run (or resubmission) that picks up this code
change.
