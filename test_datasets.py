"""
Standalone health check for every dataset loader across the *_to_text.py
training scripts (text_to_text.py/image_to_text.py/reasoning_to_text.py/
ocr_to_text.py) -- calls each SOURCE_LOADERS entry (capped small, so this
stays fast/cheap regardless of a source's real, uncapped production size)
and reports which ones fail, without running any of the full compile/train
pipelines those scripts normally do. Meant to be run standalone on the
cluster (see test_datasets.sh) whenever a new source is added or an
existing one's upstream (HF dataset schema, availability, etc.) might have
drifted -- exactly the kind of thing compile_dataset()'s own per-source
try/except already tolerates silently, this just surfaces it directly and
on demand, without needing a full compile run first.

Also covers enc_pre.py's image-encoder training step (SSL pretraining
pools + the downstream probe tasks that evaluate the pretrained encoder --
see test_enc_pre_image_sources) -- deliberately NOT via those sources' own
download()/_materialize_*() entry points, which have real persistent side
effects (see that function's docstring for why), but via the exact same
underlying load_dataset(...) call each one makes internally.

Sources with an already-root-caused, known-broken upstream id (see
KNOWN_PENDING_REPLACEMENT) are [SKIP]ped rather than tested -- they need a
dataset swap, not a retest, and re-running them every time would just
reproduce the same already-understood failure and clutter the "doesn't
work" list with issues that aren't new regressions. Everything else is
tested normally, so the "doesn't work" list only ever shows genuinely new
or unexplained breakage.

--cap (default 10) is a TEST-ONLY row limit, kept small purely so this
script stays fast/cheap -- it has no bearing on real training, which uses
each script's own HYPERPARAMS caps (typically far larger, or None/"the
entire split"). See test_enc_pre_precompute_cost for why that gap matters:
enc_pre.py's image sources are fully materialized to local disk ONCE,
up front, before any real training step runs (see
_materialize_image_pool/Task.build's docstrings) -- at production-scale
caps (500,000 images, or an uncapped ~1.28M-image split for imagenet1k),
that one-time cost can run into hours, which matters against this
project's 48-hour SLURM limit. test_enc_pre_precompute_cost measures real
per-row throughput on a modest sample and extrapolates the production-scale
cost, to answer "does this source need to be pre-downloaded as a separate
job before a real training run, or is it fast enough to materialize inline
at the start of one?"

Every check validates real OUTPUT, not just "did a row arrive": an
image-bearing row is actually decoded via the same .convert("RGB") call
every real consumer uses (_validate_image), so a present-but-corrupt image
field is caught instead of silently passing. Every failure is also
classified as ACCESS_DENIED (a gated/unauthorized HF Hub response --
printed as [FAIL-ACCESS-DENIED], distinct remediation: accept that
dataset's terms + use a token from that account) vs. a generic FAIL (schema
drift, network, missing dependency, etc.) -- see _classify_error. This
matters because IMAGENET_HF ("ILSVRC/imagenet-1k") is a genuinely gated
dataset real training depends on; see _classify_error's docstring for the
historical incident (slurm-4631354) this guards against.

Usage: python3 test_datasets.py [--token HF_TOKEN] [--cap N] [--precompute-sample N]
"""
import argparse
import copy
import importlib
import os
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

# (module name, human label) -- every *_to_text.py script that defines a
# SOURCE_LOADERS dict. ocr_to_text.py is handled separately below (its
# single source, load_docvqa, takes a plain cap argument directly rather
# than a SOURCE_LOADERS entry -- see its own module docstring/compile_dataset).
SOURCE_LOADER_MODULES = ["text_to_text", "image_to_text", "reasoning_to_text"]

# Formerly here: geoqa/clevr/vrd, each blocked on a broken or unverified HF
# mirror id. All three were moved to their ORIGINAL official academic
# sources instead (Geometry3K for geoqa, real CLEVR v1.0 for clevr, real
# GQA for vrd -- see image_to_text.py/reasoning_to_text.py/enc_pre.py's
# updated loader comments for the full provenance rationale), so this dict
# is empty for now. Re-add an entry here if a source becomes broken again.
KNOWN_PENDING_REPLACEMENT = {}

# Sources that now do a real, sizable ONE-TIME download on first call
# (clevr: 18GB; enc_pre.vrd: ~20GB images) rather than a capped HF stream --
# same "cache once, reuse forever" pattern enc_pre.py's SSL pools already
# use (see _materialize_image_pool), just extended to these three. Running
# the normal loader here would trigger that full download just to check
# "does this work", defeating this script's "fast, cheap health check"
# purpose -- these get a lightweight reachability check instead (HEAD
# request against the real download URL) unless --full-download-test is
# passed. geoqa (Geometry3K) is NOT in this set -- confirmed small (~100MB
# total across train/val/test), cheap enough to test for real every time.
LARGE_DOWNLOAD_SOURCES = {
    ("image_to_text", "clevr"): "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
    ("enc_pre", "vrd (probe task)"): "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
}


def parse_args():
    p = argparse.ArgumentParser(description="Test every dataset loader across the *_to_text.py scripts.")
    p.add_argument("--token", default="_", help="Hugging Face token (needed for any gated datasets).")
    p.add_argument("--cap", type=int, default=10,
                   help="Row cap passed to every loader -- kept small so this stays fast.")
    p.add_argument("--precompute-sample", type=int, default=25,
                   help="Rows sampled per enc_pre.py image source to measure real "
                        "throughput and extrapolate production-scale precompute cost "
                        "(see test_enc_pre_precompute_cost). Larger = more accurate "
                        "estimate but slower to run.")
    p.add_argument("--full-download-test", action="store_true",
                   help="Actually run LARGE_DOWNLOAD_SOURCES' real loaders (clevr, "
                        "enc_pre.vrd) instead of just a HEAD-request reachability "
                        "check -- triggers their real one-time multi-GB download. "
                        "Off by default so a routine test run stays fast/cheap.")
    return p.parse_args()


def _check_url_reachable(url):
    """Lightweight HEAD request -- confirms a large-download source's URL
    still resolves without pulling the multi-GB body. Returns
    (ok, detail_str)."""
    import requests

    try:
        r = requests.head(url, timeout=30, allow_redirects=True)
        if r.status_code >= 400:
            return False, f"HTTP {r.status_code} from {url}"
        size = r.headers.get("content-length")
        size_str = f", {int(size) / 1e9:.1f}GB" if size else ""
        return True, f"reachable (HTTP {r.status_code}{size_str}) -- not downloaded " \
                     f"(pass --full-download-test to actually fetch+parse)"
    except Exception as e:
        return False, f"unreachable: {e}"


def _fmt_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"


# Signatures of a gated/access-denied HF Hub response, as opposed to a
# generic schema/network failure -- distinguished because the remediation is
# completely different (accept the dataset's terms + use a token from that
# account, vs. investigate a schema drift or connectivity issue). Historical
# motivation: IMAGENET_HF ("ILSVRC/imagenet-1k") is a genuinely gated
# dataset (real training already depends on token access to it working --
# see _materialize_image_pool's callers); a past real run
# (slurm-4631354.out/.err) died materializing it, though the actual root
# cause there turned out to be a distributed-training socket timeout from
# the uncapped materialization taking too long against the (at-the-time)
# 30-minute default process-group timeout, not a literal access-denied
# response -- dist_utils.py's _PROCESS_GROUP_TIMEOUT=24h (committed after
# that run) already addresses that specific failure mode. This classifier
# exists as a second, independent safeguard: a gated dataset can always
# start rejecting requests later (token expiring/being revoked, account
# losing access), and that failure needs to be immediately obvious rather
# than looking like just another generic FAIL.
# Checked BEFORE _ACCESS_DENIED_SIGNATURES below -- confirmed via a real
# run (slurm-4781715) that the cluster's own outbound proxy actively
# filters certain external hosts (doc-*-docs.googleusercontent.com for
# Google Drive's large-file redirect, downloads.cs.stanford.edu for GQA;
# dl.fbaipublicfiles.com for CLEVR was NOT filtered in that same run, so
# this is a per-host policy, not a blanket block) with
# `ProxyError('Unable to connect to proxy', OSError('Tunnel connection
# failed: 403 Filtered'))`. That message contains "403", which would
# otherwise match _ACCESS_DENIED_SIGNATURES and mislabel a network-policy
# block as a dataset-side access denial -- a completely different
# remediation (scp the file from an unfiltered machine, vs. accept terms/
# use a different token). Checked first since "proxyerror"/"tunnel
# connection failed" are unambiguous signals a generic "401"/"403" isn't.
_PROXY_BLOCKED_SIGNATURES = (
    "proxyerror", "tunnel connection failed", "unable to connect to proxy",
)

_ACCESS_DENIED_SIGNATURES = (
    "gated", "403", "401", "unauthorized", "permission", "access to model",
    "access to dataset", "you need to accept", "restricted", "authentication",
    "please log in", "gatedrepoerror", "not authorized",
)


def _classify_error(e):
    msg = str(e).lower()
    if any(sig in msg for sig in _PROXY_BLOCKED_SIGNATURES):
        return "PROXY_BLOCKED"
    if any(sig in msg for sig in _ACCESS_DENIED_SIGNATURES):
        return "ACCESS_DENIED"
    return "FAIL"


def _error_tag(e, fail_label="FAIL"):
    """Consistent printed tag for an exception at every call site in this
    file -- distinguishes a genuine access-denial from this cluster's own
    proxy blocking the request (see _classify_error's PROXY_BLOCKED
    comment), since the remediation is completely different (scp the file
    from an unfiltered machine, vs. accept the dataset's terms/use a
    different token)."""
    category = _classify_error(e)
    if category == "PROXY_BLOCKED":
        return f"{fail_label}-PROXY-BLOCKED"
    if category == "ACCESS_DENIED":
        return f"{fail_label}-ACCESS-DENIED"
    return fail_label


def _validate_image(img):
    """Mirrors the exact real downstream decode step every image consumer
    in this codebase uses (enc_pre.py's and image_to_text.py's identical
    _load_image_tensor: `img.convert("RGB").resize(...)`) -- confirms a
    sampled row's image is an actually-decodable, non-degenerate PIL image,
    not just that the row's image field is non-None. Returns
    (ok, detail_str)."""
    try:
        rgb = img.convert("RGB")
        w, h = rgb.size
        if w <= 0 or h <= 0:
            return False, f"decoded but degenerate size {w}x{h}"
        return True, f"{w}x{h} mode={img.mode}"
    except Exception as e:
        return False, f"image failed to decode: {e}"


def _extract_image_field(ex):
    """Same fallback chain _materialize_image_pool uses to pull a usable
    PIL image out of a raw HF example dict -- see that function's docstring.
    Returns a PIL Image or None (never raises)."""
    from PIL import Image as PILImage

    img = ex.get("image")
    if img is not None:
        return img
    for key in ("jpeg", "jpg", "png"):
        raw = ex.get(key)
        if raw is None:
            continue
        if isinstance(raw, PILImage.Image):
            return raw
        if isinstance(raw, (bytes, bytearray)):
            from io import BytesIO
            return PILImage.open(BytesIO(raw))
        try:
            return PILImage.open(raw)   # file path or file-like object
        except Exception:
            return None
    return None


def capped_config(hyperparams, cap):
    """Deep-copies HYPERPARAMS and overrides every *_cap entry in ["data"]
    to `cap` -- generic across scripts (works regardless of exactly which
    cap keys a given script defines) rather than hardcoding each one."""
    cfg = copy.deepcopy(hyperparams)
    for k in cfg.get("data", {}):
        cfg["data"][k] = cap
    return cfg


def test_enc_pre_image_sources(cap, results, full_download_test=False):
    """enc_pre.py's image-encoder training step: the SSL pretraining pools
    (imagenet1k/imagenet21k/pass/inaturalist, see SSLImageStream) plus the
    downstream probe tasks (recon/depth/instance/vrd/semseg, see the Task
    subclasses) that evaluate the pretrained encoder.

    Deliberately does NOT call _materialize_image_pool/
    _materialize_inaturalist_pool or any Task.download() directly -- those
    have real, persistent side effects this health check must not trigger:
    _materialize_image_pool permanently caches whatever it downloads under
    DATA_DIR/<pool_name>, and any later real SSL run reuses that cache
    as-is (see its own docstring), so calling it here with a tiny test cap
    would silently poison real training data with a near-empty pool;
    VRDTask.download() overwrites the shared gqa_predicates.json vocabulary
    file as a side effect. Every check here instead reissues the exact same
    load_dataset(...) call each source uses internally (same HF id/split/
    streaming/trust_remote_code), reading only a few examples through HF's
    own (idempotent, shared, safe-to-reuse) download cache -- confirming
    the dataset is reachable and correctly shaped without writing anything
    persistent. recon/instance share imagenet1k's exact source (IMAGENET_HF)
    -- tested once, not three times."""
    print("\n===== enc_pre.py (image encoder training) =====")
    enc_pre = importlib.import_module("enc_pre")
    from datasets import load_dataset

    def check(label, hf_id, config_name=None, depth_field=None, **kwargs):
        """Streams `cap` rows (same call every real source uses internally)
        AND validates the actual output content the same way the real
        training-stage consumer would: extracts+decodes the image field via
        the exact _materialize_image_pool fallback chain (not just "the row
        arrived"), and -- for depth -- also validates `depth_field`. This is
        what actually answers "is the output fine", not just "did it not
        crash"."""
        try:
            ds = load_dataset(hf_id, name=config_name, split="train", streaming=True,
                              cache_dir=enc_pre.HF_CACHE, **kwargs)
            n, first_ex = 0, None
            for ex in ds:
                if first_ex is None:
                    first_ex = ex
                n += 1
                if n >= cap:
                    break
            if n == 0:
                detail = "0 rows streamed"
                print(f"  [EMPTY] {label}: {detail}")
                results.append((f"enc_pre.{label}", False, detail))
                return

            img = _extract_image_field(first_ex)
            if img is None:
                ok, img_detail = False, "no decodable image field found on a sampled row"
            else:
                ok, img_detail = _validate_image(img)
            detail = f"{n} row(s), sample image: {img_detail}"

            if ok and depth_field:
                dm = first_ex.get(depth_field)
                if dm is None:
                    ok, detail = False, detail + f" | '{depth_field}' field missing"
                else:
                    ok, dm_detail = _validate_image(dm) if hasattr(dm, "convert") else (True, f"{type(dm).__name__}")
                    detail = detail + f" | {depth_field}: {dm_detail}"

            print(f"  [{'OK' if ok else 'FAIL'}] {label}: {detail}")
            results.append((f"enc_pre.{label}", ok, detail))
        except Exception as e:
            print(f"  [{_error_tag(e)}] {label}: {e}")
            results.append((f"enc_pre.{label}", False, str(e)))

    check("imagenet1k (SSL pool; also covers recon/instance probe tasks, same source) "
          "[GATED -- requires accepted terms at huggingface.co/datasets/ILSVRC/imagenet-1k "
          "+ a --token from that account]",
          enc_pre.IMAGENET_HF)
    check("imagenet21k (SSL pool)", enc_pre.IMAGENET21K_HF)
    # pass/depth: name="default" -- see _materialize_image_pool's and
    # DepthTask.download's docstrings for why a script-based source needs
    # this passed explicitly (works around a UnicodeDecodeError bug in
    # datasets 2.19.2's own config-probing code, confirmed via direct
    # traceback, not a schema/network issue). depth also needs h5py
    # installed in the environment -- confirmed missing from mulEnv as of
    # this fix, so depth will still [FAIL] with an ImportError until that's
    # added; that failure is expected/actionable, not a code bug.
    check("pass (SSL pool)", enc_pre.PASS_HF, config_name="default")
    check("depth (probe task)", enc_pre.NYU_DEPTH_HF, config_name="default", depth_field="depth_map")
    check("semseg (probe task)", enc_pre.ADE20K_HF, trust_remote_code=True)

    # vrd (probe task): now GQA's official release (see VRDTask.download's
    # comment for the full provenance rationale -- replaces the earlier
    # broken Voxel51/GQA-Scene-Graph HF mirror). Real download is ~20GB
    # (images) + 42.7MB (scene graphs) -- see LARGE_DOWNLOAD_SOURCES --
    # so this only does a HEAD-request reachability check by default,
    # same as image_to_text.clevr.
    label = "vrd (probe task)"
    if not full_download_test:
        url = LARGE_DOWNLOAD_SOURCES[("enc_pre", label)]
        ok, detail = _check_url_reachable(url)
        print(f"  [{'OK' if ok else 'FAIL'}] {label}: {detail}")
        results.append((f"enc_pre.{label}", ok, detail))
    else:
        # Unlike every other check in this function, --full-download-test
        # DOES call Task.download() directly here (writes DATA_DIR/
        # gqa_predicates.json as a side effect) -- an intentional, opt-in
        # exception to this function's usual side-effect-free design, since
        # verifying the new GQA download+parse pipeline for real requires
        # actually exercising VRDTask's own code path, not a reimplemented
        # probe of it. Unlike the streaming-HF-source precompute estimator
        # (test_enc_pre_precompute_cost), GQA's download cost is FIXED --
        # the whole ~20GB zip has to come down regardless of cap -- so the
        # measured wall-clock time here IS the real one-time cost directly,
        # no extrapolation needed.
        try:
            t0 = time.time()
            task = enc_pre.VRDTask()
            ds = task.download(cap)
            elapsed_s = time.time() - t0
            n = len(ds)
            ok = n > 0
            detail = f"{n} row(s) in {_fmt_duration(elapsed_s)}" if ok else "0 rows"
            if ok:
                img_ok, img_detail = _validate_image(ds[0]["image"])
                ok = img_ok
                detail += f", sample image: {img_detail}, predicate={ds[0]['predicate']!r}"
                verdict = ("NEEDS EXTERNAL PRECOMPUTE" if elapsed_s > PRECOMPUTE_THRESHOLD_S
                          else "fine to materialize inline")
                detail += f" >>> {verdict}"
            print(f"  [{'OK' if ok else 'FAIL'}] {label}: {detail}")
            results.append((f"enc_pre.{label}", ok, detail))
        except Exception as e:
            print(f"  [{_error_tag(e)}] {label}: {e}")
            results.append((f"enc_pre.{label}", False, str(e)))

    # inaturalist (SSL pool): already-downloaded local directory tree, not
    # an HF dataset (see INATURALIST_ROOT's definition) -- no network call,
    # just confirm it's actually there and non-empty.
    label = "inaturalist (SSL pool)"
    path = enc_pre.INATURALIST_ROOT
    entries = os.listdir(path) if os.path.isdir(path) else []
    ok = len(entries) > 0
    detail = f"{len(entries)} entries" if ok else f"missing or empty: {path}"
    print(f"  [{'OK' if ok else 'FAIL'}] {label}: {detail}")
    results.append((f"enc_pre.{label}", ok, detail))


# Above this, a source's estimated one-time production-scale materialization
# cost is treated as "needs a separate precompute job" -- chosen as a
# fraction of the 48-hour SLURM limit these jobs run under (see dist_utils.py's
# _PROCESS_GROUP_TIMEOUT docstring for that same limit's other implications);
# an inline materialization anywhere near this long eats meaningfully into
# the real training budget for a run that hasn't trained a single step yet.
PRECOMPUTE_THRESHOLD_S = 1800  # 30 min

# ImageNet-1k's train split size is public/well-documented (ILSVRC2012) and
# used only when a source's production cap is None ("the whole split") and
# we have no cheaper way to know its size without downloading it -- purely
# informational, not measured by this script.
KNOWN_FULL_SPLIT_SIZES = {"imagenet1k (SSL pool)": 1_281_167}


def test_enc_pre_precompute_cost(sample_size, enc_pre):
    """Answers "does this enc_pre.py image source need to be pre-downloaded
    as a separate job before a real training run, or is it fast enough to
    materialize inline at the start of one?" -- see this module's docstring
    for why that gap matters (production caps are 500,000/uncapped, not
    this script's own tiny --cap).

    Measures real wall-clock: time-to-first-row (connection/resolution
    setup cost -- confirmed elsewhere in this investigation to sometimes be
    the dominant cost, e.g. a raw-loose-file HF repo needing to enumerate a
    huge file listing before streaming starts at all) and per-row streaming
    time over `sample_size` further rows, then extrapolates linearly out to
    each source's REAL production cap (pulled directly from
    enc_pre.HYPERPARAMS, not this script's own capped_config) to estimate
    the total one-time cost _materialize_image_pool/Task.build would
    actually pay on a real run's first invocation.

    This is an ESTIMATE, not a guarantee -- HF network throughput varies
    (rate limiting, proxy conditions, time of day), and per-row cost is
    rarely perfectly uniform across an entire split. Treat the verdict as
    "clearly fine" vs. "clearly needs precomputing separately", not as a
    precise runtime prediction. Skips sources already in
    KNOWN_PENDING_REPLACEMENT (no point costing out a source being
    replaced) and inaturalist (already-local directory read, not a network
    download -- see its check above, effectively instant regardless of cap)."""
    print(f"\n===== enc_pre.py precompute-cost estimate (sample={sample_size} rows, "
          f"extrapolated to production caps) =====")
    from datasets import load_dataset

    ssl_caps = enc_pre.HYPERPARAMS["ssl"]
    image_cap = enc_pre.HYPERPARAMS["image"]["per_task_cap"]
    # (label, hf_id, load_dataset kwargs, real production cap [None = full split])
    sources = [
        ("imagenet1k (SSL pool)", enc_pre.IMAGENET_HF, {}, ssl_caps["imagenet1k_pool_cap"]),
        ("imagenet21k (SSL pool)", enc_pre.IMAGENET21K_HF, {}, ssl_caps["imagenet21k_pool_cap"]),
        ("pass (SSL pool)", enc_pre.PASS_HF, {"name": "default"}, ssl_caps["pass_pool_cap"]),
        ("depth (probe task)", enc_pre.NYU_DEPTH_HF, {"name": "default"}, image_cap),
        ("semseg (probe task)", enc_pre.ADE20K_HF, {"trust_remote_code": True}, image_cap),
    ]

    for label, hf_id, kwargs, real_cap in sources:
        if ("enc_pre", label) in KNOWN_PENDING_REPLACEMENT:
            continue
        try:
            t_start = time.time()
            it = iter(load_dataset(hf_id, split="train", streaming=True,
                                   cache_dir=enc_pre.HF_CACHE, **kwargs))
            next(it)
            setup_s = time.time() - t_start
            n, t_sample_start = 1, time.time()
            for _ in it:
                n += 1
                if n >= sample_size:
                    break
            per_row_s = (time.time() - t_sample_start) / max(1, n - 1) if n > 1 else 0.0

            total_cap = real_cap if real_cap is not None else KNOWN_FULL_SPLIT_SIZES.get(label)
            estimated_s = setup_s + max(0, (total_cap or 0) - 1) * per_row_s
            cap_desc = (f"{real_cap:,}" if real_cap is not None else
                       (f"uncapped (~{total_cap:,} known full-split size)" if total_cap
                        else "uncapped, unknown full-split size"))
            if total_cap is not None:
                verdict = ("NEEDS EXTERNAL PRECOMPUTE" if estimated_s > PRECOMPUTE_THRESHOLD_S
                          else "fine to materialize inline")
                est_desc = _fmt_duration(estimated_s)
            else:
                verdict = "NEEDS EXTERNAL PRECOMPUTE (cautious default -- unknown split size)"
                est_desc = "unknown"
            print(f"  {label}: setup={setup_s:.1f}s, {per_row_s * 1000:.0f}ms/row "
                 f"(sampled {n} rows) | production cap: {cap_desc} -> "
                 f"estimated one-time cost: {est_desc} >>> {verdict}")
        except Exception as e:
            category = _classify_error(e)
            tag = {"ACCESS_DENIED": "ACCESS-DENIED", "PROXY_BLOCKED": "PROXY-BLOCKED"}.get(
                category, "could not measure")
            print(f"  {label}: {tag} ({e})")


def main():
    args = parse_args()
    if args.token and args.token != "_":
        try:
            from huggingface_hub import login
            login(token=args.token)
        except Exception as e:
            print(f"[hf] login skipped: {e}")

    results = []  # (label, ok: bool, detail: str)

    for modname in SOURCE_LOADER_MODULES:
        print(f"\n===== {modname} =====")
        mod = importlib.import_module(modname)
        cfg = capped_config(mod.HYPERPARAMS, args.cap)
        for source, loader in mod.SOURCE_LOADERS.items():
            label = f"{modname}.{source}"
            if (modname, source) in KNOWN_PENDING_REPLACEMENT:
                reason = KNOWN_PENDING_REPLACEMENT[(modname, source)]
                print(f"  [SKIP] {source}: pending replacement -- {reason}")
                continue
            if (modname, source) in LARGE_DOWNLOAD_SOURCES and not args.full_download_test:
                url = LARGE_DOWNLOAD_SOURCES[(modname, source)]
                ok, detail = _check_url_reachable(url)
                print(f"  [{'OK' if ok else 'FAIL'}] {source}: {detail}")
                results.append((label, ok, detail))
                continue
            is_large_download = (modname, source) in LARGE_DOWNLOAD_SOURCES
            try:
                t0 = time.time()
                rows = loader(cfg)
                elapsed_s = time.time() - t0
                n = len(rows)
                if n == 0:
                    detail = "loader returned 0 rows"
                    print(f"  [EMPTY] {source}: {detail}")
                    results.append((label, False, detail))
                    continue
                # Validate the actual first row's content the same way the
                # real training-stage consumer would, not just "n > 0" --
                # every SOURCE_LOADERS row already passed _mkrow's own
                # non-empty input/target check by construction, so the
                # remaining thing worth confirming here is an image field
                # (when the row has one) actually decodes, since a row with
                # a present-but-corrupt image would otherwise pass silently.
                r0 = rows[0]
                ok, detail = True, f"{n} row(s)"
                if r0.get("image") is not None:
                    img_ok, img_detail = _validate_image(r0["image"])
                    ok = img_ok
                    detail += f", sample image: {img_detail}"
                if is_large_download and ok:
                    # Download cost is FIXED (the whole zip comes down
                    # regardless of cap) -- this wall-clock time IS the real
                    # one-time cost, no extrapolation needed (unlike
                    # test_enc_pre_precompute_cost's streaming-HF sources).
                    detail += f" | total time: {_fmt_duration(elapsed_s)}"
                    verdict = ("NEEDS EXTERNAL PRECOMPUTE" if elapsed_s > PRECOMPUTE_THRESHOLD_S
                              else "fine to materialize inline")
                    detail += f" >>> {verdict}"
                print(f"  [{'OK' if ok else 'FAIL'}] {source}: {detail}")
                results.append((label, ok, detail))
            except Exception as e:
                print(f"  [{_error_tag(e)}] {source}: {e}")
                results.append((label, False, str(e)))

    print("\n===== ocr_to_text =====")
    try:
        ocr_to_text = importlib.import_module("ocr_to_text")
        rows = ocr_to_text.load_docvqa(args.cap)
        n = len(rows)
        if n == 0:
            detail = "loader returned 0 rows"
            print(f"  [EMPTY] docvqa: {detail}")
            results.append(("ocr_to_text.docvqa", False, detail))
        else:
            r0 = rows[0]
            ok, detail = True, f"{n} row(s)"
            if r0.get("image") is not None:
                img_ok, img_detail = _validate_image(r0["image"])
                ok = img_ok
                detail += f", sample image: {img_detail}"
            print(f"  [{'OK' if ok else 'FAIL'}] docvqa: {detail}")
            results.append(("ocr_to_text.docvqa", ok, detail))
    except Exception as e:
        print(f"  [{_error_tag(e)}] docvqa: {e}")
        results.append(("ocr_to_text.docvqa", False, str(e)))

    test_enc_pre_image_sources(args.cap, results, full_download_test=args.full_download_test)
    test_enc_pre_precompute_cost(args.precompute_sample, importlib.import_module("enc_pre"))

    broken = [(label, detail) for label, ok, detail in results if not ok]
    print("\n===== Datasets that don't work =====")
    if broken:
        for label, detail in broken:
            print(f"  {label}: {detail}")
    else:
        print("  (none -- every dataset loaded successfully)")
    print(f"\n{len(results) - len(broken)}/{len(results)} tested datasets OK.")

    print("\n===== Known issues pending replacement (not tested above) =====")
    if KNOWN_PENDING_REPLACEMENT:
        for (modname, source), reason in KNOWN_PENDING_REPLACEMENT.items():
            print(f"  {modname}.{source}: {reason}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
