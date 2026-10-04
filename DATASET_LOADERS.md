# Dataset loader health, provenance, and precompute costs

This document summarizes an investigation into the dataset loaders used by
`text_to_text.py`, `image_to_text.py`, `reasoning_to_text.py`, `ocr_to_text.py`,
and `enc_pre.py`. It covers three things:

1. Which loaders were broken, why, and what replaced them
2. A cluster-specific network issue (proxy blocking) and its workaround
3. Which data sources need dedicated large-scale precompute vs. a simple
   local download

Use `test_datasets.py` / `test_datasets.sh` to re-verify any of this after a
change — see [Tooling reference](#tooling-reference) below.

## Summary of what changed

| Source | Was | Is now | Why |
|---|---|---|---|
| `image_to_text.geoqa` / `reasoning_to_text.geoqa` | `MathLLMs/GeoQA` (HF id) | **Geometry3K** (Lu et al., ACL 2021) | The HF id doesn't exist on the Hub |
| `image_to_text.clevr` | `HuggingFaceM4/CLEVR` (HF id) | **Original CLEVR v1.0** (Johnson et al., CVPR 2017) | HF mirror is script-based with "no supported data files found" |
| `enc_pre.vrd` | `Voxel51/GQA-Scene-Graph` (HF id) | **Original GQA** (Stanford) | FiftyOne export whose relationship annotations the generic HF loader never actually ingests |

All three replacements use each dataset's **original, official academic
source** rather than a third-party HF re-upload — a deliberate choice for
publication provenance, not just convenience.

## Replacement details

### geoqa &rarr; Geometry3K

- Source: [github.com/lupantech/InterGPS](https://github.com/lupantech/InterGPS), MIT licensed
- Hosting: Google Drive folder `1d05WYXtlgKIoaPpK1v94LYph_heiXM7Z` (`train.zip`/`val.zip`/`test.zip`; `logic_forms.zip`/`symbols.zip` are InterGPS's own symbolic-solver annotations, unused here)
- Fetched via `gdown` (`pip install gdown` — added to `image_to_text.sh`/`reasoning_to_text.sh`/`test_datasets.sh` setup steps; pure Python, no compilation risk)
- Schema per problem (`data.json` + `img_problem.png` or `img_diagram.png`): `{"problem_text": "...", "choices": ["...", "...", "...", "..."], "answer": "A"-"D"}` — a plain 4-choice MCQ, formatted via the existing `_format_mcq`/`_mcq_answer_letter` helpers (same pattern as `ai2d`/`scienceqa`)
- Real size: ~127MB total (train.zip 89MB, val.zip 12.5MB, test.zip 25.5MB) — small, no precompute needed
- **Note**: the *original* chen-judge/GeoQA (the dataset this project's code first referenced) is Chinese-language, pre-tokenized, with formal-DSL-program answers rather than a natural-language Q/A pair — not directly usable in this pipeline. Geometry3K is a *different* original English-language geometry benchmark chosen to fill the same functional role, per explicit project direction, not a translation of GeoQA.

### clevr &rarr; original CLEVR v1.0

- Source: [cs.stanford.edu/people/jcjohns/clevr](https://cs.stanford.edu/people/jcjohns/clevr/), CC BY 4.0
- Hosting: direct HTTPS from Meta/FAIR's CDN, `https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip` (18GB)
- Schema: `questions/CLEVR_<split>_questions.json` has a top-level `"questions"` list, each `{"image_filename", "question", "answer", ...}`; images at `images/<split>/<image_filename>`
- Verified real download: 19.0GB in 5.8 min (54.6 MB/s) on this cluster, extraction 2.1 min — **not** blocked by this cluster's proxy (see below)

### vrd &rarr; original GQA

- Source: [cs.stanford.edu/people/dorarad/gqa](https://cs.stanford.edu/people/dorarad/gqa/about.html), CC BY 4.0
- Hosting: direct HTTPS from Stanford's CDN — `sceneGraphs.zip` (42.7MB) and `images.zip` (~20GB) at `https://downloads.cs.stanford.edu/nlp/data/gqa/`
- Schema (confirmed against the real downloaded `sceneGraphs.zip`): `train_sceneGraphs.json`/`val_sceneGraphs.json`, each a dict keyed by image_id &rarr; `{"width", "height", "objects": {object_id: {"name", "x", "y", "w", "h", "attributes": [...], "relations": [{"name": <predicate>, "object": <target_object_id>}]}}}`
- Only the base `images.zip` is fetched, not GQA's optional precomputed spatial/object feature files (~53GB more, meant for older non-end-to-end VQA models — irrelevant here since `VRDTask` encodes images itself via the frozen vision encoder)

## Cluster network issue: proxy blocks specific hosts

Confirmed via real runs (`slurm-4781715`, `slurm-4781718`): this cluster's
outbound proxy actively filters some external hosts with

```
ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Filtered'))
```

This is **not** a dataset-side access denial — it's the proxy server itself
returning "403 Filtered", a deliberate network policy. Confirmed hosts:

| Host | Blocked? |
|---|---|
| `doc-*-docs.googleusercontent.com` (Google Drive large-file redirect) | Yes |
| `downloads.cs.stanford.edu` (GQA) | Yes |
| `dl.fbaipublicfiles.com` (CLEVR) | No |

### Workaround

Download the file on a machine that isn't behind this proxy, then place it
directly at the expected cache path (via `scp`, or this project's local
mount if you have one) — every download function in this codebase checks
for an already-present zip **before** attempting a network fetch, so it
extracts what's there instead of re-downloading:

| File | Destination (relative to each script's `DATA_DIR`) |
|---|---|
| `train.zip`/`val.zip`/`test.zip` (Geometry3K) | `image_to_text_data/geometry3k_raw/` **and** `reasoning_to_text_data/geometry3k_raw/` (needed in both — each script keeps its own cache) |
| `sceneGraphs.zip` (GQA) | `enc_pre_data/gqa_scenegraphs_raw/sceneGraphs.zip` |
| `images.zip` (GQA) | `enc_pre_data/gqa_images_raw/images.zip` |

`DATA_DIR` for each script is `SCRATCH_DIR/<script>_data`, e.g.
`/gpfs/u/home/ENMM/ENMMsrtk/scratch/enc_pre_data`.

**Status as of this writing**: Geometry3K (both locations) and GQA's
scene graphs are already placed. GQA's `images.zip` (~20GB) is the one
piece still needing a manual transfer.

## Precompute cost: what needs dedicated large-scale compute vs. a simple download

### Needs dedicated large-scale compute (Azure VM fleet or a dedicated precompute job)

These are not file transfers — they're sustained CPU/IO-bound
materialization (streaming, decoding, resizing, and saving large volumes of
images via `_materialize_image_pool`/`Task.download`), measured in hours:

| Source | Scale | Estimated one-time cost |
|---|---|---|
| `enc_pre.imagenet1k` | uncapped, ~1.28M images (full split) | ~3.7 hours |
| `enc_pre.pass` | 500,000 images | ~11.7 hours |
| `enc_pre.depth` | 30,000 images (HDF5 parsing via h5py) | ~2.4 hours |

### Small-scale — local download + scp/mount-copy is sufficient

One-time file transfers, not sustained compute:

| Source | Status |
|---|---|
| Geometry3K (`geoqa`) | Done — ~127MB, already placed |
| GQA scene graphs | Done — 44.8MB, already placed |
| GQA images (`vrd`) | **Still needed** — ~20GB, proxy-blocked on this cluster |
| CLEVR (`clevr`) | No action needed — not proxy-blocked, downloads fine on the cluster itself (37s&ndash;8.5min depending on cache state) |

### Already fine, no precompute needed

| Source | Estimated cost |
|---|---|
| `enc_pre.imagenet21k` | ~15.3 min |
| `enc_pre.semseg` | ~5.9 min |

### Open item

`vrd`'s actual *processing* time at its real 30,000-row cap hasn't been
measured yet — only the download itself was tested so far (and is still
blocked on `images.zip`). Once the images are in place, re-run
`test_datasets.sh --full-download-test` (or `RUN_FULL_DOWNLOAD_TEST=1
sbatch test_datasets.sh`) to get a real number. Given its parsing is plain
JSON + JPEG decode (no HDF5 complexity like `depth`), it's plausible this
lands in the "fine inline" camp similar to `semseg` — but this should be
confirmed, not assumed.

## Tooling reference

- **`test_datasets.py`** / **`test_datasets.sh`**: standalone health check
  for every dataset loader across the `*_to_text.py` scripts and
  `enc_pre.py`'s image sources. Fast/cheap by default (`--cap 10` per
  loader). Submit with `sbatch test_datasets.sh`.
- **`RUN_FULL_DOWNLOAD_TEST=1`** (env var on `test_datasets.sh`) /
  **`--full-download-test`** (direct flag on `test_datasets.py`): actually
  runs `clevr`/`vrd`'s real multi-GB download+parse instead of just a
  HEAD-request reachability check. Off by default so routine runs stay
  fast. Bumps `test_datasets.sh`'s time limit to 4 hours to cover it.
- **Error classification** in `test_datasets.py`'s output:
  - `[FAIL]` &mdash; generic failure (schema drift, network, missing dependency)
  - `[FAIL-ACCESS-DENIED]` &mdash; a genuine gated/unauthorized response from the
    remote service (e.g. a dataset requiring accepted terms + a token)
  - `[FAIL-PROXY-BLOCKED]` &mdash; this cluster's own proxy filtered the
    request (see [Cluster network issue](#cluster-network-issue-proxy-blocks-specific-hosts)
    above) &mdash; the fix is a manual transfer, not a token or terms acceptance
- **`enc_pre.depth`** requires `h5py`, not installed in `mulEnv` by
  default. Install via `conda install -n mulEnv -c conda-forge h5py`
  (not `pip install` &mdash; h5py has no prebuilt wheel for this cluster's
  ppc64le architecture and fails compiling from source via pip; conda-forge
  has a prebuilt binary).
- **Geometry3K** requires `gdown` (`pip install gdown` &mdash; already added
  to the relevant `.sh` scripts' setup steps). Pure Python, no
  compilation risk.

## Slurm output files

Every `.sh` script's `#SBATCH --output`/`--error` now prefixes the
standard `slurm-%A.%a.out`/`.err` pattern with the script's own name, e.g.
`enc_pre_slurm-4781718.4294967294.out`, `test_datasets_slurm-....out`,
`collapse_fix_harness_slurm-....out` &mdash; makes it possible to tell which
job a log file belongs to at a glance in a directory with many of them.
