#!/bin/bash
#SBATCH --mail-user=surata@rpi.edu
#SBATCH --mail-type=end,fail
#SBATCH --job-name=test_datasets
#SBATCH --output=test_datasets_slurm-%A.%a.out
#SBATCH --error=test_datasets_slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --time=04:00:00

# Standalone dataset-loader health check (see test_datasets.py) -- plain
# single-process CPU job, no GPU/torchrun needed since this only loads and
# counts rows, it doesn't train anything.
#
# --time bumped 1h -> 4h from this script's original value: SBATCH
# directives can't read RUN_FULL_DOWNLOAD_TEST (no shell expansion in
# #SBATCH lines), so this has to be sized for the slower case up front --
# with RUN_FULL_DOWNLOAD_TEST=1, image_to_text.clevr (18GB) + enc_pre.vrd
# (~20GB) download+extract for real. The default (fast, HEAD-request-only)
# path still finishes in a couple minutes regardless of this ceiling.

# --- User-configurable ---
PROJECT_DIR="/gpfs/u/home/ENMM/ENMMsrtk/barn/mul_research"
PYTHON_SCRIPT_NAME="test_datasets.py"
VENV_NAME="mulEnv"
HF_TOKEN="_"

# Off by default -- LARGE_DOWNLOAD_SOURCES (image_to_text.clevr, 18GB;
# enc_pre.vrd/GQA, ~20GB) normally only get a cheap HEAD-request
# reachability check, so a routine run of this script stays fast. Set to
# "1" (e.g. `RUN_FULL_DOWNLOAD_TEST=1 sbatch test_datasets.sh`) to actually
# run their real downloads and get real precompute-cost timing for them --
# see test_datasets.py's --full-download-test flag.
RUN_FULL_DOWNLOAD_TEST="${RUN_FULL_DOWNLOAD_TEST:-0}"

# --- Sanity Checks ---
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project directory $PROJECT_DIR does not exist."
    exit 1
fi

PYTHON_SCRIPT_PATH="$PROJECT_DIR/$PYTHON_SCRIPT_NAME"
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Error: Python script $PYTHON_SCRIPT_PATH does not exist."
    exit 1
fi

# --- Environment Setup ---
echo "Loading modules..."
module purge
module load gcc

export http_proxy=http://proxy:8888
export https_proxy=http://proxy:8888
export HTTP_PROXY=http://proxy:8888
export HTTPS_PROXY=http://proxy:8888

ENV_BIN="/gpfs/u/home/ENMM/ENMMsrtk/barn/miniconda3/envs/$VENV_NAME/bin"
export PATH="$ENV_BIN:$PATH"

echo "Python executable: $(which python)"

# image_to_text.geoqa (Geometry3K) is Google-Drive-hosted -- needs gdown,
# not in mulEnv by default. Pure-Python package, plain pip install is
# reliable; idempotent, cheap no-op on repeat runs.
pip install --quiet gdown

echo "Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

export HF_HUB_HTTP_TIMEOUT=300
export REQUESTS_TIMEOUT=300

EXTRA_ARGS=""
if [ "$RUN_FULL_DOWNLOAD_TEST" = "1" ]; then
    echo "RUN_FULL_DOWNLOAD_TEST=1 -- will run clevr/vrd's real ~38GB combined download."
    EXTRA_ARGS="--full-download-test"
fi

echo "Testing all dataset loaders..."
python "$PYTHON_SCRIPT_PATH" --token "$HF_TOKEN" --cap 10 $EXTRA_ARGS

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "Python script exited with error code $EXIT_CODE."
fi

echo "Job finished."
