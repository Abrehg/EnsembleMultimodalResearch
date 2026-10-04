#!/bin/bash
#SBATCH --mail-user=surata@rpi.edu
#SBATCH --mail-type=end,fail
#SBATCH --job-name=collapse_fix_harness
#SBATCH --output=collapse_fix_harness_slurm-%A.%a.out
#SBATCH --error=collapse_fix_harness_slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --qos=dcs-48hr

# Offline verification harness for the anti-collapse fixes (P1.1/P1.2 --
# see collapse_fix_harness.py's module docstring and
# /Users/adityaasuratkal/.claude/plans/dazzling-toasting-moler.md). Single
# GPU, single process -- no torchrun/multi-node needed, this only ever runs
# one model on one device (see text_to_text.sh for the real multi-node
# training launch this harness does NOT need to match).

# --- User-configurable ---
PROJECT_DIR="/gpfs/u/home/ENMM/ENMMsrtk/barn/mul_research"
PYTHON_SCRIPT_NAME="collapse_fix_harness.py"
VENV_NAME="mulEnv"
HF_TOKEN="_"

# Stopping: whichever comes first of HARNESS_TARGET_STEPS or
# HARNESS_MAX_HOURS. HARNESS_MAX_HOURS is deliberately well under the
# --time=48:00:00 above (real per-step GPU throughput is unmeasured going
# in -- local CPU-only testing measured ~44s/step, dominated by
# t2t_rl_term's full 32-token generation, which does not predict GPU
# throughput) so the script exits cleanly with a results summary instead of
# being hard-killed by SLURM mid-checkpoint.
#
# Each ${VAR:-default} so a short diagnostic-only run (see the plan's Phase
# 0) can override any of these from the sbatch command line without editing
# this file, e.g.:
#   HARNESS_TARGET_STEPS=600 HARNESS_EVAL_EVERY=100 HARNESS_MAX_HOURS=2 \
#       sbatch --time=01:00:00 collapse_fix_harness.sh
export HARNESS_TARGET_STEPS="${HARNESS_TARGET_STEPS:-8000}"
export HARNESS_MAX_HOURS="${HARNESS_MAX_HOURS:-40}"
export HARNESS_EVAL_EVERY="${HARNESS_EVAL_EVERY:-300}"
export HARNESS_POOL_SIZE="${HARNESS_POOL_SIZE:-4000}"
export HARNESS_VAL_SAMPLE="${HARNESS_VAL_SAMPLE:-200}"

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

export OMP_NUM_THREADS=4
# Required for deterministic cuBLAS (must be set before CUDA initialises).
# Paired with the torch.use_deterministic_algorithms call in collapse_fix_harness.py.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ENV_BIN="/gpfs/u/home/ENMM/ENMMsrtk/barn/miniconda3/envs/$VENV_NAME/bin"
export PATH="$ENV_BIN:$PATH"

echo "Python executable: $(which python)"

echo "Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

export HF_HUB_HTTP_TIMEOUT=300
export REQUESTS_TIMEOUT=300

if [ -n "$HF_TOKEN" ] && [ "$HF_TOKEN" != "_" ]; then
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')" 2>&1 || true
fi

echo "Starting collapse-fix offline verification harness (single GPU)..."
echo "Config: target_steps=$HARNESS_TARGET_STEPS max_hours=$HARNESS_MAX_HOURS " \
     "eval_every=$HARNESS_EVAL_EVERY pool_size=$HARNESS_POOL_SIZE val_sample=$HARNESS_VAL_SAMPLE"
python "$PYTHON_SCRIPT_PATH"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "Python script exited with error code $EXIT_CODE."
fi

echo "Job finished."
