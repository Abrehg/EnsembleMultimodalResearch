#!/bin/bash
#SBATCH --mail-user=surata@rpi.edu
#SBATCH --mail-type=end,fail
#SBATCH --job-name=diagnose_state_collapse
#SBATCH --output=diagnose_state_collapse_slurm-%A.%a.out
#SBATCH --error=diagnose_state_collapse_slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# Phase 2 diagnostic (see diagnose_state_collapse.py's module docstring and
# /Users/adityaasuratkal/.claude/plans/dazzling-toasting-moler.md) -- single
# GPU. Originally run as a CPU-only job on the (mistaken) assumption that a
# forward-pass-only workload (no generation, no backward pass) would be
# cheap regardless of device -- job 4789535 didn't even clear 20 of 60 rows
# within a 30-minute CPU budget (d_model=1024 plus HybridLinearEncoder's two
# full-attention layers, quadratic in sequence length, over some genuinely
# long logi_glue/deepmind_math reasoning-chain inputs, without GPU
# parallelism). Moved to GPU instead of just raising the CPU time budget,
# since collapse_fix_harness.py already establishes this exact model runs
# fast there even with a full forward+backward+generation pass per step --
# this diagnostic only ever does a bare forward pass, so it should be a
# small fraction of that per-step cost.

# --- User-configurable ---
PROJECT_DIR="/gpfs/u/home/ENMM/ENMMsrtk/barn/mul_research"
PYTHON_SCRIPT_NAME="diagnose_state_collapse.py"
VENV_NAME="mulEnv"

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

echo "Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

echo "Tracing state diversity across encode_to_state's pipeline stages..."
python "$PYTHON_SCRIPT_PATH"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "Python script exited with error code $EXIT_CODE."
fi

echo "Job finished."
