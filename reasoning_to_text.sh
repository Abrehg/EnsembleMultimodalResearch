#!/bin/bash
#SBATCH --mail-user=surata@rpi.edu
#SBATCH --mail-type=end,fail
#SBATCH --job-name=reasoning_to_text_training
#SBATCH --output=reasoning_to_text_slurm-%A.%a.out
#SBATCH --error=reasoning_to_text_slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --time=48:00:00
#SBATCH --qos=dcs-48hr

# --- User-configurable ---
PROJECT_DIR="/gpfs/u/home/ENMM/ENMMsrtk/barn/mul_research"
PYTHON_SCRIPT_NAME="reasoning_to_text.py"
VENV_NAME="mulEnv"
HF_TOKEN="_"
WANDB_API_KEY="_"

# The staged pipeline (traj_gen -> tune -> reasoning_train_trm ->
# reasoning_train_vae -> reasoning_rl) stops after EVERY stage for review;
# just resubmit this job to continue to the next one. Progress is tracked
# on barn, so finished stages are skipped. This script expects a frozen
# text encoder (enc_pre.py's text_train stage) AND a frozen vision encoder
# (enc_pre.py's image_train stage) already present in the shared weights/
# directory -- if either is missing it falls back to a randomly initialised
# encoder for that modality, with a warning. Run this *after* text_to_text.py
# (and, for the best text_output warm start, after image_to_text.py too).
#
# The traj_gen stage additionally loads Qwen2.5-VL-7B-Instruct via
# transformers to generate reasoning trajectories -- needs a transformers
# version with Qwen2.5-VL support (Qwen2_5_VLForConditionalGeneration). It
# runs single-process (like the tune stage) and releases the model before
# returning, so it never coexists in memory with the training model in any
# later stage.

# --- Sanity Checks ---
if [ -z "$HF_TOKEN" ]; then
    echo "Error: Hugging Face token not provided."
    echo "Usage: sbatch $0 YOUR_HF_TOKEN"
    exit 1
fi

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
export WANDB_CORE=false

export OMP_NUM_THREADS=4

ENV_BIN="/gpfs/u/home/ENMM/ENMMsrtk/barn/miniconda3/envs/$VENV_NAME/bin"
export PATH="$ENV_BIN:$PATH"

# Verify Python and pip are from the venv
echo "Python executable: $(which python)"

# geoqa (Geometry3K) is Google-Drive-hosted, not a HF/direct-HTTP source --
# needs gdown, not in mulEnv by default. Pure-Python package (no C
# extensions to compile, unlike h5py's ppc64le build issues) so plain pip
# install is reliable here; idempotent, cheap no-op on repeat runs.
pip install --quiet gdown

# Navigate to the project directory
echo "Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# --- Networking / HF settings ---
export HF_HUB_HTTP_TIMEOUT=300
export REQUESTS_TIMEOUT=300

# --- One-time dataset preparation ---
# Download + normalise all five source tasks (logi_glue, NuminaMath-CoT,
# GeoQA, TruthfulQA, CommonsenseQA) into a single compiled arrow dataset on
# scratch (shuffled, then split into train/val per source), so it persists
# for future runs. On later jobs this just loads what's already on disk.
# Plain single-process invocation -- data prep doesn't touch the GPU and
# gains nothing from being launched across multiple ranks.
echo "Ensuring reasoning_to_text dataset is compiled locally..."
python "$PYTHON_SCRIPT_PATH" \
    --token "$HF_TOKEN" \
    --wandb_key "$WANDB_API_KEY" \
    --prepare_data_only

# --- Running the staged pipeline ---
# Distributed across every GPU this job was allocated on this one node (see
# reasoning_to_text.py's dist_utils.py for how train()/reasoning_rl() use
# this; the traj_gen and tune stages stay single-process regardless of
# world size -- see that same module docstring for why). NPROC defaults to
# the node's actual allocated GPU count (SLURM_GPUS_ON_NODE) rather than a
# hardcoded number, so it can't drift out of sync with the #SBATCH --gres
# line above.
NPROC="${SLURM_GPUS_ON_NODE:-1}"
echo "Starting reasoning_to_text pipeline across $NPROC GPU(s) (stops after each stage)..."
if [ "$NPROC" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NPROC" "$PYTHON_SCRIPT_PATH" \
        --token "$HF_TOKEN" \
        --wandb_key "$WANDB_API_KEY"
else
    python "$PYTHON_SCRIPT_PATH" \
        --token "$HF_TOKEN" \
        --wandb_key "$WANDB_API_KEY"
fi


EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "Python script exited with error code $EXIT_CODE."
fi

echo "Job finished."
