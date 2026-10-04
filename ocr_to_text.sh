#!/bin/bash
#SBATCH --mail-user=surata@rpi.edu
#SBATCH --mail-type=end,fail
#SBATCH --job-name=ocr_to_text_training
#SBATCH --output=ocr_to_text_slurm-%A.%a.out
#SBATCH --error=ocr_to_text_slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --time=48:00:00
#SBATCH --qos=dcs-48hr

# --- User-configurable ---
PROJECT_DIR="/gpfs/u/home/ENMM/ENMMsrtk/barn/mul_research"
PYTHON_SCRIPT_NAME="ocr_to_text.py"
VENV_NAME="mulEnv"
HF_TOKEN="_"
WANDB_API_KEY="_"

# The staged pipeline (tune -> train) stops after the tuning stage for
# review; just resubmit this job to continue. Progress is tracked on
# scratch, so finished stages are skipped. This script expects a frozen
# text encoder (enc_pre.py's text_train stage) AND a frozen OCR encoder
# (enc_pre.py's ocr_train stage) already present in the shared weights/
# directory -- if either is missing it falls back to a randomly initialised
# encoder for that modality, with a warning. Run this *after*
# text_to_text.py, once both frozen encoders it depends on exist.

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

# Navigate to the project directory
echo "Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# --- Networking / HF settings ---
export HF_HUB_HTTP_TIMEOUT=300
export REQUESTS_TIMEOUT=300

# --- One-time dataset preparation ---
# Download + normalise the DocVQA source into a compiled arrow dataset on
# scratch (shuffled, then split into train/val), so it persists for future
# runs. On later jobs this just loads what's already on disk. Plain
# single-process invocation -- data prep doesn't touch the GPU and gains
# nothing from being launched across multiple ranks.
echo "Ensuring ocr_to_text dataset is compiled locally..."
python "$PYTHON_SCRIPT_PATH" \
    --token "$HF_TOKEN" \
    --wandb_key "$WANDB_API_KEY" \
    --prepare_data_only

# --- Running the staged tune -> train pipeline ---
# Distributed across every GPU this job was allocated on this one node (see
# ocr_to_text.py's dist_utils.py for how train() uses this; the tuning
# stage stays single-process regardless of world size -- see that same
# module docstring for why). NPROC defaults to the node's actual allocated
# GPU count (SLURM_GPUS_ON_NODE) rather than a hardcoded number, so it
# can't drift out of sync with the #SBATCH --gres line above.
NPROC="${SLURM_GPUS_ON_NODE:-1}"
echo "Starting ocr_to_text tuning/training across $NPROC GPU(s) (stops after the tuning stage)..."
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
