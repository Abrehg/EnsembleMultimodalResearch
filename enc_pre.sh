#!/bin/bash
#SBATCH --mail-user=surata@rpi.edu
#SBATCH --mail-type=end,fail
#SBATCH --job-name=multi_encode_training
#SBATCH --output=slurm-%A.%a.out
#SBATCH --error=slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --qos=dcs-48hr

# --- User-configurable ---
PROJECT_DIR="/gpfs/u/home/ENMM/ENMMsrtk/barn/research"
PYTHON_SCRIPT_NAME="enc_pre.py"
VENV_NAME="mulEnv"
HF_TOKEN="_"
WANDB_API_KEY="_"

# The staged pipeline (text_tune -> text_train -> image_tune -> image_train)
# stops after every tuning stage for review; just resubmit this job to
# continue. Progress is tracked on scratch, so finished stages are skipped.

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
# Encoder pretraining is a single-process, single-GPU job (no distributed
# training), so the torchrun rendezvous setup is not needed.
export HF_HUB_HTTP_TIMEOUT=300
export REQUESTS_TIMEOUT=300

# --- One-time dataset preparation ---
# Download + cache every (non-OntoNotes) dataset into scratch so it persists
# for future runs. On later jobs this just loads from local disk.
echo "Ensuring encoder-pretraining datasets are available locally..."
python "$PYTHON_SCRIPT_PATH" \
    --token "$HF_TOKEN" \
    --wandb_key "$WANDB_API_KEY" \
    --prepare_data_only

# --- Running the staged encoder-pretraining pipeline ---
echo "Starting encoder pretraining (stops after each tuning stage)..."
python "$PYTHON_SCRIPT_PATH" \
    --token "$HF_TOKEN" \
    --wandb_key "$WANDB_API_KEY"


EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "Python script exited with error code $EXIT_CODE."
fi

echo "Job finished."