#!/bin/bash
#SBATCH --mail-user=surata@rpi.edu
#SBATCH --mail-type=end,fail
#SBATCH --job-name=dim_tuning
#SBATCH --output=slurm-%A.%a.out
#SBATCH --error=slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --qos=dcs-48hr

# --- User-configurable ---
PROJECT_DIR="/gpfs/u/home/ENMM/ENMMsrtk/barn/research"
PYTHON_SCRIPT_NAME="dim_tune.py"
VENV_NAME="mulEnv"
HF_TOKEN="_"

# Optuna dimension-tuning knobs.
N_TRIALS=20
MAX_OPT_STEPS=3000
TRAIN_EXAMPLES=20000
EVAL_EXAMPLES=2000
EVAL_SAMPLES=500

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
# The Optuna dimension search is a single-process job (no distributed
# training), so we skip the torchrun rendezvous setup used by the other jobs.
export HF_HUB_HTTP_TIMEOUT=300
export REQUESTS_TIMEOUT=300

# --- One-time dataset preparation ---
# Download + cache TriviaQA into Datasets/ so it persists for future runs.
# On subsequent jobs this is a no-op that just loads from local disk.
echo "Ensuring TriviaQA is available locally..."
python "$PYTHON_SCRIPT_PATH" \
    --token "$HF_TOKEN" \
    --prepare_data_only

# --- Running the dimension-tuning study ---
echo "Starting Optuna dimension tuning..."
python "$PYTHON_SCRIPT_PATH" \
    --token "$HF_TOKEN" \
    --n_trials "$N_TRIALS" \
    --max_opt_steps "$MAX_OPT_STEPS" \
    --train_examples "$TRAIN_EXAMPLES" \
    --eval_examples "$EVAL_EXAMPLES" \
    --eval_samples "$EVAL_SAMPLES"


EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "Python script exited with error code $EXIT_CODE."
fi

echo "Job finished."