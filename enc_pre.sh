#!/bin/bash
#SBATCH --mail-user=surata@rpi.edu
#SBATCH --mail-type=end,fail
#SBATCH --job-name=multi_encode_training
#SBATCH --output=enc_pre_slurm-%A.%a.out
#SBATCH --error=enc_pre_slurm-%A.%a.err
#SBATCH --nodes=5
#SBATCH --gres=gpu:6
#SBATCH --time=48:00:00
#SBATCH --qos=dcs-48hr

# --- User-configurable ---
PROJECT_DIR="/gpfs/u/home/ENMM/ENMMsrtk/barn/mul_research"
PYTHON_SCRIPT_NAME="enc_pre.py"
VENV_NAME="mulEnv"
HF_TOKEN="_"
WANDB_API_KEY="_"

# The staged pipeline (text_tune -> text_train -> vision_ssl_pretrain ->
# vision_gram_anchor -> vision_hires_adapt -> image_tune -> image_train)
# stops after every tuning stage, AND after vision_ssl_pretrain (see that
# stage's docstring -- it's a manual review point, not just a tuning one),
# for review; just resubmit this job to continue. Progress is tracked on
# scratch, so finished stages are skipped.

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
# Matches enc_pre.py's own HF_CACHE (= DATA_DIR/hf_cache, always passed
# explicitly as cache_dir= to every load_dataset() call, so these env vars
# don't change any code behavior) -- set here purely so a manual
# `huggingface-cli download` from a login node lands in the same place the
# training job already looks.
export HF_HOME="/gpfs/u/home/ENMM/ENMMsrtk/scratch/enc_pre_data/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME"

# --- One-time dataset preparation ---
# Download + cache every (non-OntoNotes) dataset into scratch so it persists
# for future runs. On later jobs this just loads from local disk. Plain
# single-process invocation -- data prep doesn't touch the GPU and gains
# nothing from being launched across multiple ranks.
echo "Ensuring encoder-pretraining datasets are available locally..."
python "$PYTHON_SCRIPT_PATH" \
    --token "$HF_TOKEN" \
    --wandb_key "$WANDB_API_KEY" \
    --prepare_data_only

# --- Running the staged encoder-pretraining pipeline ---
# Distributed across every node/GPU this job was allocated (see
# enc_pre.py's dist_utils.py for how the training/SSL stages use this; the
# Optuna tuning stages stay single-process regardless of world size -- see
# that same module docstring for why). NNODES/NPROC default to the job's
# actual allocation (SLURM_NNODES / SLURM_GPUS_ON_NODE) rather than a
# hardcoded number, so they can't drift out of sync with the #SBATCH lines
# above. Multi-node needs a real rendezvous (srun + torchrun's c10d
# backend, MASTER_ADDR resolved from the node list) -- --standalone only
# covers the single-node case (see text_to_text.sh's identical block --
# this one was missing here even after --nodes was bumped to 10, which
# silently left 9 of the 10 allocated nodes completely idle).
NNODES="${SLURM_NNODES:-1}"
NPROC="${SLURM_GPUS_ON_NODE:-1}"
echo "Starting encoder pretraining across $NNODES node(s) x $NPROC GPU(s) (stops after each tuning stage, and after vision_ssl_pretrain)..."
if [ "$NNODES" -gt 1 ]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    MASTER_PORT=29500
    srun --nodes="$NNODES" --ntasks-per-node=1 torchrun \
        --nnodes="$NNODES" \
        --nproc_per_node="$NPROC" \
        --rdzv_id="$SLURM_JOB_ID" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        "$PYTHON_SCRIPT_PATH" \
        --token "$HF_TOKEN" \
        --wandb_key "$WANDB_API_KEY"
elif [ "$NPROC" -gt 1 ]; then
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