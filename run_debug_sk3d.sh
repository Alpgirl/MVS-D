#!/usr/bin/env bash
#SBATCH --job-name='i.larina.mvs-d.run_debug_sk3d'
#SBATCH --output=./sbatch_logs/%x@%A_%a.out 
#SBATCH --error=./sbatch_logs/%x@%A_%a.err
#SBATCH --time=40:00:00
#SBATCH --partition=ais-gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=80G

# Load WandB API key
source ./wandb/export_wandb.sh # exports WANDB_API_KEY
source ./wandb/fix_wandb.sh

# Set the configuration filename (passed as the first argument)
CONFIG_FILENAME="./config/mvsformer++_debug_sk3d.json"
BNV_CONFIG_FILENAME="./config/bnvfusion_sk3d.json"

# Set the experiment name (optional, passed as the second argument)
if [ -z "$1" ]; then
    EXPERIMENT_NAME="MVSD++_debug_$(date +%Y%m%d_%H%M%S)"  # Default experiment name with timestamp
else
    EXPERIMENT_NAME="MVSD++_debug_$(date +%Y%m%d_%H%M%S)_$1"
fi

# export MASTER_PORT=30371
# ### get the first node name as master address - customized for vgg slurm
# ### e.g. master(gnodee[2-5],gnoded1) == gnodee2
# echo "NODELIST="${SLURM_NODELIST}
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo "MASTER_ADDR="$MASTER_ADDR

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

# Debugging options (e - stops the script on the 1st error, x - prints each command before executing it)
set -ex

# Command for the srun with singularity container and WandB setup
SINGULARITY_SHELL=/bin/bash \
srun singularity exec --nv \
    --bind /gpfs/gpfs0/3ddl/datasets/sk3d:/sk3d/ \
    --bind /trinity/home/i.larina/MVS-D/:/app \
    /trinity/home/i.larina/docker_imgs/larina_bnvmvs-zh-2025-02-25-d24ee6e6ce09.sif /bin/bash << EOF

# Initialize Conda
source ~/.bashrc                   # Or source ~/.bash_profile, depending on your setup
source /opt/conda/bin/activate # Replace with the path to your Conda installation
conda activate bnvmvs

# Log into WandB using the API key from the environment
python -m wandb login --relogin \$WANDB_API_KEY

# Navigate to the project directory
cd /app/

export PYTHONPATH=$PYTHONPATH:$PWD

cd MVSFormerPlusPlus/

# Run the training script with the specified config file and experiment name
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 train_sk3d.py \
            --config $CONFIG_FILENAME \
            --bnvconfig $BNV_CONFIG_FILENAME \
            --exp_name $EXPERIMENT_NAME \
            --DDP

conda deactivate

EOF
