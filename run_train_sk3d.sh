#!/usr/bin/env bash
#SBATCH --job-name='i.larina.mvs-d.run_train_sk3d'
#SBATCH --output=./sbatch_logs/%x@%A_%a.out 
#SBATCH --error=./sbatch_logs/%x@%A_%a.err
#SBATCH --time=200:00:00
#SBATCH --partition=ais-gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --reservation=HPC-2455-3
#SBATCH --mem=800G

# Load WandB API key
source ./wandb/export_wandb.sh # exports WANDB_API_KEY
source ./wandb/fix_wandb.sh

# Set the configuration filename (passed as the first argument)
CONFIG_FILENAME="./config/mvsformer++_sk3d.json"
BNV_CONFIG_FILENAME="./config/bnvfusion_sk3d.json"

RESUME_CHECKPOINT="./saved/models/DINOv2/MVSD++_train_20250425_181928/model_best.pth"

# Set the experiment name (optional, passed as the second argument)
if [ -z "$1" ]; then
    EXPERIMENT_NAME="MVSD++_train_$(date +%Y%m%d_%H%M%S)"  # Default experiment name with timestamp
else
    EXPERIMENT_NAME="MVSD++_train_$(date +%Y%m%d_%H%M%S)_$1"
fi

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
    --bind /trinity/home/g.bobrovskih/ongoing/mvsw3dfeatures/MVSFormerPlusPlus/saved/models/DINOv2/MVSFormer++_20241125_213612/:/weights \
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
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 train_sk3d.py \
            --config $CONFIG_FILENAME \
            --bnvconfig $BNV_CONFIG_FILENAME \
            --resume $RESUME_CHECKPOINT \
            --exp_name $EXPERIMENT_NAME \
            --DDP

conda deactivate

EOF
