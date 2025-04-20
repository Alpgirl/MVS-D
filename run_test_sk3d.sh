#!/usr/bin/env bash
#SBATCH --job-name='i.larina.mvs-d.run_train_sk3d'
#SBATCH --output=./sbatch_logs/%x@%A_%a.out 
#SBATCH --error=./sbatch_logs/%x@%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=ais-gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=200G

# Load WandB API key
source ./wandb/export_wandb.sh # exports WANDB_API_KEY
source ./wandb/fix_wandb.sh

# Set the configuration filename (passed as the first argument)
CONFIG_FILENAME="./config/mvsformer++_test_sk3d.json"
BNV_CONFIG_FILENAME="./config/bnvfusion_sk3d.json"

# Set the experiment name (optional, passed as the second argument)
EXPERIMENT_NAME="MVSD++_train_20250408_032611"

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
    /trinity/home/i.larina/docker_imgs/larina_bnvmvs-zh-fusibile-2025-04-15-d006d7d1e6b4.sif /bin/bash << EOF

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

# Run the testing script with the specified config file and experiment name
CUDA_VISIBLE_DEVICES=0 python test.py \
            --config $CONFIG_FILENAME \
            --bnvconfig $BNV_CONFIG_FILENAME \
            --dataset sk3d \
            --batch_size 1  \
            --testpath /sk3d/ \
            --testlist ./lists/sk3d/test.txt \
            --resume ./saved/models/DINOv2/$EXPERIMENT_NAME/model_best.pth  \
            --outdir ./saved/models/DINOv2/$EXPERIMENT_NAME/test_nv10_prob0.6/  \
            --interval_scale 1.00 \
            --num_view 10 \
            --numdepth 256 \
            --max_h 1920 --max_w 2368 \
            --prob_threshold 0.6 \
            --filter_method gipuma \
            --combine_conf \
            --tmps 5.0,5.0,5.0,1.0

conda deactivate

EOF
