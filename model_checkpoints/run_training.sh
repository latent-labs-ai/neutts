#!/bin/bash
# =============================================================================
# Multi-Node Training Launcher for SLURM + Enroot
# =============================================================================
#
# This script is executed by srun on each node inside the enroot container.
# It automatically determines node rank from SLURM environment variables.
#
# Usage (from salloc session):
#   srun --nodes=4 --ntasks-per-node=1 --gres=gpu:8 \
#       enroot start --rw \
#       --mount "/scratch:/scratch" \
#       --mount "$HOME:/workspace" \
#       --env "SLURM_NNODES=$SLURM_NNODES" \
#       --env "SLURM_NODEID=\$SLURM_NODEID" \
#       --env "SLURM_GPUS_ON_NODE=8" \
#       --env "SLURM_LAUNCH_NODE_IPADDR=$SLURM_LAUNCH_NODE_IPADDR" \
#       pytorch2401 \
#       bash /scratch/vikram.solanki/workspace/vs/neutts/model_checkpoints/run_training.sh
#
# =============================================================================

set -e

# Change to working directory
cd /scratch/vikram.solanki/workspace/vs/neutts/model_checkpoints

# Get node rank from SLURM (0, 1, 2, 3, ...)
NODE_RANK=${SLURM_NODEID:-0}
NNODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
MASTER_ADDR=${SLURM_LAUNCH_NODE_IPADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# Print node info
echo "=============================================="
echo "Node $NODE_RANK / $NNODES starting"
echo "Hostname: $(hostname)"
echo "GPUs: $GPUS_PER_NODE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "=============================================="

# Check GPUs
nvidia-smi -L

# Launch training with torchrun
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    finetune_1.5b_hf.py config_1.5b_multinode.yaml

echo "Node $NODE_RANK finished"
