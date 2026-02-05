#!/bin/bash
#SBATCH --job-name=encode-dialectal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/encode_%x_%j.out
#SBATCH --error=logs/encode_%x_%j.err

# =============================================================================
# Multi-Node Dialectal Speech Encoding
# 4 nodes x 8 H100 GPUs = 32 GPUs
# =============================================================================
#
# Submit: sbatch slurm_encode_dialectal.sh
# Monitor: squeue -u $USER
# Cancel: scancel <job_id>
#
# IMPORTANT: Uses --nproc_per_node=1 (each torchrun process manages 8 GPUs)
# =============================================================================

set -e

mkdir -p logs

echo "=============================================="
echo "Multi-Node Dialectal Encoder"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: 8"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * 8))"
echo "Node list: $SLURM_JOB_NODELIST"
echo "=============================================="

# Network settings for NCCL/Gloo
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3

# Get master address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT

echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Paths - EDIT THESE
WORK_DIR="/scratch/vikram.solanki/workspace/vs/neutts/model_checkpoints"
INPUT_FILE="/scratch/vikram.solanki/workspace/vs/data/dialected_op_data/processed_datasets/final_manifest.jsonl"
OUTPUT_FILE="/scratch/vikram.solanki/workspace/vs/neutts/datasets/encoded/encoded_dialectal.json"

cd $WORK_DIR

# Launch encoding
# CRITICAL: --nproc_per_node=1 (not 8!)
srun --export=ALL \
    --kill-on-bad-exit=1 \
    /scratch/vikram.solanki/miniconda3/envs/torch-m4/bin/torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    encode_dialectal_multinode.py \
        $INPUT_FILE \
        $OUTPUT_FILE \
        --seed 42 \
        --batch_size 48 \
        --barrier_timeout 900

echo "Encoding complete!"
