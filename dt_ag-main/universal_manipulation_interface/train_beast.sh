#!/bin/bash
# on beast, activate umi env
eval "$(mamba shell hook --shell bash)"
mamba activate
mamba activate umi
# Define training parameters
ALGO="train_xarm_baseline_2d_timm"
TAG="front_wrist_timm_vitp16"
DEBUG="false"
SEED="42"
WANDB_MODE="online" # options: "online", "offline", "disabled"
PORT="25905"
NPROCS="16"
# batch size for both training and validation dataloaders
BATCH_SIZE="64"
NUM_GPUS="1"  # Change this to desired number of GPUs

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Using $NUM_GPUS GPU (single GPU mode)"
    GPU_TEXT="GPU"
else
    echo "Using $NUM_GPUS GPUs (multi-GPU mode)"
    GPU_TEXT="GPUs"
fi

# example usage
# Single GPU: bash train_xarm_baseline_2d.sh train_xarm_baseline_2d_timm xarm_baseline_2d_timm front_wrist_timm 42 29510 0
# Multi GPU: bash train_xarm_baseline_2d.sh train_xarm_baseline_2d_timm xarm_baseline_2d_timm front_wrist_timm 42 29510 0,1
EXP="${TAG}"
RUN_DIR="runs/${EXP}"

# Dynamic accelerate launch command based on number of GPUs
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU command (no --multi_gpu flag)
    HYDRA_FULL_ERROR=1 accelerate launch --num_processes ${NUM_GPUS} --main_process_port ${PORT} train.py \
    --config-name "${ALGO}.yaml" \
    hydra.run.dir="${RUN_DIR}" \
    training.debug="${DEBUG}" \
    training.seed="${SEED}" \
    exp_name="${EXP}" \
    logging.mode="${WANDB_MODE}" \
    dataloader.batch_size="${BATCH_SIZE}" \
    val_dataloader.batch_size="${BATCH_SIZE}"
else
    # Multi GPU command (with --multi_gpu flag)
    HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port ${PORT} train.py \
    --config-name "${ALGO}.yaml" \
    hydra.run.dir="${RUN_DIR}" \
    training.debug="${DEBUG}" \
    training.seed="${SEED}" \
    exp_name="${EXP}" \
    logging.mode="${WANDB_MODE}" \
    dataloader.batch_size="${BATCH_SIZE}" \
    val_dataloader.batch_size="${BATCH_SIZE}"
fi

echo "Done"