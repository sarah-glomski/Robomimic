#!/bin/bash
#SBATCH --partition=humanoid --qos=normal
#SBATCH --account=arm
#SBATCH --nodelist=humanoid1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# only use the following on partition with GPUs
#SBATCH --gres=gpu:3
#SBATCH --job-name="dt_ag_2d_timm"
#SBATCH --output=runs/output/dt_ag_2d_strawberry_baseline-%j.out
####SBATCH --output=output/dt_ag_2d_strawberry_baseline.out
# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL
# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
# sample process (list hostnames of the nodes you've requested)
#NPROCS=srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l
NPROCS=${SLURM_NNODES}
echo NPROCS=$NPROCS
echo NPROCS=$NPROCS
# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery
# done


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/arm/u/swann/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/arm/u/swann/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/arm/u/swann/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/arm/u/swann/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate umi

# Define training parameters
ALG="train_xarm_baseline_2d_timm"
TAG="front_wrist_timm_vitp16"
DEBUG="false"
SEED="42"
WANDB_MODE="online"  # options: "online", "offline", "disabled"
PORT="25905"

# batch size for both training and validation dataloaders
BATCH_SIZE="64"

NUM_GPUS=$(grep -oP '#SBATCH --gres=gpu:\K\d+' "$0" || echo "1")
echo "Using $NUM_GPUS GPUs"

# bash train_xarm_baseline_2d.sh train_xarm_baseline_2d_timm xarm_baseline_2d_timm front_wrist_timm 42 29510 0,1

# EXP="${TASK}-${ALG}-${TAG}"
EXP="${TAG}"
# RUN_DIR="runs/${EXP}_seed${SEED}"
RUN_DIR="runs/${EXP}"

accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port ${PORT} train.py \
  --config-name "${ALG}.yaml" \
  hydra.run.dir="${RUN_DIR}" \
  training.debug="${DEBUG}" \
  training.seed="${SEED}" \
  exp_name="${EXP}" \
  logging.mode="${WANDB_MODE}" \
  dataloader.batch_size="${BATCH_SIZE}" \
  val_dataloader.batch_size="${BATCH_SIZE}"

echo "Done"
