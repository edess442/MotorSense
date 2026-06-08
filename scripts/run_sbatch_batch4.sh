#!/bin/bash
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --gres=gpu:h200-sxm:4
#SBATCH --partition=vulcan-ampere
#SBATCH --account=vulcan-ruoshi
#SBATCH --qos=vulcan-high
#SBATCH --array=0-11%1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=128g
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --job-name=emg_hier_batch4
#SBATCH --chdir=.

source ~/.bashrc
conda activate iws

BATCH_INDEX=${BATCH_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}
BATCH_SIZE=${BATCH_SIZE:-4}
GPU_IDS=${GPU_IDS:-0,1,2,3}
CONFIG_PATH=${CONFIG_PATH:-emg_pretraining/configs/default.yaml}

python -u -m emg_pretraining.local_batch_launch \
  --config "$CONFIG_PATH" \
  --batch-index "$BATCH_INDEX" \
  --batch-size "$BATCH_SIZE" \
  --gpu-ids "$GPU_IDS"
