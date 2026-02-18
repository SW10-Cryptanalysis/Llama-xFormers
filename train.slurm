#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --output=/ceph/project/SW10-CausalLM/CausalLM-flash/logs/slurm_%x_%j.out
#SBATCH --error=/ceph/project/SW10-CausalLM/CausalLM-flash/logs/slurm_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l4:1

set -e  # exit if any command fails

PROJECT_DIR=/ceph/project/SW10-CausalLM/CausalLM-flash
cd $PROJECT_DIR

mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Job started on $(hostname) at $(date)" | tee logs/train_live_${SLURM_JOB_ID}.log

uv run python -u src/train.py 2>&1 | tee -a logs/train_live_${SLURM_JOB_ID}.log

echo "Job finished at $(date)" | tee -a logs/train_live_${SLURM_JOB_ID}.log
