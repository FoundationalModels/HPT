#!/bin/bash
#SBATCH --job-name=hpt_run
#SBATCH --output=/cluster/tufts/hrilab/hlu07/HPT/logs/hpt_%j.out
#SBATCH --error=/cluster/tufts/hrilab/hlu07/HPT/logs/hpt_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu,preempt
#SBATCH --gres=gpu:a100:1

# Load the Tufts AI module (includes CUDA 12.2, compatible with A100/L40/H200)
# NOTE: Does NOT work with T4 GPUs
module load anaconda/2023.07.tuftsai

# Activate conda environment
conda activate /cluster/tufts/hrilab/hlu07/hpt310

# Navigate to project and run
cd /cluster/tufts/hrilab/hlu07/HPT
python -m hpt.run