#!/bin/bash
#SBATCH --job-name=hpt_run
#SBATCH --output=/cluster/tufts/hrilab/hlu07/HPT/logs/hpt_%j.out
#SBATCH --error=/cluster/tufts/hrilab/hlu07/HPT/logs/hpt_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu,preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hlu07@tufts.edu


singularity exec --nv /cluster/tufts/hrilab/hlu07/hpt.sif \
  python -m hpt.run