#!/bin/bash
#SBATCH --job-name=hpt_test
#SBATCH --output=/cluster/tufts/hrilab/hlu07/HPT/logs/hpt_test_%j.out
#SBATCH --error=/cluster/tufts/hrilab/hlu07/HPT/logs/hpt_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p preempt
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hlu07@tufts.edu

module load singularity/3.8.4
export WANDB_API_KEYwandb_v1_A7qwj0pfFcU6FDIOP1iZ6cWhkdR_YOjqeigXtfPw3V49OxXHMdifX0F89Kg1UbVBrl5kmzm3Oq2AQ

singularity exec --nv \
  --bind /cluster/tufts/hrilab/hlu07/HPT:/workspace \
  /cluster/tufts/hrilab/hlu07/hpt.sif \
  bash -c "cd /workspace && python -m hpt.run +mode=debug"