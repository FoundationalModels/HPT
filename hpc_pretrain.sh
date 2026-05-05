#!/bin/bash
#SBATCH --job-name=hpt_pretrain
#SBATCH --output=/cluster/tufts/hrilab/hlu07/HPT/logs/hpt_pretrain_%j.out
#SBATCH --error=/cluster/tufts/hrilab/hlu07/HPT/logs/hpt_pretrain_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=128G
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p gpu,preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hlu07@tufts.edu

# Full pretraining across all domains defined in config.yaml.
#
# Prerequisites (run once):
#   sbatch hpc_download_sim_data.sh
#
# On first run, dataset generators in config.yaml build zarr caches for any
# domain whose cache does not yet exist.  Re-submit the same command if
# preempted; auto_resume picks up from the last checkpoint and cached zarrs
# are loaded directly without re-generation.

set -euo pipefail

module load singularity/3.8.4

REPO_DIR=${REPO_DIR:-/cluster/tufts/hrilab/hlu07/HPT}
CONTAINER=${CONTAINER:-/cluster/tufts/hrilab/hlu07/hpt.sif}

mkdir -p "${REPO_DIR}/logs"

singularity exec --nv \
  --bind "${REPO_DIR}:/workspace" \
  "${CONTAINER}" \
  bash -s <<'IN_CONTAINER'
set -euo pipefail
cd /workspace

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=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
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/opt/mujoco/mujoco210/bin:/usr/lib/nvidia"
export TORCH_HOME=/workspace/data/.cache/torch

# All domains, dataset params, train params, and dataset generators come from
# config.yaml.  Generators are only invoked when a domain's zarr cache does not
# yet exist; subsequent runs (including restarts after preemption) load the cache
# directly and skip generation.
python -m hpt.run_pretrain \
  script_name=hpc_pretrain_all_data \
  wb_tag=hpc_pretrain_all_data
IN_CONTAINER
