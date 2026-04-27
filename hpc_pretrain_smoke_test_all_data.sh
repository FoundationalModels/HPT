#!/bin/bash
#SBATCH --job-name=hpt_pretrain_all_smoke
#SBATCH --output=/cluster/tufts/hrilab/hlu07/HPT/logs/hpt_pretrain_all_smoke_%j.out
#SBATCH --error=/cluster/tufts/hrilab/hlu07/HPT/logs/hpt_pretrain_all_smoke_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu,preempt
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hlu07@tufts.edu

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
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/hrilab/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export TORCH_HOME=/workspace/data/.cache/torch

EPISODE_CNT="${EPISODE_CNT:-25}"
DOMAINS="${DOMAINS:-mujoco_metaworld,mujoco_robomimic,mujoco_adroit,pybullet_trifinger,pybullet_grasping_image,drake_toulouse,isaac_arnold_image,maniskill}"

cache_action_path() {
  echo "data/zarr_${1}_resnet_traj${EPISODE_CNT}/data/action"
}

require_file() {
  if [ ! -s "$1" ]; then
    echo "[preflight] missing required file: $1" >&2
    exit 2
  fi
}

require_domain_cache_or_files() {
  domain="$1"
  shift
  cache_path="$(cache_action_path "${domain}")"
  if [ -e "${cache_path}" ]; then
    echo "[preflight] ${domain}: using existing cache ${cache_path}"
    return 0
  fi
  echo "[preflight] ${domain}: no cache yet, checking source files"
  for path in "$@"; do
    require_file "${path}"
  done
}

require_domain_cache_or_any_file() {
  domain="$1"
  shift
  cache_path="$(cache_action_path "${domain}")"
  if [ -e "${cache_path}" ]; then
    echo "[preflight] ${domain}: using existing cache ${cache_path}"
    return 0
  fi
  echo "[preflight] ${domain}: no cache yet, checking source alternatives"
  for path in "$@"; do
    if [ -s "${path}" ]; then
      echo "[preflight] ${domain}: found ${path}"
      return 0
    fi
  done
  echo "[preflight] ${domain}: missing all source alternatives:" >&2
  for path in "$@"; do
    echo "  - ${path}" >&2
  done
  exit 2
}

require_domain_cache_or_files mujoco_robomimic \
  data/robomimic/can/ph/image_v141.hdf5 \
  data/robomimic/lift/ph/image_v141.hdf5 \
  data/robomimic/square/ph/image_v141.hdf5 \
  data/robomimic/transport/ph/image_v141.hdf5

require_domain_cache_or_files mujoco_adroit \
  data/adroit/pen/pen-expert-v1.hdf5 \
  data/adroit/hammer/hammer-expert-v1.hdf5 \
  data/adroit/door/door-expert-v1.hdf5 \
  data/adroit/relocate/relocate-expert-v1.hdf5

require_domain_cache_or_files pybullet_trifinger \
  data/pybullet_trifinger/cube_reach/demo_state.npz \
  data/pybullet_trifinger/cube_push/demo_state.npz \
  data/pybullet_trifinger/cube_lift/demo_state.npz

require_domain_cache_or_files drake_toulouse \
  data/demo_drake/FrankaDrakeHammerforhammerEnv-Tool0/demo_state.npz \
  data/demo_drake/FrankaDrakeSpatulaforspatulaEnv-Tool0/demo_state.npz \
  data/demo_drake/FrankaDrakeKnifeforknifeEnv-Tool0/demo_state.npz \
  data/demo_drake/FrankaDrakeWrenchforwrenchEnv-Tool0/demo_state.npz

require_domain_cache_or_files isaac_arnold_image \
  data/arnold/tasks/close_cabinet.zip \
  data/arnold/tasks/close_drawer.zip \
  data/arnold/tasks/open_cabinet.zip \
  data/arnold/tasks/open_drawer.zip \
  data/arnold/tasks/pickup_object.zip \
  data/arnold/tasks/pour_water.zip \
  data/arnold/tasks/reorient_object.zip \
  data/arnold/tasks/transfer_water.zip

require_domain_cache_or_files pybullet_grasping_image \
  data/pybullet_grasping_image/grasping_image_default/demo_state.npz

require_domain_cache_or_files maniskill \
  data/maniskill/demos/v0/rigid_body/PickCube-v0/trajectory.h5 \
  data/maniskill/demos/v0/rigid_body/PickCube-v0/trajectory.json

python -m hpt.run_pretrain \
  env=mujoco_metaworld \
  domains="${DOMAINS}" \
  +dataset_generators.mujoco_metaworld._target_=env.mujoco.metaworld.rollout_runner.generate_dataset_rollouts \
  "+dataset_generators.mujoco_metaworld.env_names=[reach-v2,push-v2,button-press-topdown-v2,door-open-v2]" \
  +dataset_generators.mujoco_metaworld.max_total_transition=500000 \
  +dataset_generators.mujoco_metaworld.episode_num_pertask="${EPISODE_CNT}" \
  +dataset_generators.mujoco_robomimic._target_=env.mujoco.robomimic.rollout_runner.generate_dataset_rollouts \
  +dataset_generators.mujoco_robomimic.embodiment=franka \
  +dataset_generators.mujoco_robomimic.env_names=[can,lift,square,transport] \
  +dataset_generators.mujoco_robomimic.max_total_transition=500000 \
  +dataset_generators.mujoco_robomimic.episode_num_pertask="${EPISODE_CNT}" \
  +dataset_generators.mujoco_robomimic.dataset_root=data/robomimic \
  +dataset_generators.mujoco_robomimic.dataset_type=ph \
  +dataset_generators.mujoco_robomimic.download=False \
  +dataset_generators.mujoco_robomimic.convert=False \
  +dataset_generators.mujoco_adroit._target_=env.mujoco.adroit.rollout_runner.generate_dataset_rollouts \
  +dataset_generators.mujoco_adroit.env_names=[pen,hammer,door,relocate] \
  +dataset_generators.mujoco_adroit.max_total_transition=500000 \
  +dataset_generators.mujoco_adroit.episode_num_pertask="${EPISODE_CNT}" \
  +dataset_generators.mujoco_adroit.download=False \
  +dataset_generators.pybullet_trifinger._target_=env.pybullet.trifinger.rollout_runner.generate_dataset_rollouts \
  +dataset_generators.pybullet_trifinger.env_names=[cube_reach,cube_push,cube_lift] \
  +dataset_generators.pybullet_trifinger.max_total_transition=500000 \
  +dataset_generators.pybullet_trifinger.episode_num_pertask="${EPISODE_CNT}" \
  +dataset_generators.pybullet_trifinger.online=False \
  +dataset_generators.pybullet_trifinger.download=False \
  +dataset_generators.drake_toulouse._target_=env.drake.toulouse.rollout_runner.generate_dataset_rollouts \
  +dataset_generators.drake_toulouse.env_names=[hammer,spatula,knife,wrench] \
  +dataset_generators.drake_toulouse.fleet_root=external/Fleet-Tools \
  +dataset_generators.drake_toulouse.fleet_data_root=data/fleet_tools \
  +dataset_generators.drake_toulouse.processed_demo_root=data/demo_drake \
  +dataset_generators.drake_toulouse.generate=False \
  +dataset_generators.drake_toulouse.tool_idx=0 \
  +dataset_generators.drake_toulouse.max_total_transition=500000 \
  +dataset_generators.drake_toulouse.episode_num_pertask="${EPISODE_CNT}" \
  +dataset_generators.isaac_arnold_image._target_=env.isaac.arnold_image.rollout_runner.generate_dataset_rollouts \
  +dataset_generators.isaac_arnold_image.env_names=[arnold_image_default] \
  +dataset_generators.isaac_arnold_image.dataset_root=data/arnold \
  +dataset_generators.isaac_arnold_image.split=train \
  +dataset_generators.isaac_arnold_image.max_total_transition=500000 \
  +dataset_generators.isaac_arnold_image.episode_num_pertask="${EPISODE_CNT}" \
  +dataset_generators.pybullet_grasping_image._target_=env.pybullet.grasping_image.rollout_runner.generate_dataset_rollouts \
  "+dataset_generators.pybullet_grasping_image.env_names=[grasping_image_default]" \
  +dataset_generators.pybullet_grasping_image.max_total_transition=500000 \
  +dataset_generators.pybullet_grasping_image.episode_num_pertask="${EPISODE_CNT}" \
  +dataset_generators.pybullet_grasping_image.download=False \
  +dataset_generators.maniskill._target_=env.maniskill.offline.rollout_runner.generate_dataset_rollouts \
  "+dataset_generators.maniskill.env_names=[PickCube-v0]" \
  +dataset_generators.maniskill.dataset_root=data/maniskill \
  +dataset_generators.maniskill.max_total_transition=500000 \
  +dataset_generators.maniskill.episode_num_pertask="${EPISODE_CNT}" \
  dataset.precompute_feat=True \
  dataset.image_encoder=resnet \
  dataset.regenerate=False \
  dataset.use_disk=True \
  dataset.episode_cnt="${EPISODE_CNT}" \
  dataset.step_cnt=2000 \
  dataset.action_horizon=1 \
  train.total_iters=10 \
  train.epoch_iters=2 \
  train.validation_iters=1 \
  pretrain.max_train_trajectories_per_source=20 \
  pretrain.max_val_trajectories_per_source=5 \
  pretrain.sampling_weights=[] \
  dataloader.batch_size=16 \
  val_dataloader.batch_size=16 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  dataloader.persistent_workers=False \
  val_dataloader.persistent_workers=False \
  script_name=pretrain_all_data_hpc_smoke \
  wb_tag=hpc_pretrain_all_data_smoke
IN_CONTAINER
