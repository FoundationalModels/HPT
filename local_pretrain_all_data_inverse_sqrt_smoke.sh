#!/bin/bash

set -euo pipefail

REPO_DIR=${REPO_DIR:-$(pwd)}
PYTHON=${PYTHON:-python}

cd "${REPO_DIR}"

export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export HPT_DEVICE="${HPT_DEVICE:-cpu}"
export TORCH_HOME="${TORCH_HOME:-${REPO_DIR}/data/.cache/torch}"

"${PYTHON}" -m hpt.run_pretrain \
  env=mujoco_metaworld \
  "domains='mujoco_metaworld,mujoco_robomimic,mujoco_adroit,pybullet_trifinger,pybullet_grasping_image,drake_toulouse,isaac_arnold_image,maniskill'" \
  +dataset_generators.mujoco_metaworld._target_=env.mujoco.metaworld.rollout_runner.generate_dataset_rollouts \
  +dataset_generators.mujoco_metaworld.env_names=[reach-v2] \
  +dataset_generators.mujoco_metaworld.max_total_transition=500000 \
  +dataset_generators.mujoco_metaworld.episode_num_pertask=3 \
  +dataset_generators.pybullet_grasping_image._target_=env.pybullet.grasping_image.rollout_runner.generate_dataset_rollouts \
  +dataset_generators.pybullet_grasping_image.env_names=[grasping_image_default] \
  +dataset_generators.pybullet_grasping_image.max_total_transition=500000 \
  +dataset_generators.pybullet_grasping_image.episode_num_pertask=3 \
  +dataset_generators.pybullet_grasping_image.download=False \
  +dataset_generators.mujoco_robomimic._target_=env.mujoco.robomimic.rollout_runner.generate_dataset_rollouts \
  +dataset_generators.mujoco_robomimic.embodiment=franka \
  +dataset_generators.mujoco_robomimic.env_names=[can] \
  +dataset_generators.mujoco_robomimic.max_total_transition=500000 \
  +dataset_generators.mujoco_robomimic.episode_num_pertask=3 \
  +dataset_generators.mujoco_robomimic.dataset_root=data/robomimic \
  +dataset_generators.mujoco_robomimic.dataset_type=ph \
  +dataset_generators.mujoco_robomimic.download=False \
  +dataset_generators.mujoco_robomimic.convert=False \
  +dataset_generators.mujoco_adroit._target_=env.mujoco.adroit.rollout_runner.generate_dataset_rollouts \
  +dataset_generators.mujoco_adroit.env_names=[door] \
  +dataset_generators.mujoco_adroit.max_total_transition=500000 \
  +dataset_generators.mujoco_adroit.episode_num_pertask=3 \
  +dataset_generators.mujoco_adroit.download=False \
  +dataset_generators.pybullet_trifinger._target_=env.pybullet.trifinger.rollout_runner.generate_dataset_rollouts \
  +dataset_generators.pybullet_trifinger.env_names=[cube_reach] \
  +dataset_generators.pybullet_trifinger.max_total_transition=500000 \
  +dataset_generators.pybullet_trifinger.episode_num_pertask=3 \
  +dataset_generators.pybullet_trifinger.online=False \
  +dataset_generators.pybullet_trifinger.download=False \
  +dataset_generators.drake_toulouse._target_=env.drake.toulouse.rollout_runner.generate_dataset_rollouts \
  +dataset_generators.drake_toulouse.env_names=[hammer] \
  +dataset_generators.drake_toulouse.fleet_root=external/Fleet-Tools \
  +dataset_generators.drake_toulouse.fleet_data_root=data/fleet_tools \
  +dataset_generators.drake_toulouse.processed_demo_root=data/demo_drake \
  +dataset_generators.drake_toulouse.generate=False \
  +dataset_generators.drake_toulouse.tool_idx=0 \
  +dataset_generators.drake_toulouse.max_total_transition=500000 \
  +dataset_generators.drake_toulouse.episode_num_pertask=3 \
  +dataset_generators.isaac_arnold_image._target_=env.isaac.arnold_image.rollout_runner.generate_dataset_rollouts \
  +dataset_generators.isaac_arnold_image.env_names=[pickup_object] \
  +dataset_generators.isaac_arnold_image.dataset_root=data/arnold \
  +dataset_generators.isaac_arnold_image.split=train \
  +dataset_generators.isaac_arnold_image.max_total_transition=500000 \
  +dataset_generators.isaac_arnold_image.episode_num_pertask=3 \
  +dataset_generators.maniskill._target_=env.maniskill.offline.rollout_runner.generate_dataset_rollouts \
  +dataset_generators.maniskill.env_names=[PickCube-v0] \
  +dataset_generators.maniskill.dataset_root=data/maniskill \
  +dataset_generators.maniskill.max_total_transition=500000 \
  +dataset_generators.maniskill.episode_num_pertask=3 \
  dataset.precompute_feat=True \
  dataset.image_encoder=resnet \
  dataset.regenerate=True \
  dataset.use_disk=True \
  dataset.episode_cnt=3 \
  dataset.step_cnt=500 \
  dataset.val_ratio=0.34 \
  dataset.action_horizon=8 \
  train.total_iters=2 \
  train.epoch_iters=1 \
  train.validation_iters=1 \
  train.auto_resume=False \
  train.freeze_trunk=False \
  pretrain.max_train_trajectories_per_source=2 \
  pretrain.max_val_trajectories_per_source=1 \
  pretrain.sampling_weights=[] \
  dataloader.batch_size=2 \
  val_dataloader.batch_size=2 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  dataloader.persistent_workers=False \
  val_dataloader.persistent_workers=False \
  script_name=local_all_data_inverse_sqrt_smoke \
  wb_tag=local_all_data_inverse_sqrt_smoke \
  device="${HPT_DEVICE}"
