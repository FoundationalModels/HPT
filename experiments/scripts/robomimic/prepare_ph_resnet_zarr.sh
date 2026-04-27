#!/usr/bin/env bash
set -euo pipefail

# Build the HPT zarr cache with precomputed ResNet features for RoboMimic PH data.
# The RoboMimic adapter downloads raw PH HDF5 files when missing, converts them to
# RGB image HDF5 files, and LocalTrajDataset writes data/zarr_mujoco_robomimic_resnet_traj${EPISODE_CNT}.

EPISODE_CNT="${1:-25}"
STEP_CNT="${2:-2000}"
TASKS="${3:-can,lift,square,transport}"

export HYDRA_FULL_ERROR=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export MUJOCO_GL="${MUJOCO_GL:-egl}"

python -m hpt.run_pretrain \
  env=mujoco_robomimic \
  domains=mujoco_robomimic \
  dataset.precompute_feat=True \
  dataset.image_encoder=resnet \
  dataset.regenerate=False \
  dataset.use_disk=True \
  dataset.episode_cnt="${EPISODE_CNT}" \
  dataset.step_cnt="${STEP_CNT}" \
  dataset_generator_func.env_names="[${TASKS}]" \
  train.total_iters=1 \
  train.epoch_iters=1 \
  train.validation_iters=1 \
  pretrain.max_train_trajectories_per_source="${EPISODE_CNT}" \
  pretrain.max_val_trajectories_per_source=1 \
  dataloader.batch_size=16 \
  val_dataloader.batch_size=16 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  dataloader.persistent_workers=False \
  val_dataloader.persistent_workers=False \
  script_name=prepare_robomimic_resnet_zarr \
  wb_tag=prepare_robomimic_resnet_zarr
