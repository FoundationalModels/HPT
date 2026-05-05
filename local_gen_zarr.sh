#!/bin/bash
# Convert raw demonstration data at $DATA_DIR into HPT zarr caches and symlink
# them under the project's data/ directory.
#
# Expects raw data to already be present (run local_download_sim_data.sh first).
# mujoco_metaworld and pybullet_trifinger have no raw files — they are generated
# online from simulation, same as in local_download_sim_data.sh.
#
# All zarr directories are written to $DATA_DIR and symlinked at data/zarr_<domain>.
# A .zarr_complete sentinel marks fully finished zarrs so reruns are idempotent.
#
# Usage:
#   ./local_gen_zarr.sh
#   DATA_DIR=/path/to/drive DATASETS=mujoco_robomimic,mujoco_adroit ./local_gen_zarr.sh
#
# Training: generated zarrs have no encoder postfix (precompute_feat=False).
# Pass these overrides to run_pretrain.py so it finds them:
#   dataset.precompute_feat=False stem.precompute_feat=False \
#   dataset.dataset_postfix="" dataset.dataset_encoder_postfix=""

set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")" && pwd)}
DATA_DIR=${DATA_DIR:-/media/hrilab/HelenMacPc/data}

if [ ! -d "${DATA_DIR}" ]; then
  echo "ERROR: data dir not found: ${DATA_DIR}" >&2
  echo "Make sure the external drive is mounted before running this script." >&2
  exit 1
fi

export DATA_DIR
cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/data"

CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/experiments/configs/config.yaml}"
CONFIG_DOMAINS="$(
  sed -n 's/^[[:space:]]*domains:[[:space:]]*//p' "${CONFIG_PATH}" \
    | head -n 1 \
    | sed 's/[[:space:]]*#.*$//' \
    | tr -d "\"'"
)"
DATASETS="${DATASETS:-${CONFIG_DOMAINS}}"
DATASETS="${DATASETS// /}"

EPISODE_CNT="${EPISODE_CNT:-10000}"
ROBOMIMIC_TASKS="${ROBOMIMIC_TASKS:-lift,can,square,transport,tool_hang}"
ADROIT_TASKS="${ADROIT_TASKS:-door,hammer,pen,relocate}"
METAWORLD_TASKS="${METAWORLD_TASKS:-all}"
METAWORLD_EPISODES="${METAWORLD_EPISODES:-1200}"
METAWORLD_MAX_TOTAL_TRANSITION="${METAWORLD_MAX_TOTAL_TRANSITION:-500000}"
DRAKE_TASKS="${DRAKE_TASKS:-hammer,spatula,knife,wrench}"
PYBULLET_TRIFINGER_EPISODES="${PYBULLET_TRIFINGER_EPISODES:-1000}"

# ── helpers ──────────────────────────────────────────────────────────────────

zarr_is_complete() {
  [ -f "${DATA_DIR}/$1/.zarr_complete" ]
}

# After LocalTrajDataset writes data/zarr_<name>, move it to DATA_DIR and
# create a data/zarr_<name> -> DATA_DIR/zarr_<name> symlink.  Stamps a
# .zarr_complete sentinel so reruns skip fully finished zarrs.
move_and_link_zarr() {
  local zarr_name="$1"
  local src="${REPO_DIR}/data/${zarr_name}"
  local dst="${DATA_DIR}/${zarr_name}"
  local link="${REPO_DIR}/data/${zarr_name}"
  if [ -d "${src}" ] && [ ! -L "${src}" ]; then
    echo "[zarr] moving ${src} -> ${dst}"
    mv "${src}" "${dst}"
  fi
  if [ -d "${dst}" ]; then
    touch "${dst}/.zarr_complete"
    ln -sfn "$(realpath "${dst}")" "${link}"
    echo "[zarr] data/${zarr_name} -> ${dst}"
  else
    echo "WARNING: ${zarr_name} not found after generation" >&2
  fi
}

# Symlink data/<name> -> $DATA_DIR/<name> so rollout runners that hardcode
# "data/..." paths can find the raw datasets on the external drive.
create_raw_data_symlinks() {
  local pairs=(
    "robomimic:robomimic"
    "adroit:adroit"
    "demo_drake:demo_drake"
    "fleet_tools:fleet_tools"
    "arnold:arnold"
    "maniskill:maniskill"
  )
  for pair in "${pairs[@]}"; do
    local drive_name="${pair%%:*}"
    local link_name="${pair##*:}"
    local target="${DATA_DIR}/${drive_name}"
    local link="${REPO_DIR}/data/${link_name}"
    if [ ! -d "${target}" ]; then
      continue
    fi
    if [ -d "${link}" ] && [ ! -L "${link}" ]; then
      echo "[symlink] skipping data/${link_name}: exists as a real directory"
      continue
    fi
    ln -sfn "$(realpath "${target}")" "${link}"
    echo "[symlink] data/${link_name} -> ${target}"
  done
}

# ── per-domain zarr generators ────────────────────────────────────────────────

generate_zarr_mujoco_metaworld() {
  local zarr_name="zarr_mujoco_metaworld"
  if zarr_is_complete "${zarr_name}"; then
    echo "[zarr] mujoco_metaworld: exists at ${DATA_DIR}/${zarr_name}"
    ln -sfn "$(realpath "${DATA_DIR}/${zarr_name}")" "${REPO_DIR}/data/${zarr_name}"
    return 0
  fi
  rm -rf "${DATA_DIR}/${zarr_name}" "${REPO_DIR}/data/${zarr_name}"
  echo "[zarr] mujoco_metaworld: generating online (tasks=${METAWORLD_TASKS}, ${METAWORLD_EPISODES} eps/task)"
  python - <<PYEOF
import os, sys
sys.path.insert(0, '${REPO_DIR}')
os.chdir('${REPO_DIR}')
from env.mujoco.metaworld.rollout_runner import generate_dataset_rollouts
from hpt.dataset.local_traj_dataset import LocalTrajDataset

tasks = '${METAWORLD_TASKS}'
if tasks != 'all':
    tasks = [t.strip() for t in tasks.split(',') if t.strip()]

LocalTrajDataset(
    dataset_name='mujoco_metaworld',
    env_rollout_fn=generate_dataset_rollouts(
        env_names=tasks,
        save_video=False,
        max_total_transition=${METAWORLD_MAX_TOTAL_TRANSITION},
        episode_num_pertask=${METAWORLD_EPISODES},
    ),
    use_disk=True,
    episode_cnt=${METAWORLD_EPISODES},
    precompute_feat=False,
    observation_horizon=4,
    action_horizon=8,
)
print('[zarr] mujoco_metaworld done')
PYEOF
  move_and_link_zarr "${zarr_name}"
}

generate_zarr_mujoco_robomimic() {
  local zarr_name="zarr_mujoco_robomimic"
  if zarr_is_complete "${zarr_name}"; then
    echo "[zarr] mujoco_robomimic: exists at ${DATA_DIR}/${zarr_name}"
    ln -sfn "$(realpath "${DATA_DIR}/${zarr_name}")" "${REPO_DIR}/data/${zarr_name}"
    return 0
  fi
  if [ ! -d "${DATA_DIR}/robomimic" ]; then
    echo "ERROR: mujoco_robomimic: raw data not found at ${DATA_DIR}/robomimic" >&2
    echo "       Run local_download_sim_data.sh first." >&2
    return 1
  fi
  rm -rf "${DATA_DIR}/${zarr_name}" "${REPO_DIR}/data/${zarr_name}"
  echo "[zarr] mujoco_robomimic: generating from ${DATA_DIR}/robomimic"
  python - <<PYEOF
import os, sys
sys.path.insert(0, '${REPO_DIR}')
os.chdir('${REPO_DIR}')
from env.mujoco.robomimic.rollout_runner import generate_dataset_rollouts
from hpt.dataset.local_traj_dataset import LocalTrajDataset

LocalTrajDataset(
    dataset_name='mujoco_robomimic',
    env_rollout_fn=generate_dataset_rollouts(
        env_names='${ROBOMIMIC_TASKS}',
        dataset_root='data/robomimic',
        download=False,
        convert=False,
        max_total_transition=500000,
        episode_num_pertask=${EPISODE_CNT},
    ),
    use_disk=True,
    episode_cnt=${EPISODE_CNT},
    precompute_feat=False,
    observation_horizon=4,
    action_horizon=8,
)
print('[zarr] mujoco_robomimic done')
PYEOF
  move_and_link_zarr "${zarr_name}"
}

generate_zarr_mujoco_adroit() {
  local zarr_name="zarr_mujoco_adroit"
  if zarr_is_complete "${zarr_name}"; then
    echo "[zarr] mujoco_adroit: exists at ${DATA_DIR}/${zarr_name}"
    ln -sfn "$(realpath "${DATA_DIR}/${zarr_name}")" "${REPO_DIR}/data/${zarr_name}"
    return 0
  fi
  if [ ! -d "${DATA_DIR}/adroit" ]; then
    echo "ERROR: mujoco_adroit: raw data not found at ${DATA_DIR}/adroit" >&2
    echo "       Run local_download_sim_data.sh first." >&2
    return 1
  fi
  rm -rf "${DATA_DIR}/${zarr_name}" "${REPO_DIR}/data/${zarr_name}"
  echo "[zarr] mujoco_adroit: generating from ${DATA_DIR}/adroit"
  python - <<PYEOF
import os, sys
sys.path.insert(0, '${REPO_DIR}')
os.chdir('${REPO_DIR}')
from env.mujoco.adroit.rollout_runner import generate_dataset_rollouts
from hpt.dataset.local_traj_dataset import LocalTrajDataset

LocalTrajDataset(
    dataset_name='mujoco_adroit',
    env_rollout_fn=generate_dataset_rollouts(
        env_names='${ADROIT_TASKS}',
        dataset_root='data/adroit',
        download=False,
        max_total_transition=500000,
        episode_num_pertask=${EPISODE_CNT},
    ),
    use_disk=True,
    episode_cnt=${EPISODE_CNT},
    precompute_feat=False,
    observation_horizon=4,
    action_horizon=8,
)
print('[zarr] mujoco_adroit done')
PYEOF
  move_and_link_zarr "${zarr_name}"
}

generate_zarr_drake_toulouse() {
  local zarr_name="zarr_drake_toulouse"
  if zarr_is_complete "${zarr_name}"; then
    echo "[zarr] drake_toulouse: exists at ${DATA_DIR}/${zarr_name}"
    ln -sfn "$(realpath "${DATA_DIR}/${zarr_name}")" "${REPO_DIR}/data/${zarr_name}"
    return 0
  fi
  if [ ! -d "${DATA_DIR}/demo_drake" ] && [ ! -d "${DATA_DIR}/fleet_tools" ]; then
    echo "ERROR: drake_toulouse: demos not found at ${DATA_DIR}/demo_drake or ${DATA_DIR}/fleet_tools" >&2
    echo "       Run local_download_sim_data.sh first." >&2
    return 1
  fi
  rm -rf "${DATA_DIR}/${zarr_name}" "${REPO_DIR}/data/${zarr_name}"
  echo "[zarr] drake_toulouse: generating from ${DATA_DIR}/demo_drake"
  python - <<PYEOF
import os, sys
sys.path.insert(0, '${REPO_DIR}')
os.chdir('${REPO_DIR}')
from env.drake.toulouse.rollout_runner import generate_dataset_rollouts
from hpt.dataset.local_traj_dataset import LocalTrajDataset

LocalTrajDataset(
    dataset_name='drake_toulouse',
    env_rollout_fn=generate_dataset_rollouts(
        env_names='${DRAKE_TASKS}',
        fleet_root='external/Fleet-Tools',
        fleet_data_root='data/fleet_tools',
        processed_demo_root='data/demo_drake',
        generate=False,
        max_total_transition=500000,
        episode_num_pertask=${EPISODE_CNT},
    ),
    use_disk=True,
    episode_cnt=${EPISODE_CNT},
    precompute_feat=False,
    observation_horizon=4,
    action_horizon=8,
)
print('[zarr] drake_toulouse done')
PYEOF
  move_and_link_zarr "${zarr_name}"
}

generate_zarr_pybullet_trifinger() {
  local zarr_name="zarr_pybullet_trifinger"
  if zarr_is_complete "${zarr_name}"; then
    echo "[zarr] pybullet_trifinger: exists at ${DATA_DIR}/${zarr_name}"
    ln -sfn "$(realpath "${DATA_DIR}/${zarr_name}")" "${REPO_DIR}/data/${zarr_name}"
    return 0
  fi
  rm -rf "${DATA_DIR}/${zarr_name}" "${REPO_DIR}/data/${zarr_name}"
  echo "[zarr] pybullet_trifinger: generating online (${PYBULLET_TRIFINGER_EPISODES} eps/task)"
  pip install --quiet "trifinger_simulation" "numpy<2.0"
  python - <<PYEOF
import os, sys
sys.path.insert(0, '${REPO_DIR}')
os.chdir('${REPO_DIR}')
from env.pybullet.trifinger.rollout_runner import generate_dataset_rollouts
from hpt.dataset.local_traj_dataset import LocalTrajDataset

episode_per_task = ${PYBULLET_TRIFINGER_EPISODES}
LocalTrajDataset(
    dataset_name='pybullet_trifinger',
    env_rollout_fn=generate_dataset_rollouts(
        env_names=['cube_reach', 'cube_push', 'cube_lift'],
        online=True,
        max_total_transition=500000,
        episode_num_pertask=episode_per_task,
    ),
    use_disk=True,
    episode_cnt=episode_per_task * 3,
    precompute_feat=False,
    observation_horizon=4,
    action_horizon=8,
)
print('[zarr] pybullet_trifinger done')
PYEOF
  move_and_link_zarr "${zarr_name}"
}

generate_zarr_isaac_arnold_image() {
  local zarr_name="zarr_isaac_arnold_image"
  if zarr_is_complete "${zarr_name}"; then
    echo "[zarr] isaac_arnold_image: exists at ${DATA_DIR}/${zarr_name}"
    ln -sfn "$(realpath "${DATA_DIR}/${zarr_name}")" "${REPO_DIR}/data/${zarr_name}"
    return 0
  fi
  if [ ! -d "${DATA_DIR}/arnold" ]; then
    echo "ERROR: isaac_arnold_image: raw data not found at ${DATA_DIR}/arnold" >&2
    echo "       Run local_download_sim_data.sh first." >&2
    return 1
  fi
  rm -rf "${DATA_DIR}/${zarr_name}" "${REPO_DIR}/data/${zarr_name}"
  echo "[zarr] isaac_arnold_image: generating from ${DATA_DIR}/arnold"
  pip install --quiet usd-core 2>/dev/null || true
  python - <<PYEOF
import os, sys
sys.path.insert(0, '${REPO_DIR}')
os.chdir('${REPO_DIR}')
from env.isaac.arnold_image.rollout_runner import generate_dataset_rollouts
from hpt.dataset.local_traj_dataset import LocalTrajDataset

LocalTrajDataset(
    dataset_name='isaac_arnold_image',
    env_rollout_fn=generate_dataset_rollouts(
        env_names=['close_cabinet', 'close_drawer', 'open_cabinet', 'open_drawer',
                   'pickup_object', 'pour_water', 'reorient_object', 'transfer_water'],
        dataset_root='data/arnold',
        split='train',
        max_total_transition=500000,
        episode_num_pertask=${EPISODE_CNT},
    ),
    use_disk=True,
    episode_cnt=${EPISODE_CNT},
    pad_after=10,  # from isaac_arnold_image.yaml: action_horizon(8) + observation_horizon(4) - 2
    precompute_feat=False,
    observation_horizon=4,
    action_horizon=8,
)
print('[zarr] isaac_arnold_image done')
PYEOF
  move_and_link_zarr "${zarr_name}"
}

generate_zarr_maniskill() {
  local zarr_name="zarr_maniskill"
  if zarr_is_complete "${zarr_name}"; then
    echo "[zarr] maniskill: exists at ${DATA_DIR}/${zarr_name}"
    ln -sfn "$(realpath "${DATA_DIR}/${zarr_name}")" "${REPO_DIR}/data/${zarr_name}"
    return 0
  fi
  if [ ! -d "${DATA_DIR}/maniskill" ]; then
    echo "ERROR: maniskill: raw data not found at ${DATA_DIR}/maniskill" >&2
    echo "       Run local_download_sim_data.sh first." >&2
    return 1
  fi
  rm -rf "${DATA_DIR}/${zarr_name}" "${REPO_DIR}/data/${zarr_name}"
  echo "[zarr] maniskill: generating from ${DATA_DIR}/maniskill"
  python - <<PYEOF
import os, sys
sys.path.insert(0, '${REPO_DIR}')
os.chdir('${REPO_DIR}')
from env.maniskill.offline.rollout_runner import generate_dataset_rollouts
from hpt.dataset.local_traj_dataset import LocalTrajDataset

LocalTrajDataset(
    dataset_name='maniskill',
    env_rollout_fn=generate_dataset_rollouts(
        env_names=['PickCube-v0'],
        dataset_root='data/maniskill',
        max_total_transition=500000,
        episode_num_pertask=${EPISODE_CNT},
    ),
    use_disk=True,
    episode_cnt=${EPISODE_CNT},
    precompute_feat=False,
    observation_horizon=4,
    action_horizon=8,
)
print('[zarr] maniskill done')
PYEOF
  move_and_link_zarr "${zarr_name}"
}

# ── main ──────────────────────────────────────────────────────────────────────

create_raw_data_symlinks

IFS=',' read -r -a requested_datasets <<< "${DATASETS}"
for dataset in "${requested_datasets[@]}"; do
  case "${dataset}" in
    mujoco_metaworld)            generate_zarr_mujoco_metaworld ;;
    mujoco_robomimic|robomimic)  generate_zarr_mujoco_robomimic ;;
    mujoco_adroit|adroit)        generate_zarr_mujoco_adroit ;;
    drake_toulouse)              generate_zarr_drake_toulouse ;;
    pybullet_trifinger)          generate_zarr_pybullet_trifinger ;;
    isaac_arnold_image)          generate_zarr_isaac_arnold_image ;;
    maniskill)                   generate_zarr_maniskill ;;
    "")                          ;;
    *)  echo "WARNING: unsupported dataset '${dataset}', skipping" >&2 ;;
  esac
done

echo "[done] zarr datasets: ${DATASETS}"
echo "[done] zarr root: ${DATA_DIR}"
