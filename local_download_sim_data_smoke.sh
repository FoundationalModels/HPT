#!/bin/bash

set -euo pipefail

REPO_DIR=${REPO_DIR:-$(pwd)}
cd "${REPO_DIR}"

export DATA_DIR="${DATA_DIR:-${REPO_DIR}/data}"
export HF_HOME="${HF_HOME:-${DATA_DIR}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_DIR}/.cache/pip}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/hrilab/.mujoco/mujoco210/bin:/usr/lib/nvidia"

mkdir -p "${DATA_DIR}" "${HF_HOME}" "${PIP_CACHE_DIR}"

SMOKE_EPISODES="${SMOKE_EPISODES:-3}"
DATASETS="${DATASETS:-robomimic,adroit,pybullet_trifinger,drake_toulouse,isaac_arnold_image,maniskill}"
DATASETS="${DATASETS// /}"

ROBOMIMIC_TASKS="${ROBOMIMIC_TASKS:-can}"
ROBOMIMIC_CONVERT_N="${ROBOMIMIC_CONVERT_N:-${SMOKE_EPISODES}}"
ADROIT_TASKS="${ADROIT_TASKS:-door}"
DRAKE_TASKS="${DRAKE_TASKS:-hammer}"
DRAKE_EPISODES="${DRAKE_EPISODES:-${SMOKE_EPISODES}}"
MANISKILL_HF_REPO="${MANISKILL_HF_REPO:-haosulab/ManiSkill2}"
MANISKILL_HF_INCLUDE="${MANISKILL_HF_INCLUDE:-demos/v0/rigid_body/PickCube-v0/**}"
ARNOLD_DRIVE_URL="${ARNOLD_DRIVE_URL:-https://drive.google.com/drive/folders/1yaEItqU9_MdFVQmkKA6qSvfXy_cPnKGA?usp=sharing}"
ALLOW_FULL_ARCHIVE_DOWNLOADS="${ALLOW_FULL_ARCHIVE_DOWNLOADS:-1}"
PYBULLET_GRASPING_IMAGE_HF_REPO="${PYBULLET_GRASPING_IMAGE_HF_REPO:-}"
PYBULLET_TRIFINGER_HF_REPO="${PYBULLET_TRIFINGER_HF_REPO:-}"
TRIFINGER_TASKS="${TRIFINGER_TASKS:-cube_reach}"
TRIFINGER_ONLINE="${TRIFINGER_ONLINE:-1}"
TRIFINGER_HORIZON="${TRIFINGER_HORIZON:-25}"

contains_dataset() {
  case ",${DATASETS}," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

ensure_python_module() {
  local module_name="$1"
  local package_name="${2:-$1}"
  if python -c "import ${module_name}" >/dev/null 2>&1; then
    return 0
  fi
  python -m pip install --user "${package_name}"
}

ensure_huggingface_cli() {
  if command -v huggingface-cli >/dev/null 2>&1; then
    return 0
  fi
  ensure_python_module huggingface_hub "huggingface_hub[cli]"
  export PATH="${HOME}/.local/bin:${PATH}"
}

download_file() {
  local url="$1"
  local output="$2"
  if [ -s "${output}" ]; then
    echo "[download] exists: ${output}"
    return 0
  fi
  mkdir -p "$(dirname "${output}")"
  echo "[download] ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 10 -o "${output}" "${url}"
  else
    wget --tries=5 --waitretry=10 -O "${output}" "${url}"
  fi
}

download_hf_repo() {
  local repo_id="$1"
  local target_dir="$2"
  local include_pattern="${3:-**}"
  if [ -z "${repo_id}" ]; then
    echo "[huggingface] missing repo id for target ${target_dir}" >&2
    return 2
  fi
  ensure_huggingface_cli
  mkdir -p "${target_dir}"
  echo "[huggingface] ${repo_id} -> ${target_dir} (${include_pattern})"
  huggingface-cli download "${repo_id}" \
    --repo-type dataset \
    --include "${include_pattern}" \
    --local-dir "${target_dir}"
}

download_robomimic() {
  IFS=',' read -r -a tasks <<< "${ROBOMIMIC_TASKS}"
  for task in "${tasks[@]}"; do
    raw_path="${DATA_DIR}/robomimic/${task}/ph/demo_v141.hdf5"
    image_path="${DATA_DIR}/robomimic/${task}/ph/image_v141.hdf5"
    if [ -s "${raw_path}" ]; then
      echo "[robomimic] exists: ${raw_path}"
    else
      echo "[robomimic] raw PH dataset for ${task}"
      python -m robomimic.scripts.download_datasets \
        --download_dir "${DATA_DIR}/robomimic" \
        --tasks "${task}" \
        --dataset_types ph \
        --hdf5_types raw
    fi

    if [ -s "${image_path}" ]; then
      echo "[robomimic] exists: ${image_path}"
      continue
    fi

    case "${task}" in
      lift|can|square)
        cameras=(agentview robot0_eye_in_hand)
        height=84
        width=84
        ;;
      transport)
        cameras=(shouldercamera0 shouldercamera1 robot0_eye_in_hand robot1_eye_in_hand)
        height=84
        width=84
        ;;
      tool_hang)
        cameras=(sideview robot0_eye_in_hand)
        height=240
        width=240
        ;;
      *)
        echo "[robomimic] unknown camera config for task ${task}" >&2
        return 2
        ;;
    esac

    echo "[robomimic] converting first ${ROBOMIMIC_CONVERT_N} demos from ${raw_path}"
    python -m robomimic.scripts.dataset_states_to_obs \
      --dataset "${raw_path}" \
      --output_name image_v141.hdf5 \
      --done_mode 2 \
      --camera_names "${cameras[@]}" \
      --camera_height "${height}" \
      --camera_width "${width}" \
      --n "${ROBOMIMIC_CONVERT_N}"
  done
}

download_adroit() {
  IFS=',' read -r -a tasks <<< "${ADROIT_TASKS}"
  for task in "${tasks[@]}"; do
    case "${task}" in
      door|hammer|pen|relocate)
        url="https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg_v1/${task}-expert-v1.hdf5"
        ;;
      *)
        echo "[adroit] unknown task ${task}" >&2
        return 2
        ;;
    esac
    download_file "${url}" "${DATA_DIR}/adroit/${task}/${task}-expert-v1.hdf5"
  done
}

download_pybullet_grasping_image() {
  if [ -z "${PYBULLET_GRASPING_IMAGE_HF_REPO}" ]; then
    echo "[pybullet_grasping_image] Set PYBULLET_GRASPING_IMAGE_HF_REPO for local smoke download." >&2
    return 2
  fi
  download_hf_repo "${PYBULLET_GRASPING_IMAGE_HF_REPO}" "${DATA_DIR}/pybullet_grasping_image" "${PYBULLET_GRASPING_IMAGE_HF_INCLUDE:-**}"
}

download_pybullet_trifinger() {
  if [ -n "${PYBULLET_TRIFINGER_HF_REPO}" ]; then
    download_hf_repo "${PYBULLET_TRIFINGER_HF_REPO}" "${DATA_DIR}/pybullet_trifinger" "${PYBULLET_TRIFINGER_HF_INCLUDE:-**}"
    return 0
  fi
  if [ "${TRIFINGER_ONLINE}" != "1" ]; then
    echo "[pybullet_trifinger] Set PYBULLET_TRIFINGER_HF_REPO, or set TRIFINGER_ONLINE=1 for local smoke generation." >&2
    return 2
  fi

  IFS=',' read -r -a tasks <<< "${TRIFINGER_TASKS}"
  for task in "${tasks[@]}"; do
    output_path="${DATA_DIR}/pybullet_trifinger/${task}/demo_state.npz"
    if [ -s "${output_path}" ]; then
      echo "[pybullet_trifinger] exists: ${output_path}"
      continue
    fi
    mkdir -p "$(dirname "${output_path}")"
    echo "[pybullet_trifinger] generating ${SMOKE_EPISODES} smoke episodes for ${task}"
    TASK_NAME="${task}" OUTPUT_PATH="${output_path}" EPISODES="${SMOKE_EPISODES}" HORIZON="${TRIFINGER_HORIZON}" python - <<'PY'
import os
import numpy as np

from env.pybullet.trifinger.rollout_runner import generate_dataset_rollouts

task_name = os.environ["TASK_NAME"]
output_path = os.environ["OUTPUT_PATH"]
episode_count = int(os.environ["EPISODES"])
horizon = int(os.environ["HORIZON"])

actions = []
states = []
dones = []
for episode in generate_dataset_rollouts(
    env_names=[task_name],
    episode_num_pertask=episode_count,
    horizon=horizon,
    online=True,
    visualization=False,
):
    steps = episode["steps"]
    for idx, step in enumerate(steps):
        actions.append(np.asarray(step["action"], dtype=np.float32))
        states.append(np.asarray(step["observation"]["state"], dtype=np.float32))
        dones.append(idx == len(steps) - 1)

np.savez_compressed(
    output_path,
    action=np.asarray(actions, dtype=np.float32),
    state=np.asarray(states, dtype=np.float32),
    done=np.asarray(dones, dtype=bool),
)
print(f"[pybullet_trifinger] wrote {len(actions)} transitions to {output_path}")
PY
  done
}

fleet_env_name() {
  case "$1" in
    hammer) echo FrankaDrakeHammerEnv ;;
    spatula) echo FrankaDrakeSpatulaEnv ;;
    knife) echo FrankaDrakeKnifeEnv ;;
    wrench) echo FrankaDrakeWrenchEnv ;;
    *) return 2 ;;
  esac
}

fleet_task_config() {
  case "$1" in
    hammer) echo FrankaDrakeHammerEnvMergingWeights ;;
    spatula) echo FrankaDrakeSpatulaEnvMergingWeights ;;
    knife) echo FrankaDrakeKnifeEnvMergingWeights ;;
    wrench) echo FrankaDrakeWrenchEnvMergingWeights ;;
    *) return 2 ;;
  esac
}

download_drake_toulouse() {
  if [ ! -d external/Fleet-Tools ]; then
    git clone https://github.com/FoundationalModels/Fleet-Tools.git external/Fleet-Tools
  fi

  pushd external/Fleet-Tools >/dev/null
  IFS=',' read -r -a tasks <<< "${DRAKE_TASKS}"
  for task in "${tasks[@]}"; do
    env_name="$(fleet_env_name "${task}")"
    task_config="$(fleet_task_config "${task}")"
    raw_demo_root="${DATA_DIR}/fleet_tools/demonstrations"
    processed_root="${DATA_DIR}/fleet_tools/demonstrations/processed"
    processed_demo_root="${DATA_DIR}/demo_drake"
    collapsed_path="${processed_demo_root}/FrankaDrake${task^}for${task}Env-Tool0/demo_state.npz"

    if [ -s "${collapsed_path}" ]; then
      echo "[fleet-tools] exists: ${collapsed_path}"
      continue
    fi

    echo "[fleet-tools] generating ${DRAKE_EPISODES} smoke episodes for ${task}"
    python -m core.run \
      cuda=False \
      render=False \
      env_name="${env_name}" \
      num_envs=1 \
      run_expert=True \
      save_demonstrations=True \
      demonstration_dir="${raw_demo_root}" \
      start_episode_position=0 \
      num_workers=0 \
      task="${task_config}" \
      train=FrankaDrakeEnv \
      save_demo_suffix=tool_0 \
      task.tool_fix_idx=0 \
      max_episodes="${DRAKE_EPISODES}" \
      num_episode="${DRAKE_EPISODES}" \
      record_video=False \
      training=False \
      +task.data_collection=True \
      task.env.use_image=False

    python -m scripts.collapse_dataset \
      -e "${task}for${task}" \
      --tool 0 \
      --max_num "$((DRAKE_EPISODES * 200))" \
      --saved_path "${raw_demo_root}" \
      --output_path "${processed_root}" \
      --processed_output_path "${processed_demo_root}"
  done
  popd >/dev/null
}

download_arnold() {
  if [ -d "${DATA_DIR}/arnold/tasks" ]; then
    echo "[arnold] exists: ${DATA_DIR}/arnold/tasks"
    return 0
  fi
  if [ "${ALLOW_FULL_ARCHIVE_DOWNLOADS}" != "1" ]; then
    echo "[arnold] The known source is a Google Drive folder download, not per-episode files." >&2
    echo "[arnold] Set ALLOW_FULL_ARCHIVE_DOWNLOADS=1 to pull the full Arnold archive locally." >&2
    return 2
  fi
  ensure_python_module gdown gdown
  ensure_python_module pxr usd-core
  python -m gdown --folder "${ARNOLD_DRIVE_URL}" --output "${DATA_DIR}/arnold"
}

download_maniskill() {
  download_hf_repo "${MANISKILL_HF_REPO}" "${DATA_DIR}/maniskill" "${MANISKILL_HF_INCLUDE}"
}

if contains_dataset robomimic; then
  download_robomimic
fi
if contains_dataset adroit; then
  download_adroit
fi
if contains_dataset pybullet_grasping_image; then
  download_pybullet_grasping_image
fi
if contains_dataset pybullet_trifinger; then
  download_pybullet_trifinger
fi
if contains_dataset drake_toulouse; then
  download_drake_toulouse
fi
if contains_dataset isaac_arnold_image; then
  download_arnold
fi
if contains_dataset maniskill; then
  download_maniskill
fi

echo "[done] local smoke datasets: ${DATASETS}"
echo "[done] data root: ${DATA_DIR}"
