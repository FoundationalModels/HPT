#!/bin/bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")" && pwd)}
DATA_DIR=${DATA_DIR:-/media/hrilab/HelenMacPc}

if [ ! -d "${DATA_DIR}" ]; then
  echo "ERROR: data dir not found: ${DATA_DIR}" >&2
  echo "Make sure the external drive is mounted before running this script." >&2
  exit 1
fi

export DATA_DIR
export HF_HOME="${HF_HOME:-${DATA_DIR}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_DIR}/.cache/pip}"
export HYDRA_FULL_ERROR=1

mkdir -p "${DATA_DIR}" "${HF_HOME}" "${PIP_CACHE_DIR}"

cd "${REPO_DIR}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/experiments/configs/config.yaml}"
CONFIG_DOMAINS="$(
  sed -n 's/^[[:space:]]*domains:[[:space:]]*//p' "${CONFIG_PATH}" \
    | head -n 1 \
    | sed 's/[[:space:]]*#.*$//' \
    | tr -d "\"'"
)"

DATASETS="${DATASETS:-${CONFIG_DOMAINS}}"
DATASETS="${DATASETS// /}"

ROBOMIMIC_TASKS="${ROBOMIMIC_TASKS:-lift,can,square,transport,tool_hang}"
ROBOMIMIC_CONVERT_N="${ROBOMIMIC_CONVERT_N:-}"
ADROIT_TASKS="${ADROIT_TASKS:-door,hammer,pen,relocate}"
METAWORLD_TASKS="${METAWORLD_TASKS:-all}"
METAWORLD_EPISODES="${METAWORLD_EPISODES:-1200}"
METAWORLD_MAX_TOTAL_TRANSITION="${METAWORLD_MAX_TOTAL_TRANSITION:-500000}"
DRAKE_TASKS="${DRAKE_TASKS:-hammer,spatula,knife,wrench}"
DRAKE_EPISODES="${DRAKE_EPISODES:-300}"
ARNOLD_DRIVE_URL="${ARNOLD_DRIVE_URL:-https://drive.google.com/drive/folders/1yaEItqU9_MdFVQmkKA6qSvfXy_cPnKGA?usp=sharing}"
MANISKILL_HF_REPO="${MANISKILL_HF_REPO:-haosulab/ManiSkill2}"
PYBULLET_TRIFINGER_EPISODES="${PYBULLET_TRIFINGER_EPISODES:-1000}"

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
  local tmp="${output}.tmp"
  echo "[download] ${url}"
  if command -v curl >/dev/null 2>&1; then
    # -C - resumes from an existing partial .tmp; falls back to a full download
    # if the server does not support range requests (curl exit 33).
    curl -L --fail --retry 5 --retry-delay 10 -C - -o "${tmp}" "${url}" \
      || curl -L --fail --retry 5 --retry-delay 10 -o "${tmp}" "${url}"
  else
    wget --tries=5 --waitretry=10 -c -O "${tmp}" "${url}"
  fi
  mv "${tmp}" "${output}"
}

# Returns 0 if the file is a readable HDF5, 1 otherwise.
is_valid_hdf5() {
  python -c "import h5py, sys; h5py.File(sys.argv[1], 'r').close()" "$1" 2>/dev/null
}

# Returns 0 if the zarr at DATA_DIR/<name> was fully generated (sentinel present).
zarr_is_complete() {
  [ -f "${DATA_DIR}/$1/.zarr_complete" ]
}

# After LocalTrajDataset writes data/zarr_<name>, move it to DATA_DIR and
# create data/zarr_<name> -> DATA_DIR/zarr_<name> symlink.  Stamps a sentinel
# file so reruns can skip completed zarrs even if data/action already existed
# from a prior interrupted run.
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
  ensure_python_module robomimic robomimic
  IFS=',' read -r -a tasks <<< "${ROBOMIMIC_TASKS}"
  for task in "${tasks[@]}"; do
    raw_path="${DATA_DIR}/robomimic/${task}/ph/demo_v141.hdf5"
    image_path="${DATA_DIR}/robomimic/${task}/ph/image_v141.hdf5"

    if [ -s "${image_path}" ]; then
      if is_valid_hdf5 "${image_path}"; then
        echo "[robomimic] exists: ${image_path}, skipping"
        continue
      fi
      echo "[robomimic] corrupt image HDF5: ${image_path}, removing"
      rm -f "${image_path}"
    fi

    if [ -s "${raw_path}" ] && ! is_valid_hdf5 "${raw_path}"; then
      echo "[robomimic] corrupt raw HDF5: ${raw_path}, removing"
      rm -f "${raw_path}"
    fi

    if [ ! -s "${raw_path}" ]; then
      echo "[robomimic] raw PH dataset for ${task}"
      python -m robomimic.scripts.download_datasets \
        --download_dir "${DATA_DIR}/robomimic" \
        --tasks "${task}" \
        --dataset_types ph \
        --hdf5_types raw
    else
      echo "[robomimic] raw exists: ${raw_path}, skipping download"
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

    echo "[robomimic] converting ${raw_path} -> image_v141.hdf5"
    convert_limit=()
    if [ -n "${ROBOMIMIC_CONVERT_N}" ]; then
      convert_limit=(--n "${ROBOMIMIC_CONVERT_N}")
    fi
    python -m robomimic.scripts.dataset_states_to_obs \
      --dataset "${raw_path}" \
      --output_name image_v141.hdf5 \
      --done_mode 2 \
      --camera_names "${cameras[@]}" \
      --camera_height "${height}" \
      --camera_width "${width}" \
      "${convert_limit[@]}"
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

generate_pybullet_trifinger() {
  local zarr_name="zarr_pybullet_trifinger"
  if zarr_is_complete "${zarr_name}"; then
    echo "[pybullet_trifinger] exists: ${DATA_DIR}/${zarr_name}"
    ln -sfn "$(realpath "${DATA_DIR}/${zarr_name}")" "${REPO_DIR}/data/${zarr_name}"
    return 0
  fi
  # Remove any partial zarr from a previous interrupted run so LocalTrajDataset
  # regenerates from scratch rather than loading an incomplete dataset.
  rm -rf "${DATA_DIR}/${zarr_name}" "${REPO_DIR}/data/${zarr_name}"
  echo "[pybullet_trifinger] installing trifinger_simulation for online generation"
  pip install --quiet "trifinger_simulation" "numpy<2.0"
  echo "[pybullet_trifinger] generating ${PYBULLET_TRIFINGER_EPISODES} episodes per task online"
  python - <<PYEOF
import os, sys
sys.path.insert(0, '${REPO_DIR}')
os.chdir('${REPO_DIR}')
from env.pybullet.trifinger.rollout_runner import generate_dataset_rollouts
from hpt.dataset.local_traj_dataset import LocalTrajDataset

# All three tasks in one call so a single zarr holds all task episodes.
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
print('[trifinger] done')
PYEOF
  move_and_link_zarr "${zarr_name}"
}

generate_mujoco_metaworld() {
  local zarr_name="zarr_mujoco_metaworld"
  if zarr_is_complete "${zarr_name}"; then
    echo "[mujoco_metaworld] exists: ${DATA_DIR}/${zarr_name}"
    ln -sfn "$(realpath "${DATA_DIR}/${zarr_name}")" "${REPO_DIR}/data/${zarr_name}"
    return 0
  fi
  # Remove any partial zarr from a previous interrupted run.
  rm -rf "${DATA_DIR}/${zarr_name}" "${REPO_DIR}/data/${zarr_name}"
  echo "[mujoco_metaworld] generating ${METAWORLD_EPISODES} episodes for tasks: ${METAWORLD_TASKS}"
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
print('[mujoco_metaworld] done')
PYEOF
  move_and_link_zarr "${zarr_name}"
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

count_drake_demos() {
  # Count episode_* subdirectories that contain traj_data.npz
  local demo_dir="$1"
  if [ ! -d "${demo_dir}" ]; then
    echo 0
    return
  fi
  find "${demo_dir}" -maxdepth 2 -name "traj_data.npz" | wc -l
}

max_drake_episode_num() {
  # Return the highest episode number among existing episode_* dirs (for start_episode_position).
  # Using the count instead would collide with existing dirs that have gaps from failed attempts.
  local demo_dir="$1"
  if [ ! -d "${demo_dir}" ]; then
    echo 0
    return
  fi
  find "${demo_dir}" -maxdepth 1 -type d -name "episode_*" \
    | grep -o '[0-9]*$' | sort -n | tail -1 || echo 0
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
    demo_dir="${raw_demo_root}/${task}for${task}tool_0"

    # Drake's C++ backend can segfault after ~100 episodes per process run.
    # Collect demos in a retry loop: each run resumes from the number of demos
    # already on disk so episode indices stay unique across restarts.
    # Cap total attempted episodes to avoid spinning forever if success rate is zero.
    max_total_attempts=$(( DRAKE_EPISODES * 30 ))
    total_attempts=0

    while true; do
      existing_demos=$(count_drake_demos "${demo_dir}")
      echo "[fleet-tools] ${task}: ${existing_demos}/${DRAKE_EPISODES} demos collected"

      if [ "${existing_demos}" -ge "${DRAKE_EPISODES}" ]; then
        break
      fi
      if [ "${total_attempts}" -ge "${max_total_attempts}" ]; then
        echo "[fleet-tools] ${task}: reached attempt limit (${max_total_attempts}) with ${existing_demos}/${DRAKE_EPISODES} demos — stopping" >&2
        break
      fi

      batch=$(( DRAKE_EPISODES < 100 ? DRAKE_EPISODES : 100 ))
      start_pos=$(max_drake_episode_num "${demo_dir}")
      # Recount immediately before launching to avoid off-by-one if disk changed since top of loop.
      remaining=$(( DRAKE_EPISODES - $(count_drake_demos "${demo_dir}") ))
      echo "[fleet-tools] ${task}: running up to ${batch} more episodes (start_episode_position=${start_pos}, need=${remaining})"

      # Temporarily disable exit-on-error so a segfault doesn't kill the script.
      set +e
      python -m core.run \
        cuda=False \
        render=False \
        env_name="${env_name}" \
        num_envs=1 \
        run_expert=True \
        save_demonstrations=True \
        demonstration_dir="${raw_demo_root}" \
        start_episode_position="${start_pos}" \
        num_workers=0 \
        task="${task_config}" \
        train=FrankaDrakeEnv \
        save_demo_suffix=tool_0 \
        task.tool_fix_idx=0 \
        max_episodes="${remaining}" \
        num_episode="${batch}" \
        record_video=False \
        training=False \
        +task.data_collection=True \
        task.env.use_image=False
      run_exit=$?
      set -e

      total_attempts=$(( total_attempts + batch ))

      if [ "${run_exit}" -ne 0 ]; then
        echo "[fleet-tools] ${task}: run exited with code ${run_exit} (likely Drake segfault), retrying..."
      fi
    done

    existing_demos=$(count_drake_demos "${demo_dir}")
    if [ "${existing_demos}" -eq 0 ]; then
      echo "[fleet-tools] ${task}: no demos collected — skipping collapse" >&2
      continue
    fi

    echo "[fleet-tools] collapsing ${task} (${existing_demos} demos)"
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
  local tasks=(close_cabinet close_drawer open_cabinet open_drawer
               pickup_object pour_water reorient_object transfer_water)
  local need_download=false
  for task in "${tasks[@]}"; do
    local zip="${DATA_DIR}/arnold/tasks/${task}.zip"
    if [ ! -s "${zip}" ]; then
      need_download=true
    elif ! python -c "
import zipfile, sys
result = zipfile.ZipFile(sys.argv[1]).testzip()
sys.exit(0 if result is None else 1)
" "${zip}" 2>/dev/null; then
      echo "[arnold] corrupt zip: ${zip}, removing"
      rm -f "${zip}"
      need_download=true
    fi
  done
  if ! ${need_download}; then
    echo "[arnold] all task zips valid: ${DATA_DIR}/arnold/tasks"
    return 0
  fi
  ensure_python_module gdown gdown
  ensure_python_module pxr usd-core
  python -m gdown --folder "${ARNOLD_DRIVE_URL}" --output "${DATA_DIR}/arnold"
}

download_maniskill() {
  download_hf_repo "${MANISKILL_HF_REPO}" "${DATA_DIR}/maniskill" "${MANISKILL_HF_INCLUDE:-demos/**}"
}

# Create symlinks from data/{name} -> $DATA_DIR/{name} so that zarr-generation
# code (which hardcodes "data/..." relative paths) can find the raw datasets on
# the external drive.  Zarr caches generated online (metaworld, trifinger) are
# moved to DATA_DIR and symlinked by move_and_link_zarr.  Skips any entry where
# the drive directory doesn't exist yet or where data/{name} is already a real
# (non-symlink) directory.
create_data_symlinks() {
  # drive_folder : relative path the runners expect under data/
  local pairs=(
    "adroit:adroit"
    "robomimic:robomimic"
    "demo_drake:demo_drake"
    "fleet_tools:fleet_tools"
    "pybullet_trifinger:pybullet_trifinger"
    "arnold:arnold"
    "maniskill:maniskill"
  )

  for pair in "${pairs[@]}"; do
    drive_name="${pair%%:*}"
    link_name="${pair##*:}"
    target="${DATA_DIR}/${drive_name}"
    link="data/${link_name}"

    if [ ! -d "${target}" ]; then
      continue  # not downloaded yet
    fi
    if [ -d "${link}" ] && [ ! -L "${link}" ]; then
      echo "[symlink] skipping ${link}: exists as a real directory (remove it to use drive data)"
      continue
    fi
    ln -sfn "$(realpath "${target}")" "${link}"
    echo "[symlink] data/${link_name} -> ${target}"
  done
}

IFS=',' read -r -a requested_datasets <<< "${DATASETS}"
for dataset in "${requested_datasets[@]}"; do
  case "${dataset}" in
    robomimic|mujoco_robomimic)
      download_robomimic
      ;;
    adroit|mujoco_adroit)
      download_adroit
      ;;
    pybullet_trifinger)
      generate_pybullet_trifinger
      ;;
    mujoco_metaworld)
      generate_mujoco_metaworld
      ;;
    drake_toulouse)
      download_drake_toulouse
      ;;
    isaac_arnold_image)
      download_arnold
      ;;
    maniskill)
      download_maniskill
      ;;
    "")
      ;;
    *)
      echo "ERROR: unsupported dataset/domain requested: ${dataset}" >&2
      exit 2
      ;;
  esac
done

create_data_symlinks

echo "[done] requested datasets: ${DATASETS}"
echo "[done] data root: ${DATA_DIR}"
