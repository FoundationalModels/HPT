#!/bin/bash
#SBATCH --job-name=hpt_download_sim_data
#SBATCH --output=/cluster/tufts/hrilab/hlu07/HPT/logs/hpt_download_sim_data_%j.out
#SBATCH --error=/cluster/tufts/hrilab/hlu07/HPT/logs/hpt_download_sim_data_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p gpu,preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hlu07@tufts.edu

set -euo pipefail

module load singularity/3.8.4

REPO_DIR=${REPO_DIR:-/cluster/tufts/hrilab/hlu07/HPT}
CONTAINER=${CONTAINER:-/cluster/tufts/hrilab/hlu07/hpt.sif}

mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/data"

singularity exec \
  --bind "${REPO_DIR}:/workspace" \
  "${CONTAINER}" \
  bash -s <<'IN_CONTAINER'
set -euo pipefail

cd /workspace

export HYDRA_FULL_ERROR=1
export DATA_DIR="${DATA_DIR:-/workspace/data}"
export HF_HOME="${HF_HOME:-${DATA_DIR}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_DIR}/.cache/pip}"

mkdir -p "${DATA_DIR}" "${HF_HOME}" "${PIP_CACHE_DIR}"

DATASETS="${DATASETS:-robomimic,adroit,pybullet_trifinger,drake_toulouse,isaac_arnold_image,maniskill}"
DATASETS="${DATASETS// /}"

ROBOMIMIC_TASKS="${ROBOMIMIC_TASKS:-lift,can,square,transport,tool_hang}"
ROBOMIMIC_CONVERT_N="${ROBOMIMIC_CONVERT_N:-}"
ADROIT_TASKS="${ADROIT_TASKS:-door,hammer,pen,relocate}"
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
    echo "[robomimic] raw PH dataset for ${task}"
    python -m robomimic.scripts.download_datasets \
      --download_dir "${DATA_DIR}/robomimic" \
      --tasks "${task}" \
      --dataset_types ph \
      --hdf5_types raw

    raw_path="${DATA_DIR}/robomimic/${task}/ph/demo_v141.hdf5"
    image_path="${DATA_DIR}/robomimic/${task}/ph/image_v141.hdf5"
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

download_pybullet_trifinger() {
  local tasks=("cube_reach" "cube_push" "cube_lift")
  local all_exist=true
  for task in "${tasks[@]}"; do
    if [ ! -s "${DATA_DIR}/pybullet_trifinger/${task}/demo_state.npz" ]; then
      all_exist=false
      break
    fi
  done
  if ${all_exist}; then
    echo "[pybullet_trifinger] exists: using cached data"
    return 0
  fi
  echo "[pybullet_trifinger] installing trifinger_simulation for online generation"
  pip install --quiet trifinger_simulation
  echo "[pybullet_trifinger] generating ${PYBULLET_TRIFINGER_EPISODES} episodes per task online"
  python - <<PYEOF
import os, sys
sys.path.insert(0, '/workspace')
from env.pybullet.trifinger.rollout_runner import generate_dataset_rollouts
from hpt.dataset.local_traj_dataset import LocalTrajDataset

data_dir = os.environ.get('DATA_DIR', '/workspace/data')
for task in ['cube_reach', 'cube_push', 'cube_lift']:
    cache = f"{data_dir}/pybullet_trifinger/{task}/demo_state.npz"
    if os.path.exists(cache):
        print(f"[trifinger] {task}: cached")
        continue
    LocalTrajDataset(
        dataset_name='pybullet_trifinger',
        env_rollout_fn=generate_dataset_rollouts(
            env_names=[task],
            dataset_root=f"{data_dir}/pybullet_trifinger",
            online=True,
            max_total_transition=500000,
            episode_num_pertask=int(os.environ.get('PYBULLET_TRIFINGER_EPISODES', 1000)),
        ),
        use_disk=True,
        episode_cnt=int(os.environ.get('PYBULLET_TRIFINGER_EPISODES', 1000)),
        precompute_feat=False,
        observation_horizon=4,
        action_horizon=8,
    )
    print(f"[trifinger] {task}: done")
PYEOF
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

    echo "[fleet-tools] generating ${DRAKE_EPISODES} episodes for ${task}"
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

    echo "[fleet-tools] collapsing ${task}"
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
  ensure_python_module gdown gdown
  ensure_python_module pxr usd-core
  python -m gdown --folder "${ARNOLD_DRIVE_URL}" --output "${DATA_DIR}/arnold"
}

download_maniskill() {
  download_hf_repo "${MANISKILL_HF_REPO}" "${DATA_DIR}/maniskill" "${MANISKILL_HF_INCLUDE:-demos/**}"
}

if contains_dataset robomimic; then
  download_robomimic
fi
if contains_dataset adroit; then
  download_adroit
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

echo "[done] requested datasets: ${DATASETS}"
echo "[done] data root: ${DATA_DIR}"
IN_CONTAINER
