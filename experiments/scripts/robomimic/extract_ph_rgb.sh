#!/usr/bin/env bash
set -euo pipefail

# Convert robomimic PH raw datasets under ./data into RGB image datasets.
# Usage:
#   bash experiments/scripts/robomimic/extract_ph_rgb.sh [DATA_DIR] [CONDA_ENV]
# Example:
#   bash experiments/scripts/robomimic/extract_ph_rgb.sh ./data hpt311

DATA_DIR="${1:-./data/sim/robomimic_ph_raw}"
CONDA_ENV="${2:-hpt311}"

run_extract() {
  local dataset_path="$1"
  local output_name="$2"
  local camera_height="$3"
  local camera_width="$4"
  shift 4
  local camera_names=("$@")

  if [[ ! -f "$dataset_path" ]]; then
    echo "[skip] Missing raw dataset: $dataset_path"
    return 0
  fi

  local output_path
  output_path="$(dirname "$dataset_path")/$output_name"

  echo "[run] Converting $dataset_path -> $output_path"
  conda run -n "$CONDA_ENV" python -m robomimic.scripts.dataset_states_to_obs \
    --dataset "$dataset_path" \
    --output_name "$output_name" \
    --done_mode 2 \
    --camera_names "${camera_names[@]}" \
    --camera_height "$camera_height" \
    --camera_width "$camera_width"
}

echo "Using DATA_DIR=$DATA_DIR"
echo "Using CONDA_ENV=$CONDA_ENV"

# PH task camera settings from robomimic's extract_obs_from_raw_datasets.sh
run_extract "$DATA_DIR/lift/ph/demo_v141.hdf5" image_v141.hdf5 84 84 \
  agentview robot0_eye_in_hand

run_extract "$DATA_DIR/can/ph/demo_v141.hdf5" image_v141.hdf5 84 84 \
  agentview robot0_eye_in_hand

run_extract "$DATA_DIR/square/ph/demo_v141.hdf5" image_v141.hdf5 84 84 \
  agentview robot0_eye_in_hand

run_extract "$DATA_DIR/transport/ph/demo_v141.hdf5" image_v141.hdf5 84 84 \
  shouldercamera0 shouldercamera1 robot0_eye_in_hand robot1_eye_in_hand

run_extract "$DATA_DIR/tool_hang/ph/demo_v141.hdf5" image_v141.hdf5 240 240 \
  sideview robot0_eye_in_hand

echo "Done."
