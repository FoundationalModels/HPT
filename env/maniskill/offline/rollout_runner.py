# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np

from env.offline_source_adapter import UnsupportedRolloutRunner, as_task_list, missing_source_error


DEFAULT_TASKS = ["PickCube-v0"]
TASK_LANGUAGE = {
    "PickCube-v0": "pick up the cube.",
}
TASK_PATHS = {
    "PickCube-v0": Path("demos") / "v0" / "rigid_body" / "PickCube-v0" / "trajectory.h5",
}
SOURCE_NOTE = (
    "Download ManiSkill demos into data/maniskill with Hugging Face, for example "
    "data/maniskill/demos/v0/rigid_body/PickCube-v0/trajectory.h5."
)


def _task_h5_path(dataset_root, task):
    root = Path(dataset_root)
    candidates = [
        root / TASK_PATHS.get(task, Path(task) / "trajectory.h5"),
        root / task / "trajectory.h5",
        Path("data") / "maniskill" / TASK_PATHS.get(task, Path(task) / "trajectory.h5"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _demo_names(h5_file):
    return sorted(
        h5_file.keys(),
        key=lambda name: int(name.split("_")[-1]) if name.startswith("traj_") else name,
    )


def generate_dataset_rollouts(
    env_names,
    dataset_root="data/maniskill",
    max_total_transition=500000,
    episode_num_pertask=100,
    **kwargs,
):
    del kwargs
    for task in as_task_list(env_names, DEFAULT_TASKS):
        path = _task_h5_path(dataset_root, task)
        if not path.exists():
            raise missing_source_error("maniskill", task, path, SOURCE_NOTE)

        language = TASK_LANGUAGE.get(task, task)
        total_transitions = 0
        with h5py.File(path, "r") as f:
            for demo_name in _demo_names(f)[:episode_num_pertask]:
                demo = f[demo_name]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                states = np.asarray(demo["env_states"], dtype=np.float32)[: len(actions)]
                steps = [
                    OrderedDict(
                        observation={"state": states[idx].reshape(-1)},
                        action=actions[idx],
                        language_instruction=language,
                    )
                    for idx in range(len(actions))
                ]
                if not steps:
                    continue

                total_transitions += len(steps)
                yield {"steps": steps}
                if max_total_transition is not None and total_transitions >= max_total_transition:
                    break


RolloutRunner = UnsupportedRolloutRunner
