# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


TASK_CAMERA_CONFIG = {
    "lift": (84, 84, ["agentview", "robot0_eye_in_hand"]),
    "can": (84, 84, ["agentview", "robot0_eye_in_hand"]),
    "square": (84, 84, ["agentview", "robot0_eye_in_hand"]),
    "transport": (84, 84, ["shouldercamera0", "shouldercamera1", "robot0_eye_in_hand", "robot1_eye_in_hand"]),
    "tool_hang": (240, 240, ["sideview", "robot0_eye_in_hand"]),
}

TASK_LANGUAGE = {
    "lift": "lift the object.",
    "can": "pick up the can and place it in the target bin.",
    "square": "pick up the square nut and place it on the peg.",
    "transport": "transport the object to the target container.",
    "tool_hang": "hang the tool on the frame.",
}


def _as_task_list(env_names):
    if env_names == "all":
        return ["lift", "can", "square", "transport", "tool_hang"]
    if isinstance(env_names, str):
        return [task.strip() for task in env_names.split(",") if task.strip()]
    return list(env_names)


def _raw_dataset_path(dataset_root, task, dataset_type, raw_name):
    return Path(dataset_root) / task / dataset_type / raw_name


def _image_dataset_path(dataset_root, task, dataset_type, image_name):
    return Path(dataset_root) / task / dataset_type / image_name


def _download_raw_dataset(dataset_root, task, dataset_type):
    cmd = [
        sys.executable,
        "-m",
        "robomimic.scripts.download_datasets",
        "--download_dir",
        str(dataset_root),
        "--tasks",
        task,
        "--dataset_types",
        dataset_type,
        "--hdf5_types",
        "raw",
    ]
    print("[robomimic] downloading raw dataset:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _convert_raw_to_image(raw_path, image_name, task, convert_n=None):
    height, width, camera_names = TASK_CAMERA_CONFIG[task]
    cmd = [
        sys.executable,
        "-m",
        "robomimic.scripts.dataset_states_to_obs",
        "--dataset",
        str(raw_path),
        "--output_name",
        image_name,
        "--done_mode",
        "2",
        "--camera_names",
        *camera_names,
        "--camera_height",
        str(height),
        "--camera_width",
        str(width),
    ]
    if convert_n is not None:
        cmd.extend(["--n", str(convert_n)])
    print("[robomimic] converting raw dataset to image dataset:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _ensure_image_dataset(
    dataset_root,
    task,
    dataset_type,
    raw_name,
    image_name,
    download=True,
    convert=True,
    convert_n=None,
):
    raw_path = _raw_dataset_path(dataset_root, task, dataset_type, raw_name)
    image_path = _image_dataset_path(dataset_root, task, dataset_type, image_name)

    if image_path.exists():
        return image_path

    if not raw_path.exists():
        if not download:
            raise FileNotFoundError(
                f"Missing RoboMimic raw dataset {raw_path}. "
                "Set download=True or place the raw PH dataset at this path."
            )
        _download_raw_dataset(dataset_root, task, dataset_type)

    if not image_path.exists():
        if not convert:
            raise FileNotFoundError(
                f"Missing RoboMimic image dataset {image_path}. "
                "Set convert=True or create it with robomimic.scripts.dataset_states_to_obs."
            )
        _convert_raw_to_image(raw_path, image_name=image_name, task=task, convert_n=convert_n)

    if not image_path.exists():
        raise FileNotFoundError(f"Expected RoboMimic image dataset was not created: {image_path}")

    return image_path


def _demo_names(h5_file):
    names = list(h5_file["data"].keys())
    return sorted(names, key=lambda name: int(name.split("_")[-1]) if name.startswith("demo_") else name)


def _select_state(demo_group, step_idx):
    if "obs" in demo_group:
        obs = demo_group["obs"]
        low_dim_keys = [
            key
            for key in sorted(obs.keys())
            if ("image" not in key and np.issubdtype(obs[key].dtype, np.number) and len(obs[key].shape) <= 2)
        ]
        if len(low_dim_keys) > 0:
            return np.concatenate([np.asarray(obs[key][step_idx]).reshape(-1) for key in low_dim_keys], axis=0)

    return np.asarray(demo_group["states"][step_idx]).reshape(-1)


def _select_images(demo_group, step_idx):
    obs = demo_group.get("obs", None)
    if obs is None:
        return {}

    images = {}
    for key in sorted(obs.keys()):
        if "image" not in key:
            continue
        image = np.asarray(obs[key][step_idx])
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.moveaxis(image, 0, -1)
        images[key] = image.astype(np.uint8)
    return images


def _iter_task_episodes(image_path, task, max_episodes=None, max_total_transition=None):
    language = TASK_LANGUAGE.get(task, task)
    total_transitions = 0

    with h5py.File(image_path, "r") as f:
        for demo_name in _demo_names(f):
            demo = f["data"][demo_name]
            actions = np.asarray(demo["actions"])
            steps = []

            for step_idx, action in enumerate(actions):
                observation = {"state": _select_state(demo, step_idx)}
                observation.update(_select_images(demo, step_idx))

                steps.append(
                    {
                        "action": action,
                        "observation": observation,
                        "language_instruction": language,
                    }
                )

            if len(steps) == 0:
                continue

            total_transitions += len(steps)
            yield {"steps": steps}

            if max_episodes is not None and max_episodes > 0:
                max_episodes -= 1
                if max_episodes == 0:
                    break
            if max_total_transition is not None and total_transitions >= max_total_transition:
                break


def generate_dataset_rollouts(
    env_names,
    embodiment="franka",
    dataset_root="data/robomimic",
    dataset_type="ph",
    raw_name="demo_v141.hdf5",
    image_name="image_v141.hdf5",
    download=True,
    convert=True,
    convert_n=None,
    max_total_transition=500000,
    episode_num_pertask=100,
    **kwargs,
):
    """Yield RoboMimic PH trajectories in the format consumed by LocalTrajDataset.

    Raw RoboMimic PH datasets are downloadable, but image PH datasets are generated
    locally from raw states. LocalTrajDataset then writes the HPT zarr cache and
    precomputes ResNet features when dataset.precompute_feat=True.
    """
    del embodiment, kwargs
    tasks = _as_task_list(env_names)
    cycles = max(1, episode_num_pertask // max(1, len(tasks)))

    print("robomimic env names:", tasks)
    for task in tasks:
        if task not in TASK_CAMERA_CONFIG:
            raise ValueError(f"Unsupported RoboMimic task '{task}'. Supported tasks: {sorted(TASK_CAMERA_CONFIG)}")

        image_path = _ensure_image_dataset(
            dataset_root=dataset_root,
            task=task,
            dataset_type=dataset_type,
            raw_name=raw_name,
            image_name=image_name,
            download=download,
            convert=convert,
            convert_n=convert_n,
        )
        print(f"[robomimic] reading {task} episodes from {image_path}")
        yield from _iter_task_episodes(
            image_path=image_path,
            task=task,
            max_episodes=cycles,
            max_total_transition=max_total_transition,
        )


class RolloutRunner:
    """Placeholder evaluator for config compatibility."""

    def __init__(self, env_names, episode_num, embodiment="franka", save_video=False):
        self.env_names = env_names
        self.episode_num = episode_num
        self.embodiment = embodiment
        self.save_video = save_video

    def run(self, *args, **kwargs):
        raise NotImplementedError("RoboMimic evaluation rollouts are not implemented in this HPT adapter.")
