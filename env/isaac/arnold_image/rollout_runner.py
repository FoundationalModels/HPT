# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import io
import zipfile
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from env.offline_source_adapter import UnsupportedRolloutRunner, as_task_list, missing_source_error


DEFAULT_TASKS = [
    "close_cabinet",
    "close_drawer",
    "open_cabinet",
    "open_drawer",
    "pickup_object",
    "pour_water",
    "reorient_object",
    "transfer_water",
]
TASK_LANGUAGE = {
    "close_cabinet": "close the cabinet.",
    "close_drawer": "close the drawer.",
    "open_cabinet": "open the cabinet.",
    "open_drawer": "open the drawer.",
    "pickup_object": "pick up the object.",
    "pour_water": "pour water.",
    "reorient_object": "reorient the object.",
    "transfer_water": "transfer water.",
}
SOURCE_NOTE = (
    "Download Arnold with gdown into data/arnold/ so the task zips exist at "
    "data/arnold/tasks/{task}.zip. Arnold NPZ files contain USD/Pixar "
    "objects, so the Python environment also needs usd-core installed."
)
RESOLUTION = (224, 224)


def generate_dataset_rollouts(
    env_names,
    dataset_root="data/arnold",
    split="train",
    max_total_transition=500000,
    episode_num_pertask=100,
    **kwargs,
):
    del kwargs
    for task in _arnold_task_list(env_names):
        path = _task_zip_path(dataset_root, task)
        if not path.exists():
            raise missing_source_error("isaac_arnold_image", task, path, SOURCE_NOTE)
        yield from iter_arnold_zip_episodes(
            path,
            task=task,
            split=split,
            episode_num_pertask=episode_num_pertask,
            max_total_transition=max_total_transition,
        )


def _arnold_task_list(env_names):
    tasks = []
    for name in as_task_list(env_names, DEFAULT_TASKS):
        if name == "arnold_image_default":
            tasks.extend(DEFAULT_TASKS)
        else:
            tasks.append(name)
    return tasks


def _task_zip_path(dataset_root, task):
    root = Path(dataset_root)
    candidates = [
        root / "tasks" / f"{task}.zip",
        root / f"{task}.zip",
        Path("data") / "arnold" / "tasks" / f"{task}.zip",
        Path("data") / "arnold" / "arnold-data" / "tasks" / f"{task}.zip",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def iter_arnold_zip_episodes(path, task, split, episode_num_pertask, max_total_transition):
    total_transitions = 0
    episode_count = 0
    with zipfile.ZipFile(path) as zf:
        episode_names = sorted(
            name for name in zf.namelist() if name.endswith(".npz") and f"/{split}/" in name
        )
        if not episode_names:
            episode_names = sorted(name for name in zf.namelist() if name.endswith(".npz"))

        for episode_name in episode_names:
            demo = _load_arnold_npz(zf, episode_name)
            steps = arnold_demo_to_steps(demo["gt"], default_language=TASK_LANGUAGE.get(task, task))
            if not steps:
                continue

            episode_count += 1
            total_transitions += len(steps)
            yield {"steps": steps}

            if episode_count >= episode_num_pertask:
                break
            if max_total_transition is not None and total_transitions >= max_total_transition:
                break


def _load_arnold_npz(zip_file, episode_name):
    try:
        with np.load(io.BytesIO(zip_file.read(episode_name)), allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    except ModuleNotFoundError as exc:
        if exc.name == "pxr":
            raise ModuleNotFoundError(
                "Arnold demos require usd-core because their NPZ files pickle pxr objects. "
                "Install it with `pip install usd-core`."
            ) from exc
        raise


def arnold_demo_to_steps(gt, default_language):
    steps = []
    for entry in gt:
        image = select_arnold_rgb(entry)
        state = select_arnold_state(entry)
        action = select_arnold_action(entry)
        language = entry.get("instruction") or default_language
        steps.append(
            OrderedDict(
                observation={
                    "state": state,
                    "image_0": image,
                },
                action=action,
                language_instruction=language,
            )
        )
    return steps


def select_arnold_rgb(entry):
    images = entry.get("images") or []
    if not images:
        raise KeyError("Arnold demo entry is missing RGB images.")
    rgb = np.asarray(images[0]["rgb"])[..., :3]
    if rgb.shape[:2] != RESOLUTION:
        rgb = cv2.resize(rgb, RESOLUTION, interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


def select_arnold_state(entry):
    parts = []
    for key in ("joint_positions", "joint_velocities", "gripper_joint_positions"):
        if key in entry:
            parts.append(np.asarray(entry[key], dtype=np.float32).reshape(-1))

    for key in ("position_rotation_world", "robot_base"):
        if key in entry:
            for value in entry[key]:
                parts.append(np.asarray(value, dtype=np.float32).reshape(-1))

    if not parts:
        raise KeyError("Arnold demo entry is missing state-like fields.")
    return np.concatenate(parts).astype(np.float32)


def select_arnold_action(entry):
    actions = entry.get("template_actions")
    if actions:
        return np.concatenate([np.asarray(value, dtype=np.float32).reshape(-1) for value in actions]).astype(np.float32)

    target = entry.get("target")
    if target is not None:
        return np.asarray(target, dtype=np.float32).reshape(-1)
    raise KeyError("Arnold demo entry is missing template_actions/target.")


RolloutRunner = UnsupportedRolloutRunner
