# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import urllib.request
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np


def as_task_list(env_names, default_tasks):
    if env_names is None or env_names == "all":
        return list(default_tasks)
    if isinstance(env_names, str):
        return [task.strip() for task in env_names.split(",") if task.strip()]
    return list(env_names)


def source_path(dataset_root, task, filename):
    return Path(dataset_root) / task / filename


def maybe_download_file(path, source_url=None, download=False):
    if path.exists():
        return
    if not download or source_url is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[offline-source] downloading {source_url} -> {path}")
    urllib.request.urlretrieve(source_url, path)


def missing_source_error(domain, task, path, source_note):
    return FileNotFoundError(
        f"Missing source dataset for {domain}/{task}: {path}\n"
        f"{source_note}\n"
        "Place a converted source file at that path, or pass source_url_template with "
        "'{task}' and '{filename}' placeholders plus download=True."
    )


def iter_source_episodes(
    path,
    language,
    episode_num_pertask,
    max_total_transition,
):
    suffix = path.suffix.lower()
    if suffix in (".hdf5", ".h5"):
        yield from iter_hdf5_episodes(path, language, episode_num_pertask, max_total_transition)
        return
    if suffix == ".npz":
        yield from iter_npz_episodes(path, language, episode_num_pertask, max_total_transition)
        return
    raise ValueError(f"Unsupported source file type for {path}. Expected .hdf5, .h5, or .npz.")


def iter_hdf5_episodes(path, language, episode_num_pertask, max_total_transition):
    total_transitions = 0
    with h5py.File(path, "r") as f:
        if "data" not in f:
            yield from iter_flat_hdf5_episodes(path, f, language, episode_num_pertask, max_total_transition)
            return
        demo_names = sorted(
            f["data"].keys(),
            key=lambda name: int(name.split("_")[-1]) if name.startswith("demo_") else name,
        )

        for demo_name in demo_names[:episode_num_pertask]:
            demo = f["data"][demo_name]
            if "actions" not in demo:
                raise KeyError(f"{path}:{demo_name} does not contain 'actions'.")

            actions = np.asarray(demo["actions"])
            states = select_hdf5_states(demo)
            images = select_hdf5_images(demo)
            steps = []
            for step_idx, action in enumerate(actions):
                observation = {"state": states[step_idx]}
                for key, values in images.items():
                    observation[key] = values[step_idx].astype(np.uint8)
                steps.append(
                    OrderedDict(
                        observation=observation,
                        action=action.astype(np.float32),
                        language_instruction=language,
                    )
                )

            if len(steps) == 0:
                continue
            total_transitions += len(steps)
            yield {"steps": steps}
            if max_total_transition is not None and total_transitions >= max_total_transition:
                break


def iter_flat_hdf5_episodes(path, h5_file, language, episode_num_pertask, max_total_transition):
    """Read D4RL-style flat HDF5 datasets such as Adroit expert demos."""
    action_key = "actions" if "actions" in h5_file else "action"
    state_key = "observations" if "observations" in h5_file else "states"
    if action_key not in h5_file or state_key not in h5_file:
        raise KeyError(
            f"{path} must contain either data/demo_* groups or flat '{action_key}' and '{state_key}' datasets."
        )

    actions = np.asarray(h5_file[action_key]).astype(np.float32)
    states = np.asarray(h5_file[state_key]).reshape(actions.shape[0], -1).astype(np.float32)
    dones = np.zeros(actions.shape[0], dtype=bool)
    for key in ("terminals", "timeouts", "dones", "done"):
        if key in h5_file:
            dones |= np.asarray(h5_file[key]).astype(bool)

    if not dones.any():
        dones[-1] = True

    start = 0
    total_transitions = 0
    episode_count = 0
    for end in np.where(dones)[0]:
        steps = []
        for idx in range(start, end + 1):
            steps.append(
                OrderedDict(
                    observation={"state": states[idx]},
                    action=actions[idx],
                    language_instruction=language,
                )
            )
        start = end + 1
        if len(steps) == 0:
            continue

        total_transitions += len(steps)
        episode_count += 1
        yield {"steps": steps}
        if episode_count >= episode_num_pertask:
            break
        if max_total_transition is not None and total_transitions >= max_total_transition:
            break


def select_hdf5_states(demo):
    if "obs" in demo:
        obs = demo["obs"]
        state_keys = [
            key
            for key in sorted(obs.keys())
            if "image" not in key and np.issubdtype(obs[key].dtype, np.number) and len(obs[key].shape) <= 2
        ]
        if state_keys:
            return np.concatenate([np.asarray(obs[key]).reshape(obs[key].shape[0], -1) for key in state_keys], axis=-1)
    if "states" in demo:
        states = np.asarray(demo["states"])
        return states.reshape(states.shape[0], -1)
    raise KeyError("Demo contains neither low-dimensional obs entries nor a 'states' dataset.")


def select_hdf5_images(demo):
    if "obs" not in demo:
        return {}
    images = {}
    for key in sorted(demo["obs"].keys()):
        if "image" not in key:
            continue
        value = np.asarray(demo["obs"][key])
        if value.ndim == 4 and value.shape[1] in (1, 3):
            value = np.moveaxis(value, 1, -1)
        images[key] = value
    return images


def iter_npz_episodes(path, language, episode_num_pertask, max_total_transition):
    data = dict(np.load(path, allow_pickle=True))
    required = ["action", "done"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")

    action = data["action"].astype(np.float32)
    done_indexes = np.where(data["done"])[0]
    if len(done_indexes) == 0:
        done_indexes = np.array([len(action)])

    image_keys = [key for key in sorted(data) if "image" in key]
    state = select_npz_state(data)
    start = 0
    total_transitions = 0
    for end in done_indexes[:episode_num_pertask]:
        steps = []
        for idx in range(start, min(end, len(action))):
            observation = {"state": state[idx]}
            for key in image_keys:
                observation[key] = data[key][idx][..., :3].astype(np.uint8)
            steps.append(
                OrderedDict(
                    observation=observation,
                    action=action[idx],
                    language_instruction=language,
                )
            )
        start = end
        if len(steps) == 0:
            continue
        total_transitions += len(steps)
        yield {"steps": steps}
        if max_total_transition is not None and total_transitions >= max_total_transition:
            break


def select_npz_state(data):
    for key in ("state", "ee_pose", "proprio", "qpos"):
        if key in data:
            value = data[key].astype(np.float32)
            return value.reshape(value.shape[0], -1)
    raise KeyError("NPZ source needs one of: state, ee_pose, proprio, qpos.")


class UnsupportedRolloutRunner:
    def __init__(self, env_names, episode_num, save_video=False, **kwargs):
        self.env_names = env_names
        self.episode_num = episode_num
        self.save_video = save_video
        self.kwargs = kwargs

    def run(self, *args, **kwargs):
        raise NotImplementedError("Evaluation rollouts are not implemented for this offline source adapter.")
