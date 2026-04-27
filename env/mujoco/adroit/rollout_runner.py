# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from env.offline_source_adapter import (
    UnsupportedRolloutRunner,
    as_task_list,
    iter_source_episodes,
    maybe_download_file,
    missing_source_error,
    source_path,
)


DEFAULT_TASKS = ["pen", "hammer", "door", "relocate"]
ADROIT_EXPERT_URLS = {
    "door": "https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg_v1/door-expert-v1.hdf5",
    "hammer": "https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg_v1/hammer-expert-v1.hdf5",
    "pen": "https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg_v1/pen-expert-v1.hdf5",
    "relocate": "https://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg_v1/relocate-expert-v1.hdf5",
}
TASK_LANGUAGE = {
    "pen": "rotate the pen to the target orientation.",
    "hammer": "hammer the nail into the board.",
    "door": "open the door.",
    "relocate": "move the object to the target location.",
}
SOURCE_NOTE = (
    "Adroit expert demonstrations are downloadable from the Berkeley RAIL hand_dapg_v1 mirror. "
    "Use download=True, or place the corresponding {task}-expert-v1.hdf5 file under data/adroit/{task}/."
)


def generate_dataset_rollouts(
    env_names,
    dataset_root="data/adroit",
    filename=None,
    source_url_template=None,
    download=True,
    max_total_transition=500000,
    episode_num_pertask=100,
    **kwargs,
):
    del kwargs
    for task in as_task_list(env_names, DEFAULT_TASKS):
        task_filename = filename or f"{task}-expert-v1.hdf5"
        path = source_path(dataset_root, task, task_filename)
        if source_url_template:
            source_url = source_url_template.format(task=task, filename=task_filename)
        else:
            source_url = ADROIT_EXPERT_URLS.get(task)
        maybe_download_file(path, source_url=source_url, download=download)
        if not path.exists():
            raise missing_source_error("mujoco_adroit", task, path, SOURCE_NOTE)
        yield from iter_source_episodes(
            path=path,
            language=TASK_LANGUAGE.get(task, task),
            episode_num_pertask=episode_num_pertask,
            max_total_transition=max_total_transition,
        )


RolloutRunner = UnsupportedRolloutRunner
