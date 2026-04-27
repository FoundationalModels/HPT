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


DEFAULT_TASKS = ["grasping_image_default"]
SOURCE_NOTE = (
    "No public HPT PyBullet grasping-image zarr cache was found in this checkout. "
    "This adapter expects a local converted source file such as "
    "data/pybullet_grasping_image/{task}/demo_state.npz or demo.hdf5."
)


def generate_dataset_rollouts(
    env_names,
    dataset_root="data/pybullet_grasping_image",
    filename="demo_state.npz",
    source_url_template=None,
    download=False,
    max_total_transition=500000,
    episode_num_pertask=100,
    **kwargs,
):
    del kwargs
    for task in as_task_list(env_names, DEFAULT_TASKS):
        path = source_path(dataset_root, task, filename)
        source_url = source_url_template.format(task=task, filename=filename) if source_url_template else None
        maybe_download_file(path, source_url=source_url, download=download)
        if not path.exists():
            raise missing_source_error("pybullet_grasping_image", task, path, SOURCE_NOTE)
        yield from iter_source_episodes(
            path=path,
            language="grasp the object.",
            episode_num_pertask=episode_num_pertask,
            max_total_transition=max_total_transition,
        )


RolloutRunner = UnsupportedRolloutRunner
