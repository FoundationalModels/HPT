# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from collections import OrderedDict

import numpy as np

from env.offline_source_adapter import (
    UnsupportedRolloutRunner,
    as_task_list,
    iter_source_episodes,
    maybe_download_file,
    missing_source_error,
    source_path,
)


DEFAULT_TASKS = ["cube_reach", "cube_push", "cube_lift"]
TASK_LANGUAGE = {
    "cube_reach": "move the fingers to the cube.",
    "cube_push": "push the cube to the target.",
    "cube_lift": "lift the cube.",
}
SOURCE_NOTE = (
    "No public HPT PyBullet TriFinger zarr cache was found in this checkout. "
    "Use online=True to generate trajectories with trifinger_simulation, or "
    "place a local TriFinger demo file at data/pybullet_trifinger/{task}/."
)

ROBOT_POSITION_LOW = np.deg2rad([-70, -70, -160] * 3)
ROBOT_POSITION_HIGH = np.deg2rad([70, 0, -2] * 3)
IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _sample_robot_position(rng):
    return rng.uniform(ROBOT_POSITION_LOW, ROBOT_POSITION_HIGH).astype(np.float64)


def _safe_goal(difficulty, rng):
    from trifinger_simulation.tasks import move_cube

    old_random = move_cube.random
    move_cube.random = rng
    try:
        return move_cube.sample_goal(difficulty=difficulty)
    finally:
        move_cube.random = old_random


def _tip_positions_around(position):
    from trifinger_simulation import sample

    return sample.get_tip_positions_around_position(3, position)


def _target_robot_position(platform, task_name, goal_pose, rng, current_position):
    if task_name == "cube_reach":
        target_position = goal_pose.position
    elif task_name == "cube_push":
        target_position = goal_pose.position
    elif task_name == "cube_lift":
        target_position = goal_pose.position + np.array([0.0, 0.0, 0.015])
    else:
        return _sample_robot_position(rng)

    try:
        target, _ = platform.simfinger.kinematics.inverse_kinematics(
            _tip_positions_around(target_position),
            np.asarray(current_position, dtype=np.float64),
            max_iterations=200,
        )
        return np.clip(target, ROBOT_POSITION_LOW, ROBOT_POSITION_HIGH)
    except Exception:
        return _sample_robot_position(rng)


def _platform_state(platform, time_index, goal_pose, goal_tip_positions):
    robot_obs = platform.get_robot_observation(time_index)
    camera_obs = platform.get_camera_observation(time_index)
    object_pose = camera_obs.filtered_object_pose
    tip_positions = np.concatenate(platform.forward_kinematics(robot_obs.position))

    return np.concatenate(
        [
            np.asarray(robot_obs.position, dtype=np.float32),
            np.asarray(robot_obs.velocity, dtype=np.float32),
            np.asarray(tip_positions, dtype=np.float32),
            np.asarray(object_pose.position, dtype=np.float32),
            np.asarray(object_pose.orientation, dtype=np.float32),
            np.asarray(goal_tip_positions, dtype=np.float32),
            np.asarray(goal_pose.orientation, dtype=np.float32),
        ]
    )


def _generate_online_episode(task_name, episode_index, horizon, seed, visualization):
    from trifinger_simulation import TriFingerPlatform

    rng = np.random.RandomState(seed + episode_index)
    if task_name == "cube_lift":
        goal_pose = _safe_goal(difficulty=2, rng=rng)
    elif task_name == "cube_push":
        goal_pose = _safe_goal(difficulty=1, rng=rng)
    else:
        goal_pose = _safe_goal(difficulty=-1, rng=rng)

    initial_robot_position = _sample_robot_position(rng)
    platform = TriFingerPlatform(
        visualization=visualization,
        initial_robot_position=initial_robot_position,
        initial_object_pose=_safe_goal(difficulty=-1, rng=rng),
        enable_cameras=False,
    )
    current_position = np.asarray(initial_robot_position, dtype=np.float64)
    target_position = _target_robot_position(platform, task_name, goal_pose, rng, current_position)
    goal_tip_positions = np.concatenate(_tip_positions_around(goal_pose.position))
    steps = []

    try:
        for step_idx in range(horizon):
            if step_idx % 25 == 0:
                target_position = _target_robot_position(platform, task_name, goal_pose, rng, current_position)
                target_position = 0.85 * target_position + 0.15 * _sample_robot_position(rng)

            current_position = 0.92 * current_position + 0.08 * target_position
            action = np.clip(current_position, ROBOT_POSITION_LOW, ROBOT_POSITION_HIGH)
            time_index = platform.append_desired_action(platform.Action(position=action))

            steps.append(
                OrderedDict(
                    observation={
                        "state": _platform_state(platform, time_index, goal_pose, goal_tip_positions),
                    },
                    action=action.astype(np.float32),
                    language_instruction=TASK_LANGUAGE.get(task_name, task_name),
                )
            )
    finally:
        platform.simfinger._disconnect_from_pybullet()

    return {"steps": steps}


def _generate_online_rollouts(
    env_names,
    episode_num_pertask,
    horizon,
    max_total_transition,
    seed,
    visualization,
):
    tasks = as_task_list(env_names, DEFAULT_TASKS)
    total_transitions = 0
    episode_index = 0

    for task_name in tasks:
        for _ in range(episode_num_pertask):
            episode = _generate_online_episode(
                task_name=task_name,
                episode_index=episode_index,
                horizon=horizon,
                seed=seed,
                visualization=visualization,
            )
            episode_index += 1
            total_transitions += len(episode["steps"])
            yield episode
            if max_total_transition is not None and total_transitions >= max_total_transition:
                return


def generate_dataset_rollouts(
    env_names,
    dataset_root="data/pybullet_trifinger",
    filename="demo_state.npz",
    source_url_template=None,
    download=False,
    online=True,
    horizon=100,
    seed=233,
    visualization=False,
    max_total_transition=500000,
    episode_num_pertask=100,
    **kwargs,
):
    del kwargs
    if online:
        yield from _generate_online_rollouts(
            env_names=env_names,
            episode_num_pertask=episode_num_pertask,
            horizon=horizon,
            max_total_transition=max_total_transition,
            seed=seed,
            visualization=visualization,
        )
        return

    for task_name in as_task_list(env_names, DEFAULT_TASKS):
        path = source_path(dataset_root, task_name, filename)
        source_url = source_url_template.format(task=task_name, filename=filename) if source_url_template else None
        maybe_download_file(path, source_url=source_url, download=download)
        if not path.exists():
            raise missing_source_error("pybullet_trifinger", task_name, path, SOURCE_NOTE)
        yield from iter_source_episodes(
            path=path,
            language=TASK_LANGUAGE.get(task_name, task_name),
            episode_num_pertask=episode_num_pertask,
            max_total_transition=max_total_transition,
        )


RolloutRunner = UnsupportedRolloutRunner
