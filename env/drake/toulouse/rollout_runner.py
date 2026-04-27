# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import subprocess
import sys
from pathlib import Path

from env.offline_source_adapter import (
    UnsupportedRolloutRunner,
    as_task_list,
    iter_source_episodes,
    maybe_download_file,
    missing_source_error,
    source_path,
)


DEFAULT_TASKS = ["hammer", "spatula", "knife", "wrench"]
TASK_TO_ENV = {
    "hammer": "FrankaDrakeHammerEnv",
    "spatula": "FrankaDrakeSpatulaEnv",
    "knife": "FrankaDrakeKnifeEnv",
    "wrench": "FrankaDrakeWrenchEnv",
}
TASK_TO_CONFIG = {
    "hammer": "FrankaDrakeHammerEnvMergingWeights",
    "spatula": "FrankaDrakeSpatulaEnvMergingWeights",
    "knife": "FrankaDrakeKnifeEnvMergingWeights",
    "wrench": "FrankaDrakeWrenchEnvMergingWeights",
}
TASK_LANGUAGE = {
    "hammer": "use the hammer to hit the object.",
    "spatula": "use the spatula to move the object.",
    "knife": "use the knife to cut the object.",
    "wrench": "use the wrench to turn the object.",
}
SOURCE_NOTE = (
    "Fleet-Tools was added as external/Fleet-Tools, but it ships simulator code rather than "
    "prebuilt demonstrations. Generate demos with generate=True after installing Fleet-Tools "
    "dependencies, or place a collapsed Fleet-Tools demo_state.npz at the expected path."
)


def _fleet_task_list(env_names):
    tasks = []
    for name in as_task_list(env_names, DEFAULT_TASKS):
        if name == "fleet_tools_default":
            tasks.extend(DEFAULT_TASKS)
        else:
            tasks.append(name)
    return tasks


def _candidate_paths(fleet_root, dataset_root, task, tool_idx, filename):
    env_name = TASK_TO_ENV[task]
    collapse_env_name = f"{task}for{task}"
    return [
        Path("data") / "demo_drake" / f"FrankaDrake{collapse_env_name.capitalize()}Env-Tool{tool_idx}" / filename,
        Path("data") / "fleet_tools" / "demonstrations" / "processed" / f"{task}for{task}tool_{tool_idx}" / filename,
        Path(fleet_root) / "data" / "demo_drake" / f"{env_name}-Tool{tool_idx}" / filename,
        Path(fleet_root) / "data" / "demo_drake" / f"FrankaDrake{collapse_env_name.capitalize()}Env-Tool{tool_idx}" / filename,
        Path(fleet_root) / "assets" / "demonstrations" / "processed" / f"{task}for{task}tool_{tool_idx}" / filename,
        source_path(dataset_root, task, filename),
    ]


def _find_existing_path(fleet_root, dataset_root, task, tool_idx, filename):
    for path in _candidate_paths(fleet_root, dataset_root, task, tool_idx, filename):
        if path.exists():
            return path
    return None


def _run_fleet_generation(
    fleet_root,
    task,
    tool_idx,
    episode_num_pertask,
    use_image,
    render,
    fleet_data_root,
    processed_demo_root,
):
    env_name = TASK_TO_ENV[task]
    task_config = TASK_TO_CONFIG[task]
    raw_demo_root = (Path(fleet_data_root) / "demonstrations").resolve()
    processed_root = (Path(fleet_data_root) / "demonstrations" / "processed").resolve()
    processed_demo_root = Path(processed_demo_root).resolve()
    cmd = [
        sys.executable,
        "-m",
        "core.run",
        "cuda=False",
        f"render={str(render)}",
        f"env_name={env_name}",
        "num_envs=1",
        "run_expert=True",
        "save_demonstrations=True",
        f"demonstration_dir={raw_demo_root}",
        "start_episode_position=0",
        "num_workers=0",
        f"task={task_config}",
        "train=FrankaDrakeEnv",
        f"save_demo_suffix=tool_{tool_idx}",
        f"task.tool_fix_idx={tool_idx}",
        f"max_episodes={episode_num_pertask}",
        f"num_episode={episode_num_pertask}",
        "record_video=False",
        "training=False",
        "+task.data_collection=True",
        f"task.env.use_image={str(use_image)}",
    ]
    print("[fleet-tools] generating demos:", " ".join(cmd))
    subprocess.run(cmd, cwd=fleet_root, check=True)

    collapse_cmd = [
        sys.executable,
        "-m",
        "scripts.collapse_dataset",
        "-e",
        f"{task}for{task}",
        "--tool",
        str(tool_idx),
        "--max_num",
        str(episode_num_pertask * 200),
        "--saved_path",
        str(raw_demo_root),
        "--output_path",
        str(processed_root),
        "--processed_output_path",
        str(processed_demo_root),
    ]
    print("[fleet-tools] collapsing demos:", " ".join(collapse_cmd))
    subprocess.run(collapse_cmd, cwd=fleet_root, check=True)


def generate_dataset_rollouts(
    env_names,
    fleet_root="external/Fleet-Tools",
    dataset_root="data/drake_toulouse",
    filename="demo_state.npz",
    source_url_template=None,
    download=False,
    generate=False,
    tool_idx=0,
    fleet_data_root="data/fleet_tools",
    processed_demo_root="data/demo_drake",
    use_image=False,
    render=False,
    max_total_transition=500000,
    episode_num_pertask=100,
    **kwargs,
):
    del kwargs
    for task in _fleet_task_list(env_names):
        if task not in TASK_TO_ENV:
            raise ValueError(f"Unsupported Fleet-Tools task '{task}'. Supported tasks: {sorted(TASK_TO_ENV)}")

        path = _find_existing_path(fleet_root, dataset_root, task, tool_idx, filename)
        if path is None and generate:
            _run_fleet_generation(
                fleet_root=fleet_root,
                task=task,
                tool_idx=tool_idx,
                episode_num_pertask=episode_num_pertask,
                use_image=use_image,
                render=render,
                fleet_data_root=fleet_data_root,
                processed_demo_root=processed_demo_root,
            )
            path = _find_existing_path(fleet_root, dataset_root, task, tool_idx, filename)

        fallback_path = source_path(dataset_root, task, filename)
        if path is None and source_url_template:
            source_url = source_url_template.format(task=task, filename=filename)
            maybe_download_file(fallback_path, source_url=source_url, download=download)
            if fallback_path.exists():
                path = fallback_path

        if path is None:
            expected = _candidate_paths(fleet_root, dataset_root, task, tool_idx, filename)[0]
            raise missing_source_error("drake_toulouse", task, expected, SOURCE_NOTE)

        print(f"[fleet-tools] reading {task} demos from {path}")
        yield from iter_source_episodes(
            path=path,
            language=TASK_LANGUAGE.get(task, task),
            episode_num_pertask=episode_num_pertask,
            max_total_transition=max_total_transition,
        )


RolloutRunner = UnsupportedRolloutRunner
