#!/usr/bin/env python3
"""
Count valid training samples per domain from zarr caches.

Loads domains from experiments/configs/config.yaml and per-domain pad_before /
pad_after overrides from experiments/configs/env/<domain>.yaml.

A valid sample is a sliding window of length `horizon` that fits inside an
episode (with optional padding), matching SequenceSampler / create_indices in
hpt/utils/sampler.py:

    horizon        = observation_horizon + action_horizon - 1
    samples_per_ep = max(0, ep_len - horizon + pad_before + pad_after + 1)

Run from anywhere:
    python hpt/dataset/count_valid_samples.py [--data-dir data]
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import yaml
import zarr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_YAML = PROJECT_ROOT / "experiments" / "configs" / "config.yaml"
ENV_CONFIG_DIR = PROJECT_ROOT / "experiments" / "configs" / "env"


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_nested(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, {})
    return d if d != {} else default


def load_episode_ends(zarr_path: str) -> np.ndarray:
    root = zarr.open(zarr_path, mode="r")
    return np.array(root["meta"]["episode_ends"])


def episode_lengths(episode_ends: np.ndarray) -> np.ndarray:
    ends = np.array(episode_ends)
    starts = np.concatenate([[0], ends[:-1]])
    return ends - starts


def count_valid_samples(ep_lengths: np.ndarray, horizon: int, pad_before: int, pad_after: int) -> int:
    per_ep = np.maximum(0, ep_lengths - horizon + pad_before + pad_after + 1)
    return int(per_ep.sum())


def find_zarr_dirs(data_dir: str, domain: str) -> list:
    return sorted(glob.glob(os.path.join(data_dir, f"zarr_{domain}_*")))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"),
                        help="directory containing zarr_* caches")
    args = parser.parse_args()

    cfg = load_yaml(CONFIG_YAML)
    domains = [d.strip() for d in cfg.get("domains", "").split(",") if d.strip()]

    dataset_cfg = cfg.get("dataset", {})
    default_obs        = dataset_cfg.get("observation_horizon", 4)
    default_act        = dataset_cfg.get("action_horizon", 8)
    default_pad_before = dataset_cfg.get("pad_before", 0)
    default_pad_after  = dataset_cfg.get("pad_after", 0)
    val_ratio          = dataset_cfg.get("val_ratio", 0.1)

    print(f"Config                        : {CONFIG_YAML.relative_to(PROJECT_ROOT)}")
    print(f"observation_horizon (default) : {default_obs}")
    print(f"action_horizon      (default) : {default_act}")
    print(f"pad_before          (default) : {default_pad_before}")
    print(f"pad_after           (default) : {default_pad_after}")
    print(f"val_ratio                     : {val_ratio}")
    print(f"data_dir                      : {args.data_dir}")
    print()

    col_w = max(len(d) for d in domains) + 2
    header = (
        f"{'domain':<{col_w}}  {'zarr cache':<45}  "
        f"{'pb':>4}  {'pa':>4}  {'episodes':>9}  "
        f"{'total steps':>11}  {'train samples':>13}  {'val samples':>11}"
    )
    print(header)
    print("-" * len(header))

    grand_train = 0
    grand_val = 0

    for domain in domains:
        env_yaml = ENV_CONFIG_DIR / f"{domain}.yaml"
        pad_before = default_pad_before
        pad_after  = default_pad_after
        if env_yaml.exists():
            env_cfg   = load_yaml(env_yaml)
            pad_before = get_nested(env_cfg, "dataset", "pad_before", default=pad_before)
            pad_after  = get_nested(env_cfg, "dataset", "pad_after",  default=pad_after)

        horizon = default_obs + default_act - 1

        zarr_dirs = find_zarr_dirs(args.data_dir, domain)
        if not zarr_dirs:
            print(f"{domain:<{col_w}}  {'(no zarr cache found)':.<45}")
            continue

        for zarr_dir in zarr_dirs:
            cache_name = os.path.basename(zarr_dir)
            try:
                ep_ends = load_episode_ends(zarr_dir)
            except Exception as e:
                print(f"{domain:<{col_w}}  {cache_name:<45}  ERROR: {e}")
                continue

            ep_lens    = episode_lengths(ep_ends)
            n_episodes = len(ep_lens)
            total_steps = int(ep_lens.sum())

            if val_ratio > 0 and n_episodes > 1:
                n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
            else:
                n_val = 0
            n_train = n_episodes - n_val

            train_samples = count_valid_samples(ep_lens[:n_train], horizon, pad_before, pad_after)
            val_samples   = count_valid_samples(ep_lens[n_train:], horizon, pad_before, pad_after)

            grand_train += train_samples
            grand_val   += val_samples

            print(
                f"{domain:<{col_w}}  {cache_name:<45}  "
                f"{pad_before:>4}  {pad_after:>4}  {n_episodes:>9,}  "
                f"{total_steps:>11,}  {train_samples:>13,}  {val_samples:>11,}"
            )

    print("-" * len(header))
    grand_total = grand_train + grand_val
    print(
        f"{'TOTAL':<{col_w}}  {'':45}  {'':>4}  {'':>4}  {'':>9}  "
        f"{'':>11}  {grand_train:>13,}  {grand_val:>11,}"
    )
    print(f"\nGrand total valid samples (train + val): {grand_total:,}")


if __name__ == "__main__":
    main()
