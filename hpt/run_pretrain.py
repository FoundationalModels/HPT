# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import math
import os
from collections import OrderedDict
from typing import List

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.utils import data
from torch.utils.data import ConcatDataset
from tqdm import trange

from hpt import train_test
from hpt.utils import utils
from hpt.utils.warmup_lr_wrapper import WarmupLR

MAX_EPOCHS = 100000
TEST_FREQ = 3


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class WeightedDomainBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Sample batches by first choosing a source domain, then sampling data from that domain."""

    def __init__(
        self,
        domain_lengths: List[int],
        batch_size: int,
        num_batches: int,
        seed: int,
        domain_probs: List[float] = None,
    ):
        super().__init__()
        if len(domain_lengths) == 0:
            raise ValueError("domain_lengths cannot be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if num_batches <= 0:
            raise ValueError("num_batches must be > 0")
        if any(length <= 0 for length in domain_lengths):
            raise ValueError(f"all domain lengths must be > 0, got {domain_lengths}")

        self.domain_lengths = list(domain_lengths)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.epoch = 0

        offsets = [0]
        for length in self.domain_lengths[:-1]:
            offsets.append(offsets[-1] + length)
        self.domain_offsets = offsets

        if domain_probs is None:
            weights = np.array([1.0 / math.sqrt(length) for length in self.domain_lengths], dtype=np.float64)
            domain_probs = weights / weights.sum()
        else:
            domain_probs = np.array(domain_probs, dtype=np.float64)
            if domain_probs.shape[0] != len(self.domain_lengths):
                raise ValueError(
                    f"domain_probs length ({domain_probs.shape[0]}) must equal num domains ({len(self.domain_lengths)})"
                )
            if np.any(domain_probs < 0):
                raise ValueError("domain_probs cannot contain negative values")
            prob_sum = domain_probs.sum()
            if prob_sum <= 0:
                raise ValueError("domain_probs must sum to a positive value")
            domain_probs = domain_probs / prob_sum

        self.domain_probs = domain_probs

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.num_batches):
            domain_idx = int(rng.choice(len(self.domain_lengths), p=self.domain_probs))
            local_indices = rng.integers(0, self.domain_lengths[domain_idx], size=self.batch_size, endpoint=False)
            offset = self.domain_offsets[domain_idx]
            yield (local_indices + offset).tolist()


def _get_dataset_rollout_fn(cfg, domain: str):
    if "dataset_generators" in cfg and domain in cfg.dataset_generators:
        generator_cfg = cfg.dataset_generators[domain]
        if generator_cfg is not None and "_target_" in generator_cfg:
            return hydra.utils.instantiate(generator_cfg)

    if hasattr(cfg, "dataset_generator_func") and cfg.dataset_generator_func is not None:
        return hydra.utils.instantiate(cfg.dataset_generator_func)

    return None


def _init_policy(cfg, domain_datasets, device):
    pretrained_exists = len(cfg.train.pretrained_dir) > len("output/") and os.path.exists(
        os.path.join(cfg.train.pretrained_dir, "trunk.pth")
    )

    if pretrained_exists:
        print("load pretrained trunk config")
        pretrained_cfg = OmegaConf.load(cfg.train.pretrained_dir + "/config.yaml")
        pretrained_cfg = OmegaConf.structured(pretrained_cfg)
        pretrained_cfg.network["_target_"] = "hpt.models.policy.Policy"
        policy = hydra.utils.instantiate(pretrained_cfg.network).to(device)
        print("load trunk from local disk")
    elif "hf" in cfg.train.pretrained_dir:
        from hpt.models.policy import Policy

        policy = Policy.from_pretrained(cfg.train.pretrained_dir)
        print("load trunk from cloud")
    else:
        policy = hydra.utils.instantiate(cfg.network).to(device)
        print("pretrain from scratch")

    # Initialize one stem/head pair per source domain.
    for domain, dataset in domain_datasets.items():
        utils.update_network_dim(cfg, dataset, policy)
        policy.init_domain_stem(domain, cfg.stem)
        policy.init_domain_head(domain, dataset.get_normalizer(), cfg.head)

    policy.finalize_modules()

    if pretrained_exists:
        policy.load_trunk(os.path.join(cfg.train.pretrained_dir, "trunk.pth"))

    if cfg.train.freeze_trunk:
        policy.freeze_trunk()
        print("trunk frozen")

    policy.print_model_stats()
    policy.to(device)
    return policy


@hydra.main(config_path="../experiments/configs", config_name="config", version_base="1.2")
def run(cfg):
    date = cfg.output_dir.split("/")[1]
    run = wandb.init(
        project=cfg.pretrain.wandb_project,
        tags=[cfg.wb_tag, "pretrain"],
        name=f"{date}_{cfg.script_name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=False,
        save_code=False,
        resume="allow",
    )
    utils.set_seed(cfg.seed)

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    domain_list = [d.strip() for d in cfg.domains.split(",") if len(d.strip()) > 0]
    if len(domain_list) == 0:
        raise ValueError("No domains provided. Set cfg.domains to a comma-separated list of sources.")

    print(f"pretraining domains: {domain_list}")

    domain_train_datasets = OrderedDict()
    domain_val_datasets = OrderedDict()

    for domain in domain_list:
        rollout_fn = _get_dataset_rollout_fn(cfg, domain)
        dataset = hydra.utils.instantiate(
            cfg.dataset,
            dataset_name=domain,
            env_rollout_fn=rollout_fn,
            max_train_episodes=cfg.pretrain.max_train_trajectories_per_source,
            max_val_episodes=cfg.pretrain.max_val_trajectories_per_source,
            **cfg.dataset,
        )
        domain_train_datasets[domain] = dataset
        domain_val_datasets[domain] = dataset.get_validation_dataset()

    concat_train_dataset = ConcatDataset(list(domain_train_datasets.values()))
    domain_sizes = [len(dataset) for dataset in domain_train_datasets.values()]

    if cfg.pretrain.sampling_weights:
        domain_probs = list(cfg.pretrain.sampling_weights)
    else:
        domain_probs = None

    batch_sampler = WeightedDomainBatchSampler(
        domain_lengths=domain_sizes,
        batch_size=cfg.dataloader.batch_size,
        num_batches=cfg.train.epoch_iters,
        seed=cfg.seed,
        domain_probs=domain_probs,
    )

    dataloader_kwargs = OmegaConf.to_container(cfg.dataloader, resolve=True)
    dataloader_kwargs.pop("batch_size", None)
    dataloader_kwargs.pop("shuffle", None)
    dataloader_kwargs.pop("drop_last", None)
    train_loader = data.DataLoader(concat_train_dataset, batch_sampler=batch_sampler, **dataloader_kwargs)

    val_loader_kwargs = OmegaConf.to_container(cfg.val_dataloader, resolve=True)
    val_loaders = {
        domain: data.DataLoader(dataset, **val_loader_kwargs) for domain, dataset in domain_val_datasets.items()
    }

    policy = _init_policy(cfg, domain_train_datasets, device)

    opt = utils.get_optimizer(cfg.optimizer, policy, cfg.optimizer_misc)
    cfg.lr_scheduler.T_max = int(cfg.train.total_iters)
    sch = utils.get_scheduler(cfg.lr_scheduler, optimizer=opt)
    sch = WarmupLR(sch, init_lr=0, num_warmup=cfg.warmup_lr.step, warmup_strategy="linear")

    utils.save_args_hydra(cfg.output_dir, cfg)

    total_traj = int(sum(dataset.train_mask.sum() for dataset in domain_train_datasets.values()))
    cfg.total_num_traj = total_traj
    model_path = os.path.join(cfg.output_dir, "model.pth")
    trunk_path = os.path.join(cfg.output_dir, "trunk.pth")
    checkpoint_name = cfg.train.get("checkpoint_name", "training_state.pth")
    checkpoint_every = int(cfg.train.get("checkpoint_every", 1))
    resume_from = cfg.train.get("resume_from", "")
    auto_resume = bool(cfg.train.get("auto_resume", True))
    training_state_path = os.path.join(cfg.output_dir, checkpoint_name)

    start_epoch = 0
    global_step = 0
    resume_path = utils.find_resume_checkpoint(
        output_dir=cfg.output_dir,
        checkpoint_name=checkpoint_name,
        resume_from=resume_from,
        pretrained_dir=cfg.train.pretrained_dir,
        auto_resume=auto_resume,
    )
    if len(resume_path) > 0:
        state_info = utils.load_training_state(
            checkpoint_path=resume_path,
            model=policy,
            optimizer=opt,
            scheduler=sch,
            map_location=device,
            strict=True,
        )
        if state_info["restored"]:
            start_epoch = int(state_info["epoch"])
            global_step = int(state_info["global_step"])
            print(f"resumed training state from {resume_path} at epoch={start_epoch} step={global_step}")

    print(f"Epoch size: {len(train_loader)} Traj(train total): {total_traj}")
    print(f"Domain batch probs: {batch_sampler.domain_probs.tolist()}")

    # train / test loop
    pbar = trange(start_epoch, MAX_EPOCHS, position=0)
    for epoch in pbar:
        batch_sampler.set_epoch(epoch)
        train_stats = train_test.train(cfg.log_interval, policy, device, train_loader, opt, sch, epoch)
        global_step += len(train_loader)
        train_steps = global_step

        if epoch % TEST_FREQ == 0:
            all_test_losses = {}
            for domain, domain_val_loader in val_loaders.items():
                domain_test_loss = train_test.test(policy, device, domain_val_loader, epoch)
                wandb.log({"validate/epoch": epoch, f"validate/{domain}_test_loss": domain_test_loss})
                all_test_losses[domain] = domain_test_loss

            if len(all_test_losses) > 0:
                mean_test_loss = float(np.mean(list(all_test_losses.values())))
                wandb.log({"validate/epoch": epoch, "validate/mean_test_loss": mean_test_loss})

        if "loss" in train_stats:
            print(f"Steps: {train_steps}. Train loss: {train_stats['loss']:.4f}")

        should_save_checkpoint = checkpoint_every > 0 and (
            (epoch + 1) % checkpoint_every == 0 or train_steps >= cfg.train.total_iters
        )
        if should_save_checkpoint:
            policy.save(model_path)
            torch.save(policy.trunk.state_dict(), trunk_path)
            utils.save_training_state(
                checkpoint_path=training_state_path,
                model=policy,
                optimizer=opt,
                scheduler=sch,
                epoch=epoch + 1,
                global_step=train_steps,
            )

        if train_steps >= cfg.train.total_iters:
            break

    print("model saved to:", model_path)
    print("trunk saved to:", trunk_path)

    utils.save_args_hydra(cfg.output_dir, cfg)
    pbar.close()
    run.finish()
    wandb.finish()


if __name__ == "__main__":
    run()
