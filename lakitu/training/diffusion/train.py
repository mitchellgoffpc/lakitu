#!/usr/bin/env python
import time
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Union, Iterator

import torch
from torch.optim import Optimizer

from lakitu.datasets.dataset import EpisodeDataset
from lakitu.training.diffusion.eval import EvalConfig, eval_policy, set_seed
from lakitu.training.diffusion.policy import DiffusionConfig, DiffusionPolicy
from lakitu.training.helpers.checkpoint import save_checkpoint
from lakitu.training.helpers.config import BaseConfig
from lakitu.training.helpers.metrics import AverageMeter, MetricsTracker, format_big_number
from lakitu.training.helpers.wandb import WandBLogger, WandBConfig

OUTPUT_DIR = Path(__file__).parents[1] / "experiments"
LRScheduler = Any

@dataclass
class DatasetConfig:
    data_dir: Path = Path(__file__).parent.parent / 'data' / 'episodes'
    episodes: list[int] | None = None

@dataclass
class AdamConfig:
    lr: float = 1e-4
    betas: tuple[float, float] = (0.95, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-6
    grad_clip_norm: float = 10.0
    fused: bool = True

@dataclass
class LRSchedulerConfig:
    name: str = "cosine"
    num_warmup_steps: int = 500

@dataclass
class TrainConfig(BaseConfig):
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    policy: DiffusionConfig = field(default_factory=DiffusionConfig)
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    resume: bool = False
    # `seed` is used for training (eg: model initialization, dataset shuffling) AND for the evaluation environments.
    seed: int | None = 1000
    num_workers: int = 4
    batch_size: int = 64
    steps: int = 100_000
    eval_freq: int = 25_000
    log_freq: int = 200
    save_checkpoint: bool = True
    # Checkpoint is saved every `save_freq` training iterations and after the last training step.
    save_freq: int = 25_000
    optimizer: AdamConfig = field(default_factory=AdamConfig)
    scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)


class EpisodeAwareSampler:
    def __init__(
        self,
        episode_data_index: dict,
        episode_indices_to_use: Union[list, None] = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        indices: list[int] = []
        for episode_idx, (start_index, end_index) in enumerate(zip(episode_data_index["from"], episode_data_index["to"], strict=True)):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


def resolve_delta_timestamps(cfg: DiffusionConfig) -> dict[str, list[int]]:
    delta_timestamps = {}
    for key in cfg.input_features.keys() | cfg.output_features.keys():
        if key.startswith("action."):
            delta_timestamps[key] = cfg.action_delta_indices
        elif key.startswith("observation."):
            delta_timestamps[key] = cfg.observation_delta_indices
    return delta_timestamps

def get_step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def update_policy(
    train_metrics: MetricsTracker,
    policy: DiffusionPolicy,
    batch: Any,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    grad_clip_norm: float,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    policy.train()

    loss, output_dict = policy.forward(batch)
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    optimizer.step()
    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if train_metrics:
        train_metrics.loss = loss.item()
        train_metrics.grad_norm = grad_norm.item()
        train_metrics.lr = optimizer.param_groups[0]["lr"]
        train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def train(cfg: TrainConfig) -> None:
    wandb_logger = WandBLogger(replace(cfg.wandb, resume=cfg.resume, output_dir=cfg.output_dir), cfg) if cfg.wandb.enable else None

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = torch.device(cfg.policy.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    print("Creating dataset")
    delta_timestamps = resolve_delta_timestamps(cfg.policy)
    dataset = EpisodeDataset(data_dir=cfg.dataset.data_dir, deltas=delta_timestamps)
    num_episodes = len(dataset.episodes)
    num_frames = sum(len(ep.data) for ep in dataset.episodes.values())
    episode_data_index = {
        "from": [ep.start_idx for ep in dataset.episodes.values()],
        "to": [ep.end_idx for ep in dataset.episodes.values()],
    }

    print("Creating policy")
    policy = DiffusionPolicy(cfg.policy).to(cfg.policy.device)

    print("Creating optimizer and scheduler")
    kwargs = {k: v for k, v in asdict(cfg.optimizer).items() if k not in ("type", "grad_clip_norm")}
    optimizer = torch.optim.Adam(policy.parameters(), **kwargs)

    from diffusers.optimization import get_scheduler
    kwargs = {k: v for k, v in asdict(cfg.scheduler).items() if k != 'type'}
    kwargs = kwargs | {"num_training_steps": cfg.steps, "optimizer": optimizer}
    lr_scheduler = get_scheduler(**kwargs)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        raise RuntimeError("Resuming training is not supported yet")

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    print(f"Output dir: {cfg.output_dir}")
    print(f"cfg.steps={format_big_number(cfg.steps)}")
    print(f"num_frames={format_big_number(num_frames)}")
    print(f"num_episodes={num_episodes}")
    print(f"num_learnable_params={format_big_number(num_learnable_params)}")
    print(f"num_total_params={format_big_number(num_total_params)}")

    # create dataloader for offline training
    sampler = EpisodeAwareSampler(
        episode_data_index,
        drop_n_last_frames=cfg.policy.drop_n_last_frames,
        shuffle=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(cfg.batch_size, num_frames, num_episodes, train_metrics, initial_step=step)

    print("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            lr_scheduler,
            cfg.optimizer.grad_clip_norm,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we increment `step` here
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            print(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            print(f"Checkpoint policy after step {step}")
            checkpoint_dir = Path(cfg.output_dir) / 'checkpoints' / get_step_identifier(step, cfg.steps)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if is_eval_step:
            print(f"Eval policy at step {step}")
            eval_dir = Path(cfg.output_dir) / 'evals' / get_step_identifier(step, cfg.steps)
            eval_info = eval_policy(replace(cfg.eval, output_dir=eval_dir), policy)
            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(cfg.batch_size, num_frames, num_episodes, eval_metrics, initial_step=step)
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            print(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")


if __name__ == "__main__":
    train(TrainConfig.from_cli())
