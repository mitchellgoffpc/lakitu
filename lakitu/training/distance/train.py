#!/usr/bin/env python3
import random
import time
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib import recfunctions as rfn
from torch.optim import Optimizer

from lakitu.datasets.dataset import EpisodeDataset
from lakitu.training.distance.model import DistanceEstimatorConfig, DistanceEstimator
from lakitu.training.helpers.checkpoint import save_checkpoint
from lakitu.training.helpers.config import BaseConfig
from lakitu.training.helpers.metrics import AverageMeter, MetricsTracker, format_big_number
from lakitu.training.helpers.wandb import WandBLogger, WandBConfig

OUTPUT_DIR = Path(__file__).parents[1] / "experiments"
LRScheduler = Any

@dataclass
class DatasetConfig:
    data_dirs: list[Path] = field(default_factory=lambda: [Path(__file__).parents[2] / 'data' / 'episodes'])
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
    model: DistanceEstimatorConfig = field(default_factory=DistanceEstimatorConfig)
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    resume: bool = False
    seed: int | None = 1000
    num_workers: int = 4
    batch_size: int = 64
    steps: int = 100_000
    log_freq: int = 200
    save_checkpoint: bool = True
    save_freq: int = 25_000
    optimizer: AdamConfig = field(default_factory=AdamConfig)
    scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)


class DistanceDataset(EpisodeDataset):
    def __init__(self, data_dirs: list[Path], deltas: dict[str, list[int]]) -> None:
        super().__init__(data_dirs=data_dirs, deltas=deltas)
        for episode in self.episodes.values():
            distances = np.full(len(episode.data), 1000, dtype=np.int32)
            episode.data = rfn.append_fields(episode.data, 'info.distance', distances, dtypes=[np.dtype(np.int32)], usemask=False)


def get_step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def update_model(
    train_metrics: MetricsTracker,
    model: DistanceEstimator,
    batch: Any,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    grad_clip_norm: float,
) -> tuple[MetricsTracker, dict[str, float]]:
    start_time = time.perf_counter()
    model.train()

    loss, output_dict = model.compute_loss(batch)
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
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
    return train_metrics, {} # k: v.item() for k, v in output_dict.items()}


def train(cfg: TrainConfig) -> None:
    wandb_logger = WandBLogger(replace(cfg.wandb, resume=cfg.resume, output_dir=cfg.output_dir), cfg) if cfg.wandb.enable else None

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = torch.device(cfg.model.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    print("Creating dataset")
    delta_timestamps = {k: [0] for k in cfg.model.input_features.keys() | cfg.model.output_features.keys()}
    dataset = DistanceDataset(cfg.dataset.data_dirs, deltas=delta_timestamps)
    num_episodes = len(dataset.episodes)
    num_frames = sum(len(ep.data) for ep in dataset.episodes.values())

    print("Creating model")
    if cfg.resume:
        print(f"Resuming from checkpoint {cfg.output_dir}")
        checkpoints_dir = Path(cfg.output_dir) / 'checkpoints'
        step = int(sorted(checkpoints_dir.iterdir(), reverse=True)[0].name)
        checkpoint_dir = checkpoints_dir / get_step_identifier(step, cfg.steps) / 'pretrained_model'
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
        model = DistanceEstimator.from_pretrained(checkpoint_dir, device=device)
    else:
        model = DistanceEstimator(cfg.model).to(device)
        step = 0

    print("Creating optimizer and scheduler")
    kwargs = {k: v for k, v in asdict(cfg.optimizer).items() if k not in ("type", "grad_clip_norm")}
    optimizer = torch.optim.Adam(model.parameters(), **kwargs)

    from diffusers.optimization import get_scheduler
    kwargs = {k: v for k, v in asdict(cfg.scheduler).items() if k != 'type'}
    kwargs = kwargs | {"num_training_steps": cfg.steps, "optimizer": optimizer}
    lr_scheduler = get_scheduler(**kwargs)

    num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())

    print(f"Output dir: {cfg.output_dir}")
    print(f"cfg.steps={format_big_number(cfg.steps)}")
    print(f"num_frames={format_big_number(num_frames)}")
    print(f"num_episodes={num_episodes}")
    print(f"num_learnable_params={format_big_number(num_learnable_params)}")
    print(f"num_total_params={format_big_number(num_total_params)}")

    # create dataloader for offline training
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    model.train()

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

        train_tracker, output_dict = update_model(
            train_tracker,
            model,
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

        if is_log_step:
            print(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            print(f"Checkpoint model after step {step}")
            checkpoint_dir = Path(cfg.output_dir) / 'checkpoints' / get_step_identifier(step, cfg.steps)
            save_checkpoint(checkpoint_dir, step, cfg, model, optimizer, lr_scheduler)
            if wandb_logger:
                wandb_logger.log_checkpoint(checkpoint_dir)


if __name__ == "__main__":
    train(TrainConfig.from_cli())
