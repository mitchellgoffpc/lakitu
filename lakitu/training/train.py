#!/usr/bin/env python
import os
import re
import glob
import json
import time
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Union, Iterator

import torch
from torch.optim import Optimizer

from lakitu.datasets.dataset import EpisodeDataset
from lakitu.training.helpers import BaseConfig
from lakitu.training.eval import EvalConfig, eval_policy, set_seed
from lakitu.training.models.diffusion import DiffusionConfig, DiffusionPolicy

OUTPUT_DIR = Path(__file__).parents[1] / "experiments"
LRScheduler = Any

@dataclass
class ImageTransformConfig:
    weight: float = 1.0
    type: str = "Identity"
    kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class ImageTransformsConfig:
    enable: bool = False
    max_num_transforms: int = 3
    random_order: bool = False
    tfs: dict[str, ImageTransformConfig] = field(default_factory=dict)

@dataclass
class DatasetConfig:
    data_dir: Path = Path(__file__).parent.parent / 'data' / 'episodes'
    episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)

@dataclass
class WandBConfig:
    enable: bool = False
    silent: bool = True
    disable_artifact: bool = False
    project: str = "mario64"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    job_name: str | None = None
    mode: str | None = None  # Allowed values: 'online', 'offline' 'disabled'. Defaults to 'online'

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


class AverageMeter:
    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:{avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class MetricsTracker:
    __keys__ = [
        "_batch_size",
        "_num_frames",
        "_avg_samples_per_ep",
        "metrics",
        "steps",
        "samples",
        "episodes",
        "epochs",
    ]

    def __init__(
        self,
        batch_size: int,
        num_frames: int,
        num_episodes: int,
        metrics: dict[str, AverageMeter],
        initial_step: int = 0,
    ):
        self.__dict__.update(dict.fromkeys(self.__keys__))
        self._batch_size = batch_size
        self._num_frames = num_frames
        self._avg_samples_per_ep = num_frames / num_episodes
        self.metrics = metrics

        self.steps = initial_step
        # A sample is an (observation,action) pair, where observation and action
        # can be on multiple timestamps. In a batch, we have `batch_size` number of samples.
        self.samples = self.steps * self._batch_size
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self._num_frames

    def __getattr__(self, name: str) -> int | dict[str, AverageMeter] | AverageMeter | Any:
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.metrics:
            return self.metrics[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dict__:
            super().__setattr__(name, value)
        elif name in self.metrics:
            self.metrics[name].update(value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def step(self) -> None:
        """Updates metrics that depend on 'step' for one step."""
        self.steps += 1
        self.samples += self._batch_size
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self._num_frames

    def __str__(self) -> str:
        display_list = [
            f"step:{format_big_number(self.steps)}",
            # number of samples seen during training
            f"smpl:{format_big_number(self.samples)}",
            # number of episodes seen during training
            f"ep:{format_big_number(self.episodes)}",
            # number of time all unique samples are seen
            f"epch:{self.epochs:.2f}",
            *[str(m) for m in self.metrics.values()],
        ]
        return " ".join(display_list)

    def to_dict(self, use_avg: bool = True) -> dict[str, int | float]:
        """Returns the current metric values (or averages if `use_avg=True`) as a dict."""
        return {
            "steps": self.steps,
            "samples": self.samples,
            "episodes": self.episodes,
            "epochs": self.epochs,
            **{k: m.avg if use_avg else m.val for k, m in self.metrics.items()},
        }

    def reset_averages(self) -> None:
        """Resets average meters."""
        for m in self.metrics.values():
            m.reset()


class WandBLogger:
    """A helper class to log object using wandb."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg.wandb
        self.log_dir = cfg.output_dir
        self.env_fps = cfg.eval.env.fps

        # Set up WandB
        if cfg.wandb.silent:
            os.environ["WANDB_SILENT"] = "True"
        import wandb

        if cfg.wandb.run_id:
            wandb_run_id = cfg.wandb.run_id
        elif cfg.resume:
            wandb_run_id = get_wandb_run_id_from_filesystem(self.log_dir)
        else:
            wandb_run_id = None

        wandb.init(
            id=wandb_run_id,
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.cfg.job_name,
            notes=self.cfg.notes,
            dir=self.log_dir,
            config=asdict(cfg),
            # TODO(rcadene): try set to True
            save_code=False,
            # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
            job_type="train_eval",
            resume="must" if cfg.resume else None,
            mode=self.cfg.mode if self.cfg.mode in ["online", "offline", "disabled"] else "online",
        )
        print(f"Track this run --> {wandb.run.url}")
        self._wandb = wandb

    def log_policy(self, checkpoint_dir: Path) -> None:
        """Checkpoints the policy to wandb."""
        if self.cfg.disable_artifact:
            return

        step_id = checkpoint_dir.name
        artifact_name = f"{step_id}"
        artifact_name = artifact_name.replace(":", "_").replace("/", "_")  # WandB artifacts don't accept ":" or "/" in their name.
        artifact = self._wandb.Artifact(artifact_name, type="model")
        artifact.add_file(checkpoint_dir / 'pretrained_model' / 'model.safetensors')
        self._wandb.log_artifact(artifact)

    def log_dict(self, d: dict, step: int, mode: str = "train") -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        for k, v in d.items():
            if not isinstance(v, (int, float, str)):
                print(f'WandB logging of key "{k}" was ignored as its type is not handled by this wrapper.')
                continue
            self._wandb.log({f"{mode}/{k}": v}, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        wandb_video = self._wandb.Video(video_path, fps=self.env_fps, format="mp4")
        self._wandb.log({f"{mode}/video": wandb_video}, step=step)


def get_wandb_run_id_from_filesystem(log_dir: Path) -> str:
    paths = glob.glob(str(log_dir / "wandb/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    wandb_run_id = match.groups(0)[0]
    assert isinstance(wandb_run_id, str)  # to make mypy happy
    return wandb_run_id

def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

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


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainConfig,
    policy: DiffusionPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler,
) -> None:
    import safetensors

    pretrained_dir = checkpoint_dir / 'pretrained_model'
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(policy.state_dict(), str(pretrained_dir / "model.safetensors"))
    with open(pretrained_dir / "train_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=lambda x: str(x) if isinstance(x, Path) else x)

    save_dir = checkpoint_dir / 'training_state'
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "training_step.json", "w") as f:
        json.dump({"step": step}, f, indent=2)

    if optimizer is not None:
        state = optimizer.state_dict()
        param_groups = state.pop("param_groups")
        flat_state = flatten_dict(state)
        safetensors.torch.save_file(flat_state, save_dir / "optimizer_state.safetensors")
        with open(save_dir / "optimizer_param_groups.json", "w") as f:
            json.dump(param_groups, f, indent=2)
    if scheduler is not None:

        state_dict = scheduler.state_dict()
        with open(save_dir / "scheduler_state.json", "w") as f:
            json.dump(state_dict, f, indent=2)

def format_big_number(num, precision=0):
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor

    return num

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def make_optimizer_and_scheduler(cfg: TrainConfig, policy: DiffusionPolicy) -> tuple[Optimizer, LRScheduler]:
    kwargs = {k: v for k, v in asdict(cfg.optimizer).items() if k not in ("type", "grad_clip_norm")}
    optimizer = torch.optim.Adam(policy.parameters(), **kwargs)

    from diffusers.optimization import get_scheduler
    kwargs = {k: v for k, v in asdict(cfg.scheduler).items() if k != 'type'}
    kwargs = kwargs | {"num_training_steps": cfg.steps, "optimizer": optimizer}
    lr_scheduler = get_scheduler(**kwargs)

    return optimizer, lr_scheduler


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
    # print(json.dumps(asdict(cfg), indent=2))
    wandb_logger = WandBLogger(cfg) if cfg.wandb.enable else None

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
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

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
