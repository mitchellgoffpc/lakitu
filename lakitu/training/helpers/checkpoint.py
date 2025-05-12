import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from torch import nn
from torch.optim import Optimizer

LRScheduler = Any

def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: Any,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
) -> None:
    import safetensors

    pretrained_dir = checkpoint_dir / 'pretrained_model'
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(model.state_dict(), str(pretrained_dir / "model.safetensors"))
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
