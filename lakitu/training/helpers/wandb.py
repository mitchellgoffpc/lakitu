import glob
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

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


@dataclass
class WandBConfig:
    enable: bool = False
    silent: bool = True
    resume: bool = False
    disable_artifact: bool = False
    output_dir: Path | None = None
    project: str = "mario64"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    job_name: str | None = None
    mode: str | None = None  # Allowed values: 'online', 'offline' 'disabled'. Defaults to 'online'


class WandBLogger:
    """A helper class to log object using wandb."""

    def __init__(self, cfg: WandBConfig, train_cfg: Any):
        self.cfg = cfg
        assert cfg.output_dir is not None, "output_dir must be set in the config"

        # Set up WandB
        if cfg.silent:
            os.environ["WANDB_SILENT"] = "True"
        import wandb

        if cfg.run_id:
            wandb_run_id = cfg.run_id
        elif cfg.resume:
            wandb_run_id = get_wandb_run_id_from_filesystem(cfg.output_dir)
        else:
            wandb_run_id = None

        wandb.init(
            id=wandb_run_id,
            project=cfg.project,
            entity=cfg.entity,
            name=cfg.job_name,
            notes=cfg.notes,
            dir=cfg.output_dir,
            config=asdict(train_cfg),
            save_code=False,
            job_type="train_eval",
            resume="must" if cfg.resume else None,
            mode=cfg.mode if cfg.mode in ["online", "offline", "disabled"] else "online",
        )
        print(f"Track this run --> {wandb.run.url}")
        self._wandb = wandb

    def log_policy(self, checkpoint_dir: Path) -> None:
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

        wandb_video = self._wandb.Video(video_path)
        self._wandb.log({f"{mode}/video": wandb_video}, step=step)
