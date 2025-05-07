#!/usr/bin/env python
import av
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import einops
import gymnasium as gym
import numpy as np
import torch
from tqdm import trange
from omegaconf import OmegaConf

from lakitu.env.gym import N64Env, m64_get_level
from lakitu.training.models.diffusion import DiffusionConfig, DiffusionPolicy

@dataclass
class EnvConfig:
    task: str = "Mario64"
    rom_path: Path = Path(__file__).parents[2] / "Super Mario 64 (USA).z64"
    savestate_path: Path | None = Path(__file__).parents[1] / "data" / "savestates" / "savestate_0.m64p"
    render_mode: str = "rgb_array"
    episode_length: int = 1000

    @property
    def gym_kwargs(self) -> dict[str, Any]:
        return {
            "rom_path": self.rom_path,
            "savestate_path": self.savestate_path,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length
        }

@dataclass
class EvalConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    output_dir: Path | None = None
    num_episodes: int = 4
    num_envs: int = 4
    seed: int = 1000

@dataclass
class EvalPolicyConfig:
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: DiffusionConfig = field(default_factory=DiffusionConfig)
    policy_path: Path | None = None

    @classmethod
    def create(cls, *args: Any) -> "EvalPolicyConfig":
        schema = OmegaConf.structured(cls)
        config = OmegaConf.merge(schema, *args)
        result: EvalPolicyConfig = OmegaConf.to_object(config)  # type: ignore
        return result

    @classmethod
    def from_cli(cls) -> "EvalPolicyConfig":
        return cls.create(OmegaConf.from_cli())


class Mario64Env(N64Env):
    def __init__(self, *args, max_episode_steps=1000, **kwargs):
        super().__init__(*args, **{**kwargs, 'info_hooks': {'level': m64_get_level}})
        self._max_episode_steps = max_episode_steps
        self._step = 0

    def step(self, action):
        self._step += 1
        obs, reward, done, trunc, info = super().step(action)
        info['success'] = info['level'] != 16  # Level 16 is the castle courtyard
        done = done or info['success']
        trunc = self._step >= self._max_episode_steps
        return obs, reward, done, trunc, info


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rollout(env: gym.vector.AsyncVectorEnv, policy: DiffusionPolicy, seeds: list[int], output_paths: list[Path]) -> dict:
    observation: np.ndarray
    reward: np.ndarray

    device = torch.device(policy.config.device)
    policy.reset()
    observation, info = env.reset(seed=seeds)

    containers = []
    streams = []
    for output_path in output_paths:
        width, height, fps = 320, 240, 30
        container = av.open(str(output_path), mode='w')
        stream = container.add_stream('h264', rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        stream.codec_context.options = {'crf': '23', 'g': '30'}
        containers.append(container)
        streams.append(stream)

    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    done = np.array([False] * env.num_envs)
    success = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(max_steps, desc="Running rollouts", leave=False)
    while not np.all(done):
        observation_tensor = torch.as_tensor(observation).to(device)
        observation_tensor = einops.rearrange(observation_tensor, "b h w c -> b c h w").contiguous().float() / 255.0
        action_tensor = policy.select_action({'observation.image': observation_tensor})
        action = {k.removeprefix("action."): v.cpu().numpy() for k, v in action_tensor.items()}

        if output_paths:
            for frame, container, stream in zip(observation, containers, streams, strict=True):
                packet = stream.encode(av.VideoFrame.from_ndarray(frame.astype(np.uint8), format='rgb24'))
                container.mux(packet)

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated | done
        success = success | info['success']

        all_actions.append({k: torch.as_tensor(v) for k, v in action.items()})
        all_rewards.append(torch.as_tensor(reward))
        all_dones.append(torch.as_tensor(done))
        all_successes.append(torch.tensor(success))

        running_success_rate = einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    for container, stream in zip(containers, streams, strict=True):
        packet = stream.encode(None)
        container.mux(packet)
        container.close()

    return {
        "action": {key: torch.stack([action[key] for action in all_actions], dim=1) for key in all_actions[0].keys()},
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }


@torch.no_grad()
def eval_policy(config: EvalConfig, policy: DiffusionPolicy) -> dict:
    env = gym.vector.AsyncVectorEnv([lambda: Mario64Env(**config.env.gym_kwargs)] * config.num_envs, daemon=False)

    policy.eval()
    start = time.time()
    start_seed = config.seed
    num_episodes = config.num_episodes
    n_batches = int(math.ceil(num_episodes / env.num_envs))
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving videos to {config.output_dir}")

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []

    for batch_idx in range(n_batches):
        start_idx, end_idx = batch_idx * env.num_envs, (batch_idx + 1) * env.num_envs
        output_paths = [config.output_dir / f"episode_{i}.mp4" for i in range(start_idx, end_idx)] if config.output_dir else []
        seeds = list(range(start_seed + start_idx, start_seed + end_idx))
        rollout_data = rollout(env, policy, seeds=seeds, output_paths=output_paths)

        # Figure out where in each rollout sequence the first done condition was encountered (results after this won't be included).
        n_steps = rollout_data["done"].shape[1]
        done_indices = torch.argmax(rollout_data["done"].long(), dim=1)  # Get the first done index for each batch element

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()

        sum_rewards.extend(einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum").tolist())
        max_rewards.extend(einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max").tolist())
        all_successes.extend(einops.reduce((rollout_data["success"] * mask), "b n -> b", "any").tolist())
        all_seeds.extend(seeds)

    env.close()

    return {
        "per_episode": [
            {"episode_idx": i, "sum_reward": sum_reward, "max_reward": max_reward, "success": success, "seed": seed}
                for i, (sum_reward, max_reward, success, seed)
                in enumerate(zip(sum_rewards, max_rewards, all_successes, all_seeds, strict=True))
        ][:num_episodes],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:num_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:num_episodes])),
            "pc_success": float(np.nanmean(all_successes[:num_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / num_episodes,
        },
    }


def eval_main(config: EvalPolicyConfig) -> None:
    device = torch.device(config.policy.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(config.eval.seed)

    policy = DiffusionPolicy(config.policy).to(device)
    # policy = DiffusionPolicy.from_pretrained(config.policy, config.policy_path)

    info = eval_policy(config.eval, policy)
    print(info["aggregated"])


if __name__ == "__main__":
    eval_main(EvalPolicyConfig.from_cli())
