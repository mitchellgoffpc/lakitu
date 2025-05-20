#!/usr/bin/env pythona
import cv2
import math
import multiprocessing as mp
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

from lakitu.datasets.write import encode
from lakitu.env.defs import M64pButtons
from lakitu.env.games import m64_get_level
from lakitu.env.gym import N64Env
from lakitu.training.helpers.config import BaseConfig
from lakitu.training.diffusion.policy import DiffusionPolicy

@dataclass
class EnvConfig:
    rom_path: Path = Path(__file__).parents[3] / "Super Mario 64 (USA).z64"
    savestate_path: Path | None = Path(__file__).parents[2] / "data" / "savestates" / "courtyard.m64p"
    render_mode: str = "rgb_array"
    episode_length: int = 1000
    fps: int = 30

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
class EvalPolicyConfig(BaseConfig):
    policy_path: Path
    num_inference_steps: int = 10
    eval: EvalConfig = field(default_factory=EvalConfig)


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

    def reset(self, *args, **kwargs):
        self._step = 0
        return super().reset(*args, **kwargs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rollout(config: EvalConfig, policy: DiffusionPolicy, env: gym.vector.AsyncVectorEnv, indices: list[int]) -> dict:
    observation: np.ndarray
    reward: np.ndarray

    device = torch.device(policy.config.device)
    policy.reset()
    observation, info = env.reset(seed=[config.seed + idx for idx in indices])

    # Set up encoder processes if output directory is specified
    ctx = mp.get_context('spawn')
    data_queues = []
    encoder_processes = []
    if config.output_dir:
        info_fields = [('level', np.dtype(np.uint8), ())]
        for idx in indices:
            data_queue = ctx.Queue()
            rollout_dir = config.output_dir / f"{idx:04d}"
            savestate_path = config.env.savestate_path
            encoder_process = ctx.Process(target=encode, args=(data_queue, rollout_dir, savestate_path, info_fields), daemon=True)
            encoder_process.start()
            data_queues.append(data_queue)
            encoder_processes.append(encoder_process)

    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    done = np.array([False] * env.num_envs)
    success = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(max_steps, desc="Running rollouts", leave=False)
    while not np.all(done):
        observation = np.stack([cv2.resize(obs, (320, 240)) for obs in observation], axis=0)
        observation_tensor = torch.as_tensor(observation).to(device)
        observation_tensor = einops.rearrange(observation_tensor, "b h w c -> b c h w").contiguous().float() / 255.0
        action_tensor = policy.select_action({'observation.image': observation_tensor})
        action = {k.removeprefix("action."): v.cpu().numpy() for k, v in action_tensor.items()}

        # Send frame data to encoders
        for i, data_queue in enumerate(data_queues):
            controller_state = M64pButtons()
            controller_state.X_AXIS = int(action['joystick'][i][0] * 127)
            controller_state.Y_AXIS = int(action['joystick'][i][1] * 127)
            for j, button_name in enumerate(M64pButtons.get_button_fields()):
                setattr(controller_state, button_name, int(action['buttons'][i][j]))
            data_queue.put((observation[i], [controller_state], {'level': info['level'][i]}))

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

    # Clean up encoder processes
    for data_queue, encoder_process in zip(data_queues, encoder_processes, strict=True):
        data_queue.put(None)
        encoder_process.join()

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
    num_episodes = config.num_episodes
    n_batches = int(math.ceil(num_episodes / env.num_envs))
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving episodes to {config.output_dir}")

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []

    for batch_idx in range(n_batches):
        start_idx, end_idx = batch_idx * env.num_envs, min((batch_idx + 1) * env.num_envs, num_episodes)
        rollout_data = rollout(config, policy, env, indices=list(range(start_idx, end_idx)))

        # Figure out where in each rollout sequence the first done condition was encountered (results after this won't be included).
        n_steps = rollout_data["done"].shape[1]
        done_indices = torch.argmax(rollout_data["done"].long(), dim=1)  # Get the first done index for each batch element

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()

        sum_rewards.extend(einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum").tolist())
        max_rewards.extend(einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max").tolist())
        all_successes.extend(einops.reduce((rollout_data["success"] * mask), "b n -> b", "any").tolist())

    env.close()

    return {
        "per_episode": [
            {"episode_idx": i, "sum_reward": sum_reward, "max_reward": max_reward, "success": success}
                for i, (sum_reward, max_reward, success)
                in enumerate(zip(sum_rewards, max_rewards, all_successes, strict=True))
        ][:num_episodes],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:num_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:num_episodes])),
            "pc_success": float(np.nanmean(all_successes[:num_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / num_episodes,
        },
    }


if __name__ == "__main__":
    config = EvalPolicyConfig.from_cli()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(config.eval.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = DiffusionPolicy.from_pretrained(config.policy_path, num_inference_steps=config.num_inference_steps, device=device)
    info = eval_policy(config.eval, policy)
    print(info["aggregated"])
