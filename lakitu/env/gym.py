import struct
import numpy as np
import multiprocessing
import gymnasium as gym
from pathlib import Path

from lakitu.env.core import Core
from lakitu.env.hooks import VideoExtension, InputExtension
from lakitu.env.defs import PluginType, ErrorType, CoreState, M64pButtons

LIBRARY_PATH = Path('/usr/local/lib')
PLUGINS_PATH = LIBRARY_PATH / 'mupen64plus'
CONFIG_PATH = Path(__file__).parent / 'lib'
DATA_PATH = Path('/usr/local/share/mupen64plus')

class RemoteInputExtension(InputExtension):
    """Input plugin that receives controller states from a queue"""

    def __init__(self, core, input_queue, data_queue, savestate_path=None, info_hooks=None):
        super().__init__(core, data_queue, savestate_path, info_hooks)
        self.input_queue = input_queue

    def get_controller_states(self):
        controller_states = [M64pButtons() for _ in range(4)]
        next_input = self.input_queue.get()
        match next_input:
            case "STOP":
                self.core.stop()
            case "RESET":
                self.core.reset()
            case ("SAVE", savestate_path):
                savestate_path.parent.mkdir(parents=True, exist_ok=True)
                self.core.state_save(str(savestate_path))
                return self.get_controller_states()  # Wait for the next input
            case list(states):
                for i, state in enumerate(states):
                    controller_states[i] = state
            case _:
                raise ValueError(f"Received unknown command in RemoteInputExtension: {next_input}")
        return controller_states


def emulator_process(rom_path, savestate_path, input_queue, data_queue, info_hooks):
    """Process that runs the emulator"""
    # Load the core and plugins
    core = Core(log_level=0)  # No logging
    input_extension = RemoteInputExtension(core, input_queue, data_queue, savestate_path, info_hooks)
    video_extension = VideoExtension(input_extension, offscreen=True)
    core.core_startup(vidext=video_extension, inputext=input_extension)
    core.load_plugins()

    # Open the ROM file
    with open(rom_path, 'rb') as f:
        romfile = f.read()
    rval = core.rom_open(romfile)
    if rval == ErrorType.SUCCESS:
        core.rom_get_header()
        core.rom_get_settings()

    # Run the game
    core.attach_plugins([PluginType.GFX, PluginType.INPUT, PluginType.RSP])
    core.core_state_set(CoreState.SPEED_LIMITER, 0)
    core.execute()
    core.detach_plugins()
    core.rom_close()


class N64Env(gym.Env):
    """Gymnasium environment for N64"""
    action_space: gym.spaces.Dict
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, rom_path, savestate_path=None, render_mode=None, info_hooks=None):
        super().__init__()
        self.rom_path = str(rom_path)
        self.savestate_path = Path(savestate_path) if savestate_path else None
        self.render_mode = render_mode
        self.info_hooks = info_hooks
        self.emulator_proc = None
        self.current_frame = None

        # Create communication queues
        self.ctx = multiprocessing.get_context('spawn')
        self.input_queue = self.ctx.Queue()
        self.data_queue = self.ctx.Queue()

        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Dict({
            'joystick': gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            'buttons': gym.spaces.MultiBinary(14),
        })

        if self.savestate_path and not self.savestate_path.exists():
            raise FileNotFoundError(f"Could not read savestate file at {savestate_path}")

    def _start_emulator(self):
        """Start the emulator in a separate process"""
        if self.emulator_proc is not None and self.emulator_proc.is_alive():
            return

        self.emulator_proc = self.ctx.Process(
            target=emulator_process,
            args=(self.rom_path, str(self.savestate_path), self.input_queue, self.data_queue, self.info_hooks),
            daemon=True
        )
        self.emulator_proc.start()

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)

        # Start the emulator if not already running
        self.current_frame = None
        if self.emulator_proc is None:
            self._start_emulator()
            self.input_queue.put([M64pButtons()])  # Send an initial empty state to the emulator
        else:
            self.input_queue.put("RESET")
        frame, _, info = self.data_queue.get()  # Wait for the emulator to reset

        return frame, info

    def step(self, action):
        """Take a step in the environment"""
        if self.emulator_proc is None:
            raise RuntimeError("You must call reset() before step()")
        if not self.emulator_proc.is_alive():
            raise RuntimeError("Emulator process has died")

        if not isinstance(action, dict):
            raise ValueError("action must be a dictionary")
        if not isinstance(action['joystick'], np.ndarray) or action['joystick'].dtype != np.float32:
            raise ValueError("action['joystick'] must be an array with dtype float32")
        if not isinstance(action['buttons'], np.ndarray) or action['buttons'].dtype != bool:
            raise ValueError("action['buttons'] must be an array with dtype bool")

        # Create controller state
        joystick = action['joystick']
        magnitude = np.linalg.norm(joystick)
        if magnitude > 1.0:
            joystick = joystick / magnitude  # Normalize to unit circle

        controller_state = M64pButtons()
        controller_state.X_AXIS = int(joystick[0] * 127)
        controller_state.Y_AXIS = int(joystick[1] * 127)
        for i, button_name in enumerate(M64pButtons.get_button_fields()):
            setattr(controller_state, button_name, int(action['buttons'][i]))

        self.input_queue.put([controller_state])
        frame, _, info = self.data_queue.get()

        observation = frame[::-1]  # Flip vertically
        terminated = False
        truncated = False
        reward = 0.0

        self.current_frame = observation
        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment"""
        if self.render_mode == "rgb_array" and self.current_frame is not None:
            return self.current_frame
        return None

    def close(self):
        """Clean up resources"""
        if self.emulator_proc and self.emulator_proc.is_alive():
            self.input_queue.put("STOP")
            self.emulator_proc.join(timeout=5)
            if self.emulator_proc.is_alive():
                self.emulator_proc.kill()

        # Clear queues
        while not self.input_queue.empty():
            self.input_queue.get_nowait()
        while not self.data_queue.empty():
            self.data_queue.get_nowait()

    def savestate(self, path):
        """Save the current state of the emulator"""
        if self.emulator_proc and self.emulator_proc.is_alive():
            self.input_queue.put(("SAVE", path))


# Info hooks

def m64_get_level(core):
    mem = core.core_mem_read(0x8032DDF8, 2)
    return struct.unpack('>H', mem)[0]  # n64 is big endian


# Example usage

if __name__ == "__main__":
    import argparse
    import cv2
    import einops
    import pygame
    import torch

    from lakitu.training.diffusion.policy import DiffusionPolicy
    from lakitu.datasets.dataset import draw_actions, draw_info
    from lakitu.datasets.format import load_data

    parser = argparse.ArgumentParser(description='Run N64 Gym Environment')
    parser.add_argument('rom_path', type=str, help='Path to the ROM file')
    parser.add_argument('-s', '--savestate', type=str, default=None, help='Path to save state file')
    parser.add_argument('-p', '--policy', type=str, default=None, help='Path to policy file')
    parser.add_argument('-r', '--replay', type=str, default=None, help="Path of episode to replay")
    args = parser.parse_args()

    # Initialize Pygame
    W, H = 320, 240
    pygame.init()
    screen = pygame.display.set_mode((W * 2, H * 2 + 100))
    pygame.display.set_caption('N64')
    clock = pygame.time.Clock()

    savestate_path = Path(args.replay) / "initial_state.m64p" if args.replay and not args.savestate else args.savestate
    env = N64Env(args.rom_path, savestate_path, render_mode="rgb_array", info_hooks={'level': m64_get_level})
    observation, info = env.reset()

    if args.policy:
        policy = DiffusionPolicy.from_pretrained(Path(args.policy), device='cuda' if torch.cuda.is_available() else 'cpu')
        policy.reset()
    elif args.replay:
        episode_data = load_data(Path(args.replay) / 'episode.data')

    frame_idx = 0
    while True:
        if any(event.type == pygame.QUIT for event in pygame.event.get()):
            break
        elif args.replay and frame_idx >= len(episode_data['action.joystick']):
            break

        # Sample action from policy or random
        if args.policy:
            observation = cv2.resize(observation, (W, H))
            observation_tensor = torch.as_tensor(observation[None]).to(policy.config.device)
            observation_tensor = einops.rearrange(observation_tensor, "b h w c -> b c h w").contiguous().float() / 255.0
            action_tensor = policy.select_action({'observation.image': observation_tensor})
            action = {k.removeprefix("action."): v.cpu().numpy()[0] for k, v in action_tensor.items()}
        elif args.replay:
            action = {k: episode_data[f'action.{k}'][frame_idx] for k in env.action_space.keys()}
        else:
            action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        # Convert numpy array to pygame surface and display
        surf = pygame.surfarray.make_surface(observation.swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        draw_actions(screen, action['joystick'], action['buttons'], H * 2, W * 2, 100)
        draw_info(screen, info, H * 2, W * 2)

        pygame.display.flip()
        clock.tick(30)
        frame_idx += 1

        if terminated or truncated:
            observation, info = env.reset()
            if args.policy:
                policy.reset()

    env.close()
    pygame.quit()
