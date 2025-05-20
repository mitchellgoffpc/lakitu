import gymnasium as gym
import multiprocessing as mp
import numpy as np
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional, Callable

from lakitu.env.core import Core
from lakitu.env.hooks import VideoExtension, InputExtension
from lakitu.env.defs import PluginType, ErrorType, CoreState, M64pButtons

LIBRARY_PATH = Path('/usr/local/lib')
PLUGINS_PATH = LIBRARY_PATH / 'mupen64plus'
CONFIG_PATH = Path(__file__).parent / 'lib'
DATA_PATH = Path('/usr/local/share/mupen64plus')

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GRAY = (128, 128, 128)

class ControlMode(IntEnum):
    HUMAN = 0
    MODEL = 1
    REPLAY = 2


# Helper functions

def draw_info(screen, info, base_y, width):
    """Draw environment info on the bottom right of the pygame window"""
    import pygame
    font = pygame.font.Font(None, 24)

    # Draw text
    x, y = width - 20, base_y
    for key, value in info.items():
        text = font.render(f"{key}: {value}", True, WHITE)
        text_rect = text.get_rect(topright=(x, y))
        screen.blit(text, text_rect)
        y += 25

def draw_actions(screen, joystick, buttons, base_y, width, height):
    """Draw action inputs at the bottom of the pygame window"""
    import pygame
    font = pygame.font.Font(None, 24)

    # Draw background
    pygame.draw.rect(screen, BLACK, (0, base_y, width * 2, height))

    # Draw joystick
    joy_x, joy_y = joystick
    center_x, center_y = 80, base_y + 50
    pygame.draw.circle(screen, GRAY, (center_x, center_y), 30)
    stick_x = center_x + joy_x.item() * 25
    stick_y = center_y - joy_y.item() * 25
    pygame.draw.circle(screen, RED, (stick_x, stick_y), 10)

    # Draw buttons
    button_names = M64pButtons.get_button_fields()
    button_states = dict(zip(button_names, buttons, strict=True))
    button_states = {k.split('_')[0]: button_states[k] for k in ('A_BUTTON', 'B_BUTTON', 'Z_TRIG', 'START_BUTTON')}
    for i, (name, state) in enumerate(button_states.items()):
        x = 160 + (i * 60)
        y = base_y + 50
        color = RED if state else GRAY
        pygame.draw.circle(screen, color, (x, y), 15)
        text = font.render(name, True, WHITE)
        text_rect = text.get_rect(center=(x, y))
        screen.blit(text, text_rect)


# Gym environment

def emulator_process(
    rom_path: Path,
    savestate_path: Optional[Path],
    input_queue: mp.Queue,
    data_queue: mp.Queue,
    info_hooks: Optional[dict[str, Callable]]
) -> None:
    """Process that runs the emulator"""
    core = Core(log_level=0)  # No logging
    input_extension = RemoteInputExtension(core, input_queue, data_queue, savestate_path, info_hooks)
    video_extension = VideoExtension(input_extension, offscreen=True)
    core.core_startup(vidext=video_extension, inputext=input_extension)
    core.load_plugins()

    with open(rom_path, 'rb') as f:
        romfile = f.read()
    rval = core.rom_open(romfile)
    if rval == ErrorType.SUCCESS:
        core.rom_get_header()
        core.rom_get_settings()

    core.attach_plugins([PluginType.GFX, PluginType.INPUT, PluginType.RSP])
    core.core_state_set(CoreState.SPEED_LIMITER, 0)
    core.execute()
    core.detach_plugins()
    core.rom_close()


class RemoteInputExtension(InputExtension):
    """Input plugin that receives controller states from a queue"""

    def __init__(self,
        core: Core,
        input_queue: mp.Queue,
        data_queue: mp.Queue,
        savestate_path: Optional[Path] = None,
        info_hooks: Optional[dict[str, Callable]] = None
    ) -> None:
        super().__init__(core, data_queue, savestate_path, info_hooks)
        self.input_queue = input_queue

    def get_controller_states(self) -> list[M64pButtons]:
        controller_states = [M64pButtons() for _ in range(4)]
        next_input = self.input_queue.get()
        match next_input:
            case "STOP":
                self.core.stop()
            case "RESET":
                self.core.reset()
                if self.savestate_path:
                    self.core.state_load(str(self.savestate_path))
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


class N64Env(gym.Env):
    """Gymnasium environment for N64"""
    action_space: gym.spaces.Dict
    observation_space: gym.spaces.Box
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self,
        rom_path: Path,
        savestate_path: Optional[Path] = None,
        render_mode: Optional[str] = None,
        info_hooks: Optional[dict[str, Callable]] = None
    ) -> None:
        super().__init__()
        self.rom_path = rom_path
        self.savestate_path = savestate_path
        self.render_mode = render_mode
        self.info_hooks = info_hooks
        self.emulator_proc: Optional[mp.process.BaseProcess] = None
        self.current_frame: Optional[np.ndarray] = None

        self.ctx = mp.get_context('spawn')
        self.input_queue: mp.Queue = self.ctx.Queue()
        self.data_queue: mp.Queue = self.ctx.Queue()

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Dict({
            'joystick': gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            'buttons': gym.spaces.MultiBinary(14),
        })

        if self.savestate_path and not self.savestate_path.exists():
            raise FileNotFoundError(f"Could not read savestate file at {savestate_path}")

    def _start_emulator(self) -> None:
        """Start the emulator in a separate process"""
        if self.emulator_proc is not None and self.emulator_proc.is_alive():
            return

        self.emulator_proc = self.ctx.Process(
            target=emulator_process,
            args=(self.rom_path, self.savestate_path, self.input_queue, self.data_queue, self.info_hooks),
            daemon=True
        )
        self.emulator_proc.start()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment"""
        super().reset(seed=seed)

        self.current_frame = None
        if self.emulator_proc is None:
            self._start_emulator()
            self.input_queue.put([M64pButtons()])  # Send an initial empty state to the emulator
        else:
            self.input_queue.put("RESET")
        frame, _, info = self.data_queue.get()  # Wait for the emulator to reset

        return frame, info

    def step(self, action: dict[str, np.ndarray]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
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
        observation, _, info = self.data_queue.get()

        terminated = False
        truncated = False
        reward = 0.0

        self.current_frame = observation
        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:  # type: ignore[override]
        """Render the environment"""
        if self.render_mode == "rgb_array" and self.current_frame is not None:
            return self.current_frame
        return None

    def close(self) -> None:
        """Clean up resources"""
        if self.emulator_proc and self.emulator_proc.is_alive():
            self.input_queue.put("STOP")
            self.emulator_proc.join(timeout=5)
            if self.emulator_proc.is_alive():
                self.emulator_proc.kill()

        while not self.input_queue.empty():
            self.input_queue.get_nowait()
        while not self.data_queue.empty():
            self.data_queue.get_nowait()

    def savestate(self, path: Path) -> None:
        """Save the current state of the emulator"""
        if self.emulator_proc and self.emulator_proc.is_alive():
            self.input_queue.put(("SAVE", path))


# Entry point for testing

if __name__ == "__main__":
    import argparse
    import cv2
    import datetime
    import einops
    import pygame
    import torch

    from lakitu.datasets.format import load_data
    from lakitu.datasets.write import encode
    from lakitu.env.games import m64_get_level
    from lakitu.env.run import GamepadController, KeyboardController, combine_controller_states
    from lakitu.training.diffusion.policy import DiffusionPolicy

    class PygameKeyboardController(KeyboardController):
        JOYSTICK = {
            'X_AXIS': {pygame.K_LEFT: -1, pygame.K_RIGHT: 1},
            'Y_AXIS': {pygame.K_DOWN: -1, pygame.K_UP: 1},
        }

        KEYMAP = {
            'R_DPAD': pygame.K_l,
            'L_DPAD': pygame.K_j,
            'U_DPAD': pygame.K_i,
            'D_DPAD': pygame.K_k,
            'START_BUTTON': pygame.K_RETURN,
            'Z_TRIG': pygame.K_x,
            'B_BUTTON': pygame.K_c,
            'A_BUTTON': pygame.K_SPACE,
            'R_CBUTTON': pygame.K_d,
            'L_CBUTTON': pygame.K_a,
            'D_CBUTTON': pygame.K_s,
            'U_CBUTTON': pygame.K_w,
            'R_TRIG': pygame.K_PERIOD,
            'L_TRIG': pygame.K_COMMA,
        }

    parser = argparse.ArgumentParser(description='Run N64 Gym Environment')
    parser.add_argument('rom_path', type=str, help='Path to the ROM file')
    parser.add_argument('-s', '--savestate', type=str, default=None, help='Path to save state file')
    parser.add_argument('-p', '--policy', type=str, default=None, help='Path to policy file')
    parser.add_argument('-r', '--replay', type=str, default=None, help="Path of episode to replay")
    parser.add_argument('-o', '--output', type=str, default=None, help='Path to output directory')
    args = parser.parse_args()

    savestate_path = None
    if args.savestate:
        savestate_path = Path(args.savestate)
    elif args.replay:
        savestate_path = Path(args.replay) / "initial_state.m64p"

    if not Path(args.rom_path).is_file():
        raise FileNotFoundError(f"ROM file {Path(args.rom_path)} does not exist")
    if savestate_path and not savestate_path.is_file():
        raise FileNotFoundError(f"Savestate file {savestate_path} does not exist")

    # Initialize Pygame
    W, H = 320, 240
    pygame.init()
    screen = pygame.display.set_mode((W * 2, H * 2 + 100))
    pygame.display.set_caption('N64')
    clock = pygame.time.Clock()
    gamepad = GamepadController()
    keyboard = PygameKeyboardController()

    # Create the encoder process if recording
    data_queue = None
    ctx = mp.get_context('spawn')
    if args.output:
        data_queue = ctx.Queue()
        info_fields = [('level', np.dtype(np.uint8), ()), ('control_mode', np.dtype(np.uint8), ())]
        output_path = Path(args.output) / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        encoder_process = ctx.Process(target=encode, args=(data_queue, output_path, savestate_path, info_fields))
        encoder_process.start()

    env = N64Env(Path(args.rom_path), savestate_path, render_mode="rgb_array", info_hooks={'level': m64_get_level})
    observation, info = env.reset()

    if args.policy:
        policy = DiffusionPolicy.from_pretrained(Path(args.policy), device='cuda' if torch.cuda.is_available() else 'cpu').eval()
        policy.reset()
        control_mode = ControlMode.MODEL
    elif args.replay:
        episode_data = load_data(Path(args.replay) / 'episode.data')
        control_mode = ControlMode.REPLAY
    else:
        control_mode = ControlMode.HUMAN

    frame_idx = 0
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYUP:
                keyboard.keyup(event.key)
            elif event.type == pygame.KEYDOWN:
                keyboard.keydown(event.key)
                if event.key == pygame.K_TAB and args.policy and control_mode is not ControlMode.MODEL:
                    control_mode = ControlMode.MODEL
                    policy.reset()
                if event.key == pygame.K_r:
                    env.reset()
                    if args.policy:
                        policy.reset()

        gamepad_state = gamepad.get_controller_state()
        keyboard_state = keyboard.get_controller_state()
        controller_state = combine_controller_states(gamepad_state, keyboard_state)
        if any(getattr(controller_state, k) for k, *_ in M64pButtons._fields_):
            control_mode = ControlMode.HUMAN
        if control_mode is ControlMode.REPLAY and frame_idx >= len(episode_data['action.joystick']):
            control_mode = ControlMode.HUMAN

        # Get action based on control mode
        if control_mode is ControlMode.MODEL:
            observation_resized = cv2.resize(observation, (W, H))
            observation_tensor = torch.as_tensor(observation_resized[None]).to(policy.config.device)
            observation_tensor = einops.rearrange(observation_tensor, "b h w c -> b c h w").contiguous().float() / 255.0
            action_tensor = policy.select_action({'observation.image': observation_tensor})
            action = {k.removeprefix("action."): v.cpu().numpy()[0] for k, v in action_tensor.items()}
        elif control_mode is ControlMode.REPLAY:
            action = {k: episode_data[f'action.{k}'][frame_idx] for k in env.action_space.keys()}
            action['buttons'] = action['buttons'].astype(bool)  # TODO: remove this after recording the next dataset
        elif control_mode == ControlMode.HUMAN:
            action = {
                'joystick': np.array([getattr(controller_state, k) / 127 for k in M64pButtons.get_joystick_fields()], dtype=np.float32),
                'buttons': np.array([getattr(controller_state, k) for k in M64pButtons.get_button_fields()], dtype=bool),
            }

        # Take step in environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Send frame data to encoder
        if data_queue:
            controller_state = M64pButtons()
            controller_state.X_AXIS = int(action['joystick'][0] * 127)
            controller_state.Y_AXIS = int(action['joystick'][1] * 127)
            for i, button_name in enumerate(M64pButtons.get_button_fields()):
                setattr(controller_state, button_name, int(action['buttons'][i]))
            data_queue.put((observation, [controller_state], {**info, 'control_mode': control_mode.value}))

        # Render
        surf = pygame.surfarray.make_surface(observation.swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        draw_actions(screen, action['joystick'], action['buttons'], H * 2, W * 2, 100)
        draw_info(screen, {**info, 'controller': control_mode.name}, H * 2, W * 2)

        pygame.display.flip()
        clock.tick(30)
        frame_idx += 1

        if terminated or truncated:
            observation, info = env.reset()
            if args.policy:
                policy.reset()

    # Clean up
    if data_queue:
        data_queue.put(None)
        encoder_process.join()

    env.close()
    pygame.quit()
