import cv2
import time
import glfw
import numpy as np
import multiprocessing
import logging as log
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from collections import deque

from lakitu.env.core import Core
from lakitu.env.defs import *
from lakitu.env.hooks import VideoExtension, InputPlugin
from lakitu.env.platforms import DEFAULT_DYNLIB

LIBRARY_PATH = Path('/usr/local/lib')
PLUGINS_PATH = LIBRARY_PATH / 'mupen64plus'
CONFIG_PATH = Path('~/Developer/lakitu/config').expanduser()
DATA_PATH = Path('/usr/local/share/mupen64plus')

# Button mapping (same order as in the action space)
BUTTON_NAMES = [
    'A_BUTTON', 'B_BUTTON', 'Z_TRIG', 'START_BUTTON',
    'U_DPAD', 'D_DPAD', 'L_DPAD', 'R_DPAD',
    'U_CBUTTON', 'D_CBUTTON', 'L_CBUTTON', 'R_CBUTTON',
    'L_TRIG', 'R_TRIG'
]

class RemoteInputPlugin(InputPlugin):
    """Input plugin that receives controller states from a queue"""

    def __init__(self, core, input_queue, data_queue):
        super().__init__(core, data_queue)
        self.input_queue = input_queue

    def get_keys(self, controller, buttons):
        assert self.data_queue.empty(), "Data queue should be empty before processing input"
        if not self.controller_states:
            self.controller_states = {i: Buttons() for i in range(4)}
            match self.input_queue.get():
                case "STOP":
                    self.core.stop()
                case "RESET":
                    self.core.reset()
                case inputs:
                    for i, state in enumerate(inputs):
                        self.controller_states[i] = state

        for field, *_ in Buttons._fields_:
            if controller in self.controller_states:
                setattr(buttons.contents, field, getattr(self.controller_states[controller], field))

    def render_callback(self):
        if self.controller_states:  # Only generate observations once we've received input
            super().render_callback()
        self.controller_states = {}


def emulator_process(rom_path, input_queue, data_queue):
    """Process that runs the emulator"""
    # Load the core and plugins
    core = Core(str(LIBRARY_PATH / 'libmupen64plus.dylib'))
    input_plugin = RemoteInputPlugin(core, input_queue, data_queue)
    video_extension = VideoExtension(input_plugin, offscreen=True)

    core.core_startup(str(CONFIG_PATH), str(DATA_PATH), vidext=video_extension)

    plugin_files = []
    for f in PLUGINS_PATH.iterdir():
        if f.name.startswith("mupen64plus") and f.name.endswith(DLL_EXT) and f.name != DEFAULT_DYNLIB:
            plugin_files.append(str(f))

    for plugin_path in plugin_files:
        core.plugin_load_try(plugin_path)

    for plugin_type in core.plugin_map.keys():
        for plugin_handle, plugin_path, plugin_name, plugin_desc, plugin_version in core.plugin_map[plugin_type].values():
            core.plugin_startup(plugin_handle, plugin_name, plugin_desc)

    # Open the ROM file
    with open(rom_path, 'rb') as f:
        romfile = f.read()
    rval = core.rom_open(romfile)
    if rval == M64ERR_SUCCESS:
        core.rom_get_header()
        core.rom_get_settings()

    # Attach plugins
    core.attach_plugins({plugin_type: PLUGIN_DEFAULT[plugin_type] for plugin_type in (M64PLUGIN_GFX, M64PLUGIN_RSP)})
    core.override_input_plugin(input_plugin)

    # Run the game
    core.execute()
    core.detach_plugins()
    core.rom_close()


class N64Env(gym.Env):
    """Gymnasium environment for N64"""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, rom_path, render_mode=None):
        super().__init__()
        self.rom_path = rom_path
        self.render_mode = render_mode
        self.emulator_proc = None
        self.current_frame = None

        # Create communication queues
        self.ctx = multiprocessing.get_context('spawn')
        self.input_queue = self.ctx.Queue()
        self.data_queue = self.ctx.Queue()

        # Define observation and action spaces
        self.observation_space = spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
        self.action_space = spaces.Dict({
            'joystick': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            'buttons': spaces.MultiBinary(14),
        })

    def _start_emulator(self):
        """Start the emulator in a separate process"""
        if self.emulator_proc is not None and self.emulator_proc.is_alive():
            return

        self.emulator_proc = self.ctx.Process(
            target=emulator_process,
            args=(self.rom_path, self.input_queue, self.data_queue),
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
        else:
            self.input_queue.put("RESET")
            self.data_queue.get()  # Wait for the emulator to reset

        return None, {}

    def step(self, action):
        """Take a step in the environment"""
        if self.emulator_proc is None:
            raise RuntimeError("You must call reset() before step()")
        if not self.emulator_proc.is_alive():
            raise RuntimeError("Emulator process has died")

        # Create controller state
        joystick = action['joystick']
        magnitude = np.linalg.norm(joystick)
        if magnitude > 1.0:
            joystick = joystick / magnitude  # Normalize to unit circle

        controller_state = Buttons()
        controller_state.X_AXIS = int(joystick[0] * 127)
        controller_state.Y_AXIS = int(joystick[1] * 127)
        for i, button_name in enumerate(BUTTON_NAMES):
            setattr(controller_state, button_name, int(action['buttons'][i]))

        self.input_queue.put([controller_state])
        frame, controller_states = self.data_queue.get()

        observation = frame[::-1]  # Flip vertically
        terminated = False
        truncated = False
        reward = 0.0
        info = {}

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


# Example usage
if __name__ == "__main__":
    import argparse
    import pygame
    parser = argparse.ArgumentParser(description='Run N64 Gym Environment')
    parser.add_argument('rom_path', type=str, help='Path to the ROM file')
    args = parser.parse_args()

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption('N64')
    clock = pygame.time.Clock()

    env = N64Env(args.rom_path, render_mode="rgb_array")
    observation, info = env.reset()

    fidx = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Random action
        action = {
            'joystick': np.random.uniform(-1, 1, size=2),
            'buttons': np.random.randint(0, 2, size=14)
        }
        observation, reward, terminated, truncated, info = env.step(action)

        # Convert numpy array to pygame surface and display
        surf = pygame.surfarray.make_surface(observation.swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)

        if terminated or truncated:
            observation, info = env.reset()

        # Reset every 200 frames
        fidx += 1
        if fidx % 200 == 0:
            observation, info = env.reset()

    env.close()
    pygame.quit()
