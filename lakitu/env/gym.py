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

# Button mapping (same order as in the action space)
BUTTON_NAMES = [
    'A_BUTTON', 'B_BUTTON', 'Z_TRIG', 'START_BUTTON',
    'U_DPAD', 'D_DPAD', 'L_DPAD', 'R_DPAD',
    'U_CBUTTON', 'D_CBUTTON', 'L_CBUTTON', 'R_CBUTTON',
    'L_TRIG', 'R_TRIG'
]

class RemoteInputExtension(InputExtension):
    """Input plugin that receives controller states from a queue"""

    def __init__(self, core, input_queue, data_queue, savestate_path=None):
        super().__init__(core, data_queue, savestate_path)
        self.input_queue = input_queue

    def get_controller_states(self):
        controller_states = [M64pButtons() for _ in range(4)]
        next_input = self.input_queue.get()
        if next_input == "STOP":
            self.core.stop()
        elif next_input == "RESET":
            self.core.reset()
        else:
            for i, state in enumerate(next_input):
                controller_states[i] = state
        return controller_states


def emulator_process(rom_path, savestate_path, input_queue, data_queue):
    """Process that runs the emulator"""
    # Load the core and plugins
    core = Core()
    input_extension = RemoteInputExtension(core, input_queue, data_queue, savestate_path)
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
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, rom_path, savestate_path=None, render_mode=None):
        super().__init__()
        self.rom_path = rom_path
        self.savestate_path = savestate_path
        self.render_mode = render_mode
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

    def _start_emulator(self):
        """Start the emulator in a separate process"""
        if self.emulator_proc is not None and self.emulator_proc.is_alive():
            return

        self.emulator_proc = self.ctx.Process(
            target=emulator_process,
            args=(self.rom_path, self.savestate_path, self.input_queue, self.data_queue),
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

        controller_state = M64pButtons()
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
        info: dict = {}

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
    parser.add_argument('-s', '--savestate', type=str, default=None, help='Path to save state file')
    args = parser.parse_args()

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption('N64')
    clock = pygame.time.Clock()

    env = N64Env(args.rom_path, args.savestate, render_mode="rgb_array")
    observation, info = env.reset()

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
        clock.tick(30)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    pygame.quit()
