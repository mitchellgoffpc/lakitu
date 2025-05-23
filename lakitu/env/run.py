#!/usr/bin/env python3
import argparse
import datetime
import multiprocessing as mp
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import hid
import glfw
import numpy as np

from lakitu.datasets.write import encode
from lakitu.env.core import Core
from lakitu.env.defs import PluginType, ErrorType, M64pButtons
from lakitu.env.games import M64_INFO_HOOKS, M64_INFO_FIELDS, SAVESTATE_DIR
from lakitu.env.hooks import VideoExtension, InputExtension

def parse_stick_data(report: list[int], left: bool = True) -> tuple[float, float]:
    data = report[6 if left else 9:]
    x_axis_raw = (data[0] | ((data[1] & 0xF) << 8))
    y_axis_raw = ((data[1] >> 4) | (data[2] << 4))
    x_axis = (x_axis_raw - 1900) / 1500 if abs(x_axis_raw - 1900) > 300 else 0  # scale and deadzone
    y_axis = (y_axis_raw - 1900) / 1500 if abs(y_axis_raw - 1900) > 300 else 0
    x_axis = max(-1, min(1, x_axis))  # clamp to [-1, 1]
    y_axis = max(-1, min(1, y_axis))
    return x_axis, y_axis

def axis_to_float(value: int) -> float:
    return value / 127.0  # scale to [-1, 1] range

def float_to_axis(value: float) -> int:
    value = max(-1, min(1, value))  # clamp to [-1, 1]
    return int(value * 127)  # scale to [-127, 127] range

def combine_controller_states(*states: M64pButtons) -> M64pButtons:
    combined_state = M64pButtons()
    for button in M64pButtons.get_button_fields():
        setattr(combined_state, button, int(any(getattr(state, button) for state in states)))
    for axis in M64pButtons.get_joystick_fields():
        setattr(combined_state, axis, float_to_axis(sum(axis_to_float(getattr(state, axis)) for state in states)))
    return combined_state


class GamepadController:
    KEYMAP = {
        'R_DPAD': lambda report: (report[5] >> 2) & 1,
        'L_DPAD': lambda report: (report[5] >> 3) & 1,
        'U_DPAD': lambda report: (report[5] >> 1) & 1,
        'D_DPAD': lambda report: (report[5] >> 0) & 1,
        'START_BUTTON': lambda report: (report[4] >> 1) & 1,
        'Z_TRIG': lambda report: ((report[5] >> 7) & 1) or ((report[3] >> 7) & 1),
        'B_BUTTON': lambda report: ((report[3] >> 2) & 1) or ((report[3] >> 1) & 1),
        'A_BUTTON': lambda report: ((report[3] >> 3) & 1) or ((report[3] >> 0) & 1),
        'R_CBUTTON': lambda report: parse_stick_data(report, left=False)[0] > 0.5,
        'L_CBUTTON': lambda report: parse_stick_data(report, left=False)[0] < -0.5,
        'D_CBUTTON': lambda report: parse_stick_data(report, left=False)[1] > 0.5,
        'U_CBUTTON': lambda report: parse_stick_data(report, left=False)[1] < -0.5,
        'R_TRIG': lambda report: (report[3] >> 6) & 1,
        'L_TRIG': lambda report: (report[5] >> 6) & 1,
    }

    def __init__(self) -> None:
        self.active = False
        self.report: list[int] = [0] * 64
        self.report[6:9] = [0b01101100, 0b11000111, 0b01110110]
        self.report[9:12] = [0b01101100, 0b11000111, 0b01110110]
        self.reader = threading.Thread(target=self.read_gamepad_data, args=(), daemon=True)
        self.reader.start()

    def read_gamepad_data(self) -> None:
        gamepad_device = next((device for device in hid.enumerate() if device['product_string'] == "Pro Controller"), None)
        if gamepad_device is None:
            return
        gamepad = hid.Device(gamepad_device['vendor_id'], gamepad_device['product_id'])
        gamepad.nonblocking = True
        while True:
            if (report := gamepad.read(64)):
                self.active = True
                self.report = report

    def get_controller_state(self) -> M64pButtons:
        controller_state = M64pButtons()
        for button in self.KEYMAP:
            setattr(controller_state, button, int(self.KEYMAP[button](self.report)))
        x_axis, y_axis = parse_stick_data(self.report, left=True)
        controller_state.X_AXIS = float_to_axis(x_axis)
        controller_state.Y_AXIS = float_to_axis(y_axis)
        return controller_state


class KeyboardController:
    JOYSTICK = {
        'X_AXIS': {glfw.KEY_LEFT: -1, glfw.KEY_RIGHT: 1},
        'Y_AXIS': {glfw.KEY_DOWN: -1, glfw.KEY_UP: 1},
    }

    KEYMAP = {
        'R_DPAD': glfw.KEY_L,
        'L_DPAD': glfw.KEY_J,
        'U_DPAD': glfw.KEY_I,
        'D_DPAD': glfw.KEY_K,
        'START_BUTTON': glfw.KEY_ENTER,
        'Z_TRIG': glfw.KEY_X,
        'B_BUTTON': glfw.KEY_C,
        'A_BUTTON': glfw.KEY_SPACE,
        'R_CBUTTON': glfw.KEY_D,
        'L_CBUTTON': glfw.KEY_A,
        'D_CBUTTON': glfw.KEY_S,
        'U_CBUTTON': glfw.KEY_W,
        'R_TRIG': glfw.KEY_PERIOD,
        'L_TRIG': glfw.KEY_COMMA,
    }

    def __init__(self) -> None:
        self.pressed_keys: set[int] = set()

    def keyup(self, key: int) -> None:
        self.pressed_keys.discard(key)

    def keydown(self, key: int) -> None:
        self.pressed_keys.add(key)

    def get_controller_state(self) -> M64pButtons:
        controller_state = M64pButtons()
        for button in self.KEYMAP:
            setattr(controller_state, button, int(self.KEYMAP[button] in self.pressed_keys))
        x_axis = sum(value for key, value in self.JOYSTICK['X_AXIS'].items() if key in self.pressed_keys)
        y_axis = sum(value for key, value in self.JOYSTICK['Y_AXIS'].items() if key in self.pressed_keys)
        magnitude = np.sqrt(x_axis**2 + y_axis**2) + 1e-6
        controller_state.X_AXIS = float_to_axis(x_axis / magnitude)
        controller_state.Y_AXIS = float_to_axis(y_axis / magnitude)
        return controller_state


class KeyboardInputExtension(InputExtension):
    def __init__(
        self,
        core: Core,
        data_queue: Optional[mp.Queue] = None,
        savestate_path: Optional[Path] = None,
        info_hooks: Optional[dict[str, Callable]] = None
    ) -> None:
        super().__init__(core, data_queue, savestate_path, info_hooks)
        self.keyboard = KeyboardController()
        self.gamepad = GamepadController()

    def init(self, window: Any) -> None:
        super().init(window)
        glfw.set_key_callback(self.window, self.key_callback)

    def key_callback(self, window: Any, key: int, scancode: int, action: int, mods: int) -> None:
        if action == glfw.RELEASE:
            self.keyboard.keyup(key)
        elif action == glfw.PRESS:
            self.keyboard.keydown(key)
            if key == glfw.KEY_ESCAPE:
                self.core.stop()
            elif key == glfw.KEY_L:
                self.core.toggle_speed_limit()
            elif key == glfw.KEY_M:
                self.core.toggle_mute()
            elif key == glfw.KEY_GRAVE_ACCENT:
                SAVESTATE_DIR.mkdir(parents=True, exist_ok=True)
                savestate_paths = (SAVESTATE_DIR / f'savestate_{i}.m64p' for i in range(100))
                savestate_path = next(path for path in savestate_paths if not path.exists())
                self.core.state_save(str(savestate_path))

    def get_controller_states(self) -> list[M64pButtons]:
        gamepad_state = self.gamepad.get_controller_state()
        keyboard_state = self.keyboard.get_controller_state()
        controller_state = combine_controller_states(keyboard_state, gamepad_state)
        return [controller_state] + [M64pButtons()] * 3


# Entry point

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Lakitu environment')
    parser.add_argument('rom_path', type=str, help='Path to the ROM file')
    parser.add_argument('-s', '--savestate', type=str, default=None, help='Path to savestate file')
    parser.add_argument('-o', '--output', type=str, default=None, help='Path to output directory')
    args = parser.parse_args()

    if not Path(args.rom_path).is_file():
        raise FileNotFoundError(f"ROM file {Path(args.rom_path)} does not exist")
    if args.savestate and not Path(args.savestate).is_file():
        raise FileNotFoundError(f"Savestate file {Path(args.savestate)} does not exist")

    # Create the encoder thread
    data_queue = None
    ctx = mp.get_context('spawn')
    if args.output:
        data_queue = ctx.Queue()
        output_path = Path(args.output) / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        savestate_path = Path(args.savestate) if args.savestate else None
        encoder_thread = ctx.Process(target=encode, args=(data_queue, output_path, savestate_path, M64_INFO_FIELDS))
        encoder_thread.start()

    # Load the core and plugins
    core = Core()
    input_extension = KeyboardInputExtension(core, data_queue, args.savestate, info_hooks=M64_INFO_HOOKS)
    video_extension = VideoExtension(input_extension)
    core.core_startup(vidext=video_extension, inputext=input_extension)
    core.load_plugins()

    # Open the ROM file
    with open(args.rom_path, 'rb') as f:
        romfile = f.read()
    rval = core.rom_open(romfile)
    if rval == ErrorType.SUCCESS:
        core.rom_get_header()
        core.rom_get_settings()

    # Run the game
    core.attach_plugins([PluginType.GFX, PluginType.AUDIO, PluginType.INPUT, PluginType.RSP])
    core.execute()

    # Cleanup
    if args.output:
        assert data_queue is not None  # make mypy happy
        data_queue.put(None)
        encoder_thread.join()
    core.detach_plugins()
    core.rom_close()
