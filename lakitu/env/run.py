import av
import cv2
import hid
import glfw
import json
import struct
import shutil
import argparse
import datetime
import threading
import numpy as np
import multiprocessing
from pathlib import Path

from lakitu.env.core import Core
from lakitu.env.hooks import VideoExtension, InputExtension
from lakitu.env.defs import PluginType, ErrorType, M64pButtons

KEYBOARD_AXES = {
    'X_AXIS': {glfw.KEY_LEFT: -1, glfw.KEY_RIGHT: 1},
    'Y_AXIS': {glfw.KEY_DOWN: -1, glfw.KEY_UP: 1},
}

KEYBOARD_BUTTONS = {
    'R_DPAD': glfw.KEY_L,
    'L_DPAD': glfw.KEY_J,
    'U_DPAD': glfw.KEY_I,
    'D_DPAD': glfw.KEY_K,
    'START_BUTTON': glfw.KEY_ENTER,
    'Z_TRIG': glfw.KEY_C,
    'B_BUTTON': glfw.KEY_X,
    'A_BUTTON': glfw.KEY_SPACE,
    'R_CBUTTON': glfw.KEY_D,
    'L_CBUTTON': glfw.KEY_A,
    'D_CBUTTON': glfw.KEY_S,
    'U_CBUTTON': glfw.KEY_W,
    'R_TRIG': glfw.KEY_PERIOD,
    'L_TRIG': glfw.KEY_COMMA,
}

CONTROLLER_BUTTONS = {
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

def parse_stick_data(report, left=True):
    data = report[6 if left else 9:]
    x_axis = (data[0] | ((data[1] & 0xF) << 8))
    y_axis = ((data[1] >> 4) | (data[2] << 4))
    x_axis = (x_axis - 1900) / 1500 if abs(x_axis - 1900) > 300 else 0  # scale and deadzone
    y_axis = (y_axis - 1900) / 1500 if abs(y_axis - 1900) > 300 else 0
    return x_axis, y_axis


# Input extension for keyboard and gamepad

class KeyboardInputExtension(InputExtension):
    def __init__(self, core, data_queue=None, savestate_path=None):
        super().__init__(core, data_queue, savestate_path)
        self.pressed_keys = set()
        self.gamepad_report = [0] * 64
        self.gamepad_report[6:9] = [0b01101100, 0b11000111, 0b01110110]
        self.gamepad_report[9:12] = [0b01101100, 0b11000111, 0b01110110]
        self.gamepad_thread = threading.Thread(target=self.read_gamepad_data, args=(), daemon=True)
        self.gamepad_thread.start()

    def init(self, window):
        super().init(window)
        glfw.set_key_callback(self.window, self.key_callback)

    def read_gamepad_data(self):
        gamepad_device = next((device for device in hid.enumerate() if device['product_string'] == "Pro Controller"), None)
        if gamepad_device is None:
            return
        gamepad = hid.Device(gamepad_device['vendor_id'], gamepad_device['product_id'])
        gamepad.nonblocking = True
        while True:
            if (report := gamepad.read(64)):
                self.gamepad_report = report

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.RELEASE:
            self.pressed_keys.discard(key)
        elif action == glfw.PRESS:
            self.pressed_keys.add(key)
            if key == glfw.KEY_ESCAPE:
                self.core.stop()
            elif key == glfw.KEY_L:
                self.core.toggle_speed_limit(),
            elif key == glfw.KEY_M:
                self.core.toggle_mute()
            elif key == glfw.KEY_GRAVE_ACCENT:
                savestate_dir = Path(__file__).parent.parent / 'data' / 'savestates'
                savestate_dir.mkdir(parents=True, exist_ok=True)
                savestate_paths = (savestate_dir / f'savestate_{i}.m64p' for i in range(100))
                savestate_path = next(path for path in savestate_paths if not path.exists())
                self.core.state_save(str(savestate_path))

    def get_controller_states(self):
        controller_state = M64pButtons()
        for button in KEYBOARD_BUTTONS:
            pressed = KEYBOARD_BUTTONS[button] in self.pressed_keys or CONTROLLER_BUTTONS[button](self.gamepad_report)
            setattr(controller_state, button, int(pressed))
        ctl_x_axis, ctl_y_axis = parse_stick_data(self.gamepad_report, left=True)
        kb_x_axis = sum(value for key, value in KEYBOARD_AXES['X_AXIS'].items() if key in self.pressed_keys)
        kb_y_axis = sum(value for key, value in KEYBOARD_AXES['Y_AXIS'].items() if key in self.pressed_keys)
        magnitude = np.sqrt(kb_x_axis**2 + kb_y_axis**2) + 1e-6
        x_axis = max(-1, min(1, kb_x_axis / magnitude + ctl_x_axis))
        y_axis = max(-1, min(1, kb_y_axis / magnitude + ctl_y_axis))
        controller_state.X_AXIS = int(x_axis * 127)
        controller_state.Y_AXIS = int(y_axis * 127)
        return [controller_state] + [M64pButtons()] * 3


# Encoder thread

def encode(data_queue, savestate_path):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_path = Path(__file__).parent.parent / 'data' / 'episodes' / current_time
    result_path.mkdir(parents=True, exist_ok=True)

    if savestate_path:
        shutil.copy(savestate_path, result_path / 'initial_state.m64p')

    width, height, fps = 320, 240, 30
    container = av.open(str(result_path / 'episode.mp4'), mode='w')
    stream = container.add_stream('h264', rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'
    stream.codec_context.options = {'crf': '23', 'g': '10'}

    # Header should follow the spec defined in lakitu/datasets/dataset.py
    data_path = result_path / 'episode.data'
    field_defs = [('frame_index', np.uint32, ()), ('action.joystick', np.float32, (2,)), ('action.buttons', np.uint8, (14,))]
    fields = [{'name': name, 'dtype': np.dtype(dtype).name, 'shape': shape} for name, dtype, shape in field_defs]
    header = json.dumps(fields).encode('utf-8')
    header = struct.pack('<I', len(header)) + header
    row_size = sum(np.dtype(dtype).itemsize * (np.prod(shape) if shape else 1) for _, dtype, shape in field_defs)

    frame_count = 0
    with open(data_path, 'wb') as f:
        f.write(header)
        while (data := data_queue.get()) is not None:
            frame, controller_states, _ = data
            frame = cv2.resize(frame[::-1], (width, height))
            av_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            packet = stream.encode(av_frame)
            container.mux(packet)
            joystick = [float(getattr(controller_states[0], field)) / 127 for field in M64pButtons.get_joystick_fields()]
            buttons = [int(getattr(controller_states[0], field)) for field in M64pButtons.get_button_fields()]
            row = struct.pack('<I2f14B', frame_count, *joystick, *buttons)
            assert len(row) == row_size, f"Row size mismatch: {len(row)} != {row_size}"
            f.write(row)
            frame_count += 1

    packet = stream.encode(None)
    container.mux(packet)
    container.close()


# Entry point

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Lakitu environment')
    parser.add_argument('path', type=str, help='ROM Path')
    parser.add_argument('-s', '--savestate', type=str, default=None, help='Path to save state file')
    parser.add_argument('-r', '--record', action='store_true', default=False, help='Record the episode')
    args = parser.parse_args()

    if not Path(args.path).is_file():
        raise FileNotFoundError(f"ROM file {args.path!r} does not exist")
    if args.savestate and not Path(args.savestate).is_file():
        raise FileNotFoundError(f"Savestate file {args.savestate!r} does not exist")

    # Create the encoder thread
    ctx = multiprocessing.get_context('spawn')
    if args.record:
        data_queue = ctx.Queue()
        encoder_thread = ctx.Process(target=encode, args=(data_queue, args.savestate))
        encoder_thread.start()

    # Load the core and plugins
    core = Core()
    input_extension = KeyboardInputExtension(core, data_queue=data_queue if args.record else None, savestate_path=args.savestate)
    video_extension = VideoExtension(input_extension)
    core.core_startup(vidext=video_extension, inputext=input_extension)
    core.load_plugins()

    # Open the ROM file
    with open(args.path, 'rb') as f:
        romfile = f.read()
    rval = core.rom_open(romfile)
    if rval == ErrorType.SUCCESS:
        core.rom_get_header()
        core.rom_get_settings()

    # Run the game
    core.attach_plugins([PluginType.GFX, PluginType.AUDIO, PluginType.INPUT, PluginType.RSP])
    core.execute()

    # Cleanup
    if args.record:
        data_queue.put(None)
        encoder_thread.join()
    core.detach_plugins()
    core.rom_close()
