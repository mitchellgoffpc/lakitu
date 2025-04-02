import cv2
import csv
import math
import glfw
import argparse
import multiprocessing
from pathlib import Path

from lakitu.env.core import Core, PLUGIN_DEFAULT
from lakitu.env.hooks import VideoExtension, InputPlugin
from lakitu.env.platforms import DEFAULT_DYNLIB, DLL_EXT
from lakitu.env.defs import PluginType, ErrorType, M64pButtons

LIBRARY_PATH = Path('/usr/local/lib')
PLUGINS_PATH = LIBRARY_PATH / 'mupen64plus'
CONFIG_PATH = Path('~/Developer/lakitu/config').expanduser()
DATA_PATH = Path('/usr/local/share/mupen64plus')

BUTTONS = {
    glfw.KEY_L: 'R_DPAD',
    glfw.KEY_J: 'L_DPAD',
    glfw.KEY_I: 'U_DPAD',
    glfw.KEY_K: 'D_DPAD',
    glfw.KEY_ENTER: 'START_BUTTON',
    glfw.KEY_C: 'Z_TRIG',
    glfw.KEY_X: 'B_BUTTON',
    glfw.KEY_SPACE: 'A_BUTTON',
    glfw.KEY_D: 'R_CBUTTON',
    glfw.KEY_A: 'L_CBUTTON',
    glfw.KEY_S: 'D_CBUTTON',
    glfw.KEY_W: 'U_CBUTTON',
    glfw.KEY_PERIOD: 'R_TRIG',
    glfw.KEY_COMMA: 'L_TRIG',
}
AXES = {
    'X_AXIS': {glfw.KEY_LEFT: -1, glfw.KEY_RIGHT: 1},
    'Y_AXIS': {glfw.KEY_DOWN: -1, glfw.KEY_UP: 1},
}

class KeyboardInputPlugin(InputPlugin):
    def __init__(self, core, data_queue=None):
        super().__init__(core, data_queue)
        self.pressed_keys = set()

    def init(self, window):
        super().init(window)
        glfw.set_key_callback(self.window, self.key_callback)

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

    def get_controller_states(self):
        controller_state = M64pButtons()
        for key, button in BUTTONS.items():
            setattr(controller_state, button, int(key in self.pressed_keys))
        x_axis = sum(value for key, value in AXES['X_AXIS'].items() if key in self.pressed_keys)
        y_axis = sum(value for key, value in AXES['Y_AXIS'].items() if key in self.pressed_keys)
        magnitude = math.sqrt(x_axis**2 + y_axis**2) + 1e-6
        controller_state.X_AXIS = int(x_axis / magnitude * 127)
        controller_state.Y_AXIS = int(y_axis / magnitude * 127)
        return [controller_state] + [M64pButtons()] * 3


def encode(data_q):
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    width, height, fps = 640, 480, 30
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    with open('controller_states.csv', 'w', newline='') as csvfile:
        fieldnames = ['frame_index', 'controller_index'] + [field for field, *_ in M64pButtons._fields_]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        frame_count = 0
        while (data := data_q.get()) is not None:
            frame, controller_states = data
            out.write(frame[::-1, :, ::-1])  # Convert RGB to BGR
            for i, state in enumerate(controller_states):
                state_dict = {field: getattr(state, field) for field, *_ in M64pButtons._fields_}
                writer.writerow({'frame_index': frame_count, 'controller_index': i, **state_dict})
            frame_count += 1

    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Lakitu environment')
    parser.add_argument('path', type=str, help='ROM Path')
    args = parser.parse_args()
    assert Path(args.path).is_file(), f"File {args.path!r} does not exist"

    # Create a queue for saving data to disk
    ctx = multiprocessing.get_context('spawn')
    data_queue = ctx.Queue()
    encoder_thread = ctx.Process(target=encode, args=(data_queue,))

    # Load the core and plugins
    core = Core(str(LIBRARY_PATH / 'libmupen64plus.dylib'))
    input_plugin = KeyboardInputPlugin(core, data_queue=data_queue)
    video_extension = VideoExtension(input_plugin)

    core.core_startup(str(CONFIG_PATH), str(DATA_PATH), vidext=video_extension)

    plugin_files = []
    for path in PLUGINS_PATH.iterdir():
        if path.name.startswith("mupen64plus") and path.name.endswith(DLL_EXT) and path.name != DEFAULT_DYNLIB:
            plugin_files.append(str(path))

    for plugin_path in plugin_files:
        core.plugin_load_try(plugin_path)

    for plugin_type in core.plugin_map.keys():
        for plugin_handle, _plugin_path, plugin_name, plugin_desc, _plugin_version in core.plugin_map[plugin_type].values():
            core.plugin_startup(plugin_handle, plugin_name, plugin_desc)

    # Open the ROM file
    with open(args.path, 'rb') as f:
        romfile = f.read()
    rval = core.rom_open(romfile)
    if rval == ErrorType.SUCCESS:
        core.rom_get_header()
        core.rom_get_settings()

    # Attach plugins
    core.attach_plugins({plugin_type: PLUGIN_DEFAULT[plugin_type] for plugin_type in (PluginType.GFX, PluginType.AUDIO, PluginType.RSP)})
    core.override_input_plugin(input_plugin)

    # Run the game
    encoder_thread.start()
    core.execute()

    data_queue.put(None)
    encoder_thread.join()
    core.detach_plugins()
    core.rom_close()
