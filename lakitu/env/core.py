# -*- coding: utf-8 -*-
# Author: Milan Nikolic <gen2brain@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import struct
import ctypes as C
import logging as log
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from lakitu.env.loader import load, unload_library, get_dll_ext
from lakitu.env.defs import LogLevel, ErrorType, PluginType, CoreFlags, CoreState, CoreCommand, EmulationState
from lakitu.env.defs import M64pRomHeader, M64pRomSettings

if TYPE_CHECKING:
    from lakitu.env.hooks import InputExtension, VideoExtension

CORE_API_VERSION = 0x20001
CONFIG_API_VERSION = 0x20302
VIDEXT_API_VERSION = 0x030300
MINIMUM_CORE_VERSION = 0x020600

CORE_LIB_PATH = Path(__file__).parent / 'lib'

ROM_TYPE = {
    b'80371240': 'z64 (native)',
    b'37804012': 'v64 (byteswapped)',
    b'40123780': 'n64 (wordswapped)'
}

PLUGIN_NAMES = {
    PluginType.RSP: "RSP",
    PluginType.GFX: "Video",
    PluginType.AUDIO: "Audio",
    PluginType.INPUT: "Input"
}

PLUGIN_PATHS = {
    PluginType.RSP: f"mupen64plus-rsp-hle.{get_dll_ext()}",
    PluginType.GFX: f"mupen64plus-video-GLideN64.{get_dll_ext()}",
    PluginType.AUDIO: f"mupen64plus-audio-sdl.{get_dll_ext()}",
    PluginType.INPUT: f"mupen64plus-input-ext.{get_dll_ext()}",
}

SKIP_MESSAGES = [
    ('Core', 'init block'),  # debugger messages
    ('Core', 'block recompiled'),
    ('Core', 'NOTCOMPILED'),
    ('Audio', 'sdl_push_samples:'),  # audio skipping bytes
]

DEBUGFUNC = C.CFUNCTYPE(None, C.c_char_p, C.c_int, C.c_char_p)
STATEFUNC = C.CFUNCTYPE(None, C.c_char_p, C.c_int, C.c_int)

def version_split(ver: int) -> str:
    return f"{(ver >> 16) & 0xffff}.{(ver >> 8) & 0xff}.{ver & 0xff}"


class Core:
    """Core class to manage the Mupen64Plus emulator core and its plugins."""

    def __init__(self, log_level: int = LogLevel.INFO) -> None:
        self.plugins: list[int] = []
        self.plugin_map: dict[int, tuple[C.CDLL, str, str, str, int]] = {}
        self.inputext: Optional['InputExtension'] = None
        self.rom_header = M64pRomHeader()
        self.rom_settings = M64pRomSettings()
        self.core_path = str(CORE_LIB_PATH / f'libmupen64plus.{get_dll_ext()}')
        self.core_name = "Mupen64Plus Core"
        self.core_version = ""
        self.log_level = log_level
        self.logging_callback = DEBUGFUNC(self.handle_log_message)
        self.state_callback = STATEFUNC(self.handle_state_update)
        self.m64p = load(self.core_path)
        self.check_version()

    def check_version(self) -> None:
        """Checks core API version."""
        version = self.plugin_get_version(self.m64p, self.core_path)
        if version:
            plugin_type, plugin_version, plugin_api, plugin_name, plugin_cap = version
            if plugin_type != PluginType.CORE:
                raise RuntimeError(
                    f"library '{os.path.basename(self.core_path)}' is invalid, this is not the emulator core.")
            elif plugin_version < MINIMUM_CORE_VERSION:
                raise RuntimeError(
                    f"library '{os.path.basename(self.core_path)}' is incompatible, "
                    f"core version {version_split(plugin_version)} is below minimum supported {version_split(MINIMUM_CORE_VERSION)}.")
            elif plugin_api & 0xffff0000 != CORE_API_VERSION & 0xffff0000:
                raise RuntimeError(
                    f"library '{os.path.basename(self.core_path)}' is incompatible, "
                    f"core API major version {version_split(plugin_version)} doesn't match application: "
                    f"({version_split(CORE_API_VERSION)}).")
            else:
                api_versions = self.get_api_versions()
                if api_versions:
                    config_ver, debug_ver, vidext_ver = api_versions
                    if config_ver & 0xffff0000 != CONFIG_API_VERSION & 0xffff0000:
                        raise RuntimeError(
                            f"emulator core '{os.path.basename(self.core_path)}' is incompatible, "
                            f"config API major version {version_split(config_ver)} doesn't match application: "
                            f"({version_split(CONFIG_API_VERSION)})")
                    if vidext_ver & 0xffff0000 != VIDEXT_API_VERSION & 0xffff0000:
                        raise RuntimeError(
                            f"emulator core '{os.path.basename(self.core_path)}' is incompatible, "
                            f"vidext API major version {version_split(config_ver)} doesn't match application: "
                            f"({version_split(CONFIG_API_VERSION)})")

                self.core_name = plugin_name
                self.core_version = version_split(plugin_version)

                log.info(f"attached to library '{self.core_name}' version {self.core_version}")
                if plugin_cap & CoreFlags.DYNAREC:
                    log.info("includes support for Dynamic Recompiler.")
                if plugin_cap & CoreFlags.DEBUGGER:
                    log.info("includes support for MIPS r4300 Debugger.")
                if plugin_cap & CoreFlags.CORE_COMPARE:
                    log.info("includes support for r4300 Core Comparison.")

    def error_message(self, return_code: int) -> str:
        """Returns a human-readable error message for the given return code."""
        self.m64p.CoreErrorMessage.restype = C.c_char_p
        msg: str = self.m64p.CoreErrorMessage(return_code).decode()
        return msg

    def handle_log_message(self, context: bytes, level: int, message: bytes) -> None:
        """Callback to handle log messages from the core."""
        try:
            if any(context.decode() == ctx and message.decode().startswith(msg) for ctx, msg in SKIP_MESSAGES):
                return
            if self.log_level >= level:
                sys.stderr.write(f"{context.decode()}: {message.decode()}\n")
        except UnicodeDecodeError:
            sys.stderr.write(f"{context!r}: {message!r}\n")

    def handle_state_update(self, context: bytes, param: int, value: int) -> None:
        """Callback to handle state updates from the core."""
        pass

    def core_startup(self, vidext: Optional['VideoExtension'] = None, inputext: Optional['InputExtension'] = None) -> None:
        """Initializes libmupen64plus and attaches the video extension."""
        rval = self.m64p.CoreStartup(
            C.c_int(CORE_API_VERSION), C.c_char_p(str(CORE_LIB_PATH).encode()), C.c_char_p(str(CORE_LIB_PATH).encode()),
            C.c_char_p(b"Core"), self.logging_callback, C.c_char_p(b"State"), self.state_callback)
        if rval == ErrorType.SUCCESS:
            self.inputext = inputext
            if vidext:
                self.override_vidext(vidext)
        else:
            log.debug("core_startup()")
            log.warning(f"error starting '{self.core_name}' library")

    def core_shutdown(self) -> int:
        """Shuts down the core and unloads the library."""
        if self.m64p:
            self.m64p.CoreShutdown()
        return ErrorType.SUCCESS

    def plugin_get_version(self, handle: C.CDLL, path: str) -> Optional[tuple[int, int, int, str, int]]:
        """Returns the plugin version information."""
        try:
            type_ptr = C.pointer(C.c_int())
            ver_ptr = C.pointer(C.c_int())
            api_ptr = C.pointer(C.c_int())
            name_ptr = C.pointer(C.c_char_p())
            cap_ptr = C.pointer(C.c_int())
            rval = handle.PluginGetVersion(type_ptr, ver_ptr, api_ptr, name_ptr, cap_ptr)
        except AttributeError:
            unload_library(handle)
            log.warning(f"library '{os.path.basename(path)}' is invalid, no PluginGetVersion() function found.")
        except OSError as err:
            log.debug("plugin_get_version()")
            log.warning(str(err))
        else:
            if rval == ErrorType.SUCCESS:
                assert name_ptr.contents.value is not None
                return (
                    type_ptr.contents.value, ver_ptr.contents.value, api_ptr.contents.value,
                    name_ptr.contents.value.decode(), cap_ptr.contents.value)
            else:
                log.debug("plugin_get_version()")
                log.warning(self.error_message(rval))
        return None

    def get_api_versions(self) -> Optional[tuple[int, int, int]]:
        """Returns the API versions of the core."""
        config_ver_ptr = C.pointer(C.c_int())
        debug_ver_ptr = C.pointer(C.c_int())
        vidext_ver_ptr = C.pointer(C.c_int())
        rval = self.m64p.CoreGetAPIVersions(config_ver_ptr, debug_ver_ptr, vidext_ver_ptr, None)
        if rval == ErrorType.SUCCESS:
            return config_ver_ptr.contents.value, debug_ver_ptr.contents.value, vidext_ver_ptr.contents.value
        else:
            log.debug("get_api_versions()")
            log.warning(self.error_message(rval))
            return None

    def load_plugins(self) -> None:
        """Loads and initializes all of the plugins."""
        for plugin_type, plugin_path in PLUGIN_PATHS.items():
            self.plugin_load(str(CORE_LIB_PATH / plugin_path))
            self.plugin_startup(plugin_type)

    def plugin_load(self, plugin_path: str) -> None:
        """Loads the given plugin library and retrieves its version information."""
        try:
            plugin_handle = C.cdll.LoadLibrary(plugin_path)
            version = self.plugin_get_version(plugin_handle, plugin_path)
            if version:
                plugin_type, plugin_version, _, plugin_desc, plugin_cap = version
                self.plugin_map[plugin_type] = (plugin_handle, plugin_path, PLUGIN_NAMES[plugin_type], plugin_desc, plugin_version)
        except OSError as e:
            log.debug("plugin_load()")
            log.error(f"failed to load plugin {plugin_path}: {e}")

    def plugin_startup(self, plugin_type: int) -> None:
        """Initializes the specified plugin and sets up the logging callback."""
        plugin_handle, _, plugin_name, plugin_desc, _ = self.plugin_map[plugin_type]
        rval = plugin_handle.PluginStartup(C.c_void_p(self.m64p._handle), plugin_name.encode(), self.logging_callback)
        if rval != ErrorType.SUCCESS:
            log.debug("plugin_startup()")
            log.warning(self.error_message(rval))
            log.warning(f"{plugin_desc} failed to start.")
        elif plugin_type == PluginType.INPUT and self.inputext:
            self.override_inputext(self.inputext)

    def plugin_shutdown(self, plugin_type: int) -> None:
        """Shuts down the specified plugin and unloads it."""
        plugin_handle, _, _, plugin_desc, _ = self.plugin_map[plugin_type]
        rval = plugin_handle.PluginShutdown()
        if rval != ErrorType.SUCCESS:
            log.debug("plugin_shutdown()")
            log.warning(self.error_message(rval))
            log.warning(f"{plugin_desc} failed to stop.")

    def attach_plugins(self, plugin_types: list[int]) -> None:
        """Attaches the specified plugins to the core."""
        self.plugins = plugin_types
        for plugin_type in plugin_types:
            plugin_handle, _, plugin_name, plugin_desc, plugin_version = self.plugin_map[plugin_type]

            rval = self.m64p.CoreAttachPlugin(C.c_int(plugin_type), C.c_void_p(plugin_handle._handle))
            if rval != ErrorType.SUCCESS:
                log.debug("attach_plugins()")
                log.warning(self.error_message(rval))
                log.warning(f"core failed to attach {plugin_name} plugin.")
            else:
                log.info(f"using {plugin_name} plugin: '{plugin_desc}' v{version_split(plugin_version)}")

    def detach_plugins(self) -> None:
        """Detaches all plugins from the core."""
        for plugin_type in self.plugins:
            _, _, plugin_name, _, _ = self.plugin_map[plugin_type]

            rval = self.m64p.CoreDetachPlugin(plugin_type)
            if rval != ErrorType.SUCCESS:
                log.debug("detach_plugins()")
                log.warning(self.error_message(rval))
                log.warning(f"core failed to detach {plugin_name} plugin.")

    def rom_open(self, romfile: bytes) -> int:
        """Opens the specified ROM file and initializes the core."""
        rombuffer = C.c_buffer(romfile)
        rval: int = self.m64p.CoreDoCommand(CoreCommand.ROM_OPEN, C.c_int(len(romfile)), C.byref(rombuffer))
        if rval != ErrorType.SUCCESS:
            log.debug("rom_open()")
            log.warning(self.error_message(rval))
            log.error("core failed to open ROM file.")
        del rombuffer
        return rval

    def rom_close(self) -> int:
        """Closes the currently open ROM file."""
        rval: int = self.m64p.CoreDoCommand(CoreCommand.ROM_CLOSE)
        if rval != ErrorType.SUCCESS:
            log.debug("rom_close()")
            log.warning(self.error_message(rval))
            log.error("core failed to close ROM image file.")
        return rval

    def rom_get_header(self) -> int:
        """Retrieves the header data of the currently open ROM."""
        rval: int = self.m64p.CoreDoCommand(CoreCommand.ROM_GET_HEADER, C.c_int(C.sizeof(self.rom_header)), C.pointer(self.rom_header))
        if rval != ErrorType.SUCCESS:
            log.debug("rom_get_header()")
            log.warning("core failed to get ROM header.")
        return rval

    def rom_get_settings(self) -> int:
        """Retrieves the settings data of the currently open ROM."""
        rom_settings_size = C.c_int(C.sizeof(self.rom_settings))
        rval: int = self.m64p.CoreDoCommand(CoreCommand.ROM_GET_SETTINGS, rom_settings_size, C.pointer(self.rom_settings))
        if rval != ErrorType.SUCCESS:
            log.debug("rom_get_settings()")
            log.warning("core failed to get ROM settings.")
        return rval

    def execute(self) -> int:
        """Starts the emulator and begin executing the ROM image."""
        rval: int = self.m64p.CoreDoCommand(CoreCommand.EXECUTE, 0, None)
        if rval != ErrorType.SUCCESS:
            log.warning(self.error_message(rval))
        return rval

    def stop(self) -> int:
        """Stops the emulator, if it is currently running."""
        rval: int = self.m64p.CoreDoCommand(CoreCommand.STOP, 0, None)
        if rval != ErrorType.SUCCESS:
            log.debug("stop()")
            log.warning(self.error_message(rval))
        return rval

    def pause(self) -> int:
        """Pause the emulator if it is running."""
        rval: int = self.m64p.CoreDoCommand(CoreCommand.PAUSE, 0, None)
        if rval != ErrorType.SUCCESS:
            log.debug("pause()")
            log.warning(self.error_message(rval))
        return rval

    def resume(self) -> int:
        """Resumes execution of the emulator if it is paused."""
        rval: int = self.m64p.CoreDoCommand(CoreCommand.RESUME, 0, None)
        if rval != ErrorType.SUCCESS:
            log.debug("resume()")
            log.warning(self.error_message(rval))
        return rval

    def core_state_query(self, state: int) -> int:
        """Query the emulator core for the value of a state parameter."""
        state_ptr = C.pointer(C.c_int())
        rval: int = self.m64p.CoreDoCommand(CoreCommand.CORE_STATE_QUERY, C.c_int(state), state_ptr)
        if rval != ErrorType.SUCCESS:
            log.debug("core_state_query()")
            log.warning(self.error_message(rval))
        return state_ptr.contents.value

    def core_state_set(self, state: int, value: int) -> int:
        """Sets the value of a state parameter in the emulator core."""
        value_ptr = C.pointer(C.c_int(value))
        rval: int = self.m64p.CoreDoCommand(CoreCommand.CORE_STATE_SET, C.c_int(state), value_ptr)
        if rval != ErrorType.SUCCESS:
            log.debug("core_state_set()")
            log.warning(self.error_message(rval))
        return value_ptr.contents.value

    def core_mem_read(self, address: int, size: int) -> bytes:
        """Reads a block of memory from the emulator core."""
        dwords = []
        for _ in range(0, size, 4):
            rval: int = self.m64p.DebugMemRead32(C.c_uint(address))
            dwords.append(struct.pack(">I", rval))  # n64 is big endian
        return b''.join(dwords)[:size]

    def state_load(self, state_path: Optional[str] = None) -> int:
        """Loads a saved state file from the current slot."""
        path = C.c_char_p(state_path.encode()) if state_path else None
        rval: int = self.m64p.CoreDoCommand(CoreCommand.STATE_LOAD, C.c_int(1), path)
        if rval != ErrorType.SUCCESS:
            log.debug("state_load()")
            log.warning(self.error_message(rval))
        return rval

    def state_save(self, state_path: Optional[str] = None, state_type: int = 1) -> int:
        """Saves a state file to the current slot."""
        path = C.c_char_p(state_path.encode()) if state_path else None
        rval: int = self.m64p.CoreDoCommand(CoreCommand.STATE_SAVE, C.c_int(state_type), path)
        if rval != ErrorType.SUCCESS:
            log.debug("state_save()")
            log.warning(self.error_message(rval))
        return rval

    def state_set_slot(self, slot: int) -> int:
        """Sets the currently selected save slot index."""
        rval: int = self.m64p.CoreDoCommand(CoreCommand.STATE_SET_SLOT, C.c_int(slot))
        if rval != ErrorType.SUCCESS:
            log.debug("state_set_slot()")
            log.warning(self.error_message(rval))
        return rval

    def reset(self, soft: bool = False) -> int:
        """Reset the emulated machine."""
        rval: int = self.m64p.CoreDoCommand(CoreCommand.RESET, C.c_int(int(soft)))
        if rval != ErrorType.SUCCESS:
            log.debug("reset()")
            log.warning(self.error_message(rval))
        return rval

    def advance_frame(self) -> int:
        """Advance one frame. The emulator will run until the next frame, then pause."""
        rval: int = self.m64p.CoreDoCommand(CoreCommand.ADVANCE_FRAME, C.c_int(), C.c_int())
        if rval != ErrorType.SUCCESS:
            log.debug("advance_frame()")
            log.warning(self.error_message(rval))
        return rval

    def toggle_pause(self) -> None:
        """Toggles pause."""
        state = self.core_state_query(CoreState.EMU_STATE)
        if state == EmulationState.RUNNING:
            self.pause()
        elif state == EmulationState.PAUSED:
            self.resume()

    def toggle_mute(self) -> None:
        """Toggles mute."""
        if self.core_state_query(CoreState.AUDIO_MUTE):
            self.core_state_set(CoreState.AUDIO_MUTE, 0)
        else:
            self.core_state_set(CoreState.AUDIO_MUTE, 1)

    def toggle_speed_limit(self) -> None:
        """Toggles speed limiter."""
        if self.core_state_query(CoreState.SPEED_LIMITER):
            self.core_state_set(CoreState.SPEED_LIMITER, 0)
            log.info("Speed limiter disabled")
        else:
            self.core_state_set(CoreState.SPEED_LIMITER, 1)
            log.info("Speed limiter enabled")

    def get_rom_settings(self, crc1: int, crc2: int) -> Optional[M64pRomSettings]:
        """Searches through the data in the ini file for given crc hashes,
        if found, fills in the RomSettings structure with the data."""
        rom_settings = M64pRomSettings()
        rval: int = self.m64p.CoreGetRomSettings(C.byref(rom_settings), C.c_int(C.sizeof(rom_settings)), C.c_int(crc1), C.c_int(crc2))
        if rval != ErrorType.SUCCESS:
            return None
        return rom_settings

    def override_vidext(self, vidext: 'VideoExtension') -> int:
        """Overrides the core's internal SDL-based OpenGL functions."""
        rval: int = self.m64p.CoreOverrideVidExt(C.pointer(vidext.extension))
        if rval != ErrorType.SUCCESS:
            log.debug("override_vidext()")
            log.warning(self.error_message(rval))
        else:
            log.info("video extension enabled")
        return rval

    def override_inputext(self, inputext: 'InputExtension') -> int:
        """Overrides the input plugin's callbacks."""
        plugin_handle, *_ = self.plugin_map[PluginType.INPUT]
        rval: int = plugin_handle.OverrideInputExt(C.pointer(inputext.extension))
        if rval != ErrorType.SUCCESS:
            log.debug("override_inputext()")
            log.warning(self.error_message(rval))
        else:
            log.info("input extension enabled")
        return rval
