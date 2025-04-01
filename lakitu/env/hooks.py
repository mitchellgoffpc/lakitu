import glfw
import queue
import ctypes
import numpy as np
import logging as log

from lakitu.env.defs import *

class VideoExtension:
    """Mupen64Plus video extension"""

    def __init__(self, input_plugin, offscreen=False):
        """Constructor."""
        self.window = None
        self.input_plugin = input_plugin
        self.offscreen = offscreen
        self.render_mode = M64P_RENDER_OPENGL

        # OpenGL attributes
        self.gl_major_version = 3
        self.gl_minor_version = 3
        self.gl_profile = None
        self.gl_depth_size = 24
        self.gl_swap_interval = 0
        self.gl_doublebuffer = True
        self.gl_red_size = 8
        self.gl_green_size = 8
        self.gl_blue_size = 8
        self.gl_alpha_size = 8
        self.gl_multisample_samples = 0
        self.gl_context_profile = M64P_GL_CONTEXT_PROFILE_CORE

        # Video callbacks
        self.callbacks = M64pVideoExtensionFunctions()
        self.callbacks.Functions = 17
        self.callbacks.VidExtFuncInit = FuncInit(self.init)
        self.callbacks.VidExtFuncQuit = FuncQuit(self.quit)
        self.callbacks.VidExtFuncListModes = FuncListModes(self.list_modes)
        self.callbacks.VidExtFuncListRates = FuncListRates(self.list_rates)
        self.callbacks.VidExtFuncSetMode = FuncSetMode(self.set_mode)
        self.callbacks.VidExtFuncSetModeWithRate = FuncSetModeWithRate(self.set_mode_with_rate)
        self.callbacks.VidExtFuncGLGetProc = FuncGLGetProc(self.gl_get_proc)
        self.callbacks.VidExtFuncGLSetAttr = FuncGLSetAttr(self.gl_set_attr)
        self.callbacks.VidExtFuncGLGetAttr = FuncGLGetAttr(self.gl_get_attr)
        self.callbacks.VidExtFuncGLSwapBuf = FuncGLSwapBuf(self.gl_swap_buf)
        self.callbacks.VidExtFuncSetCaption = FuncSetCaption(self.set_caption)
        self.callbacks.VidExtFuncToggleFS = FuncToggleFS(self.toggle_fs)
        self.callbacks.VidExtFuncResizeWindow = FuncResizeWindow(self.resize_window)
        self.callbacks.VidExtFuncGLGetDefaultFramebuffer = FuncGLGetDefaultFramebuffer(self.gl_get_default_framebuffer)
        self.callbacks.VidExtFuncInitWithRenderMode = FuncInitWithRenderMode(self.init_with_render_mode)
        self.callbacks.VidExtFuncVKGetSurface = FuncVKGetSurface(self.vk_get_surface)
        self.callbacks.VidExtFuncVKGetInstanceExtensions = FuncVKGetInstanceExtensions(self.vk_get_instance_extensions)

    def init(self):
        if self.render_mode == M64P_RENDER_OPENGL:
            if not glfw.init():
                log.error("Failed to initialize GLFW")
                return M64ERR_SYSTEM_FAIL

            # Configure GLFW
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, self.gl_major_version)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, self.gl_minor_version)

            if self.gl_context_profile == M64P_GL_CONTEXT_PROFILE_CORE:
                glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            elif self.gl_context_profile == M64P_GL_CONTEXT_PROFILE_COMPATIBILITY:
                glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
            elif self.gl_context_profile == M64P_GL_CONTEXT_PROFILE_ES:
                glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)

            glfw.window_hint(glfw.DEPTH_BITS, self.gl_depth_size)
            glfw.window_hint(glfw.RED_BITS, self.gl_red_size)
            glfw.window_hint(glfw.GREEN_BITS, self.gl_green_size)
            glfw.window_hint(glfw.BLUE_BITS, self.gl_blue_size)
            glfw.window_hint(glfw.ALPHA_BITS, self.gl_alpha_size)
            glfw.window_hint(glfw.SAMPLES, self.gl_multisample_samples)
            glfw.window_hint(glfw.DOUBLEBUFFER, self.gl_doublebuffer)
            glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, glfw.FALSE)

            # Create a window
            self.window = glfw.create_window(640, 480, "M64Py", None, None)
            if not self.window:
                glfw.terminate()
                log.error("Failed to create GLFW window")
                return M64ERR_SYSTEM_FAIL

            # Make the window's context current
            glfw.make_context_current(self.window)
            glfw.swap_interval(self.gl_swap_interval)
            self.input_plugin.init(self.window)

        elif self.render_mode == M64P_RENDER_VULKAN:
            pass

        return M64ERR_SUCCESS

    def init_with_render_mode(self, mode):
        self.render_mode = mode
        return self.init()

    def quit(self):
        if self.render_mode == M64P_RENDER_OPENGL:
            if self.window:
                glfw.destroy_window(self.window)
                self.window = None
                glfw.terminate()

        return M64ERR_SUCCESS

    def list_modes(self, size_array, num_sizes):
        """Enumerate the available resolutions for fullscreen video modes."""
        num_sizes.contents.value = 0
        return M64ERR_SUCCESS

    def list_rates(self, size_array, num_rates, rates):
        """Enumerate the available rates for fullscreen video modes."""
        num_rates.contents.value = 0
        return M64ERR_SUCCESS

    def set_mode(self, width, height, bits, mode, flags):
        if self.render_mode == M64P_RENDER_OPENGL:
            glfw.set_window_size(self.window, width, height)
            if not self.offscreen:
                glfw.show_window(self.window)
            glfw.make_context_current(self.window)

        return M64ERR_SUCCESS

    def set_mode_with_rate(self, width, height, rate, bits, mode, flags):
        return self.set_mode(width, height, bits, mode, flags)

    def set_caption(self, title):
        """Set the caption text of the emulator rendering window. """
        title_str = "M64Py :: %s" % title.decode()
        if self.window:
            glfw.set_window_title(self.window, title_str)
        return M64ERR_SUCCESS

    def toggle_fs(self):
        """Toggle between fullscreen and windowed rendering modes. """
        return M64ERR_SUCCESS

    def gl_get_proc(self, proc):
        """Get a pointer to an OpenGL extension function."""
        if self.render_mode != M64P_RENDER_OPENGL:
            return 0
        if not self.window:
            return 0

        proc_str = proc.decode()
        addr = glfw.get_proc_address(proc_str)
        if addr:
            return addr
        else:
            log.warn("VidExtFuncGLGetProc: '%s'" % proc_str)
            return 0

    def gl_set_attr(self, attr, value):
        """Set OpenGL attributes."""
        if self.render_mode != M64P_RENDER_OPENGL:
            return M64ERR_INVALID_STATE

        # Store the attribute for later use when creating the window
        if attr == M64P_GL_DOUBLEBUFFER:
            self.gl_doublebuffer = bool(value)
        elif attr == M64P_GL_BUFFER_SIZE:
            val = int(value/4)
            self.gl_red_size = val
            self.gl_green_size = val
            self.gl_blue_size = val
            self.gl_alpha_size = val
        elif attr == M64P_GL_DEPTH_SIZE:
            self.gl_depth_size = value
        elif attr == M64P_GL_RED_SIZE:
            self.gl_red_size = value
        elif attr == M64P_GL_GREEN_SIZE:
            self.gl_green_size = value
        elif attr == M64P_GL_BLUE_SIZE:
            self.gl_blue_size = value
        elif attr == M64P_GL_ALPHA_SIZE:
            self.gl_alpha_size = value
        elif attr == M64P_GL_SWAP_CONTROL:
            self.gl_swap_interval = value
            if self.window:
                glfw.swap_interval(value)
        elif attr == M64P_GL_MULTISAMPLESAMPLES:
            self.gl_multisample_samples = value
        elif attr == M64P_GL_CONTEXT_MAJOR_VERSION:
            self.gl_major_version = value
        elif attr == M64P_GL_CONTEXT_MINOR_VERSION:
            self.gl_minor_version = value
        elif attr == M64P_GL_CONTEXT_PROFILE_MASK:
            self.gl_context_profile = value

        return M64ERR_SUCCESS

    def gl_get_attr(self, attr, value):
        """Get OpenGL attributes."""
        if self.render_mode != M64P_RENDER_OPENGL:
            return M64ERR_INVALID_STATE

        if attr == M64P_GL_DOUBLEBUFFER:
            new_value = 1 if self.gl_doublebuffer else 0
        elif attr == M64P_GL_BUFFER_SIZE:
            new_value = self.gl_red_size + self.gl_green_size + self.gl_blue_size + self.gl_alpha_size
        elif attr == M64P_GL_DEPTH_SIZE:
            new_value = self.gl_depth_size
        elif attr == M64P_GL_RED_SIZE:
            new_value = self.gl_red_size
        elif attr == M64P_GL_GREEN_SIZE:
            new_value = self.gl_green_size
        elif attr == M64P_GL_BLUE_SIZE:
            new_value = self.gl_blue_size
        elif attr == M64P_GL_ALPHA_SIZE:
            new_value = self.gl_alpha_size
        elif attr == M64P_GL_SWAP_CONTROL:
            new_value = self.gl_swap_interval
        elif attr == M64P_GL_MULTISAMPLESAMPLES:
            new_value = self.gl_multisample_samples
        elif attr == M64P_GL_CONTEXT_MAJOR_VERSION:
            new_value = self.gl_major_version
        elif attr == M64P_GL_CONTEXT_MINOR_VERSION:
            new_value = self.gl_minor_version
        elif attr == M64P_GL_CONTEXT_PROFILE_MASK:
            new_value = self.gl_context_profile
        else:
            return M64ERR_INPUT_INVALID

        value.contents.value = new_value
        if new_value != value.contents.value:  # Hmm very suspicious, not sure why this would ever happen...
            return M64ERR_SYSTEM_FAIL
        return M64ERR_SUCCESS

    def gl_swap_buf(self):
        """Swap the front/back buffers after rendering an output video frame."""
        if self.render_mode != M64P_RENDER_OPENGL:
            return M64ERR_INVALID_STATE

        if self.window:
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        return M64ERR_SUCCESS

    def resize_window(self, width, height):
        """Called when the video plugin has resized its OpenGL output viewport in response to a ResizeVideoOutput() call"""
        if self.window:
            glfw.set_window_size(self.window, width, height)
        return M64ERR_SUCCESS

    def gl_get_default_framebuffer(self):
        if self.render_mode != M64P_RENDER_OPENGL:
            return 0
        return 0  # GLFW uses the default framebuffer (0)

    def vk_get_surface(self, a, b):
        if self.render_mode != M64P_RENDER_VULKAN:
            return M64ERR_INVALID_STATE
        return M64ERR_SUCCESS

    def vk_get_instance_extensions(self, a, b):
        if self.render_mode != M64P_RENDER_VULKAN:
            return M64ERR_INVALID_STATE
        return M64ERR_SUCCESS


class InputPlugin:
    def __init__(self, core, data_queue=None):
        self.window = None
        self.core = core
        self.data_queue = data_queue
        self.controller_states = {}

        # Input callbacks
        self.gfx_funcs = M64pGfxPluginFunctions.in_dll(core.m64p, 'gfx')
        self.callbacks = M64pInputPluginFunctions()
        self.callbacks.getKeys = FuncGetKeys(self.get_keys)
        self.callbacks.initiateControllers = FuncInitiateControllers(self.initiate_controllers)
        self.callbacks.renderCallback = FuncRenderCallback(self.render_callback)

    def init(self, window):
        self.window = window

    def initiate_controllers(self, control_info):
        control_info.Controls[0].Present = 1
        control_info.Controls[0].Plugin = ControlPlugin.PLUGIN_MEMPAK
        control_info.Controls[1].Present = 0
        control_info.Controls[1].Plugin = ControlPlugin.PLUGIN_NONE
        control_info.Controls[2].Present = 0
        control_info.Controls[2].Plugin = ControlPlugin.PLUGIN_NONE
        control_info.Controls[3].Present = 0
        control_info.Controls[3].Plugin = ControlPlugin.PLUGIN_NONE

    def get_keys(self, controller, buttons):
        raise NotImplementedError()

    def render_callback(self):
        if not self.window:
            return
        if glfw.window_should_close(self.window):
            self.core.stop()
        if self.data_queue:
            # NOTE: We can also use glReadPixels to read the framebuffer, but using the official API removes the dependency on PyOpenGL
            width, height = glfw.get_window_size(self.window)
            buffer = np.zeros((height, width, 3), dtype=np.uint8)
            self.gfx_funcs.readScreen2(buffer.ctypes.data_as(C.POINTER(C.c_uint8)), C.byref(C.c_int(width)), C.byref(C.c_int(height)), 0)
            self.data_queue.put((buffer, self.controller_states))
