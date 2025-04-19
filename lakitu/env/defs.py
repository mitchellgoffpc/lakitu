import ctypes as C

class LogLevel:
    ERROR = 1
    WARNING = 2
    INFO = 3
    STATUS = 4
    VERBOSE = 5

class ErrorType:
    SUCCESS = 0
    NOT_INIT = 1
    ALREADY_INIT = 2
    INCOMPATIBLE = 3
    INPUT_ASSERT = 4
    INPUT_INVALID = 5
    INPUT_NOT_FOUND = 6
    NO_MEMORY = 7
    FILES = 8
    INTERNAL = 9
    INVALID_STATE = 10
    PLUGIN_FAIL = 11
    SYSTEM_FAIL = 12
    UNSUPPORTED = 13
    WRONG_TYPE = 14

class CoreFlags:
    DYNAREC = 1
    DEBUGGER = 2
    CORE_COMPARE = 4

class PluginType:
    RSP = 1
    GFX = 2
    AUDIO = 3
    INPUT = 4
    CORE = 5

class EmulationState:
    STOPPED = 1
    RUNNING = 2
    PAUSED = 3

class CoreState:
    EMU_STATE = 1
    VIDEO_MODE = 2
    SAVESTATE_SLOT = 3
    SPEED_FACTOR = 4
    SPEED_LIMITER = 5
    VIDEO_SIZE = 6
    AUDIO_VOLUME = 7
    AUDIO_MUTE = 8
    INPUT_GAMESHARK = 9
    STATE_LOADCOMPLETE = 10
    STATE_SAVECOMPLETE = 11
    SCREENSHOT_CAPTURED = 12

class CoreCommand:
    NOP = 0
    ROM_OPEN = 1
    ROM_CLOSE = 2
    ROM_GET_HEADER = 3
    ROM_GET_SETTINGS = 4
    EXECUTE = 5
    STOP = 6
    PAUSE = 7
    RESUME = 8
    CORE_STATE_QUERY = 9
    STATE_LOAD = 10
    STATE_SAVE = 11
    STATE_SET_SLOT = 12
    SEND_SDL_KEYDOWN = 13
    SEND_SDL_KEYUP = 14
    SET_FRAME_CALLBACK = 15
    TAKE_NEXT_SCREENSHOT = 16
    CORE_STATE_SET = 17
    READ_SCREEN = 18
    RESET = 19
    ADVANCE_FRAME = 20
    SET_MEDIA_LOADER = 21
    NETPLAY_INIT = 22
    NETPLAY_CONTROL_PLAYER = 23
    NETPLAY_GET_VERSION = 24
    NETPLAY_CLOSE = 25
    PIF_OPEN = 26
    ROM_SET_SETTINGS = 27
    DISK_OPEN = 28
    DISK_CLOSE = 29

class GLAttribute:
    DOUBLEBUFFER = 1
    BUFFER_SIZE = 2
    DEPTH_SIZE = 3
    RED_SIZE = 4
    GREEN_SIZE = 5
    BLUE_SIZE = 6
    ALPHA_SIZE = 7
    SWAP_CONTROL = 8
    MULTISAMPLEBUFFERS = 9
    MULTISAMPLESAMPLES = 10
    CONTEXT_MAJOR_VERSION = 11
    CONTEXT_MINOR_VERSION = 12
    CONTEXT_PROFILE_MASK = 13

class GLProfile:
    CORE = 0
    COMPATIBILITY = 1
    ES = 2

class RenderMode:
    OPENGL = 0
    VULKAN = 1

class ControllerPluginType:
    NONE = 1
    MEMPAK = 2


#  Video Extension Types

m64p_error = C.c_int
m64p_gl_attr = C.c_int

class M64pRomHeader(C.Structure):
    _fields_ = [
        ('init_PI_BSB_DOM1_LAT_REG', C.c_uint8),
        ('init_PI_BSB_DOM1_PGS_REG', C.c_uint8),
        ('init_PI_BSB_DOM1_PWD_REG', C.c_uint8),
        ('init_PI_BSB_DOM1_PGS_REG2', C.c_uint8),
        ('ClockRate', C.c_uint32),
        ('PC', C.c_uint32),
        ('Release', C.c_uint32),
        ('CRC1', C.c_uint32),
        ('CRC2', C.c_uint32),
        ('Unknown', C.c_uint32 * 2),
        ('Name', C.c_char * 20),
        ('unknown', C.c_uint32),
        ('Manufacturer_ID', C.c_uint32),
        ('Cartridge_ID', C.c_uint16),
        ('Country_code', C.c_uint8),
        ('Version', C.c_uint8)
    ]

class M64pRomSettings(C.Structure):
    _fields_ = [
        ('goodname', C.c_char * 256),
        ('MD5', C.c_char * 33),
        ('savetype', C.c_ubyte),
        ('status', C.c_ubyte),
        ('players', C.c_ubyte),
        ('rumble', C.c_ubyte),
        ('transferpak', C.c_ubyte),
        ('mempak', C.c_ubyte),
        ('biopak', C.c_ubyte),
        ('disableextramem', C.c_ubyte),
        ('countperop', C.c_uint),
        ('sidmaduration', C.c_uint),
        ('aidmamodifier', C.c_uint)
    ]

class M64p2dSize(C.Structure):
    _fields_ = [
        ('uiWidth', C.c_uint),
        ('uiHeight', C.c_uint)
    ]


class VidExtFuncs:
    Init = C.CFUNCTYPE(m64p_error)
    Quit = C.CFUNCTYPE(m64p_error)
    ListModes = C.CFUNCTYPE(m64p_error, C.POINTER(M64p2dSize), C.POINTER(C.c_int))
    ListRates = C.CFUNCTYPE(m64p_error, M64p2dSize, C.POINTER(C.c_int), C.POINTER(C.c_int))
    SetMode = C.CFUNCTYPE(m64p_error, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int)
    SetModeWithRate = C.CFUNCTYPE(m64p_error, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int)
    GLGetProc = C.CFUNCTYPE(C.c_void_p, C.c_char_p)
    GLSetAttr = C.CFUNCTYPE(m64p_error, m64p_gl_attr, C.c_int)
    GLGetAttr = C.CFUNCTYPE(m64p_error, m64p_gl_attr, C.POINTER(C.c_int))
    GLSwapBuf = C.CFUNCTYPE(m64p_error)
    SetCaption = C.CFUNCTYPE(m64p_error, C.c_char_p)
    ToggleFS = C.CFUNCTYPE(m64p_error)
    ResizeWindow = C.CFUNCTYPE(m64p_error, C.c_int, C.c_int)
    GLGetDefaultFramebuffer = C.CFUNCTYPE(C.c_uint32)
    InitWithRenderMode = C.CFUNCTYPE(m64p_error, C.c_int)
    VKGetSurface = C.CFUNCTYPE(m64p_error, C.POINTER(C.c_void_p), C.c_void_p)
    VKGetInstanceExtensions = C.CFUNCTYPE(m64p_error, C.POINTER(C.c_char_p), C.POINTER(C.c_uint32))

class M64pVideoExtension(C.Structure):
    _fields_ = [
        ('Functions', C.c_uint),
        ('VidExtFuncInit', VidExtFuncs.Init),
        ('VidExtFuncQuit', VidExtFuncs.Quit),
        ('VidExtFuncListModes', VidExtFuncs.ListModes),
        ('VidExtFuncListRates', VidExtFuncs.ListRates),
        ('VidExtFuncSetMode', VidExtFuncs.SetMode),
        ('VidExtFuncSetModeWithRate', VidExtFuncs.SetModeWithRate),
        ('VidExtFuncGLGetProc', VidExtFuncs.GLGetProc),
        ('VidExtFuncGLSetAttr', VidExtFuncs.GLSetAttr),
        ('VidExtFuncGLGetAttr', VidExtFuncs.GLGetAttr),
        ('VidExtFuncGLSwapBuf', VidExtFuncs.GLSwapBuf),
        ('VidExtFuncSetCaption', VidExtFuncs.SetCaption),
        ('VidExtFuncToggleFS', VidExtFuncs.ToggleFS),
        ('VidExtFuncResizeWindow', VidExtFuncs.ResizeWindow),
        ('VidExtFuncGLGetDefaultFramebuffer', VidExtFuncs.GLGetDefaultFramebuffer),
        ('VidExtFuncInitWithRenderMode', VidExtFuncs.InitWithRenderMode),
        ('VidExtFuncVKGetSurface', VidExtFuncs.VKGetSurface),
        ('VidExtFuncVKGetInstanceExtensions', VidExtFuncs.VKGetInstanceExtensions)
    ]


# Input Extension Types

class M64pButtons(C.Structure):
    _pack_ = 1
    _fields_ = [
        ('R_DPAD', C.c_uint16, 1),
        ('L_DPAD', C.c_uint16, 1),
        ('D_DPAD', C.c_uint16, 1),
        ('U_DPAD', C.c_uint16, 1),
        ('START_BUTTON', C.c_uint16, 1),
        ('Z_TRIG', C.c_uint16, 1),
        ('B_BUTTON', C.c_uint16, 1),
        ('A_BUTTON', C.c_uint16, 1),
        ('R_CBUTTON', C.c_uint16, 1),
        ('L_CBUTTON', C.c_uint16, 1),
        ('D_CBUTTON', C.c_uint16, 1),
        ('U_CBUTTON', C.c_uint16, 1),
        ('R_TRIG', C.c_uint16, 1),
        ('L_TRIG', C.c_uint16, 1),
        ('Reserved1', C.c_uint16, 1),
        ('Reserved2', C.c_uint16, 1),
        ('X_AXIS', C.c_int8),
        ('Y_AXIS', C.c_int8),
    ]

    @classmethod
    def get_joystick_fields(cls):
        return ['X_AXIS', 'Y_AXIS']

    @classmethod
    def get_button_fields(cls):
        return [field for field, *_ in cls._fields_ if field not in cls.get_joystick_fields() and not field.startswith('Reserved')]

class M64pControl(C.Structure):
    _fields_ = [
        ('Present', C.c_int),
        ('RawData', C.c_int),
        ('Plugin', C.c_int),
        ('Type', C.c_int)
    ]

class M64pControlInfo(C.Structure):
    _fields_ = [
        ('Controls', C.POINTER(M64pControl))
    ]

class InputExtFuncs:
    GetKeys = C.CFUNCTYPE(None, C.c_int, C.POINTER(M64pButtons))
    InitiateControllers = C.CFUNCTYPE(None, M64pControlInfo)
    RenderCallback = C.CFUNCTYPE(None)

class M64pInputExtension(C.Structure):
    _fields_ = [
        ('InputExtFuncGetKeys', InputExtFuncs.GetKeys),
        ('InputExtFuncInitiateControllers', InputExtFuncs.InitiateControllers),
        ('InputExtFuncRenderCallback', InputExtFuncs.RenderCallback),
    ]
