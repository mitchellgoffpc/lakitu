import sys

if sys.platform.startswith("linux"):
    DLL_EXT = ".so"
    DLL_FILTER = ".so.2"
    DEFAULT_DYNLIB = "libmupen64plus.so.2"
    SEARCH_DIRS = [
        "/usr/local/lib/mupen64plus",
        "/usr/lib64/mupen64plus",
        "/usr/lib/mupen64plus",
        "/usr/games/lib64/mupen64plus",
        "/usr/games/lib/mupen64plus",
        "/usr/lib/x86_64-linux-gnu/mupen64plus",
        "/usr/lib/i386-linux-gnu/mupen64plus",
        "/app/lib/mupen64plus",
        "."
    ]
elif "bsd" in sys.platform:
    DLL_EXT = ".so"
    DLL_FILTER = ""
    DEFAULT_DYNLIB = "libmupen64plus.so"
    SEARCH_DIRS = [
        "/usr/local/lib/mupen64plus",
        "."
    ]
elif sys.platform == "darwin":
    DLL_EXT = ".dylib"
    DLL_FILTER = ".dylib"
    DEFAULT_DYNLIB = "libmupen64plus.dylib"
    SEARCH_DIRS = [
        "/usr/local/lib/mupen64plus",
        "/usr/lib/mupen64plus",
        "."
    ]
elif sys.platform == "win32":
    DLL_EXT = ".dll"
    DLL_FILTER = ".dll"
    DEFAULT_DYNLIB = "mupen64plus.dll"
    SEARCH_DIRS = ["."]
