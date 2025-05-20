import hashlib
import numpy as np
import struct
from pathlib import Path

from lakitu.env.core import Core

def m64_get_level(core: Core) -> int:
    """Get the active level"""
    mem = core.core_mem_read(0x8032DDF8, 2)
    result: int = struct.unpack('>H', mem)[0]  # n64 is big endian
    return result

def m64_get_level_act(core: Core) -> int:
    """Get the active level entry"""
    mem = core.core_mem_read(0x80331620, 1)
    result: int = struct.unpack('>B', mem)[0]
    return result

def m64_get_health(core: Core) -> int:
    """Get the current healthbar"""
    mem = core.core_mem_read(0x8033B21E, 2)
    result: int = struct.unpack('>H', mem)[0]
    return result

def m64_get_num_lives(core: Core) -> int:
    """Get the number of lives remaining"""
    mem = core.core_mem_read(0x8033B21D, 1)
    result: int = struct.unpack('>B', mem)[0]
    return result

def m64_get_num_stars(core: Core) -> int:
    """Get the number of collected stars"""
    mem = core.core_mem_read(0x8033B21A, 2)
    result: int = struct.unpack('>H', mem)[0]
    return result

def m64_get_num_coins(core: Core) -> int:
    """Get the number of collected coins"""
    mem = core.core_mem_read(0x8033B218, 2)
    result: int = struct.unpack('>H', mem)[0]
    return result

def m64_get_position(core: Core) -> list[float]:
    """Get the current position of the player"""
    mem = core.core_mem_read(0x8033B1AC, 12)
    result: list[float] = list(struct.unpack('>fff', mem))
    return result

M64_INFO_HOOKS = {
    "level": m64_get_level,
    "level_act": m64_get_level_act,
    'num_stars': m64_get_num_stars,
    'num_lives': m64_get_num_lives,
    'num_coins': m64_get_num_coins,
    'health': m64_get_health,
    'position': m64_get_position,
}

M64_INFO_FIELDS = [
    ("level", np.dtype(np.uint8), ()),
    ("level_act", np.dtype(np.uint8), ()),
    ("health", np.dtype(np.uint16), ()),
    ("num_lives", np.dtype(np.uint8), ()),
    ("num_stars", np.dtype(np.uint16), ()),
    ("num_coins", np.dtype(np.uint16), ()),
    ("position", np.dtype(np.float32), (3,)),
]

M64_INFO_ACTIVE_KEYS = ['level', 'level_act', 'num_stars']
M64_INFO_HOOKS = {k: v for k, v in M64_INFO_HOOKS.items() if k in M64_INFO_ACTIVE_KEYS}
M64_INFO_FIELDS = [(k, d, s) for k, d, s in M64_INFO_FIELDS if k in M64_INFO_ACTIVE_KEYS]

M64_OBJECTIVES = {
    'courtyard.m64p': lambda initial_state, current_state: current_state['level'] == 6,  # Enter the castle
    'castle_entry.m64p': lambda initial_state, current_state: current_state['level'] == 9,  # Enter bobomb battlefield
    'princess_slide.m64p': lambda initial_state, current_state: current_state['num_stars'] == initial_state['num_stars'] + 1,
}

SAVESTATE_DIR = Path(__file__).parents[2] / "data" / "savestates"
SAVESTATE_HASHES = {}
for savestate_file in SAVESTATE_DIR.glob("*.m64p"):
    savestate_data = savestate_file.read_bytes()
    savestate_hash = hashlib.sha256(savestate_data).hexdigest()
    SAVESTATE_HASHES[savestate_hash] = savestate_file.name

def get_savestate_name(savestate_file: Path) -> str:
    """Get the savestate name from the file path"""
    savestate_data = savestate_file.read_bytes()
    savestate_hash = hashlib.sha256(savestate_data).hexdigest()
    if savestate_hash not in SAVESTATE_HASHES:
        raise ValueError(f"Unknown savestate hash: {savestate_hash}")
    return SAVESTATE_HASHES[savestate_hash]
