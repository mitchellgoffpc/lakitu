# Mario 64 Helpers

import struct
from lakitu.env.core import Core

def m64_get_level(core: Core) -> int:
    """Get the current level from memory"""
    mem = core.core_mem_read(0x8032DDF8, 2)
    result: int = struct.unpack('>H', mem)[0]  # n64 is big endian
    return result
