"""
thaw_common.format — thaw binary file format read/write primitives.

Must stay byte-compatible with crates/thaw-core (Rust implementation).
Files written here can be read by Rust and vice versa.
"""

import struct
from typing import Optional

# -- thaw file format constants, must match crates/thaw-core --

MAGIC = b"THAW"
VERSION = 1
HEADER_SIZE = 4096
REGION_ENTRY_SIZE = 32

# Region kind discriminants
KIND_WEIGHTS = 0
KIND_KV_LIVE_BLOCK = 1
KIND_METADATA = 2


def write_header(
    f,
    num_regions: int,
    engine_commit: Optional[bytes] = None,
):
    """Write a 4096-byte thaw header."""
    buf = bytearray(HEADER_SIZE)
    buf[0:4] = MAGIC
    struct.pack_into("<I", buf, 4, VERSION)
    struct.pack_into("<Q", buf, 8, num_regions)
    struct.pack_into("<Q", buf, 16, HEADER_SIZE)  # region_table_offset
    if engine_commit is not None:
        if len(engine_commit) != 40:
            raise ValueError(f"engine_commit must be 40 bytes, got {len(engine_commit)}")
        buf[24:64] = engine_commit
    f.write(buf)


def write_region_entry(
    f,
    kind: int,
    logical_id: int,
    size: int,
    file_offset: int,
):
    """Write a 32-byte region table entry."""
    buf = bytearray(REGION_ENTRY_SIZE)
    struct.pack_into("<I", buf, 0, kind)
    struct.pack_into("<I", buf, 4, logical_id)
    struct.pack_into("<Q", buf, 8, size)
    struct.pack_into("<Q", buf, 16, file_offset)
    f.write(buf)


def read_header(f):
    """Read and validate a thaw header. Returns (num_regions, engine_commit)."""
    buf = f.read(HEADER_SIZE)
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"file too short for header: {len(buf)} bytes")
    if buf[0:4] != MAGIC:
        raise ValueError(f"bad magic: {buf[0:4]!r}")
    version = struct.unpack_from("<I", buf, 4)[0]
    if version != VERSION:
        raise ValueError(f"unsupported version {version}, expected {VERSION}")
    num_regions = struct.unpack_from("<Q", buf, 8)[0]
    engine_commit = bytes(buf[24:64])
    return num_regions, engine_commit


def read_region_entry(f):
    """Read a 32-byte region entry. Returns (kind, logical_id, size, file_offset)."""
    buf = f.read(REGION_ENTRY_SIZE)
    if len(buf) < REGION_ENTRY_SIZE:
        raise ValueError("truncated region entry")
    kind = struct.unpack_from("<I", buf, 0)[0]
    logical_id = struct.unpack_from("<I", buf, 4)[0]
    size = struct.unpack_from("<Q", buf, 8)[0]
    file_offset = struct.unpack_from("<Q", buf, 16)[0]
    return kind, logical_id, size, file_offset
