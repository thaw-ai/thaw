"""
Tests for thaw_common — file format primitives and shared utilities.

Runs on Mac without GPU. Tests cover:
  - Header write/read roundtrip
  - Region entry write/read roundtrip
  - Bad magic detection
  - Unsupported version detection
  - Truncated file handling
  - engine_commit field encoding
  - rank_snapshot_path for TP file naming
  - Constants match expected values
"""

import io
import struct

import pytest

from thaw_common.format import (
    MAGIC,
    VERSION,
    HEADER_SIZE,
    REGION_ENTRY_SIZE,
    KIND_WEIGHTS,
    KIND_KV_LIVE_BLOCK,
    KIND_METADATA,
    write_header,
    write_region_entry,
    read_header,
    read_region_entry,
)
from thaw_common.util import rank_snapshot_path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_magic(self):
        assert MAGIC == b"THAW"

    def test_version(self):
        assert VERSION == 1

    def test_header_size(self):
        assert HEADER_SIZE == 4096

    def test_region_entry_size(self):
        assert REGION_ENTRY_SIZE == 32

    def test_kind_discriminants(self):
        assert KIND_WEIGHTS == 0
        assert KIND_KV_LIVE_BLOCK == 1
        assert KIND_METADATA == 2


# ---------------------------------------------------------------------------
# Header read/write
# ---------------------------------------------------------------------------


class TestHeader:
    def test_roundtrip(self):
        """Write a header, read it back, verify fields match."""
        buf = io.BytesIO()
        write_header(buf, num_regions=42)
        assert buf.tell() == HEADER_SIZE

        buf.seek(0)
        num_regions, engine_commit = read_header(buf)
        assert num_regions == 42
        assert engine_commit == b"\x00" * 40

    def test_roundtrip_with_commit(self):
        """Header with engine_commit preserves the 40-byte hash."""
        commit = b"a" * 40
        buf = io.BytesIO()
        write_header(buf, num_regions=7, engine_commit=commit)

        buf.seek(0)
        num_regions, engine_commit = read_header(buf)
        assert num_regions == 7
        assert engine_commit == commit

    def test_commit_wrong_length(self):
        """engine_commit must be exactly 40 bytes."""
        buf = io.BytesIO()
        with pytest.raises(ValueError, match="40 bytes"):
            write_header(buf, num_regions=1, engine_commit=b"short")

    def test_bad_magic(self):
        """Reading a file with wrong magic raises ValueError."""
        buf = io.BytesIO(b"NOPE" + b"\x00" * (HEADER_SIZE - 4))
        with pytest.raises(ValueError, match="bad magic"):
            read_header(buf)

    def test_bad_version(self):
        """Reading a file with unsupported version raises ValueError."""
        buf = io.BytesIO()
        write_header(buf, num_regions=1)
        # Patch version to 99
        buf.seek(4)
        buf.write(struct.pack("<I", 99))
        buf.seek(0)
        with pytest.raises(ValueError, match="unsupported version"):
            read_header(buf)

    def test_truncated_header(self):
        """File shorter than HEADER_SIZE raises ValueError."""
        buf = io.BytesIO(b"THAW" + b"\x00" * 10)
        with pytest.raises(ValueError, match="too short"):
            read_header(buf)

    def test_zero_regions(self):
        """Header with zero regions is valid (empty snapshot)."""
        buf = io.BytesIO()
        write_header(buf, num_regions=0)
        buf.seek(0)
        num_regions, _ = read_header(buf)
        assert num_regions == 0

    def test_large_region_count(self):
        """Header can encode large region counts."""
        buf = io.BytesIO()
        write_header(buf, num_regions=100_000)
        buf.seek(0)
        num_regions, _ = read_header(buf)
        assert num_regions == 100_000


# ---------------------------------------------------------------------------
# Region entry read/write
# ---------------------------------------------------------------------------


class TestRegionEntry:
    def test_roundtrip(self):
        """Write a region entry, read it back, verify fields match."""
        buf = io.BytesIO()
        write_region_entry(buf, kind=KIND_WEIGHTS, logical_id=5,
                           size=1024, file_offset=4096)
        assert buf.tell() == REGION_ENTRY_SIZE

        buf.seek(0)
        kind, logical_id, size, file_offset = read_region_entry(buf)
        assert kind == KIND_WEIGHTS
        assert logical_id == 5
        assert size == 1024
        assert file_offset == 4096

    def test_kv_block_kind(self):
        """Region entry with KV block kind roundtrips correctly."""
        buf = io.BytesIO()
        write_region_entry(buf, kind=KIND_KV_LIVE_BLOCK, logical_id=0,
                           size=2**20, file_offset=2**16)
        buf.seek(0)
        kind, logical_id, size, file_offset = read_region_entry(buf)
        assert kind == KIND_KV_LIVE_BLOCK
        assert size == 2**20

    def test_metadata_kind(self):
        buf = io.BytesIO()
        write_region_entry(buf, kind=KIND_METADATA, logical_id=0,
                           size=256, file_offset=4096)
        buf.seek(0)
        kind, _, _, _ = read_region_entry(buf)
        assert kind == KIND_METADATA

    def test_truncated_entry(self):
        """Truncated region entry raises ValueError."""
        buf = io.BytesIO(b"\x00" * 16)  # only 16 of 32 bytes
        with pytest.raises(ValueError, match="truncated"):
            read_region_entry(buf)

    def test_multiple_entries(self):
        """Multiple entries written sequentially can be read back."""
        buf = io.BytesIO()
        for i in range(10):
            write_region_entry(buf, kind=KIND_WEIGHTS, logical_id=i,
                               size=(i + 1) * 1000, file_offset=4096 + i * 1000)

        buf.seek(0)
        for i in range(10):
            kind, lid, size, offset = read_region_entry(buf)
            assert kind == KIND_WEIGHTS
            assert lid == i
            assert size == (i + 1) * 1000

    def test_large_size(self):
        """Region entry can encode multi-GB sizes (uint64)."""
        big_size = 16 * (2**30)  # 16 GB
        buf = io.BytesIO()
        write_region_entry(buf, kind=KIND_WEIGHTS, logical_id=0,
                           size=big_size, file_offset=4096)
        buf.seek(0)
        _, _, size, _ = read_region_entry(buf)
        assert size == big_size


# ---------------------------------------------------------------------------
# Full file format roundtrip
# ---------------------------------------------------------------------------


class TestFullFileRoundtrip:
    def test_header_plus_regions(self):
        """Write a complete header + region table, read it all back."""
        buf = io.BytesIO()
        num_regions = 3
        write_header(buf, num_regions=num_regions)

        sizes = [1024, 2048, 4096]
        offset = HEADER_SIZE + num_regions * REGION_ENTRY_SIZE
        for i, s in enumerate(sizes):
            write_region_entry(buf, kind=KIND_WEIGHTS, logical_id=i,
                               size=s, file_offset=offset)
            offset += s

        # Read back
        buf.seek(0)
        nr, _ = read_header(buf)
        assert nr == num_regions

        for i in range(num_regions):
            kind, lid, size, foff = read_region_entry(buf)
            assert kind == KIND_WEIGHTS
            assert lid == i
            assert size == sizes[i]


# ---------------------------------------------------------------------------
# rank_snapshot_path
# ---------------------------------------------------------------------------


class TestRankSnapshotPath:
    def test_rank_0(self):
        assert rank_snapshot_path("weights.thaw", 0) == "weights.thaw"

    def test_rank_1(self):
        assert rank_snapshot_path("weights.thaw", 1) == "weights.rank1.thaw"

    def test_rank_3(self):
        assert rank_snapshot_path("weights.thaw", 3) == "weights.rank3.thaw"

    def test_rank_7(self):
        assert rank_snapshot_path("weights.thaw", 7) == "weights.rank7.thaw"

    def test_nested_path(self):
        assert rank_snapshot_path("/data/models/llama.thaw", 2) == \
            "/data/models/llama.rank2.thaw"

    def test_no_extension(self):
        assert rank_snapshot_path("weights", 1) == "weights.rank1"

    def test_rank_0_passthrough(self):
        """Rank 0 always returns the exact base path — backward compat."""
        path = "/some/path/model.thaw"
        assert rank_snapshot_path(path, 0) is path
