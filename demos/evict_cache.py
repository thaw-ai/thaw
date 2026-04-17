#!/usr/bin/env python3
"""evict_cache.py — force page-cache eviction for a file on high-RAM hosts.

posix_fadvise(POSIX_FADV_DONTNEED) is advisory. On a pod with 1+ TB free,
the kernel ignores it because there is no memory pressure. This script
allocates memory until the target file is evicted, then releases it — so
the next read of the file hits NVMe instead of page cache.

Usage:
    EVICT_PATH=/workspace/snap.thaw EVICT_GB=100 python3 demos/evict_cache.py
"""
import os
import ctypes
import time

path = os.environ.get("EVICT_PATH", "/workspace/snap.thaw")
size_gb = int(os.environ.get("EVICT_GB", "100"))

libc = ctypes.CDLL("libc.so.6", use_errno=True)
POSIX_FADV_DONTNEED = 4

# First try the polite ask
fd = os.open(path, os.O_RDONLY)
try:
    os.fsync(fd)
    libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
finally:
    os.close(fd)

print(f"path:    {path}")
print(f"target:  allocate {size_gb} GB to force cache eviction")

blocks = []
try:
    for i in range(size_gb):
        blocks.append(bytearray(1 << 30))
        if (i + 1) % 10 == 0:
            with open("/proc/meminfo") as f:
                cached = next(l for l in f if l.startswith("Cached:")).strip()
            print(f"  +{i+1} GB allocated — {cached}")
except MemoryError:
    print(f"hit MemoryError at {len(blocks)} GB allocated")

print("releasing memory...")
blocks.clear()
time.sleep(2)
print("done — rerun your benchmark now for a cold read")
