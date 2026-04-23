// crates/thaw-runtime/src/pipeline.rs
//
// =============================================================================
// PIPELINED RESTORE — DOUBLE-BUFFERED ASYNC DMA
// =============================================================================
//
// This is the performance-critical restore path. Instead of reading
// the entire file into memory and then copying it to the GPU in one
// shot, we:
//
//   1. Open the file (O_DIRECT when possible).
//   2. Parse the header to get the region table.
//   3. Resolve all regions to device addresses upfront.
//   4. Double-buffer: while one pinned buffer is being DMA'd to the
//      GPU via cudaMemcpyAsync, pread fills the other from disk.
//
// The result is that disk I/O and PCIe DMA overlap, and the total
// time is bounded by the slower of the two (usually disk).
//
// =============================================================================

use std::cell::RefCell;
use std::io::Write;
use std::path::Path;

use thaw_core::{ByteRegionWriter, RegionKind, Snapshot, SnapshotError, HEADER_SIZE};

use crate::backend::{DeviceRegion, PipelinedBackend};
use crate::direct_io::{
    fsync_dir, fsync_file, open_direct, open_direct_write, pread_exact, pwrite_exact,
    truncate,
};
use crate::freeze::{FreezeConfig, FreezeError, FreezeRequest};
use crate::restore::{crc_verification_disabled, RestoreError, RestoreStats};

// =============================================================================
// WRITE-COMBINING PINNED BUFFER CACHE (per-thread, amortized across calls)
// =============================================================================
//
// `alloc_pinned_wc` calls `cudaHostAlloc` / `cudaHostRegister`, which is
// O(pages) — pinning 128 MiB (two 64 MiB WC chunk buffers) on a cold
// thread is ~tens of milliseconds; repeat that for every `thaw serve`
// hot-swap or agent fork and it becomes a visible floor.
//
// The fix: cache one pair of WC buffers per thread, keyed on chunk_size.
// Subsequent calls with the same chunk_size reuse the cached pair and
// skip allocation entirely. If a later call comes in with a different
// chunk_size, the old pair is dropped (freed) and a new pair is
// allocated — no memory leak, worst case is the same as no cache.
//
// Thread-local (not Mutex'd global) because:
//   - PinnedBuffer is Send but not Sync. Two threads can't share one.
//   - CUDA contexts are typically thread-bound; a pinned buffer
//     allocated under one context is not guaranteed to be valid under
//     another.
//   - No lock contention on the fast path.
//
// Disable with `THAW_DISABLE_WC_CACHE=1` for debugging.

thread_local! {
    // WC (write-combined) cache for the one code path that writes to the
    // pinned buffer and never reads it back on CPU:
    // `restore_pipelined_from_bytes`, which pre-verifies CRCs against the
    // source slice.
    static WC_BUF_CACHE: RefCell<Option<WcBufCache>> = const { RefCell::new(None) };
    // Plain-pinned cache for every path that CPU-reads the pinned buffer —
    // freeze (CRC + pwrite) and disk-backed `restore_pipelined` (chunked
    // CRC fold). WC reads are 2-3 orders of magnitude slower on most
    // hosts; A100 SXM showed ~50x on the freeze side (2026-04-19),
    // H100 SXM showed similar on the restore side (2026-04-19). Plain
    // pinned is therefore the default for both directions.
    static PINNED_BUF_CACHE: RefCell<Option<PinnedBufCache>> = const { RefCell::new(None) };
}

struct WcBufCache {
    chunk_size: usize,
    bufs: [crate::backend::PinnedBuffer; 2],
}

struct PinnedBufCache {
    chunk_size: usize,
    bufs: [crate::backend::PinnedBuffer; 2],
}

fn wc_cache_disabled() -> bool {
    std::env::var("THAW_DISABLE_WC_CACHE")
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Force every CPU-reads-the-buffer path (freeze + disk-backed restore)
/// back onto WRITE_COMBINED pinned memory. Both paths CPU-read the buffer
/// (freeze for CRC + pwrite, restore for chunked CRC fold), and WC reads
/// are 2-3 orders of magnitude slower than plain pinned on most host CPUs
/// — so plain pinned is the default. This env exists as an opt-in escape
/// hatch for hardware where WC reads are measurably fast.
///
/// Env name kept for back-compat even though it now governs restore too.
fn freeze_use_wc() -> bool {
    std::env::var("THAW_FREEZE_USE_WC")
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Acquire a pair of WC pinned buffers sized `chunk_size`. Returns the
/// cached pair if one is available, otherwise allocates fresh.
fn acquire_wc_bufs<B: PipelinedBackend>(
    backend: &B,
    chunk_size: usize,
) -> Result<[crate::backend::PinnedBuffer; 2], crate::backend::BackendError> {
    if !wc_cache_disabled() {
        let cached = WC_BUF_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            match c.take() {
                Some(cache) if cache.chunk_size == chunk_size => Some(cache.bufs),
                other => {
                    // Stale size or empty — drop the cached pair (if any)
                    // and fall through to a fresh allocation below.
                    drop(other);
                    None
                }
            }
        });
        if let Some(bufs) = cached {
            return Ok(bufs);
        }
    }
    Ok([
        backend.alloc_pinned_wc(chunk_size)?,
        backend.alloc_pinned_wc(chunk_size)?,
    ])
}

/// Return a pair of WC pinned buffers to the per-thread cache. If the
/// cache already holds a pair, the new pair is dropped (freed). This
/// keeps cached footprint bounded to one pair per thread.
fn release_wc_bufs(bufs: [crate::backend::PinnedBuffer; 2], chunk_size: usize) {
    if wc_cache_disabled() {
        drop(bufs);
        return;
    }
    WC_BUF_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.is_none() {
            *c = Some(WcBufCache { chunk_size, bufs });
        }
        // else: existing cache wins; `bufs` drops here = freed.
    });
}

/// Drop this thread's cached WC pinned buffer pair (if any). Safe to
/// call when no pair is cached. Use for explicit shutdown and for
/// tests that need deterministic allocation.
pub fn clear_wc_buf_cache() {
    WC_BUF_CACHE.with(|c| *c.borrow_mut() = None);
    PINNED_BUF_CACHE.with(|c| *c.borrow_mut() = None);
}

/// Acquire a pair of plain (non-WC) pinned buffers for the freeze path.
/// Cached separately from WC so freeze and restore don't evict each
/// other. If `THAW_FREEZE_USE_WC=1` is set, routes to the WC cache
/// instead (see `freeze_use_wc`).
fn acquire_pinned_bufs<B: PipelinedBackend>(
    backend: &B,
    chunk_size: usize,
) -> Result<[crate::backend::PinnedBuffer; 2], crate::backend::BackendError> {
    if freeze_use_wc() {
        return acquire_wc_bufs(backend, chunk_size);
    }
    if !wc_cache_disabled() {
        let cached = PINNED_BUF_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            match c.take() {
                Some(cache) if cache.chunk_size == chunk_size => Some(cache.bufs),
                other => {
                    drop(other);
                    None
                }
            }
        });
        if let Some(bufs) = cached {
            return Ok(bufs);
        }
    }
    Ok([
        backend.alloc_pinned(chunk_size)?,
        backend.alloc_pinned(chunk_size)?,
    ])
}

/// Return a pair of plain pinned buffers to the freeze-path cache.
fn release_pinned_bufs(bufs: [crate::backend::PinnedBuffer; 2], chunk_size: usize) {
    if freeze_use_wc() {
        return release_wc_bufs(bufs, chunk_size);
    }
    if wc_cache_disabled() {
        drop(bufs);
        return;
    }
    PINNED_BUF_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.is_none() {
            *c = Some(PinnedBufCache { chunk_size, bufs });
        }
        // else: existing cache wins; `bufs` drops here = freed.
    });
}

/// Configuration for the pipelined restore.
pub struct PipelineConfig {
    /// Size of each I/O chunk in bytes. Each of the two pinned
    /// buffers will be this size. Default: 64 MiB.
    pub chunk_size: usize,

    /// Whether to attempt O_DIRECT when opening the file.
    /// Falls back to buffered I/O if the platform/filesystem
    /// does not support it.
    pub try_direct_io: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            chunk_size: 64 * 1024 * 1024, // 64 MiB
            try_direct_io: true,
        }
    }
}

/// Stats returned by `freeze_pipelined`.
#[derive(Debug, Default)]
pub struct FreezeStats {
    /// Number of device regions frozen.
    pub regions_frozen: usize,
    /// Total bytes DMA'd from device to host.
    pub bytes_copied: u64,
}

// =============================================================================
// PIPELINED FREEZE — DOUBLE-BUFFERED ASYNC D2H
// =============================================================================
//
// The reverse of the pipelined restore. Instead of overlapping disk
// reads with H2D DMA, we overlap D2H DMA with disk writes:
//
//   GPU --cudaMemcpyAsync(stream 0)--> [Pinned Buffer A] --> Disk
//     ^                                                       |
//     |  (swap)                                               v
//   GPU --cudaMemcpyAsync(stream 1)--> [Pinned Buffer B] --> Disk
//
// While stream 0 copies device memory into buffer A, the previous
// chunk (in buffer B) is being written to disk. The bottleneck
// becomes max(disk, PCIe) instead of disk + PCIe.

/// A single entry in the freeze plan: which device region maps to
/// which byte range in the output file.
///
/// Mirror of `CopyEntry` (see below), with the same field layout —
/// the freeze and restore pipelines share the same chunked
/// algorithm, they just flip the copy direction and the I/O
/// direction.
struct FreezePlan {
    /// Region kind — needed at the end of the freeze to rebuild the
    /// region table with the computed CRCs stamped on it.
    kind: RegionKind,
    /// Logical id (meaningful for KV blocks).
    logical_id: u32,
    /// Absolute byte offset in the output file where this region's
    /// payload starts. Determined by the region table layout.
    file_offset: u64,
    /// Size of this region in bytes.
    size: u64,
    /// Source device region to D2H copy from.
    device_region: DeviceRegion,
}

/// Build the freeze plan + serialized prelude bytes.
///
/// The prelude (header + region table) is constructed in memory so
/// the chunked pipeline can carry it as the head of chunk 0 without
/// a separate write call. Returns `(plan, prelude_bytes, total_file_size)`.
fn build_freeze_plan(
    requests: &[FreezeRequest],
    config: &FreezeConfig,
) -> Result<(Vec<FreezePlan>, Vec<u8>, u64), FreezeError> {
    let mut writer = ByteRegionWriter::new();
    if let Some(commit) = config.vllm_commit {
        writer.set_vllm_commit(commit);
    }
    for request in requests {
        writer.push_region_metadata(
            request.kind,
            request.logical_id,
            request.device_region.size,
        );
    }
    let snapshot = writer.build_snapshot();

    // Serialize prelude to a Vec<u8>. Small (4 KiB + 24 B/region).
    let mut prelude_bytes: Vec<u8> = Vec::with_capacity(snapshot.prelude_size() as usize);
    snapshot.write_to(&mut prelude_bytes)?;

    // Extract file_offsets from the region table.
    let mut plan: Vec<FreezePlan> = Vec::with_capacity(requests.len());
    let mut total_bytes: u64 = 0;
    for (i, request) in requests.iter().enumerate() {
        let entry = snapshot
            .table()
            .get(i)
            .expect("region_table entry for freeze request");
        plan.push(FreezePlan {
            kind: request.kind,
            logical_id: request.logical_id,
            file_offset: entry.file_offset(),
            size: entry.size(),
            device_region: request.device_region.clone(),
        });
        total_bytes += request.device_region.size;
    }

    let prelude_size = prelude_bytes.len() as u64;
    let total_file_size = if requests.is_empty() {
        prelude_size
    } else {
        // Payload ends at the highest (file_offset + size). Region table
        // packs them sequentially, so this equals prelude_size + total_bytes.
        prelude_size + total_bytes
    };

    Ok((plan, prelude_bytes, total_file_size))
}

/// Determine which plan entries overlap a given chunk and issue
/// async D2H copies into the buffer. Mirror of `launch_uploads` in
/// restore.
fn launch_dumps<B: PipelinedBackend>(
    backend: &B,
    buf: &mut crate::backend::PinnedBuffer,
    stream: &crate::backend::StreamHandle,
    chunk_start: u64,
    chunk_end: u64,
    plan: &[FreezePlan],
) -> Result<(), FreezeError> {
    for entry in plan {
        let region_start = entry.file_offset;
        let region_end = entry.file_offset + entry.size;

        // Skip entries that don't overlap this chunk.
        if region_end <= chunk_start || region_start >= chunk_end {
            continue;
        }

        // Compute the overlap in file coordinates.
        let overlap_start = region_start.max(chunk_start);
        let overlap_end = region_end.min(chunk_end);
        let overlap_len = overlap_end - overlap_start;

        // Offset within the pinned buffer (where in the chunk to land).
        let buf_offset = (overlap_start - chunk_start) as usize;

        // Offset within the source device region.
        let dev_offset = overlap_start - region_start;

        // Sub-region of the device allocation.
        let sub_region = DeviceRegion::new(
            crate::backend::DevicePtr(entry.device_region.ptr.0 + dev_offset),
            overlap_len,
        );

        backend.memcpy_d2h_async(buf, buf_offset, &sub_region, stream)?;
    }
    Ok(())
}

/// Per-region CRC state while reading a chunked restore. Carries the
/// running CRC32C and the number of bytes folded in so the restore
/// loop can tell when a region is complete and ready to check.
#[derive(Debug, Clone, Copy)]
struct RestoreCrcState {
    crc: u32,
    bytes_seen: u64,
}

impl RestoreCrcState {
    fn new() -> Self {
        Self { crc: 0, bytes_seen: 0 }
    }
}

/// Verify every region's CRC32C from the full in-memory `data` slice
/// in a single sequential pass before any DMA starts. Used by the
/// in-memory restore paths where we can afford an upfront scan — in
/// exchange we get atomic "verify then DMA" semantics: a corrupt
/// snapshot never touches device memory. Skipped when `THAW_VERIFY=0`
/// or a region records CRC 0 (forward compat with older snapshots).
fn verify_plan_crcs_from_bytes(
    data: &[u8],
    plan: &[CopyEntry],
) -> Result<(), RestoreError> {
    if crc_verification_disabled() {
        return Ok(());
    }
    for entry in plan.iter() {
        if entry.expected_crc32c == 0 {
            continue;
        }
        let start = entry.file_offset as usize;
        let end = start + entry.size as usize;
        let actual = crc32c_append_parallel(0, &data[start..end]);
        if actual != entry.expected_crc32c {
            return Err(RestoreError::ChecksumMismatch {
                kind: entry.kind,
                logical_id: entry.logical_id,
                expected: entry.expected_crc32c,
                actual,
            });
        }
    }
    Ok(())
}

/// Extend each region's running CRC by the overlap between a chunk
/// and the region, reading the bytes from `src`. `src_base` is the
/// file coordinate of `src[0]` so overlap math works regardless of
/// whether the buffer is a pread target (chunk_start == src_base) or
/// an absolute file slice (src_base == 0).
///
/// Verifies and returns `ChecksumMismatch` the instant a region
/// finishes (its last chunk is folded in) and its accumulated CRC
/// disagrees with the expected one.
/// Extend `initial_crc` by the bytes of `slice`, splitting across N
/// worker threads and stitching via `crc32c_combine`. Serial fallback
/// for slices below the threshold avoids thread-pool overhead for small
/// region overlaps.
///
/// On modern x86, single-core `crc32c_append` hits ~6–8 GB/s; 8 cores
/// with combine stitching reaches ~30 GB/s before memory bandwidth
/// starts to bite. This is the hot path for the restore pipeline, so
/// keeping it out of the critical-path serialization budget matters.
fn crc32c_append_parallel(initial_crc: u32, slice: &[u8]) -> u32 {
    const PARALLEL_THRESHOLD: usize = 4 * 1024 * 1024; // 4 MiB
    const NUM_SHARDS: usize = 8;

    if slice.len() < PARALLEL_THRESHOLD {
        return crc32c::crc32c_append(initial_crc, slice);
    }

    // Split into NUM_SHARDS roughly equal sub-slices. The last shard
    // absorbs any length remainder so total == slice.len().
    let shard_len = slice.len() / NUM_SHARDS;
    let shards: Vec<&[u8]> = (0..NUM_SHARDS)
        .map(|i| {
            let start = i * shard_len;
            let end = if i == NUM_SHARDS - 1 { slice.len() } else { (i + 1) * shard_len };
            &slice[start..end]
        })
        .collect();

    let crcs: Vec<u32> = std::thread::scope(|s| {
        let handles: Vec<_> = shards
            .iter()
            .map(|sub| s.spawn(move || crc32c::crc32c(sub)))
            .collect();
        handles.into_iter().map(|h| h.join().expect("crc worker panic")).collect()
    });

    // Fold the shard CRCs into the running CRC via combine.
    let mut total = initial_crc;
    for (crc, sub) in crcs.iter().zip(shards.iter()) {
        total = crc32c::crc32c_combine(total, *crc, sub.len());
    }
    total
}

fn fold_and_verify_chunk_crc(
    src: &[u8],
    src_base: u64,
    chunk_start: u64,
    chunk_end: u64,
    plan: &[CopyEntry],
    state: &mut [RestoreCrcState],
) -> Result<(), RestoreError> {
    debug_assert_eq!(state.len(), plan.len());
    if crc_verification_disabled() {
        return Ok(());
    }
    for (i, entry) in plan.iter().enumerate() {
        if entry.expected_crc32c == 0 {
            continue;
        }
        let region_start = entry.file_offset;
        let region_end = entry.file_offset + entry.size;
        if region_end <= chunk_start || region_start >= chunk_end {
            continue;
        }
        let overlap_start = region_start.max(chunk_start);
        let overlap_end = region_end.min(chunk_end);
        let overlap_len = (overlap_end - overlap_start) as usize;
        let src_offset = (overlap_start - src_base) as usize;
        let slice = &src[src_offset..src_offset + overlap_len];
        state[i].crc = crc32c_append_parallel(state[i].crc, slice);
        state[i].bytes_seen += overlap_len as u64;

        if state[i].bytes_seen == entry.size && state[i].crc != entry.expected_crc32c {
            return Err(RestoreError::ChecksumMismatch {
                kind: entry.kind,
                logical_id: entry.logical_id,
                expected: entry.expected_crc32c,
                actual: state[i].crc,
            });
        }
    }
    Ok(())
}

/// Rebuild the on-disk region-table bytes from the freeze plan with
/// per-region CRCs stamped in. Produces exactly
/// `plan.len() * REGION_ENTRY_SIZE` bytes that overwrite the initial
/// CRC-zero table written at prelude time.
fn region_table_bytes_with_crcs(plan: &[FreezePlan], crcs: &[u32]) -> Vec<u8> {
    debug_assert_eq!(crcs.len(), plan.len());
    let mut table = thaw_core::RegionTable::new();
    for (entry, crc) in plan.iter().zip(crcs.iter()) {
        let rec = thaw_core::RegionEntry::new(entry.kind)
            .with_logical_id(entry.logical_id)
            .with_size(entry.size)
            .with_file_offset(entry.file_offset)
            .with_crc32c(*crc);
        table.push(rec);
    }
    table.to_bytes()
}

/// After D2H for a chunk has landed in the pinned buffer, extend the
/// per-region running CRC32C by the overlap slice in file order. This
/// is called once the chunk's stream has synced, so the bytes are
/// fully materialized in the buffer.
///
/// Panics if `crcs.len() != plan.len()` — caller bug. Returns the
/// file region of pinned-buffer bytes covered so the freeze loop can
/// compute write offsets consistently.
fn accumulate_freeze_crcs(
    buf: &crate::backend::PinnedBuffer,
    chunk_start: u64,
    chunk_end: u64,
    chunk_buf_base: u64,
    plan: &[FreezePlan],
    crcs: &mut [u32],
) {
    debug_assert_eq!(crcs.len(), plan.len());
    for (i, entry) in plan.iter().enumerate() {
        let region_start = entry.file_offset;
        let region_end = entry.file_offset + entry.size;
        if region_end <= chunk_start || region_start >= chunk_end {
            continue;
        }
        let overlap_start = region_start.max(chunk_start);
        let overlap_end = region_end.min(chunk_end);
        let overlap_len = (overlap_end - overlap_start) as usize;
        let buf_offset = (overlap_start - chunk_buf_base) as usize;
        let slice = &buf.as_slice()[buf_offset..buf_offset + overlap_len];
        crcs[i] = crc32c_append_parallel(crcs[i], slice);
    }
}

/// Pipelined freeze: double-buffered async D2H with overlapped
/// writes (generic writer).
///
/// Produces the same `.thaw` file format as `freeze`, but:
///   - One `memcpy_d2h_async` per region, packed into fixed-size
///     chunks instead of allocating per-region pinned buffers.
///   - Two write-combining pinned buffers, reused across the entire
///     freeze, so allocation cost is O(1) not O(regions).
///   - D2H of chunk N overlaps the write of chunk N-1.
///
/// This is the in-memory / test-friendly entry point. For production
/// freezes to disk use `freeze_pipelined_to_file`, which adds
/// O_DIRECT writes via `pwrite`.
pub fn freeze_pipelined<B, W>(
    backend: &B,
    requests: &[FreezeRequest],
    config: &FreezeConfig,
    sink: &mut W,
) -> Result<FreezeStats, FreezeError>
where
    B: PipelinedBackend,
    W: Write,
{
    let (plan, prelude_bytes, total_file_size) = build_freeze_plan(requests, config)?;

    if requests.is_empty() {
        // Prelude-only file — nothing to CRC, write and return.
        sink.write_all(&prelude_bytes)
            .map_err(|e| FreezeError::Snapshot(SnapshotError::from(e)))?;
        return Ok(FreezeStats::default());
    }

    // The region table carries per-region CRCs, but the CRCs are only
    // known after D2H completes. Since the generic `Write` sink isn't
    // seek-able, we buffer the whole file in RAM and patch the table
    // bytes before handing them to the sink. This path is test- and
    // bench-only; production freezes use `freeze_pipelined_to_file`
    // which uses `pwrite` to avoid the RAM buffer.
    let mut file_buf: Vec<u8> = vec![0u8; total_file_size as usize];
    file_buf[..prelude_bytes.len()].copy_from_slice(&prelude_bytes);

    let chunk_size = config.chunk_size;
    if chunk_size == 0 {
        return Err(FreezeError::InvalidConfig {
            message: "chunk_size must be > 0".to_string(),
        });
    }

    let payload_start = prelude_bytes.len() as u64;
    let payload_end = total_file_size;
    let payload_len = (payload_end - payload_start) as usize;
    if payload_len == 0 {
        sink.write_all(&file_buf)
            .map_err(|e| FreezeError::Snapshot(SnapshotError::from(e)))?;
        return Ok(FreezeStats {
            regions_frozen: requests.len(),
            bytes_copied: 0,
        });
    }
    let num_chunks = (payload_len + chunk_size - 1) / chunk_size;

    let total_bytes: u64 = plan.iter().map(|p| p.size).sum();

    // Two plain pinned buffers + two streams. Plain (non-WC) pinned
    // because the freeze path READS these buffers on the CPU (for CRC
    // and the write-back to the file buffer), and WC reads are orders
    // of magnitude slower than plain pinned on most host CPUs. The
    // per-thread cache amortizes `cudaHostAlloc` across calls.
    let mut bufs = acquire_pinned_bufs(backend, chunk_size)?;
    let streams = [backend.stream_create()?, backend.stream_create()?];

    // Per-region CRC accumulators. Index-aligned with `plan`.
    let mut crcs: Vec<u32> = vec![0u32; plan.len()];

    // Helper: compute this chunk's [start, end) in file coordinates.
    let chunk_range = |idx: usize| -> (u64, u64) {
        let start = payload_start + (idx as u64) * (chunk_size as u64);
        let end = (start + chunk_size as u64).min(payload_end);
        (start, end)
    };

    // Helper: copy the first `len` pinned-buffer bytes into the
    // in-memory file buffer at `file_offset`, and extend per-region
    // CRCs by the overlap slices.
    let commit_chunk = |buf: &crate::backend::PinnedBuffer,
                        file_offset: u64,
                        chunk_end: u64,
                        len: usize,
                        file_buf: &mut [u8],
                        crcs: &mut [u32]| {
        let dst_start = file_offset as usize;
        file_buf[dst_start..dst_start + len].copy_from_slice(&buf.as_slice()[..len]);
        accumulate_freeze_crcs(buf, file_offset, chunk_end, file_offset, &plan, crcs);
    };

    // Prime: D2H chunk 0 into bufs[0], sync.
    {
        let (c0_start, c0_end) = chunk_range(0);
        launch_dumps(backend, &mut bufs[0], &streams[0], c0_start, c0_end, &plan)?;
        backend.stream_sync(&streams[0])?;
    }

    if num_chunks == 1 {
        let (c0_start, c0_end) = chunk_range(0);
        let len = (c0_end - c0_start) as usize;
        commit_chunk(&bufs[0], c0_start, c0_end, len, &mut file_buf, &mut crcs);
    } else {
        // Steady state: D2H(chunk i) overlaps memcpy(chunk i-1).
        for chunk_idx in 1..num_chunks {
            let prev = (chunk_idx - 1) % 2;
            let curr = chunk_idx % 2;

            let (prev_start, prev_end) = chunk_range(chunk_idx - 1);
            let prev_len = (prev_end - prev_start) as usize;

            let (curr_start, curr_end) = chunk_range(chunk_idx);

            // Kick off D2H for the current chunk into its buffer.
            launch_dumps(
                backend,
                &mut bufs[curr],
                &streams[curr],
                curr_start,
                curr_end,
                &plan,
            )?;

            // While D2H is in flight on streams[curr], commit the
            // previous chunk to the in-memory buffer.
            commit_chunk(
                &bufs[prev],
                prev_start,
                prev_end,
                prev_len,
                &mut file_buf,
                &mut crcs,
            );

            // Wait for the current chunk's D2H to complete before
            // the next iteration reuses its buffer.
            backend.stream_sync(&streams[curr])?;
        }

        // Commit the final chunk.
        let last = (num_chunks - 1) % 2;
        let (last_start, last_end) = chunk_range(num_chunks - 1);
        let last_len = (last_end - last_start) as usize;
        commit_chunk(
            &bufs[last],
            last_start,
            last_end,
            last_len,
            &mut file_buf,
            &mut crcs,
        );
    }

    // Cleanup streams.
    backend
        .stream_destroy(streams[0])
        .map_err(FreezeError::Backend)?;
    backend
        .stream_destroy(streams[1])
        .map_err(FreezeError::Backend)?;

    // Return plain pinned buffers to the per-thread freeze cache.
    release_pinned_bufs(bufs, chunk_size);

    // Patch the region-table bytes with computed CRCs, then flush the
    // whole file to the sink in one `write_all`.
    let table_bytes = region_table_bytes_with_crcs(&plan, &crcs);
    let table_start = HEADER_SIZE as usize;
    file_buf[table_start..table_start + table_bytes.len()].copy_from_slice(&table_bytes);

    sink.write_all(&file_buf)
        .map_err(|e| FreezeError::Snapshot(SnapshotError::from(e)))?;

    Ok(FreezeStats {
        regions_frozen: requests.len(),
        bytes_copied: total_bytes,
    })
}

/// Pipelined freeze to a file path, using O_DIRECT writes when
/// supported.
///
/// Mirror of `restore_pipelined`: fixed-size chunks, write-combining
/// pinned buffers, double-buffered D2H + I/O overlap, O_DIRECT on
/// aligned chunks with a buffered fallback for ragged edges.
///
/// The prelude (header + region table) is carried as the head of
/// chunk 0 so the first pwrite can start at offset 0. Final chunk
/// is padded up to 4 KiB for O_DIRECT and then the file is
/// truncated down to the exact size afterward.
///
/// The write is atomic from the caller's perspective: bytes land in
/// `{path}.thaw-incomplete` and are renamed over `path` only after
/// fsync succeeds. Any error, panic, or early return unlinks the
/// tmp file via `TmpFileGuard`. A process-level SIGKILL orphans the
/// tmp but cannot produce a partially-valid `path`, so `thaw
/// restore` will never observe torn state.
pub fn freeze_pipelined_to_file<B>(
    backend: &B,
    requests: &[FreezeRequest],
    config: &FreezeConfig,
    path: &Path,
) -> Result<FreezeStats, FreezeError>
where
    B: PipelinedBackend,
{
    let (plan, prelude_bytes, total_file_size) = build_freeze_plan(requests, config)?;

    let io_err = |e: std::io::Error| {
        FreezeError::Snapshot(SnapshotError::Io {
            kind: e.kind(),
            message: e.to_string(),
        })
    };

    // Derive the tmp path (`{path}.thaw-incomplete`). The suffix
    // is deliberately distinct from `.tmp` so users cannot confuse
    // it with their own scratch files, and distinct from `.thaw`
    // so glob patterns over snapshot dirs won't pick it up.
    let tmp_path = tmp_path_for(path);

    // RAII guard: unlinks tmp_path on drop unless `disarm()` is
    // called. Covers early returns, panics, and `?` propagation
    // from any of the open/write/fsync/rename calls below.
    let mut guard = TmpFileGuard::arm(&tmp_path);

    // Open both fds on the tmp path. The first creates+truncates;
    // the second attaches without disturbing length (both fds
    // view the same inode).
    let file_direct =
        open_direct_write(&tmp_path, config.try_direct_io, true).map_err(io_err)?;
    let file_buffered = open_direct_write(&tmp_path, false, false).map_err(io_err)?;

    let chunk_size = config.chunk_size;
    if chunk_size == 0 {
        return Err(FreezeError::InvalidConfig {
            message: "chunk_size must be > 0".to_string(),
        });
    }
    if chunk_size % 4096 != 0 {
        return Err(FreezeError::InvalidConfig {
            message: format!(
                "chunk_size must be a multiple of 4096 for O_DIRECT, got {}",
                chunk_size
            ),
        });
    }

    // In the file-based path we write starting at offset 0 and
    // the prelude lives at the head of chunk 0. So "chunks" here
    // span the entire file, not just the payload region.
    let file_len = total_file_size;
    if file_len == 0 {
        // Empty snapshot — just write the prelude (prelude-only file),
        // fsync, then atomically rename tmp → path.
        pwrite_exact(&file_buffered, &prelude_bytes, 0).map_err(io_err)?;
        fsync_file(&file_buffered).map_err(io_err)?;
        drop(file_direct);
        drop(file_buffered);
        commit_tmp_to_final(&tmp_path, path).map_err(io_err)?;
        guard.disarm();
        return Ok(FreezeStats::default());
    }

    let num_chunks = ((file_len as usize) + chunk_size - 1) / chunk_size;

    let total_bytes: u64 = plan.iter().map(|p| p.size).sum();

    // Align-up helper for O_DIRECT write sizes.
    fn align_up_4k(n: usize) -> usize {
        (n + 4095) & !4095
    }

    // Two plain pinned buffers + two streams. Plain (non-WC) pinned is
    // deliberate: the freeze path reads these buffers on the CPU (for
    // CRC accumulation and pwrite), and WC reads are orders of
    // magnitude slower than plain pinned reads on most host CPUs. The
    // per-thread cache amortizes `cudaHostAlloc` across successive
    // freeze calls (e.g. `thaw serve` hot-swap).
    let mut bufs = acquire_pinned_bufs(backend, chunk_size)?;
    let streams = [backend.stream_create()?, backend.stream_create()?];

    // Per-region CRC accumulators. The initial table written in chunk
    // 0 has zero CRCs; we pwrite a corrected table at the end.
    let mut crcs: Vec<u32> = vec![0u32; plan.len()];

    // Helper: chunk range in absolute file coordinates.
    let chunk_range = |idx: usize| -> (u64, u64) {
        let start = (idx as u64) * (chunk_size as u64);
        let end = (start + chunk_size as u64).min(file_len);
        (start, end)
    };

    // Helper: prepare a chunk's buffer — seed any prelude bytes that
    // land in this chunk's range and kick off D2H for any overlapping
    // region payloads. The prelude lives at [0, prelude_bytes.len())
    // in file coordinates; it may span multiple chunks if the region
    // table is large relative to chunk_size.
    let prepare_chunk =
        |backend: &B,
         buf: &mut crate::backend::PinnedBuffer,
         stream: &crate::backend::StreamHandle,
         chunk_start: u64,
         chunk_end: u64,
         plan: &[FreezePlan]|
         -> Result<(), FreezeError> {
            let pre_end = prelude_bytes.len() as u64;
            if chunk_start < pre_end {
                let overlap_start = chunk_start;
                let overlap_end = chunk_end.min(pre_end);
                let overlap_len = (overlap_end - overlap_start) as usize;
                let pre_offset = overlap_start as usize;
                let buf_offset = 0usize;
                buf.as_mut_slice()[buf_offset..buf_offset + overlap_len]
                    .copy_from_slice(&prelude_bytes[pre_offset..pre_offset + overlap_len]);
            }
            // D2H-dump any region payload bytes that fall in this chunk.
            launch_dumps(backend, buf, stream, chunk_start, chunk_end, plan)?;
            Ok(())
        };

    // Helper: write a chunk to disk, picking the right fd.
    // Takes `&mut` so we can zero-fill the ragged tail before an
    // O_DIRECT-aligned write (otherwise we'd leak uninit pinned
    // memory to disk, even though we truncate right after).
    let write_chunk_to_file = |buf: &mut crate::backend::PinnedBuffer,
                               chunk_start: u64,
                               chunk_len: usize|
     -> Result<(), FreezeError> {
        let aligned = align_up_4k(chunk_len);
        if file_direct.is_direct() && aligned <= chunk_size {
            // Aligned O_DIRECT write. If the tail of the aligned
            // region is past the true file length, zero-fill the
            // padding and truncate at the very end.
            if aligned > chunk_len {
                for b in &mut buf.as_mut_slice()[chunk_len..aligned] {
                    *b = 0;
                }
            }
            pwrite_exact(&file_direct, &buf.as_slice()[..aligned], chunk_start).or_else(
                |e| {
                    if e.kind() == std::io::ErrorKind::InvalidInput {
                        // Some filesystems reject O_DIRECT at ragged edges.
                        pwrite_exact(&file_buffered, &buf.as_slice()[..chunk_len], chunk_start)
                    } else {
                        Err(e)
                    }
                },
            )
        } else {
            pwrite_exact(&file_buffered, &buf.as_slice()[..chunk_len], chunk_start)
        }
        .map_err(io_err)
    };

    // Prime: prepare chunk 0, sync.
    {
        let (c0_start, c0_end) = chunk_range(0);
        prepare_chunk(backend, &mut bufs[0], &streams[0], c0_start, c0_end, &plan)?;
        backend.stream_sync(&streams[0])?;
    }

    if num_chunks == 1 {
        let (c0_start, c0_end) = chunk_range(0);
        let len = (c0_end - c0_start) as usize;
        accumulate_freeze_crcs(&bufs[0], c0_start, c0_end, c0_start, &plan, &mut crcs);
        write_chunk_to_file(&mut bufs[0], c0_start, len)?;
    } else {
        for chunk_idx in 1..num_chunks {
            let prev = (chunk_idx - 1) % 2;
            let curr = chunk_idx % 2;

            let (prev_start, prev_end) = chunk_range(chunk_idx - 1);
            let prev_len = (prev_end - prev_start) as usize;

            let (curr_start, curr_end) = chunk_range(chunk_idx);

            // Kick off D2H for current chunk.
            prepare_chunk(
                backend,
                &mut bufs[curr],
                &streams[curr],
                curr_start,
                curr_end,
                &plan,
            )?;

            // Accumulate CRC for the just-synced previous chunk, then
            // write it to disk (overlaps the current chunk's D2H).
            accumulate_freeze_crcs(
                &bufs[prev],
                prev_start,
                prev_end,
                prev_start,
                &plan,
                &mut crcs,
            );
            write_chunk_to_file(&mut bufs[prev], prev_start, prev_len)?;

            // Wait for current chunk's D2H before reuse.
            backend.stream_sync(&streams[curr])?;
        }

        // Accumulate CRC for + write final chunk.
        let last = (num_chunks - 1) % 2;
        let (last_start, last_end) = chunk_range(num_chunks - 1);
        let last_len = (last_end - last_start) as usize;
        accumulate_freeze_crcs(
            &bufs[last],
            last_start,
            last_end,
            last_start,
            &plan,
            &mut crcs,
        );
        write_chunk_to_file(&mut bufs[last], last_start, last_len)?;
    }

    // Patch the region-table bytes on disk with the computed CRCs,
    // then fsync so the patched table is durable before the rename.
    // Uses the buffered fd because the table is only 32 bytes per
    // region and pwrite doesn't care about O_DIRECT alignment.
    let table_bytes = region_table_bytes_with_crcs(&plan, &crcs);
    pwrite_exact(&file_buffered, &table_bytes, HEADER_SIZE).map_err(io_err)?;

    // Truncate to exact file size (strips O_DIRECT 4 KiB padding).
    truncate(&file_direct, file_len).map_err(io_err)?;

    // fsync data + metadata before the rename so the bytes are
    // durable. Both fds reference the same inode; one fsync covers
    // the file.
    fsync_file(&file_direct).map_err(io_err)?;

    // Release fds before renaming — some filesystems (e.g. AFS,
    // older NFS) are finicky about renaming an open file.
    drop(file_direct);
    drop(file_buffered);

    // Atomic rename tmp → path, then fsync the parent dir so the
    // new name survives a power loss.
    commit_tmp_to_final(&tmp_path, path).map_err(io_err)?;

    // Write succeeded — don't unlink tmp in Drop (already renamed).
    guard.disarm();

    // Cleanup streams.
    backend
        .stream_destroy(streams[0])
        .map_err(FreezeError::Backend)?;
    backend
        .stream_destroy(streams[1])
        .map_err(FreezeError::Backend)?;

    // Return plain pinned buffers to the per-thread freeze cache.
    release_pinned_bufs(bufs, chunk_size);

    Ok(FreezeStats {
        regions_frozen: requests.len(),
        bytes_copied: total_bytes,
    })
}

/// Derive the intermediate-write path for an atomic freeze. The
/// `.thaw-incomplete` suffix is deliberate: distinct from generic
/// `.tmp` and from `.thaw` so glob patterns over snapshot
/// directories won't see partial writes as valid snapshots.
fn tmp_path_for(path: &Path) -> std::path::PathBuf {
    let mut os = path.as_os_str().to_owned();
    os.push(".thaw-incomplete");
    std::path::PathBuf::from(os)
}

/// Commit a completed tmp file to its final path. Performs an
/// atomic `rename` then fsyncs the parent directory so the new
/// name is durable. Caller must have already fsynced the tmp's
/// contents before calling this.
fn commit_tmp_to_final(tmp: &Path, final_path: &Path) -> std::io::Result<()> {
    std::fs::rename(tmp, final_path)?;
    if let Some(parent) = final_path.parent() {
        if !parent.as_os_str().is_empty() {
            // Best-effort: some filesystems (e.g. tmpfs) don't need
            // or support dir fsync. Swallow ENOTSUP-style errors.
            let _ = fsync_dir(parent);
        }
    }
    Ok(())
}

/// RAII guard that unlinks a tmp file on drop unless `disarm()` is
/// called first. Covers the "freeze errored or panicked before
/// rename" case so callers don't leak half-written `.thaw-incomplete`
/// files into the snapshot directory.
struct TmpFileGuard {
    path: Option<std::path::PathBuf>,
}

impl TmpFileGuard {
    fn arm(path: &Path) -> Self {
        TmpFileGuard {
            path: Some(path.to_path_buf()),
        }
    }

    fn disarm(&mut self) {
        self.path = None;
    }
}

impl Drop for TmpFileGuard {
    fn drop(&mut self) {
        if let Some(p) = self.path.take() {
            let _ = std::fs::remove_file(&p);
        }
    }
}

// =============================================================================
// PIPELINED RESTORE
// =============================================================================

/// A single entry in the copy plan: which region of the file maps
/// to which device region.
struct CopyEntry {
    /// Region kind — preserved so a checksum-mismatch error can
    /// report "which region" to the operator.
    kind: RegionKind,
    /// Logical id (meaningful for KV blocks).
    logical_id: u32,
    /// Absolute byte offset in the file where this region's payload
    /// starts.
    file_offset: u64,
    /// Size of this region in bytes.
    size: u64,
    /// Where on the device to put it.
    device_region: DeviceRegion,
    /// CRC32C recorded by the freeze writer. Zero if the file was
    /// produced before CRCs existed; restore skips verification in
    /// that case.
    expected_crc32c: u32,
}

/// Pipelined restore from a `.thaw` file.
///
/// Takes a file path (not a byte slice) because the pipeline needs
/// control over I/O (O_DIRECT, pread) rather than receiving the
/// whole file pre-loaded.
///
/// `resolve` is called once per region in the file's region table,
/// before any I/O starts. It must return the `DeviceRegion` where
/// the bytes should land. All mappings are validated upfront so a
/// missing or mismatched region fails before any DMA work begins.
pub fn restore_pipelined<B, F>(
    backend: &B,
    path: &Path,
    mut resolve: F,
    config: &PipelineConfig,
) -> Result<RestoreStats, RestoreError>
where
    B: PipelinedBackend,
    F: FnMut(RegionKind, u32) -> Option<DeviceRegion>,
{
    // -- Phase 1: Parse header and build copy plan --------------------
    //
    // We use a buffered fd for the header/region-table because:
    //   - O_DIRECT requires page-aligned buffers, but Vec<u8> from the
    //     heap is only guaranteed 16-byte alignment.
    //   - The header is tiny (4 KiB), so page cache is fine here.
    //
    // A separate O_DIRECT fd is opened later for the bulk payload.

    fn io_err(e: std::io::Error) -> RestoreError {
        RestoreError::Snapshot(thaw_core::SnapshotError::Io {
            kind: e.kind(),
            message: e.to_string(),
        })
    }

    let buffered_file = open_direct(path, false).map_err(io_err)?;

    let mut header_buf = vec![0u8; HEADER_SIZE as usize];
    pread_exact(&buffered_file, &mut header_buf, 0).map_err(io_err)?;

    // We need header + region table to parse the snapshot. The region
    // table sits right after the header. Read enough to cover it.
    let snapshot = Snapshot::from_prelude_bytes(&header_buf).or_else(|_| {
        // Header parsed, but the buffer was too short for the region
        // table. Read the full prelude.
        let num_regions = {
            let h = thaw_core::SnapshotHeader::from_bytes(&header_buf)
                .map_err(|e| RestoreError::Snapshot(thaw_core::SnapshotError::Header(e)))?;
            h.num_regions() as usize
        };
        let prelude_size =
            HEADER_SIZE as usize + num_regions * thaw_core::REGION_ENTRY_SIZE as usize;
        let mut prelude_buf = vec![0u8; prelude_size];
        pread_exact(&buffered_file, &mut prelude_buf, 0).map_err(io_err)?;
        Snapshot::from_prelude_bytes(&prelude_buf).map_err(RestoreError::Snapshot)
    })?;

    // Build the copy plan: resolve every region to a device address.
    let mut plan: Vec<CopyEntry> = Vec::with_capacity(snapshot.len());
    let mut total_bytes: u64 = 0;

    for i in 0..snapshot.len() {
        let entry = snapshot
            .table()
            .get(i)
            .ok_or(RestoreError::Snapshot(thaw_core::SnapshotError::TruncatedTable {
                got: 0,
                need: 0,
            }))?;

        let kind = entry.kind();
        let logical_id = entry.logical_id();
        let size = entry.size();
        let file_offset = entry.file_offset();
        let expected_crc32c = entry.crc32c();

        let device_region =
            resolve(kind, logical_id).ok_or(RestoreError::UnmappedRegion {
                kind,
                logical_id,
            })?;

        if device_region.size != size {
            return Err(RestoreError::DeviceSizeMismatch {
                kind,
                logical_id,
                file_size: size,
                device_size: device_region.size,
            });
        }

        total_bytes += size;
        plan.push(CopyEntry {
            kind,
            logical_id,
            file_offset,
            size,
            device_region,
            expected_crc32c,
        });
    }

    // Handle empty file.
    if plan.is_empty() {
        return Ok(RestoreStats::default());
    }

    // -- Phase 2: Double-buffered pipelined restore -------------------
    //
    // Open O_DIRECT fd for the bulk payload reads. The pinned buffers
    // from cudaMallocHost are page-aligned (CUDA guarantee), satisfying
    // O_DIRECT's buffer alignment requirement. We round chunk read
    // sizes up to 4096 so the kernel never sees an unaligned request.
    // The buffered_file is kept around as a fallback for the last chunk
    // if the file size isn't block-aligned.

    let file = open_direct(path, config.try_direct_io).map_err(io_err)?;

    let chunk_size = config.chunk_size;
    if chunk_size == 0 || chunk_size % 4096 != 0 {
        return Err(RestoreError::InvalidConfig {
            message: format!(
                "chunk_size must be a positive multiple of 4096, got {}",
                chunk_size
            ),
        });
    }

    // Determine the byte range we need to read from the file.
    let payload_start = plan.first().unwrap().file_offset;
    let payload_end = plan
        .iter()
        .map(|e| e.file_offset + e.size)
        .max()
        .unwrap();

    // Align the start down to the nearest page boundary for O_DIRECT.
    let read_start = payload_start & !0xFFF; // round down to 4096
    let read_len = (payload_end - read_start) as usize;
    let num_chunks = (read_len + chunk_size - 1) / chunk_size;

    /// Round `n` up to the next multiple of 4096.
    fn align_up_4k(n: usize) -> usize {
        (n + 4095) & !4095
    }

    // Allocate two pinned buffers and two streams.
    //
    // We use PLAIN pinned (not write-combined) here. Superficially WC
    // looks right — CPU writes via pread, GPU reads via DMA — but this
    // path also CPU-READS the buffer to fold it into per-region CRCs
    // (`fold_and_verify_chunk_crc` below). WC memory is catastrophically
    // slow for CPU reads (2-3 orders of magnitude), which made TP=1
    // restore stall at ~0.05 GB/s on H100 SXM even though the DMA itself
    // was fine. See commit 2de24bf for the matching freeze-side fix.
    //
    // The sibling `restore_pipelined_from_bytes` path keeps WC because
    // it pre-verifies CRCs directly against the input slice and never
    // CPU-reads the pinned buffer.
    let mut bufs = acquire_pinned_bufs(backend, chunk_size).map_err(RestoreError::Backend)?;
    let streams = [
        backend.stream_create().map_err(RestoreError::Backend)?,
        backend.stream_create().map_err(RestoreError::Backend)?,
    ];

    // Helper: for a given chunk byte range [chunk_start, chunk_end),
    // find all plan entries that overlap and issue async copies.
    let launch_uploads = |backend: &B,
                          buf: &crate::backend::PinnedBuffer,
                          stream: &crate::backend::StreamHandle,
                          chunk_start: u64,
                          chunk_end: u64,
                          plan: &[CopyEntry]|
     -> Result<(), RestoreError> {
        for entry in plan {
            let region_start = entry.file_offset;
            let region_end = entry.file_offset + entry.size;

            // Skip entries that don't overlap this chunk.
            if region_end <= chunk_start || region_start >= chunk_end {
                continue;
            }

            // Compute the overlap.
            let overlap_start = region_start.max(chunk_start);
            let overlap_end = region_end.min(chunk_end);
            let overlap_len = overlap_end - overlap_start;

            // Offset within the pinned buffer.
            let buf_offset = (overlap_start - chunk_start) as usize;

            // Offset within the device region.
            let dev_offset = overlap_start - region_start;

            // Sub-region of the device allocation.
            let sub_region = DeviceRegion::new(
                crate::backend::DevicePtr(entry.device_region.ptr.0 + dev_offset),
                overlap_len,
            );

            backend
                .memcpy_h2d_async(&sub_region, buf, buf_offset, stream)
                .map_err(RestoreError::Backend)?;
        }
        Ok(())
    };

    // Helper: read a chunk, picking the right fd and aligned size.
    // O_DIRECT requires size aligned to 4096. If the aligned size fits
    // in the pinned buffer, use the O_DIRECT fd. Otherwise fall back to
    // the buffered fd (only happens for the last chunk if the file size
    // isn't block-aligned).
    let read_chunk =
        |buf: &mut crate::backend::PinnedBuffer, offset: u64, needed: usize| -> Result<(), RestoreError> {
            let aligned = align_up_4k(needed);
            if file.is_direct() && aligned <= chunk_size {
                // Aligned read via O_DIRECT into pinned (page-aligned) buffer.
                pread_exact(&file, &mut buf.as_mut_slice()[..aligned], offset).or_else(|e| {
                    // If we get EINVAL or UnexpectedEof (reading past file end
                    // with aligned size), fall back to buffered.
                    if e.kind() == std::io::ErrorKind::InvalidInput
                        || e.kind() == std::io::ErrorKind::UnexpectedEof
                    {
                        pread_exact(&buffered_file, &mut buf.as_mut_slice()[..needed], offset)
                    } else {
                        Err(e)
                    }
                })
            } else {
                pread_exact(&buffered_file, &mut buf.as_mut_slice()[..needed], offset)
            }
            .map_err(io_err)
        };

    // Per-region CRC verification state.
    let mut crc_state: Vec<RestoreCrcState> = vec![RestoreCrcState::new(); plan.len()];

    // Prime the pump: read chunk 0.
    let chunk0_start = read_start;
    let chunk0_len = chunk_size.min(read_len);
    read_chunk(&mut bufs[0], chunk0_start, chunk0_len)?;

    if num_chunks == 1 {
        // Single chunk: verify CRC over the buffer, then upload.
        let chunk_end = chunk0_start + chunk0_len as u64;
        fold_and_verify_chunk_crc(
            &bufs[0].as_slice()[..chunk0_len],
            chunk0_start,
            chunk0_start,
            chunk_end,
            &plan,
            &mut crc_state,
        )?;
        launch_uploads(
            backend,
            &bufs[0],
            &streams[0],
            chunk0_start,
            chunk_end,
            &plan,
        )?;
        backend.stream_sync(&streams[0]).map_err(RestoreError::Backend)?;
    } else {
        // Steady-state double-buffer loop.
        for chunk_idx in 1..num_chunks {
            let prev = (chunk_idx - 1) % 2;

            let prev_start = read_start + (chunk_idx - 1) as u64 * chunk_size as u64;
            let prev_len = chunk_size.min(read_len - (chunk_idx - 1) * chunk_size);
            let prev_end = prev_start + prev_len as u64;

            let curr_start = read_start + chunk_idx as u64 * chunk_size as u64;
            let curr_len = chunk_size.min(read_len - chunk_idx * chunk_size);

            // Launch async uploads from the previous chunk's buffer FIRST.
            // cudaMemcpyAsync only enqueues the DMA; the pinned buffer
            // stays CPU-readable, so we can fold its CRC in a worker
            // thread while the main thread preads the next chunk.
            launch_uploads(
                backend,
                &bufs[prev],
                &streams[prev],
                prev_start,
                prev_end,
                &plan,
            )?;

            // Split bufs so we can hold &bufs[prev] (for CRC on worker)
            // and &mut bufs[curr] (for pread on main) simultaneously.
            let (slot0, slot1) = bufs.split_at_mut(1);
            let (prev_buf_ref, curr_buf_ref): (
                &crate::backend::PinnedBuffer,
                &mut crate::backend::PinnedBuffer,
            ) = if prev == 0 {
                (&slot0[0], &mut slot1[0])
            } else {
                (&slot1[0], &mut slot0[0])
            };
            let prev_slice: &[u8] = &prev_buf_ref.as_slice()[..prev_len];
            let plan_ref: &[CopyEntry] = &plan;
            let crc_state_ref: &mut [RestoreCrcState] = &mut crc_state;

            // Overlap CRC fold (CPU) with pread (disk I/O). Before this
            // reorder, CRC and pread were serialized on the main thread;
            // on 16 GB snapshots that added ~2.5s (CRC32C ≈ 6.5 GB/s
            // single-core) on top of the pipelined pread. With them
            // parallel, the critical path per chunk becomes max(CRC,
            // pread) instead of CRC + pread.
            std::thread::scope(|s| -> Result<(), RestoreError> {
                let crc_handle = s.spawn(move || {
                    fold_and_verify_chunk_crc(
                        prev_slice,
                        prev_start,
                        prev_start,
                        prev_end,
                        plan_ref,
                        crc_state_ref,
                    )
                });
                let read_result = read_chunk(curr_buf_ref, curr_start, curr_len);
                let crc_result = crc_handle
                    .join()
                    .expect("CRC worker panicked");
                read_result.and(crc_result)
            })?;

            // Sync the previous stream before its buffer gets reused
            // in the next iteration.
            backend
                .stream_sync(&streams[prev])
                .map_err(RestoreError::Backend)?;
        }

        // Fold + upload the final chunk.
        let last = (num_chunks - 1) % 2;
        let last_start = read_start + (num_chunks - 1) as u64 * chunk_size as u64;
        let last_len = chunk_size.min(read_len - (num_chunks - 1) * chunk_size);
        let last_end = last_start + last_len as u64;
        fold_and_verify_chunk_crc(
            &bufs[last].as_slice()[..last_len],
            last_start,
            last_start,
            last_end,
            &plan,
            &mut crc_state,
        )?;
        launch_uploads(
            backend,
            &bufs[last],
            &streams[last],
            last_start,
            last_end,
            &plan,
        )?;
        backend
            .stream_sync(&streams[0])
            .map_err(RestoreError::Backend)?;
        backend
            .stream_sync(&streams[1])
            .map_err(RestoreError::Backend)?;
    }

    // Cleanup.
    backend
        .stream_destroy(streams[0])
        .map_err(RestoreError::Backend)?;
    backend
        .stream_destroy(streams[1])
        .map_err(RestoreError::Backend)?;

    // Return pinned buffers to the per-thread cache.
    release_pinned_bufs(bufs, chunk_size);

    Ok(RestoreStats {
        regions_restored: plan.len(),
        bytes_copied: total_bytes,
    })
}

// =============================================================================
// PIPELINED RESTORE FROM MEMORY (RAM-BACKED)
// =============================================================================
//
// Same double-buffered pipeline as `restore_pipelined`, but reads from
// a byte slice already in host memory instead of from disk. This
// eliminates disk I/O entirely — the bottleneck becomes pure PCIe DMA
// bandwidth.
//
// Use case: a serverless platform (Modal, Beam, RunPod) pre-stages
// snapshots in RAM (tmpfs, page cache, or shared memory). The
// orchestrator reads the file once, then hands the bytes to this
// function for every cold start. The "disk read" step becomes a
// memcpy at ~50 GB/s (DDR5), making the total restore time
// PCIe-limited: 16 GB / 25 GB/s ≈ 640ms on Gen4, ~310ms on Gen5.

/// Pipelined restore from an in-memory byte slice.
///
/// Identical algorithm to [`restore_pipelined`], but the source data
/// is a `&[u8]` that is already in host memory. Each "read" is a
/// `copy_from_slice` (~50 GB/s) instead of a `pread` syscall.
///
/// The double-buffering still helps: while one pinned buffer is being
/// DMA'd to the GPU, the next chunk is memcpy'd into the other buffer.
/// On PCIe Gen4, memcpy (50 GB/s) fully hides behind DMA (25 GB/s).
pub fn restore_pipelined_from_bytes<B, F>(
    backend: &B,
    data: &[u8],
    mut resolve: F,
    config: &PipelineConfig,
) -> Result<RestoreStats, RestoreError>
where
    B: PipelinedBackend,
    F: FnMut(RegionKind, u32) -> Option<DeviceRegion>,
{
    // -- Phase 1: Parse header and build copy plan from byte slice ---

    if data.len() < HEADER_SIZE as usize {
        return Err(RestoreError::Snapshot(thaw_core::SnapshotError::Io {
            kind: std::io::ErrorKind::UnexpectedEof,
            message: format!("data too short for header: {} bytes", data.len()),
        }));
    }

    let snapshot = Snapshot::from_prelude_bytes(data).map_err(RestoreError::Snapshot)?;

    let mut plan: Vec<CopyEntry> = Vec::with_capacity(snapshot.len());
    let mut total_bytes: u64 = 0;

    for i in 0..snapshot.len() {
        let entry = snapshot
            .table()
            .get(i)
            .ok_or(RestoreError::Snapshot(thaw_core::SnapshotError::TruncatedTable {
                got: 0,
                need: 0,
            }))?;

        let kind = entry.kind();
        let logical_id = entry.logical_id();
        let size = entry.size();
        let file_offset = entry.file_offset();
        let expected_crc32c = entry.crc32c();

        let device_region =
            resolve(kind, logical_id).ok_or(RestoreError::UnmappedRegion {
                kind,
                logical_id,
            })?;

        if device_region.size != size {
            return Err(RestoreError::DeviceSizeMismatch {
                kind,
                logical_id,
                file_size: size,
                device_size: device_region.size,
            });
        }

        total_bytes += size;
        plan.push(CopyEntry {
            kind,
            logical_id,
            file_offset,
            size,
            device_region,
            expected_crc32c,
        });
    }

    if plan.is_empty() {
        return Ok(RestoreStats::default());
    }

    // -- Phase 2: Double-buffered pipelined restore from memory -----

    let chunk_size = config.chunk_size;
    if chunk_size == 0 || chunk_size % 4096 != 0 {
        return Err(RestoreError::InvalidConfig {
            message: format!(
                "chunk_size must be a positive multiple of 4096, got {}",
                chunk_size
            ),
        });
    }

    let payload_start = plan.first().unwrap().file_offset;
    let payload_end = plan
        .iter()
        .map(|e| e.file_offset + e.size)
        .max()
        .unwrap();

    let read_start = payload_start & !0xFFF;
    let read_len = (payload_end - read_start) as usize;
    let num_chunks = (read_len + chunk_size - 1) / chunk_size;

    let mut bufs = acquire_wc_bufs(backend, chunk_size).map_err(RestoreError::Backend)?;
    let streams = [
        backend.stream_create().map_err(RestoreError::Backend)?,
        backend.stream_create().map_err(RestoreError::Backend)?,
    ];

    // Helper: issue async H2D copies for regions overlapping a chunk.
    let launch_uploads = |backend: &B,
                          buf: &crate::backend::PinnedBuffer,
                          stream: &crate::backend::StreamHandle,
                          chunk_start: u64,
                          chunk_end: u64,
                          plan: &[CopyEntry]|
     -> Result<(), RestoreError> {
        for entry in plan {
            let region_start = entry.file_offset;
            let region_end = entry.file_offset + entry.size;

            if region_end <= chunk_start || region_start >= chunk_end {
                continue;
            }

            let overlap_start = region_start.max(chunk_start);
            let overlap_end = region_end.min(chunk_end);
            let overlap_len = overlap_end - overlap_start;

            let buf_offset = (overlap_start - chunk_start) as usize;
            let dev_offset = overlap_start - region_start;

            let sub_region = DeviceRegion::new(
                crate::backend::DevicePtr(entry.device_region.ptr.0 + dev_offset),
                overlap_len,
            );

            backend
                .memcpy_h2d_async(&sub_region, buf, buf_offset, stream)
                .map_err(RestoreError::Backend)?;
        }
        Ok(())
    };

    // Helper: copy a chunk from the byte slice into a pinned buffer.
    let copy_chunk =
        |buf: &mut crate::backend::PinnedBuffer,
         offset: u64,
         len: usize|
         -> Result<(), RestoreError> {
            let src_start = offset as usize;
            let src_end = src_start + len;
            if src_end > data.len() {
                return Err(RestoreError::Snapshot(thaw_core::SnapshotError::Io {
                    kind: std::io::ErrorKind::UnexpectedEof,
                    message: format!(
                        "data too short: need byte {}..{} but len is {}",
                        src_start, src_end, data.len()
                    ),
                }));
            }
            buf.as_mut_slice()[..len].copy_from_slice(&data[src_start..src_end]);
            Ok(())
        };

    // Verify every region's CRC up front — `data` is already resident
    // in RAM, so a single sequential scan gives atomic "verify then
    // DMA" semantics: a corrupt snapshot cannot touch device memory.
    verify_plan_crcs_from_bytes(data, &plan)?;

    // Prime: copy chunk 0 into pinned buffer.
    let chunk0_start = read_start;
    let chunk0_len = chunk_size.min(read_len);
    copy_chunk(&mut bufs[0], chunk0_start, chunk0_len)?;

    if num_chunks == 1 {
        let chunk_end = chunk0_start + chunk0_len as u64;
        launch_uploads(
            backend,
            &bufs[0],
            &streams[0],
            chunk0_start,
            chunk_end,
            &plan,
        )?;
        backend
            .stream_sync(&streams[0])
            .map_err(RestoreError::Backend)?;
    } else {
        for chunk_idx in 1..num_chunks {
            let prev = (chunk_idx - 1) % 2;
            let curr = chunk_idx % 2;

            let prev_start = read_start + (chunk_idx - 1) as u64 * chunk_size as u64;
            let prev_len = chunk_size.min(read_len - (chunk_idx - 1) * chunk_size);
            let prev_end = prev_start + prev_len as u64;

            let curr_start = read_start + chunk_idx as u64 * chunk_size as u64;
            let curr_len = chunk_size.min(read_len - chunk_idx * chunk_size);

            // Launch async DMA from previous buffer.
            launch_uploads(
                backend,
                &bufs[prev],
                &streams[prev],
                prev_start,
                prev_end,
                &plan,
            )?;

            // Memcpy next chunk into current buffer (overlaps with DMA).
            copy_chunk(&mut bufs[curr], curr_start, curr_len)?;

            // Sync previous stream before reusing its buffer.
            backend
                .stream_sync(&streams[prev])
                .map_err(RestoreError::Backend)?;
        }

        // Upload final chunk.
        let last = (num_chunks - 1) % 2;
        let last_start = read_start + (num_chunks - 1) as u64 * chunk_size as u64;
        let last_len = chunk_size.min(read_len - (num_chunks - 1) * chunk_size);
        let last_end = last_start + last_len as u64;
        launch_uploads(
            backend,
            &bufs[last],
            &streams[last],
            last_start,
            last_end,
            &plan,
        )?;
        backend
            .stream_sync(&streams[0])
            .map_err(RestoreError::Backend)?;
        backend
            .stream_sync(&streams[1])
            .map_err(RestoreError::Backend)?;
    }

    // Cleanup.
    backend
        .stream_destroy(streams[0])
        .map_err(RestoreError::Backend)?;
    backend
        .stream_destroy(streams[1])
        .map_err(RestoreError::Backend)?;

    // Return WC buffers to the per-thread cache.
    release_wc_bufs(bufs, chunk_size);

    Ok(RestoreStats {
        regions_restored: plan.len(),
        bytes_copied: total_bytes,
    })
}

// =============================================================================
// ZERO-COPY PIPELINED RESTORE — DMA DIRECTLY FROM REGISTERED HOST MEMORY
// =============================================================================
//
// Identical shape to `restore_pipelined_from_bytes`, but skips the
// memcpy-into-pinned-buffer hop by `cudaHostRegister`-ing the input
// byte range once up front. After registration, each region's bytes
// can be DMA'd directly from the mapped pages via the raw async
// memcpy path — no intermediate staging buffer, no double-buffering.
//
// This is the fix for the pre-staged-RAM path. The previous
// `restore_pipelined_from_bytes` was bottlenecked by
// `buf.as_mut_slice().copy_from_slice(&data[..])` at ~7 GB/s per core.
// Registering an mmap of the snapshot lets us fire async DMAs on two
// streams from the mapped pages directly; wall-clock converges on the
// PCIe Gen5 x16 ceiling (~25 GB/s on the H100 SXM pod).
//
// The fallback is still available via `restore_pipelined_from_bytes`
// for hosts where `cudaHostRegister` fails (low `ulimit -l`, sealed
// mmap, etc.); the Python wrapper tries this path first and only
// falls back on error.

/// Zero-copy pipelined restore from a registered host byte range.
///
/// Calls `backend.host_register` on the full input range once, then
/// issues `memcpy_h2d_async_raw` for each region slice directly from
/// the registered pointer. Returns when all copies have completed on
/// both streams.
///
/// No double-buffering: the registered range IS the DMA source, so
/// there is nothing to fill. Two streams are still used so the driver
/// can overlap the per-region DMAs across the GPU's two copy engines.
///
/// # Safety / Requirements
///
/// - `data` must come from a stable allocation that lives for the
///   duration of this call. Typical source: a Python mmap passed
///   through PyO3 with the GIL held.
/// - For real CUDA, `data.as_ptr()` must be page-aligned (mmap
///   always is).
/// - The registration can fail if the system's `ulimit -l` is too
///   low for `data.len()` bytes; callers handle this by catching
///   `RestoreError::Backend { .. BackendError::Cuda { .. } }` and
///   falling back to `restore_pipelined_from_bytes`.
pub fn restore_pipelined_from_registered_bytes<B, F>(
    backend: &B,
    data: &[u8],
    resolve: F,
    config: &PipelineConfig,
) -> Result<RestoreStats, RestoreError>
where
    B: PipelinedBackend,
    F: FnMut(RegionKind, u32) -> Option<DeviceRegion>,
{
    // Build the copy plan under the `data` borrow.
    let (plan, total_bytes) = plan_copies_from_bytes(data, resolve)?;

    if plan.is_empty() {
        return Ok(RestoreStats::default());
    }

    // Register the whole buffer once for this call. The RAII guard
    // unpins on the way out regardless of which branch we take.
    //
    // SAFETY: `data` is borrowed for the duration of this function so
    // the pointer is valid for exactly `data.len()` bytes. The guard
    // drops before we return, so the registration does not outlive
    // the borrow.
    let _registration = unsafe {
        backend
            .host_register(data.as_ptr() as *mut u8, data.len())
            .map_err(RestoreError::Backend)?
    };

    run_pre_registered_plan(backend, data, &plan, config)?;

    Ok(RestoreStats {
        regions_restored: plan.len(),
        bytes_copied: total_bytes,
    })
}

/// DMA from a buffer the caller has **already** registered via
/// `cudaHostRegister` (or equivalent). This is the hot path for
/// `thaw serve`: the slot holds a persistent registration over its
/// mmap, and every restore into that slot skips the O(pages) cost of
/// re-registering 16+ GB on each call.
///
/// Behavior is otherwise identical to
/// `restore_pipelined_from_registered_bytes` — same plan construction,
/// same two-stream round-robin DMA, same stream sync on exit.
///
/// # Safety / Requirements
///
/// - `data` must point at bytes currently registered for DMA. If the
///   bytes are pageable, `cudaMemcpyAsync` degrades to synchronous
///   transfer (slow, not UB).
/// - `data` must outlive this call. The caller typically holds the
///   registration and the underlying mmap alive via a `PinnedMmap`
///   PyO3 handle on the Python side.
pub fn restore_pipelined_from_pre_registered_bytes<B, F>(
    backend: &B,
    data: &[u8],
    resolve: F,
    config: &PipelineConfig,
) -> Result<RestoreStats, RestoreError>
where
    B: PipelinedBackend,
    F: FnMut(RegionKind, u32) -> Option<DeviceRegion>,
{
    let (plan, total_bytes) = plan_copies_from_bytes(data, resolve)?;

    if plan.is_empty() {
        return Ok(RestoreStats::default());
    }

    run_pre_registered_plan(backend, data, &plan, config)?;

    Ok(RestoreStats {
        regions_restored: plan.len(),
        bytes_copied: total_bytes,
    })
}

/// Parse the snapshot header, resolve every region through `resolve`,
/// and return the ordered copy plan plus total bytes to transfer.
/// Shared by both the self-registering and pre-registered variants.
fn plan_copies_from_bytes<F>(
    data: &[u8],
    mut resolve: F,
) -> Result<(Vec<CopyEntry>, u64), RestoreError>
where
    F: FnMut(RegionKind, u32) -> Option<DeviceRegion>,
{
    if data.len() < HEADER_SIZE as usize {
        return Err(RestoreError::Snapshot(thaw_core::SnapshotError::Io {
            kind: std::io::ErrorKind::UnexpectedEof,
            message: format!("data too short for header: {} bytes", data.len()),
        }));
    }

    let snapshot = Snapshot::from_prelude_bytes(data).map_err(RestoreError::Snapshot)?;

    let mut plan: Vec<CopyEntry> = Vec::with_capacity(snapshot.len());
    let mut total_bytes: u64 = 0;

    for i in 0..snapshot.len() {
        let entry = snapshot.table().get(i).ok_or(RestoreError::Snapshot(
            thaw_core::SnapshotError::TruncatedTable { got: 0, need: 0 },
        ))?;

        let kind = entry.kind();
        let logical_id = entry.logical_id();
        let size = entry.size();
        let file_offset = entry.file_offset();
        let expected_crc32c = entry.crc32c();

        let device_region = resolve(kind, logical_id)
            .ok_or(RestoreError::UnmappedRegion { kind, logical_id })?;

        if device_region.size != size {
            return Err(RestoreError::DeviceSizeMismatch {
                kind,
                logical_id,
                file_size: size,
                device_size: device_region.size,
            });
        }

        total_bytes += size;
        plan.push(CopyEntry {
            kind,
            logical_id,
            file_offset,
            size,
            device_region,
            expected_crc32c,
        });
    }

    Ok((plan, total_bytes))
}

/// Execute an already-built plan against `data`, assuming `data` is
/// pinned for DMA. Creates two streams, round-robins regions across
/// them, syncs, and tears the streams down. Does not touch
/// registration — the caller owns that lifecycle.
fn run_pre_registered_plan<B>(
    backend: &B,
    data: &[u8],
    plan: &[CopyEntry],
    _config: &PipelineConfig,
) -> Result<(), RestoreError>
where
    B: PipelinedBackend,
{
    // Verify every region's CRC32C from the source slice before any
    // DMA fires. `data` is already resident (mmap or read-through),
    // so an upfront scan is cheap and buys atomic semantics: a
    // corrupt snapshot cannot touch device memory.
    verify_plan_crcs_from_bytes(data, plan)?;

    let streams = [
        backend.stream_create().map_err(RestoreError::Backend)?,
        backend.stream_create().map_err(RestoreError::Backend)?,
    ];

    // Round-robin regions across streams so the GPU's two copy
    // engines can run concurrently. Each DMA goes straight from the
    // registered pages — no staging copy.
    for (i, entry) in plan.iter().enumerate() {
        // SAFETY: `src_ptr` points into `data`, which is live for the
        // call. The caller's registration (or the wrapper's guard)
        // keeps it pinned through the syncs below.
        let src_ptr = unsafe { data.as_ptr().add(entry.file_offset as usize) };
        let stream = &streams[i % 2];

        unsafe {
            backend
                .memcpy_h2d_async_raw(
                    &entry.device_region,
                    src_ptr,
                    entry.size as usize,
                    stream,
                )
                .map_err(RestoreError::Backend)?;
        }
    }

    backend
        .stream_sync(&streams[0])
        .map_err(RestoreError::Backend)?;
    backend
        .stream_sync(&streams[1])
        .map_err(RestoreError::Backend)?;

    backend
        .stream_destroy(streams[0])
        .map_err(RestoreError::Backend)?;
    backend
        .stream_destroy(streams[1])
        .map_err(RestoreError::Backend)?;

    Ok(())
}

// =============================================================================
// UNIFIED PIPELINED RESTORE — ZERO-COPY WITH AUTOMATIC STAGING FALLBACK
// =============================================================================
//
// Callers shouldn't have to pick between the zero-copy path and the
// staging path — the "right" answer is "try zero-copy; if the host
// can't register the bytes, transparently fall back to the staging
// buffer copy path." That's what this function does.
//
// The zero-copy path can fail for a handful of reasons, all of which
// are recoverable:
//   - `ulimit -l` too low for `data.len()` bytes
//   - buffer not page-aligned (rare — mmap always is, `bytes()` often
//     is too, but a sliced `bytearray` may not be)
//   - the CUDA driver transiently out of pinned-host address space
//
// The staging path works against any `&[u8]` regardless of alignment
// or host memory state; it's just slower (one extra memcpy per chunk
// at ~7 GB/s per core vs. zero-copy PCIe-saturating DMA).
//
// Semantics: `host_register` is attempted before any plan is built or
// any stream is created. If it succeeds we run the zero-copy pipeline;
// if it fails we fall back to the staging pipeline. Either way the
// caller observes the same `RestoreStats` and the same error surface
// (a backend failure *during* DMA still propagates as `RestoreError`).

/// Pipelined restore from an in-memory byte slice with automatic
/// zero-copy / staging dispatch.
///
/// First attempts [`restore_pipelined_from_registered_bytes`] (DMA
/// directly from a `cudaHostRegister`-ed view of `data`). If the
/// registration call fails — typically due to low `ulimit -l`,
/// non-page-aligned bytes, or transient host-memory pressure — falls
/// back to [`restore_pipelined_from_bytes`] (copy through a pinned
/// staging buffer). The fallback is logged at WARN level via
/// `eprintln!` with the failure reason so operators can diagnose why a
/// given call landed on the slower path.
///
/// This is the correct entry point for callers that have snapshot
/// bytes in host memory and don't know, or don't want to know, whether
/// the current environment can support zero-copy DMA. The fast path
/// is taken whenever possible; correctness is preserved in all cases.
pub fn restore_pipelined_from_bytes_auto<B, F>(
    backend: &B,
    data: &[u8],
    resolve: F,
    config: &PipelineConfig,
) -> Result<RestoreStats, RestoreError>
where
    B: PipelinedBackend,
    F: FnMut(RegionKind, u32) -> Option<DeviceRegion>,
{
    // Try the zero-copy path first. The ONLY failure we consider
    // recoverable is `host_register` itself — that's the "environment
    // can't pin this buffer" signal. A failure during plan
    // construction or during the DMA itself surfaces as a real
    // `RestoreError` and must not be silently retried on the slow
    // path: the slow path would hit the same error.
    //
    // SAFETY: `data` is borrowed for the duration of this function, so
    // the pointer is valid for exactly `data.len()` bytes. The guard
    // (`_registration`) drops before we return, so the registration
    // does not outlive the borrow.
    let registration = unsafe { backend.host_register(data.as_ptr() as *mut u8, data.len()) };

    match registration {
        Ok(_guard) => {
            // Zero-copy path — same body as
            // `restore_pipelined_from_registered_bytes`, inlined so we
            // own the guard lifetime. The guard drops at the end of
            // this match arm, after `run_pre_registered_plan` has
            // synced both streams.
            let (plan, total_bytes) = plan_copies_from_bytes(data, resolve)?;

            if plan.is_empty() {
                return Ok(RestoreStats::default());
            }

            run_pre_registered_plan(backend, data, &plan, config)?;

            Ok(RestoreStats {
                regions_restored: plan.len(),
                bytes_copied: total_bytes,
            })
        }
        Err(register_err) => {
            // Staging-buffer fallback. Log at WARN level so operators
            // can see which call site landed on the slow path and
            // why. The codebase uses `eprintln!` for crate-level
            // diagnostics (no `tracing` dep in thaw-runtime); matching
            // that convention here keeps this wrapper dependency-free.
            eprintln!(
                "thaw: WARN restore_pipelined_from_bytes_auto: zero-copy \
                 host_register failed ({register_err}) for {} bytes; falling \
                 back to staging-buffer restore (slower, but works)",
                data.len()
            );

            restore_pipelined_from_bytes(backend, data, resolve, config)
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::DevicePtr;
    use crate::freeze::{freeze, FreezeConfig, FreezeRequest};
    use crate::mock::MockCuda;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use thaw_core::RegionKind;

    /// Helper: freeze regions to a temp file, return the path.
    fn freeze_to_tempfile(
        backend: &MockCuda,
        requests: &[FreezeRequest],
    ) -> NamedTempFile {
        let mut file: Vec<u8> = Vec::new();
        freeze(backend, requests, &FreezeConfig::default(), &mut file)
            .expect("freeze");
        let mut tmp = NamedTempFile::new().expect("tempfile");
        tmp.write_all(&file).expect("write");
        tmp.flush().expect("flush");
        tmp
    }

    /// Basic round-trip: freeze two regions, restore via pipeline.
    #[test]
    fn pipelined_restore_round_trips() {
        let src = MockCuda::new();
        let w_ptr = DevicePtr(0x1000);
        let m_ptr = DevicePtr(0x2000);
        let weights: Vec<u8> = (0..512).map(|i| ((i * 3) & 0xFF) as u8).collect();
        let metadata: Vec<u8> = (0..64).map(|i| (0xF0 ^ i) as u8).collect();
        src.register_region(w_ptr, weights.clone());
        src.register_region(m_ptr, metadata.clone());

        let requests = vec![
            FreezeRequest::new(
                RegionKind::Weights,
                0,
                DeviceRegion::new(w_ptr, weights.len() as u64),
            ),
            FreezeRequest::new(
                RegionKind::Metadata,
                0,
                DeviceRegion::new(m_ptr, metadata.len() as u64),
            ),
        ];
        let tmp = freeze_to_tempfile(&src, &requests);

        // Restore into a fresh backend.
        let dst = MockCuda::new();
        let w_ptr2 = DevicePtr(0xA000);
        let m_ptr2 = DevicePtr(0xB000);
        dst.register_region(w_ptr2, vec![0u8; weights.len()]);
        dst.register_region(m_ptr2, vec![0u8; metadata.len()]);

        let config = PipelineConfig {
            chunk_size: 4096, // small chunks to exercise the loop
            try_direct_io: false,
        };
        let stats = restore_pipelined(
            &dst,
            tmp.path(),
            |kind, _| match kind {
                RegionKind::Weights => {
                    Some(DeviceRegion::new(w_ptr2, weights.len() as u64))
                }
                RegionKind::Metadata => {
                    Some(DeviceRegion::new(m_ptr2, metadata.len() as u64))
                }
                _ => None,
            },
            &config,
        )
        .expect("restore");

        assert_eq!(stats.regions_restored, 2);
        assert_eq!(
            stats.bytes_copied,
            (weights.len() + metadata.len()) as u64
        );
        assert_eq!(dst.read_region(w_ptr2).unwrap(), weights);
        assert_eq!(dst.read_region(m_ptr2).unwrap(), metadata);
    }

    /// Large region spanning multiple chunks.
    #[test]
    fn large_region_spanning_chunks() {
        let src = MockCuda::new();
        let ptr = DevicePtr(0x1000);
        // 10000 bytes with 4096-byte chunks = ~3 chunks
        let data: Vec<u8> = (0..10000).map(|i| ((i * 7) & 0xFF) as u8).collect();
        src.register_region(ptr, data.clone());

        let requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(ptr, data.len() as u64),
        )];
        let tmp = freeze_to_tempfile(&src, &requests);

        let dst = MockCuda::new();
        let dst_ptr = DevicePtr(0xF000);
        dst.register_region(dst_ptr, vec![0u8; data.len()]);

        let config = PipelineConfig {
            chunk_size: 4096,
            try_direct_io: false,
        };
        let stats = restore_pipelined(
            &dst,
            tmp.path(),
            |_, _| Some(DeviceRegion::new(dst_ptr, data.len() as u64)),
            &config,
        )
        .expect("restore");

        assert_eq!(stats.regions_restored, 1);
        assert_eq!(stats.bytes_copied, data.len() as u64);
        assert_eq!(dst.read_region(dst_ptr).unwrap(), data);
    }

    /// Many small regions that fit within a single chunk.
    #[test]
    fn many_small_regions_in_one_chunk() {
        let src = MockCuda::new();
        let mut requests = Vec::new();
        let mut expected = Vec::new();

        for i in 0..20u32 {
            let ptr = DevicePtr(0x1000 + i as u64 * 0x100);
            let data: Vec<u8> = (0..32).map(|b| (b + i as u8 * 5) & 0xFF).collect();
            src.register_region(ptr, data.clone());
            requests.push(FreezeRequest::new(
                RegionKind::Weights,
                i,
                DeviceRegion::new(ptr, 32),
            ));
            expected.push((ptr, data));
        }
        let tmp = freeze_to_tempfile(&src, &requests);

        let dst = MockCuda::new();
        let mut dst_ptrs = Vec::new();
        for i in 0..20u32 {
            let ptr = DevicePtr(0xA000 + i as u64 * 0x100);
            dst.register_region(ptr, vec![0u8; 32]);
            dst_ptrs.push(ptr);
        }

        let config = PipelineConfig {
            chunk_size: 64 * 1024, // big chunk, all regions fit
            try_direct_io: false,
        };
        let stats = restore_pipelined(
            &dst,
            tmp.path(),
            |_kind, logical_id| {
                Some(DeviceRegion::new(dst_ptrs[logical_id as usize], 32))
            },
            &config,
        )
        .expect("restore");

        assert_eq!(stats.regions_restored, 20);
        for (i, (_, orig_data)) in expected.iter().enumerate() {
            assert_eq!(
                dst.read_region(dst_ptrs[i]).unwrap(),
                *orig_data,
                "region {} mismatch",
                i
            );
        }
    }

    // =================================================================
    // freeze_pipelined tests
    // =================================================================

    /// Basic round-trip: freeze_pipelined then restore via pipeline.
    #[test]
    fn pipelined_freeze_round_trips() {
        let backend = MockCuda::new();
        let w_ptr = DevicePtr(0x1000);
        let m_ptr = DevicePtr(0x2000);
        let weights: Vec<u8> = (0..512).map(|i| ((i * 3) & 0xFF) as u8).collect();
        let metadata: Vec<u8> = (0..64).map(|i| (0xF0 ^ i) as u8).collect();
        backend.register_region(w_ptr, weights.clone());
        backend.register_region(m_ptr, metadata.clone());

        let requests = vec![
            FreezeRequest::new(
                RegionKind::Weights,
                0,
                DeviceRegion::new(w_ptr, weights.len() as u64),
            ),
            FreezeRequest::new(
                RegionKind::Metadata,
                0,
                DeviceRegion::new(m_ptr, metadata.len() as u64),
            ),
        ];

        let mut file: Vec<u8> = Vec::new();
        let stats = freeze_pipelined(
            &backend,
            &requests,
            &FreezeConfig::default(),
            &mut file,
        )
        .expect("freeze_pipelined");
        assert_eq!(stats.regions_frozen, 2);
        assert_eq!(stats.bytes_copied, (weights.len() + metadata.len()) as u64);

        // Verify the file is parseable and byte-exact.
        let parsed = thaw_core::Snapshot::from_prelude_bytes(&file).expect("parse");
        assert_eq!(parsed.len(), 2);
        let e0 = parsed.table().get(0).unwrap();
        let start = e0.file_offset() as usize;
        assert_eq!(&file[start..start + weights.len()], weights.as_slice());
        let e1 = parsed.table().get(1).unwrap();
        let start = e1.file_offset() as usize;
        assert_eq!(&file[start..start + metadata.len()], metadata.as_slice());
    }

    /// Pipelined freeze then pipelined restore: full round-trip
    /// through both pipelines using MockCuda.
    #[test]
    fn pipelined_freeze_then_pipelined_restore() {
        let src = MockCuda::new();
        let ptr = DevicePtr(0x5000);
        let data: Vec<u8> = (0..4096).map(|i| ((i * 7) & 0xFF) as u8).collect();
        src.register_region(ptr, data.clone());

        let requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(ptr, data.len() as u64),
        )];

        // Freeze to in-memory buffer.
        let mut file: Vec<u8> = Vec::new();
        freeze_pipelined(
            &src,
            &requests,
            &FreezeConfig::default(),
            &mut file,
        )
        .expect("freeze");

        // Write to tempfile for pipelined restore (needs file path).
        let mut tmp = NamedTempFile::new().expect("tempfile");
        tmp.write_all(&file).expect("write");
        tmp.flush().expect("flush");

        // Restore into fresh backend.
        let dst = MockCuda::new();
        let dst_ptr = DevicePtr(0xA000);
        dst.register_region(dst_ptr, vec![0u8; data.len()]);

        let restore_config = PipelineConfig {
            chunk_size: 4096,
            try_direct_io: false,
        };
        restore_pipelined(
            &dst,
            tmp.path(),
            |_, _| Some(DeviceRegion::new(dst_ptr, data.len() as u64)),
            &restore_config,
        )
        .expect("restore");

        assert_eq!(dst.read_region(dst_ptr).unwrap(), data);
    }

    /// Many regions: exercises the double-buffer alternation pattern.
    #[test]
    fn pipelined_freeze_many_regions() {
        let backend = MockCuda::new();
        let mut requests = Vec::new();
        let mut expected = Vec::new();

        for i in 0..20u32 {
            let ptr = DevicePtr(0x1000 + i as u64 * 0x100);
            let data: Vec<u8> = (0..32).map(|b| (b + i as u8 * 5) & 0xFF).collect();
            backend.register_region(ptr, data.clone());
            requests.push(FreezeRequest::new(
                RegionKind::Weights,
                i,
                DeviceRegion::new(ptr, 32),
            ));
            expected.push(data);
        }

        let mut file: Vec<u8> = Vec::new();
        let stats = freeze_pipelined(
            &backend,
            &requests,
            &FreezeConfig::default(),
            &mut file,
        )
        .expect("freeze");
        assert_eq!(stats.regions_frozen, 20);

        let parsed = thaw_core::Snapshot::from_prelude_bytes(&file).expect("parse");
        for i in 0..20 {
            let entry = parsed.table().get(i).unwrap();
            let start = entry.file_offset() as usize;
            let end = start + entry.size() as usize;
            assert_eq!(
                &file[start..end],
                expected[i].as_slice(),
                "region {i} mismatch"
            );
        }
    }

    /// Empty request list produces prelude-only file.
    #[test]
    fn pipelined_freeze_empty() {
        let backend = MockCuda::new();
        let mut file: Vec<u8> = Vec::new();
        let stats = freeze_pipelined(
            &backend,
            &[],
            &FreezeConfig::default(),
            &mut file,
        )
        .expect("freeze");
        assert_eq!(stats.regions_frozen, 0);
        assert_eq!(stats.bytes_copied, 0);
        let parsed = thaw_core::Snapshot::from_prelude_bytes(&file).expect("parse");
        assert_eq!(parsed.len(), 0);
    }

    /// Single region (degenerate case — no double-buffering needed).
    #[test]
    fn pipelined_freeze_single_region() {
        let backend = MockCuda::new();
        let ptr = DevicePtr(0x7000);
        let data: Vec<u8> = (0..256).map(|i| (i & 0xFF) as u8).collect();
        backend.register_region(ptr, data.clone());

        let requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(ptr, data.len() as u64),
        )];

        let mut file: Vec<u8> = Vec::new();
        freeze_pipelined(
            &backend,
            &requests,
            &FreezeConfig::default(),
            &mut file,
        )
        .expect("freeze");

        let parsed = thaw_core::Snapshot::from_prelude_bytes(&file).expect("parse");
        let entry = parsed.table().get(0).unwrap();
        let start = entry.file_offset() as usize;
        assert_eq!(&file[start..start + data.len()], data.as_slice());
    }

    // =================================================================
    // freeze_pipelined_to_file tests — O_DIRECT path
    // =================================================================

    /// freeze_pipelined_to_file writes the same bytes as the
    /// writer-based freeze_pipelined. Proves the O_DIRECT path
    /// produces a format-identical file.
    #[test]
    fn freeze_pipelined_to_file_matches_writer_bytes() {
        let backend = MockCuda::new();
        // Non-overlapping device ranges (weights+metadata = 8448 < 0x10000).
        let w_ptr = DevicePtr(0x1000);
        let m_ptr = DevicePtr(0x20_000);
        let weights: Vec<u8> = (0..8192).map(|i| ((i * 11) & 0xFF) as u8).collect();
        let metadata: Vec<u8> = (0..256).map(|i| (0x5A ^ i) as u8).collect();
        backend.register_region(w_ptr, weights.clone());
        backend.register_region(m_ptr, metadata.clone());

        let requests = vec![
            FreezeRequest::new(
                RegionKind::Weights,
                0,
                DeviceRegion::new(w_ptr, weights.len() as u64),
            ),
            FreezeRequest::new(
                RegionKind::Metadata,
                0,
                DeviceRegion::new(m_ptr, metadata.len() as u64),
            ),
        ];

        // Small chunk_size forces the double-buffer loop to iterate.
        let config = FreezeConfig {
            chunk_size: 4096,
            try_direct_io: false, // macOS: no O_DIRECT; test the buffered pwrite path
            ..FreezeConfig::default()
        };

        // Writer-based reference bytes.
        let mut ref_bytes: Vec<u8> = Vec::new();
        freeze_pipelined(&backend, &requests, &config, &mut ref_bytes).expect("ref freeze");

        // File-based path writes to a temp file.
        let tmp = NamedTempFile::new().expect("tempfile");
        let stats = freeze_pipelined_to_file(&backend, &requests, &config, tmp.path())
            .expect("freeze_to_file");
        assert_eq!(stats.regions_frozen, 2);
        assert_eq!(stats.bytes_copied, (weights.len() + metadata.len()) as u64);

        let file_bytes = std::fs::read(tmp.path()).expect("read temp");
        assert_eq!(file_bytes.len(), ref_bytes.len(), "file size mismatch");
        assert_eq!(file_bytes, ref_bytes, "file bytes differ from writer path");
    }

    /// freeze_pipelined_to_file output round-trips through
    /// restore_pipelined. Proves the whole freeze → disk → restore
    /// pipeline end-to-end.
    #[test]
    fn freeze_to_file_then_restore_pipelined_round_trips() {
        let src = MockCuda::new();
        // Non-overlapping device ranges: weights occupy [0x1000, 0x1000+20000].
        // Metadata sits well past the weights tail to avoid the mock's
        // address-range-based region resolver from matching the wrong one.
        let w_ptr = DevicePtr(0x1000);
        let m_ptr = DevicePtr(0x100_000);
        let weights: Vec<u8> = (0..20_000).map(|i| ((i * 3) & 0xFF) as u8).collect();
        let metadata: Vec<u8> = (0..512).map(|i| (0xAB ^ i) as u8).collect();
        src.register_region(w_ptr, weights.clone());
        src.register_region(m_ptr, metadata.clone());

        let requests = vec![
            FreezeRequest::new(
                RegionKind::Weights,
                0,
                DeviceRegion::new(w_ptr, weights.len() as u64),
            ),
            FreezeRequest::new(
                RegionKind::Metadata,
                0,
                DeviceRegion::new(m_ptr, metadata.len() as u64),
            ),
        ];

        // 8 KiB chunks — multiple of 4096, smaller than weights.
        let freeze_config = FreezeConfig {
            chunk_size: 8192,
            try_direct_io: false,
            ..FreezeConfig::default()
        };

        let tmp = NamedTempFile::new().expect("tempfile");
        freeze_pipelined_to_file(&src, &requests, &freeze_config, tmp.path())
            .expect("freeze_to_file");

        // Restore into a fresh backend via the existing pipelined path.
        // Use non-overlapping device ranges on the restore side too.
        let dst = MockCuda::new();
        let w_ptr2 = DevicePtr(0x10_000);
        let m_ptr2 = DevicePtr(0x200_000);
        dst.register_region(w_ptr2, vec![0u8; weights.len()]);
        dst.register_region(m_ptr2, vec![0u8; metadata.len()]);

        let restore_config = PipelineConfig {
            chunk_size: 4096,
            try_direct_io: false,
        };
        let stats = restore_pipelined(
            &dst,
            tmp.path(),
            |kind, _| match kind {
                RegionKind::Weights => {
                    Some(DeviceRegion::new(w_ptr2, weights.len() as u64))
                }
                RegionKind::Metadata => {
                    Some(DeviceRegion::new(m_ptr2, metadata.len() as u64))
                }
                _ => None,
            },
            &restore_config,
        )
        .expect("restore");

        assert_eq!(stats.regions_restored, 2);
        assert_eq!(
            stats.bytes_copied,
            (weights.len() + metadata.len()) as u64
        );
        assert_eq!(dst.read_region(w_ptr2).unwrap(), weights);
        assert_eq!(dst.read_region(m_ptr2).unwrap(), metadata);
    }

    /// Many regions through the file-based path, each spanning
    /// multiple chunks. Stresses the multi-region-per-chunk
    /// packing logic on the freeze side.
    #[test]
    fn freeze_to_file_many_regions_multi_chunk() {
        let backend = MockCuda::new();
        let mut requests = Vec::new();
        let mut expected: Vec<(DevicePtr, Vec<u8>)> = Vec::new();

        // 50 regions of varying sizes that straddle multiple chunks.
        for i in 0..50u32 {
            let ptr = DevicePtr(0x1000 + i as u64 * 0x2000);
            let size = 500 + (i as usize) * 37; // 500..2313 bytes, all different
            let data: Vec<u8> = (0..size).map(|b| ((b + i as usize) & 0xFF) as u8).collect();
            backend.register_region(ptr, data.clone());
            requests.push(FreezeRequest::new(
                RegionKind::Weights,
                i,
                DeviceRegion::new(ptr, size as u64),
            ));
            expected.push((ptr, data));
        }

        let config = FreezeConfig {
            chunk_size: 4096,
            try_direct_io: false,
            ..FreezeConfig::default()
        };

        let tmp = NamedTempFile::new().expect("tempfile");
        let stats = freeze_pipelined_to_file(&backend, &requests, &config, tmp.path())
            .expect("freeze");
        assert_eq!(stats.regions_frozen, 50);

        // Restore and verify.
        let dst = MockCuda::new();
        for (ptr, data) in &expected {
            let dst_ptr = DevicePtr(0x100000 + ptr.0);
            dst.register_region(dst_ptr, vec![0u8; data.len()]);
        }
        let restore_config = PipelineConfig {
            chunk_size: 4096,
            try_direct_io: false,
        };
        restore_pipelined(
            &dst,
            tmp.path(),
            |_, logical_id| {
                let idx = logical_id as usize;
                let (src_ptr, data) = &expected[idx];
                Some(DeviceRegion::new(
                    DevicePtr(0x100000 + src_ptr.0),
                    data.len() as u64,
                ))
            },
            &restore_config,
        )
        .expect("restore");

        for (src_ptr, data) in &expected {
            let dst_ptr = DevicePtr(0x100000 + src_ptr.0);
            assert_eq!(&dst.read_region(dst_ptr).unwrap()[..], data.as_slice());
        }
    }

    /// After a successful freeze, no `.thaw-incomplete` file
    /// remains next to the final path — the atomic rename and the
    /// RAII guard both have to agree the write succeeded.
    #[test]
    fn freeze_to_file_leaves_no_tmp_on_success() {
        let backend = MockCuda::new();
        let ptr = DevicePtr(0x1000);
        let bytes: Vec<u8> = (0..4096).map(|i| (i & 0xFF) as u8).collect();
        backend.register_region(ptr, bytes.clone());
        let requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(ptr, bytes.len() as u64),
        )];
        let config = FreezeConfig {
            chunk_size: 4096,
            try_direct_io: false,
            ..FreezeConfig::default()
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("snap.thaw");
        freeze_pipelined_to_file(&backend, &requests, &config, &path).expect("freeze");

        assert!(path.exists(), "final file should exist");
        let tmp = super::tmp_path_for(&path);
        assert!(!tmp.exists(), "tmp file should have been renamed away");
    }

    /// If the freeze fails mid-pipeline (here: unknown device ptr),
    /// the tmp file is unlinked by the RAII guard and any
    /// pre-existing file at the final path is untouched. No
    /// half-written `.thaw` can leak out to confuse `thaw restore`.
    #[test]
    fn freeze_to_file_cleans_tmp_on_error_and_preserves_final() {
        let backend = MockCuda::new();
        // A request pointing at a DevicePtr that was never registered
        // triggers UnknownDevicePtr during D2H, after the tmp file
        // is already open.
        let bad_requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(DevicePtr(0xDEAD), 8192),
        )];
        let config = FreezeConfig {
            chunk_size: 4096,
            try_direct_io: false,
            ..FreezeConfig::default()
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("snap.thaw");
        // Seed a pre-existing "old snapshot" at the final path.
        std::fs::write(&path, b"OLD_SNAPSHOT_DO_NOT_TOUCH").expect("seed");

        let err = freeze_pipelined_to_file(&backend, &bad_requests, &config, &path)
            .expect_err("should fail");
        match err {
            FreezeError::Backend(_) => {}
            other => panic!("expected Backend error, got {other:?}"),
        }

        let tmp = super::tmp_path_for(&path);
        assert!(!tmp.exists(), "tmp file must be cleaned up on error");
        assert_eq!(
            std::fs::read(&path).expect("read final"),
            b"OLD_SNAPSHOT_DO_NOT_TOUCH",
            "pre-existing file must not be touched on failure"
        );
    }

    /// Empty request list via file path: prelude-only file.
    #[test]
    fn freeze_to_file_empty() {
        let backend = MockCuda::new();
        let config = FreezeConfig {
            try_direct_io: false,
            ..FreezeConfig::default()
        };
        let tmp = NamedTempFile::new().expect("tempfile");
        let stats = freeze_pipelined_to_file(&backend, &[], &config, tmp.path())
            .expect("freeze");
        assert_eq!(stats.regions_frozen, 0);

        let bytes = std::fs::read(tmp.path()).expect("read");
        let parsed = thaw_core::Snapshot::from_prelude_bytes(&bytes).expect("parse");
        assert_eq!(parsed.len(), 0);
    }

    // =================================================================
    // restore_pipelined_from_bytes tests
    // =================================================================

    /// Helper: freeze to in-memory bytes (no tempfile needed).
    fn freeze_to_bytes(
        backend: &MockCuda,
        requests: &[FreezeRequest],
    ) -> Vec<u8> {
        let mut file: Vec<u8> = Vec::new();
        freeze(backend, requests, &FreezeConfig::default(), &mut file)
            .expect("freeze");
        file
    }

    /// Basic round-trip: freeze to bytes, restore_from_bytes.
    #[test]
    fn from_bytes_round_trips() {
        let src = MockCuda::new();
        let w_ptr = DevicePtr(0x1000);
        let m_ptr = DevicePtr(0x2000);
        let weights: Vec<u8> = (0..512).map(|i| ((i * 3) & 0xFF) as u8).collect();
        let metadata: Vec<u8> = (0..64).map(|i| (0xF0 ^ i) as u8).collect();
        src.register_region(w_ptr, weights.clone());
        src.register_region(m_ptr, metadata.clone());

        let requests = vec![
            FreezeRequest::new(
                RegionKind::Weights,
                0,
                DeviceRegion::new(w_ptr, weights.len() as u64),
            ),
            FreezeRequest::new(
                RegionKind::Metadata,
                0,
                DeviceRegion::new(m_ptr, metadata.len() as u64),
            ),
        ];
        let data = freeze_to_bytes(&src, &requests);

        let dst = MockCuda::new();
        let w_ptr2 = DevicePtr(0xA000);
        let m_ptr2 = DevicePtr(0xB000);
        dst.register_region(w_ptr2, vec![0u8; weights.len()]);
        dst.register_region(m_ptr2, vec![0u8; metadata.len()]);

        let config = PipelineConfig {
            chunk_size: 4096,
            try_direct_io: false,
        };
        let stats = restore_pipelined_from_bytes(
            &dst,
            &data,
            |kind, _| match kind {
                RegionKind::Weights => {
                    Some(DeviceRegion::new(w_ptr2, weights.len() as u64))
                }
                RegionKind::Metadata => {
                    Some(DeviceRegion::new(m_ptr2, metadata.len() as u64))
                }
                _ => None,
            },
            &config,
        )
        .expect("restore");

        assert_eq!(stats.regions_restored, 2);
        assert_eq!(
            stats.bytes_copied,
            (weights.len() + metadata.len()) as u64
        );
        assert_eq!(dst.read_region(w_ptr2).unwrap(), weights);
        assert_eq!(dst.read_region(m_ptr2).unwrap(), metadata);
    }

    /// Large region spanning multiple chunks (from bytes).
    #[test]
    fn from_bytes_large_region_spanning_chunks() {
        let src = MockCuda::new();
        let ptr = DevicePtr(0x1000);
        let data_region: Vec<u8> = (0..10000).map(|i| ((i * 7) & 0xFF) as u8).collect();
        src.register_region(ptr, data_region.clone());

        let requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(ptr, data_region.len() as u64),
        )];
        let file_bytes = freeze_to_bytes(&src, &requests);

        let dst = MockCuda::new();
        let dst_ptr = DevicePtr(0xF000);
        dst.register_region(dst_ptr, vec![0u8; data_region.len()]);

        let config = PipelineConfig {
            chunk_size: 4096,
            try_direct_io: false,
        };
        let stats = restore_pipelined_from_bytes(
            &dst,
            &file_bytes,
            |_, _| Some(DeviceRegion::new(dst_ptr, data_region.len() as u64)),
            &config,
        )
        .expect("restore");

        assert_eq!(stats.regions_restored, 1);
        assert_eq!(stats.bytes_copied, data_region.len() as u64);
        assert_eq!(dst.read_region(dst_ptr).unwrap(), data_region);
    }

    /// Freeze pipelined then restore from bytes: full pipeline round-trip.
    #[test]
    fn pipelined_freeze_then_restore_from_bytes() {
        let src = MockCuda::new();
        let ptr = DevicePtr(0x5000);
        let data_region: Vec<u8> = (0..4096).map(|i| ((i * 7) & 0xFF) as u8).collect();
        src.register_region(ptr, data_region.clone());

        let requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(ptr, data_region.len() as u64),
        )];

        let mut file_bytes: Vec<u8> = Vec::new();
        freeze_pipelined(
            &src,
            &requests,
            &FreezeConfig::default(),
            &mut file_bytes,
        )
        .expect("freeze");

        let dst = MockCuda::new();
        let dst_ptr = DevicePtr(0xA000);
        dst.register_region(dst_ptr, vec![0u8; data_region.len()]);

        let config = PipelineConfig {
            chunk_size: 4096,
            try_direct_io: false,
        };
        let stats = restore_pipelined_from_bytes(
            &dst,
            &file_bytes,
            |_, _| Some(DeviceRegion::new(dst_ptr, data_region.len() as u64)),
            &config,
        )
        .expect("restore");

        assert_eq!(dst.read_region(dst_ptr).unwrap(), data_region);
        assert_eq!(stats.regions_restored, 1);
    }

    /// Payload tampering after freeze is caught by per-region
    /// CRC32C before any bytes hit the device. Mirrors the
    /// `restore::restore_detects_single_byte_payload_corruption`
    /// test but against the pipelined-from-bytes path, so a future
    /// refactor that skips the fold-and-verify call breaks a test.
    #[test]
    fn pipelined_from_bytes_detects_payload_corruption() {
        let src = MockCuda::new();
        let ptr = DevicePtr(0x5000);
        let data_region: Vec<u8> = (0..4096).map(|i| ((i * 11) & 0xFF) as u8).collect();
        src.register_region(ptr, data_region.clone());

        let requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(ptr, data_region.len() as u64),
        )];
        let mut file_bytes: Vec<u8> = Vec::new();
        freeze_pipelined(
            &src,
            &requests,
            &FreezeConfig::default(),
            &mut file_bytes,
        )
        .expect("freeze");

        // Flip a byte deep in the payload, past the prelude (header +
        // 32-byte region-table entry), so we tamper with data bytes,
        // not table bytes.
        let pos = file_bytes.len() - 100;
        file_bytes[pos] ^= 0xFF;

        let dst = MockCuda::new();
        let dst_ptr = DevicePtr(0xA000);
        dst.register_region(dst_ptr, vec![0u8; data_region.len()]);

        let config = PipelineConfig {
            chunk_size: 4096,
            try_direct_io: false,
        };
        let err = restore_pipelined_from_bytes(
            &dst,
            &file_bytes,
            |_, _| Some(DeviceRegion::new(dst_ptr, data_region.len() as u64)),
            &config,
        )
        .expect_err("tampered payload must fail verify");

        match err {
            RestoreError::ChecksumMismatch { kind, .. } => {
                assert_eq!(kind, RegionKind::Weights);
            }
            other => panic!("expected ChecksumMismatch, got {:?}", other),
        }

        // Device memory must be untouched — verify runs before DMA.
        assert_eq!(dst.read_region(dst_ptr).unwrap(), vec![0u8; data_region.len()]);
    }

    /// Zero-copy path: tampered payload must fail verify before any
    /// DMA fires. Uses the mock's `host_register`-noop path so the
    /// exact function under test is `run_pre_registered_plan`.
    #[test]
    fn pipelined_from_registered_bytes_detects_payload_corruption() {
        let src = MockCuda::new();
        let ptr = DevicePtr(0x6000);
        let data_region: Vec<u8> = (0..2048).map(|i| ((i * 13) & 0xFF) as u8).collect();
        src.register_region(ptr, data_region.clone());

        let requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(ptr, data_region.len() as u64),
        )];
        let mut file_bytes: Vec<u8> = Vec::new();
        freeze_pipelined(
            &src,
            &requests,
            &FreezeConfig::default(),
            &mut file_bytes,
        )
        .expect("freeze");

        // Corrupt a byte deep in the payload, past the prelude.
        let pos = file_bytes.len() - 50;
        file_bytes[pos] ^= 0x5A;

        let dst = MockCuda::new();
        let dst_ptr = DevicePtr(0xB000);
        dst.register_region(dst_ptr, vec![0u8; data_region.len()]);

        let config = PipelineConfig {
            chunk_size: 4096,
            try_direct_io: false,
        };
        let err = restore_pipelined_from_registered_bytes(
            &dst,
            &file_bytes,
            |_, _| Some(DeviceRegion::new(dst_ptr, data_region.len() as u64)),
            &config,
        )
        .expect_err("tampered payload must fail verify on zero-copy path");

        match err {
            RestoreError::ChecksumMismatch { kind, .. } => {
                assert_eq!(kind, RegionKind::Weights);
            }
            other => panic!("expected ChecksumMismatch, got {:?}", other),
        }

        assert_eq!(dst.read_region(dst_ptr).unwrap(), vec![0u8; data_region.len()]);
    }

    /// Empty data restores with zero stats (from bytes).
    #[test]
    fn from_bytes_empty() {
        let src = MockCuda::new();
        let file_bytes = freeze_to_bytes(&src, &[]);

        let dst = MockCuda::new();
        let config = PipelineConfig {
            chunk_size: 4096,
            try_direct_io: false,
        };
        let stats = restore_pipelined_from_bytes(
            &dst,
            &file_bytes,
            |_, _| panic!("should not be called"),
            &config,
        )
        .expect("restore");

        assert_eq!(stats.regions_restored, 0);
        assert_eq!(stats.bytes_copied, 0);
    }

    /// Zero-copy round-trip: freeze to bytes, restore via the
    /// host-registered pipeline. The mock's `host_register` is a
    /// no-op and `memcpy_h2d_async_raw` snapshots the source bytes
    /// just like the pinned-buffer variant, so a correct
    /// orchestration against the trait must byte-exactly restore
    /// the frozen regions.
    #[test]
    fn from_registered_bytes_round_trips() {
        let src = MockCuda::new();
        let w_ptr = DevicePtr(0x1000);
        let m_ptr = DevicePtr(0x2000);
        let weights: Vec<u8> = (0..512).map(|i| ((i * 3) & 0xFF) as u8).collect();
        let metadata: Vec<u8> = (0..64).map(|i| (0xF0 ^ i) as u8).collect();
        src.register_region(w_ptr, weights.clone());
        src.register_region(m_ptr, metadata.clone());

        let requests = vec![
            FreezeRequest::new(
                RegionKind::Weights,
                0,
                DeviceRegion::new(w_ptr, weights.len() as u64),
            ),
            FreezeRequest::new(
                RegionKind::Metadata,
                0,
                DeviceRegion::new(m_ptr, metadata.len() as u64),
            ),
        ];
        let data = freeze_to_bytes(&src, &requests);

        let dst = MockCuda::new();
        let w_ptr2 = DevicePtr(0xA000);
        let m_ptr2 = DevicePtr(0xB000);
        dst.register_region(w_ptr2, vec![0u8; weights.len()]);
        dst.register_region(m_ptr2, vec![0u8; metadata.len()]);

        let config = PipelineConfig::default();
        let stats = restore_pipelined_from_registered_bytes(
            &dst,
            &data,
            |kind, _| match kind {
                RegionKind::Weights => {
                    Some(DeviceRegion::new(w_ptr2, weights.len() as u64))
                }
                RegionKind::Metadata => {
                    Some(DeviceRegion::new(m_ptr2, metadata.len() as u64))
                }
                _ => None,
            },
            &config,
        )
        .expect("restore");

        assert_eq!(stats.regions_restored, 2);
        assert_eq!(
            stats.bytes_copied,
            (weights.len() + metadata.len()) as u64
        );
        assert_eq!(dst.read_region(w_ptr2).unwrap(), weights);
        assert_eq!(dst.read_region(m_ptr2).unwrap(), metadata);
    }

    /// Empty input is a no-op on the zero-copy path too. Verifies
    /// that `host_register(.., 0)` and the empty-plan short-circuit
    /// do not attempt any FFI work.
    #[test]
    fn from_registered_bytes_empty() {
        let src = MockCuda::new();
        let data = freeze_to_bytes(&src, &[]);

        let dst = MockCuda::new();
        let stats = restore_pipelined_from_registered_bytes(
            &dst,
            &data,
            |_, _| panic!("should not be called"),
            &PipelineConfig::default(),
        )
        .expect("restore");

        assert_eq!(stats.regions_restored, 0);
        assert_eq!(stats.bytes_copied, 0);
    }

    /// The pre-registered variant must be call-repeatable against the
    /// same input buffer: two back-to-back restores produce identical
    /// bytes and stats. This is the slot-warm-up → N-restores pattern
    /// that `thaw serve` relies on.
    #[test]
    fn from_pre_registered_bytes_reusable() {
        let src = MockCuda::new();
        let w_ptr = DevicePtr(0x1000);
        let weights: Vec<u8> = (0..256).map(|i| ((i * 7) & 0xFF) as u8).collect();
        src.register_region(w_ptr, weights.clone());

        let requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(w_ptr, weights.len() as u64),
        )];
        let data = freeze_to_bytes(&src, &requests);

        let dst = MockCuda::new();
        let w_ptr2 = DevicePtr(0xA000);
        dst.register_region(w_ptr2, vec![0u8; weights.len()]);

        let config = PipelineConfig::default();

        // First restore: same as the registering variant would do.
        let stats1 = restore_pipelined_from_pre_registered_bytes(
            &dst,
            &data,
            |kind, _| match kind {
                RegionKind::Weights => {
                    Some(DeviceRegion::new(w_ptr2, weights.len() as u64))
                }
                _ => None,
            },
            &config,
        )
        .expect("first restore");

        assert_eq!(stats1.regions_restored, 1);
        assert_eq!(stats1.bytes_copied, weights.len() as u64);
        assert_eq!(dst.read_region(w_ptr2).unwrap(), weights);

        // Zero the destination and restore a second time from the same
        // buffer — no registration cost, same result. This is the
        // slot-reuse case.
        dst.register_region(w_ptr2, vec![0u8; weights.len()]);
        let stats2 = restore_pipelined_from_pre_registered_bytes(
            &dst,
            &data,
            |kind, _| match kind {
                RegionKind::Weights => {
                    Some(DeviceRegion::new(w_ptr2, weights.len() as u64))
                }
                _ => None,
            },
            &config,
        )
        .expect("second restore");

        assert_eq!(stats2.regions_restored, 1);
        assert_eq!(stats2.bytes_copied, weights.len() as u64);
        assert_eq!(dst.read_region(w_ptr2).unwrap(), weights);
    }

    /// Unified auto-dispatch path: the mock's `host_register` always
    /// succeeds, so the auto function must take the zero-copy branch
    /// and byte-exactly round-trip. This is the common-case happy
    /// path — it just has to work against the mock, same as the
    /// hand-written zero-copy variant.
    #[test]
    fn from_bytes_auto_round_trips() {
        let src = MockCuda::new();
        let w_ptr = DevicePtr(0x1000);
        let m_ptr = DevicePtr(0x2000);
        let weights: Vec<u8> = (0..512).map(|i| ((i * 3) & 0xFF) as u8).collect();
        let metadata: Vec<u8> = (0..64).map(|i| (0xF0 ^ i) as u8).collect();
        src.register_region(w_ptr, weights.clone());
        src.register_region(m_ptr, metadata.clone());

        let requests = vec![
            FreezeRequest::new(
                RegionKind::Weights,
                0,
                DeviceRegion::new(w_ptr, weights.len() as u64),
            ),
            FreezeRequest::new(
                RegionKind::Metadata,
                0,
                DeviceRegion::new(m_ptr, metadata.len() as u64),
            ),
        ];
        let data = freeze_to_bytes(&src, &requests);

        let dst = MockCuda::new();
        let w_ptr2 = DevicePtr(0xA000);
        let m_ptr2 = DevicePtr(0xB000);
        dst.register_region(w_ptr2, vec![0u8; weights.len()]);
        dst.register_region(m_ptr2, vec![0u8; metadata.len()]);

        let config = PipelineConfig::default();
        let stats = restore_pipelined_from_bytes_auto(
            &dst,
            &data,
            |kind, _| match kind {
                RegionKind::Weights => {
                    Some(DeviceRegion::new(w_ptr2, weights.len() as u64))
                }
                RegionKind::Metadata => {
                    Some(DeviceRegion::new(m_ptr2, metadata.len() as u64))
                }
                _ => None,
            },
            &config,
        )
        .expect("restore");

        assert_eq!(stats.regions_restored, 2);
        assert_eq!(
            stats.bytes_copied,
            (weights.len() + metadata.len()) as u64
        );
        assert_eq!(dst.read_region(w_ptr2).unwrap(), weights);
        assert_eq!(dst.read_region(m_ptr2).unwrap(), metadata);
    }

    /// Empty input is a no-op on the unified path too.
    #[test]
    fn from_bytes_auto_empty() {
        let src = MockCuda::new();
        let data = freeze_to_bytes(&src, &[]);

        let dst = MockCuda::new();
        let stats = restore_pipelined_from_bytes_auto(
            &dst,
            &data,
            |_, _| panic!("should not be called"),
            &PipelineConfig::default(),
        )
        .expect("restore");

        assert_eq!(stats.regions_restored, 0);
        assert_eq!(stats.bytes_copied, 0);
    }

    /// Zero-region file restores with zero stats.
    #[test]
    fn zero_region_file() {
        let src = MockCuda::new();
        let tmp = freeze_to_tempfile(&src, &[]);

        let dst = MockCuda::new();
        let config = PipelineConfig {
            chunk_size: 4096,
            try_direct_io: false,
        };
        let stats = restore_pipelined(
            &dst,
            tmp.path(),
            |_, _| panic!("should not be called"),
            &config,
        )
        .expect("restore");

        assert_eq!(stats.regions_restored, 0);
        assert_eq!(stats.bytes_copied, 0);
    }

    // =========================================================================
    // WC BUFFER CACHE TESTS
    // =========================================================================

    /// Peek at the current thread's cached chunk_size for the freeze
    /// path, if any. Test-only helper that touches the module-private
    /// thread_local. Freeze now routes through PINNED_BUF_CACHE (since
    /// 2de24bf); WC_BUF_CACHE is only populated by
    /// `restore_pipelined_from_bytes` these days.
    fn cached_chunk_size() -> Option<usize> {
        super::PINNED_BUF_CACHE.with(|c| c.borrow().as_ref().map(|x| x.chunk_size))
    }

    /// After a successful pipelined freeze, the cache holds a pair sized
    /// to that call's chunk_size. Calling again with the same chunk_size
    /// leaves the cache still primed — proving `release_wc_bufs` put
    /// buffers back and the next `acquire_wc_bufs` found them.
    #[test]
    fn wc_cache_primes_and_stays_primed_across_matching_calls() {
        clear_wc_buf_cache();
        assert_eq!(cached_chunk_size(), None);

        let backend = MockCuda::new();
        let w_ptr = DevicePtr(0x1_000);
        let weights = vec![7u8; 4096];
        backend.register_region(w_ptr, weights.clone());

        let requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(w_ptr, weights.len() as u64),
        )];
        let cfg = FreezeConfig::default();

        let tmp1 = NamedTempFile::new().expect("tmp1");
        freeze_pipelined_to_file(&backend, &requests, &cfg, tmp1.path())
            .expect("freeze 1");
        assert_eq!(cached_chunk_size(), Some(cfg.chunk_size));

        let tmp2 = NamedTempFile::new().expect("tmp2");
        freeze_pipelined_to_file(&backend, &requests, &cfg, tmp2.path())
            .expect("freeze 2");
        // Cache still primed at same size — reuse path exercised.
        assert_eq!(cached_chunk_size(), Some(cfg.chunk_size));
    }

    /// Calling with a different chunk_size drops the stale pair and
    /// replaces the cache entry with the new size. Prior contents are
    /// freed — the old pair doesn't leak.
    #[test]
    fn wc_cache_updates_on_chunk_size_change() {
        clear_wc_buf_cache();

        let backend = MockCuda::new();
        let w_ptr = DevicePtr(0x1_000);
        let weights = vec![3u8; 4096];
        backend.register_region(w_ptr, weights.clone());

        let requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(w_ptr, weights.len() as u64),
        )];

        // First call: default FreezeConfig chunk_size.
        let tmp1 = NamedTempFile::new().expect("tmp1");
        let cfg_default = FreezeConfig::default();
        freeze_pipelined_to_file(&backend, &requests, &cfg_default, tmp1.path())
            .expect("freeze default");
        assert_eq!(cached_chunk_size(), Some(cfg_default.chunk_size));

        // Second call: different chunk_size → cache entry is replaced.
        let cfg_small = FreezeConfig {
            chunk_size: 4096 * 16, // 64 KiB, multiple of 4096 for O_DIRECT
            ..FreezeConfig::default()
        };
        let tmp2 = NamedTempFile::new().expect("tmp2");
        freeze_pipelined_to_file(&backend, &requests, &cfg_small, tmp2.path())
            .expect("freeze small");
        assert_eq!(cached_chunk_size(), Some(cfg_small.chunk_size));
    }

    /// `clear_wc_buf_cache` empties this thread's cache so the next
    /// pipelined call allocates from scratch.
    #[test]
    fn wc_cache_clear_empties_cache() {
        clear_wc_buf_cache();

        let backend = MockCuda::new();
        let w_ptr = DevicePtr(0x1_000);
        let weights = vec![5u8; 4096];
        backend.register_region(w_ptr, weights.clone());

        let requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(w_ptr, weights.len() as u64),
        )];
        let cfg = FreezeConfig::default();
        let tmp = NamedTempFile::new().expect("tmp");
        freeze_pipelined_to_file(&backend, &requests, &cfg, tmp.path()).expect("freeze");
        assert!(cached_chunk_size().is_some());

        clear_wc_buf_cache();
        assert_eq!(cached_chunk_size(), None);
    }

    /// Invalid freeze config surfaces as `InvalidConfig`, not a panic.
    /// The CLI and Python layer need a recoverable error here; an
    /// assert would unwind the process with no context.
    #[test]
    fn freeze_rejects_invalid_chunk_size() {
        let backend = MockCuda::new();
        let w_ptr = DevicePtr(0x1_000);
        backend.register_region(w_ptr, vec![0u8; 1024]);
        let requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(w_ptr, 1024),
        )];

        let bad = FreezeConfig {
            chunk_size: 0,
            ..FreezeConfig::default()
        };
        let mut sink: Vec<u8> = Vec::new();
        let err = freeze_pipelined(&backend, &requests, &bad, &mut sink)
            .expect_err("chunk_size=0 must fail");
        assert!(matches!(err, FreezeError::InvalidConfig { .. }), "got {err:?}");

        let bad_align = FreezeConfig {
            chunk_size: 1234,
            ..FreezeConfig::default()
        };
        let tmp = NamedTempFile::new().expect("tmp");
        let err = freeze_pipelined_to_file(&backend, &requests, &bad_align, tmp.path())
            .expect_err("non-4KiB chunk_size must fail on O_DIRECT path");
        assert!(matches!(err, FreezeError::InvalidConfig { .. }), "got {err:?}");
    }

    /// Invalid restore config surfaces as `InvalidConfig`, not a panic.
    /// With a non-empty plan the validation path is reached before the
    /// `.first().unwrap()` payload math, so the error is the config one.
    #[test]
    fn restore_rejects_invalid_chunk_size() {
        let backend = MockCuda::new();
        let w_ptr = DevicePtr(0x1_000);
        backend.register_region(w_ptr, vec![1u8; 1024]);
        let reqs = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(w_ptr, 1024),
        )];
        let file_bytes = freeze_to_bytes(&backend, &reqs);

        let dst = MockCuda::new();
        let dst_ptr = DevicePtr(0x2_000);
        dst.register_region(dst_ptr, vec![0u8; 1024]);

        let cfg = PipelineConfig {
            chunk_size: 1234,
            try_direct_io: false,
        };
        let err = restore_pipelined_from_bytes(
            &dst,
            &file_bytes,
            |_, _| Some(DeviceRegion::new(dst_ptr, 1024)),
            &cfg,
        )
        .expect_err("non-4KiB chunk_size must fail");
        assert!(matches!(err, RestoreError::InvalidConfig { .. }), "got {err:?}");
    }
}
