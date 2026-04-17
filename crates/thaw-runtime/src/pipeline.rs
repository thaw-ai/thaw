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
    open_direct, open_direct_write, pread_exact, pwrite_exact, truncate,
};
use crate::freeze::{FreezeConfig, FreezeError, FreezeRequest};
use crate::restore::{RestoreError, RestoreStats};

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
    static WC_BUF_CACHE: RefCell<Option<WcBufCache>> = const { RefCell::new(None) };
}

struct WcBufCache {
    chunk_size: usize,
    bufs: [crate::backend::PinnedBuffer; 2],
}

fn wc_cache_disabled() -> bool {
    std::env::var("THAW_DISABLE_WC_CACHE")
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

    // Phase 1: write prelude directly. For the writer-based path we
    // don't need to carry the prelude in the chunk 0 buffer — we can
    // just write it and then write payload chunks in order.
    sink.write_all(&prelude_bytes)
        .map_err(|e| FreezeError::Snapshot(SnapshotError::from(e)))?;

    if requests.is_empty() {
        return Ok(FreezeStats::default());
    }

    // Phase 2: chunked double-buffered D2H + write.
    let chunk_size = config.chunk_size;
    assert!(chunk_size > 0, "chunk_size must be > 0");

    let payload_start = prelude_bytes.len() as u64;
    let payload_end = total_file_size;
    let payload_len = (payload_end - payload_start) as usize;
    if payload_len == 0 {
        return Ok(FreezeStats {
            regions_frozen: requests.len(),
            bytes_copied: 0,
        });
    }
    let num_chunks = (payload_len + chunk_size - 1) / chunk_size;

    let total_bytes: u64 = plan.iter().map(|p| p.size).sum();

    // Two pinned buffers (write-combining for faster D2H) + two streams.
    // Buffers are acquired from a per-thread cache — subsequent calls
    // with the same chunk_size skip the `cudaHostAlloc` cost entirely.
    let mut bufs = acquire_wc_bufs(backend, chunk_size)?;
    let streams = [backend.stream_create()?, backend.stream_create()?];

    // Helper: compute this chunk's [start, end) in file coordinates.
    let chunk_range = |idx: usize| -> (u64, u64) {
        let start = payload_start + (idx as u64) * (chunk_size as u64);
        let end = (start + chunk_size as u64).min(payload_end);
        (start, end)
    };

    // Helper: write `len` bytes from the buffer into the sink.
    let write_chunk_to_sink = |buf: &crate::backend::PinnedBuffer,
                               len: usize,
                               sink: &mut W|
     -> Result<(), FreezeError> {
        sink.write_all(&buf.as_slice()[..len])
            .map_err(|e| FreezeError::Snapshot(SnapshotError::from(e)))?;
        Ok(())
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
        write_chunk_to_sink(&bufs[0], len, sink)?;
    } else {
        // Steady state: D2H(chunk i) overlaps write(chunk i-1).
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

            // While D2H is in flight on streams[curr], write the
            // previous chunk to the sink.
            write_chunk_to_sink(&bufs[prev], prev_len, sink)?;

            // Wait for the current chunk's D2H to complete before
            // the next iteration reuses its buffer.
            backend.stream_sync(&streams[curr])?;
        }

        // Write the final chunk.
        let last = (num_chunks - 1) % 2;
        let (last_start, last_end) = chunk_range(num_chunks - 1);
        let last_len = (last_end - last_start) as usize;
        write_chunk_to_sink(&bufs[last], last_len, sink)?;
    }

    // Cleanup streams.
    backend
        .stream_destroy(streams[0])
        .map_err(FreezeError::Backend)?;
    backend
        .stream_destroy(streams[1])
        .map_err(FreezeError::Backend)?;

    // Return WC buffers to the per-thread cache.
    release_wc_bufs(bufs, chunk_size);

    let _ = total_file_size; // silence unused on non-file path
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

    // Open both fds: O_DIRECT for aligned middle chunks, buffered
    // for ragged final chunk or when O_DIRECT isn't supported.
    let io_err = |e: std::io::Error| {
        FreezeError::Snapshot(SnapshotError::Io {
            kind: e.kind(),
            message: e.to_string(),
        })
    };
    let file_direct = open_direct_write(path, config.try_direct_io).map_err(io_err)?;
    let file_buffered = open_direct_write(path, false).map_err(io_err)?;

    let chunk_size = config.chunk_size;
    assert!(chunk_size > 0, "chunk_size must be > 0");
    assert!(
        chunk_size % 4096 == 0,
        "chunk_size must be a multiple of 4096 for O_DIRECT"
    );

    // In the file-based path we write starting at offset 0 and
    // the prelude lives at the head of chunk 0. So "chunks" here
    // span the entire file, not just the payload region.
    let file_len = total_file_size;
    if file_len == 0 {
        // Empty snapshot — just write the prelude (prelude-only file).
        pwrite_exact(&file_buffered, &prelude_bytes, 0).map_err(io_err)?;
        return Ok(FreezeStats::default());
    }

    let num_chunks = ((file_len as usize) + chunk_size - 1) / chunk_size;

    let total_bytes: u64 = plan.iter().map(|p| p.size).sum();

    // Align-up helper for O_DIRECT write sizes.
    fn align_up_4k(n: usize) -> usize {
        (n + 4095) & !4095
    }

    // Two write-combining pinned buffers + two streams. Buffers come
    // from the per-thread cache, amortizing the allocation cost across
    // successive freeze calls (e.g. thaw serve hot-swap).
    let mut bufs = acquire_wc_bufs(backend, chunk_size)?;
    let streams = [backend.stream_create()?, backend.stream_create()?];

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

            // Write previous chunk (overlaps D2H).
            write_chunk_to_file(&mut bufs[prev], prev_start, prev_len)?;

            // Wait for current chunk's D2H before reuse.
            backend.stream_sync(&streams[curr])?;
        }

        // Write final chunk.
        let last = (num_chunks - 1) % 2;
        let (last_start, last_end) = chunk_range(num_chunks - 1);
        let last_len = (last_end - last_start) as usize;
        write_chunk_to_file(&mut bufs[last], last_start, last_len)?;
    }

    // Truncate to exact file size (strips O_DIRECT 4 KiB padding).
    truncate(&file_direct, file_len).map_err(io_err)?;

    // Cleanup streams.
    backend
        .stream_destroy(streams[0])
        .map_err(FreezeError::Backend)?;
    backend
        .stream_destroy(streams[1])
        .map_err(FreezeError::Backend)?;

    // Return WC buffers to the per-thread cache.
    release_wc_bufs(bufs, chunk_size);

    Ok(FreezeStats {
        regions_frozen: requests.len(),
        bytes_copied: total_bytes,
    })
}

// =============================================================================
// PIPELINED RESTORE
// =============================================================================

/// A single entry in the copy plan: which region of the file maps
/// to which device region.
struct CopyEntry {
    /// Absolute byte offset in the file where this region's payload
    /// starts.
    file_offset: u64,
    /// Size of this region in bytes.
    size: u64,
    /// Where on the device to put it.
    device_region: DeviceRegion,
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
            file_offset,
            size,
            device_region,
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
    assert!(chunk_size % 4096 == 0, "chunk_size must be a multiple of 4096");

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
    // Use write-combining memory: the CPU writes (from pread/mmap) and the
    // GPU reads (via DMA). WC bypasses cache snooping → up to 40% faster H2D.
    // Buffers come from the per-thread cache — amortizes `cudaHostAlloc`
    // across successive restores (critical for thaw serve hot-swap).
    let mut bufs = acquire_wc_bufs(backend, chunk_size).map_err(RestoreError::Backend)?;
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

    // Prime the pump: read chunk 0.
    let chunk0_start = read_start;
    let chunk0_len = chunk_size.min(read_len);
    read_chunk(&mut bufs[0], chunk0_start, chunk0_len)?;

    if num_chunks == 1 {
        // Single chunk: upload and done.
        let chunk_end = chunk0_start + chunk0_len as u64;
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
            let curr = chunk_idx % 2;

            let prev_start = read_start + (chunk_idx - 1) as u64 * chunk_size as u64;
            let prev_len = chunk_size.min(read_len - (chunk_idx - 1) * chunk_size);
            let prev_end = prev_start + prev_len as u64;

            let curr_start = read_start + chunk_idx as u64 * chunk_size as u64;
            let curr_len = chunk_size.min(read_len - chunk_idx * chunk_size);

            // Launch async uploads from the previous chunk's buffer.
            launch_uploads(
                backend,
                &bufs[prev],
                &streams[prev],
                prev_start,
                prev_end,
                &plan,
            )?;

            // Read the next chunk into the current buffer (overlaps
            // with the DMA from the previous buffer).
            read_chunk(&mut bufs[curr], curr_start, curr_len)?;

            // Sync the previous stream before its buffer gets reused
            // in the next iteration.
            backend
                .stream_sync(&streams[prev])
                .map_err(RestoreError::Backend)?;
        }

        // Upload the final chunk.
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
            file_offset,
            size,
            device_region,
        });
    }

    if plan.is_empty() {
        return Ok(RestoreStats::default());
    }

    // -- Phase 2: Double-buffered pipelined restore from memory -----

    let chunk_size = config.chunk_size;
    assert!(
        chunk_size % 4096 == 0,
        "chunk_size must be a multiple of 4096"
    );

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
            file_offset,
            size,
            device_region,
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

    /// Peek at the current thread's cached chunk_size, if any.
    /// Test-only helper that touches the module-private thread_local.
    fn cached_chunk_size() -> Option<usize> {
        super::WC_BUF_CACHE.with(|c| c.borrow().as_ref().map(|x| x.chunk_size))
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
}
