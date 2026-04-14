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

use std::io::Write;
use std::path::Path;

use thaw_core::{ByteRegionWriter, RegionKind, Snapshot, SnapshotError, HEADER_SIZE};

use crate::backend::{DeviceRegion, PipelinedBackend};
use crate::direct_io::{open_direct, pread_exact};
use crate::freeze::{FreezeConfig, FreezeError, FreezeRequest};
use crate::restore::{RestoreError, RestoreStats};

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

/// Pipelined freeze: double-buffered async D2H with overlapped
/// disk writes.
///
/// Same file format as `freeze`, but significantly faster for large
/// models because D2H DMA and disk I/O run concurrently. Also
/// avoids the per-region `alloc_pinned` overhead by reusing two
/// pre-allocated buffers.
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
    // Phase 1: Write header + region table.
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
    snapshot.write_to(sink)?;

    if requests.is_empty() {
        return Ok(FreezeStats::default());
    }

    // Phase 2: Double-buffered D2H + write pipeline.
    //
    // Allocate two pinned buffers sized to the largest region. Each
    // region gets one D2H async call, and the double-buffering
    // overlaps the disk write of region N with the D2H of region N+1.
    let max_region_size = requests
        .iter()
        .map(|r| r.device_region.size as usize)
        .max()
        .unwrap();

    let mut buf_a = backend.alloc_pinned(max_region_size)?;
    let mut buf_b = backend.alloc_pinned(max_region_size)?;
    let stream_a = backend.stream_create()?;
    let stream_b = backend.stream_create()?;

    let total_bytes: u64 = requests.iter().map(|r| r.device_region.size).sum();

    // Helper to write a region's worth of data from a buffer.
    let write_region = |buf: &crate::backend::PinnedBuffer,
                        size: usize,
                        sink: &mut W|
     -> Result<(), FreezeError> {
        sink.write_all(&buf.as_slice()[..size])
            .map_err(|e| FreezeError::Snapshot(SnapshotError::from(e)))?;
        Ok(())
    };

    // Prime: D2H first region into buf_a, sync.
    backend.memcpy_d2h_async(
        &mut buf_a,
        0,
        &requests[0].device_region,
        &stream_a,
    )?;
    backend.stream_sync(&stream_a)?;

    // Steady-state: overlap D2H(current) with write(previous).
    let mut prev_in_a = true;

    for i in 1..requests.len() {
        let prev_size = requests[i - 1].device_region.size as usize;

        if prev_in_a {
            // D2H current region into buf_b (async, returns immediately).
            backend.memcpy_d2h_async(
                &mut buf_b,
                0,
                &requests[i].device_region,
                &stream_b,
            )?;
            // Write buf_a to disk (overlaps with D2H into buf_b).
            write_region(&buf_a, prev_size, sink)?;
            // Wait for buf_b to be ready.
            backend.stream_sync(&stream_b)?;
        } else {
            // D2H current region into buf_a (async).
            backend.memcpy_d2h_async(
                &mut buf_a,
                0,
                &requests[i].device_region,
                &stream_a,
            )?;
            // Write buf_b to disk (overlaps with D2H into buf_a).
            write_region(&buf_b, prev_size, sink)?;
            // Wait for buf_a to be ready.
            backend.stream_sync(&stream_a)?;
        }

        prev_in_a = !prev_in_a;
    }

    // Write final region.
    let last_size = requests.last().unwrap().device_region.size as usize;
    if prev_in_a {
        write_region(&buf_a, last_size, sink)?;
    } else {
        write_region(&buf_b, last_size, sink)?;
    }

    // Cleanup.
    backend
        .stream_destroy(stream_a)
        .map_err(FreezeError::Backend)?;
    backend
        .stream_destroy(stream_b)
        .map_err(FreezeError::Backend)?;

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
    let mut bufs = [
        backend.alloc_pinned(chunk_size).map_err(RestoreError::Backend)?,
        backend.alloc_pinned(chunk_size).map_err(RestoreError::Backend)?,
    ];
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

    let mut bufs = [
        backend
            .alloc_pinned(chunk_size)
            .map_err(RestoreError::Backend)?,
        backend
            .alloc_pinned(chunk_size)
            .map_err(RestoreError::Backend)?,
    ];
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

    Ok(RestoreStats {
        regions_restored: plan.len(),
        bytes_copied: total_bytes,
    })
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
}
