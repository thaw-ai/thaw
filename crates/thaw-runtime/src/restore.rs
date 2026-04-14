// crates/thaw-runtime/src/restore.rs
//
// =============================================================================
// THE RESTORE ORCHESTRATOR
// =============================================================================
//
// The mirror of `freeze`. Given a `.thaw` file already read into
// memory and a function that tells us where each region should be
// pushed back onto the device, this module walks the region table
// and issues one `memcpy_h2d` per entry through the `CudaBackend`.
//
// Why a resolver closure instead of a "restore request" list:
//
//   Freeze is easy — the caller already knows exactly which device
//   regions they want captured, so they hand us a list and we walk
//   it. Restore is asymmetric: the *file* knows the list of
//   regions. The caller only has to tell us, for each entry in that
//   list, "if you see this (kind, logical_id), put the bytes at
//   this DeviceRegion." A closure is the most compact and least
//   allocation-y way to express that mapping without forcing every
//   caller to build a HashMap first. vLLM's restore path will
//   typically be "look up the device address from the block
//   table," which is a fast O(1) lookup — no HashMap needed.
//
// Why "unmapped region" is an error and not a silent skip:
//
//   A partially-restored model is worse than no model at all. If
//   the caller's resolver does not recognize a region in the file,
//   that is either a programming bug (forgot to wire up a region
//   kind) or a semantic mismatch ("this file was produced by a
//   different deployment and you should not be loading it"). In
//   either case we want the operator to see a typed error and
//   decide, not for thaw to silently drop model weights on the
//   floor.
//
// What this does NOT do yet, and why:
//
//   - No streaming. The whole file sits in `&[u8]`; we don't pull
//     one region at a time off disk. Same Phase 1 simplification
//     as `freeze`: correctness first, the streaming version reuses
//     the same trait with an added `Read + Seek` bound later.
//
//   - No cross-region ordering concerns. Each region is an
//     independent `memcpy_h2d`. The real CUDA backend will
//     eventually want to pipeline these on two streams, but that
//     optimization is a backend concern — the orchestrator just
//     says "copy this region," and the backend decides how.
//
//   - No vllm_commit check. The restore path *should* verify the
//     file's vllm_commit matches the running vLLM's commit, but
//     that comparison needs the caller's commit string, which
//     belongs in the higher-level integration layer (the Python
//     bindings that know how to ask vLLM for its own commit).
//     This module exposes the parsed header so the caller can do
//     the check themselves with one line.
//
// =============================================================================

use thiserror::Error;

use thaw_core::{RegionKind, Snapshot, SnapshotError};

use crate::backend::{BackendError, CudaBackend, DeviceRegion};

/// Summary statistics from a successful `restore` call.
///
/// A small struct instead of a tuple so adding fields later (e.g.
/// "how many bytes were zero-skipped by a future optimization")
/// is not a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RestoreStats {
    /// Number of regions successfully copied back onto the device.
    pub regions_restored: usize,
    /// Total bytes moved host-to-device across all regions.
    pub bytes_copied: u64,
}

/// Errors produced by the restore orchestrator.
///
/// Wraps the three underlying error sources it can see: the file
/// parser, the backend, and the resolver closure. Keeping them
/// distinct at the enum level lets the caller tell "the file is
/// bad" from "the GPU refused" from "I forgot to map a region."
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RestoreError {
    /// The `.thaw` file failed to parse. The inner error carries
    /// the specific reason (bad header, bad table, truncated).
    #[error("snapshot parse failed: {0}")]
    Snapshot(#[from] SnapshotError),

    /// A `CudaBackend` call failed (allocation, unknown device
    /// pointer, size mismatch). Restore aborts on the first such
    /// error — a partial restore leaves the device in a
    /// never-observed state that is worse than the original
    /// cold-start path.
    #[error("backend error during restore: {0}")]
    Backend(#[from] BackendError),

    /// The resolver closure returned `None` for a region that
    /// appears in the file. This is almost always a programming
    /// bug in the caller's mapping logic; see the module-level
    /// comment for why we treat it as fatal.
    #[error("no device mapping for region: kind={kind:?}, logical_id={logical_id}")]
    UnmappedRegion {
        kind: RegionKind,
        logical_id: u32,
    },

    /// The resolver returned a `DeviceRegion` whose `size` does
    /// not match the region's size as recorded in the file. This
    /// means the caller's device allocation disagrees with the
    /// frozen layout — loading the bytes anyway would either
    /// truncate or overflow, and both are silent data corruption.
    #[error(
        "device region size mismatch: kind={kind:?}, logical_id={logical_id}, \
         file_size={file_size}, device_size={device_size}"
    )]
    DeviceSizeMismatch {
        kind: RegionKind,
        logical_id: u32,
        file_size: u64,
        device_size: u64,
    },

    /// The file claims a region whose `file_offset + size` lands
    /// beyond the end of the in-memory byte slice. Defensive: the
    /// writer we control never produces this, but a truncated or
    /// corrupted `.thaw` would, and we want to say so precisely
    /// rather than panicking on a slice index.
    #[error(
        "region payload out of bounds: file_size={file_size}, \
         payload_end={payload_end}"
    )]
    PayloadOutOfBounds {
        file_size: usize,
        payload_end: usize,
    },
}

/// Restore a `.thaw` file onto a GPU backend.
///
/// `file_bytes` is the complete file contents (prelude + payload)
/// already read into memory. The caller is responsible for getting
/// the bytes there; a later streaming restore will lift that
/// responsibility into this module.
///
/// `resolve` is called once per region in the file, in table order,
/// with the region's `kind` and `logical_id`. It must return the
/// `DeviceRegion` where the bytes should land, or `None` if the
/// caller does not recognize that region (which aborts the restore
/// — see the module comment).
///
/// On success, returns a `RestoreStats` describing what was copied.
/// On failure, returns a `RestoreError` and the backend is left in
/// whatever state the failed call produced; callers that need
/// atomicity should either pre-validate the mapping or be prepared
/// to treat a failed restore as a "fall back to cold start" signal.
pub fn restore<B, F>(
    backend: &B,
    file_bytes: &[u8],
    mut resolve: F,
) -> Result<RestoreStats, RestoreError>
where
    B: CudaBackend + ?Sized,
    F: FnMut(RegionKind, u32) -> Option<DeviceRegion>,
{
    // Parse the prelude out of the leading bytes. `from_prelude_bytes`
    // only reads as far as `HEADER_SIZE + num_regions * ENTRY_SIZE`,
    // so handing it the full file is fine — it ignores the payload
    // tail. We keep the payload around because we're about to slice
    // into it.
    let snapshot = Snapshot::from_prelude_bytes(file_bytes)?;

    let mut stats = RestoreStats::default();

    for i in 0..snapshot.len() {
        // `get` is infallible here because we're iterating in
        // bounds, but the table returns an Option so we propagate
        // the "impossible" case as a parse error rather than
        // `unwrap`. A future parser change that could leave holes
        // in the table would then surface here, not as a panic.
        let entry = snapshot
            .table()
            .get(i)
            .ok_or(RestoreError::Snapshot(SnapshotError::TruncatedTable {
                got: file_bytes.len(),
                need: 0,
            }))?;

        let kind = entry.kind();
        let logical_id = entry.logical_id();
        let size = entry.size();
        let offset = entry.file_offset() as usize;

        // Ask the caller where this region goes. `None` is fatal
        // for the reasons documented on `RestoreError::UnmappedRegion`.
        let device_region = resolve(kind, logical_id).ok_or(RestoreError::UnmappedRegion {
            kind,
            logical_id,
        })?;

        // The resolver is trusted for the device pointer but not
        // for the size — a caller that hands us a differently-
        // sized allocation is a bug we want to catch before it
        // becomes silent corruption.
        if device_region.size != size {
            return Err(RestoreError::DeviceSizeMismatch {
                kind,
                logical_id,
                file_size: size,
                device_size: device_region.size,
            });
        }

        // Bounds-check the payload slice *before* indexing. A
        // corrupted file that claims a region past EOF would
        // otherwise panic on the slice below.
        let end = offset
            .checked_add(size as usize)
            .ok_or(RestoreError::PayloadOutOfBounds {
                file_size: file_bytes.len(),
                payload_end: usize::MAX,
            })?;
        if end > file_bytes.len() {
            return Err(RestoreError::PayloadOutOfBounds {
                file_size: file_bytes.len(),
                payload_end: end,
            });
        }

        // Allocate a pinned buffer of exactly the region's size,
        // fill it from the file bytes, and hand it to the backend.
        // Same copy-once shape as freeze: a later streaming impl
        // will replace the intermediate `copy_from_slice` with a
        // direct pread into pinned memory.
        let mut pinned = backend.alloc_pinned(size as usize)?;
        pinned.as_mut_slice().copy_from_slice(&file_bytes[offset..end]);
        backend.memcpy_h2d(&device_region, &pinned)?;

        stats.regions_restored += 1;
        stats.bytes_copied += size;
    }

    Ok(stats)
}

// =============================================================================
// TESTS
// =============================================================================
//
// Tier-2 mock-integration tests: freeze into a buffer with one
// `MockCuda`, then restore that same buffer onto a *fresh*
// `MockCuda` and assert the device side matches. This is the first
// test in the project that round-trips bytes through the full
// freeze/restore pipeline, so it is the ultimate correctness
// smoke test for everything below the Python bindings.
//
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::DevicePtr;
    use crate::freeze::{freeze, FreezeConfig, FreezeRequest};
    use crate::mock::MockCuda;

    /// The basic round-trip: freeze two regions from a seeded
    /// backend into a byte buffer, then restore that buffer onto a
    /// second backend whose device regions are pre-allocated but
    /// zeroed. After restore, the second backend's device bytes
    /// should equal what the first backend held.
    ///
    /// If any single piece of the freeze/restore pipeline is
    /// wrong — the writer, the parser, the orchestrators, or the
    /// mock — this test fires. Which is exactly why it is worth
    /// writing before the real GPU impl exists.
    #[test]
    fn restore_round_trips_through_freeze() {
        // Source side: populate a mock, freeze it.
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
        let mut file: Vec<u8> = Vec::new();
        freeze(&src, &requests, &FreezeConfig::default(), &mut file).expect("freeze");

        // Destination side: fresh mock with the same shapes but
        // zeroed contents, standing in for "fresh process that
        // just cudaMalloc'd empty regions."
        let dst = MockCuda::new();
        let w_ptr2 = DevicePtr(0xA000);
        let m_ptr2 = DevicePtr(0xB000);
        dst.register_region(w_ptr2, vec![0u8; weights.len()]);
        dst.register_region(m_ptr2, vec![0u8; metadata.len()]);

        let stats = restore(&dst, &file, |kind, _logical_id| match kind {
            RegionKind::Weights => Some(DeviceRegion::new(w_ptr2, weights.len() as u64)),
            RegionKind::Metadata => Some(DeviceRegion::new(m_ptr2, metadata.len() as u64)),
            _ => None,
        })
        .expect("restore");

        assert_eq!(stats.regions_restored, 2);
        assert_eq!(stats.bytes_copied, (weights.len() + metadata.len()) as u64);
        assert_eq!(dst.read_region(w_ptr2).unwrap(), weights);
        assert_eq!(dst.read_region(m_ptr2).unwrap(), metadata);
    }

    /// Restoring an empty file (prelude only, zero regions) is a
    /// no-op and returns zero stats. Edge case: the N=0 loop must
    /// not trip on anything.
    #[test]
    fn restore_with_no_regions_is_noop() {
        let backend = MockCuda::new();
        let mut file: Vec<u8> = Vec::new();
        freeze(&backend, &[], &FreezeConfig::default(), &mut file).expect("freeze");

        let stats = restore(&backend, &file, |_, _| {
            panic!("resolver should never be called on an empty file")
        })
        .expect("restore");
        assert_eq!(stats.regions_restored, 0);
        assert_eq!(stats.bytes_copied, 0);
    }

    /// A resolver that returns `None` for a region present in the
    /// file surfaces as `UnmappedRegion`, not a silent skip.
    #[test]
    fn restore_errors_on_unmapped_region() {
        let src = MockCuda::new();
        let ptr = DevicePtr(0x1000);
        src.register_region(ptr, vec![0xAB; 32]);
        let requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            7,
            DeviceRegion::new(ptr, 32),
        )];
        let mut file: Vec<u8> = Vec::new();
        freeze(&src, &requests, &FreezeConfig::default(), &mut file).expect("freeze");

        let dst = MockCuda::new();
        let err =
            restore(&dst, &file, |_, _| None).expect_err("resolver returns None -> fatal");
        match err {
            RestoreError::UnmappedRegion { kind, logical_id } => {
                assert_eq!(kind, RegionKind::Weights);
                assert_eq!(logical_id, 7);
            }
            other => panic!("expected UnmappedRegion, got {other:?}"),
        }
    }

    /// A resolver that returns a `DeviceRegion` whose size does
    /// not match the file's size surfaces as `DeviceSizeMismatch`
    /// before any bytes are copied. This is the guard against
    /// silent truncation / overflow that the module comment calls
    /// out.
    #[test]
    fn restore_errors_on_device_size_mismatch() {
        let src = MockCuda::new();
        let ptr = DevicePtr(0x1000);
        src.register_region(ptr, vec![0xCD; 64]);
        let requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(ptr, 64),
        )];
        let mut file: Vec<u8> = Vec::new();
        freeze(&src, &requests, &FreezeConfig::default(), &mut file).expect("freeze");

        let dst = MockCuda::new();
        let dst_ptr = DevicePtr(0x9000);
        // Deliberately wrong size: file is 64, device claims 32.
        dst.register_region(dst_ptr, vec![0u8; 32]);

        let err = restore(&dst, &file, |_, _| {
            Some(DeviceRegion::new(dst_ptr, 32))
        })
        .expect_err("size mismatch should abort");
        match err {
            RestoreError::DeviceSizeMismatch {
                file_size,
                device_size,
                ..
            } => {
                assert_eq!(file_size, 64);
                assert_eq!(device_size, 32);
            }
            other => panic!("expected DeviceSizeMismatch, got {other:?}"),
        }
    }

    /// A corrupted file whose prelude parse fails surfaces as
    /// `RestoreError::Snapshot`, not a panic. Pins the error-
    /// propagation wiring.
    #[test]
    fn restore_surfaces_parse_errors() {
        let backend = MockCuda::new();
        let garbage = vec![0u8; 16]; // too short to even hold a header
        let err = restore(&backend, &garbage, |_, _| None)
            .expect_err("garbage should fail to parse");
        match err {
            RestoreError::Snapshot(_) => {}
            other => panic!("expected Snapshot error, got {other:?}"),
        }
    }

    /// A backend error during the `memcpy_h2d` (here: unknown
    /// device pointer on the destination side) surfaces as
    /// `RestoreError::Backend` with the inner `BackendError`
    /// preserved. Pins the error-propagation wiring.
    #[test]
    fn restore_surfaces_backend_errors() {
        let src = MockCuda::new();
        let ptr = DevicePtr(0x1000);
        src.register_region(ptr, vec![0x55; 16]);
        let requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(ptr, 16),
        )];
        let mut file: Vec<u8> = Vec::new();
        freeze(&src, &requests, &FreezeConfig::default(), &mut file).expect("freeze");

        // Destination backend has NO registered region, so the
        // resolver will hand us a pointer that memcpy_h2d cannot
        // find.
        let dst = MockCuda::new();
        let err = restore(&dst, &file, |_, _| {
            Some(DeviceRegion::new(DevicePtr(0xDEAD), 16))
        })
        .expect_err("unknown pointer should abort");
        match err {
            RestoreError::Backend(BackendError::UnknownDevicePtr(p)) => {
                assert_eq!(p, DevicePtr(0xDEAD));
            }
            other => panic!("expected Backend(UnknownDevicePtr), got {other:?}"),
        }
    }

    /// Full three-region round-trip including a `KvLiveBlock`
    /// region with a non-zero `logical_id`. Verifies the
    /// orchestrator preserves logical_id from file to resolver
    /// call — this is the thing vLLM's restore path will actually
    /// key off of to find the right KV block in its pool.
    #[test]
    fn restore_preserves_logical_id_for_kv_blocks() {
        let src = MockCuda::new();
        let ptrs = [DevicePtr(0x1000), DevicePtr(0x2000), DevicePtr(0x3000)];
        let payloads: Vec<Vec<u8>> = (0..3)
            .map(|i| (0..128).map(|b| ((b + i * 17) & 0xFF) as u8).collect())
            .collect();
        for (p, bytes) in ptrs.iter().zip(payloads.iter()) {
            src.register_region(*p, bytes.clone());
        }
        // logical_ids 3, 17, 42 — arbitrary distinct non-zero values.
        let kv_ids = [3u32, 17, 42];
        let requests: Vec<FreezeRequest> = ptrs
            .iter()
            .zip(kv_ids.iter())
            .map(|(p, id)| {
                FreezeRequest::new(
                    RegionKind::KvLiveBlock,
                    *id,
                    DeviceRegion::new(*p, 128),
                )
            })
            .collect();
        let mut file: Vec<u8> = Vec::new();
        freeze(&src, &requests, &FreezeConfig::default(), &mut file).expect("freeze");

        // Destination maps each logical_id to its own fresh region.
        let dst = MockCuda::new();
        let dst_ptrs = [DevicePtr(0xA000), DevicePtr(0xB000), DevicePtr(0xC000)];
        for p in &dst_ptrs {
            dst.register_region(*p, vec![0u8; 128]);
        }
        let id_to_ptr = [
            (3u32, dst_ptrs[0]),
            (17, dst_ptrs[1]),
            (42, dst_ptrs[2]),
        ];

        let mut seen_ids: Vec<u32> = Vec::new();
        restore(&dst, &file, |kind, logical_id| {
            assert_eq!(kind, RegionKind::KvLiveBlock);
            seen_ids.push(logical_id);
            id_to_ptr
                .iter()
                .find(|(id, _)| *id == logical_id)
                .map(|(_, p)| DeviceRegion::new(*p, 128))
        })
        .expect("restore");

        // Resolver saw each id exactly once, in table order.
        assert_eq!(seen_ids, vec![3, 17, 42]);
        // Device contents match source.
        for (dst_ptr, original) in dst_ptrs.iter().zip(payloads.iter()) {
            assert_eq!(dst.read_region(*dst_ptr).unwrap(), *original);
        }
    }
}
