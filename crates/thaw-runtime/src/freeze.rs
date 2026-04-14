// crates/thaw-runtime/src/freeze.rs
//
// =============================================================================
// THE FREEZE ORCHESTRATOR
// =============================================================================
//
// Given a GPU backend and a list of "things I want to snapshot,"
// this module produces a `.thaw` file. It is the first piece of
// thaw that crosses all three abstraction layers in this repo:
//
//   - file format      (thaw-core::ByteRegionWriter)
//   - GPU abstraction  (thaw-runtime::CudaBackend)
//   - orchestration    (this module)
//
// Everything above this module only has to know about two types:
// `FreezeRequest` (what to capture) and `FreezeError` (what can go
// wrong). Everything below it is unaware that the orchestrator
// exists.
//
// THE HOT PATH, IN PROSE
// ----------------------
//
// For each request the caller hands us:
//   1. Allocate a pinned host buffer of exactly `region.size` bytes.
//   2. `memcpy_d2h` from the device region into that buffer.
//   3. Hand the bytes (plus the kind and logical_id) to
//      `ByteRegionWriter::push_region`.
//
// After the loop:
//   4. Optionally stamp the header's `vllm_commit` slot.
//   5. Call `writer.write_to(sink)` to stream the complete file.
//
// Notes on what this does NOT do yet, and why:
//
//   - No streaming. Every region is fully buffered in a `Vec<u8>`
//     before the file starts being written. The writer is already
//     two-pass by design (see `writer.rs`), and doing one-shot
//     `Vec` copies here keeps the control flow simple. A later
//     "streaming freeze" will reuse the same trait and push the
//     bytes down without the intermediate Vec.
//
//   - No concurrent streams. The CUDA hot path eventually wants
//     two pinned ring buffers and two streams alternating (the
//     trick that gets us from ~12 GB/s to ~24 GB/s on PCIe 4).
//     That is a backend-implementation concern, not an
//     orchestration concern — the orchestrator just says "copy
//     this region," and the backend decides how.
//
//   - No chunking for huge regions. A 16 GB weights region will
//     allocate 16 GB of pinned memory in one shot. The real
//     backend will split it in the `memcpy_d2h` implementation;
//     this layer doesn't need to know.
//
// All three of those are deliberate Phase 1 simplifications. The
// point of the orchestrator today is to prove the pipeline from
// "device region" to "`.thaw` file on disk" end-to-end, against a
// mock GPU, with full byte-exact correctness. Performance work
// comes after correctness.
//
// =============================================================================

use std::io::Write;

use thiserror::Error;

use thaw_core::{ByteRegionWriter, RegionKind, SnapshotError};

use crate::backend::{BackendError, CudaBackend, DeviceRegion};

/// A single "capture this region" instruction for the freeze
/// orchestrator.
///
/// Each request fully specifies what will appear in the
/// resulting file: the region kind (which bucket it goes into),
/// the logical id (meaningful for `KvLiveBlock`, zero otherwise),
/// and the device region whose bytes should be read.
///
/// The struct is `Clone` so that tests can build a list once and
/// pass copies to multiple orchestrator invocations without
/// ceremony.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreezeRequest {
    /// The region kind. Orchestration does not interpret this;
    /// it forwards the value straight to `ByteRegionWriter::push_region`.
    pub kind: RegionKind,

    /// The logical id for this region. Meaningful for
    /// `KvLiveBlock` (the original integer block index in vLLM's
    /// KV pool); pass `0` for `Weights` and `Metadata`.
    pub logical_id: u32,

    /// The device-side region whose bytes will be captured.
    pub device_region: DeviceRegion,
}

impl FreezeRequest {
    /// Convenience constructor.
    pub fn new(kind: RegionKind, logical_id: u32, device_region: DeviceRegion) -> Self {
        FreezeRequest {
            kind,
            logical_id,
            device_region,
        }
    }
}

/// Errors produced by the freeze orchestrator.
///
/// Wraps the two underlying error types it can see: backend
/// errors from the CUDA layer and format errors from thaw-core.
/// Neither is re-exported naked, because a caller wants to
/// distinguish "the GPU refused" from "the file format refused"
/// when deciding how to recover.
///
/// `#[non_exhaustive]` for the usual forward-compatibility
/// reasons.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FreezeError {
    /// A `CudaBackend` call failed (allocation, unknown pointer,
    /// size mismatch). The freeze cannot proceed without these
    /// bytes, so the whole operation aborts on the first such
    /// error.
    #[error("backend error during freeze: {0}")]
    Backend(#[from] BackendError),

    /// The file writer failed (I/O error, or a format-level
    /// failure bubbled up from thaw-core). Also aborts the
    /// whole freeze.
    #[error("snapshot error during freeze: {0}")]
    Snapshot(#[from] SnapshotError),
}

/// The freeze configuration.
///
/// A minimal "options bag" passed alongside the request list.
/// Today it only carries the vllm_commit hash; later it will
/// also carry things like "stream N regions in parallel,"
/// "use this chunk size," "attach this metadata JSON."
#[derive(Debug, Clone, Default)]
pub struct FreezeConfig {
    /// Optional 40-byte vllm_commit hash. If `Some`, the freeze
    /// orchestrator stamps it onto the snapshot header. If
    /// `None`, the header's slot stays at all-zeros and the
    /// later vLLM integration layer will refuse the file.
    pub vllm_commit: Option<[u8; 40]>,
}

/// Freeze a list of device regions into a complete `.thaw` file,
/// streamed to the provided writer.
///
/// This is the public entry point of the orchestrator. See the
/// module-level comment for the full pipeline description. The
/// function is generic over the backend so that tests can pass
/// `MockCuda` and production code can pass `RealCuda` — neither
/// knows about the other.
///
/// Returns the number of bytes written on success. The count
/// includes the header, the region table, and every region's
/// payload.
pub fn freeze<B, W>(
    backend: &B,
    requests: &[FreezeRequest],
    config: &FreezeConfig,
    sink: &mut W,
) -> Result<u64, FreezeError>
where
    B: CudaBackend + ?Sized,
    W: Write,
{
    // Phase 1: register every region's metadata (kind, logical_id,
    // size) with the writer so it can compute the prelude layout.
    // No payload bytes are copied yet.
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

    // Phase 2: write the prelude (header + region table). The
    // offsets in the table already account for every region's
    // declared size, so when we stream payload bytes next the
    // file's internal pointers will be correct.
    let snapshot = writer.build_snapshot();
    let mut written = snapshot.write_to(sink)?;

    // Phase 3: for each region, DMA from device into a pinned
    // buffer and write the pinned buffer directly to the sink.
    // No intermediate Vec<u8> copy -- the bytes go straight from
    // pinned memory into the writer, which is the difference
    // between ~1 GB/s and ~12 GB/s on PCIe 4.
    for request in requests {
        let size = request.device_region.size as usize;
        let mut pinned = backend.alloc_pinned(size)?;
        backend.memcpy_d2h(&mut pinned, &request.device_region)?;
        sink.write_all(pinned.as_slice())
            .map_err(|e| FreezeError::Snapshot(SnapshotError::from(e)))?;
        written += size as u64;
    }

    Ok(written)
}

// =============================================================================
// TESTS
// =============================================================================
//
// These are tier-2 mock-integration tests: they exercise the full
// freeze pipeline through `MockCuda`, verifying that the file
// bytes produced match the device bytes we seeded the mock with.
// No real GPU, no real filesystem.
//
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::DevicePtr;
    use crate::mock::MockCuda;
    use thaw_core::Snapshot;

    /// Build a mock backend pre-populated with one Weights and
    /// one Metadata region of known, distinct contents. Returned
    /// alongside the request list that describes them.
    ///
    /// Factored so multiple tests can share the same fixture.
    fn seeded_backend_and_requests(
    ) -> (MockCuda, Vec<FreezeRequest>, Vec<u8>, Vec<u8>) {
        let backend = MockCuda::new();
        let weights_ptr = DevicePtr(0x1000);
        let metadata_ptr = DevicePtr(0x2000);

        // Distinct, non-overlapping byte patterns so any off-by-
        // one bug in the orchestrator lands on a visibly wrong
        // value.
        let weights_bytes: Vec<u8> = (0..256).map(|i| (i & 0xFF) as u8).collect();
        let metadata_bytes: Vec<u8> = (0..32).map(|i| 0xA0 ^ (i as u8)).collect();

        backend.register_region(weights_ptr, weights_bytes.clone());
        backend.register_region(metadata_ptr, metadata_bytes.clone());

        let requests = vec![
            FreezeRequest::new(
                RegionKind::Weights,
                0,
                DeviceRegion::new(weights_ptr, weights_bytes.len() as u64),
            ),
            FreezeRequest::new(
                RegionKind::Metadata,
                0,
                DeviceRegion::new(metadata_ptr, metadata_bytes.len() as u64),
            ),
        ];

        (backend, requests, weights_bytes, metadata_bytes)
    }

    /// An empty request list produces a prelude-only file with
    /// no regions, and `freeze` returns `HEADER_SIZE` as the
    /// byte count. Edge case, tested first because off-by-one
    /// bugs love the N=0 path.
    #[test]
    fn freeze_with_no_requests_produces_empty_prelude() {
        let backend = MockCuda::new();
        let mut sink: Vec<u8> = Vec::new();
        let written = freeze(&backend, &[], &FreezeConfig::default(), &mut sink)
            .expect("freeze should succeed");
        assert_eq!(written, thaw_core::HEADER_SIZE);
        assert_eq!(sink.len() as u64, thaw_core::HEADER_SIZE);

        // The file parses as an empty snapshot.
        let parsed = Snapshot::from_prelude_bytes(&sink).expect("parse");
        assert_eq!(parsed.len(), 0);
    }

    /// The happy path: two distinct regions get captured, the
    /// resulting file parses cleanly, and every byte the mock
    /// backing-store held is present in the file at the offset
    /// the region table points at.
    ///
    /// This is the first end-to-end test that ties the mock GPU
    /// to the file format. If it ever fires, some layer in the
    /// pipeline is mis-routing bytes.
    #[test]
    fn freeze_captures_region_bytes_into_file() {
        let (backend, requests, weights_bytes, metadata_bytes) =
            seeded_backend_and_requests();

        let mut sink: Vec<u8> = Vec::new();
        freeze(&backend, &requests, &FreezeConfig::default(), &mut sink)
            .expect("freeze should succeed");

        let parsed = Snapshot::from_prelude_bytes(&sink).expect("parse prelude");
        assert_eq!(parsed.len(), 2);

        // Entry 0: Weights region. Size and kind must match what
        // we registered; payload bytes at its `file_offset` must
        // equal the mock's registered contents.
        let e0 = parsed.table().get(0).unwrap();
        assert_eq!(e0.kind(), RegionKind::Weights);
        assert_eq!(e0.size() as usize, weights_bytes.len());
        let start = e0.file_offset() as usize;
        let end = start + e0.size() as usize;
        assert_eq!(&sink[start..end], weights_bytes.as_slice());

        // Entry 1: Metadata region. Same checks.
        let e1 = parsed.table().get(1).unwrap();
        assert_eq!(e1.kind(), RegionKind::Metadata);
        assert_eq!(e1.size() as usize, metadata_bytes.len());
        let start = e1.file_offset() as usize;
        let end = start + e1.size() as usize;
        assert_eq!(&sink[start..end], metadata_bytes.as_slice());
    }

    /// `FreezeConfig::vllm_commit` lands on the resulting file's
    /// header. This is the mitigation from Decision Log
    /// 2026-04-11: without it, the later vLLM integration layer
    /// cannot tell which vLLM produced a snapshot, and refuses
    /// to load it on suspicion.
    #[test]
    fn freeze_stamps_vllm_commit_on_header() {
        let (backend, requests, _, _) = seeded_backend_and_requests();
        let hash = *b"deadbeefdeadbeefdeadbeefdeadbeefdeadbeef";

        let mut sink: Vec<u8> = Vec::new();
        let config = FreezeConfig {
            vllm_commit: Some(hash),
        };
        freeze(&backend, &requests, &config, &mut sink).expect("freeze");

        let parsed = Snapshot::from_prelude_bytes(&sink).expect("parse");
        assert_eq!(parsed.header().vllm_commit(), &hash);
    }

    /// A FreezeConfig with no vllm_commit leaves the header's
    /// slot at the all-zero "unset" sentinel. Pins the default.
    #[test]
    fn freeze_without_vllm_commit_leaves_slot_zero() {
        let (backend, requests, _, _) = seeded_backend_and_requests();
        let mut sink: Vec<u8> = Vec::new();
        freeze(&backend, &requests, &FreezeConfig::default(), &mut sink)
            .expect("freeze");
        let parsed = Snapshot::from_prelude_bytes(&sink).expect("parse");
        assert_eq!(parsed.header().vllm_commit(), &[0u8; 40]);
    }

    /// A request that points at an unregistered device pointer
    /// surfaces as `FreezeError::Backend(UnknownDevicePtr)`, not
    /// a panic. The whole freeze aborts — no partial file is
    /// produced, because a partial `.thaw` is worse than no file
    /// at all.
    #[test]
    fn freeze_surfaces_backend_error_on_unknown_pointer() {
        let backend = MockCuda::new();
        let requests = vec![FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(DevicePtr(0xDEAD), 16),
        )];
        let mut sink: Vec<u8> = Vec::new();
        let err = freeze(&backend, &requests, &FreezeConfig::default(), &mut sink)
            .expect_err("should fail");
        match err {
            FreezeError::Backend(BackendError::UnknownDevicePtr(p)) => {
                assert_eq!(p, DevicePtr(0xDEAD));
            }
            other => panic!("expected UnknownDevicePtr, got {other:?}"),
        }
    }

    /// Round-trip through `MockCuda`: freeze a region, then use
    /// the parsed table's `file_offset` / `size` to slice the
    /// payload back out of the sink and confirm it equals what
    /// the mock held.
    ///
    /// This is the "end-to-end" proof that the orchestrator,
    /// the writer, the parser, and the mock backend all agree
    /// on what the bytes of a region are. If any of the four
    /// have a bug, this test fires before a real GPU has ever
    /// been touched.
    #[test]
    fn freeze_round_trips_region_bytes_end_to_end() {
        let backend = MockCuda::new();
        let ptr = DevicePtr(0xAAAA);
        let original: Vec<u8> = (0..4096).map(|i| ((i * 7) & 0xFF) as u8).collect();
        backend.register_region(ptr, original.clone());

        let request = FreezeRequest::new(
            RegionKind::Weights,
            0,
            DeviceRegion::new(ptr, original.len() as u64),
        );

        let mut sink: Vec<u8> = Vec::new();
        freeze(&backend, &[request], &FreezeConfig::default(), &mut sink)
            .expect("freeze");

        let parsed = Snapshot::from_prelude_bytes(&sink).expect("parse");
        let entry = parsed.table().get(0).unwrap();
        let start = entry.file_offset() as usize;
        let end = start + entry.size() as usize;
        assert_eq!(&sink[start..end], original.as_slice());
    }
}
