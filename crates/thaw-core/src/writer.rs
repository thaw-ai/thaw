// crates/thaw-core/src/writer.rs
//
// =============================================================================
// FILE WRITER: PRELUDE + PAYLOAD
// =============================================================================
//
// `snapshot.rs` knows how to serialize the prelude (header + region
// table). What it doesn't know is how to compose that prelude with
// actual payload bytes — the weight blob, the KV blocks, the metadata
// JSON — into a single self-describing `.thaw` file.
//
// This module is the first piece that does. It is deliberately simple:
//
//   1. Caller constructs a `ByteRegionWriter`.
//   2. Caller pushes (kind, logical_id, bytes) tuples in whatever
//      order makes sense. Order is preserved.
//   3. Caller calls `write_to(sink)`. The writer computes file
//      offsets, builds the `Snapshot`, writes the prelude, and
//      streams the payload bytes straight out the back.
//
// WHY IT IS A SEPARATE MODULE
// ---------------------------
//
// `Snapshot` is a pure data type — it knows about headers and tables
// but nothing about "how do I decide what the file_offset of this
// region should be." That layout math is its own concern, and
// bundling it with `Snapshot` would force every caller that only
// wants to read a snapshot to also compile the layout code.
//
// WHY IT IS TWO-PASS (AND NOT STREAMING)
// --------------------------------------
//
// A streaming writer would accept `impl Read`s for payload bytes
// and seek-back to rewrite the header at the end. That is what
// real production code will do eventually, because you cannot hold
// 16 GiB of weights in RAM just to build a snapshot file. For now
// we are buffering in memory because:
//
//   - Every test we want to write in this crate uses small payloads
//     (KiB, not GiB). No memory pressure.
//   - Streaming requires `Seek` on the sink, which rules out
//     `Vec<u8>` and makes tests noticeably more awkward.
//   - The layout math is the hard part. Get it right in the simple
//     shape first; swap the backend for a streaming one later
//     behind the same public API.
//
// TDD STORY
// ---------
//
// Smallest possible claim first: an empty writer produces a file
// that is exactly `HEADER_SIZE` bytes long, parses as a valid
// snapshot with zero regions, and contains no payload. Every
// subsequent test adds one more behavior.
//
// =============================================================================

use std::io::Write;

use crate::header::HEADER_SIZE;
use crate::region::{RegionEntry, RegionKind, RegionTable, REGION_ENTRY_SIZE};
use crate::snapshot::{Snapshot, SnapshotError};

/// A pending region: kind, logical id, declared size, and optionally
/// the bytes that will be written to the file.
///
/// Private to this module. Callers interact via `push_region` or
/// `push_region_metadata`.
///
/// `size` is authoritative for layout computation. When `bytes` is
/// non-empty, `size == bytes.len() as u64` is an invariant
/// maintained by `push_region`. When `bytes` is empty (metadata-only
/// push), `size` carries the caller-declared payload size for the
/// prelude layout, and the caller is responsible for streaming the
/// actual bytes to the sink themselves.
#[derive(Debug, Clone)]
struct PendingRegion {
    kind: RegionKind,
    logical_id: u32,
    size: u64,
    bytes: Vec<u8>,
    /// CRC32C of the region's payload bytes. For `push_region` this is
    /// computed from `bytes` on insertion. For `push_region_metadata`
    /// it starts at zero; orchestrators that stream payload themselves
    /// should use `push_region_metadata_with_crc` to supply it.
    crc32c: u32,
}

/// Builds a complete `.thaw` file (prelude + payload) from a list
/// of in-memory byte blobs.
///
/// Use:
///
/// ```ignore
/// let mut w = ByteRegionWriter::new();
/// w.push_region(RegionKind::Weights, 0, weight_bytes);
/// w.push_region(RegionKind::Metadata, 0, metadata_json);
/// let mut file = std::fs::File::create("snapshot.thaw")?;
/// w.write_to(&mut file)?;
/// ```
///
/// See the module comment for why this is two-pass and in-memory.
#[derive(Debug, Default, Clone)]
pub struct ByteRegionWriter {
    pending: Vec<PendingRegion>,
    /// Optional vllm_commit hash to stamp on the header when
    /// `build_snapshot` runs. `None` means "leave the header's
    /// vllm_commit slot at its all-zero default," which the
    /// later vLLM integration layer will treat as "not
    /// recorded" and refuse to load. Setting this is how the
    /// freeze orchestrator signals "this file was produced
    /// against this specific vLLM."
    vllm_commit: Option<[u8; 40]>,
}

impl ByteRegionWriter {
    /// Build an empty writer. No regions yet.
    pub fn new() -> Self {
        ByteRegionWriter {
            pending: Vec::new(),
            vllm_commit: None,
        }
    }

    /// Stamp a vllm_commit hash on every snapshot this writer
    /// subsequently produces. See the field doc for semantics.
    ///
    /// Returns `&mut self` (not `self`) because the writer is
    /// typically built up with multiple `push_region` calls that
    /// already borrow it mutably; a by-value builder would force
    /// the caller to sequence things awkwardly.
    pub fn set_vllm_commit(&mut self, commit: [u8; 40]) -> &mut Self {
        self.vllm_commit = Some(commit);
        self
    }

    /// Number of regions queued for writing.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// True if no regions are queued.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Append a region to be written when `write_to` runs.
    ///
    /// The order of pushes determines the order of entries in the
    /// region table *and* the order of payload bytes in the file.
    /// That is not an accident — payload bytes get laid down
    /// back-to-back in push order, and each entry's `file_offset`
    /// points at the start of the corresponding slice.
    ///
    /// `logical_id` is only meaningful for `KvLiveBlock` regions;
    /// callers should pass 0 for `Weights` and `Metadata`.
    pub fn push_region(&mut self, kind: RegionKind, logical_id: u32, bytes: Vec<u8>) {
        let size = bytes.len() as u64;
        let crc32c = crc32c::crc32c(&bytes);
        self.pending.push(PendingRegion {
            kind,
            logical_id,
            size,
            bytes,
            crc32c,
        });
    }

    /// Register a region by metadata only (kind, logical_id, size)
    /// without providing its bytes. Used by the streaming freeze
    /// path: the orchestrator calls this to build the correct
    /// prelude layout, writes the prelude via `build_snapshot`,
    /// then streams each region's payload directly from pinned
    /// memory to the sink -- no intermediate `Vec<u8>` copy.
    ///
    /// A writer that has regions added via this method MUST NOT
    /// call `write_to`, because there are no payload bytes to
    /// write. Use `build_snapshot` to get the prelude, write it
    /// yourself, then write payload bytes in the same order.
    pub fn push_region_metadata(
        &mut self,
        kind: RegionKind,
        logical_id: u32,
        size: u64,
    ) {
        self.push_region_metadata_with_crc(kind, logical_id, size, 0);
    }

    /// Same as `push_region_metadata`, but also records a CRC32C that
    /// the orchestrator has already computed (e.g. while streaming
    /// payload from pinned memory). Zero means "CRC not available" —
    /// readers treat a zero CRC as "skip verification for this entry."
    pub fn push_region_metadata_with_crc(
        &mut self,
        kind: RegionKind,
        logical_id: u32,
        size: u64,
        crc32c: u32,
    ) {
        self.pending.push(PendingRegion {
            kind,
            logical_id,
            size,
            bytes: Vec::new(),
            crc32c,
        });
    }

    /// Compute the on-disk layout and build the `Snapshot` that
    /// describes it, but do not actually write anything.
    ///
    /// This is split out from `write_to` for two reasons:
    ///
    ///   1. Tests can inspect the computed offsets without going
    ///      through the I/O layer.
    ///   2. A future streaming writer will call `build_snapshot`
    ///      first (to stamp the header) and then stream payload
    ///      bytes region-by-region. Reusing this method means the
    ///      layout math has exactly one implementation.
    ///
    /// Layout rule: the first region's payload starts at
    /// `HEADER_SIZE + N * REGION_ENTRY_SIZE`, i.e. immediately
    /// after the prelude. Each subsequent region starts at the
    /// previous region's end. No padding, no alignment — that is
    /// a concern for a later O_DIRECT-capable writer.
    pub fn build_snapshot(&self) -> Snapshot {
        let mut table = RegionTable::new();
        let n = self.pending.len() as u64;
        // First payload byte sits right after the full prelude.
        let mut next_offset = HEADER_SIZE + n * REGION_ENTRY_SIZE;

        for region in &self.pending {
            let entry = RegionEntry::new(region.kind)
                .with_logical_id(region.logical_id)
                .with_size(region.size)
                .with_file_offset(next_offset)
                .with_crc32c(region.crc32c);
            table.push(entry);
            next_offset += region.size;
        }

        let snapshot = Snapshot::from_table(table);
        // If the caller stamped a vllm_commit hash on the writer,
        // propagate it onto the built header here. This is the
        // only place the writer reaches into `Snapshot` to
        // customize the header beyond the layout math above —
        // everything else the header encodes is derived from the
        // table.
        if let Some(commit) = self.vllm_commit {
            let stamped_header = snapshot.header().clone().with_vllm_commit(commit);
            Snapshot::from_header_and_table(stamped_header, snapshot.table().clone())
        } else {
            snapshot
        }
    }

    /// Total on-disk byte length of the file this writer will
    /// produce: `prelude_size + sum(region.bytes.len())`.
    ///
    /// Separate accessor so tests can assert file size without
    /// actually writing anything to disk.
    pub fn total_size(&self) -> u64 {
        let payload: u64 = self.pending.iter().map(|r| r.size).sum();
        let snapshot = self.build_snapshot();
        snapshot.prelude_size() + payload
    }

    /// Write the full file (prelude + payload) to the given sink.
    ///
    /// The sink can be a `Vec<u8>`, a `BufWriter<File>`, or
    /// anything else that implements `std::io::Write`. The method
    /// returns the number of bytes written on success, equal to
    /// `total_size()`.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<u64, SnapshotError> {
        // Stage 1: build the snapshot and write the prelude. This
        // reuses `Snapshot::write_to`, which is already tested.
        let snapshot = self.build_snapshot();
        let mut written = snapshot.write_to(writer)?;

        // Stage 2: stream each pending region's payload bytes in
        // the same order they were pushed. Because the snapshot
        // above was built from the same order, every entry's
        // `file_offset` already points at the right place.
        for region in &self.pending {
            writer.write_all(&region.bytes)?;
            written += region.bytes.len() as u64;
        }

        Ok(written)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::Snapshot;

    /// An empty writer queues zero regions.
    #[test]
    fn new_writer_is_empty() {
        let w = ByteRegionWriter::new();
        assert_eq!(w.len(), 0);
        assert!(w.is_empty());
    }

    /// `push_region` appends in order.
    #[test]
    fn push_region_appends() {
        let mut w = ByteRegionWriter::new();
        w.push_region(RegionKind::Weights, 0, vec![1, 2, 3]);
        w.push_region(RegionKind::Metadata, 0, vec![4, 5]);
        assert_eq!(w.len(), 2);
    }

    /// An empty writer builds a snapshot with zero regions.
    #[test]
    fn empty_writer_builds_empty_snapshot() {
        let w = ByteRegionWriter::new();
        let s = w.build_snapshot();
        assert_eq!(s.len(), 0);
        assert_eq!(s.header().num_regions(), 0);
    }

    /// Single-region layout: the region's `file_offset` sits
    /// exactly one prelude (header + one entry) past the start of
    /// the file.
    ///
    /// This pins the core layout rule: payload starts at
    /// `HEADER_SIZE + N * REGION_ENTRY_SIZE`. If this test ever
    /// fires, a future refactor has moved the payload start in a
    /// way that will break every existing writer in one shot.
    #[test]
    fn single_region_offset_sits_right_after_prelude() {
        let mut w = ByteRegionWriter::new();
        w.push_region(RegionKind::Weights, 0, vec![0xAA; 100]);
        let s = w.build_snapshot();
        let entry = s.table().get(0).unwrap();
        assert_eq!(entry.kind(), RegionKind::Weights);
        assert_eq!(entry.size(), 100);
        assert_eq!(entry.file_offset(), HEADER_SIZE + REGION_ENTRY_SIZE);
    }

    /// Multi-region layout: each region sits immediately after the
    /// previous one, with no gaps and no padding.
    ///
    /// This is the single most error-prone piece of a file writer
    /// (off-by-one in the running offset) so it gets an explicit
    /// test with three regions of different sizes.
    #[test]
    fn multi_region_offsets_are_back_to_back() {
        let mut w = ByteRegionWriter::new();
        w.push_region(RegionKind::Weights, 0, vec![0xAA; 10]);
        w.push_region(RegionKind::KvLiveBlock, 7, vec![0xBB; 20]);
        w.push_region(RegionKind::Metadata, 0, vec![0xCC; 30]);
        let s = w.build_snapshot();

        let prelude_end = HEADER_SIZE + 3 * REGION_ENTRY_SIZE;

        let e0 = s.table().get(0).unwrap();
        assert_eq!(e0.file_offset(), prelude_end);
        assert_eq!(e0.size(), 10);

        let e1 = s.table().get(1).unwrap();
        assert_eq!(e1.file_offset(), prelude_end + 10);
        assert_eq!(e1.size(), 20);
        assert_eq!(e1.logical_id(), 7);

        let e2 = s.table().get(2).unwrap();
        assert_eq!(e2.file_offset(), prelude_end + 10 + 20);
        assert_eq!(e2.size(), 30);
    }

    /// `total_size` matches `prelude_size + sum(payload sizes)`.
    #[test]
    fn total_size_sums_prelude_and_payloads() {
        let mut w = ByteRegionWriter::new();
        w.push_region(RegionKind::Weights, 0, vec![0; 100]);
        w.push_region(RegionKind::Metadata, 0, vec![0; 50]);
        let s = w.build_snapshot();
        assert_eq!(w.total_size(), s.prelude_size() + 150);
    }

    /// `write_to` produces a buffer whose length equals
    /// `total_size()`. Length-only; content comes next.
    #[test]
    fn write_to_produces_expected_length() {
        let mut w = ByteRegionWriter::new();
        w.push_region(RegionKind::Weights, 0, vec![0xAA; 8]);
        w.push_region(RegionKind::Metadata, 0, vec![0xBB; 4]);
        let mut sink: Vec<u8> = Vec::new();
        let written = w.write_to(&mut sink).expect("write_to");
        assert_eq!(written, w.total_size());
        assert_eq!(sink.len() as u64, w.total_size());
    }

    /// Payload bytes appear at the file offsets the snapshot's
    /// entries point at. This is the property that makes the whole
    /// file actually readable: if the entry says "my payload is at
    /// offset 4128," the bytes at offset 4128 had better be the
    /// bytes the caller pushed.
    #[test]
    fn payload_bytes_sit_at_entry_offsets() {
        let mut w = ByteRegionWriter::new();
        w.push_region(RegionKind::Weights, 0, vec![0xAA; 8]);
        w.push_region(RegionKind::Metadata, 0, vec![0xBB; 4]);

        let mut sink: Vec<u8> = Vec::new();
        w.write_to(&mut sink).unwrap();

        let s = w.build_snapshot();
        let e0 = s.table().get(0).unwrap();
        let e1 = s.table().get(1).unwrap();

        let start0 = e0.file_offset() as usize;
        let end0 = start0 + e0.size() as usize;
        let start1 = e1.file_offset() as usize;
        let end1 = start1 + e1.size() as usize;

        assert_eq!(&sink[start0..end0], &[0xAA; 8]);
        assert_eq!(&sink[start1..end1], &[0xBB; 4]);
    }

    /// `set_vllm_commit` propagates onto the snapshot header
    /// that `build_snapshot` produces.
    ///
    /// This is the seam between the orchestrator layer (which
    /// knows the vLLM commit hash) and the file format (which
    /// stores it). Pinning the propagation as a test means the
    /// day someone refactors `build_snapshot` and forgets to
    /// carry the stamp across, the failure shows up here and not
    /// at restore time on a real machine.
    #[test]
    fn set_vllm_commit_propagates_to_snapshot_header() {
        let hash = *b"cafef00dcafef00dcafef00dcafef00dcafef00d";
        let mut w = ByteRegionWriter::new();
        w.set_vllm_commit(hash);
        w.push_region(RegionKind::Weights, 0, vec![1, 2, 3, 4]);
        let s = w.build_snapshot();
        assert_eq!(s.header().vllm_commit(), &hash);
    }

    /// A writer that never calls `set_vllm_commit` produces a
    /// header whose slot is all zeros (the "unset" sentinel).
    #[test]
    fn unset_vllm_commit_leaves_header_slot_zero() {
        let mut w = ByteRegionWriter::new();
        w.push_region(RegionKind::Weights, 0, vec![1, 2, 3, 4]);
        let s = w.build_snapshot();
        assert_eq!(s.header().vllm_commit(), &[0u8; 40]);
    }

    /// `push_region` computes and stamps a CRC32C over the payload
    /// bytes onto the resulting snapshot entry.
    ///
    /// Pinning this behavior means a future refactor that forgets to
    /// propagate the CRC through `build_snapshot` will fail here
    /// immediately, not silently on a restore at 3am.
    #[test]
    fn push_region_stamps_crc32c() {
        let payload = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let expected = crc32c::crc32c(&payload);
        let mut w = ByteRegionWriter::new();
        w.push_region(RegionKind::Weights, 0, payload);
        let s = w.build_snapshot();
        let entry = s.table().get(0).unwrap();
        assert_eq!(entry.crc32c(), expected);
        assert_ne!(expected, 0, "non-empty payload must have non-zero CRC");
    }

    /// `push_region_metadata_with_crc` records the caller-supplied
    /// CRC verbatim (for orchestrators that compute the CRC while
    /// streaming from pinned host memory).
    #[test]
    fn push_region_metadata_with_crc_records_verbatim() {
        let mut w = ByteRegionWriter::new();
        w.push_region_metadata_with_crc(RegionKind::Weights, 0, 1024, 0xFEED_BEEF);
        let s = w.build_snapshot();
        assert_eq!(s.table().get(0).unwrap().crc32c(), 0xFEED_BEEF);
    }

    /// Round-trip: write a complete file into a `Vec<u8>`, parse
    /// the prelude back out of the same bytes, and assert the
    /// parsed snapshot equals the one the writer built.
    ///
    /// This is the first end-to-end test of the full file format:
    /// writer → bytes → parser → structs. If it ever fires, the
    /// writer and the parser disagree about the layout.
    #[test]
    fn end_to_end_round_trip_through_vec() {
        let mut w = ByteRegionWriter::new();
        w.push_region(RegionKind::Weights, 0, vec![1, 2, 3, 4, 5]);
        w.push_region(RegionKind::KvLiveBlock, 99, vec![6, 7, 8]);
        let original = w.build_snapshot();

        let mut sink: Vec<u8> = Vec::new();
        w.write_to(&mut sink).unwrap();

        // Parse the prelude only — the payload lives past
        // `prelude_size` and `from_prelude_bytes` tolerates the
        // extra bytes.
        let parsed = Snapshot::from_prelude_bytes(&sink).expect("parse prelude");
        assert_eq!(parsed, original);
    }
}
