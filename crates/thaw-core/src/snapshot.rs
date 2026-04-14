// crates/thaw-core/src/snapshot.rs
//
// =============================================================================
// THE SNAPSHOT PRELUDE
// =============================================================================
//
// `header.rs` describes the 4096-byte header. `region.rs` describes the
// fixed-size table that lives immediately after the header. This file is
// where those two pieces meet. The "prelude" of a `.thaw` file is the
// byte stream:
//
//     [ 4096-byte header ] [ num_regions * 32-byte table entries ]
//
// and then everything after that is payload (weight blob, KV blocks,
// metadata blob), whose bytes are the I/O layer's problem.
//
// WHY A SEPARATE MODULE
// --------------------
//
// We could have added a `to_prelude_bytes(table: &RegionTable)` method
// on `SnapshotHeader` and called it a day. We are not doing that, for
// two reasons:
//
//   1. The "header + table" pair has its own invariants that neither
//      half can enforce alone. The biggest one: the header's
//      `num_regions` field must match `table.len()`. If those ever
//      disagree, a reader will parse garbage — and the only place we
//      can cheaply catch the mismatch is the seam between them, which
//      is exactly here.
//
//   2. Keeping the composition in its own type lets the header and
//      table modules stay focused on their own byte layouts without
//      learning about each other. When we later add a real file
//      writer on top of this, it will take a `Snapshot` (this type)
//      and not juggle a header and a table separately.
//
// TDD STORY
// ---------
//
// Same discipline as every other module in this crate: one claim per
// test, the simplest failing test first, minimum code to make it pass.
// The very first test just builds an empty snapshot and asks how many
// regions it has. Tiny, but it pins that the type exists and the
// accessor works.
//
// =============================================================================

use std::io::{self, Read, Write};

use thiserror::Error;

use crate::header::{HeaderError, SnapshotHeader, HEADER_SIZE};
use crate::region::{RegionError, RegionTable, REGION_ENTRY_SIZE};

/// The in-memory representation of a `.thaw` file's prelude: a fixed
/// header glued to an ordered region table.
///
/// A `Snapshot` owns both halves together so that the single most
/// important invariant — "the header's `num_regions` field equals the
/// table's `len()`" — can be upheld at exactly one place: the
/// constructors. Callers who hand out a `Snapshot` never have to worry
/// about the two halves drifting apart, because the only ways to
/// build one force the header to be stamped from the table.
///
/// This type intentionally knows nothing about payload bytes
/// (weights, KV blocks, metadata). Its job is the first
/// `HEADER_SIZE + num_regions * REGION_ENTRY_SIZE` bytes of the file,
/// and nothing else. A later writer module will take `Snapshot` plus
/// the payload streams and produce the full on-disk file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    header: SnapshotHeader,
    table: RegionTable,
}

impl Snapshot {
    /// Build an empty snapshot: a fresh header and an empty table.
    ///
    /// The header's `num_regions` is zero and `region_table_offset`
    /// defaults to `HEADER_SIZE` (via `SnapshotHeader::new()`), which
    /// is where our writer will always park the table. Callers who
    /// need a table full of entries should build the `RegionTable`
    /// first and then use [`Snapshot::from_table`].
    pub fn new() -> Self {
        Snapshot {
            header: SnapshotHeader::new(),
            table: RegionTable::new(),
        }
    }

    /// Build a snapshot whose header is automatically synced to the
    /// given table. This is the only place in the crate that touches
    /// the "num_regions must equal table.len()" invariant, and it
    /// upholds it by construction: the header is freshly stamped
    /// from the table every time.
    ///
    /// `region_table_offset` stays at the default `HEADER_SIZE` —
    /// there is no reason for a caller at this layer to want it
    /// elsewhere, and the writer module can override it later if we
    /// ever need a non-default layout.
    pub fn from_table(table: RegionTable) -> Self {
        let header = SnapshotHeader::new().with_num_regions(table.len() as u64);
        Snapshot { header, table }
    }

    /// Build a snapshot from an explicit header and table.
    ///
    /// Unlike `from_table`, this constructor does *not* stamp the
    /// header's `num_regions` field — the caller is responsible for
    /// ensuring it matches `table.len()`. The header's
    /// `num_regions` is forcibly overwritten to match, because the
    /// alternative (returning an error, or silently disagreeing)
    /// would make the type unsafe to hand around.
    ///
    /// The intended use is passthrough: a caller has a header it
    /// already customized (e.g. a stamped `vllm_commit`) and a
    /// table it built separately, and wants to glue them together
    /// without losing its customizations. The writer module uses
    /// this to propagate `vllm_commit` from the `ByteRegionWriter`
    /// onto the header it hands back to callers.
    pub fn from_header_and_table(header: SnapshotHeader, table: RegionTable) -> Self {
        let header = header.with_num_regions(table.len() as u64);
        Snapshot { header, table }
    }

    /// Read-only access to the header.
    pub fn header(&self) -> &SnapshotHeader {
        &self.header
    }

    /// Read-only access to the region table.
    pub fn table(&self) -> &RegionTable {
        &self.table
    }

    /// Convenience: the number of region-table entries.
    ///
    /// This is the same number as `self.header().num_regions()` and
    /// `self.table().len()`, because the invariant upheld by
    /// `from_table` guarantees they match. It exists because callers
    /// who just want a count should not have to pick between two
    /// accessors that, by contract, return the same thing.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// True if the snapshot has no regions.
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// The total size, in bytes, of the serialized prelude.
    ///
    /// Equal to `HEADER_SIZE + num_regions * REGION_ENTRY_SIZE`. This
    /// is the exact length of the byte stream that
    /// [`Snapshot::to_prelude_bytes`] produces, and the exact number
    /// of bytes a reader has to pull off disk before it knows where
    /// the payload lives.
    pub fn prelude_size(&self) -> u64 {
        HEADER_SIZE + (self.table.len() as u64) * REGION_ENTRY_SIZE
    }

    /// Serialize the prelude to a single contiguous byte buffer.
    ///
    /// Layout:
    ///
    ///   [ 0 .. HEADER_SIZE )                         — header
    ///   [ HEADER_SIZE .. HEADER_SIZE + N*32 )        — region table
    ///
    /// where N is `self.len()`. The result is exactly
    /// `self.prelude_size()` bytes long. The caller can write this
    /// straight to a `.thaw` file and then append payload bytes; no
    /// further framing or padding is required at this layer.
    ///
    /// The table sits at byte offset `HEADER_SIZE`, which matches
    /// `header().region_table_offset()`. That redundancy is
    /// intentional — a reader that trusts the field and a reader
    /// that trusts the layout must agree, and this method is the
    /// single place that pins them together.
    pub fn to_prelude_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.prelude_size() as usize);
        buf.extend_from_slice(&self.header.to_bytes());
        buf.extend_from_slice(&self.table.to_bytes());
        buf
    }

    /// Parse a prelude back into a `Snapshot`.
    ///
    /// The input must be at least `HEADER_SIZE + num_regions * 32`
    /// bytes long, where `num_regions` is read from the header. The
    /// parser is strict: any mismatch between the header's claimed
    /// region count and the actual bytes available is a typed error,
    /// not a silent truncation. Trailing bytes (i.e. the payload
    /// section) are tolerated and ignored — a reader may hand us the
    /// whole file and only want the prelude.
    ///
    /// Round-trip: for every `Snapshot` built via `new()` or
    /// `from_table()`,
    /// `Snapshot::from_prelude_bytes(s.to_prelude_bytes()) == Ok(s)`.
    pub fn from_prelude_bytes(bytes: &[u8]) -> Result<Self, SnapshotError> {
        // Step 1: parse the header. It has its own length and field
        // checks, so anything wrong here surfaces as a typed
        // `HeaderError` that we wrap.
        let header = SnapshotHeader::from_bytes(bytes).map_err(SnapshotError::Header)?;

        // Step 2: figure out where the table lives and how long it
        // should be. We trust the header's own fields here: the
        // writer we control always stamps `region_table_offset =
        // HEADER_SIZE`, but the format allows for other values and
        // we honor them so a future writer that moves the table is
        // not silently broken by this parser.
        let table_offset = header.region_table_offset() as usize;
        let num_regions = header.num_regions() as usize;
        let table_bytes_needed = num_regions * REGION_ENTRY_SIZE as usize;
        let table_end = table_offset
            .checked_add(table_bytes_needed)
            .ok_or(SnapshotError::OffsetOverflow)?;

        if bytes.len() < table_end {
            return Err(SnapshotError::TruncatedTable {
                got: bytes.len(),
                need: table_end,
            });
        }

        // Step 3: parse the table. `from_bytes` already handles
        // short input inside the slice we give it, but we've
        // length-checked the outer buffer above so that error
        // variant is only reachable for genuinely malformed entries
        // (e.g. unknown region kind).
        let table = RegionTable::from_bytes(&bytes[table_offset..table_end], num_regions)
            .map_err(SnapshotError::Region)?;

        Ok(Snapshot { header, table })
    }

    /// Write the prelude to any `std::io::Write` sink.
    ///
    /// This is the thinnest possible adapter from the in-memory
    /// byte form to a real writer. It exists so that test code can
    /// pass a `Vec<u8>` or `Cursor`, and production code can pass a
    /// `BufWriter<File>`, and neither has to care about the other.
    ///
    /// The I/O layer proper — pinned buffers, O_DIRECT, alignment,
    /// GDS — lives in a separate module. That layer is GPU-adjacent
    /// and will not compile on a Mac. This method, by contrast, is
    /// pure std::io and runs in the unit-test sandbox like every
    /// other piece of `thaw-core`.
    ///
    /// Returns the number of bytes written on success, which is
    /// always equal to `self.prelude_size()`. The count is returned
    /// (not just `()`) so that callers wiring this into a larger
    /// file writer know exactly how far the cursor advanced without
    /// having to recompute `prelude_size` themselves.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<u64, SnapshotError> {
        let bytes = self.to_prelude_bytes();
        writer.write_all(&bytes)?;
        Ok(bytes.len() as u64)
    }

    /// Read a prelude from any `std::io::Read` source.
    ///
    /// This one is a little subtler than `write_to`. We do not know
    /// in advance how many bytes the prelude will be — that depends
    /// on the header's `num_regions` field, which we cannot read
    /// without first having the header. So the read is two-stage:
    ///
    ///   1. Pull exactly `HEADER_SIZE` bytes and parse them into a
    ///      `SnapshotHeader`. This tells us how many entries follow
    ///      and where they live.
    ///   2. Pull exactly `num_regions * REGION_ENTRY_SIZE` bytes and
    ///      hand the concatenated buffer to `from_prelude_bytes`.
    ///
    /// We reuse `from_prelude_bytes` for the actual parsing rather
    /// than duplicating the validation logic. That means every
    /// error variant already tested by the byte-slice parser is
    /// also reachable here — no new test matrix required.
    ///
    /// Short reads are rejected as `Io` errors; `read_exact` returns
    /// `ErrorKind::UnexpectedEof` for truncated streams, which is
    /// exactly the behavior we want.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self, SnapshotError> {
        // Stage 1: the header is a fixed size, so we can read it
        // straight into a stack-friendly `Vec`.
        let mut header_buf = vec![0u8; HEADER_SIZE as usize];
        reader
            .read_exact(&mut header_buf)
            ?;

        let header = SnapshotHeader::from_bytes(&header_buf).map_err(SnapshotError::Header)?;

        // Stage 2: figure out how many table bytes to pull. We
        // trust the header's own `num_regions` and
        // `region_table_offset` fields for the same reason
        // `from_prelude_bytes` does — a future format that moves
        // the table would break this parser otherwise, and we want
        // that future to be a deliberate choice rather than an
        // accident.
        //
        // For the current layout, `region_table_offset` always
        // equals `HEADER_SIZE`, so the table bytes begin immediately
        // after the header bytes we just consumed. If a future
        // format parks the table somewhere else, this method would
        // need to `seek` past the gap — we'll cross that bridge
        // when we have a real reason to build it.
        if header.region_table_offset() != HEADER_SIZE {
            return Err(SnapshotError::UnsupportedLayout {
                region_table_offset: header.region_table_offset(),
            });
        }

        let num_regions = header.num_regions() as usize;
        let table_len = num_regions * REGION_ENTRY_SIZE as usize;

        let mut table_buf = vec![0u8; table_len];
        reader
            .read_exact(&mut table_buf)
            ?;

        // Glue the two stages back together and let the existing
        // byte-slice parser do the heavy lifting. One implementation
        // of the parse logic, two front doors.
        let mut combined = header_buf;
        combined.extend_from_slice(&table_buf);
        Snapshot::from_prelude_bytes(&combined)
    }
}

impl Default for Snapshot {
    fn default() -> Self {
        Snapshot::new()
    }
}

/// Errors produced by the snapshot prelude parser.
///
/// Non-exhaustive so new variants can be added without breaking
/// downstream crates that match on this type.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum SnapshotError {
    /// The header portion failed to parse. The wrapped `HeaderError`
    /// carries the specific reason (too short, bad magic, wrong
    /// version).
    #[error("header parse failed: {0}")]
    Header(#[from] HeaderError),

    /// The region table portion failed to parse. The wrapped
    /// `RegionError` carries the specific reason (too short for the
    /// claimed count, unknown kind discriminant).
    #[error("region table parse failed: {0}")]
    Region(#[from] RegionError),

    /// The input buffer was not long enough to contain the header's
    /// claimed region table. `got` is what we saw, `need` is what
    /// the header told us to expect. This variant exists separately
    /// from `Region(RegionError::TooShort)` because the two errors
    /// have different meanings: this one means "the header and the
    /// outer buffer disagree," while `Region::TooShort` means "the
    /// table slice we carved out is itself malformed."
    #[error("snapshot prelude is truncated: got {got} bytes, need {need}")]
    TruncatedTable { got: usize, need: usize },

    /// Arithmetic on the header's offset and count fields would
    /// overflow a usize. This is only reachable on a maliciously
    /// crafted or corrupted file — no real writer produces values
    /// anywhere near the overflow limit — but we prefer a typed
    /// error to a silent wraparound.
    #[error("snapshot prelude offsets overflow usize")]
    OffsetOverflow,

    /// An underlying `std::io` error occurred while reading or
    /// writing the prelude. Kept as a separate variant from the
    /// format errors because an I/O failure (disk full, broken
    /// pipe) is a completely different class of problem from a
    /// format parse failure and callers usually want to branch
    /// on it separately.
    ///
    /// `io::Error` does not implement `PartialEq`, so we carry its
    /// `ErrorKind` and the `to_string()` message instead of the
    /// whole error. That keeps `SnapshotError` itself comparable
    /// in tests (via `#[derive(PartialEq, Eq)]`) while still
    /// preserving the signal a caller needs to diagnose the
    /// failure.
    #[error("i/o error during snapshot prelude: {kind:?}: {message}")]
    Io {
        kind: io::ErrorKind,
        message: String,
    },

    /// The header claimed a `region_table_offset` that the current
    /// streaming reader does not support. Right now that means
    /// "anything other than `HEADER_SIZE`," because we do not yet
    /// bother to `seek` past a gap between the header and the
    /// table. If a future writer starts using a non-default
    /// offset, this variant is the signal to teach the reader
    /// about it.
    #[error("unsupported snapshot layout: region_table_offset = {region_table_offset} (expected {})", HEADER_SIZE)]
    UnsupportedLayout { region_table_offset: u64 },
}

// Manual conversion from `io::Error` into our typed error. We can't
// `#[from]` it like we do for `HeaderError` and `RegionError` because
// `io::Error` isn't `PartialEq`, so we flatten it into kind + message
// so that the enum as a whole remains comparable for tests.
impl From<io::Error> for SnapshotError {
    fn from(err: io::Error) -> Self {
        SnapshotError::Io {
            kind: err.kind(),
            message: err.to_string(),
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================
//
// The tests below follow the same one-claim-per-test discipline as the
// header and region modules. Each test is named for what it asserts
// and carries a comment explaining why the claim matters.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::{RegionEntry, RegionKind};

    /// A fresh snapshot has zero regions. Smallest possible claim:
    /// the type exists, the constructor runs, and the `len` accessor
    /// agrees with the table it holds.
    #[test]
    fn new_snapshot_is_empty() {
        let s = Snapshot::new();
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
    }

    /// A fresh snapshot's header reports zero regions.
    ///
    /// This is the first place the "header and table agree" invariant
    /// gets tested. If a future refactor ever lets `new()` stamp a
    /// non-zero `num_regions` without actually adding entries to the
    /// table, this test fires.
    #[test]
    fn new_snapshot_header_and_table_agree_on_count() {
        let s = Snapshot::new();
        assert_eq!(s.header().num_regions() as usize, s.table().len());
        assert_eq!(s.header().num_regions(), 0);
    }

    /// `from_table` stamps the header's `num_regions` from the table
    /// it was built with. This is the invariant-enforcing path, and
    /// it is the whole reason `Snapshot` exists as a separate type.
    #[test]
    fn from_table_syncs_header_num_regions() {
        let mut t = RegionTable::new();
        t.push(RegionEntry::new(RegionKind::Weights).with_size(100));
        t.push(RegionEntry::new(RegionKind::Metadata).with_size(200));
        t.push(RegionEntry::new(RegionKind::KvLiveBlock).with_logical_id(5));

        let s = Snapshot::from_table(t);
        assert_eq!(s.header().num_regions(), 3);
        assert_eq!(s.table().len(), 3);
    }

    /// The prelude size of an empty snapshot is exactly `HEADER_SIZE`.
    /// No table bytes yet, because the table is empty.
    #[test]
    fn empty_snapshot_prelude_size_is_header_size() {
        let s = Snapshot::new();
        assert_eq!(s.prelude_size(), HEADER_SIZE);
    }

    /// The prelude size of a non-empty snapshot is
    /// `HEADER_SIZE + N * REGION_ENTRY_SIZE`.
    ///
    /// This is the formula every future reader and writer depends on
    /// to know where the payload begins. Pinning it in a test means
    /// no future refactor can quietly change the layout.
    #[test]
    fn non_empty_snapshot_prelude_size_counts_table() {
        let mut t = RegionTable::new();
        t.push(RegionEntry::new(RegionKind::Weights));
        t.push(RegionEntry::new(RegionKind::Metadata));
        let s = Snapshot::from_table(t);
        assert_eq!(s.prelude_size(), HEADER_SIZE + 2 * REGION_ENTRY_SIZE);
    }

    /// `to_prelude_bytes` produces a buffer of exactly `prelude_size`
    /// bytes. Length-only; contents come next.
    #[test]
    fn to_prelude_bytes_has_expected_length() {
        let mut t = RegionTable::new();
        t.push(RegionEntry::new(RegionKind::Weights).with_size(16));
        t.push(RegionEntry::new(RegionKind::KvLiveBlock).with_logical_id(42));
        let s = Snapshot::from_table(t);
        let bytes = s.to_prelude_bytes();
        assert_eq!(bytes.len() as u64, s.prelude_size());
    }

    /// The first `HEADER_SIZE` bytes of the prelude are byte-for-byte
    /// equal to the header's own `to_bytes()` output.
    ///
    /// This pins "the header lives at offset 0 of the prelude,"
    /// which is the universal convention and the only reason file
    /// magic detection works.
    #[test]
    fn prelude_starts_with_header_bytes() {
        let mut t = RegionTable::new();
        t.push(RegionEntry::new(RegionKind::Weights).with_size(1));
        let s = Snapshot::from_table(t);
        let bytes = s.to_prelude_bytes();
        let header_bytes = s.header().to_bytes();
        assert_eq!(&bytes[..HEADER_SIZE as usize], header_bytes.as_slice());
    }

    /// The bytes at `[HEADER_SIZE .. prelude_size)` are byte-for-byte
    /// equal to the table's own `to_bytes()` output.
    ///
    /// Equivalent pinning for the table: "the table begins at
    /// `region_table_offset`, which defaults to `HEADER_SIZE`."
    #[test]
    fn prelude_table_lives_right_after_header() {
        let mut t = RegionTable::new();
        t.push(
            RegionEntry::new(RegionKind::Weights)
                .with_size(0x1234)
                .with_file_offset(0x5000),
        );
        t.push(
            RegionEntry::new(RegionKind::Metadata)
                .with_size(0x9)
                .with_file_offset(0x6000),
        );
        let s = Snapshot::from_table(t);
        let bytes = s.to_prelude_bytes();
        let table_bytes = s.table().to_bytes();
        assert_eq!(
            &bytes[HEADER_SIZE as usize..],
            table_bytes.as_slice(),
            "table bytes should sit immediately after the header"
        );
    }

    /// Round-trip: an empty snapshot survives serialization and
    /// parsing. The smallest round-trip claim; anything beyond this
    /// is a bonus.
    #[test]
    fn empty_snapshot_round_trips() {
        let original = Snapshot::new();
        let bytes = original.to_prelude_bytes();
        let parsed = Snapshot::from_prelude_bytes(&bytes).expect("valid prelude should parse");
        assert_eq!(parsed, original);
    }

    /// Round-trip: a non-empty snapshot with multiple region kinds
    /// survives. This is the test that will fire first if any future
    /// change to the header, table, or composition logic breaks the
    /// format contract.
    #[test]
    fn non_empty_snapshot_round_trips() {
        let mut t = RegionTable::new();
        t.push(
            RegionEntry::new(RegionKind::Weights)
                .with_size(16 * 1024 * 1024 * 1024)
                .with_file_offset(HEADER_SIZE + 3 * REGION_ENTRY_SIZE),
        );
        t.push(
            RegionEntry::new(RegionKind::KvLiveBlock)
                .with_logical_id(17)
                .with_size(32 * 1024)
                .with_file_offset(999_999),
        );
        t.push(
            RegionEntry::new(RegionKind::Metadata)
                .with_size(512)
                .with_file_offset(1_000_000),
        );
        let original = Snapshot::from_table(t);
        let bytes = original.to_prelude_bytes();
        let parsed = Snapshot::from_prelude_bytes(&bytes).expect("should parse");
        assert_eq!(parsed, original);
    }

    /// The parser accepts a buffer that contains the full prelude
    /// plus trailing payload bytes. The payload is ignored.
    ///
    /// This is the everyday case for real readers: they mmap or read
    /// a whole `.thaw` file and hand the whole slice to
    /// `from_prelude_bytes`, which should only consume the first
    /// `prelude_size` bytes.
    #[test]
    fn parser_tolerates_trailing_payload_bytes() {
        let mut t = RegionTable::new();
        t.push(RegionEntry::new(RegionKind::Weights).with_size(10));
        let original = Snapshot::from_table(t);
        let mut bytes = original.to_prelude_bytes();
        // Append 1 MiB of "payload" garbage. A real reader would
        // seek to entry file_offsets to interpret these; we don't
        // care what's there, just that the prelude parser ignores
        // them.
        bytes.extend(std::iter::repeat(0xAB).take(1024 * 1024));
        let parsed = Snapshot::from_prelude_bytes(&bytes).expect("should parse");
        assert_eq!(parsed, original);
    }

    /// The parser rejects a buffer whose claimed region table runs
    /// past the end of the input, with a typed `TruncatedTable`
    /// error (not a panic and not `RegionError::TooShort`).
    #[test]
    fn parser_rejects_truncated_table() {
        let mut t = RegionTable::new();
        t.push(RegionEntry::new(RegionKind::Weights));
        t.push(RegionEntry::new(RegionKind::Metadata));
        t.push(RegionEntry::new(RegionKind::KvLiveBlock));
        let original = Snapshot::from_table(t);
        let mut bytes = original.to_prelude_bytes();
        // Truncate the table by one full entry. The header still
        // claims 3 entries, but only 2 worth of bytes are present
        // after the header.
        bytes.truncate(bytes.len() - REGION_ENTRY_SIZE as usize);
        let err = Snapshot::from_prelude_bytes(&bytes).expect_err("should reject");
        assert!(matches!(err, SnapshotError::TruncatedTable { .. }));
    }

    /// The parser propagates a bad-magic header error as
    /// `SnapshotError::Header`, not as a panic.
    #[test]
    fn parser_wraps_header_errors() {
        let s = Snapshot::new();
        let mut bytes = s.to_prelude_bytes();
        bytes[0] = b'X'; // Corrupt the magic.
        let err = Snapshot::from_prelude_bytes(&bytes).expect_err("should reject");
        assert!(matches!(err, SnapshotError::Header(HeaderError::BadMagic { .. })));
    }

    // ------------------------------------------------------------------
    // write_to / read_from — std::io adapters
    // ------------------------------------------------------------------
    //
    // These tests exercise the streaming front door into the same
    // parser/serializer that the byte-slice tests above already cover.
    // They use `Vec<u8>` as a `Write` sink and `std::io::Cursor` as a
    // `Read` source, so there is no actual filesystem involved —
    // these tests run the same on Mac, Linux, and Windows, with no
    // GPU and no privileged I/O. The real, GPU-adjacent I/O layer
    // (pinned buffers, O_DIRECT, GDS) will live in a separate crate
    // and have its own test story.

    /// `write_to` writes exactly `prelude_size()` bytes into a
    /// `Vec<u8>`. The count returned by `write_to` matches.
    #[test]
    fn write_to_vec_matches_prelude_size() {
        let mut t = RegionTable::new();
        t.push(RegionEntry::new(RegionKind::Weights).with_size(64));
        let s = Snapshot::from_table(t);

        let mut sink: Vec<u8> = Vec::new();
        let written = s.write_to(&mut sink).expect("write_to should succeed");
        assert_eq!(written, s.prelude_size());
        assert_eq!(sink.len() as u64, s.prelude_size());
    }

    /// The bytes produced by `write_to` are byte-for-byte equal to
    /// the bytes produced by `to_prelude_bytes`. Pinning this means
    /// the two front doors are guaranteed to agree — a future
    /// refactor of either cannot silently diverge.
    #[test]
    fn write_to_bytes_match_to_prelude_bytes() {
        let mut t = RegionTable::new();
        t.push(
            RegionEntry::new(RegionKind::Weights)
                .with_size(0xDEAD_BEEF)
                .with_file_offset(0x1000),
        );
        t.push(
            RegionEntry::new(RegionKind::KvLiveBlock)
                .with_logical_id(3)
                .with_size(0x2000)
                .with_file_offset(0x2000),
        );
        let s = Snapshot::from_table(t);

        let mut sink: Vec<u8> = Vec::new();
        s.write_to(&mut sink).unwrap();
        assert_eq!(sink, s.to_prelude_bytes());
    }

    /// `read_from` parses a `Cursor<Vec<u8>>` back into a snapshot
    /// equal to the original. The streaming round-trip.
    #[test]
    fn read_from_cursor_round_trips() {
        let mut t = RegionTable::new();
        t.push(
            RegionEntry::new(RegionKind::Weights)
                .with_size(16 * 1024 * 1024)
                .with_file_offset(0x1000),
        );
        t.push(
            RegionEntry::new(RegionKind::Metadata)
                .with_size(0x200)
                .with_file_offset(0x2000),
        );
        let original = Snapshot::from_table(t);

        let mut sink: Vec<u8> = Vec::new();
        original.write_to(&mut sink).unwrap();

        let mut cursor = std::io::Cursor::new(sink);
        let parsed = Snapshot::read_from(&mut cursor).expect("read_from should succeed");
        assert_eq!(parsed, original);
    }

    /// `read_from` advances the cursor exactly `prelude_size()`
    /// bytes and no more. This pins the contract that callers can
    /// stream more data (the payload) from the same reader after
    /// the prelude has been consumed.
    #[test]
    fn read_from_consumes_exactly_prelude_size_bytes() {
        let mut t = RegionTable::new();
        t.push(RegionEntry::new(RegionKind::Weights).with_size(10));
        let original = Snapshot::from_table(t);

        let mut buf: Vec<u8> = Vec::new();
        original.write_to(&mut buf).unwrap();
        // Append a payload tail so we can tell how far the cursor
        // has moved after the prelude is parsed.
        buf.extend_from_slice(b"PAYLOAD");

        let mut cursor = std::io::Cursor::new(buf);
        let parsed = Snapshot::read_from(&mut cursor).expect("round trip");
        assert_eq!(parsed, original);
        assert_eq!(cursor.position(), original.prelude_size());

        // The next bytes in the cursor should be the payload tail,
        // completely untouched. This is the property that lets a
        // real file reader stream-parse the prelude and then hand
        // the same reader to the payload layer.
        let mut tail = Vec::new();
        std::io::Read::read_to_end(&mut cursor, &mut tail).unwrap();
        assert_eq!(tail, b"PAYLOAD");
    }

    /// `read_from` surfaces a truncated header as a typed `Io`
    /// error (specifically, `ErrorKind::UnexpectedEof` coming out
    /// of `read_exact`). We assert on the variant and the kind,
    /// not the message, to keep the test resilient to std changes.
    #[test]
    fn read_from_rejects_short_header() {
        let bytes = vec![0u8; 100]; // way short of HEADER_SIZE
        let mut cursor = std::io::Cursor::new(bytes);
        let err = Snapshot::read_from(&mut cursor).expect_err("should reject");
        match err {
            SnapshotError::Io { kind, .. } => {
                assert_eq!(kind, io::ErrorKind::UnexpectedEof);
            }
            other => panic!("expected Io error, got {other:?}"),
        }
    }

    /// `read_from` surfaces a truncated table as a typed `Io`
    /// error, not as `TruncatedTable`. The distinction matters:
    /// `TruncatedTable` is for a complete buffer that claims more
    /// entries than it contains; a short read mid-stream is an
    /// `io::Error` with `UnexpectedEof`, because that is what
    /// `read_exact` produces.
    #[test]
    fn read_from_rejects_short_table() {
        let mut t = RegionTable::new();
        t.push(RegionEntry::new(RegionKind::Weights));
        t.push(RegionEntry::new(RegionKind::Metadata));
        let s = Snapshot::from_table(t);

        let mut buf: Vec<u8> = Vec::new();
        s.write_to(&mut buf).unwrap();
        // Chop off the last 10 bytes of the table.
        buf.truncate(buf.len() - 10);

        let mut cursor = std::io::Cursor::new(buf);
        let err = Snapshot::read_from(&mut cursor).expect_err("should reject");
        match err {
            SnapshotError::Io { kind, .. } => {
                assert_eq!(kind, io::ErrorKind::UnexpectedEof);
            }
            other => panic!("expected Io error, got {other:?}"),
        }
    }

    /// The parser propagates a bad-kind region error as
    /// `SnapshotError::Region`, not as a panic. This is the
    /// counterpart to the header test above, for the table half of
    /// the prelude.
    #[test]
    fn parser_wraps_region_errors() {
        let mut t = RegionTable::new();
        t.push(RegionEntry::new(RegionKind::Weights));
        let s = Snapshot::from_table(t);
        let mut bytes = s.to_prelude_bytes();
        // Stamp an unknown kind (99) over the first entry's kind
        // discriminant. The first entry starts at offset HEADER_SIZE.
        let entry_start = HEADER_SIZE as usize;
        bytes[entry_start..entry_start + 4].copy_from_slice(&99u32.to_le_bytes());
        let err = Snapshot::from_prelude_bytes(&bytes).expect_err("should reject");
        assert!(matches!(err, SnapshotError::Region(RegionError::UnknownKind { found: 99 })));
    }
}
