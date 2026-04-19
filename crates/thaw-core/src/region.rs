// crates/thaw-core/src/region.rs
//
// =============================================================================
// THE REGION TABLE
// =============================================================================
//
// A `.thaw` file is a header followed by a "payload," but the payload
// is not one flat blob — it's a sequence of named regions. This module
// defines the metadata that tells a reader where each region lives
// and what it contains.
//
// See DESIGN.md §3.3 for the big picture. The short version:
//
//   +-----------------+   offset 0
//   |     HEADER      |   (pinned at 4096 bytes — see header.rs)
//   +-----------------+   offset 4096
//   |  REGION TABLE   |   N fixed-size entries describing what's next
//   +-----------------+
//   |    PAYLOAD      |   the actual region bytes, back-to-back,
//   |   (weights,     |   page-aligned so each region can be O_DIRECT'd
//   |   KV blocks,    |   straight into pinned host memory
//   |   metadata)     |
//   +-----------------+
//
// Each region-table entry is exactly 32 bytes on disk. Fixed-size
// entries matter because the reader can seek to entry N with a single
// multiply; variable-length entries would force a serial walk from
// the start of the table every time a lookup happens.
//
// KINDS
// -----
//
// There are three region kinds, each serving a different purpose:
//
//   WEIGHTS         — the one big model-weight blob. Loaded into GPU
//                     memory via pinned-host DMA. One per snapshot.
//   KV_LIVE_BLOCKS  — only the live KV cache blocks, packed, tagged
//                     by their original integer block index so the
//                     scheduler's Python-side block table can be
//                     reconstructed on restore. See KEY_INSIGHTS.md
//                     insight 1 for why we can get away with integer
//                     indices instead of CUDA device pointers.
//   METADATA        — a small JSON blob with everything the Python
//                     layer needs to rebuild the scheduler state
//                     (block table, seq groups, refcounts, tokenizer
//                     config, vllm_commit hash).
//
// TDD STORY
// ---------
//
// Same rule as header.rs: one failing test at a time, minimum code
// to make it pass, commit, next test. Every field of `RegionEntry`
// comes from a specific test that demanded it.
//
// =============================================================================

use thiserror::Error;

/// The fixed on-disk size, in bytes, of a single region table entry.
///
/// Why a fixed size at all:
///   - Random access. Reader can seek to entry N with a single
///     multiply (`table_offset + N * REGION_ENTRY_SIZE`) instead of
///     walking the table from the start, which matters once you have
///     thousands of live KV blocks.
///   - Alignment. 32 is a power of two and divides evenly into a
///     4096-byte page, so the region table itself can be sized in
///     whole pages without awkward padding.
///
/// Why 32 bytes specifically:
///   - Current fields add up to 28 bytes (kind + logical_id + size +
///     file_offset + crc32c). The final 4 bytes are reserved and
///     zero-filled for forward compatibility (flags, compression tag,
///     etc.) without needing another format-version bump.
///   - 32 is cache-line-friendly (half a typical x86 line) and
///     matches the size of an AVX2 register, which is irrelevant for
///     correctness but does not hurt.
pub const REGION_ENTRY_SIZE: u64 = 32;

/// What kind of data a region holds.
///
/// The on-disk representation is a little-endian `u32`. Using a full
/// u32 instead of a u8 is wasteful on paper but gives us (a) room to
/// grow past 256 kinds if we ever need to, and (b) natural four-byte
/// alignment of the fields that follow. The `#[repr(u32)]` pins the
/// discriminant values so we can cast cleanly both directions.
///
/// Discriminant values are assigned explicitly and must never be
/// changed or reused — old snapshot files on disk will have them
/// baked in. Adding a new kind appends at the end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum RegionKind {
    /// The model weights. One per snapshot. Loaded into GPU memory
    /// via pinned-host DMA on restore. Typically 16 GB for Llama-3-8B
    /// in FP16.
    Weights = 0,

    /// A single live KV cache block, tagged with its original integer
    /// block index so the Python-side block table can be patched back
    /// into place on restore. Many per snapshot, one per live block.
    KvLiveBlock = 1,

    /// The metadata JSON blob. One per snapshot. Carries everything
    /// the Python layer needs to rebuild the scheduler state: block
    /// table, sequence groups, refcounts, tokenizer config, and the
    /// pinned vllm_commit hash.
    Metadata = 2,
}

/// An ordered collection of `RegionEntry`s. This is the in-memory
/// form of the on-disk "region table" that sits immediately after
/// the file header.
///
/// Ordering is part of the contract: entry at index N on disk is
/// entry at index N in memory, and readers walk the table in order.
/// `push` appends; there is intentionally no `insert`, `remove`, or
/// `sort` API on this type because reordering a region table after
/// the file has been written would require renumbering every
/// `file_offset` in the table, which is the kind of footgun that
/// should be opt-in behind a helper with its own tests.
///
/// The serialized form is just the entries back-to-back with no
/// length prefix — the count is carried in the file header's
/// `num_regions` field, and `from_bytes` takes that count explicitly.
/// Storing the count in exactly one place avoids the "two sources of
/// truth disagree" failure mode.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegionTable {
    entries: Vec<RegionEntry>,
}

impl RegionTable {
    /// Build an empty table.
    pub fn new() -> Self {
        RegionTable { entries: Vec::new() }
    }

    /// Number of entries currently in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if the table has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Append an entry to the end of the table.
    pub fn push(&mut self, entry: RegionEntry) {
        self.entries.push(entry);
    }

    /// Read an entry by index, or `None` if out of range.
    pub fn get(&self, index: usize) -> Option<&RegionEntry> {
        self.entries.get(index)
    }

    /// Serialize the table to its on-disk byte form.
    ///
    /// The result is exactly `self.len() * REGION_ENTRY_SIZE` bytes:
    /// each entry's `to_bytes()` output laid down in order with no
    /// separator, no length prefix, and no trailing padding. Padding
    /// to a page boundary (if required by the outer file layout) is
    /// the file writer's job, not the table's.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.entries.len() * REGION_ENTRY_SIZE as usize);
        for entry in &self.entries {
            buf.extend_from_slice(&entry.to_bytes());
        }
        buf
    }

    /// Parse a table from the on-disk byte form.
    ///
    /// `num_entries` is taken as an explicit argument — not inferred
    /// from `bytes.len()` — because the authoritative count lives in
    /// the file header and we want the mismatch between "header
    /// claims N entries" and "N entries worth of bytes" to be
    /// detectable as a typed error rather than silently truncating.
    pub fn from_bytes(bytes: &[u8], num_entries: usize) -> Result<Self, RegionError> {
        let need = num_entries * REGION_ENTRY_SIZE as usize;
        if bytes.len() < need {
            return Err(RegionError::TooShort {
                got: bytes.len(),
                need,
            });
        }

        let mut entries = Vec::with_capacity(num_entries);
        for i in 0..num_entries {
            let start = i * REGION_ENTRY_SIZE as usize;
            let end = start + REGION_ENTRY_SIZE as usize;
            entries.push(RegionEntry::from_bytes(&bytes[start..end])?);
        }
        Ok(RegionTable { entries })
    }
}

/// Errors produced when parsing or constructing a `RegionEntry` from
/// raw bytes. Non-exhaustive so new variants can be added without
/// breaking downstream crates that match on it.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegionError {
    /// The input slice was too short to contain a full entry.
    #[error("region entry slice is too short: got {got} bytes, need {need}")]
    TooShort { got: usize, need: usize },

    /// The `kind` discriminant is not one of the known variants.
    /// `found` is the raw u32 value so error messages can print it.
    #[error("unknown region kind discriminant: {found}")]
    UnknownKind { found: u32 },
}

/// A single entry in the region table.
///
/// Describes one region of the payload: what it is, where in the file
/// it starts, how long it is, and (for KV blocks) which logical block
/// index it represents.
///
/// Built with a tiny builder (`with_size`, `with_file_offset`,
/// `with_logical_id`) rather than a six-argument `new`. The builder
/// style makes call sites self-documenting ("this region is 16 GB at
/// offset 4096") and lets us add or remove fields without breaking
/// every existing caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionEntry {
    kind: RegionKind,
    /// Original integer block index for `KvLiveBlock` regions;
    /// unused (0) for `Weights` and `Metadata`.
    logical_id: u32,
    /// Length of this region's payload, in bytes.
    size: u64,
    /// Absolute byte offset into the `.thaw` file where this
    /// region's payload starts.
    file_offset: u64,
    /// CRC32C (Castagnoli) of the region's payload bytes. Zero on
    /// fresh entries and stamped by the writer once the payload is
    /// known. Restore code compares this against the CRC computed
    /// while reading to detect bit-rot, truncated uploads, or partial
    /// S3 PUTs that passed Content-Length but lost bytes in flight.
    crc32c: u32,
}

impl RegionEntry {
    /// Build a new region entry of the given kind. All numeric fields
    /// default to zero; use the `with_*` builders to set them.
    pub fn new(kind: RegionKind) -> Self {
        RegionEntry {
            kind,
            logical_id: 0,
            size: 0,
            file_offset: 0,
            crc32c: 0,
        }
    }

    /// Builder: set the payload size in bytes.
    pub fn with_size(mut self, size: u64) -> Self {
        self.size = size;
        self
    }

    /// Builder: set the absolute file offset where the payload starts.
    pub fn with_file_offset(mut self, off: u64) -> Self {
        self.file_offset = off;
        self
    }

    /// Builder: set the logical id. Only meaningful for KV block
    /// regions; ignored by readers for Weights and Metadata.
    pub fn with_logical_id(mut self, id: u32) -> Self {
        self.logical_id = id;
        self
    }

    /// Builder: set the CRC32C of the region's payload bytes.
    pub fn with_crc32c(mut self, crc: u32) -> Self {
        self.crc32c = crc;
        self
    }

    /// Read-only access to the kind.
    pub fn kind(&self) -> RegionKind {
        self.kind
    }

    /// Read-only access to the payload size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Read-only access to the absolute file offset of the payload.
    pub fn file_offset(&self) -> u64 {
        self.file_offset
    }

    /// Read-only access to the logical id (see field docs).
    pub fn logical_id(&self) -> u32 {
        self.logical_id
    }

    /// Read-only access to the CRC32C stamp. Zero means "not computed"
    /// (freshly-constructed entry that has not been through the
    /// writer's stamping pass).
    pub fn crc32c(&self) -> u32 {
        self.crc32c
    }

    /// Serialize this entry to its fixed-size on-disk byte form.
    ///
    /// Layout (pinned by tests in the module below):
    ///
    ///   0..4   kind discriminant (u32 LE)
    ///   4..8   logical_id        (u32 LE)
    ///   8..16  size              (u64 LE)
    ///   16..24 file_offset       (u64 LE)
    ///   24..28 crc32c            (u32 LE) — Castagnoli over payload
    ///   28..32 reserved, zero-filled for forward compatibility
    ///
    /// The reserved tail is load-bearing: when we eventually add a
    /// new field (flags, compression tag), we bump the header's
    /// `CURRENT_VERSION` and repurpose one of these reserved bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = vec![0u8; REGION_ENTRY_SIZE as usize];
        buf[0..4].copy_from_slice(&(self.kind as u32).to_le_bytes());
        buf[4..8].copy_from_slice(&self.logical_id.to_le_bytes());
        buf[8..16].copy_from_slice(&self.size.to_le_bytes());
        buf[16..24].copy_from_slice(&self.file_offset.to_le_bytes());
        buf[24..28].copy_from_slice(&self.crc32c.to_le_bytes());
        // Bytes 28..32 stay at zero — see doc comment above.
        buf
    }

    /// Parse a `RegionEntry` from the on-disk byte form.
    ///
    /// Inverse of `to_bytes`. Length-checks first, then rejects any
    /// unknown kind discriminant with a typed error so the caller
    /// sees "unknown region kind 3" instead of a panic or, worse, a
    /// silent misdispatch of the region bytes to the wrong handler.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RegionError> {
        if (bytes.len() as u64) < REGION_ENTRY_SIZE {
            return Err(RegionError::TooShort {
                got: bytes.len(),
                need: REGION_ENTRY_SIZE as usize,
            });
        }

        let kind_raw = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let kind = match kind_raw {
            0 => RegionKind::Weights,
            1 => RegionKind::KvLiveBlock,
            2 => RegionKind::Metadata,
            other => return Err(RegionError::UnknownKind { found: other }),
        };
        let logical_id = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let size = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let file_offset = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let crc32c = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        // Bytes 28..32 ignored on the read path — reserved slot.

        Ok(RegionEntry {
            kind,
            logical_id,
            size,
            file_offset,
            crc32c,
        })
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A freshly-constructed entry remembers its kind.
    ///
    /// The smallest possible claim about this type: you can build one
    /// and read back the single field it has. If this ever fails, the
    /// constructor is broken in a way that makes every other test in
    /// this module meaningless.
    #[test]
    fn new_region_entry_remembers_its_kind() {
        let e = RegionEntry::new(RegionKind::Weights);
        assert_eq!(e.kind(), RegionKind::Weights);
    }

    /// The `RegionKind` discriminants are pinned to their on-disk
    /// values. These numbers must never change — old `.thaw` files on
    /// disk have them baked in. This test is a tripwire: if a refactor
    /// ever shuffles the enum order or renumbers a variant, this test
    /// fires *immediately*, not at restore time.
    #[test]
    fn region_kind_discriminants_are_pinned() {
        assert_eq!(RegionKind::Weights as u32, 0);
        assert_eq!(RegionKind::KvLiveBlock as u32, 1);
        assert_eq!(RegionKind::Metadata as u32, 2);
    }

    /// An entry carries a `size` (payload length in bytes) and a
    /// `file_offset` (where the payload starts in the .thaw file).
    ///
    /// These two fields together are what makes random-access reads
    /// of a region possible: reader picks an entry, seeks to
    /// `file_offset`, reads exactly `size` bytes. Both are u64
    /// because a weights region for a 70B model is already larger
    /// than 4 GiB, and file offsets go past the 32-bit limit as soon
    /// as you have more than one such region.
    #[test]
    fn new_region_entry_accepts_size_and_file_offset() {
        let e = RegionEntry::new(RegionKind::Weights)
            .with_size(16 * 1024 * 1024 * 1024)
            .with_file_offset(4096);
        assert_eq!(e.size(), 16 * 1024 * 1024 * 1024);
        assert_eq!(e.file_offset(), 4096);
    }

    /// An entry carries a `logical_id` used to tag KV blocks with
    /// their original integer block index.
    ///
    /// For `Weights` and `Metadata` regions this field is unused and
    /// conventionally 0. For `KvLiveBlock` regions it is the integer
    /// index the block occupied in vLLM's preallocated KV pool on
    /// freeze — the exact number we patch back into the Python-side
    /// block table on restore. See KEY_INSIGHTS.md insight 1 for why
    /// this lets us sidestep CUDA virtual address reconstruction.
    ///
    /// u32 because vLLM's pool is always many orders of magnitude
    /// smaller than 4 billion blocks and we do not want to burn 4
    /// extra bytes per entry on headroom we will never use.
    #[test]
    fn new_region_entry_accepts_logical_id() {
        let e = RegionEntry::new(RegionKind::KvLiveBlock).with_logical_id(42);
        assert_eq!(e.logical_id(), 42);
    }

    /// `to_bytes` serializes one entry to exactly `REGION_ENTRY_SIZE`
    /// bytes. Length-only, contents come next.
    #[test]
    fn region_entry_to_bytes_is_fixed_size() {
        let e = RegionEntry::new(RegionKind::Weights);
        assert_eq!(e.to_bytes().len() as u64, REGION_ENTRY_SIZE);
    }

    /// `to_bytes` lays out fields at pinned offsets in little-endian.
    ///
    /// This is the equivalent of the header's field-by-field layout
    /// tests. Each offset is pinned explicitly so that future
    /// refactors cannot silently shift a field by four bytes.
    ///
    ///   0..4   kind discriminant (u32 LE)
    ///   4..8   logical_id        (u32 LE)
    ///   8..16  size              (u64 LE)
    ///   16..24 file_offset       (u64 LE)
    ///   24..28 crc32c            (u32 LE)
    ///   28..32 reserved, zeroed
    #[test]
    fn region_entry_layout_is_pinned() {
        let e = RegionEntry::new(RegionKind::KvLiveBlock)
            .with_logical_id(7)
            .with_size(0x11_22_33_44_55_66_77_88)
            .with_file_offset(0x99_AA_BB_CC_DD_EE_FF_01)
            .with_crc32c(0xDEAD_BEEF);

        let bytes = e.to_bytes();
        assert_eq!(&bytes[0..4], &1u32.to_le_bytes());
        assert_eq!(&bytes[4..8], &7u32.to_le_bytes());
        assert_eq!(&bytes[8..16], &0x11_22_33_44_55_66_77_88u64.to_le_bytes());
        assert_eq!(&bytes[16..24], &0x99_AA_BB_CC_DD_EE_FF_01u64.to_le_bytes());
        assert_eq!(&bytes[24..28], &0xDEAD_BEEFu32.to_le_bytes());
        // Last reserved slot must be zero — future-self depends on it
        // being safe to compare two serialized entries byte-for-byte.
        assert_eq!(&bytes[28..32], &[0u8; 4]);
    }

    /// Round-trip: parse(serialize(x)) == x.
    #[test]
    fn region_entry_round_trip_through_bytes() {
        let original = RegionEntry::new(RegionKind::Metadata)
            .with_logical_id(0)
            .with_size(1234)
            .with_file_offset(999_999)
            .with_crc32c(0xC0FF_EE42);
        let bytes = original.to_bytes();
        let parsed = RegionEntry::from_bytes(&bytes).expect("should parse");
        assert_eq!(parsed, original);
        assert_eq!(parsed.crc32c(), 0xC0FF_EE42);
    }

    /// Parser rejects short input with a typed error, not a panic.
    #[test]
    fn region_entry_from_bytes_rejects_short_input() {
        let bytes = [0u8; 4];
        let err = RegionEntry::from_bytes(&bytes).expect_err("should reject short");
        assert!(matches!(
            err,
            RegionError::TooShort { got: 4, need: 32 }
        ));
    }

    /// Parser rejects an unknown `kind` discriminant cleanly.
    ///
    /// A garbage kind byte is the single most likely corruption mode
    /// for this table (a future version added kind 3, an older
    /// reader sees it). We refuse early so the caller gets a clean
    /// "unknown region kind 3" message instead of silently
    /// mis-dispatching the region to the wrong handler.
    #[test]
    fn region_entry_from_bytes_rejects_unknown_kind() {
        let mut bytes = RegionEntry::new(RegionKind::Weights).to_bytes();
        // Overwrite the kind discriminant with a value that is not
        // a known variant (LE encoding of 99).
        bytes[0..4].copy_from_slice(&99u32.to_le_bytes());
        let err = RegionEntry::from_bytes(&bytes).expect_err("should reject unknown kind");
        assert!(matches!(err, RegionError::UnknownKind { found: 99 }));
    }

    // ------------------------------------------------------------------
    // RegionTable — collection of entries
    // ------------------------------------------------------------------
    //
    // A `RegionTable` is what actually lives on disk after the header.
    // It owns an ordered list of `RegionEntry`s and knows how to
    // serialize the whole list to a contiguous byte buffer and parse
    // it back. Ordering matters: readers walk the table in order, so
    // the file format contract is "entry at index N on disk is entry
    // at index N in memory."

    /// A fresh table is empty and has a `len` of zero.
    #[test]
    fn new_region_table_is_empty() {
        let t = RegionTable::new();
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
    }

    /// `push` appends an entry and `get` reads it back by index.
    #[test]
    fn push_then_get_preserves_order() {
        let mut t = RegionTable::new();
        t.push(RegionEntry::new(RegionKind::Weights).with_size(16));
        t.push(RegionEntry::new(RegionKind::Metadata).with_size(32));
        assert_eq!(t.len(), 2);
        assert_eq!(t.get(0).unwrap().kind(), RegionKind::Weights);
        assert_eq!(t.get(0).unwrap().size(), 16);
        assert_eq!(t.get(1).unwrap().kind(), RegionKind::Metadata);
        assert_eq!(t.get(1).unwrap().size(), 32);
    }

    /// `to_bytes` produces exactly `N * REGION_ENTRY_SIZE` bytes.
    ///
    /// No header, no count prefix, no trailing padding. The count is
    /// carried separately in the file header's `num_regions` field;
    /// duplicating it here would invite drift between the two
    /// sources of truth. The table is "just the entries, in order."
    #[test]
    fn region_table_to_bytes_is_n_times_entry_size() {
        let mut t = RegionTable::new();
        t.push(RegionEntry::new(RegionKind::Weights));
        t.push(RegionEntry::new(RegionKind::Metadata));
        t.push(RegionEntry::new(RegionKind::KvLiveBlock).with_logical_id(5));
        let bytes = t.to_bytes();
        assert_eq!(bytes.len() as u64, 3 * REGION_ENTRY_SIZE);
    }

    /// Within a serialized table, entry N sits at offset
    /// `N * REGION_ENTRY_SIZE`. This is what makes random access work.
    #[test]
    fn region_table_entries_sit_at_fixed_strides() {
        let mut t = RegionTable::new();
        t.push(
            RegionEntry::new(RegionKind::Weights)
                .with_size(0xAABBCCDD)
                .with_file_offset(0x1000),
        );
        t.push(
            RegionEntry::new(RegionKind::Metadata)
                .with_size(0x11223344)
                .with_file_offset(0x2000),
        );
        let bytes = t.to_bytes();
        // First entry occupies bytes 0..32.
        let first = RegionEntry::from_bytes(&bytes[0..32]).unwrap();
        assert_eq!(first.kind(), RegionKind::Weights);
        assert_eq!(first.size(), 0xAABBCCDD);
        // Second entry occupies bytes 32..64.
        let second = RegionEntry::from_bytes(&bytes[32..64]).unwrap();
        assert_eq!(second.kind(), RegionKind::Metadata);
        assert_eq!(second.file_offset(), 0x2000);
    }

    /// Round-trip: a table built in memory, serialized, and parsed
    /// back is equal to the original.
    ///
    /// The parser takes an explicit `num_entries` parameter because
    /// the file header stores the count. This is a deliberate design
    /// choice: the table itself has no length prefix, and readers
    /// always know how many entries to expect before they touch
    /// these bytes. Passing the count in explicitly makes that
    /// assumption visible in every call site.
    #[test]
    fn region_table_round_trip() {
        let mut original = RegionTable::new();
        original.push(
            RegionEntry::new(RegionKind::Weights)
                .with_size(16 * 1024 * 1024 * 1024)
                .with_file_offset(4096 + 3 * REGION_ENTRY_SIZE),
        );
        original.push(
            RegionEntry::new(RegionKind::KvLiveBlock)
                .with_logical_id(17)
                .with_size(32 * 1024)
                .with_file_offset(999_999),
        );
        original.push(
            RegionEntry::new(RegionKind::Metadata)
                .with_size(512)
                .with_file_offset(1_000_000),
        );
        let bytes = original.to_bytes();
        let parsed = RegionTable::from_bytes(&bytes, 3).expect("should parse");
        assert_eq!(parsed, original);
    }

    /// Parser refuses if the buffer is smaller than `num_entries`
    /// full entries.
    #[test]
    fn region_table_from_bytes_rejects_truncated() {
        let mut t = RegionTable::new();
        t.push(RegionEntry::new(RegionKind::Weights));
        t.push(RegionEntry::new(RegionKind::Metadata));
        let bytes = t.to_bytes();
        // Claim there are 3 entries but only pass 2 worth of bytes.
        let err = RegionTable::from_bytes(&bytes, 3).expect_err("should reject");
        assert!(matches!(
            err,
            RegionError::TooShort { got, need } if got == bytes.len() && need == 3 * 32
        ));
    }

    /// A zero-entry table round-trips cleanly: an empty input is
    /// valid iff `num_entries` is zero.
    #[test]
    fn region_table_empty_round_trip() {
        let t = RegionTable::new();
        let bytes = t.to_bytes();
        assert!(bytes.is_empty());
        let parsed = RegionTable::from_bytes(&bytes, 0).expect("empty table");
        assert_eq!(parsed.len(), 0);
    }
}
