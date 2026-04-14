// crates/thaw-core/src/header.rs
//
// =============================================================================
// THE SNAPSHOT FILE HEADER
// =============================================================================
//
// This file defines the first 4096 bytes of every `.thaw` file.
//
// If you've never designed a binary file format before, here's the mental
// model: every file on disk is a long sequence of bytes. To make sense of
// those bytes, the first thing a reader sees is a fixed-layout "header"
// that says "I am a .thaw file, version 1, here's where to find the rest
// of the data inside me." The header is like the title page of a book —
// it tells you what the book is, who wrote it, and where to find chapter 1.
//
// See DESIGN.md §3.3 for the overall file layout and why the header is
// exactly 4096 bytes (spoiler: it's so that the "real data" after the
// header starts on a page boundary, which O_DIRECT I/O requires — see
// GLOSSARY: "O_DIRECT", "Page").
//
// -----------------------------------------------------------------------------
// THE TDD STORY FOR THIS MODULE
// -----------------------------------------------------------------------------
//
// This module is the first real module in thaw. It exists to set the
// pattern for every later module, so it's commented aggressively and the
// test loop is explicit.
//
// The loop we're following (from docs/TESTING.md §5):
//
//   1. Write ONE failing test.
//   2. Watch it fail for the right reason (compile error counts).
//   3. Write the MINIMUM code to make it pass.
//   4. Run it, watch it pass.
//   5. Refactor if needed.
//   6. Write the NEXT test. Repeat.
//
// The first test (below, in the `#[cfg(test)] mod tests` block at the
// bottom of this file) claims:
//
//   "A default-constructed SnapshotHeader has the magic bytes 'THAW\0'
//    in its first four bytes."
//
// That's a very small claim. But if it's wrong, then file format detection
// is broken, and nothing else in the project can work. Small claims are
// good claims.
//
// The code below this comment block is ONLY enough to make that test
// pass. It is deliberately incomplete. We will add fields (version,
// model_id, tokenizer hash, etc.) ONE AT A TIME, each driven by its own
// failing test. Resist the temptation to pre-add them.
//
// =============================================================================

use std::fmt;

use thiserror::Error;

/// The fixed magic bytes at the start of every `.thaw` file.
///
/// Readers check these four bytes before doing anything else. If they
/// don't match, the file is not a thaw snapshot and the reader should
/// reject it immediately with a clear error instead of trying to parse
/// garbage.
///
/// The value is `THAW` in ASCII. Four bytes because that's the standard
/// magic-number width in most binary formats (PNG, ELF, WASM all use
/// 4-byte magic numbers). It's also exactly the length of the project
/// name, which is a nice coincidence.
///
/// Why `pub`? So tests and downstream crates can reference the exact
/// expected value without re-typing it and risking typos.
pub const MAGIC: [u8; 4] = *b"THAW";

/// The total size, in bytes, of the on-disk header region of a `.thaw`
/// file. The header occupies the first `HEADER_SIZE` bytes of the file,
/// and the payload (weights, KV live blocks, metadata) starts at exactly
/// this offset.
///
/// Why 4096?
///   - It is the page size on every OS and filesystem thaw will run on
///     (x86-64 Linux, Windows, macOS on Apple Silicon all use 4 KiB
///     pages, or in the Apple case a 16 KiB native page that still
///     treats 4 KiB as a valid alignment). Pinned memory and O_DIRECT
///     both require page-aligned buffers and file offsets; by starting
///     the payload at byte 4096 we guarantee both at zero cost.
///   - It is the smallest page size that still comfortably fits every
///     header field we currently plan (magic, version, model id,
///     tokenizer hash, gpu arch tag, vllm commit hash, region table
///     offset). Using the full page also leaves plenty of headroom for
///     later fields without having to bump the header size and break
///     every existing snapshot.
///   - It simplifies reasoning: "skip the first page, then the file is
///     all payload" is a rule any future reader can keep in their head.
///
/// If we ever need a header larger than 4096 bytes, that is a breaking
/// format change and we should bump `CURRENT_VERSION` in the same commit
/// that changes this constant. Readers can then branch on the version
/// byte to decide how many bytes to skip before the payload.
pub const HEADER_SIZE: u64 = 4096;

/// The current on-disk format version for `.thaw` files.
///
/// This is the single source of truth for "what version are we writing
/// right now." Every freshly-constructed `SnapshotHeader` stamps this
/// value into its `version` field, and every reader compares the version
/// it sees on disk against this constant to decide whether it knows how
/// to parse the rest of the file.
///
/// We start at 1, not 0, so that a zeroed-out buffer (e.g. a freshly
/// `mmap`d region that was never written, or a corrupted file full of
/// null bytes) is immediately distinguishable from a real v1 file. A
/// version of 0 is a canary for "this file was never actually initialized."
///
/// Bumping this constant is a breaking change to the on-disk format.
/// When we eventually need to do it, the rule is: bump this, update the
/// `new_header_has_version_one` test to match the new number (and rename
/// it), and add a migration branch in the reader. Do all three in the
/// same commit so the history is easy to bisect.
pub const CURRENT_VERSION: u32 = 1;

/// The fixed-size on-disk header of a `.thaw` snapshot file.
///
/// A `.thaw` file always begins with this struct, serialized to exactly
/// `HEADER_SIZE` bytes. Everything after the header is the "payload"
/// (weights, KV cache, metadata) and is located via offsets recorded in
/// the region table inside the header.
///
/// The header is designed to be:
///   - Fixed-size (so readers know exactly how much to read before parsing)
///   - Self-identifying (via the magic bytes)
///   - Version-tagged (so future format changes are detectable)
///
/// For now the struct is very bare — just the magic bytes — because we
/// only have one test to satisfy. Future commits will add fields: version,
/// model identifier, tokenizer hash, GPU architecture tag, and the region
/// table offset. Each field gets its own test that fails first.
///
/// # Example (future, once we have a constructor)
///
/// ```ignore
/// let header = SnapshotHeader::new();
/// assert_eq!(header.magic(), &[b'T', b'H', b'A', b'W']);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotHeader {
    /// The magic bytes identifying this as a thaw file.
    ///
    /// Kept as an owned `[u8; 4]` rather than a reference so that
    /// `SnapshotHeader` is `'static` and can be constructed at any time.
    magic: [u8; 4],

    /// The on-disk format version for this header.
    ///
    /// `u32` instead of `u8` because four bytes costs nothing on a
    /// 4096-byte header and gives us billions of version numbers if we
    /// ever need them (we won't — but "we won't" is famous last words
    /// for file format designers). u32 also aligns nicely with the
    /// 4-byte magic above, keeping the first eight bytes of the file
    /// trivially parseable.
    version: u32,

    /// The number of entries in the region table that this header
    /// points at. Zero is a valid value and means "no payload yet" —
    /// a freshly constructed header always starts here.
    ///
    /// u64 (not u32) because every other count-or-size field in the
    /// file format is u64, and keeping them uniform means a reader
    /// never has to remember which width a particular field uses.
    /// The cost of eight bytes inside a 4096-byte header is zero.
    num_regions: u64,

    /// The absolute byte offset within the `.thaw` file where the
    /// region table begins. For a default header this is equal to
    /// `HEADER_SIZE` — the table starts on the very next page after
    /// the header. Keeping this as an explicit field (rather than
    /// hard-coding `HEADER_SIZE` in every reader) means a future
    /// revision can slide the table without touching every caller.
    region_table_offset: u64,

    /// The 40-byte hex SHA-1 of the vLLM commit this snapshot was
    /// written against. All-zero means "not stamped"; the later
    /// vLLM integration layer refuses to trust a snapshot whose
    /// recorded commit does not match the hash of the currently
    /// installed vLLM.
    ///
    /// See DESIGN.md §3.3 and Decision Log entry 2026-04-11 for
    /// why this is load-bearing: it is the mitigation for the
    /// "vLLM refactors its block manager between versions and
    /// silently corrupts restored state" failure mode (risk #1).
    vllm_commit: [u8; 40],
}

impl SnapshotHeader {
    /// Construct a fresh header with the magic bytes set and all other
    /// fields at their default values.
    ///
    /// At this point in development there are no other fields, so "default"
    /// is trivial. As we add fields, this constructor will grow.
    ///
    /// Why a constructor instead of `Default::default()`? Because we want
    /// callers to pass parameters (version, model id, etc.) in future
    /// versions of this function, and a custom constructor is easier to
    /// evolve than an impl of `Default`.
    pub fn new() -> Self {
        SnapshotHeader {
            magic: MAGIC,
            version: CURRENT_VERSION,
            num_regions: 0,
            // "Empty table sitting right after the header." The very
            // first entry a writer appends to an empty snapshot will
            // live at this exact offset, so readers never need a
            // special case for "table not written yet."
            region_table_offset: HEADER_SIZE,
            // Default to all-zero: "no vLLM commit recorded." A
            // writer that forgets to stamp this produces a
            // parseable header, but the vLLM integration layer
            // will later refuse to load it. See the field doc
            // and the tests in this file for the rationale.
            vllm_commit: [0u8; 40],
        }
    }

    /// Builder: set the number of region-table entries this header
    /// claims to point at. Returns `self` so calls can chain.
    ///
    /// This is a builder (not a setter) because `SnapshotHeader` is
    /// constructed once per write and then serialized — there is no
    /// legitimate reason for a live header struct to mutate its own
    /// region count in place, and a builder signature makes that
    /// pattern the path of least resistance.
    pub fn with_num_regions(mut self, n: u64) -> Self {
        self.num_regions = n;
        self
    }

    /// Builder: set the absolute byte offset at which the region
    /// table begins inside the `.thaw` file. Returns `self` so calls
    /// can chain.
    ///
    /// Writers that keep the default layout never need to call this
    /// — `SnapshotHeader::new()` already defaults it to
    /// `HEADER_SIZE`. It exists for future formats that might park
    /// the table somewhere else (for example, after a growing
    /// metadata blob).
    pub fn with_region_table_offset(mut self, offset: u64) -> Self {
        self.region_table_offset = offset;
        self
    }

    /// Read-only access to the number of region-table entries.
    pub fn num_regions(&self) -> u64 {
        self.num_regions
    }

    /// Read-only access to the byte offset where the region table
    /// begins. Readers use this to `seek` to the start of the table
    /// before parsing `num_regions()` entries of `REGION_ENTRY_SIZE`
    /// bytes each.
    pub fn region_table_offset(&self) -> u64 {
        self.region_table_offset
    }

    /// Builder: stamp the vllm_commit slot with a 40-byte hash.
    ///
    /// Callers typically pass the raw bytes of a 40-character hex
    /// string (e.g. `*b"deadbeef..."`). This module does not
    /// validate that the bytes are actually hex — the format is
    /// agnostic, and validation is the vLLM integration layer's
    /// job. We store and return whatever the caller gave us.
    pub fn with_vllm_commit(mut self, commit: [u8; 40]) -> Self {
        self.vllm_commit = commit;
        self
    }

    /// Read-only access to the vllm_commit slot.
    ///
    /// Returns a reference rather than an owned array so the
    /// caller can avoid a 40-byte copy on every access. The
    /// value is not interpreted in any way; it is exactly the
    /// bytes that were stamped via `with_vllm_commit`, or all
    /// zeros for a fresh header.
    pub fn vllm_commit(&self) -> &[u8; 40] {
        &self.vllm_commit
    }

    /// Read-only access to the magic bytes.
    ///
    /// Returned by reference because `[u8; 4]` is small but cheap-by-ref
    /// and this way the caller can't accidentally mutate them.
    pub fn magic(&self) -> &[u8; 4] {
        &self.magic
    }

    /// Read-only access to the format version stamped into this header.
    ///
    /// Returned by value (not by reference) because `u32` is `Copy` and
    /// four bytes on the stack is cheaper than dereferencing a pointer.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// The byte offset within the `.thaw` file where the payload begins.
    ///
    /// This is the value readers and writers use to seek past the header
    /// before streaming weights or KV blocks. Right now it is a constant
    /// (`HEADER_SIZE`) because the header is always exactly one page, but
    /// keeping it as a method — rather than having callers hard-code
    /// `4096` — means any future change to the header layout is a single
    /// edit in one file, and every O_DIRECT read in the codebase keeps
    /// working without modification.
    ///
    /// Returned as `u64` rather than `usize` because file offsets in
    /// thaw are always 64-bit regardless of host pointer width. A
    /// `.thaw` file can easily be larger than 4 GiB, and we do not want
    /// a 32-bit build (if one ever exists, e.g. an embedded host tool)
    /// to silently truncate.
    pub fn payload_offset(&self) -> u64 {
        HEADER_SIZE
    }

    /// Serialize this header into its fixed-size on-disk byte form.
    ///
    /// The returned buffer is always exactly `HEADER_SIZE` bytes long,
    /// regardless of how few or how many fields the header carries. The
    /// unused tail is zero-filled: zeroes are a safe default because
    /// (a) they are unambiguous — no valid field has an all-zero value
    /// we would confuse with real data — and (b) they make the file
    /// deterministic, so byte-for-byte equality tests on two snapshots
    /// of the same state actually pass.
    ///
    /// Why `Vec<u8>` and not a stack array: callers will usually write
    /// this straight into a pinned host buffer or through O_DIRECT,
    /// which wants a heap-allocated (and preferably page-aligned)
    /// destination anyway. Returning a `Vec` keeps the signature simple
    /// and leaves alignment concerns to the caller — we will address
    /// alignment properly when we wire up the real I/O layer.
    ///
    /// This method currently only reserves the space; the next TDD
    /// cycle will start populating specific byte offsets (magic first,
    /// then version, then the rest). Every field gets its own failing
    /// test before it lands here.
    pub fn to_bytes(&self) -> Vec<u8> {
        // Start with a zero-filled page. Unused tail bytes stay at zero
        // by construction, which gives the file a deterministic,
        // reproducible layout for golden-file tests.
        let mut buf = vec![0u8; HEADER_SIZE as usize];

        // Offset 0..4 — magic bytes. Offset 0 is the universal "is this
        // my file?" check (ELF, PNG, WASM, safetensors all do it), so
        // the magic goes first and nothing else is allowed to overlap
        // it. `copy_from_slice` is the cheapest correct write for a
        // fixed 4-byte range; `buf[..4] = MAGIC;` would also work but
        // is a subtler slice-pattern trick, and we'd rather the I/O
        // layer read like a postcard than a magic spell.
        buf[0..4].copy_from_slice(&self.magic);

        // Offset 4..8 — version, little-endian u32. LE is chosen once,
        // here, so every future field in this header (and every
        // downstream region-table entry) can inherit the same rule
        // instead of each author having to pick. See the test
        // `to_bytes_encodes_version_little_endian_at_offset_four` for
        // the full rationale — the short version is "every host we
        // target is natively LE, every adjacent file format is LE, no
        // hot-path byte swapping, no cross-FFI surprises."
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());

        // Offset 8..16 — num_regions, u64 LE. This tells a reader
        // how many fixed-size entries to pull out of the region
        // table immediately after seeking to `region_table_offset`.
        buf[8..16].copy_from_slice(&self.num_regions.to_le_bytes());

        // Offset 16..24 — region_table_offset, u64 LE. Absolute
        // byte offset inside the file; a reader can seek straight
        // there without any arithmetic.
        buf[16..24].copy_from_slice(&self.region_table_offset.to_le_bytes());

        // Offset 24..64 — vllm_commit, 40 raw bytes. No encoding,
        // no length prefix: the slot is fixed, the caller owns the
        // semantics. An un-stamped header writes 40 zero bytes,
        // which readers treat as "not recorded."
        buf[24..64].copy_from_slice(&self.vllm_commit);

        buf
    }

    /// Parse a `SnapshotHeader` from the on-disk byte form.
    ///
    /// This is the inverse of [`to_bytes`]. The round-trip invariant —
    /// `from_bytes(h.to_bytes()) == Ok(h)` — is the single most
    /// important property of the file format and is pinned by a test.
    ///
    /// The parser is the only place in thaw that sees untrusted input,
    /// so it is defensive:
    ///   1. Length check first. A buffer shorter than `HEADER_SIZE` is
    ///      rejected before any indexing, so bad input produces a
    ///      typed error and not an index-out-of-bounds panic.
    ///   2. Magic check. A wrong magic means the file is not a thaw
    ///      snapshot at all — we refuse loudly instead of walking
    ///      further into a format we cannot trust.
    ///   3. Version check. A known-unknown version is refused with the
    ///      found/supported pair in the error so the caller can
    ///      report "this file was written by a newer thaw, please
    ///      upgrade" without guessing.
    ///
    /// Buffers larger than `HEADER_SIZE` are accepted: the caller may
    /// hand us the whole file's bytes and only want us to parse the
    /// first page. We validate the first `HEADER_SIZE` bytes and
    /// ignore the rest.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, HeaderError> {
        // Length gate. `HEADER_SIZE` is a u64 for on-disk reasons, but
        // here we compare against `bytes.len()`, which is a usize; the
        // cast is fine on every supported host (HEADER_SIZE = 4096).
        if (bytes.len() as u64) < HEADER_SIZE {
            return Err(HeaderError::TooShort {
                got: bytes.len(),
                need: HEADER_SIZE as usize,
            });
        }

        // Offset 0..4 — magic. Copy into an owned array so the error
        // variant can carry it without borrowing from `bytes`.
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);
        if magic != MAGIC {
            return Err(HeaderError::BadMagic { found: magic });
        }

        // Offset 4..8 — version, little-endian u32. The `try_into` +
        // unwrap is safe because we already know the slice is exactly
        // four bytes.
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != CURRENT_VERSION {
            return Err(HeaderError::UnsupportedVersion {
                found: version,
                supported: CURRENT_VERSION,
            });
        }

        // Offset 8..16 — num_regions, u64 LE.
        let num_regions = u64::from_le_bytes(bytes[8..16].try_into().unwrap());

        // Offset 16..24 — region_table_offset, u64 LE.
        let region_table_offset = u64::from_le_bytes(bytes[16..24].try_into().unwrap());

        // Offset 24..64 — vllm_commit, 40 raw bytes. Copied into
        // an owned array so the returned header has no borrow
        // against `bytes`.
        let mut vllm_commit = [0u8; 40];
        vllm_commit.copy_from_slice(&bytes[24..64]);

        Ok(SnapshotHeader {
            magic,
            version,
            num_regions,
            region_table_offset,
            vllm_commit,
        })
    }
}

/// Errors produced when parsing a `SnapshotHeader` from bytes.
///
/// Every variant carries enough context for the caller to produce a
/// useful message without having to re-inspect the input. The whole
/// enum is `#[non_exhaustive]` so that future format changes (adding,
/// say, a `TokenizerMismatch` variant) are not a breaking change for
/// downstream crates that match on this type.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum HeaderError {
    /// The input slice was smaller than a full header. The `got` field
    /// is what we saw; `need` is what we required.
    #[error("header is too short: got {got} bytes, need at least {need}")]
    TooShort { got: usize, need: usize },

    /// The first four bytes did not match `MAGIC`. The `found` bytes
    /// are included so the error message can show exactly what was
    /// seen instead — invaluable when debugging a misidentified file.
    #[error("bad magic bytes: expected b\"THAW\", found {found:?}")]
    BadMagic { found: [u8; 4] },

    /// The header carried a version number this binary does not know
    /// how to read. Both numbers are included so the caller can tell
    /// the user "file is v{found}, this thaw supports v{supported}."
    #[error("unsupported snapshot version: found {found}, this binary supports {supported}")]
    UnsupportedVersion { found: u32, supported: u32 },
}

// A human-readable string form for debugging. Prints the magic as an
// ASCII string so that `println!("{}", header)` shows "THAW" instead of
// "[84, 72, 65, 87]", and tags along the version number so that logs
// can tell v1 files apart from any future format bump at a glance.
//
// The rule for this impl, enforced by a test in the block below, is:
// every field of `SnapshotHeader` that a human might want to see at 2am
// should show up in the `Display` output in some form. If you add a
// field to the struct, add it here too, and add a substring assertion
// to the test so the next person (probably you) can't forget.
impl fmt::Display for SnapshotHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Render the vllm_commit as a lossy UTF-8 string — the
        // expected content is ASCII hex, and `from_utf8_lossy`
        // turns any stray non-ASCII bytes into the replacement
        // char rather than erroring. If the slot is all zeros
        // (un-stamped header), the result is a string of 40 null
        // chars, which we special-case into "<unset>" for log
        // readability. The test pins that *some* substring of
        // the stamped hash appears in the output, not the exact
        // format, so this block has room to evolve.
        let commit_str: String = if self.vllm_commit.iter().all(|&b| b == 0) {
            "<unset>".to_string()
        } else {
            String::from_utf8_lossy(&self.vllm_commit).into_owned()
        };
        write!(
            f,
            "SnapshotHeader(magic=\"{}\", version={}, vllm_commit={})",
            String::from_utf8_lossy(&self.magic),
            self.version,
            commit_str,
        )
    }
}

// =============================================================================
// TESTS
// =============================================================================
//
// Unit tests live in the same file as the code they test (a Rust
// convention). The `#[cfg(test)]` attribute means this module is only
// compiled during `cargo test`, so it adds nothing to release binaries.
//
// Each test is ONE assertion about ONE claim. Resist the urge to cram
// multiple checks into a single `#[test]` — when a cram-test fails, you
// don't know which claim broke.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// The first test: a newly-constructed header has the magic bytes
    /// "THAW" at the start.
    ///
    /// Why this test?
    ///   - It's the smallest possible claim that, if false, proves the
    ///     entire file format is broken.
    ///   - It doesn't depend on anything else (no I/O, no disk, no
    ///     serialization). Pure in-memory check.
    ///   - It sets the pattern: every new feature starts with a test this
    ///     simple.
    ///
    /// If this test ever fails in the future, you've probably either
    ///   (a) changed `MAGIC` on purpose and forgot to update the test,
    ///   (b) changed the layout of `SnapshotHeader` in a way that broke
    ///       construction, or
    ///   (c) accidentally overwritten the magic bytes somewhere.
    /// All three are worth knowing about immediately.
    #[test]
    fn new_header_has_thaw_magic_bytes() {
        let header = SnapshotHeader::new();
        assert_eq!(header.magic(), b"THAW");
    }

    /// The second test: a newly-constructed header reports a version of 1.
    ///
    /// Why this test?
    ///   - Every binary format that survives contact with real users ends up
    ///     needing a version number. safetensors has one, ELF has one, PNG
    ///     has one. The reason is always the same: the day you ship v1 is
    ///     the day you wish you had shipped v2, and without a version byte
    ///     you cannot tell old files apart from new ones.
    ///   - We commit to *one* format number at a time. Right now every
    ///     newly-constructed header is version 1. If we ever bump the
    ///     format, we bump this constant and this test in the same commit,
    ///     and every reader then has a single branch point ("is this a v1
    ///     or a v2 file?") to handle the transition.
    ///   - It fails today for the right reason: `SnapshotHeader` has no
    ///     `version()` accessor and no `version` field, so the compiler
    ///     will reject this test. That compile failure IS the first half
    ///     of the TDD cycle — "watch it fail."
    ///
    /// If this test ever fails in the future, you've probably either
    ///   (a) bumped the on-disk format number on purpose and forgot to
    ///       update the test, or
    ///   (b) broken the constructor so it no longer stamps the version.
    /// Both are worth knowing about immediately.
    #[test]
    fn new_header_has_version_one() {
        let header = SnapshotHeader::new();
        assert_eq!(header.version(), 1);
    }

    /// The third test: the `Display` impl mentions both the magic bytes
    /// and the version number.
    ///
    /// Why this test?
    ///   - Every time we add a field to `SnapshotHeader`, there's a risk
    ///     that the debug-friendly `Display` impl silently drifts out of
    ///     sync with the struct — new field goes in, `Display` keeps
    ///     printing the old three fields, and the first time anyone
    ///     notices is during a 2am debugging session where the log says
    ///     nothing about the field that's broken. A pinned test on the
    ///     `Display` output is a cheap way to force future-us to update
    ///     the formatter every time the struct grows.
    ///   - We assert on substring containment, not exact equality,
    ///     because we want `Display` to be free to change its formatting
    ///     (spacing, punctuation, ordering) without breaking this test.
    ///     What we care about is "the string mentions THAW and it
    ///     mentions version 1," not the exact layout.
    ///   - It fails today for the right reason: the current `Display`
    ///     impl only writes the magic, not the version, so the
    ///     `contains("version=1")` check will return false.
    #[test]
    fn display_mentions_magic_and_version() {
        let header = SnapshotHeader::new();
        let rendered = format!("{}", header);
        assert!(
            rendered.contains("THAW"),
            "Display output should mention the magic bytes, got: {rendered}"
        );
        assert!(
            rendered.contains("version=1"),
            "Display output should mention version=1, got: {rendered}"
        );
    }

    /// The fourth test: the payload (weights, KV blocks, metadata)
    /// begins at byte offset 4096, not one byte after the last header
    /// field.
    ///
    /// Why this test?
    ///   - This is the first test that bakes the "header is exactly one
    ///     OS page, payload starts on the next page boundary" rule into
    ///     the type system, instead of leaving it as a comment in
    ///     DESIGN.md §3.3 that someone could accidentally violate.
    ///   - O_DIRECT on Linux refuses to read or write a file unless the
    ///     byte offset, the buffer pointer, and the transfer length are
    ///     all multiples of the filesystem's logical block size (almost
    ///     always 4096 on modern ext4/xfs). See GLOSSARY: O_DIRECT and
    ///     Page. If our payload started at byte 17, the very first
    ///     O_DIRECT read of the weights would fail with EINVAL and no
    ///     explanation, and we'd burn an afternoon figuring out why.
    ///   - Pinning the offset as a method on the header means every
    ///     reader and writer in the future asks the header where the
    ///     payload lives, rather than hard-coding 4096 in scattered
    ///     places. If we ever change the header size, there's exactly
    ///     one place to update, and this test catches the change.
    ///   - It fails today for the right reason: `SnapshotHeader` has no
    ///     `payload_offset()` method yet, so the compiler rejects the
    ///     test.
    #[test]
    fn new_header_payload_starts_at_page_boundary() {
        let header = SnapshotHeader::new();
        assert_eq!(header.payload_offset(), 4096);
    }

    /// The fifth test: `to_bytes()` serializes the header to exactly
    /// `HEADER_SIZE` bytes.
    ///
    /// This is the first test that turns the in-memory struct into a
    /// real on-disk byte sequence. It only pins the length, not the
    /// contents — the contents get pinned by the next test, where we
    /// start asserting which bytes live at which offsets. Separating
    /// "right size" from "right bytes" keeps each failure pointing at
    /// one kind of bug.
    ///
    /// Why the length must be exactly `HEADER_SIZE` (not <=): readers
    /// do one fixed-size `read(4096)` to get the header off disk. If
    /// `to_bytes()` ever returned fewer bytes, the reader would mix
    /// trailing garbage (or uninitialized memory) into the header;
    /// if it returned more, the payload would start inside the header,
    /// which is exactly the O_DIRECT-misalignment bug we went to so
    /// much trouble to prevent in test #4.
    ///
    /// Fails today for the right reason: no `to_bytes()` method exists.
    #[test]
    fn to_bytes_is_exactly_header_size() {
        let header = SnapshotHeader::new();
        let bytes = header.to_bytes();
        assert_eq!(bytes.len() as u64, HEADER_SIZE);
    }

    /// The sixth test: the first four bytes of the serialized header
    /// are the magic, in order.
    ///
    /// This is the first test that pins actual content at a specific
    /// file offset. Offset 0 is the only offset every reader is
    /// guaranteed to hit, so magic-first is the universal convention
    /// (ELF, PNG, WASM, safetensors, every format you've ever used).
    /// A reader that opens a random file does one 4-byte read at
    /// offset 0, compares against `MAGIC`, and either proceeds or
    /// rejects the file — no ambiguity, no partial parsing.
    ///
    /// Fails today for the right reason: `to_bytes()` currently returns
    /// all zeroes, so `bytes[0..4]` is `[0, 0, 0, 0]`, not b"THAW".
    #[test]
    fn to_bytes_starts_with_magic() {
        let header = SnapshotHeader::new();
        let bytes = header.to_bytes();
        assert_eq!(&bytes[0..4], &MAGIC);
    }

    /// The seventh test: bytes 4..8 of the serialized header encode the
    /// version as a little-endian u32.
    ///
    /// This is the first multi-byte integer in the file, so it forces
    /// us to pick an endianness and commit to it *once*, in a test,
    /// before a single byte has been written to disk anywhere.
    ///
    /// Why little-endian:
    ///   - Every platform thaw will run on (x86-64 Linux, x86-64
    ///     Windows, ARM64 macOS, ARM64 Linux server) is natively
    ///     little-endian. Picking LE means reader code on the hot
    ///     path is a single aligned load with no byte swap.
    ///   - Every adjacent format we care about (safetensors, GGUF,
    ///     ELF, the CUDA driver ABI) is also LE. Staying consistent
    ///     avoids a whole category of "I forgot to swap" bugs at the
    ///     FFI boundary.
    ///   - The one argument for big-endian is "network byte order,"
    ///     which is irrelevant here — .thaw files are never sent over
    ///     a socket in this project; they live on local NVMe.
    ///
    /// Fails today for the right reason: `to_bytes()` currently only
    /// writes the magic, so bytes 4..8 are still zero, not
    /// `[1, 0, 0, 0]`.
    #[test]
    fn to_bytes_encodes_version_little_endian_at_offset_four() {
        let header = SnapshotHeader::new();
        let bytes = header.to_bytes();
        assert_eq!(&bytes[4..8], &1u32.to_le_bytes());
    }

    /// The eighth test: a header survives a round trip through bytes
    /// and back. `from_bytes(to_bytes(h)) == h` for every valid header.
    ///
    /// This is the first test that stands up the read path next to the
    /// write path, and it locks in the single most important invariant
    /// of any file format: the parser and the serializer are inverses.
    /// If this test ever fails, one of the two sides has drifted and
    /// every snapshot written by the old code is unreadable by the new
    /// code — the worst possible bug for a file format.
    ///
    /// Fails today for the right reason: `from_bytes` does not exist.
    #[test]
    fn round_trip_through_bytes() {
        let original = SnapshotHeader::new();
        let bytes = original.to_bytes();
        let parsed = SnapshotHeader::from_bytes(&bytes).expect("valid header should parse");
        assert_eq!(parsed, original);
    }

    /// The ninth test: `from_bytes` rejects a buffer whose first four
    /// bytes are not the magic.
    ///
    /// The parser is the only part of thaw that sees untrusted input
    /// (a file the user claims is a snapshot might actually be a PNG,
    /// a truncated download, or a corrupted page of disk), so it has
    /// to fail loudly and typedly on every wrong-shaped input. This
    /// test pins the magic-check behavior specifically, so that we can
    /// never accidentally accept a non-thaw file and then keep walking
    /// off the end of its bytes interpreting them as our format.
    #[test]
    fn from_bytes_rejects_bad_magic() {
        let mut bytes = SnapshotHeader::new().to_bytes();
        bytes[0] = b'X'; // Corrupt the first magic byte.
        let err = SnapshotHeader::from_bytes(&bytes).expect_err("should reject bad magic");
        assert!(matches!(err, HeaderError::BadMagic { .. }));
    }

    /// The tenth test: `from_bytes` rejects a buffer whose version
    /// field is not equal to `CURRENT_VERSION`.
    ///
    /// A version mismatch is a recoverable error in principle (some
    /// future reader could branch on it), but for the MVP we only
    /// know how to read v1 files, and silently accepting a v2 file
    /// would mean parsing a format we do not understand as if it were
    /// v1 — the next field would be misinterpreted and everything
    /// downstream would be corrupted in ways that would not show up
    /// until restore time. Refuse early, refuse clearly.
    #[test]
    fn from_bytes_rejects_wrong_version() {
        let mut bytes = SnapshotHeader::new().to_bytes();
        // Stamp version 2 (LE) over the version slot.
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
        let err = SnapshotHeader::from_bytes(&bytes).expect_err("should reject bad version");
        assert!(matches!(
            err,
            HeaderError::UnsupportedVersion { found: 2, supported: 1 }
        ));
    }

    /// The eleventh test: `from_bytes` rejects any buffer shorter than
    /// `HEADER_SIZE`.
    ///
    /// This is the "short read" case — a truncated file, a partial
    /// network download, or a reader that accidentally passed the
    /// wrong slice. If we did not check the length up front, later
    /// indexing (like `bytes[4..8]`) would panic with an
    /// out-of-bounds, which is the wrong way to report "this file is
    /// too small to be a snapshot." A typed error tells the caller
    /// exactly what went wrong.
    #[test]
    fn from_bytes_rejects_short_input() {
        let bytes = vec![0u8; 16]; // way too small, and wrong content.
        let err = SnapshotHeader::from_bytes(&bytes).expect_err("should reject short input");
        assert!(matches!(
            err,
            HeaderError::TooShort { got: 16, need: 4096 }
        ));
    }

    // ------------------------------------------------------------------
    // num_regions + region_table_offset
    // ------------------------------------------------------------------
    //
    // This is where the header stops being a standalone type and
    // becomes the thing that points at the region table. Two new
    // fields:
    //
    //   num_regions         — how many entries the table has
    //   region_table_offset — absolute byte offset where the table
    //                         starts in the .thaw file
    //
    // Both are u64 for consistency with the 64-bit fields already in
    // RegionEntry (file_offset, size). A u32 for num_regions would be
    // more than enough in practice but would force a pad field for
    // alignment; u64 is simpler and has zero cost inside a 4096-byte
    // header that is mostly reserved.
    //
    // The convention for a default/empty header is num_regions == 0
    // and region_table_offset == HEADER_SIZE, meaning "the table
    // starts right after me, it just happens to be empty right now."
    // This makes the very first write into a fresh file trivially
    // valid.

    /// A fresh header reports zero regions.
    #[test]
    fn new_header_has_no_regions() {
        let h = SnapshotHeader::new();
        assert_eq!(h.num_regions(), 0);
    }

    /// A fresh header points its region_table_offset at the byte
    /// right after the header — i.e. the first payload-area byte.
    ///
    /// This is the invariant that every writer depends on: "the
    /// table begins at payload_offset()." Pinning it as a test means
    /// the day someone reorders the header layout or moves the table
    /// elsewhere, the error shows up *here* and not at restore time
    /// when a corrupted table gets parsed.
    #[test]
    fn new_header_points_region_table_right_after_header() {
        let h = SnapshotHeader::new();
        assert_eq!(h.region_table_offset(), HEADER_SIZE);
    }

    /// Builders set both fields.
    #[test]
    fn header_builders_set_region_table_fields() {
        let h = SnapshotHeader::new()
            .with_num_regions(17)
            .with_region_table_offset(8192);
        assert_eq!(h.num_regions(), 17);
        assert_eq!(h.region_table_offset(), 8192);
    }

    /// Layout: num_regions is u64 LE at offset 8..16.
    ///
    /// Pinned so that a future refactor cannot silently slide the
    /// field by a few bytes and produce a header that parses but
    /// means something different.
    #[test]
    fn to_bytes_encodes_num_regions_at_offset_eight() {
        let h = SnapshotHeader::new().with_num_regions(0x11_22_33_44_55_66_77_88);
        let bytes = h.to_bytes();
        assert_eq!(
            &bytes[8..16],
            &0x11_22_33_44_55_66_77_88u64.to_le_bytes()
        );
    }

    /// Layout: region_table_offset is u64 LE at offset 16..24.
    #[test]
    fn to_bytes_encodes_region_table_offset_at_offset_sixteen() {
        let h = SnapshotHeader::new().with_region_table_offset(0x99_AA_BB_CC_DD_EE_FF_01);
        let bytes = h.to_bytes();
        assert_eq!(
            &bytes[16..24],
            &0x99_AA_BB_CC_DD_EE_FF_01u64.to_le_bytes()
        );
    }

    /// The tail of the header (bytes 64..4096) is still zero-filled
    /// after every declared field is accounted for. Tripwire for
    /// "I added a field and forgot to update the zero-fill
    /// invariant" bugs. The cutoff is 64 because vllm_commit
    /// occupies 24..64 (40 bytes) — see the vllm_commit section
    /// below for why that specific slot size.
    #[test]
    fn to_bytes_tail_is_zero_filled() {
        // Stamp every declared field with a non-default value so
        // the test proves the *tail* is zero, not just the slots
        // we happened to leave at their defaults.
        let h = SnapshotHeader::new()
            .with_num_regions(7)
            .with_region_table_offset(4096)
            .with_vllm_commit(*b"0123456789abcdef0123456789abcdef01234567");
        let bytes = h.to_bytes();
        assert!(
            bytes[64..].iter().all(|&b| b == 0),
            "expected all-zero tail, found non-zero byte in 64..4096"
        );
    }

    /// Round-trip: a header with non-default region fields survives
    /// serialization and parsing.
    #[test]
    fn header_with_region_fields_round_trips() {
        let original = SnapshotHeader::new()
            .with_num_regions(3)
            .with_region_table_offset(HEADER_SIZE);
        let bytes = original.to_bytes();
        let parsed = SnapshotHeader::from_bytes(&bytes).expect("valid header should parse");
        assert_eq!(parsed, original);
    }

    // ------------------------------------------------------------------
    // vllm_commit
    // ------------------------------------------------------------------
    //
    // Per DESIGN.md §3.3 and Decision Log entry 2026-04-11: the
    // snapshot header carries a 40-byte slot for the git SHA-1 hash
    // of the vLLM commit the file was written against. On restore,
    // a mismatch between the recorded hash and the hash of the
    // currently installed vLLM is a hard refuse, not a warning.
    //
    // WHY 40 BYTES AND NOT 20
    // -----------------------
    // A git SHA-1 is 20 raw bytes. We store the hex form (40 ASCII
    // characters) instead for two reasons: (a) error messages can
    // print the value directly without a hex encoder, and (b) the
    // vLLM repo uses the 40-char form in every log and API that
    // returns commit hashes, so the bytes on disk match the bytes
    // a Python caller will hand us.
    //
    // WHY IT'S A FIXED-SIZE SLOT
    // --------------------------
    // A variable-length string would need a length prefix and a
    // bounds check, both of which are load-bearing in a parser
    // that sees untrusted input. A fixed 40-byte slot removes
    // those questions entirely: the field is always at the same
    // offset, always the same size, and "unset" is a well-defined
    // value (all zeros).
    //
    // The convention for an un-stamped header (fresh `new()`) is
    // all-zero bytes. Readers interpret all-zero as "no vLLM
    // commit recorded" and, in a later integration layer, refuse
    // to trust the file unless a non-zero value is present *and*
    // matches the installed vLLM.

    /// A fresh header has an all-zero vllm_commit slot.
    ///
    /// Zero is the designated "not yet stamped" value. A writer
    /// that forgets to call `with_vllm_commit` produces a header
    /// that round-trips cleanly but will be refused by the (later)
    /// vLLM integration layer, which is exactly the behavior we
    /// want — fail loudly rather than silently accept a snapshot
    /// with no provenance.
    #[test]
    fn new_header_has_zero_vllm_commit() {
        let h = SnapshotHeader::new();
        assert_eq!(h.vllm_commit(), &[0u8; 40]);
    }

    /// Builder sets the vllm_commit slot from a 40-byte array.
    #[test]
    fn header_builder_sets_vllm_commit() {
        // A made-up 40-byte hash just to verify the accessor.
        // Uses printable ASCII so a hex-dump reader can eyeball
        // it in the serialized bytes.
        let hash = *b"0123456789abcdef0123456789abcdef01234567";
        let h = SnapshotHeader::new().with_vllm_commit(hash);
        assert_eq!(h.vllm_commit(), &hash);
    }

    /// Layout: vllm_commit occupies bytes 24..64 (40 bytes).
    ///
    /// Pinned so a future refactor cannot silently slide the slot
    /// and produce a header that parses but means something
    /// different. The offset 24 is chosen to sit immediately
    /// after region_table_offset (16..24); the length 40 is the
    /// full git SHA-1 hex form.
    #[test]
    fn to_bytes_encodes_vllm_commit_at_offset_twenty_four() {
        let hash = *b"deadbeefdeadbeefdeadbeefdeadbeefdeadbeef";
        let h = SnapshotHeader::new().with_vllm_commit(hash);
        let bytes = h.to_bytes();
        assert_eq!(&bytes[24..64], &hash);
    }

    /// Round-trip: a header with a stamped vllm_commit survives
    /// serialization and parsing.
    #[test]
    fn header_with_vllm_commit_round_trips() {
        let hash = *b"aaaabbbbccccddddeeeeffff0000111122223333";
        let original = SnapshotHeader::new()
            .with_num_regions(2)
            .with_vllm_commit(hash);
        let bytes = original.to_bytes();
        let parsed = SnapshotHeader::from_bytes(&bytes).expect("valid header");
        assert_eq!(parsed, original);
        assert_eq!(parsed.vllm_commit(), &hash);
    }

    /// The Display impl mentions the vllm_commit (or a trimmed
    /// form of it), so that a log line about a header is
    /// immediately useful for diagnosing "which vLLM wrote this."
    #[test]
    fn display_mentions_vllm_commit() {
        let hash = *b"cafef00dcafef00dcafef00dcafef00dcafef00d";
        let h = SnapshotHeader::new().with_vllm_commit(hash);
        let rendered = format!("{}", h);
        // We don't pin the exact format — short prefix or full
        // value is fine — only that the hash appears somewhere.
        assert!(
            rendered.contains("cafef00d"),
            "Display should mention vllm_commit, got: {rendered}"
        );
    }
}
