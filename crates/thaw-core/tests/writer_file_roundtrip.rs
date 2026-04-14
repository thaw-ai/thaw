// Integration test: end-to-end round-trip of a complete .thaw file.
//
// WHY THIS TEST EXISTS
// --------------------
//
// Every previous test in this crate covers one half of the write/read
// cycle:
//
//   - `snapshot.rs` unit tests exercise the byte-slice parser.
//   - `snapshot_golden.rs` pins the exact prelude bytes for drift
//     detection.
//   - `snapshot_file_roundtrip.rs` round-trips just the prelude
//     through a real file.
//   - `writer.rs` unit tests exercise `ByteRegionWriter` against a
//     `Vec<u8>`.
//
// None of those tests cover the *full path* a real restore will take:
//
//   1. A writer lays down a header, a region table, and multiple
//      payload blobs to a real file.
//   2. A reader opens that file, parses the prelude, and then uses
//      the entries' `file_offset` + `size` fields to seek to each
//      region and pull its bytes back.
//   3. Each payload comes back byte-for-byte equal to what the
//      writer handed in.
//
// This is the test that catches every "the entry said the payload was
// at offset X but the writer actually put it at offset X+3" bug. Once
// it passes, we have a self-describing file format that survives a
// real write-and-reread, which is the whole point of the file format
// work in this sprint.
//
// The test stays inside the Mac-pure rule: no CUDA, no pinned memory,
// no O_DIRECT. Those concerns belong to the I/O layer in a later
// crate. This test only proves that the *format* is internally
// consistent when it hits a real filesystem.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom};

use tempfile::NamedTempFile;
use thaw_core::{ByteRegionWriter, RegionKind, Snapshot};

/// Deterministic payload-generator: returns `len` bytes that vary
/// from region to region so that a read that crosses a region
/// boundary produces a visibly wrong value.
///
/// The formula is arbitrary but bit-sensitive — a one-byte offset
/// error between regions will fail the equality check.
fn make_payload(seed: u8, len: usize) -> Vec<u8> {
    (0..len).map(|i| seed.wrapping_add(i as u8)).collect()
}

/// The actual end-to-end round-trip.
///
/// Writes three regions of different sizes and kinds through
/// `ByteRegionWriter`, reopens the file from scratch, parses the
/// prelude, and then seeks to each entry's `file_offset` and reads
/// exactly `size` bytes. Asserts byte-for-byte equality on each
/// payload and on the total file length.
#[test]
fn writer_file_round_trip_preserves_every_byte() {
    // --- setup ---------------------------------------------------
    //
    // Three regions of different kinds and sizes. Kind diversity
    // matters because the region table encodes kind as a u32
    // discriminant, and a broken layout could easily mix the kind
    // byte up with the logical_id byte — this test catches that.
    //
    // Size diversity matters because if we only ever wrote fixed-
    // size payloads, a running-offset bug might coincidentally
    // land on the right boundary and hide itself.
    let weights = make_payload(0x10, 4096);
    let kv_block = make_payload(0x40, 1024);
    let metadata = make_payload(0x80, 300);

    let mut writer = ByteRegionWriter::new();
    writer.push_region(RegionKind::Weights, 0, weights.clone());
    writer.push_region(RegionKind::KvLiveBlock, 17, kv_block.clone());
    writer.push_region(RegionKind::Metadata, 0, metadata.clone());

    let expected_snapshot = writer.build_snapshot();
    let expected_total = writer.total_size();

    // --- write ---------------------------------------------------
    //
    // BufWriter + sync_all is the same pattern used in the prelude
    // round-trip test. The scope block drops the writer and the
    // File handle so that the read below reopens fresh — no chance
    // of seeing stale buffered bytes.
    let tmp = NamedTempFile::new().expect("create temp file");
    let path = tmp.path().to_path_buf();
    {
        let file = File::create(&path).expect("open for write");
        let mut w = BufWriter::new(file);
        let written = writer.write_to(&mut w).expect("writer write_to");
        assert_eq!(written, expected_total);
        let file = w.into_inner().expect("flush BufWriter");
        file.sync_all().expect("sync_all");
    }

    // The file on disk should be exactly `total_size` bytes. No
    // trailing padding, no silent truncation.
    let meta = std::fs::metadata(&path).expect("stat temp file");
    assert_eq!(meta.len(), expected_total);

    // --- read ----------------------------------------------------
    //
    // Reopen the file, parse the prelude, then seek-and-read each
    // payload region. We open the file directly (not through
    // BufReader) because we're going to seek around: a BufReader's
    // internal position would disagree with the underlying file
    // after a seek, which is solvable but not worth the extra
    // complexity for a test this small.
    let mut file = File::open(&path).expect("open for read");

    // Parse the prelude. `Snapshot::read_from` advances the file
    // cursor to exactly `prelude_size`, which we'll verify below.
    let parsed = {
        let mut reader = BufReader::new(&mut file);
        Snapshot::read_from(&mut reader).expect("parse prelude")
    };
    assert_eq!(parsed, expected_snapshot);

    // --- verify payloads -----------------------------------------
    //
    // Walk the parsed table in order and pull each payload back
    // out of the file using the entry's `file_offset` and `size`.
    // This is the exact sequence a real restore will follow: the
    // reader trusts the entries, seeks, reads, and hands the bytes
    // to the next layer.
    let expected_payloads: [(&[u8], &str); 3] = [
        (&weights, "weights"),
        (&kv_block, "kv_block"),
        (&metadata, "metadata"),
    ];

    for (i, (expected, name)) in expected_payloads.iter().enumerate() {
        let entry = parsed
            .table()
            .get(i)
            .unwrap_or_else(|| panic!("missing entry {i} ({name})"));

        // Entry's own size field should match what we pushed.
        assert_eq!(
            entry.size() as usize,
            expected.len(),
            "entry {i} ({name}) size mismatch"
        );

        // Seek to the entry's file_offset and pull the bytes back.
        file.seek(SeekFrom::Start(entry.file_offset()))
            .unwrap_or_else(|e| panic!("seek to entry {i} ({name}): {e}"));

        let mut buf = vec![0u8; expected.len()];
        file.read_exact(&mut buf)
            .unwrap_or_else(|e| panic!("read entry {i} ({name}): {e}"));

        assert_eq!(
            &buf, expected,
            "entry {i} ({name}) payload bytes do not match what was written"
        );
    }
}

/// Second test: an empty writer produces a valid prelude-only file.
///
/// Edge case — zero regions — because off-by-one bugs in the layout
/// math tend to manifest either on the first entry or on an empty
/// table, and the first entry is already covered by the test above.
#[test]
fn empty_writer_produces_prelude_only_file() {
    let writer = ByteRegionWriter::new();
    let tmp = NamedTempFile::new().expect("create temp file");
    let path = tmp.path().to_path_buf();

    {
        let file = File::create(&path).expect("open for write");
        let mut w = BufWriter::new(file);
        writer.write_to(&mut w).expect("write empty");
        let file = w.into_inner().expect("flush");
        file.sync_all().expect("sync_all");
    }

    // File should be exactly HEADER_SIZE bytes: empty table, no
    // payload. An empty writer is the degenerate base case and
    // should still produce a parseable snapshot.
    let meta = std::fs::metadata(&path).expect("stat");
    assert_eq!(meta.len(), thaw_core::HEADER_SIZE);

    let file = File::open(&path).expect("open for read");
    let mut reader = BufReader::new(file);
    let parsed = Snapshot::read_from(&mut reader).expect("parse empty prelude");
    assert_eq!(parsed.len(), 0);
    assert!(parsed.is_empty());
}
