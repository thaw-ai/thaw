// Integration test: write a snapshot prelude to a real file on disk,
// read it back, and assert byte/struct equality.
//
// WHY THIS TEST EXISTS
// --------------------
//
// The unit tests in `snapshot.rs` exercise `write_to` / `read_from`
// against `Vec<u8>` and `std::io::Cursor`. Those are perfectly valid
// `Write` and `Read` impls, but they live in memory — they never
// touch the OS file layer at all. A test that goes through a real
// `std::fs::File` catches a different class of bug:
//
//   - `write_all` that silently succeeds on a zero-length write
//     because the kernel hasn't flushed yet
//   - A missing `flush` or `sync_all` that leaves bytes in the
//     OS page cache and causes the read side to see a truncated
//     file
//   - Path handling quirks on Windows that only show up when a
//     real FS is involved
//   - BufReader/BufWriter interactions that a bare Vec cannot
//     reproduce
//
// This test uses `tempfile::NamedTempFile` so the file cleans itself
// up when the test finishes, regardless of pass or fail. The
// integration test runs on every platform (Mac, Linux, Windows) with
// no GPU and no privileged I/O, so it stays inside the "Mac-pure"
// constraint that `thaw-core` promises.

use std::fs::File;
use std::io::{BufReader, BufWriter};

use tempfile::NamedTempFile;
use thaw_core::{RegionEntry, RegionKind, RegionTable, Snapshot};

/// Construct a small deterministic snapshot for the test. Local
/// helper (not shared with the golden tests) because this test does
/// not care about the exact byte values — only that what we wrote
/// equals what we read back.
fn make_snapshot() -> Snapshot {
    let mut t = RegionTable::new();
    t.push(
        RegionEntry::new(RegionKind::Weights)
            .with_size(16 * 1024 * 1024)
            .with_file_offset(0x1000),
    );
    t.push(
        RegionEntry::new(RegionKind::KvLiveBlock)
            .with_logical_id(42)
            .with_size(8 * 1024)
            .with_file_offset(0x2000),
    );
    t.push(
        RegionEntry::new(RegionKind::Metadata)
            .with_size(256)
            .with_file_offset(0x3000),
    );
    Snapshot::from_table(t)
}

/// Write a snapshot through `BufWriter<File>`, then read it back
/// through `BufReader<File>`, and assert the parsed snapshot equals
/// the original.
///
/// Why `BufWriter` / `BufReader`: they are what every real file
/// writer in the production code path will use, and they have a
/// non-trivial interaction with `write_all` / `read_exact` (the
/// buffer has to flush at the right moments). If this ever breaks,
/// it breaks here first instead of at 2am on a real snapshot.
#[test]
fn snapshot_round_trips_through_real_file() {
    let original = make_snapshot();

    let tmp = NamedTempFile::new().expect("create temp file");
    let path = tmp.path().to_path_buf();

    // Write path. The `drop` of `writer` at the end of the scope
    // flushes the BufWriter; we then also call `sync_all` on the
    // raw file handle to make sure the kernel has the bytes before
    // the read side opens them. On some filesystems the test would
    // pass without `sync_all`, but making it explicit removes any
    // ambiguity.
    {
        let file = File::create(&path).expect("open for write");
        let mut writer = BufWriter::new(file);
        let written = original
            .write_to(&mut writer)
            .expect("write_to should succeed");
        assert_eq!(written, original.prelude_size());
        let file = writer.into_inner().expect("flush BufWriter");
        file.sync_all().expect("sync_all");
    }

    // On-disk size sanity check: the file should be exactly
    // `prelude_size` bytes, because `write_to` writes the prelude
    // and nothing else. A larger file would mean a stray write; a
    // smaller one would mean a silent truncation.
    let meta = std::fs::metadata(&path).expect("stat temp file");
    assert_eq!(meta.len(), original.prelude_size());

    // Read path. Same buffering strategy in reverse.
    {
        let file = File::open(&path).expect("open for read");
        let mut reader = BufReader::new(file);
        let parsed = Snapshot::read_from(&mut reader).expect("read_from should succeed");
        assert_eq!(parsed, original);
    }
}

/// Same test, but the writer appends extra "payload" bytes after
/// the prelude and the reader must still parse the prelude
/// correctly and leave the cursor positioned at the first payload
/// byte.
///
/// This is the dry-run for the real file format: prelude + payload
/// back-to-back. A full writer that composes `Snapshot` with
/// payload region bytes is coming in a later test; this one just
/// pins the "reader stops at prelude_size" contract against a real
/// file, not just a Cursor.
#[test]
fn snapshot_read_leaves_cursor_at_payload_boundary() {
    use std::io::Read;

    let original = make_snapshot();
    let payload: &[u8] = b"this would be a weight blob in real life";

    let tmp = NamedTempFile::new().expect("create temp file");
    let path = tmp.path().to_path_buf();

    {
        let file = File::create(&path).expect("open for write");
        let mut writer = BufWriter::new(file);
        original
            .write_to(&mut writer)
            .expect("write_to should succeed");
        // Append the fake payload. In a real snapshot this would
        // be written region-by-region by a ByteRegionWriter (next
        // batch).
        std::io::Write::write_all(&mut writer, payload).expect("write payload");
        let file = writer.into_inner().expect("flush BufWriter");
        file.sync_all().expect("sync_all");
    }

    let file = File::open(&path).expect("open for read");
    let mut reader = BufReader::new(file);
    let parsed = Snapshot::read_from(&mut reader).expect("parse prelude");
    assert_eq!(parsed, original);

    // Drain the rest of the reader and confirm it matches the
    // payload bytes we wrote. If `read_from` ever over-consumed,
    // this check would come up short or produce different bytes.
    let mut tail = Vec::new();
    reader.read_to_end(&mut tail).expect("drain payload");
    assert_eq!(tail, payload);
}
