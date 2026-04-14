// Integration test: golden-file check for the serialized header.
//
// WHY THIS TEST EXISTS
// --------------------
//
// The unit tests in `src/header.rs` pin individual fields of the
// serialized header (magic at offset 0, version at offset 4, etc.).
// They're great at catching bugs in the field we just touched, but
// they do not catch "silent format drift" — the situation where two
// fields accidentally overlap, a field moves by four bytes, or the
// zero-fill tail grows a non-zero byte. A golden-file test is the
// cheap, brute-force answer: we commit the exact 4096-byte sequence
// the current format writes, and every future run compares to it
// byte-for-byte.
//
// If this test ever fails, one of three things has happened:
//
//   1. You intentionally changed the on-disk format. In that case,
//      bump `CURRENT_VERSION` and regenerate the golden (instructions
//      below). The version bump is the visible signal that old
//      snapshots are no longer byte-compatible.
//
//   2. You introduced an accidental format drift. The diff will tell
//      you exactly which byte moved. Fix the code, don't "just
//      regenerate" the golden — that would paper over the bug.
//
//   3. A platform somehow flipped endianness on you. This should be
//      impossible on the hosts we target (all little-endian), but if
//      it ever happens, the first four bytes of the version field
//      will be `00 00 00 01` instead of `01 00 00 00` and the diff
//      will be obvious.
//
// REGENERATING THE GOLDEN
// -----------------------
//
// Only do this when you have deliberately bumped `CURRENT_VERSION` or
// otherwise changed the format, and you have updated every other
// header test to match. From the repo root:
//
//   cargo test -p thaw-core --test header_golden -- --ignored \
//     regenerate_header_golden
//
// That runs the regenerator test below (normally ignored) which
// overwrites the committed golden file with whatever the current code
// produces. Commit the new bytes in the same commit that bumped the
// version.

use std::path::PathBuf;

use thaw_core::{SnapshotHeader, HEADER_SIZE};

/// Resolve the path to the committed golden file.
///
/// `CARGO_MANIFEST_DIR` points at `crates/thaw-core/` when this
/// integration test is running, so the path is stable regardless of
/// what the current working directory happens to be when `cargo test`
/// is invoked.
fn golden_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("goldens");
    p.push("header");
    p.push("v1_default.bin");
    p
}

#[test]
fn default_header_matches_golden_bytes() {
    let expected = std::fs::read(golden_path())
        .expect("golden file missing — run the `regenerate_header_golden` ignored test");
    let actual = SnapshotHeader::new().to_bytes();

    assert_eq!(
        expected.len() as u64,
        HEADER_SIZE,
        "golden file has wrong length; the committed file is corrupt"
    );
    assert_eq!(
        actual.len(),
        expected.len(),
        "to_bytes() produced {} bytes, golden is {} bytes",
        actual.len(),
        expected.len()
    );

    if actual != expected {
        // Produce a focused diff: show the first byte that differs
        // and a short window around it. A full 4096-byte diff is
        // unreadable and 99% of format drifts are a single field.
        let first_diff = actual
            .iter()
            .zip(expected.iter())
            .position(|(a, b)| a != b)
            .expect("vectors differ but no mismatch found?");
        let start = first_diff.saturating_sub(4);
        let end = (first_diff + 8).min(actual.len());
        panic!(
            "header bytes do not match golden\n  first differing offset: {first_diff}\n  actual[{start}..{end}]   = {:02x?}\n  expected[{start}..{end}] = {:02x?}\n\nIf this change was intentional, bump CURRENT_VERSION and regenerate with\n  cargo test -p thaw-core --test header_golden -- --ignored regenerate_header_golden",
            &actual[start..end],
            &expected[start..end],
        );
    }
}

/// Regenerator — normally ignored. Run on purpose (`-- --ignored`)
/// to overwrite the committed golden file with the current
/// serializer's output. Only run this when you have just bumped
/// `CURRENT_VERSION` or otherwise intentionally changed the format;
/// otherwise you are papering over a real drift.
#[test]
#[ignore]
fn regenerate_header_golden() {
    let bytes = SnapshotHeader::new().to_bytes();
    std::fs::write(golden_path(), &bytes).expect("failed to write golden file");
    eprintln!(
        "regenerated golden at {} ({} bytes)",
        golden_path().display(),
        bytes.len()
    );
}
