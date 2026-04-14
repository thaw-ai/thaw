// Integration test: golden-file check for a *non-empty* snapshot prelude.
//
// WHY THIS TEST EXISTS
// --------------------
//
// `header_golden.rs` pins the 4096-byte default header. That's the
// minimum — it catches drift in the header-only bytes. But real
// `.thaw` files have a region table sitting immediately after the
// header, and the only way to catch drift in the *composed* layout
// (header + table, with the correct offsets, little-endian
// encodings, reserved zero-fills, etc.) is to pin the exact byte
// sequence a known-small prelude produces and compare to it
// byte-for-byte in CI.
//
// The golden is built from a deterministic three-entry snapshot:
//
//   entry 0: Weights,      size = 0x00_00_00_00_10_00_00_00, file_offset = 0x10_00
//   entry 1: KvLiveBlock,  logical_id = 7, size = 0x2000, file_offset = 0x20_00
//   entry 2: Metadata,     size = 0x100, file_offset = 0x30_00
//
// The exact numbers are arbitrary but chosen to be recognizable in a
// hex dump (0x1000, 0x2000, 0x3000 for file offsets; a 256 MiB
// weights blob; a single KV block of 8 KiB; a 256-byte metadata
// blob). If a byte ever moves, a human reading the hex diff can see
// exactly which field drifted.
//
// If this test fails, the same three diagnoses apply as in
// `header_golden.rs`:
//
//   1. You intentionally changed the on-disk format. Bump
//      CURRENT_VERSION and regenerate. The version bump is the
//      visible signal that old snapshots are no longer byte-compatible.
//
//   2. You introduced an accidental format drift. The diff will
//      point at the first differing byte. Fix the code, don't "just
//      regenerate" — that would paper over the bug.
//
//   3. The composition logic in `snapshot.rs` changed in a way that
//      shifted the table offset or altered the reserved zero-fill.
//
// REGENERATING THE GOLDEN
// -----------------------
//
//   cargo test -p thaw-core --test snapshot_golden -- --ignored \
//     regenerate_snapshot_golden
//
// Only run this when you have deliberately bumped the format.

use std::path::PathBuf;

use thaw_core::{RegionEntry, RegionKind, RegionTable, Snapshot};

fn golden_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("goldens");
    p.push("snapshot");
    p.push("v1_three_entries.bin");
    p
}

/// Construct the exact snapshot the golden encodes. Used by both the
/// check test and the regenerator so they cannot drift.
fn fixture_snapshot() -> Snapshot {
    let mut t = RegionTable::new();
    t.push(
        RegionEntry::new(RegionKind::Weights)
            .with_size(0x1000_0000) // 256 MiB
            .with_file_offset(0x1000),
    );
    t.push(
        RegionEntry::new(RegionKind::KvLiveBlock)
            .with_logical_id(7)
            .with_size(0x2000) // 8 KiB
            .with_file_offset(0x2000),
    );
    t.push(
        RegionEntry::new(RegionKind::Metadata)
            .with_size(0x100) // 256 B
            .with_file_offset(0x3000),
    );
    Snapshot::from_table(t)
}

#[test]
fn three_entry_prelude_matches_golden_bytes() {
    let expected = std::fs::read(golden_path())
        .expect("golden file missing — run the `regenerate_snapshot_golden` ignored test");
    let actual = fixture_snapshot().to_prelude_bytes();

    assert_eq!(
        actual.len(),
        expected.len(),
        "to_prelude_bytes() produced {} bytes, golden is {} bytes",
        actual.len(),
        expected.len()
    );

    if actual != expected {
        // Focused diff: first differing byte with a short window
        // around it. The full prelude is ~4192 bytes and a complete
        // diff is unreadable for a single-field drift.
        let first_diff = actual
            .iter()
            .zip(expected.iter())
            .position(|(a, b)| a != b)
            .expect("vectors differ but no mismatch found?");
        let start = first_diff.saturating_sub(4);
        let end = (first_diff + 12).min(actual.len());
        panic!(
            "prelude bytes do not match golden\n  first differing offset: {first_diff}\n  actual[{start}..{end}]   = {:02x?}\n  expected[{start}..{end}] = {:02x?}\n\nIf this change was intentional, bump CURRENT_VERSION and regenerate with\n  cargo test -p thaw-core --test snapshot_golden -- --ignored regenerate_snapshot_golden",
            &actual[start..end],
            &expected[start..end],
        );
    }
}

#[test]
#[ignore]
fn regenerate_snapshot_golden() {
    let bytes = fixture_snapshot().to_prelude_bytes();
    // Make sure the parent directory exists before writing.
    if let Some(parent) = golden_path().parent() {
        std::fs::create_dir_all(parent).expect("failed to create golden directory");
    }
    std::fs::write(golden_path(), &bytes).expect("failed to write golden file");
    eprintln!(
        "regenerated golden at {} ({} bytes)",
        golden_path().display(),
        bytes.len()
    );
}
