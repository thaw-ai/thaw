// thaw-core — the pure-Rust foundation.
//
// This is the crate-level documentation. If you're reading this trying to
// understand what thaw-core is, start here and follow the module pointers.
//
// ## What lives in this crate
//
// - `header`    — the fixed-size header at the start of every .thaw file.
//                 See DESIGN.md §3.3 for the on-disk layout.
// - (more modules will appear as TDD drives them in: region table, payload
//    reader, payload writer, etc. We are deliberately not creating empty
//    modules ahead of time. Every module has a failing test that motivated
//    its creation.)
//
// ## What does NOT live in this crate
//
// - Anything that talks to a GPU. That's thaw-cuda-sys and cpp/thaw-cuda.
// - Freeze/restore orchestration. That's thaw-runtime.
// - Anything to do with vLLM. That's python/thaw_vllm.
//
// ## The testing rule for this crate
//
// Every public function in thaw-core must be testable on a Mac with no
// GPU. If you find yourself wanting to `#[cfg(feature = "cuda")]` something
// in here, stop — it belongs in a different crate. See docs/TESTING.md.
//
// ## Style rules
//
// - No `unsafe` unless a comment block explains exactly why, what invariant
//   is being upheld, and what would break if it were violated.
// - No `unwrap()` or `expect()` outside of tests. Production code returns
//   `Result`. Unwrapping in production is how you get cryptic panics at
//   2am on demo day.
// - Every public item has a `///` doc comment explaining what it is, when
//   you'd use it, and any non-obvious gotchas. If you don't know what to
//   write, ask yourself "what would I want to know if I'd never seen this
//   type before?" and write that.

// Re-export the header module's public types at the crate root so that
// downstream code can say `use thaw_core::SnapshotHeader` instead of
// `use thaw_core::header::SnapshotHeader`. This is a style choice — it
// keeps the public API flat and easy to discover.
pub mod header;
pub mod region;
pub mod snapshot;
pub mod writer;

pub use header::{HeaderError, SnapshotHeader, CURRENT_VERSION, HEADER_SIZE, MAGIC};
pub use region::{RegionEntry, RegionError, RegionKind, RegionTable, REGION_ENTRY_SIZE};
pub use snapshot::{Snapshot, SnapshotError};
pub use writer::ByteRegionWriter;
