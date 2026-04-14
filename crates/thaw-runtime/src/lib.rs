// thaw-runtime — the orchestration layer.
//
// ## What lives in this crate
//
// - `backend` — the `CudaBackend` trait and the small set of types
//                (`DevicePtr`, `DeviceRegion`, `PinnedBuffer`) it
//                exchanges with its callers. This is the seam between
//                "orchestration logic" and "anything that talks to a
//                GPU."
// - `mock`    — `MockCuda`, a HashMap-backed fake that implements
//                `CudaBackend`. Used by every tier-2 test in the
//                project. Compiles on any machine.
// - (more modules will land here as TDD drives them in: the
//    freeze/restore orchestrator, pinned-buffer allocators, etc.
//    Every module has a failing test that motivates its creation.)
//
// ## The testing rule for this crate
//
// Everything in thaw-runtime must be testable on a Mac with no GPU.
// The pattern is: write orchestration code against the trait, test
// it against `MockCuda`, and trust that the later `RealCuda` impl
// (which will live in its own crate that depends on thaw-cuda-sys)
// only has to match the trait contract. See docs/TESTING.md section
// 3 for the full strategy.
//
// ## The discipline
//
// There is exactly one rule, per docs/TESTING.md section 3.3: no
// code in this crate may ever import `thaw-cuda-sys`. Everything
// that needs to touch the GPU goes through the `CudaBackend` trait.
// If you ever find yourself about to write a direct FFI call here,
// stop — that's a leak and it belongs in the `RealCuda` module in
// the sibling crate.

pub mod backend;
pub mod direct_io;
pub mod freeze;
pub mod mock;
pub mod pipeline;
#[cfg(feature = "cuda")]
pub mod real;
pub mod restore;

pub use backend::{
    BackendError, CudaBackend, DevicePtr, DeviceRegion, PinnedBuffer,
    PipelinedBackend, StreamHandle,
};
pub use freeze::{freeze, FreezeConfig, FreezeError, FreezeRequest};
pub use mock::MockCuda;
pub use pipeline::{
    freeze_pipelined, restore_pipelined, restore_pipelined_from_bytes, FreezeStats, PipelineConfig,
};
#[cfg(feature = "cuda")]
pub use real::{OwnedDeviceRegion, RealCuda};
pub use restore::{restore, RestoreError, RestoreStats};
