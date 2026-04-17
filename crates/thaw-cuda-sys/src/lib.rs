// crates/thaw-cuda-sys/src/lib.rs
//
// =============================================================================
// THE CUDA RUNTIME FFI SURFACE
// =============================================================================
//
// This is every line of `extern "C"` in the thaw project. If you are
// looking for "where does thaw talk to the GPU," this file is the only
// honest answer.
//
// The surface area is deliberately tiny. The rule is the same as in
// thaw-runtime::backend: every item in this file must correspond to a
// specific operation the real freeze/restore path will have to issue.
// Nothing here exists "just in case." A reviewer can read this file end
// to end in five minutes and be confident they have seen every CUDA call
// the project will ever make.
//
// Current surface:
//
//   Rust side (always compiled)
//     - `CudaStatus`         — safe wrapper over `cudaError_t`.
//     - `CudaMemcpyKind`     — safe wrapper over the copy-direction enum.
//     - `CUDA_AVAILABLE`     — bool const; true iff the `cuda` feature is on.
//
//   FFI side (only compiled with `--features cuda`)
//     - `cudaMalloc`            — device allocation.
//     - `cudaFree`              — device free.
//     - `cudaMallocHost`        — pinned host allocation.
//     - `cudaFreeHost`          — pinned host free.
//     - `cudaHostAlloc`         — pinned host alloc with flags (WC).
//     - `cudaHostRegister`      — pin an existing host range in place.
//     - `cudaHostUnregister`    — unpin a previously-registered range.
//     - `cudaMemcpy`            — synchronous device<->host copy.
//     - `cudaMemcpyAsync`       — asynchronous copy on a named stream.
//     - `cudaStreamCreate`      — create a named CUDA stream.
//     - `cudaStreamSynchronize` — block until stream work completes.
//     - `cudaStreamDestroy`     — destroy a named stream.
//     - `cudaGetLastError`      — clears and returns the sticky last error.
//     - `cudaGetErrorString`    — const char* describing a given status code.
//
// The split between "Rust side" and "FFI side" is the whole point of
// this crate. On a Mac without the CUDA toolkit, the Rust side still
// compiles and the rest of the workspace can link against it — they
// just cannot actually call any of the FFI functions, which is fine
// because on a Mac there is no device to call them against.
//
// =============================================================================
//
// ## Why not bindgen?
//
// bindgen is the standard tool for generating Rust bindings from C
// headers, and for most `-sys` crates it is the right answer. Here it
// would be overkill:
//
//   - The surface we need is seven functions and two enums, all of
//     which have been stable in the CUDA runtime API for a decade. The
//     signatures are not going to drift.
//
//   - bindgen at build time requires a working `libclang` on the host
//     and the CUDA headers installed. That defeats the purpose of a
//     crate that is supposed to compile on a Mac with no CUDA.
//
//   - Hand-written FFI is auditable by reading this file. Generated
//     FFI is auditable by running bindgen against a specific CUDA
//     version and diffing the output. The former is cheaper given how
//     small the surface is.
//
// If the surface ever grows past ~30 functions or we start needing the
// driver API in addition to the runtime API, reconsider. Until then,
// seven `extern "C"` declarations do not justify a build-time
// dependency.
//
// =============================================================================
//
// ## Safety contract (for callers)
//
// Every FFI function in this crate is `unsafe` for the usual reasons:
// the compiler cannot check that the pointers are valid, the sizes are
// in bounds, and the memory is not aliased. Callers must:
//
//   - Only call these functions from a thread that has a valid CUDA
//     context. The runtime API manages contexts implicitly per-thread,
//     so in practice this means "any thread of a process that has ever
//     successfully called any CUDA function."
//
//   - Never pass a host pointer to a function expecting a device
//     pointer, or vice versa. The runtime will return an error, but
//     the error is caller-visible, not a panic.
//
//   - Treat every returned `cudaError_t` as potentially non-zero. CUDA
//     errors are sticky — a non-zero return on one call can silently
//     poison the next one if you don't clear it. The `CudaStatus::ok`
//     helper exists to make ignoring errors verbose.
//
// The safe wrapper that `thaw-runtime::RealCuda` will build on top of
// these functions (in a later batch) is the layer that enforces all of
// the above at the type level.
//
// =============================================================================

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use core::ffi::c_int;
#[cfg(feature = "cuda")]
use core::ffi::c_uint;
#[cfg(feature = "cuda")]
use core::ffi::c_void;

/// Opaque handle for a CUDA stream.
///
/// On the FFI boundary this is `cudaStream_t`, which the runtime API
/// defines as a pointer to an opaque driver struct. We represent it
/// as `*mut c_void` because that is what the runtime API signatures
/// use. The null pointer is the "default stream" — thaw never uses
/// it explicitly; we always create named streams.
#[cfg(feature = "cuda")]
pub type CudaStream = *mut c_void;

/// Flag for `cudaHostAlloc`: use write-combining memory. Bypasses L1/L2
/// cache snooping on the PCIe bus, improving H2D DMA throughput by up to
/// 40% for buffers the CPU writes and the GPU reads. CPU reads from WC
/// memory are extremely slow — only use for the restore hot path.
#[cfg(feature = "cuda")]
pub const CUDA_HOST_ALLOC_WRITE_COMBINED: c_uint = 0x04;

/// Flag for `cudaHostAlloc`: default allocation (same as `cudaMallocHost`).
#[cfg(feature = "cuda")]
pub const CUDA_HOST_ALLOC_DEFAULT: c_uint = 0x00;

/// Flag for `cudaHostRegister`: default (same as no flags).
#[cfg(feature = "cuda")]
pub const CUDA_HOST_REGISTER_DEFAULT: c_uint = 0x00;

/// Flag for `cudaHostRegister`: the memory returned by this call will be
/// considered pinned memory by all CUDA contexts, not just the one that
/// performed the allocation. Required for memory shared across processes
/// (mmap on /dev/shm, etc.) to be DMA-able without re-registration.
#[cfg(feature = "cuda")]
pub const CUDA_HOST_REGISTER_PORTABLE: c_uint = 0x01;

/// Flag for `cudaHostRegister`: maps the allocation into the CUDA
/// address space. Required if callers want to use the mapped device
/// pointer via `cudaHostGetDevicePointer`. thaw does not use this
/// today — our flow uses the host pointer directly via
/// `cudaMemcpyAsync` — but it is cheap to expose.
#[cfg(feature = "cuda")]
pub const CUDA_HOST_REGISTER_MAPPED: c_uint = 0x02;

// =============================================================================
// SAFE RUST-LEVEL HELPERS
// =============================================================================
//
// These compile on every platform regardless of whether the `cuda`
// feature is enabled. They are the "Rust types" half of the crate —
// the half that sibling crates can depend on unconditionally.

/// Compile-time flag: is this build linked against libcudart?
///
/// Downstream code that wants to choose between the real and mock
/// backend at runtime without `#[cfg]` noise can read this constant.
/// The typical pattern is:
///
/// ```ignore
/// if thaw_cuda_sys::CUDA_AVAILABLE {
///     // construct RealCuda
/// } else {
///     // fall back to MockCuda
/// }
/// ```
///
/// The constant is `false` whenever the `cuda` feature is not enabled,
/// regardless of the host platform. On a 4090 box with `--features cuda`
/// it is `true`. A Mac build is always `false`.
#[cfg(feature = "cuda")]
pub const CUDA_AVAILABLE: bool = true;

/// See the feature-gated variant above. Without the `cuda` feature the
/// FFI block is not compiled in, so callers always see `false`.
#[cfg(not(feature = "cuda"))]
pub const CUDA_AVAILABLE: bool = false;

/// A small wrapper around `cudaError_t` that implements Rust traits.
///
/// The CUDA runtime represents errors as a plain C `int`. That is
/// workable but brittle: you end up comparing against magic numbers
/// everywhere, and a typo in a status code silently compiles. Wrapping
/// it in a tuple struct gives us `Debug`, `PartialEq`, and a `Display`
/// impl that looks up the human-readable name when possible.
///
/// The zero value (`CudaStatus(0)`) is "success." Every other value is
/// an error. The most common error codes are named as associated
/// constants below so call sites do not have to hardcode integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CudaStatus(pub c_int);

impl CudaStatus {
    /// `cudaSuccess` — the "no error" status.
    pub const SUCCESS: CudaStatus = CudaStatus(0);

    /// `cudaErrorInvalidValue` — a parameter was out of range.
    ///
    /// Named here because our own wrapper layer will want to map it
    /// onto `BackendError::SizeMismatch` in some cases, and we would
    /// rather do that comparison by name than by literal `3`.
    pub const INVALID_VALUE: CudaStatus = CudaStatus(1);

    /// `cudaErrorMemoryAllocation` — alloc failed because the device
    /// or host is out of the relevant memory.
    pub const MEMORY_ALLOCATION: CudaStatus = CudaStatus(2);

    /// True if this status represents success.
    ///
    /// Prefer this over comparing against `SUCCESS` directly; it reads
    /// better at call sites and is less error-prone.
    pub fn is_ok(self) -> bool {
        self == CudaStatus::SUCCESS
    }

    /// Convert to a `Result`. `CudaStatus::SUCCESS` becomes `Ok(())`,
    /// anything else becomes `Err(self)`.
    ///
    /// This is the idiomatic way to use the FFI: call the function,
    /// wrap the return in `CudaStatus`, call `.ok()`, propagate with
    /// `?`. The result type is `Result<(), CudaStatus>` so the caller
    /// can match on the specific error variant if they need to.
    pub fn ok(self) -> Result<(), CudaStatus> {
        if self.is_ok() {
            Ok(())
        } else {
            Err(self)
        }
    }
}

impl core::fmt::Display for CudaStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // On a CUDA build we could call `cudaGetErrorString` to get a
        // human-readable name. We don't here, for two reasons:
        //
        //   1. Keeping `Display` infallible-and-allocation-free means
        //      it can be used from signal handlers, panic messages,
        //      and anywhere else where calling into FFI would be a
        //      bad idea.
        //
        //   2. A higher-level wrapper in thaw-runtime will do the
        //      lookup and cache the result. That is the right layer
        //      for it — this crate's job is the raw binding.
        //
        // So we just print the numeric code with a sensible tag.
        write!(f, "CudaStatus({})", self.0)
    }
}

/// Safe wrapper over `cudaMemcpyKind`.
///
/// Making this a Rust enum instead of a loose integer means the
/// compiler will catch a copy-direction typo at the call site. The
/// integer values are the ones the CUDA runtime documents, and are
/// stable across every CUDA version we care about.
///
/// We only name the four directions thaw actually uses. A hypothetical
/// future need for `cudaMemcpyDefault` (which asks the runtime to
/// infer the direction from the pointers) would be one more variant,
/// but we deliberately do not enable it today — the whole point of
/// this layer is that thaw knows the direction and names it
/// explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum CudaMemcpyKind {
    /// Host-to-host copy. Not used by thaw today but kept for
    /// symmetry; the CUDA runtime accepts it and we want `from_raw`
    /// to round-trip cleanly.
    HostToHost = 0,
    /// Host-to-device. The restore hot path.
    HostToDevice = 1,
    /// Device-to-host. The freeze hot path.
    DeviceToHost = 2,
    /// Device-to-device. Not on the hot path today, but the real
    /// impl's "copy a KV block from the pool into a staging buffer"
    /// optimization will need it.
    DeviceToDevice = 3,
}

impl CudaMemcpyKind {
    /// Parse a raw `c_int` into a `CudaMemcpyKind`. Returns `None` for
    /// unknown values. Useful for defensive wrappers that want to log
    /// "we got an unexpected kind back" rather than trust a `transmute`.
    pub fn from_raw(value: c_int) -> Option<CudaMemcpyKind> {
        match value {
            0 => Some(CudaMemcpyKind::HostToHost),
            1 => Some(CudaMemcpyKind::HostToDevice),
            2 => Some(CudaMemcpyKind::DeviceToHost),
            3 => Some(CudaMemcpyKind::DeviceToDevice),
            _ => None,
        }
    }

    /// Raw integer form, for passing to the FFI.
    pub fn as_raw(self) -> c_int {
        self as c_int
    }
}

// =============================================================================
// RAW FFI DECLARATIONS
// =============================================================================
//
// Only compiled when the `cuda` feature is enabled. Without it, this
// block is absent and the crate exposes only the Rust-level helpers
// above.
//
// The function signatures mirror the CUDA runtime API exactly. Sizes
// are `usize` (which matches `size_t` on every platform CUDA supports),
// pointers are raw `*mut c_void`, and the error type is a plain `c_int`
// that the caller wraps in `CudaStatus`.

#[cfg(feature = "cuda")]
extern "C" {
    /// Allocate `size` bytes on the current device. On success,
    /// `*dev_ptr` is set to the base pointer of the allocation.
    ///
    /// CUDA docs: "Allocates size bytes of linear memory on the
    /// device and returns in *devPtr a pointer to the allocated
    /// memory. The allocated memory is suitably aligned for any
    /// kind of variable."
    pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int;

    /// Free a previously-allocated device pointer. Passing a null
    /// pointer is explicitly allowed by the runtime and is a no-op;
    /// the real backend will sometimes depend on that.
    pub fn cudaFree(dev_ptr: *mut c_void) -> c_int;

    /// Allocate `size` bytes of page-locked ("pinned") host memory.
    /// This is the buffer the GPU can DMA into with no copy through
    /// the kernel's page cache, which is where the freeze hot path's
    /// 24 GB/s comes from.
    pub fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> c_int;

    /// Allocate `size` bytes of page-locked host memory with flags.
    ///
    /// With `cudaHostAllocWriteCombined` (0x04), the allocation uses
    /// write-combining memory that bypasses L1/L2 cache snooping on
    /// the PCIe bus. This improves H2D DMA throughput by up to 40%
    /// for buffers the CPU writes and the GPU reads (the restore
    /// hot path). CPU reads from WC memory are extremely slow — only
    /// use this for write-then-DMA buffers.
    pub fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;

    /// Free a previously-allocated pinned host buffer. Same null-is-
    /// no-op semantics as `cudaFree`.
    pub fn cudaFreeHost(ptr: *mut c_void) -> c_int;

    /// Page-lock an existing host memory range so the GPU can DMA
    /// into or out of it without the driver's hidden-copy fallback.
    ///
    /// Unlike `cudaMallocHost`/`cudaHostAlloc`, the memory is *not*
    /// owned by CUDA — the caller retains ownership (it was
    /// `malloc`'d, `mmap`'d, etc.). The runtime pins the pages in
    /// place for the duration of registration and unpins them on
    /// `cudaHostUnregister`. Registering an `mmap`'d region of a
    /// large snapshot file lets the pipelined restore path issue
    /// `cudaMemcpyAsync` directly from the mapped pages — no
    /// intermediate memcpy through a `cudaHostAlloc`'d staging
    /// buffer, which is how we get PCIe-saturating throughput for
    /// the `thaw serve` pre-staged-RAM path.
    ///
    /// `ptr` must be page-aligned on most kernels; `size` should
    /// cover the contiguous byte range the caller intends to DMA
    /// from. `flags` is one of the `CUDA_HOST_REGISTER_*` constants
    /// above — thaw uses `CUDA_HOST_REGISTER_DEFAULT` because our
    /// flow does not need portable or mapped semantics.
    ///
    /// Failure modes to expect in production:
    ///   - `cudaErrorHostMemoryAlreadyRegistered` — an earlier call
    ///     forgot to `cudaHostUnregister`. The safe wrapper treats
    ///     this as a hard error; it is a logic bug.
    ///   - `cudaErrorInvalidValue` — often a sign the container's
    ///     `ulimit -l` (locked-memory limit) is too low for the
    ///     requested size. The Python-side fallback path handles
    ///     this by retrying with the non-registered memcpy path.
    pub fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: c_uint) -> c_int;

    /// Release a previously-registered host memory range. Must be
    /// called exactly once per successful `cudaHostRegister` before
    /// the underlying memory is freed or unmapped. The safe wrapper
    /// enforces this via an RAII guard in `thaw_runtime::backend`.
    pub fn cudaHostUnregister(ptr: *mut c_void) -> c_int;

    /// Copy `count` bytes from `src` to `dst`. `kind` selects the
    /// direction — see `CudaMemcpyKind`.
    ///
    /// This is a synchronous copy: it does not return until the copy
    /// has completed on the default stream. Orchestration code that
    /// wants concurrency should use `cudaMemcpyAsync` with an explicit
    /// stream, which thaw will add to this FFI block at the point the
    /// backend actually needs it — not before.
    pub fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
    ) -> c_int;

    /// Return the most recent error from any CUDA runtime call on the
    /// current thread, and clear the sticky state. If you see a
    /// surprising error from a function that "should" have succeeded,
    /// it is almost always because some earlier call left the state
    /// poisoned and nobody cleared it.
    pub fn cudaGetLastError() -> c_int;

    /// Return a pointer to a static, null-terminated string describing
    /// the given error code. The pointer is valid for the lifetime of
    /// the process; callers must not free it.
    ///
    /// We do not wrap this in a safe helper in this crate — see the
    /// `Display` impl on `CudaStatus` for the reason. The wrapper
    /// lives one layer up.
    pub fn cudaGetErrorString(error: c_int) -> *const core::ffi::c_char;

    // -----------------------------------------------------------------
    // Stream management + async copy — the pipelined restore hot path.
    // -----------------------------------------------------------------

    /// Create a new CUDA stream. On success, `*stream` is set to a
    /// handle for the new stream. Streams allow concurrent kernel
    /// execution and async memory copies; the pipelined restore uses
    /// two streams to overlap disk I/O with DMA transfers.
    pub fn cudaStreamCreate(stream: *mut CudaStream) -> c_int;

    /// Block the host thread until all previously submitted work on
    /// `stream` has completed. This is the fence that the double-
    /// buffer loop waits on before reusing a pinned buffer.
    pub fn cudaStreamSynchronize(stream: CudaStream) -> c_int;

    /// Destroy a previously created stream. Any pending work on the
    /// stream is completed before the stream is destroyed.
    pub fn cudaStreamDestroy(stream: CudaStream) -> c_int;

    /// Asynchronous copy of `count` bytes from `src` to `dst` on the
    /// given `stream`. Returns immediately — the copy executes on
    /// the GPU's DMA engine concurrently with host code. The caller
    /// must not free or mutate the source buffer until
    /// `cudaStreamSynchronize` confirms completion.
    ///
    /// Both `src` and `dst` must be either device memory or pinned
    /// host memory. Passing pageable host memory silently falls back
    /// to synchronous transfer — the single worst performance trap
    /// in CUDA.
    pub fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: CudaStream,
    ) -> c_int;
}

// =============================================================================
// TESTS
// =============================================================================
//
// Everything in this test module must run on a Mac with no CUDA. That
// means we only test the Rust-side helpers here. The real FFI is
// exercised by integration tests in the RealCuda wrapper crate, which
// is gated on `--features cuda` and only runs in CI on a GPU host.

#[cfg(test)]
mod tests {
    use super::*;

    /// `CudaStatus::SUCCESS` round-trips through `.ok()` to `Ok(())`.
    /// Smallest possible claim on the error-handling helper.
    #[test]
    fn status_success_is_ok() {
        assert!(CudaStatus::SUCCESS.is_ok());
        assert_eq!(CudaStatus::SUCCESS.ok(), Ok(()));
    }

    /// Any non-zero status round-trips through `.ok()` to `Err(self)`,
    /// preserving the code so callers can match on it.
    #[test]
    fn status_nonzero_is_err() {
        let err = CudaStatus(7);
        assert!(!err.is_ok());
        assert_eq!(err.ok(), Err(CudaStatus(7)));
    }

    /// Named status constants have the integer values the CUDA runtime
    /// documents. If any of these shift, every downstream caller that
    /// compares against the names would silently misclassify errors —
    /// pinning them as a test makes that impossible.
    #[test]
    fn named_status_constants_have_documented_values() {
        assert_eq!(CudaStatus::SUCCESS.0, 0);
        assert_eq!(CudaStatus::INVALID_VALUE.0, 1);
        assert_eq!(CudaStatus::MEMORY_ALLOCATION.0, 2);
    }

    /// `CudaMemcpyKind::from_raw` round-trips through `as_raw` for
    /// every defined variant. The reverse direction too: any integer
    /// outside the 0..=3 range should become `None` rather than
    /// silently landing on a neighboring variant.
    #[test]
    fn memcpy_kind_round_trips() {
        for kind in [
            CudaMemcpyKind::HostToHost,
            CudaMemcpyKind::HostToDevice,
            CudaMemcpyKind::DeviceToHost,
            CudaMemcpyKind::DeviceToDevice,
        ] {
            assert_eq!(CudaMemcpyKind::from_raw(kind.as_raw()), Some(kind));
        }
        assert_eq!(CudaMemcpyKind::from_raw(4), None);
        assert_eq!(CudaMemcpyKind::from_raw(-1), None);
        assert_eq!(CudaMemcpyKind::from_raw(99), None);
    }

    /// `CudaMemcpyKind` variants have the integer values the CUDA
    /// runtime documents. Same rationale as the status-code test.
    #[test]
    fn memcpy_kind_variants_have_documented_values() {
        assert_eq!(CudaMemcpyKind::HostToHost.as_raw(), 0);
        assert_eq!(CudaMemcpyKind::HostToDevice.as_raw(), 1);
        assert_eq!(CudaMemcpyKind::DeviceToHost.as_raw(), 2);
        assert_eq!(CudaMemcpyKind::DeviceToDevice.as_raw(), 3);
    }

    /// On a no-feature build, `CUDA_AVAILABLE` must be `false`. On a
    /// feature build, `true`. This is the contract that downstream
    /// runtime-dispatch code depends on.
    #[test]
    fn cuda_available_matches_feature() {
        #[cfg(feature = "cuda")]
        assert!(CUDA_AVAILABLE);
        #[cfg(not(feature = "cuda"))]
        assert!(!CUDA_AVAILABLE);
    }

    /// `CudaStatus` Display includes the numeric code. This is the
    /// behavior panic messages and log lines will rely on; pinning it
    /// so a future "prettier" Display impl does not accidentally drop
    /// the code.
    #[test]
    fn status_display_contains_numeric_code() {
        let s = format!("{}", CudaStatus(42));
        assert!(s.contains("42"), "Display was {s:?}");
    }
}
