// crates/thaw-runtime/src/backend.rs
//
// =============================================================================
// THE CUDA BACKEND TRAIT
// =============================================================================
//
// This file defines the single seam between "orchestration logic"
// and "anything that talks to a GPU." Every piece of code in
// thaw-runtime that needs to move bytes onto or off of a GPU goes
// through `CudaBackend`. Nothing else.
//
// Why a trait and not a direct FFI call: because we want to run the
// full freeze/restore pipeline on a Mac with no CUDA installed, and
// the only honest way to do that is to give the orchestration code
// a fake GPU it can talk to identically. See docs/TESTING.md section
// 3 for the full rationale.
//
// The trait is intentionally small. The rule is: every method on
// this trait represents one thing the real CUDA implementation
// will genuinely have to do at runtime. If a helper can be built
// on top of the existing methods without adding a new one, it lives
// above the trait, not inside it. "Smallest correct surface area"
// is the design goal.
//
// =============================================================================

use thiserror::Error;

/// An opaque pointer into device memory.
///
/// In the real `cudaMalloc`-backed implementation this is the raw
/// device pointer returned by the driver, cast to `u64`. In the
/// mock implementation it is just an integer key into a HashMap.
/// Either way, no code above this crate interprets the value — it
/// is an opaque handle that only `CudaBackend` implementations
/// know how to dereference.
///
/// u64 (not usize) so the type is the same width on 32-bit hosts
/// as on 64-bit ones. thaw does not target 32-bit GPUs, but the
/// discipline of "one width, picked deliberately, for every
/// persistent field" is the same as in thaw-core.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevicePtr(pub u64);

/// A contiguous region of GPU memory, described by its base
/// pointer and size.
///
/// `DeviceRegion` is the unit every copy in this trait operates on.
/// A region is *not* an owned allocation — it is a view into one.
/// The backend owns the allocation; the region is just the
/// (base, length) pair a caller hands to a copy operation.
///
/// `size` is `u64` for consistency with the thaw-core file format
/// (every size and offset in this project is 64-bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceRegion {
    /// Base pointer of the region.
    pub ptr: DevicePtr,
    /// Length in bytes.
    pub size: u64,
}

impl DeviceRegion {
    /// Convenience constructor.
    pub fn new(ptr: DevicePtr, size: u64) -> Self {
        DeviceRegion { ptr, size }
    }
}

/// A buffer in pinned host memory.
///
/// In the real implementation, pinned host memory is allocated via
/// `cudaHostAlloc` so that the GPU can DMA into it directly with no
/// copy through the kernel page cache. See GLOSSARY: "pinned
/// memory."
///
/// In this layer we do not expose any of that machinery. A
/// `PinnedBuffer` is defined only by what orchestration code can
/// *do* with it: read its bytes, write its bytes, and ask how long
/// it is. The mock implements this as a plain `Vec<u8>`; the real
/// impl will wrap a `cudaHostAlloc`-returned pointer behind the
/// same three operations.
///
/// Ownership: the backend that produced the buffer is responsible
/// for freeing it. The mock does this via `Drop` on the inner
/// `Vec`. The real impl will do it via `cudaFreeHost` in a `Drop`
/// written once in the `RealCuda` crate.
///
/// Concurrency: a `PinnedBuffer` is `Send` but intentionally not
/// `Sync`. Two threads that both want to poke at pinned memory
/// should do so through the backend that owns the allocation,
/// which is `Sync` and can coordinate.
pub struct PinnedBuffer {
    /// The actual storage. A small enum so that one `PinnedBuffer`
    /// type can wrap either a plain `Vec<u8>` (mock) or a raw pointer
    /// returned by `cudaMallocHost` (real), and `Drop` dispatches
    /// correctly without the caller having to know which variant is
    /// inside. The public `as_slice` / `as_mut_slice` accessors see
    /// through the enum, so orchestration code passes the same
    /// `PinnedBuffer` type in both cases.
    ///
    /// The `Cuda` variant is only present when the `cuda` feature is
    /// enabled — on a Mac build the enum is effectively a one-variant
    /// wrapper, and the match arms for the missing variant are
    /// compiled away.
    storage: PinnedStorage,
}

/// Internal discriminator for `PinnedBuffer`. Not `pub` because
/// callers must not care which variant is inside — the whole point of
/// the buffer type is to hide that from them.
enum PinnedStorage {
    /// Heap-allocated Rust memory. Used by the mock backend and by
    /// anyone constructing a `PinnedBuffer` from a plain `Vec<u8>`.
    /// Dropped by the `Vec`'s own destructor — nothing special.
    Heap(Vec<u8>),

    /// A `cudaMallocHost`-returned pointer. The pointer itself is
    /// owned by this variant and must be freed with `cudaFreeHost`
    /// when the `PinnedBuffer` is dropped. See the `Drop` impl at the
    /// bottom of this block for the unsafe ceremony.
    ///
    /// Only present on builds that link CUDA, because otherwise there
    /// is no way to have gotten a pointer to put here in the first
    /// place. Gating the variant itself keeps the Mac build from
    /// having to carry dead FFI types.
    #[cfg(feature = "cuda")]
    Cuda {
        /// Raw pointer returned by `cudaMallocHost`. Must be freed
        /// via `cudaFreeHost` in Drop. Never null for a successfully-
        /// constructed buffer; the constructor maps a null return to
        /// `BackendError::AllocationFailed` before building the enum.
        ptr: *mut core::ffi::c_void,
        /// Length in bytes. `cudaMallocHost` does not remember the
        /// size for you, so we store it here.
        len: usize,
    },
}

impl PinnedBuffer {
    /// Construct a `PinnedBuffer` directly from an owned `Vec<u8>`.
    ///
    /// Intended for the mock backend and for tests. The real backend
    /// has its own constructor path (`from_cuda_alloc`, only compiled
    /// with the `cuda` feature) that goes through `cudaMallocHost`.
    /// Both paths produce the same `PinnedBuffer` type, so
    /// orchestration code never has to know which backend allocated
    /// the buffer.
    pub fn from_vec(bytes: Vec<u8>) -> Self {
        PinnedBuffer {
            storage: PinnedStorage::Heap(bytes),
        }
    }

    /// Construct a `PinnedBuffer` from a raw pointer returned by
    /// `cudaMallocHost`. Only compiled on builds with the `cuda`
    /// feature, because otherwise there is no way to have obtained
    /// the pointer.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been returned by a successful call to
    ///   `cudaMallocHost` with exactly this `len` and not yet freed.
    /// - After calling this constructor, the pointer is owned by the
    ///   `PinnedBuffer`; the caller must not free it, reuse it, or
    ///   pass it to another `PinnedBuffer`. `Drop` will call
    ///   `cudaFreeHost` exactly once.
    /// - `len` must be the length in bytes that was passed to
    ///   `cudaMallocHost`. A mismatch silently produces a buffer
    ///   whose `len()` accessor lies, which every other layer would
    ///   then trust.
    ///
    /// This constructor is used only by the `real` module; the
    /// ordinary public path is `CudaBackend::alloc_pinned`, which is
    /// safe because it wraps this constructor with the invariants
    /// checked internally.
    #[cfg(feature = "cuda")]
    pub(crate) unsafe fn from_cuda_alloc(ptr: *mut core::ffi::c_void, len: usize) -> Self {
        PinnedBuffer {
            storage: PinnedStorage::Cuda { ptr, len },
        }
    }

    /// Length of the buffer in bytes.
    pub fn len(&self) -> usize {
        match &self.storage {
            PinnedStorage::Heap(v) => v.len(),
            #[cfg(feature = "cuda")]
            PinnedStorage::Cuda { len, .. } => *len,
        }
    }

    /// True if the buffer has length zero.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read-only view of the buffer's bytes.
    ///
    /// For the `Cuda` variant this materializes a slice over the raw
    /// pointer via `from_raw_parts`. Safe because the constructor's
    /// contract requires the pointer to be valid for exactly `len`
    /// bytes for the lifetime of the `PinnedBuffer`, and we borrow
    /// `self` for the slice — so the pointer cannot be freed out
    /// from under the slice.
    pub fn as_slice(&self) -> &[u8] {
        match &self.storage {
            PinnedStorage::Heap(v) => v,
            #[cfg(feature = "cuda")]
            PinnedStorage::Cuda { ptr, len } => {
                // SAFETY: see method doc. The `&self` borrow ensures
                // the pointer outlives the returned slice.
                unsafe { core::slice::from_raw_parts(*ptr as *const u8, *len) }
            }
        }
    }

    /// Mutable view of the buffer's bytes.
    ///
    /// Backends use this to write the result of a device-to-host
    /// copy. Orchestration code uses it to fill a buffer before a
    /// host-to-device copy.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match &mut self.storage {
            PinnedStorage::Heap(v) => v,
            #[cfg(feature = "cuda")]
            PinnedStorage::Cuda { ptr, len } => {
                // SAFETY: see `as_slice`. The `&mut self` borrow
                // upgrades the lifetime guarantee to exclusive access.
                unsafe { core::slice::from_raw_parts_mut(*ptr as *mut u8, *len) }
            }
        }
    }

    /// Internal accessor for backends that need the raw pointer form
    /// of a `PinnedBuffer` to hand to `cudaMemcpy`. Only compiled on
    /// `cuda` feature builds, and only callable from inside this
    /// crate, so the raw pointer never escapes to user code.
    ///
    /// Returns a const void pointer at the start of the buffer. Works
    /// for both the `Heap` and `Cuda` variants; the real backend
    /// usually sees the `Cuda` variant but there is no harm in
    /// allowing `cudaMemcpy` to read a `Heap`-backed buffer — that
    /// path is just slower because the memory is not pinned.
    #[cfg(feature = "cuda")]
    pub(crate) fn as_ptr(&self) -> *const core::ffi::c_void {
        match &self.storage {
            PinnedStorage::Heap(v) => v.as_ptr() as *const core::ffi::c_void,
            PinnedStorage::Cuda { ptr, .. } => *ptr,
        }
    }

    /// Mutable counterpart to `as_ptr`, for copies going host-bound.
    #[cfg(feature = "cuda")]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut core::ffi::c_void {
        match &mut self.storage {
            PinnedStorage::Heap(v) => v.as_mut_ptr() as *mut core::ffi::c_void,
            PinnedStorage::Cuda { ptr, .. } => *ptr,
        }
    }
}

// SAFETY: `PinnedBuffer` is `Send` because moving ownership of a
// pinned allocation across threads is allowed by the CUDA runtime
// (the allocation is process-wide, not thread-local). It is NOT
// `Sync` — two threads cannot safely hold `&PinnedBuffer` and issue
// concurrent memcpys through the same raw pointer, because the
// runtime serializes those through the default stream and the
// resulting interleaving is not something orchestration code is
// prepared for. The `Sync` bound is deliberately omitted; if a
// future use case needs it we will add a `SyncPinnedBuffer` wrapper
// instead of loosening this one.
//
// The automatic `Send` that Rust would infer from the raw pointer is
// `false` (raw pointers are `!Send` by default), so we assert it by
// hand for the `Cuda` variant. The `Heap` variant is `Send` via the
// `Vec<u8>` inside it regardless.
#[cfg(feature = "cuda")]
unsafe impl Send for PinnedBuffer {}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        match &mut self.storage {
            PinnedStorage::Heap(_) => {
                // Nothing to do; the Vec's own Drop runs next.
            }
            #[cfg(feature = "cuda")]
            PinnedStorage::Cuda { ptr, .. } => {
                // Best-effort free. If `cudaFreeHost` fails (which in
                // practice only happens if the CUDA context has been
                // torn down, or if the pointer is somehow invalid —
                // and our constructor rejects null), there is nothing
                // a destructor can do about it. We could log, but
                // logging from Drop is its own foot-gun and we do not
                // have a logger wired up in this crate anyway.
                //
                // SAFETY: `ptr` was returned by a successful
                // `cudaMallocHost` per the constructor's contract,
                // and has not been freed (we own the buffer). The
                // runtime accepts null `ptr` as a no-op, but our
                // constructor rejects null, so `*ptr` is always the
                // real allocation here.
                unsafe {
                    let _ = thaw_cuda_sys::cudaFreeHost(*ptr);
                }
            }
        }
    }
}

/// Errors produced by a `CudaBackend` implementation.
///
/// Every variant is reachable from *both* the mock and the real
/// implementation. Variants that would only ever fire for one or
/// the other do not belong here — push them into a more specific
/// error type in the impl's own module.
///
/// `#[non_exhaustive]` so future variants (e.g. `StreamTimeout`)
/// are not breaking changes for downstream crates.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum BackendError {
    /// Allocation failed (out of memory in the real impl;
    /// artificially-enforced limit in the mock, if any).
    #[error("backend allocation failed: requested {bytes} bytes")]
    AllocationFailed { bytes: usize },

    /// A copy was requested against a device pointer that the
    /// backend does not recognize. For the real impl this is "you
    /// passed a pointer from a different context"; for the mock it
    /// is "you passed a `DevicePtr` that was never registered."
    #[error("unknown device pointer: {0:?}")]
    UnknownDevicePtr(DevicePtr),

    /// A copy was requested with a host buffer whose length does
    /// not match the device region's size. This is almost always
    /// a bug in orchestration code (forgot to size the pinned
    /// buffer to the region), and surfacing it as a typed error
    /// makes the bug unmissable.
    #[error("size mismatch: region is {region} bytes, host buffer is {host} bytes")]
    SizeMismatch { region: u64, host: usize },

    /// A CUDA runtime call returned a non-success status. Carries
    /// the raw status code and a short tag describing which
    /// operation was being attempted, so logs and panics can tell
    /// "the alloc failed" from "the memcpy failed" without having
    /// to correlate against a stack trace.
    ///
    /// Only present on `cuda` feature builds because the
    /// `CudaStatus` type itself only exists there. The Mac build's
    /// mock backend does not use CUDA and therefore cannot produce
    /// this variant.
    #[cfg(feature = "cuda")]
    #[error("cuda runtime error during {op}: {status}")]
    Cuda {
        status: thaw_cuda_sys::CudaStatus,
        op: &'static str,
    },

    /// A stream operation failed. Used by `PipelinedBackend`
    /// implementations for stream lifecycle errors (create, sync,
    /// destroy).
    #[error("stream operation failed: {op}")]
    StreamError { op: &'static str },
}

/// The minimal surface area thaw needs from a GPU backend.
///
/// See the module-level doc for the design principle ("smallest
/// correct surface area"). Every method here maps to a specific
/// thing the real CUDA implementation will have to do. Nothing
/// here exists "just in case" — if we do not have an orchestration
/// test that calls it, it does not belong on the trait.
///
/// Implementations must be thread-safe: `Send + Sync`. The real
/// impl satisfies this because `cudaMalloc`/`cudaMemcpy` are
/// thread-safe under the CUDA runtime's documented rules; the
/// mock satisfies it by wrapping its state in a `Mutex`.
pub trait CudaBackend: Send + Sync {
    /// Allocate `bytes` bytes of pinned host memory and return the
    /// resulting buffer.
    ///
    /// In the real impl this is `cudaHostAlloc` with the default
    /// flags. In the mock it is `vec![0; bytes]`. The contract
    /// orchestration code sees is identical: "give me a buffer I
    /// can memcpy into."
    fn alloc_pinned(&self, bytes: usize) -> Result<PinnedBuffer, BackendError>;

    /// Copy `region.size` bytes from device memory at `region.ptr`
    /// into `dst`, starting at offset 0. `dst.len()` must equal
    /// `region.size`.
    ///
    /// Why strict length equality: thaw never does partial copies.
    /// Every region we care about is either fully live or not
    /// touched at all, so an orchestration bug that produces a
    /// size mismatch should fail loudly, not silently truncate.
    fn memcpy_d2h(
        &self,
        dst: &mut PinnedBuffer,
        region: &DeviceRegion,
    ) -> Result<(), BackendError>;

    /// Copy `region.size` bytes from `src` (starting at offset 0)
    /// into device memory at `region.ptr`. Same length-equality
    /// rule as `memcpy_d2h`.
    ///
    /// This is the restore-path counterpart to `memcpy_d2h`.
    /// Between these two methods, the whole freeze/restore hot
    /// path is representable without any other trait method.
    fn memcpy_h2d(
        &self,
        region: &DeviceRegion,
        src: &PinnedBuffer,
    ) -> Result<(), BackendError>;
}

// =============================================================================
// PIPELINED BACKEND — ASYNC STREAMS FOR THE HOT PATH
// =============================================================================

/// An opaque handle to a CUDA stream.
///
/// In the real implementation this wraps a `cudaStream_t` (a pointer
/// cast to `u64`). In the mock it is a monotonically increasing
/// counter used to key into a pending-copy table. Either way, no
/// code above the backend interprets the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamHandle(pub(crate) u64);

/// Extended backend with async (streamed) copy operations.
///
/// The pipelined restore needs explicit control over when copies
/// start and when they complete, because the whole point is to
/// overlap disk I/O with PCIe DMA. The base `CudaBackend` trait
/// is synchronous — every `memcpy_h2d` blocks until the copy is
/// done. This trait adds the async primitives.
///
/// Supertrait of `CudaBackend` so any `PipelinedBackend` can also
/// be used where a plain `CudaBackend` is expected.
pub trait PipelinedBackend: CudaBackend {
    /// Create a new CUDA stream. The returned handle is valid until
    /// `stream_destroy` is called on it.
    fn stream_create(&self) -> Result<StreamHandle, BackendError>;

    /// Destroy a previously created stream. The stream must have
    /// been synchronized first — destroying a stream with pending
    /// work is an error in the mock (to catch bugs) and silently
    /// completes work in the real runtime.
    fn stream_destroy(&self, stream: StreamHandle) -> Result<(), BackendError>;

    /// Block the calling thread until all work on `stream` has
    /// completed. After this returns, any buffers used as sources
    /// in prior `memcpy_h2d_async` calls on this stream may be
    /// safely reused.
    fn stream_sync(&self, stream: &StreamHandle) -> Result<(), BackendError>;

    /// Asynchronous host-to-device copy on the given stream.
    ///
    /// Copies `region.size` bytes from `src` starting at byte
    /// `src_offset` into the device region. Returns immediately;
    /// the actual DMA runs concurrently on the GPU's copy engine.
    ///
    /// The caller **must not** mutate `src` between this call and
    /// the next `stream_sync` on the same stream. Violating this
    /// is undefined behavior in real CUDA (the DMA reads from the
    /// pinned buffer asynchronously).
    ///
    /// `src_offset` allows the pipeline to issue sub-buffer copies
    /// without creating new `PinnedBuffer` types. The pipeline reads
    /// a large chunk into a pinned buffer, then issues multiple
    /// async copies for each region slice within that chunk.
    fn memcpy_h2d_async(
        &self,
        region: &DeviceRegion,
        src: &PinnedBuffer,
        src_offset: usize,
        stream: &StreamHandle,
    ) -> Result<(), BackendError>;

    /// Asynchronous device-to-host copy on the given stream.
    ///
    /// Copies `region.size` bytes from the device region into `dst`
    /// starting at byte `dst_offset`. Returns immediately; the actual
    /// DMA runs concurrently on the GPU's copy engine.
    ///
    /// The caller **must not** read `dst` between this call and the
    /// next `stream_sync` on the same stream. The buffer contents are
    /// undefined until the stream is synchronized.
    ///
    /// `dst_offset` allows the pipeline to pack multiple D2H copies
    /// into the same pinned buffer at different offsets, mirroring
    /// how `memcpy_h2d_async` uses `src_offset` for the reverse
    /// direction.
    fn memcpy_d2h_async(
        &self,
        dst: &mut PinnedBuffer,
        dst_offset: usize,
        region: &DeviceRegion,
        stream: &StreamHandle,
    ) -> Result<(), BackendError>;
}
