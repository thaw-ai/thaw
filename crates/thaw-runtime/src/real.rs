// crates/thaw-runtime/src/real.rs
//
// =============================================================================
// REAL CUDA BACKEND
// =============================================================================
//
// This is the "actual GPU" implementation of `CudaBackend`. It is the
// counterpart of `MockCuda` and obeys the same trait contract; every
// test already written against the mock will run unchanged against
// this impl on a box with a CUDA-capable device.
//
// The module is gated behind the `cuda` feature. On a Mac build
// (`default = []`), the file is not compiled at all, so the rest of
// the crate can be built and tested without a CUDA toolchain.
//
// ## What this module does NOT do
//
//   - It does not allocate device memory. The `CudaBackend` trait has
//     no `alloc_device` method, and that is deliberate: thaw always
//     operates on device regions that some other part of the system
//     (vLLM's allocator, a test harness) owns. The role of this
//     module is purely to move bytes in and out of those regions.
//
//   - It does not track "which pointers are live." The runtime does
//     not hand out revocation hooks, and the only way to detect a
//     dead pointer is to try to use it and watch for the error. We
//     pass the error through and let the caller decide.
//
//   - It does not create or manage streams. Every copy here is on
//     the default stream, which is the simplest correct thing. When
//     the real hot path needs two-stream double-buffering for
//     bandwidth, that optimization lives *inside* this module (and
//     reuses the same trait surface); the caller does not have to
//     know.
//
// ## Error mapping
//
// `cudaMalloc`/`cudaMallocHost` both return `cudaErrorMemoryAllocation`
// on OOM, which maps to `BackendError::AllocationFailed`. Everything
// else maps to `BackendError::Cuda { status, op }` — a catch-all that
// preserves the raw code plus the name of the operation we were
// attempting so the caller can tell "the memcpy failed" apart from
// "the alloc failed" without a stack trace.
//
// We could also map `cudaErrorInvalidValue` on a memcpy to
// `UnknownDevicePtr`, but that is lossy: `InvalidValue` fires for lots
// of reasons and collapsing them to "unknown pointer" would mislead
// future debugging. Better to keep the raw code.
//
// =============================================================================

use core::ffi::c_void;
use core::ptr;

use thaw_cuda_sys::{
    cudaFree, cudaGetLastError, cudaHostRegister, cudaMalloc, cudaMallocHost, cudaMemcpy,
    cudaMemcpyAsync, cudaStreamCreate, cudaStreamDestroy, cudaStreamSynchronize,
    CudaMemcpyKind, CudaStatus, CudaStream, CUDA_HOST_REGISTER_DEFAULT,
};

use crate::backend::{
    BackendError, CudaBackend, DevicePtr, DeviceRegion, HostRegistration, PinnedBuffer,
    PipelinedBackend, StreamHandle,
};

/// The real CUDA backend. Zero-sized: the backend holds no state of
/// its own, because CUDA's runtime API is per-thread context and we
/// do not need to remember anything between calls.
///
/// Constructing a `RealCuda` does NOT initialize the runtime — CUDA's
/// runtime API initializes lazily on the first call. This matters
/// because a test harness that constructs the backend but then
/// decides not to use it (say, because no GPU is present) will not
/// trigger initialization just by instantiating the type.
///
/// Safety / threading: `RealCuda` is trivially `Send + Sync` because
/// it has no fields. The underlying CUDA runtime calls are
/// thread-safe under the documented rules, so multiple threads may
/// hold `&RealCuda` and issue concurrent calls without the backend
/// itself having to synchronize.
#[derive(Debug, Default, Clone, Copy)]
pub struct RealCuda;

impl RealCuda {
    /// Construct a new `RealCuda`. Cheap — no runtime side effects.
    pub const fn new() -> Self {
        RealCuda
    }

    /// Allocate a device region of the given size via `cudaMalloc`
    /// and return a `DevicePtr` plus a drop-guard that will
    /// `cudaFree` it when dropped.
    ///
    /// The `CudaBackend` trait deliberately does not expose device
    /// allocation -- production callers (vLLM) hand us device
    /// pointers that came from their own allocator. This method
    /// exists for tests and standalone tools (the benchmark CLI)
    /// that need their own device memory without going through an
    /// external allocator.
    pub fn alloc_device_for_tests(
        &self,
        bytes: usize,
    ) -> Result<OwnedDeviceRegion, BackendError> {
        let mut raw: *mut c_void = ptr::null_mut();
        // SAFETY: `cudaMalloc` writes the allocated pointer into
        // `*raw` on success, leaving it unchanged on failure. We
        // check the return and only trust the pointer on success.
        let status = CudaStatus(unsafe { cudaMalloc(&mut raw as *mut *mut c_void, bytes) });
        if !status.is_ok() {
            if status == CudaStatus::MEMORY_ALLOCATION {
                return Err(BackendError::AllocationFailed { bytes });
            }
            return Err(BackendError::Cuda {
                status,
                op: "cudaMalloc",
            });
        }
        Ok(OwnedDeviceRegion {
            region: DeviceRegion::new(DevicePtr(raw as u64), bytes as u64),
        })
    }
}

/// A `DeviceRegion` plus ownership of the underlying `cudaMalloc`
/// allocation. Drop calls `cudaFree`. Used by tests and the
/// benchmark CLI.
pub struct OwnedDeviceRegion {
    region: DeviceRegion,
}

impl OwnedDeviceRegion {
    /// The borrowed `DeviceRegion` for use in `memcpy_*` calls.
    pub fn region(&self) -> DeviceRegion {
        self.region
    }
}

impl Drop for OwnedDeviceRegion {
    fn drop(&mut self) {
        // SAFETY: the region's pointer came from a successful
        // `cudaMalloc` in `alloc_device_for_tests` and has not been
        // freed yet (we own it). `cudaFree` accepts null as a no-op
        // but we never construct an `OwnedDeviceRegion` with a null
        // pointer.
        unsafe {
            let _ = cudaFree(self.region.ptr.0 as *mut c_void);
        }
    }
}

impl CudaBackend for RealCuda {
    fn alloc_pinned(&self, bytes: usize) -> Result<PinnedBuffer, BackendError> {
        // `cudaMallocHost` on zero bytes is explicitly undefined in
        // the runtime docs. We map it to a successful zero-length
        // heap-backed buffer instead — no FFI call, no pointer, and
        // the `PinnedBuffer` accessors behave identically. The real
        // hot path never asks for zero bytes; this is purely
        // defensive so that an edge-case caller does not get UB.
        if bytes == 0 {
            return Ok(PinnedBuffer::from_vec(Vec::new()));
        }

        let mut raw: *mut c_void = ptr::null_mut();
        // SAFETY: `cudaMallocHost` writes the allocated pointer into
        // `*raw` on success and leaves it unchanged (null) on failure.
        // We check the return and only trust the pointer on success.
        // The runtime requires `bytes > 0`, which we guaranteed above.
        let status =
            CudaStatus(unsafe { cudaMallocHost(&mut raw as *mut *mut c_void, bytes) });
        if !status.is_ok() {
            if status == CudaStatus::MEMORY_ALLOCATION {
                return Err(BackendError::AllocationFailed { bytes });
            }
            return Err(BackendError::Cuda {
                status,
                op: "cudaMallocHost",
            });
        }
        // Defensive: the runtime should never return success with a
        // null pointer, but if it somehow did, we would otherwise
        // hand out a `PinnedBuffer` whose Drop tries to
        // `cudaFreeHost(null)` — which is actually fine (no-op) —
        // but whose slices would all segfault. Treat it as alloc
        // failure.
        if raw.is_null() {
            return Err(BackendError::AllocationFailed { bytes });
        }

        // SAFETY: `raw` is a freshly-allocated, non-null pointer from
        // `cudaMallocHost` with exactly `bytes` bytes of valid
        // storage, and we have not stored it anywhere else. The
        // `PinnedBuffer` takes ownership and will `cudaFreeHost` it
        // in Drop.
        Ok(unsafe { PinnedBuffer::from_cuda_alloc(raw, bytes) })
    }

    fn alloc_pinned_wc(&self, bytes: usize) -> Result<PinnedBuffer, BackendError> {
        if bytes == 0 {
            return Ok(PinnedBuffer::from_vec(Vec::new()));
        }

        let mut raw: *mut c_void = ptr::null_mut();
        // SAFETY: same contract as cudaMallocHost but with WC flag.
        // Write-combining memory bypasses cache snooping on PCIe,
        // improving H2D DMA throughput for the restore hot path.
        let status = CudaStatus(unsafe {
            thaw_cuda_sys::cudaHostAlloc(
                &mut raw as *mut *mut c_void,
                bytes,
                thaw_cuda_sys::CUDA_HOST_ALLOC_WRITE_COMBINED,
            )
        });
        if !status.is_ok() {
            // Fall back to regular pinned memory if WC fails
            // (e.g., on platforms that don't support WC).
            return self.alloc_pinned(bytes);
        }
        if raw.is_null() {
            return self.alloc_pinned(bytes);
        }

        Ok(unsafe { PinnedBuffer::from_cuda_alloc(raw, bytes) })
    }

    fn memcpy_d2h(
        &self,
        dst: &mut PinnedBuffer,
        region: &DeviceRegion,
    ) -> Result<(), BackendError> {
        // Length check first, for the same reason the mock does it:
        // catching a size mismatch here gives a typed error that
        // names the sizes, instead of letting CUDA return a generic
        // `cudaErrorInvalidValue` that the caller has to guess the
        // meaning of.
        if dst.len() as u64 != region.size {
            return Err(BackendError::SizeMismatch {
                region: region.size,
                host: dst.len(),
            });
        }

        // SAFETY: `dst` is borrowed mutably for the duration of the
        // call, so nothing else can touch the host buffer. `region`
        // is a caller-trusted device pointer (the trait's contract
        // puts responsibility for device-side validity on the
        // caller). `cudaMemcpy` is synchronous on the default stream,
        // so the buffer is fully populated by the time we return.
        let status = CudaStatus(unsafe {
            cudaMemcpy(
                dst.as_mut_ptr(),
                region.ptr.0 as *const c_void,
                region.size as usize,
                CudaMemcpyKind::DeviceToHost.as_raw(),
            )
        });
        status.ok().map_err(|status| BackendError::Cuda {
            status,
            op: "cudaMemcpy(d2h)",
        })
    }

    fn memcpy_h2d(
        &self,
        region: &DeviceRegion,
        src: &PinnedBuffer,
    ) -> Result<(), BackendError> {
        if src.len() as u64 != region.size {
            return Err(BackendError::SizeMismatch {
                region: region.size,
                host: src.len(),
            });
        }

        // SAFETY: `src` is borrowed immutably for the duration of
        // the call. `region.ptr` is a caller-trusted device pointer.
        // Synchronous copy on the default stream, so `src` remains
        // valid for the entire DMA.
        let status = CudaStatus(unsafe {
            cudaMemcpy(
                region.ptr.0 as *mut c_void,
                src.as_ptr(),
                region.size as usize,
                CudaMemcpyKind::HostToDevice.as_raw(),
            )
        });
        status.ok().map_err(|status| BackendError::Cuda {
            status,
            op: "cudaMemcpy(h2d)",
        })
    }
}

impl PipelinedBackend for RealCuda {
    fn stream_create(&self) -> Result<StreamHandle, BackendError> {
        let mut raw: CudaStream = core::ptr::null_mut();
        // SAFETY: `cudaStreamCreate` writes the new stream handle
        // into `*raw` on success. We check the return before using it.
        let status = CudaStatus(unsafe {
            cudaStreamCreate(&mut raw as *mut CudaStream)
        });
        if !status.is_ok() {
            return Err(BackendError::Cuda {
                status,
                op: "cudaStreamCreate",
            });
        }
        Ok(StreamHandle(raw as u64))
    }

    fn stream_destroy(&self, stream: StreamHandle) -> Result<(), BackendError> {
        // SAFETY: `stream` came from a prior `stream_create` call
        // whose handle is a `cudaStream_t` cast to `u64`. Casting
        // back is the inverse of what `stream_create` did.
        let status = CudaStatus(unsafe {
            cudaStreamDestroy(stream.0 as CudaStream)
        });
        status.ok().map_err(|status| BackendError::Cuda {
            status,
            op: "cudaStreamDestroy",
        })
    }

    fn stream_sync(&self, stream: &StreamHandle) -> Result<(), BackendError> {
        // SAFETY: same handle-validity argument as `stream_destroy`.
        let status = CudaStatus(unsafe {
            cudaStreamSynchronize(stream.0 as CudaStream)
        });
        status.ok().map_err(|status| BackendError::Cuda {
            status,
            op: "cudaStreamSynchronize",
        })
    }

    fn memcpy_h2d_async(
        &self,
        region: &DeviceRegion,
        src: &PinnedBuffer,
        src_offset: usize,
        stream: &StreamHandle,
    ) -> Result<(), BackendError> {
        let size = region.size as usize;
        if src_offset + size > src.len() {
            return Err(BackendError::SizeMismatch {
                region: region.size,
                host: src.len().saturating_sub(src_offset),
            });
        }

        // SAFETY:
        // - `src` is a PinnedBuffer (pinned host memory), and we
        //   compute the source pointer as base + offset, which is
        //   within bounds (checked above).
        // - `region.ptr` is a caller-trusted device pointer.
        // - The stream handle came from `stream_create`.
        // - The caller must not mutate `src` until `stream_sync`.
        let src_ptr = unsafe {
            (src.as_ptr() as *const u8).add(src_offset) as *const c_void
        };
        let status = CudaStatus(unsafe {
            cudaMemcpyAsync(
                region.ptr.0 as *mut c_void,
                src_ptr,
                size,
                CudaMemcpyKind::HostToDevice.as_raw(),
                stream.0 as CudaStream,
            )
        });
        status.ok().map_err(|status| BackendError::Cuda {
            status,
            op: "cudaMemcpyAsync(h2d)",
        })
    }

    fn memcpy_d2h_async(
        &self,
        dst: &mut PinnedBuffer,
        dst_offset: usize,
        region: &DeviceRegion,
        stream: &StreamHandle,
    ) -> Result<(), BackendError> {
        let size = region.size as usize;
        if dst_offset + size > dst.len() {
            return Err(BackendError::SizeMismatch {
                region: region.size,
                host: dst.len().saturating_sub(dst_offset),
            });
        }

        // SAFETY:
        // - `dst` is a PinnedBuffer (pinned host memory), and we
        //   compute the destination pointer as base + offset, which
        //   is within bounds (checked above).
        // - `region.ptr` is a caller-trusted device pointer.
        // - The stream handle came from `stream_create`.
        // - The caller must not read `dst` until `stream_sync`.
        let dst_ptr = unsafe {
            (dst.as_mut_ptr() as *mut u8).add(dst_offset) as *mut c_void
        };
        let src_ptr = region.ptr.0 as *const c_void;
        let status = CudaStatus(unsafe {
            cudaMemcpyAsync(
                dst_ptr,
                src_ptr,
                size,
                CudaMemcpyKind::DeviceToHost.as_raw(),
                stream.0 as CudaStream,
            )
        });
        status.ok().map_err(|status| BackendError::Cuda {
            status,
            op: "cudaMemcpyAsync(d2h)",
        })
    }

    unsafe fn host_register(
        &self,
        ptr: *mut u8,
        size: usize,
    ) -> Result<HostRegistration, BackendError> {
        if size == 0 {
            // Registering a zero-byte range is explicitly undefined
            // in the runtime docs. A zero-byte guard is a no-op both
            // ways — the caller's `memcpy_h2d_async_raw` loop will
            // have nothing to do anyway.
            return Ok(HostRegistration::noop(ptr, 0));
        }

        // SAFETY: the caller's contract on `host_register` requires
        // `ptr` to be a live, contiguous host range of `size` bytes.
        // `cudaHostRegister` itself will return `cudaErrorInvalidValue`
        // for a misaligned pointer or a range it cannot pin — we
        // surface that error unchanged so the Python-side fallback
        // can retry via the staging-buffer path.
        let status = CudaStatus(cudaHostRegister(
            ptr as *mut c_void,
            size,
            CUDA_HOST_REGISTER_DEFAULT,
        ));
        if !status.is_ok() {
            // CUDA errors are sticky per-thread. If we return the error
            // without clearing it, the next unrelated CUDA call (e.g.
            // a torch tensor op during model warmup) will observe the
            // same error and crash — even though the fallback restore
            // path succeeded. Calling `cudaGetLastError` here consumes
            // the sticky state so the caller's fallback can proceed on
            // a clean context.
            let _ = cudaGetLastError();
            return Err(BackendError::Cuda {
                status,
                op: "cudaHostRegister",
            });
        }

        // SAFETY: `cudaHostRegister` just returned success; the
        // guard's Drop will match it with exactly one
        // `cudaHostUnregister` call.
        Ok(HostRegistration::registered_cuda(ptr, size))
    }

    unsafe fn memcpy_h2d_async_raw(
        &self,
        region: &DeviceRegion,
        src_ptr: *const u8,
        size: usize,
        stream: &StreamHandle,
    ) -> Result<(), BackendError> {
        if size as u64 > region.size {
            return Err(BackendError::SizeMismatch {
                region: region.size,
                host: size,
            });
        }

        // SAFETY: both `region.ptr` and `src_ptr` are caller-trusted
        // per the trait's Safety contract. The stream handle came
        // from `stream_create`. `cudaMemcpyAsync` reads the source
        // bytes asynchronously on the GPU's DMA engine; the caller
        // promised not to mutate them until `stream_sync`.
        let status = CudaStatus(cudaMemcpyAsync(
            region.ptr.0 as *mut c_void,
            src_ptr as *const c_void,
            size,
            CudaMemcpyKind::HostToDevice.as_raw(),
            stream.0 as CudaStream,
        ));
        status.ok().map_err(|status| BackendError::Cuda {
            status,
            op: "cudaMemcpyAsync(h2d raw)",
        })
    }
}

// =============================================================================
// TESTS
// =============================================================================
//
// These run only on a box with a working CUDA runtime — they are
// gated on `cfg(all(test, feature = "cuda"))`. On Mac they compile-
// check (the module itself is gated on `feature = "cuda"`), and on
// the 4090 box they actually fire against libcudart.
//
// The test pattern mirrors the mock tests one-for-one: same sizes,
// same patterns, same assertions. The whole point of writing RealCuda
// against the same trait contract is that the tests line up.
//
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// `alloc_pinned` returns a buffer of exactly the requested size.
    /// Zero-initialization is NOT required by CUDA (`cudaMallocHost`
    /// leaves the bytes undefined), so we only assert on length, not
    /// content.
    #[test]
    fn alloc_pinned_returns_buffer_of_requested_length() {
        let backend = RealCuda::new();
        let buf = backend.alloc_pinned(128).expect("alloc_pinned");
        assert_eq!(buf.len(), 128);
    }

    /// Zero-byte alloc returns a valid empty buffer and does not
    /// call into the FFI (the constructor short-circuits). Pins that
    /// defensive path against a regression that would make it call
    /// `cudaMallocHost(.., 0)` and hit UB.
    #[test]
    fn alloc_pinned_zero_bytes_succeeds() {
        let backend = RealCuda::new();
        let buf = backend.alloc_pinned(0).expect("alloc_pinned(0)");
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    /// `memcpy_h2d` followed by `memcpy_d2h` round-trips arbitrary
    /// bytes through a real device allocation. This is the "same
    /// test as `freeze_then_restore_round_trip_through_mock` but on
    /// real hardware" sanity check.
    #[test]
    fn d2h_and_h2d_round_trip() {
        let backend = RealCuda::new();
        let owned = backend
            .alloc_device_for_tests(256)
            .expect("alloc_device");
        let region = owned.region();

        // Distinct pattern so a zeroed or off-by-one buffer fails
        // loudly.
        let pattern: Vec<u8> = (0..256).map(|i| ((i * 11) & 0xFF) as u8).collect();

        let mut send = backend.alloc_pinned(256).expect("alloc send");
        send.as_mut_slice().copy_from_slice(&pattern);
        backend.memcpy_h2d(&region, &send).expect("h2d");

        let mut recv = backend.alloc_pinned(256).expect("alloc recv");
        backend.memcpy_d2h(&mut recv, &region).expect("d2h");

        assert_eq!(recv.as_slice(), pattern.as_slice());
    }

    /// Size mismatch on `memcpy_d2h` surfaces as `SizeMismatch`
    /// before any FFI call. Same assertion shape as the mock's
    /// `memcpy_d2h_rejects_size_mismatch` test — mirroring the
    /// contract intentionally.
    #[test]
    fn memcpy_d2h_rejects_size_mismatch() {
        let backend = RealCuda::new();
        let owned = backend
            .alloc_device_for_tests(64)
            .expect("alloc_device");
        let region = owned.region();

        // Deliberately wrong host size: 32 bytes for a 64-byte region.
        let mut host = backend.alloc_pinned(32).expect("alloc");
        let err = backend
            .memcpy_d2h(&mut host, &region)
            .expect_err("should reject");
        assert_eq!(
            err,
            BackendError::SizeMismatch {
                region: 64,
                host: 32
            }
        );
    }

    /// Size mismatch on `memcpy_h2d`, mirror of the above.
    #[test]
    fn memcpy_h2d_rejects_size_mismatch() {
        let backend = RealCuda::new();
        let owned = backend.alloc_device_for_tests(16).expect("alloc_device");
        let region = owned.region();

        let host = backend.alloc_pinned(8).expect("alloc");
        let err = backend
            .memcpy_h2d(&region, &host)
            .expect_err("should reject");
        assert_eq!(
            err,
            BackendError::SizeMismatch {
                region: 16,
                host: 8
            }
        );
    }

    /// End-to-end through the freeze/restore orchestrators against
    /// `RealCuda`. If this passes on the 4090 box, the entire Rust
    /// side of thaw works against real hardware — no new
    /// orchestration code, just a second impl of `CudaBackend`.
    #[test]
    fn freeze_restore_round_trip_through_real_cuda() {
        use crate::freeze::{freeze, FreezeConfig, FreezeRequest};
        use crate::restore::restore;
        use thaw_core::RegionKind;

        let backend = RealCuda::new();

        // Seed a device region with a known pattern via h2d, then
        // freeze it using the orchestrator.
        let src = backend.alloc_device_for_tests(1024).expect("alloc src");
        let src_region = src.region();
        let pattern: Vec<u8> = (0..1024).map(|i| ((i * 13) & 0xFF) as u8).collect();
        {
            let mut stage = backend.alloc_pinned(1024).expect("alloc stage");
            stage.as_mut_slice().copy_from_slice(&pattern);
            backend.memcpy_h2d(&src_region, &stage).expect("seed h2d");
        }

        let requests = vec![FreezeRequest::new(RegionKind::Weights, 0, src_region)];
        let mut file: Vec<u8> = Vec::new();
        freeze(&backend, &requests, &FreezeConfig::default(), &mut file)
            .expect("freeze");

        // Allocate a fresh device region, restore into it, and
        // verify the bytes came back byte-exact.
        let dst = backend.alloc_device_for_tests(1024).expect("alloc dst");
        let dst_region = dst.region();
        restore(&backend, &file, |kind, _| {
            assert_eq!(kind, RegionKind::Weights);
            Some(dst_region)
        })
        .expect("restore");

        let mut check = backend.alloc_pinned(1024).expect("alloc check");
        backend.memcpy_d2h(&mut check, &dst_region).expect("d2h");
        assert_eq!(check.as_slice(), pattern.as_slice());
    }
}
