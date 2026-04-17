// crates/thaw-runtime/src/mock.rs
//
// =============================================================================
// MOCK CUDA BACKEND
// =============================================================================
//
// `MockCuda` is a fake GPU. It implements `CudaBackend` using a
// `HashMap<DevicePtr, Vec<u8>>` that pretends to be device memory.
// Every tier-2 test in the project uses it; see docs/TESTING.md
// section 3 for the testing strategy it unlocks.
//
// Deliberate non-goals: the mock does not simulate timing, does
// not simulate allocation failure under realistic memory
// pressure, does not simulate the driver's concurrency quirks.
// It is only there to obey the `CudaBackend` contract in the
// simplest correct way, so that orchestration code written
// against the trait has something to run against on a Mac.
//
// If the orchestration logic is right, it should work with any
// `CudaBackend` that obeys the contract. The mock obeys the
// contract. The real impl will obey the contract. Therefore a
// bug that appears in the real impl but not the mock is a bug
// in `RealCuda` or the C++ layer — and those have their own test
// story.
//
// =============================================================================

use std::collections::HashMap;
use std::sync::Mutex;

use crate::backend::{
    BackendError, CudaBackend, DevicePtr, DeviceRegion, HostRegistration, PinnedBuffer,
    PipelinedBackend, StreamHandle,
};

/// HashMap-backed fake GPU. Thread-safe via an inner `Mutex`.
///
/// Construction: `MockCuda::new()` returns an empty backend with
/// no registered regions. Tests call `register_region` to tell
/// the mock "pretend there is a device allocation at this pointer
/// with these contents." Orchestration code then hands copy
/// operations to the mock through the `CudaBackend` trait, exactly
/// as it will later hand them to the real impl.
/// Deferred copy queued by `memcpy_h2d_async`. Applied to device
/// memory on `stream_sync`.
struct PendingCopy {
    dst_ptr: DevicePtr,
    data: Vec<u8>,
}

pub struct MockCuda {
    /// The fake device memory. Keyed by `DevicePtr`, each entry is
    /// the full byte content of that region. An allocation of N
    /// bytes appears as an N-element `Vec<u8>` in this map.
    ///
    /// Wrapped in a `Mutex` so the backend is `Sync`. The mock
    /// does not simulate concurrent DMA, so a single coarse lock
    /// is fine.
    regions: Mutex<HashMap<DevicePtr, Vec<u8>>>,

    /// Monotonically increasing counter for stream handle IDs.
    next_stream_id: Mutex<u64>,

    /// Pending async copies keyed by stream ID. Copies are deferred
    /// here by `memcpy_h2d_async` and applied to `regions` on
    /// `stream_sync`. This models the real CUDA async contract:
    /// data is not visible on the device until the stream is synced.
    pending: Mutex<HashMap<u64, Vec<PendingCopy>>>,

    /// Count of pending D2H async ops per stream. D2H copies are
    /// applied immediately to the host buffer (the mock doesn't need
    /// to defer them), but we track the count so `stream_destroy`
    /// catches unsynced streams.
    pending_d2h_count: Mutex<HashMap<u64, usize>>,
}

impl MockCuda {
    /// Construct an empty mock backend.
    pub fn new() -> Self {
        MockCuda {
            regions: Mutex::new(HashMap::new()),
            next_stream_id: Mutex::new(1), // 0 reserved for "default stream"
            pending: Mutex::new(HashMap::new()),
            pending_d2h_count: Mutex::new(HashMap::new()),
        }
    }

    /// Register a fake device region with the given pointer and
    /// initial contents.
    ///
    /// This is the "setup" half of the mock: it takes the place
    /// of whatever allocator would have produced the region in a
    /// real run. Tests call this to seed the mock with known
    /// content, then call `memcpy_d2h` through the `CudaBackend`
    /// trait and assert on the bytes that come back.
    ///
    /// Idempotent-ish: calling it twice for the same pointer
    /// overwrites the previous contents, which is the behavior
    /// tests want when resetting state between cases. If you do
    /// not want that, use a fresh `MockCuda`.
    pub fn register_region(&self, ptr: DevicePtr, contents: Vec<u8>) {
        self.regions
            .lock()
            .expect("MockCuda lock poisoned")
            .insert(ptr, contents);
    }

    /// Read-only snapshot of the bytes at a registered region.
    ///
    /// Tests call this after a `memcpy_h2d` to verify that the
    /// device side now contains what the caller uploaded. Real
    /// callers (orchestration code) never use it — they only see
    /// the backend through the trait, which deliberately does not
    /// expose a "give me the device bytes directly" method,
    /// because that is not an operation the real GPU backend can
    /// cheaply provide.
    ///
    /// Returns `None` if the pointer has never been registered.
    pub fn read_region(&self, ptr: DevicePtr) -> Option<Vec<u8>> {
        self.regions
            .lock()
            .expect("MockCuda lock poisoned")
            .get(&ptr)
            .cloned()
    }
}

impl Default for MockCuda {
    fn default() -> Self {
        MockCuda::new()
    }
}

impl CudaBackend for MockCuda {
    fn alloc_pinned(&self, bytes: usize) -> Result<PinnedBuffer, BackendError> {
        // The mock has no artificial allocation limit, so this
        // always succeeds. `vec![0; bytes]` gives us a zeroed
        // buffer of exactly the requested length — the same
        // contract the real `cudaHostAlloc` provides for
        // `cudaHostAllocDefault`.
        Ok(PinnedBuffer::from_vec(vec![0u8; bytes]))
    }

    fn memcpy_d2h(
        &self,
        dst: &mut PinnedBuffer,
        region: &DeviceRegion,
    ) -> Result<(), BackendError> {
        // Length check first. Orchestration bugs that mis-size a
        // pinned buffer are the most common failure mode this
        // check catches, and catching it here means the test
        // output says "size mismatch" instead of some confusing
        // downstream panic.
        if dst.len() as u64 != region.size {
            return Err(BackendError::SizeMismatch {
                region: region.size,
                host: dst.len(),
            });
        }

        let guard = self.regions.lock().expect("MockCuda lock poisoned");
        let src = guard
            .get(&region.ptr)
            .ok_or(BackendError::UnknownDevicePtr(region.ptr))?;

        // The registered region's length is the authoritative
        // size. If a caller registered 100 bytes but the
        // `DeviceRegion` claims 200, the mock treats that as a
        // size mismatch too — the trait contract says a region's
        // size is what the caller says it is, and a mismatch
        // between the claim and the actual backing buffer is a
        // caller bug.
        if src.len() as u64 != region.size {
            return Err(BackendError::SizeMismatch {
                region: region.size,
                host: src.len(),
            });
        }

        dst.as_mut_slice().copy_from_slice(src);
        Ok(())
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

        let mut guard = self.regions.lock().expect("MockCuda lock poisoned");
        let dst = guard
            .get_mut(&region.ptr)
            .ok_or(BackendError::UnknownDevicePtr(region.ptr))?;

        if dst.len() as u64 != region.size {
            return Err(BackendError::SizeMismatch {
                region: region.size,
                host: dst.len(),
            });
        }

        dst.copy_from_slice(src.as_slice());
        Ok(())
    }
}

impl PipelinedBackend for MockCuda {
    fn stream_create(&self) -> Result<StreamHandle, BackendError> {
        let mut id = self.next_stream_id.lock().expect("lock poisoned");
        let handle = StreamHandle(*id);
        *id += 1;
        self.pending
            .lock()
            .expect("lock poisoned")
            .insert(handle.0, Vec::new());
        self.pending_d2h_count
            .lock()
            .expect("lock poisoned")
            .insert(handle.0, 0);
        Ok(handle)
    }

    fn stream_destroy(&self, stream: StreamHandle) -> Result<(), BackendError> {
        let mut pending = self.pending.lock().expect("lock poisoned");
        let mut d2h = self.pending_d2h_count.lock().expect("lock poisoned");
        let d2h_count = d2h.remove(&stream.0).unwrap_or(0);
        match pending.remove(&stream.0) {
            Some(queue) if !queue.is_empty() || d2h_count > 0 => {
                // In the real runtime, destroying a stream with pending
                // work silently completes it. In the mock we treat it
                // as a bug so tests catch misuse.
                Err(BackendError::StreamError {
                    op: "stream_destroy: stream has pending copies (call stream_sync first)",
                })
            }
            Some(_) => Ok(()),
            None => Err(BackendError::StreamError {
                op: "stream_destroy: unknown stream handle",
            }),
        }
    }

    fn stream_sync(&self, stream: &StreamHandle) -> Result<(), BackendError> {
        // Reset D2H pending count (D2H copies were applied eagerly).
        {
            let mut d2h = self.pending_d2h_count.lock().expect("lock poisoned");
            if let Some(count) = d2h.get_mut(&stream.0) {
                *count = 0;
            }
        }

        // Drain pending H2D copies for this stream and apply them.
        let copies = {
            let mut pending = self.pending.lock().expect("lock poisoned");
            let queue = pending
                .get_mut(&stream.0)
                .ok_or(BackendError::StreamError {
                    op: "stream_sync: unknown stream handle",
                })?;
            std::mem::take(queue)
        };

        let mut regions = self.regions.lock().expect("lock poisoned");
        for copy in copies {
            // The copy may target a sub-region (base_ptr + offset).
            // Find the registered region that contains this pointer
            // by checking if copy.dst_ptr falls within any registered
            // region's range.
            let target_addr = copy.dst_ptr.0;
            let mut found = false;
            for (base_ptr, data) in regions.iter_mut() {
                let base = base_ptr.0;
                let end = base + data.len() as u64;
                if target_addr >= base && target_addr < end {
                    let offset = (target_addr - base) as usize;
                    let copy_end = offset + copy.data.len();
                    if copy_end > data.len() {
                        return Err(BackendError::SizeMismatch {
                            region: copy.data.len() as u64,
                            host: data.len() - offset,
                        });
                    }
                    data[offset..copy_end].copy_from_slice(&copy.data);
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(BackendError::UnknownDevicePtr(copy.dst_ptr));
            }
        }
        Ok(())
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

        // Snapshot the source bytes now, just like real CUDA DMA would
        // start reading from pinned memory at call time.
        let data = src.as_slice()[src_offset..src_offset + size].to_vec();

        let mut pending = self.pending.lock().expect("lock poisoned");
        let queue = pending.get_mut(&stream.0).ok_or(BackendError::StreamError {
            op: "memcpy_h2d_async: unknown stream handle",
        })?;
        queue.push(PendingCopy {
            dst_ptr: region.ptr,
            data,
        });
        Ok(())
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

        // Verify stream exists.
        {
            let pending = self.pending.lock().expect("lock poisoned");
            if !pending.contains_key(&stream.0) {
                return Err(BackendError::StreamError {
                    op: "memcpy_d2h_async: unknown stream handle",
                });
            }
        }

        // Read from device memory. Supports sub-region reads: the
        // region's ptr may point into the middle of a registered
        // allocation (e.g. when the pipeline splits a large region
        // across multiple chunks).
        let guard = self.regions.lock().expect("lock poisoned");
        let target_addr = region.ptr.0;

        for (base_ptr, data) in guard.iter() {
            let base = base_ptr.0;
            let end = base + data.len() as u64;
            if target_addr >= base && target_addr + region.size <= end {
                let dev_offset = (target_addr - base) as usize;
                dst.as_mut_slice()[dst_offset..dst_offset + size]
                    .copy_from_slice(&data[dev_offset..dev_offset + size]);

                // Track pending count for stream lifecycle.
                let mut d2h = self.pending_d2h_count.lock().expect("lock poisoned");
                *d2h.entry(stream.0).or_insert(0) += 1;

                return Ok(());
            }
        }

        Err(BackendError::UnknownDevicePtr(region.ptr))
    }

    unsafe fn host_register(
        &self,
        ptr: *mut u8,
        size: usize,
    ) -> Result<HostRegistration, BackendError> {
        // The mock has nothing to pin. The caller still gets a
        // valid guard so orchestration code can exercise the
        // zero-copy pipeline against the mock backend — we just
        // won't issue any actual FFI call when the guard drops.
        Ok(HostRegistration::noop(ptr, size))
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

        // Snapshot the source bytes now, mirroring what real CUDA
        // would do at DMA start. The `unsafe` signature is the
        // caller's promise that `src_ptr..src_ptr+size` is valid,
        // readable host memory.
        let data = core::slice::from_raw_parts(src_ptr, size).to_vec();

        let mut pending = self.pending.lock().expect("lock poisoned");
        let queue = pending.get_mut(&stream.0).ok_or(BackendError::StreamError {
            op: "memcpy_h2d_async_raw: unknown stream handle",
        })?;
        queue.push(PendingCopy {
            dst_ptr: region.ptr,
            data,
        });
        Ok(())
    }
}

// =============================================================================
// TESTS
// =============================================================================
//
// These are the tier-2 "mock integration" tests described in
// docs/TESTING.md section 2: they exercise the full public API of
// the backend abstraction using `MockCuda`, with no real hardware.
// They pin the trait contract so that any future `RealCuda` impl
// can be written against the same assertions.
//
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A freshly-constructed mock has no registered regions.
    /// Smallest possible claim: the type exists and the constructor
    /// runs.
    #[test]
    fn new_mock_has_no_regions() {
        let m = MockCuda::new();
        assert!(m.read_region(DevicePtr(0)).is_none());
    }

    /// `alloc_pinned` returns a buffer of the requested length,
    /// zero-filled.
    #[test]
    fn alloc_pinned_returns_zeroed_buffer_of_requested_length() {
        let m = MockCuda::new();
        let buf = m.alloc_pinned(128).expect("alloc_pinned");
        assert_eq!(buf.len(), 128);
        assert!(buf.as_slice().iter().all(|&b| b == 0));
    }

    /// `register_region` + `read_region` round-trip an arbitrary
    /// byte vector. This is the setup/inspection path tests use
    /// to seed and verify mock state; it is not part of the
    /// trait contract but is the mock's own affordance.
    #[test]
    fn register_then_read_round_trip() {
        let m = MockCuda::new();
        m.register_region(DevicePtr(0x1000), vec![1, 2, 3, 4, 5]);
        assert_eq!(m.read_region(DevicePtr(0x1000)), Some(vec![1, 2, 3, 4, 5]));
    }

    /// `memcpy_d2h` copies the full contents of a registered
    /// region into a pinned buffer of matching size.
    ///
    /// This is the "freeze" direction of the hot path, reduced
    /// to its simplest possible shape. Orchestration code on top
    /// will call this method once per region it freezes.
    #[test]
    fn memcpy_d2h_copies_region_bytes_into_pinned_buffer() {
        let m = MockCuda::new();
        let ptr = DevicePtr(0x2000);
        let payload = vec![0xAA, 0xBB, 0xCC, 0xDD];
        m.register_region(ptr, payload.clone());

        let mut host = m.alloc_pinned(payload.len()).expect("alloc");
        let region = DeviceRegion::new(ptr, payload.len() as u64);
        m.memcpy_d2h(&mut host, &region).expect("memcpy_d2h");

        assert_eq!(host.as_slice(), payload.as_slice());
    }

    /// `memcpy_h2d` copies the full contents of a pinned buffer
    /// into a registered region of matching size. The "restore"
    /// direction of the hot path.
    #[test]
    fn memcpy_h2d_copies_pinned_buffer_into_region() {
        let m = MockCuda::new();
        let ptr = DevicePtr(0x3000);
        // Pre-register the destination region with a known-bad
        // pattern so we can tell the difference between "write
        // happened" and "write silently noop'd."
        m.register_region(ptr, vec![0xFF; 8]);

        let mut host = m.alloc_pinned(8).expect("alloc");
        host.as_mut_slice().copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);

        let region = DeviceRegion::new(ptr, 8);
        m.memcpy_h2d(&region, &host).expect("memcpy_h2d");

        assert_eq!(m.read_region(ptr).unwrap(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    /// Full round-trip through the mock: seed a region, copy out,
    /// zero the pinned buffer, copy in a different pattern, verify.
    ///
    /// This is the smoke test for "the mock actually behaves like
    /// a GPU" — if anything in this chain is wrong, later
    /// orchestration tests built on top of it will all break in
    /// confusing ways. Better to fail here than three levels up.
    #[test]
    fn freeze_then_restore_round_trip_through_mock() {
        let m = MockCuda::new();
        let ptr = DevicePtr(0x4000);
        let original = vec![0x11, 0x22, 0x33, 0x44, 0x55];
        m.register_region(ptr, original.clone());
        let region = DeviceRegion::new(ptr, original.len() as u64);

        // Freeze: device -> host.
        let mut host = m.alloc_pinned(original.len()).expect("alloc");
        m.memcpy_d2h(&mut host, &region).expect("d2h");
        let frozen_bytes = host.as_slice().to_vec();
        assert_eq!(frozen_bytes, original);

        // Clobber the device side to simulate a fresh process
        // where nothing has been restored yet.
        m.register_region(ptr, vec![0u8; original.len()]);
        assert_ne!(m.read_region(ptr).unwrap(), original);

        // Restore: host -> device.
        m.memcpy_h2d(&region, &host).expect("h2d");
        assert_eq!(m.read_region(ptr).unwrap(), original);
    }

    /// Copying from an unregistered region surfaces as a typed
    /// error, not a panic. This is the same "defensive parser"
    /// discipline as thaw-core's `from_bytes` methods.
    #[test]
    fn memcpy_d2h_rejects_unknown_pointer() {
        let m = MockCuda::new();
        let mut host = m.alloc_pinned(4).expect("alloc");
        let region = DeviceRegion::new(DevicePtr(0x9999), 4);
        let err = m
            .memcpy_d2h(&mut host, &region)
            .expect_err("should reject");
        assert_eq!(err, BackendError::UnknownDevicePtr(DevicePtr(0x9999)));
    }

    /// Copying with a mis-sized host buffer is a `SizeMismatch`,
    /// not a panic or a silent truncation.
    #[test]
    fn memcpy_d2h_rejects_size_mismatch() {
        let m = MockCuda::new();
        let ptr = DevicePtr(0x5000);
        m.register_region(ptr, vec![0xAA; 10]);
        let mut host = m.alloc_pinned(6).expect("alloc"); // wrong size
        let region = DeviceRegion::new(ptr, 10);
        let err = m
            .memcpy_d2h(&mut host, &region)
            .expect_err("should reject");
        assert_eq!(
            err,
            BackendError::SizeMismatch {
                region: 10,
                host: 6
            }
        );
    }

    /// Same mismatch check for the other direction.
    #[test]
    fn memcpy_h2d_rejects_size_mismatch() {
        let m = MockCuda::new();
        let ptr = DevicePtr(0x6000);
        m.register_region(ptr, vec![0; 4]);
        let host = m.alloc_pinned(3).expect("alloc"); // wrong size
        let region = DeviceRegion::new(ptr, 4);
        let err = m.memcpy_h2d(&region, &host).expect_err("should reject");
        assert_eq!(
            err,
            BackendError::SizeMismatch {
                region: 4,
                host: 3
            }
        );
    }

    /// Copying to an unregistered region via h2d is the mirror of
    /// the d2h unknown-pointer test.
    #[test]
    fn memcpy_h2d_rejects_unknown_pointer() {
        let m = MockCuda::new();
        let host = m.alloc_pinned(4).expect("alloc");
        let region = DeviceRegion::new(DevicePtr(0xDEAD), 4);
        let err = m.memcpy_h2d(&region, &host).expect_err("should reject");
        assert_eq!(err, BackendError::UnknownDevicePtr(DevicePtr(0xDEAD)));
    }

    /// The trait object flavor works: orchestration code written
    /// against `&dyn CudaBackend` still routes correctly to the
    /// mock. Pinning this as a test means a future refactor that
    /// accidentally makes the trait non-object-safe fires here
    /// instead of at some distant call site.
    #[test]
    fn mock_can_be_used_as_dyn_trait() {
        let m = MockCuda::new();
        let ptr = DevicePtr(0x7000);
        m.register_region(ptr, vec![9, 8, 7, 6]);
        let backend: &dyn CudaBackend = &m;

        let mut host = backend.alloc_pinned(4).expect("alloc");
        backend
            .memcpy_d2h(&mut host, &DeviceRegion::new(ptr, 4))
            .expect("d2h");
        assert_eq!(host.as_slice(), &[9, 8, 7, 6]);
    }

    // =================================================================
    // PipelinedBackend tests
    // =================================================================

    /// stream_create returns distinct handles.
    #[test]
    fn stream_create_returns_distinct_handles() {
        let m = MockCuda::new();
        let s1 = m.stream_create().expect("create 1");
        let s2 = m.stream_create().expect("create 2");
        assert_ne!(s1, s2);
    }

    /// Async copy does NOT modify device memory until stream_sync.
    #[test]
    fn async_copy_deferred_until_sync() {
        let m = MockCuda::new();
        let ptr = DevicePtr(0x8000);
        m.register_region(ptr, vec![0u8; 4]);

        let stream = m.stream_create().expect("create");
        let mut host = m.alloc_pinned(4).expect("alloc");
        host.as_mut_slice().copy_from_slice(&[1, 2, 3, 4]);

        let region = DeviceRegion::new(ptr, 4);
        m.memcpy_h2d_async(&region, &host, 0, &stream)
            .expect("async copy");

        // Before sync: device should still be zeroed.
        assert_eq!(m.read_region(ptr).unwrap(), vec![0, 0, 0, 0]);

        // After sync: device should have the data.
        m.stream_sync(&stream).expect("sync");
        assert_eq!(m.read_region(ptr).unwrap(), vec![1, 2, 3, 4]);

        m.stream_destroy(stream).expect("destroy");
    }

    /// Async copy with src_offset reads from the correct position.
    #[test]
    fn async_copy_with_src_offset() {
        let m = MockCuda::new();
        let ptr = DevicePtr(0x9000);
        m.register_region(ptr, vec![0u8; 3]);

        let stream = m.stream_create().expect("create");
        let mut host = m.alloc_pinned(8).expect("alloc");
        host.as_mut_slice()
            .copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22]);

        // Copy 3 bytes starting at offset 4 in the host buffer.
        let region = DeviceRegion::new(ptr, 3);
        m.memcpy_h2d_async(&region, &host, 4, &stream)
            .expect("async copy");
        m.stream_sync(&stream).expect("sync");

        assert_eq!(m.read_region(ptr).unwrap(), vec![0xEE, 0xFF, 0x11]);
        m.stream_destroy(stream).expect("destroy");
    }

    /// stream_destroy on a stream with pending copies is an error.
    #[test]
    fn stream_destroy_with_pending_copies_is_error() {
        let m = MockCuda::new();
        let ptr = DevicePtr(0xA000);
        m.register_region(ptr, vec![0u8; 4]);

        let stream = m.stream_create().expect("create");
        let host = m.alloc_pinned(4).expect("alloc");
        m.memcpy_h2d_async(&DeviceRegion::new(ptr, 4), &host, 0, &stream)
            .expect("async copy");

        // Destroying without sync should fail.
        let err = m.stream_destroy(stream).expect_err("should fail");
        matches!(err, BackendError::StreamError { .. });
    }

    /// D2H async copies bytes from device to host buffer at offset.
    #[test]
    fn d2h_async_copies_device_to_host() {
        let m = MockCuda::new();
        let ptr = DevicePtr(0xD000);
        m.register_region(ptr, vec![0xAA, 0xBB, 0xCC, 0xDD]);

        let stream = m.stream_create().expect("create");
        let mut host = m.alloc_pinned(8).expect("alloc");

        let region = DeviceRegion::new(ptr, 4);
        m.memcpy_d2h_async(&mut host, 2, &region, &stream)
            .expect("d2h async");
        m.stream_sync(&stream).expect("sync");

        assert_eq!(&host.as_slice()[2..6], &[0xAA, 0xBB, 0xCC, 0xDD]);
        m.stream_destroy(stream).expect("destroy");
    }

    /// D2H async with sub-region read (pointer into middle of allocation).
    #[test]
    fn d2h_async_sub_region_read() {
        let m = MockCuda::new();
        let ptr = DevicePtr(0xE000);
        m.register_region(ptr, vec![0x10, 0x20, 0x30, 0x40, 0x50, 0x60]);

        let stream = m.stream_create().expect("create");
        let mut host = m.alloc_pinned(3).expect("alloc");

        // Read 3 bytes starting at offset 2 within the device region.
        let sub_region = DeviceRegion::new(DevicePtr(0xE002), 3);
        m.memcpy_d2h_async(&mut host, 0, &sub_region, &stream)
            .expect("d2h async");
        m.stream_sync(&stream).expect("sync");

        assert_eq!(host.as_slice(), &[0x30, 0x40, 0x50]);
        m.stream_destroy(stream).expect("destroy");
    }

    /// stream_destroy with pending D2H ops is an error.
    #[test]
    fn stream_destroy_with_pending_d2h_is_error() {
        let m = MockCuda::new();
        let ptr = DevicePtr(0xF000);
        m.register_region(ptr, vec![1, 2, 3, 4]);

        let stream = m.stream_create().expect("create");
        let mut host = m.alloc_pinned(4).expect("alloc");
        m.memcpy_d2h_async(&mut host, 0, &DeviceRegion::new(ptr, 4), &stream)
            .expect("d2h async");

        // Destroying without sync should fail.
        let err = m.stream_destroy(stream).expect_err("should fail");
        matches!(err, BackendError::StreamError { .. });
    }

    /// Multiple async copies on the same stream are applied in order.
    #[test]
    fn multiple_async_copies_applied_in_order() {
        let m = MockCuda::new();
        let ptr = DevicePtr(0xB000);
        m.register_region(ptr, vec![0u8; 4]);

        let stream = m.stream_create().expect("create");

        // First copy: write [1,2,3,4]
        let mut h1 = m.alloc_pinned(4).expect("alloc");
        h1.as_mut_slice().copy_from_slice(&[1, 2, 3, 4]);
        m.memcpy_h2d_async(&DeviceRegion::new(ptr, 4), &h1, 0, &stream)
            .expect("copy 1");

        // Second copy: overwrite with [5,6,7,8]
        let mut h2 = m.alloc_pinned(4).expect("alloc");
        h2.as_mut_slice().copy_from_slice(&[5, 6, 7, 8]);
        m.memcpy_h2d_async(&DeviceRegion::new(ptr, 4), &h2, 0, &stream)
            .expect("copy 2");

        m.stream_sync(&stream).expect("sync");
        // Second copy wins (applied in order).
        assert_eq!(m.read_region(ptr).unwrap(), vec![5, 6, 7, 8]);

        m.stream_destroy(stream).expect("destroy");
    }
}
