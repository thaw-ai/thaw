// crates/thaw-py/src/lib.rs
//
// =============================================================================
// THE PYTHON BINDING LAYER
// =============================================================================
//
// This file is the outermost skin of thaw. It turns Python calls into
// Rust calls and Rust errors into Python exceptions. Nothing else.
//
// Design rule: the PyO3 wrappers (`#[pyfunction]`) must be *dumb*.
// Every non-trivial piece of logic — parsing region kinds, building
// request lists, mapping tuples to typed structs — is a plain Rust
// function that returns `Result<_, ThawPyError>`. The wrappers call
// those functions, handle I/O, and translate `ThawPyError` into
// `PyErr`. This means:
//
//   - Unit tests can exercise the interesting logic without spinning
//     up a Python interpreter.
//
//   - The wrappers are small enough that a reader can audit the
//     Python <-> Rust boundary in one page.
//
//   - When the bindings change (a new function, a renamed argument),
//     the diff lives in one place.
//
// ## The two entry points
//
// `freeze_to_file(path, requests, vllm_commit=None)` — takes a path,
// a list of `(kind_str, logical_id, device_ptr_int, size_int)`
// tuples, and an optional 40-character hex commit string. Writes the
// resulting `.thaw` file to disk. Returns the number of bytes
// written on success.
//
// `restore_from_file(path, mapping)` — takes a path and a list of
// `(kind_str, logical_id, device_ptr_int, size_int)` tuples that
// describe where each region in the file should land. Returns a dict
// `{"regions_restored": int, "bytes_copied": int}` on success.
//
// Both functions dispatch to `RealCuda` when the `cuda` feature is
// enabled. Without the feature they raise `NotImplementedError`
// immediately — this is the Mac development path, where the Python
// glue itself can be iterated on without a GPU in the loop.
//
// ## Why strings for region kinds
//
// `"weights"` / `"kv_live_block"` / `"metadata"` are easier to
// debug than integer enum values, don't require Python-side imports
// of a constants module, and are future-proof: adding a new kind
// adds a match arm in one function and a docstring line. vLLM
// integration code that calls us does not have to track our enum
// discriminants.
//
// =============================================================================

use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::fs::File;
#[cfg(feature = "cuda")]
use std::io::{BufReader, BufWriter, Read, Write};
#[cfg(feature = "cuda")]
use std::path::Path;

use pyo3::exceptions::{PyIOError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

use thaw_core::RegionKind;
use thaw_runtime::{DevicePtr, DeviceRegion};

// FreezeRequest is only constructable when we can actually call the
// orchestrator, which is a feature-gated path below. Importing it
// unconditionally would be fine, but keeping the import next to its
// use keeps `cargo check` quieter on the Mac build.
#[cfg(feature = "cuda")]
use thaw_runtime::{
    freeze, freeze_pipelined, restore, restore_pipelined, restore_pipelined_from_bytes,
    restore_pipelined_from_pre_registered_bytes, restore_pipelined_from_registered_bytes,
    FreezeConfig, FreezeRequest, HostRegistration, PipelineConfig, PipelinedBackend, RealCuda,
};

// =============================================================================
// ERROR TYPE
// =============================================================================
//
// A dedicated error enum for this crate, because `BackendError`,
// `SnapshotError`, and `FreezeError` each carry different context
// and we want the Python-side exception messages to name the
// problem in the caller's vocabulary. Every variant has a clear
// mapping to a Python exception class.

/// Errors produced by the thaw-py binding layer. See the `From`
/// impl at the bottom of the file for how each variant maps to a
/// Python exception class.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ThawPyError {
    /// The caller passed an unknown region kind string. Lists the
    /// known strings in the message so the Python error tells the
    /// user exactly what to fix.
    #[error(
        "unknown region kind {0:?}: expected one of \"weights\", \
         \"kv_live_block\", \"metadata\""
    )]
    UnknownRegionKind(String),

    /// A freeze/restore tuple had the wrong number of elements or a
    /// non-numeric field. PyO3's own extraction errors land here.
    #[error("invalid region tuple: {0}")]
    InvalidTuple(String),

    /// The caller passed a `vllm_commit` string that was not
    /// exactly 40 bytes. We refuse to silently pad or truncate
    /// because the header slot is fixed-size by design.
    #[error("vllm_commit must be exactly 40 bytes, got {0}")]
    InvalidVllmCommit(usize),

    /// The resolver map is missing an entry for a region that
    /// appears in the file. Distinct from `InvalidTuple` so the
    /// Python-side error can say "you forgot to map this region"
    /// instead of "your input is malformed."
    #[error("no device mapping for region: kind={kind:?}, logical_id={logical_id}")]
    UnmappedRegion {
        kind: RegionKind,
        logical_id: u32,
    },

    /// An I/O error from opening, reading, or writing the file.
    #[error("i/o error: {0}")]
    Io(String),

    /// The orchestrator or backend returned an error that is not
    /// specifically one of the categories above. Carries the
    /// upstream message verbatim so the Python user can paste it
    /// into a bug report without losing context.
    #[error("thaw runtime error: {0}")]
    Runtime(String),

    /// This build was compiled without the `cuda` feature, so the
    /// real backend is not available. Raised by the wrappers when
    /// Python calls them on a Mac-style build.
    #[error(
        "thaw was built without the cuda feature; freeze/restore are \
         not available in this build"
    )]
    CudaUnavailable,
}

// =============================================================================
// PURE LOGIC: TESTABLE WITHOUT PYO3
// =============================================================================
//
// Everything below this line is plain Rust that returns `Result<_,
// ThawPyError>`. The `#[pyfunction]` wrappers further down invoke
// these and translate the error into a `PyErr`. Putting the logic
// here means the unit tests can cover it without any Python in the
// loop — no pyo3::prepare_freethreaded_python, no pytest subprocess.

/// Parse a Python-side region kind string into the typed enum.
///
/// The accepted strings mirror the Rust enum variants in
/// snake_case, which is the Python convention. If we ever need to
/// add a new kind, this function and `ThawPyError::UnknownRegionKind`
/// are the only two places that change.
pub fn parse_region_kind(s: &str) -> Result<RegionKind, ThawPyError> {
    match s {
        "weights" => Ok(RegionKind::Weights),
        "kv_live_block" => Ok(RegionKind::KvLiveBlock),
        "metadata" => Ok(RegionKind::Metadata),
        other => Err(ThawPyError::UnknownRegionKind(other.to_string())),
    }
}

/// Parse a 40-character `vllm_commit` string into the fixed-size
/// byte array the header expects.
///
/// Why 40 bytes: the header slot is 40 bytes (see thaw-core
/// header.rs). The canonical value is a 40-character ASCII git SHA,
/// which fits exactly. We refuse to pad or truncate because a
/// silent transform would make it possible for two different vLLM
/// commits to produce header slots that compare equal — which
/// would defeat the whole purpose of the field.
///
/// The input is taken as `&str` because that is what PyO3 extracts
/// a Python `str` into with no copy. We convert to bytes at the
/// boundary.
pub fn parse_vllm_commit(s: &str) -> Result<[u8; 40], ThawPyError> {
    let bytes = s.as_bytes();
    if bytes.len() != 40 {
        return Err(ThawPyError::InvalidVllmCommit(bytes.len()));
    }
    let mut out = [0u8; 40];
    out.copy_from_slice(bytes);
    Ok(out)
}

/// Build a list of freeze requests from the Python-side tuple list.
///
/// Each tuple is `(kind_str, logical_id, device_ptr, size)`. We
/// convert each field to its typed form and collect them into a
/// `Vec<FreezeRequest>`. The `#[cfg(feature = "cuda")]` gate is
/// because `FreezeRequest` is only imported on feature builds; the
/// rest of the function would otherwise be redundant with
/// `parse_region_kind` called in a loop.
///
/// The tuple list type is `Vec<(String, u32, u64, u64)>` — we
/// receive the tuples from Python already extracted into Rust types
/// by PyO3, so this function is pure Rust on both ends.
#[cfg(feature = "cuda")]
pub fn build_freeze_requests(
    tuples: Vec<(String, u32, u64, u64)>,
) -> Result<Vec<FreezeRequest>, ThawPyError> {
    let mut out = Vec::with_capacity(tuples.len());
    for (kind_str, logical_id, ptr, size) in tuples {
        let kind = parse_region_kind(&kind_str)?;
        out.push(FreezeRequest::new(
            kind,
            logical_id,
            DeviceRegion::new(DevicePtr(ptr), size),
        ));
    }
    Ok(out)
}

/// Build the restore-side lookup table from a tuple list.
///
/// Returns a `HashMap<(RegionKind, u32), DeviceRegion>` keyed on
/// `(kind, logical_id)` — the same key the file entries carry. The
/// restore orchestrator accepts a closure, so the wrapper function
/// captures this map and closes over it.
///
/// Duplicate keys in the input are not an error here: later entries
/// overwrite earlier ones. That is the principle of least surprise
/// for a Python dict-like input, and the restore pipeline itself
/// would have no way to act on two conflicting mappings anyway.
pub fn build_restore_map(
    tuples: Vec<(String, u32, u64, u64)>,
) -> Result<HashMap<(RegionKind, u32), DeviceRegion>, ThawPyError> {
    let mut out = HashMap::with_capacity(tuples.len());
    for (kind_str, logical_id, ptr, size) in tuples {
        let kind = parse_region_kind(&kind_str)?;
        out.insert(
            (kind, logical_id),
            DeviceRegion::new(DevicePtr(ptr), size),
        );
    }
    Ok(out)
}

// =============================================================================
// PYTHON ENTRY POINTS
// =============================================================================
//
// These are the `#[pyfunction]` wrappers. They are deliberately tiny:
// each one does the I/O, calls into the testable logic above or
// directly into thaw-runtime, and converts the result. Any logic you
// find tempting to add here almost certainly belongs one function up.

/// Freeze a list of device regions into a `.thaw` file.
///
/// Python signature:
///
/// ```python
/// thaw.freeze_to_file(
///     path: str,
///     requests: list[tuple[str, int, int, int]],
///     vllm_commit: str | None = None,
/// ) -> int
/// ```
///
/// `requests` is a list of `(kind, logical_id, device_ptr, size)`
/// tuples where `kind` is one of `"weights"`, `"kv_live_block"`,
/// `"metadata"`. Returns the number of bytes written.
///
/// Raises:
///   - `NotImplementedError` if the extension was built without
///     the `cuda` feature.
///   - `ValueError` for malformed inputs (unknown region kind,
///     wrong-length vllm_commit).
///   - `IOError` for filesystem problems.
///   - `RuntimeError` for anything the orchestrator returned that
///     does not fit the above.
#[pyfunction]
#[pyo3(signature = (path, requests, vllm_commit=None))]
fn freeze_to_file(
    path: &str,
    requests: Vec<(String, u32, u64, u64)>,
    vllm_commit: Option<&str>,
) -> PyResult<u64> {
    #[cfg(not(feature = "cuda"))]
    {
        // Keep the argument signatures referenced so the compiler
        // does not complain about unused parameters on Mac builds.
        let _ = (path, requests, vllm_commit);
        return Err(PyErr::from(ThawPyError::CudaUnavailable));
    }

    #[cfg(feature = "cuda")]
    {
        let backend = RealCuda::new();
        let typed_requests = build_freeze_requests(requests)?;

        let config = match vllm_commit {
            Some(s) => FreezeConfig {
                vllm_commit: Some(parse_vllm_commit(s)?),
            },
            None => FreezeConfig::default(),
        };

        let file = File::create(Path::new(path))
            .map_err(|e| ThawPyError::Io(format!("open {path}: {e}")))?;
        let mut writer = BufWriter::new(file);

        let written = freeze(&backend, &typed_requests, &config, &mut writer)
            .map_err(|e| ThawPyError::Runtime(format!("{e}")))?;
        writer
            .flush()
            .map_err(|e| ThawPyError::Io(format!("flush {path}: {e}")))?;
        Ok(written)
    }
}

/// Pipelined freeze: double-buffered async D2H with overlapped writes.
///
/// Python signature:
///
/// ```python
/// thaw.freeze_to_file_pipelined(
///     path: str,
///     requests: list[tuple[str, int, int, int]],
///     vllm_commit: str | None = None,
/// ) -> dict
/// ```
///
/// Same semantics as `freeze_to_file`, but uses double-buffered async
/// D2H DMA to overlap GPU copies with disk writes. Returns a dict
/// `{"regions_frozen": int, "bytes_copied": int}`.
#[pyfunction]
#[pyo3(signature = (path, requests, vllm_commit=None))]
fn freeze_to_file_pipelined(
    py: Python<'_>,
    path: &str,
    requests: Vec<(String, u32, u64, u64)>,
    vllm_commit: Option<&str>,
) -> PyResult<PyObject> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (py, path, requests, vllm_commit);
        return Err(PyErr::from(ThawPyError::CudaUnavailable));
    }

    #[cfg(feature = "cuda")]
    {
        let backend = RealCuda::new();
        let typed_requests = build_freeze_requests(requests)?;

        let config = match vllm_commit {
            Some(s) => FreezeConfig {
                vllm_commit: Some(parse_vllm_commit(s)?),
            },
            None => FreezeConfig::default(),
        };

        let file = File::create(Path::new(path))
            .map_err(|e| ThawPyError::Io(format!("open {path}: {e}")))?;
        let mut writer = BufWriter::new(file);

        let stats = freeze_pipelined(&backend, &typed_requests, &config, &mut writer)
            .map_err(|e| ThawPyError::Runtime(format!("{e}")))?;
        writer
            .flush()
            .map_err(|e| ThawPyError::Io(format!("flush {path}: {e}")))?;

        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("regions_frozen", stats.regions_frozen)?;
        dict.set_item("bytes_copied", stats.bytes_copied)?;
        Ok(dict.into())
    }
}

/// Restore a `.thaw` file onto a list of device regions.
///
/// Python signature:
///
/// ```python
/// thaw.restore_from_file(
///     path: str,
///     mapping: list[tuple[str, int, int, int]],
/// ) -> dict
/// ```
///
/// `mapping` describes where each region in the file should land:
/// each tuple is `(kind, logical_id, device_ptr, size)`. The call
/// returns `{"regions_restored": int, "bytes_copied": int}`.
///
/// Raises:
///   - `NotImplementedError` if the extension was built without
///     the `cuda` feature.
///   - `ValueError` for unknown region kinds or missing mappings
///     (any region in the file with no entry in `mapping`).
///   - `IOError` for filesystem problems.
///   - `RuntimeError` for anything else the orchestrator returned.
#[pyfunction]
fn restore_from_file(
    py: Python<'_>,
    path: &str,
    mapping: Vec<(String, u32, u64, u64)>,
) -> PyResult<PyObject> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (py, path, mapping);
        return Err(PyErr::from(ThawPyError::CudaUnavailable));
    }

    #[cfg(feature = "cuda")]
    {
        let backend = RealCuda::new();
        let lookup = build_restore_map(mapping)?;

        // Read the whole file into memory. Same Phase 1
        // simplification as the orchestrator itself — a later
        // streaming restore will take a `Read` directly.
        let mut file = BufReader::new(
            File::open(Path::new(path))
                .map_err(|e| ThawPyError::Io(format!("open {path}: {e}")))?,
        );
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| ThawPyError::Io(format!("read {path}: {e}")))?;

        let stats = restore(&backend, &bytes, |kind, logical_id| {
            lookup.get(&(kind, logical_id)).copied()
        })
        .map_err(|e| ThawPyError::Runtime(format!("{e}")))?;

        // Return a Python dict instead of a tuple. Dicts are self-
        // describing at the call site (`stats["bytes_copied"]`
        // reads better than `stats[1]`) and are trivially
        // extensible if we add more fields later.
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("regions_restored", stats.regions_restored)?;
        dict.set_item("bytes_copied", stats.bytes_copied)?;
        Ok(dict.into())
    }
}

/// Pipelined restore: double-buffered async DMA with O_DIRECT.
///
/// Python signature:
///
/// ```python
/// thaw.restore_from_file_pipelined(
///     path: str,
///     mapping: list[tuple[str, int, int, int]],
///     chunk_size_mb: int = 64,
///     direct_io: bool = True,
/// ) -> dict
/// ```
///
/// Same semantics as `restore_from_file`, but uses the pipelined
/// restore path: two pinned buffers, two CUDA streams, overlapped
/// disk I/O and DMA transfers. Significantly faster for large files.
#[pyfunction]
#[pyo3(signature = (path, mapping, chunk_size_mb=64, direct_io=true))]
fn restore_from_file_pipelined(
    py: Python<'_>,
    path: &str,
    mapping: Vec<(String, u32, u64, u64)>,
    chunk_size_mb: usize,
    direct_io: bool,
) -> PyResult<PyObject> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (py, path, mapping, chunk_size_mb, direct_io);
        return Err(PyErr::from(ThawPyError::CudaUnavailable));
    }

    #[cfg(feature = "cuda")]
    {
        let backend = RealCuda::new();
        let lookup = build_restore_map(mapping)?;

        let config = PipelineConfig {
            chunk_size: chunk_size_mb * 1024 * 1024,
            try_direct_io: direct_io,
        };

        let stats = restore_pipelined(
            &backend,
            std::path::Path::new(path),
            |kind, logical_id| lookup.get(&(kind, logical_id)).copied(),
            &config,
        )
        .map_err(|e| ThawPyError::Runtime(format!("{e}")))?;

        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("regions_restored", stats.regions_restored)?;
        dict.set_item("bytes_copied", stats.bytes_copied)?;
        Ok(dict.into())
    }
}

/// RAM-backed pipelined restore: reads from a byte buffer instead of disk.
///
/// Python signature:
///
/// ```python
/// thaw.restore_from_bytes_pipelined(
///     data: bytes | mmap | bytearray,  # any buffer-protocol object
///     mapping: list[tuple[str, int, int, int]],
///     chunk_size_mb: int = 64,
/// ) -> dict
/// ```
///
/// Same as `restore_from_file_pipelined`, but the snapshot data is
/// passed as a Python buffer object already in memory. Accepts `bytes`,
/// `mmap.mmap`, `bytearray`, or any object implementing the buffer
/// protocol. With mmap on /dev/shm this is zero-copy: the kernel maps
/// the same physical RAM pages, no allocation or memcpy needed.
#[pyfunction]
#[pyo3(signature = (data, mapping, chunk_size_mb=64))]
fn restore_from_bytes_pipelined(
    py: Python<'_>,
    data: pyo3::buffer::PyBuffer<u8>,
    mapping: Vec<(String, u32, u64, u64)>,
    chunk_size_mb: usize,
) -> PyResult<PyObject> {
    // SAFETY: PyBuffer<u8> guarantees the buffer is valid, contiguous,
    // and lives as long as the PyBuffer handle. We hold the GIL (py)
    // and don't release it during restore, so the buffer won't be
    // invalidated. For mmap this points directly at the mapped pages.
    let data_slice = unsafe {
        std::slice::from_raw_parts(data.buf_ptr() as *const u8, data.len_bytes())
    };

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (py, data_slice, mapping, chunk_size_mb);
        return Err(PyErr::from(ThawPyError::CudaUnavailable));
    }

    #[cfg(feature = "cuda")]
    {
        let backend = RealCuda::new();
        let lookup = build_restore_map(mapping)?;

        let config = PipelineConfig {
            chunk_size: chunk_size_mb * 1024 * 1024,
            try_direct_io: false, // not relevant for memory path
        };

        let stats = restore_pipelined_from_bytes(
            &backend,
            data_slice,
            |kind, logical_id| lookup.get(&(kind, logical_id)).copied(),
            &config,
        )
        .map_err(|e| ThawPyError::Runtime(format!("{e}")))?;

        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("regions_restored", stats.regions_restored)?;
        dict.set_item("bytes_copied", stats.bytes_copied)?;
        Ok(dict.into())
    }
}

/// Zero-copy RAM-backed pipelined restore: DMA directly from the buffer.
///
/// Python signature:
///
/// ```python
/// thaw.restore_from_bytes_pipelined_zerocopy(
///     data: bytes | mmap | bytearray,
///     mapping: list[tuple[str, int, int, int]],
/// ) -> dict
/// ```
///
/// Same semantics as `restore_from_bytes_pipelined`, but uses
/// `cudaHostRegister` to pin the caller's buffer in place and
/// `cudaMemcpyAsync` directly from the mapped pages — no intermediate
/// staging buffer, no per-chunk memcpy. For a `/dev/shm` or file-
/// backed mmap this is a true PCIe-saturating zero-copy restore.
///
/// Failure modes the caller should handle:
///   - `RuntimeError: cudaHostRegister` — usually `ulimit -l` is too
///     low for the snapshot size, or the buffer is not page-aligned.
///     Fall back to `restore_from_bytes_pipelined`.
///   - `ValueError` — malformed mapping, same as the other restore
///     entry points.
#[pyfunction]
fn restore_from_bytes_pipelined_zerocopy(
    py: Python<'_>,
    data: pyo3::buffer::PyBuffer<u8>,
    mapping: Vec<(String, u32, u64, u64)>,
) -> PyResult<PyObject> {
    // SAFETY: see `restore_from_bytes_pipelined`. Same invariants: the
    // PyBuffer keeps the underlying allocation alive, we hold the GIL
    // for the whole call, and for mmap this points straight at the
    // mapped pages (page-aligned, contiguous — exactly what
    // `cudaHostRegister` wants).
    let data_slice = unsafe {
        std::slice::from_raw_parts(data.buf_ptr() as *const u8, data.len_bytes())
    };

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (py, data_slice, mapping);
        return Err(PyErr::from(ThawPyError::CudaUnavailable));
    }

    #[cfg(feature = "cuda")]
    {
        let backend = RealCuda::new();
        let lookup = build_restore_map(mapping)?;

        // The zero-copy path does not chunk — `chunk_size` is
        // unused — but we construct a default `PipelineConfig` so
        // the function signature is symmetric with the other
        // restore entry points.
        let config = PipelineConfig::default();

        let stats = restore_pipelined_from_registered_bytes(
            &backend,
            data_slice,
            |kind, logical_id| lookup.get(&(kind, logical_id)).copied(),
            &config,
        )
        .map_err(|e| ThawPyError::Runtime(format!("{e}")))?;

        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("regions_restored", stats.regions_restored)?;
        dict.set_item("bytes_copied", stats.bytes_copied)?;
        Ok(dict.into())
    }
}

// =============================================================================
// PERSISTENT PINNED REGISTRATION (slot-warm path for thaw serve)
// =============================================================================
//
// `cudaHostRegister` is O(pages) — pinning 16 GB takes ~7 s. That cost
// dominates every per-restore zero-copy call and makes the zero-copy
// path slower than the chunked path when measured per-load. For
// `thaw serve`, where a slot mmaps the snapshot once at warm-up and
// then reloads the same bytes into the GPU many times, we want to pay
// the registration cost exactly once.
//
// `PinnedMmap` is the amortization vehicle: construct it on slot warm
// (`cudaHostRegister` happens in `__init__`) and reuse it on every
// subsequent `restore_from_pinned_mmap` call (which skips registration
// and goes straight to DMA). Drop unpins.

/// A Python buffer that has been pinned in place for DMA.
///
/// Python signature:
///
/// ```python
/// pinned = thaw.PinnedMmap(data)  # data: mmap | bytes | bytearray | any buffer
/// stats = thaw.restore_from_pinned_mmap(pinned, mapping)
/// # ... reuse `pinned` across many restores ...
/// del pinned                        # unpins via cudaHostUnregister
/// ```
///
/// Construction calls `cudaHostRegister` on the underlying pages. The
/// resulting handle keeps a reference to the source buffer (so the
/// Python mmap cannot be closed under us) and the CUDA registration
/// (so the pages remain DMA-addressable). Dropping the handle on the
/// Python side runs `cudaHostUnregister` and releases the buffer.
///
/// Constructing one is the slow step (O(pages), seconds for large
/// snapshots). Reusing one across many `restore_from_pinned_mmap`
/// calls is the whole point — that is what turns the amortized
/// per-restore cost into just the PCIe DMA itself.
///
/// Raises `RuntimeError` if `cudaHostRegister` fails — usually
/// `ulimit -l` is too low for the snapshot size, or the buffer is not
/// page-aligned. The caller should fall back to
/// `restore_from_bytes_pipelined`.
#[pyclass(unsendable)]
pub struct PinnedMmap {
    // Field drop order is declaration order. We want the CUDA
    // registration to drop first (cudaHostUnregister) and then the
    // PyBuffer to release its Python reference. Reversing this would
    // risk asking CUDA to unpin memory whose underlying mmap had
    // already been torn down by Python.
    #[cfg(feature = "cuda")]
    _registration: Option<HostRegistration>,
    _buffer: pyo3::buffer::PyBuffer<u8>,
    // Raw ptr into the buffer's bytes. Read by `restore_from_pinned_mmap`
    // on cuda builds; unused on non-cuda where construction always errors.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    ptr: u64,
    size: usize,
}

#[pymethods]
impl PinnedMmap {
    /// Pin a Python buffer in place for DMA.
    #[new]
    fn new(data: pyo3::buffer::PyBuffer<u8>) -> PyResult<Self> {
        let ptr = data.buf_ptr() as *mut u8;
        let size = data.len_bytes();

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (ptr, size, data);
            Err(PyErr::from(ThawPyError::CudaUnavailable))
        }

        #[cfg(feature = "cuda")]
        {
            let backend = RealCuda::new();
            // SAFETY: the PyBuffer keeps the Python mmap (or other
            // buffer-protocol object) alive for the lifetime of the
            // returned PinnedMmap. The pages are contiguous by the
            // buffer protocol's contract. If `ptr` is not page-aligned
            // or the range cannot be pinned, `cudaHostRegister`
            // returns an error which we propagate.
            let registration = unsafe { backend.host_register(ptr, size) }
                .map_err(|e| ThawPyError::Runtime(format!("{e}")))?;
            Ok(PinnedMmap {
                _registration: Some(registration),
                _buffer: data,
                ptr: ptr as u64,
                size,
            })
        }
    }

    /// Length in bytes of the registered range.
    fn __len__(&self) -> usize {
        self.size
    }

    /// Number of bytes pinned.
    #[getter]
    fn size(&self) -> usize {
        self.size
    }
}

/// Restore a snapshot using a pre-registered `PinnedMmap`.
///
/// Python signature:
///
/// ```python
/// thaw.restore_from_pinned_mmap(
///     pinned: PinnedMmap,
///     mapping: list[tuple[str, int, int, int]],
/// ) -> dict
/// ```
///
/// Skips the `cudaHostRegister` call entirely — the pages were pinned
/// when `pinned` was constructed. The per-restore cost is the PCIe
/// DMA itself plus a handful of microseconds of plan construction. In
/// `thaw serve` this is the path that turns a 7-second slot reload
/// into a sub-second one.
#[pyfunction]
fn restore_from_pinned_mmap(
    py: Python<'_>,
    pinned: &PinnedMmap,
    mapping: Vec<(String, u32, u64, u64)>,
) -> PyResult<PyObject> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (py, pinned, mapping);
        return Err(PyErr::from(ThawPyError::CudaUnavailable));
    }

    #[cfg(feature = "cuda")]
    {
        let backend = RealCuda::new();
        let lookup = build_restore_map(mapping)?;
        let config = PipelineConfig::default();

        // SAFETY: `pinned` holds the PyBuffer (keeps the mmap alive)
        // and the HostRegistration (keeps the pages pinned), both for
        // the duration of this call because we borrow `&PinnedMmap`.
        let data_slice = unsafe {
            std::slice::from_raw_parts(pinned.ptr as *const u8, pinned.size)
        };

        let stats = restore_pipelined_from_pre_registered_bytes(
            &backend,
            data_slice,
            |kind, logical_id| lookup.get(&(kind, logical_id)).copied(),
            &config,
        )
        .map_err(|e| ThawPyError::Runtime(format!("{e}")))?;

        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("regions_restored", stats.regions_restored)?;
        dict.set_item("bytes_copied", stats.bytes_copied)?;
        Ok(dict.into())
    }
}

// =============================================================================
// MODULE REGISTRATION
// =============================================================================
//
// This is the magic symbol Python's import machinery looks for. The
// function name matches the cdylib name (`thaw`), which is what
// appears in the manifest's `[lib] name = "thaw"` stanza.

/// Register the `thaw` Python module. PyO3 finds this via the
/// `#[pymodule]` attribute and wires it up as the extension module
/// entry point.
#[pymodule]
fn thaw(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(freeze_to_file, m)?)?;
    m.add_function(wrap_pyfunction!(freeze_to_file_pipelined, m)?)?;
    m.add_function(wrap_pyfunction!(restore_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(restore_from_file_pipelined, m)?)?;
    m.add_function(wrap_pyfunction!(restore_from_bytes_pipelined, m)?)?;
    m.add_function(wrap_pyfunction!(restore_from_bytes_pipelined_zerocopy, m)?)?;
    m.add_class::<PinnedMmap>()?;
    m.add_function(wrap_pyfunction!(restore_from_pinned_mmap, m)?)?;
    Ok(())
}

// =============================================================================
// ERROR CONVERSION
// =============================================================================
//
// Each `ThawPyError` variant lands on the Python exception class
// that most naturally describes the problem. `ValueError` for
// malformed inputs, `IOError` for filesystem trouble,
// `NotImplementedError` for the feature-disabled case,
// `RuntimeError` for everything else.

impl From<ThawPyError> for PyErr {
    fn from(e: ThawPyError) -> PyErr {
        use ThawPyError::*;
        match &e {
            UnknownRegionKind(_) | InvalidTuple(_) | InvalidVllmCommit(_)
            | UnmappedRegion { .. } => PyValueError::new_err(e.to_string()),
            Io(_) => PyIOError::new_err(e.to_string()),
            CudaUnavailable => PyNotImplementedError::new_err(e.to_string()),
            Runtime(_) => pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================
//
// Only the pure-Rust logic is tested here. The PyO3 wrappers
// themselves are thin enough that a unit test would mostly be
// testing PyO3 itself, which is not our job. Actual
// import-and-call tests live in a separate pytest directory and
// run in CI on the GPU box.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_known_region_kinds() {
        assert_eq!(parse_region_kind("weights").unwrap(), RegionKind::Weights);
        assert_eq!(
            parse_region_kind("kv_live_block").unwrap(),
            RegionKind::KvLiveBlock
        );
        assert_eq!(
            parse_region_kind("metadata").unwrap(),
            RegionKind::Metadata
        );
    }

    #[test]
    fn parse_unknown_region_kind_errors() {
        let err = parse_region_kind("WEIGHTS").expect_err("case-sensitive");
        match err {
            ThawPyError::UnknownRegionKind(s) => assert_eq!(s, "WEIGHTS"),
            other => panic!("expected UnknownRegionKind, got {other:?}"),
        }
        // Hyphens are a common typo for underscores in Python.
        assert!(parse_region_kind("kv-live-block").is_err());
        // Empty string.
        assert!(parse_region_kind("").is_err());
    }

    #[test]
    fn parse_vllm_commit_round_trips_40_bytes() {
        let s = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef";
        let bytes = parse_vllm_commit(s).expect("40 bytes");
        assert_eq!(&bytes, s.as_bytes());
    }

    #[test]
    fn parse_vllm_commit_rejects_wrong_length() {
        // Too short.
        let err = parse_vllm_commit("deadbeef").expect_err("too short");
        match err {
            ThawPyError::InvalidVllmCommit(n) => assert_eq!(n, 8),
            other => panic!("expected InvalidVllmCommit, got {other:?}"),
        }
        // Too long.
        let err = parse_vllm_commit(&"a".repeat(41)).expect_err("too long");
        match err {
            ThawPyError::InvalidVllmCommit(n) => assert_eq!(n, 41),
            other => panic!("expected InvalidVllmCommit, got {other:?}"),
        }
        // Exactly 39 — the off-by-one to watch for.
        assert!(parse_vllm_commit(&"a".repeat(39)).is_err());
    }

    #[test]
    fn build_restore_map_produces_keyed_lookup() {
        let tuples = vec![
            ("weights".to_string(), 0, 0x1000, 128),
            ("kv_live_block".to_string(), 7, 0x2000, 64),
            ("metadata".to_string(), 0, 0x3000, 32),
        ];
        let map = build_restore_map(tuples).expect("build");
        assert_eq!(map.len(), 3);

        let weights = map.get(&(RegionKind::Weights, 0)).expect("weights");
        assert_eq!(weights.ptr, DevicePtr(0x1000));
        assert_eq!(weights.size, 128);

        let kv = map
            .get(&(RegionKind::KvLiveBlock, 7))
            .expect("kv block 7");
        assert_eq!(kv.ptr, DevicePtr(0x2000));
        assert_eq!(kv.size, 64);
    }

    /// Two entries with the same `(kind, logical_id)` key: the
    /// later one wins. Pins the "last write wins" policy so a
    /// future refactor that accidentally rejects duplicates
    /// surfaces here rather than breaking caller expectations.
    #[test]
    fn build_restore_map_last_duplicate_wins() {
        let tuples = vec![
            ("weights".to_string(), 0, 0x1000, 128),
            ("weights".to_string(), 0, 0x9000, 128),
        ];
        let map = build_restore_map(tuples).expect("build");
        assert_eq!(map.len(), 1);
        assert_eq!(
            map.get(&(RegionKind::Weights, 0)).unwrap().ptr,
            DevicePtr(0x9000)
        );
    }

    #[test]
    fn build_restore_map_rejects_unknown_kind() {
        let tuples = vec![("gargabe".to_string(), 0, 0x1000, 32)];
        let err = build_restore_map(tuples).expect_err("unknown kind");
        match err {
            ThawPyError::UnknownRegionKind(s) => assert_eq!(s, "gargabe"),
            other => panic!("expected UnknownRegionKind, got {other:?}"),
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn build_freeze_requests_round_trips_tuple_list() {
        let tuples = vec![
            ("weights".to_string(), 0, 0xA000, 256),
            ("kv_live_block".to_string(), 42, 0xB000, 128),
        ];
        let reqs = build_freeze_requests(tuples).expect("build");
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0].kind, RegionKind::Weights);
        assert_eq!(reqs[0].logical_id, 0);
        assert_eq!(reqs[0].device_region.ptr, DevicePtr(0xA000));
        assert_eq!(reqs[0].device_region.size, 256);
        assert_eq!(reqs[1].kind, RegionKind::KvLiveBlock);
        assert_eq!(reqs[1].logical_id, 42);
    }
}
