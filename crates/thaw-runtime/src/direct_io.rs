// crates/thaw-runtime/src/direct_io.rs
//
// =============================================================================
// DIRECT I/O ABSTRACTION
// =============================================================================
//
// Provides a thin wrapper over file I/O that attempts O_DIRECT on
// Linux (bypassing the kernel page cache) and falls back gracefully
// on platforms that do not support it (macOS, some filesystems).
//
// The pipelined restore reads directly into pinned memory via pread,
// so the page cache would just waste DRAM bandwidth copying data
// through buffers the CPU will never touch again. O_DIRECT skips
// that entirely, making disk→pinned DMA the only copy.
//
// =============================================================================

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

/// A file handle that may have been opened with O_DIRECT.
///
/// The `direct` flag records whether O_DIRECT actually took effect,
/// so callers can log it without re-checking.
pub struct DirectFile {
    file: File,
    direct: bool,
}

impl DirectFile {
    /// The underlying `File` handle.
    pub fn file(&self) -> &File {
        &self.file
    }

    /// Whether O_DIRECT (or equivalent) is active.
    pub fn is_direct(&self) -> bool {
        self.direct
    }
}

/// Open a file for reading, attempting O_DIRECT if `try_direct` is
/// true. Falls back silently to buffered I/O if the platform or
/// filesystem does not support it.
pub fn open_direct(path: &Path, try_direct: bool) -> io::Result<DirectFile> {
    if try_direct {
        if let Some(f) = try_open_direct(path) {
            return Ok(DirectFile {
                file: f,
                direct: true,
            });
        }
    }
    let file = OpenOptions::new().read(true).open(path)?;
    Ok(DirectFile {
        file,
        direct: false,
    })
}

/// Open a file for writing, attempting O_DIRECT if `try_direct`
/// is true. Falls back silently to buffered I/O if the platform or
/// filesystem does not support it.
///
/// When `truncate` is true the file is created+truncated; when
/// false the file is expected to exist and is opened write-only
/// without length change. The `freeze_pipelined_to_file` path opens
/// a single underlying file twice (one O_DIRECT fd, one buffered
/// fallback for ragged edges) — the first call truncates, the
/// second attaches without disturbing the first.
///
/// Used by `freeze_pipelined` to write `.thaw` files without
/// thrashing the page cache — the bytes come straight out of pinned
/// host memory that the CPU will never touch again.
pub fn open_direct_write(
    path: &Path,
    try_direct: bool,
    truncate: bool,
) -> io::Result<DirectFile> {
    if try_direct {
        if let Some(f) = try_open_direct_write(path, truncate) {
            return Ok(DirectFile {
                file: f,
                direct: true,
            });
        }
    }
    let file = OpenOptions::new()
        .write(true)
        .create(truncate)
        .truncate(truncate)
        .open(path)?;
    Ok(DirectFile {
        file,
        direct: false,
    })
}

/// Platform-specific O_DIRECT attempt. Returns `Some(File)` on
/// success, `None` if the platform doesn't support it or the open
/// fails.
#[cfg(target_os = "linux")]
fn try_open_direct(path: &Path) -> Option<File> {
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .ok()
}

/// O_DIRECT for write with the direct flag. When `truncate` is
/// true the file is created+truncated; otherwise the existing file
/// is opened without length change.
#[cfg(target_os = "linux")]
fn try_open_direct_write(path: &Path, truncate: bool) -> Option<File> {
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .write(true)
        .create(truncate)
        .truncate(truncate)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .ok()
}

/// macOS does not support O_DIRECT. We could use F_NOCACHE via
/// fcntl, but it is not equivalent (it still goes through the VFS
/// buffer layer, just marks pages for eviction). For simplicity we
/// return None and let the caller use buffered I/O. The pipeline
/// structure is the same either way — the difference is only in
/// kernel-side caching behavior.
#[cfg(not(target_os = "linux"))]
fn try_open_direct(_path: &Path) -> Option<File> {
    None
}

#[cfg(not(target_os = "linux"))]
fn try_open_direct_write(_path: &Path, _truncate: bool) -> Option<File> {
    None
}

/// Positional read: read up to `buf.len()` bytes from `file` at
/// the given absolute `offset`, without changing the file's seek
/// position. Returns the number of bytes actually read.
///
/// Uses `libc::pread` on Unix, which is atomic with respect to
/// the file offset — multiple threads can pread from the same fd
/// concurrently without synchronization.
#[cfg(unix)]
pub fn pread_into(file: &DirectFile, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::io::AsRawFd;
    let fd = file.file.as_raw_fd();
    let ret = unsafe {
        libc::pread(
            fd,
            buf.as_mut_ptr() as *mut libc::c_void,
            buf.len(),
            offset as libc::off_t,
        )
    };
    if ret < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(ret as usize)
    }
}

/// Read exactly `buf.len()` bytes from `file` at `offset`, looping
/// on short reads (which can happen with O_DIRECT at EOF or on
/// interrupted syscalls). Returns `Ok(())` on success.
pub fn pread_exact(file: &DirectFile, buf: &mut [u8], offset: u64) -> io::Result<()> {
    let mut pos = 0usize;
    while pos < buf.len() {
        let n = pread_into(file, &mut buf[pos..], offset + pos as u64)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "pread_exact: hit EOF after {} of {} bytes at offset {}",
                    pos,
                    buf.len(),
                    offset
                ),
            ));
        }
        pos += n;
    }
    Ok(())
}

/// Positional write: write up to `buf.len()` bytes to `file` at
/// the given absolute `offset`, without changing the file's seek
/// position. Returns the number of bytes actually written.
///
/// Mirror of `pread_into`. Uses `libc::pwrite` on Unix.
#[cfg(unix)]
pub fn pwrite_from(file: &DirectFile, buf: &[u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::io::AsRawFd;
    let fd = file.file.as_raw_fd();
    let ret = unsafe {
        libc::pwrite(
            fd,
            buf.as_ptr() as *const libc::c_void,
            buf.len(),
            offset as libc::off_t,
        )
    };
    if ret < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(ret as usize)
    }
}

/// Write exactly `buf.len()` bytes to `file` at `offset`, looping
/// on short writes (which can happen on interrupted syscalls).
/// Returns `Ok(())` on success.
///
/// Mirror of `pread_exact`. With O_DIRECT, `buf` must be page-
/// aligned and `buf.len()` a multiple of the block size (4096 on
/// all supported filesystems). Pinned buffers from `cudaMallocHost`
/// satisfy alignment; the caller is responsible for length rounding.
pub fn pwrite_exact(file: &DirectFile, buf: &[u8], offset: u64) -> io::Result<()> {
    let mut pos = 0usize;
    while pos < buf.len() {
        let n = pwrite_from(file, &buf[pos..], offset + pos as u64)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                format!(
                    "pwrite_exact: wrote 0 after {} of {} bytes at offset {}",
                    pos,
                    buf.len(),
                    offset
                ),
            ));
        }
        pos += n;
    }
    Ok(())
}

/// Truncate the file to `len` bytes. Used by the O_DIRECT write
/// path to trim the 4 KiB-aligned tail padding after all chunks are
/// written — O_DIRECT writes must be block-aligned but the final
/// `.thaw` file length is whatever the region table says.
pub fn truncate(file: &DirectFile, len: u64) -> io::Result<()> {
    file.file.set_len(len)
}

/// fsync the file — flushes both data and metadata to stable storage.
/// Required before an atomic rename to guarantee the file's contents
/// survive a power loss: without fsync, the rename can land on disk
/// before the data does, producing a valid directory entry pointing
/// at garbage.
pub fn fsync_file(file: &DirectFile) -> io::Result<()> {
    file.file.sync_all()
}

/// fsync a directory, flushing the directory's metadata (including
/// any `rename` that just landed inside it) to stable storage. On
/// Linux the rename call alone is atomic but not durable; the
/// parent dir must be fsynced to persist the new name.
#[cfg(unix)]
pub fn fsync_dir(path: &Path) -> io::Result<()> {
    let dir = File::open(path)?;
    dir.sync_all()
}

#[cfg(not(unix))]
pub fn fsync_dir(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn pread_reads_at_offset() {
        let mut f = NamedTempFile::new().expect("tempfile");
        f.write_all(b"Hello, pread world!").expect("write");
        f.flush().expect("flush");

        let df = open_direct(f.path(), false).expect("open");
        assert!(!df.is_direct()); // Mac: always buffered

        let mut buf = [0u8; 5];
        pread_exact(&df, &mut buf, 7).expect("pread");
        assert_eq!(&buf, b"pread");
    }

    #[test]
    fn pread_exact_detects_eof() {
        let mut f = NamedTempFile::new().expect("tempfile");
        f.write_all(b"short").expect("write");
        f.flush().expect("flush");

        let df = open_direct(f.path(), false).expect("open");
        let mut buf = [0u8; 100];
        let err = pread_exact(&df, &mut buf, 0).expect_err("should EOF");
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn pwrite_roundtrip_via_pread() {
        let f = NamedTempFile::new().expect("tempfile");

        let wf = open_direct_write(f.path(), false, true).expect("open write");
        assert!(!wf.is_direct()); // Mac: always buffered
        pwrite_exact(&wf, b"Hello, pwrite world!", 0).expect("pwrite");

        // Re-open for read.
        let rf = open_direct(f.path(), false).expect("open read");
        let mut buf = [0u8; 20];
        pread_exact(&rf, &mut buf, 0).expect("pread");
        assert_eq!(&buf, b"Hello, pwrite world!");
    }

    #[test]
    fn pwrite_at_offset_skips_intervening_bytes() {
        let f = NamedTempFile::new().expect("tempfile");

        let wf = open_direct_write(f.path(), false, true).expect("open write");
        pwrite_exact(&wf, b"HEAD", 0).expect("pwrite head");
        pwrite_exact(&wf, b"TAIL", 100).expect("pwrite tail at 100");

        let rf = open_direct(f.path(), false).expect("open read");
        let mut head = [0u8; 4];
        pread_exact(&rf, &mut head, 0).expect("pread head");
        assert_eq!(&head, b"HEAD");

        let mut tail = [0u8; 4];
        pread_exact(&rf, &mut tail, 100).expect("pread tail");
        assert_eq!(&tail, b"TAIL");
    }

    #[test]
    fn truncate_shrinks_file() {
        let f = NamedTempFile::new().expect("tempfile");
        let wf = open_direct_write(f.path(), false, true).expect("open");
        pwrite_exact(&wf, &[0xAAu8; 8192], 0).expect("pwrite");
        truncate(&wf, 100).expect("truncate");
        drop(wf);

        let meta = std::fs::metadata(f.path()).expect("metadata");
        assert_eq!(meta.len(), 100);
    }

    #[test]
    fn pread_at_various_offsets() {
        let mut f = NamedTempFile::new().expect("tempfile");
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        f.write_all(&data).expect("write");
        f.flush().expect("flush");

        let df = open_direct(f.path(), false).expect("open");

        // Read from offset 100, 10 bytes
        let mut buf = [0u8; 10];
        pread_exact(&df, &mut buf, 100).expect("pread");
        assert_eq!(buf, [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]);

        // Read from offset 0, 4 bytes
        let mut buf2 = [0u8; 4];
        pread_exact(&df, &mut buf2, 0).expect("pread");
        assert_eq!(buf2, [0, 1, 2, 3]);
    }
}
