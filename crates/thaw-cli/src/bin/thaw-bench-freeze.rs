// thaw-bench-freeze: A/B bench the old writer-based freeze_pipelined
// against the new freeze_pipelined_to_file (O_DIRECT pwrite) path.
//
// Usage (needs CUDA toolkit):
//   cargo run --release -p thaw-cli --bin thaw-bench-freeze --features cuda \
//     -- [size_mb] [chunk_mb]
//
// Default: 1024 MB region, 64 MB chunks.
// Writes both files to $TMPDIR (use a fast SSD — /workspace on RunPod is
// usually a local NVMe, /tmp may be tmpfs on some pods).
//
// Expected on H100 SXM + PCIe Gen5 NVMe: old path ~3 GB/s (buffered writes
// through page cache), new path 10+ GB/s (O_DIRECT, WC pinned memory, no
// page-cache bounce).

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("thaw-bench-freeze requires the `cuda` feature. Build with:");
    eprintln!(
        "  cargo run --release -p thaw-cli --bin thaw-bench-freeze --features cuda"
    );
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;
    use thaw_core::RegionKind;
    use thaw_runtime::{
        freeze_pipelined, freeze_pipelined_to_file, CudaBackend, FreezeConfig,
        FreezeRequest, RealCuda,
    };

    fn fmt(bytes: u64, dur: std::time::Duration) -> String {
        let secs = dur.as_secs_f64();
        let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        format!("{:.3} s, {:.2} GB/s", secs, gb / secs)
    }

    let mut args = std::env::args().skip(1);
    let size_mb: u64 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let chunk_mb: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let size_bytes = size_mb * 1024 * 1024;
    let chunk_size = chunk_mb * 1024 * 1024;

    println!("thaw-bench-freeze");
    println!("=================");
    println!("region size: {} MB", size_mb);
    println!("chunk size:  {} MB", chunk_mb);
    println!("tmpdir:      {}", std::env::temp_dir().display());
    println!();

    let backend = RealCuda::new();

    // Allocate + seed the source device region.
    println!("[setup] cudaMalloc + H2D seed...");
    let src = backend
        .alloc_device_for_tests(size_bytes as usize)
        .expect("cudaMalloc (is a GPU available?)");
    let src_region = src.region();

    let mut stage = backend
        .alloc_pinned(size_bytes as usize)
        .expect("cudaMallocHost for seed");
    for (i, b) in stage.as_mut_slice().iter_mut().enumerate() {
        *b = ((i as u64).wrapping_mul(13) & 0xFF) as u8;
    }
    backend
        .memcpy_h2d(&src_region, &stage)
        .expect("seed memcpy_h2d");
    drop(stage);

    let requests = vec![FreezeRequest::new(RegionKind::Weights, 0, src_region)];
    let config = FreezeConfig {
        chunk_size,
        try_direct_io: true,
        ..FreezeConfig::default()
    };

    // -------- Path A: old freeze_pipelined (BufWriter<File>) ---------
    println!("[A] freeze_pipelined (BufWriter<File>) ...");
    let tmp_a = tempfile::NamedTempFile::new().expect("tempfile A");
    let path_a = tmp_a.path().to_path_buf();
    drop(tmp_a); // reopen below
    let t = Instant::now();
    {
        use std::io::{BufWriter, Write};
        let f = std::fs::File::create(&path_a).expect("create A");
        let mut w = BufWriter::new(f);
        freeze_pipelined(&backend, &requests, &config, &mut w).expect("freeze A");
        w.flush().expect("flush A");
    }
    let dur_a = t.elapsed();
    let size_a = std::fs::metadata(&path_a).expect("stat A").len();
    println!(
        "  {} ({} bytes, file={})",
        fmt(size_bytes, dur_a),
        size_a,
        path_a.display()
    );

    // -------- Path B: new freeze_pipelined_to_file (O_DIRECT) --------
    println!("[B] freeze_pipelined_to_file (O_DIRECT pwrite + WC pinned) ...");
    let tmp_b = tempfile::NamedTempFile::new().expect("tempfile B");
    let path_b = tmp_b.path().to_path_buf();
    drop(tmp_b);
    let t = Instant::now();
    freeze_pipelined_to_file(&backend, &requests, &config, &path_b)
        .expect("freeze B");
    let dur_b = t.elapsed();
    let size_b = std::fs::metadata(&path_b).expect("stat B").len();
    println!(
        "  {} ({} bytes, file={})",
        fmt(size_bytes, dur_b),
        size_b,
        path_b.display()
    );

    // -------- Path B-no-direct: new path with try_direct_io=false ----
    // Isolates the "double-buffer + WC + pwrite" wins from the O_DIRECT win.
    println!("[B'] freeze_pipelined_to_file (buffered pwrite) ...");
    let config_buf = FreezeConfig {
        chunk_size,
        try_direct_io: false,
        ..FreezeConfig::default()
    };
    let tmp_c = tempfile::NamedTempFile::new().expect("tempfile C");
    let path_c = tmp_c.path().to_path_buf();
    drop(tmp_c);
    let t = Instant::now();
    freeze_pipelined_to_file(&backend, &requests, &config_buf, &path_c)
        .expect("freeze B'");
    let dur_c = t.elapsed();
    let size_c = std::fs::metadata(&path_c).expect("stat C").len();
    println!(
        "  {} ({} bytes, file={})",
        fmt(size_bytes, dur_c),
        size_c,
        path_c.display()
    );

    // -------- Byte-equality check: all three files identical? -------
    println!("[verify] file bytes identical across paths...");
    let a = std::fs::read(&path_a).expect("read A");
    let b = std::fs::read(&path_b).expect("read B");
    let c = std::fs::read(&path_c).expect("read B'");
    assert_eq!(a.len(), b.len(), "size mismatch A vs B");
    assert_eq!(a.len(), c.len(), "size mismatch A vs B'");
    assert_eq!(a, b, "bytes differ A vs B");
    assert_eq!(a, c, "bytes differ A vs B'");
    println!("  PASS: all three paths wrote identical bytes.");

    // Cleanup temp files (the NamedTempFile guards were dropped earlier).
    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);
    let _ = std::fs::remove_file(&path_c);

    println!();
    println!("Summary ({} MB)", size_mb);
    println!("--------");
    println!("  A   writer path:           {}", fmt(size_bytes, dur_a));
    println!("  B'  to_file, buffered:     {}", fmt(size_bytes, dur_c));
    println!("  B   to_file, O_DIRECT:     {}", fmt(size_bytes, dur_b));
    if dur_b < dur_a {
        let speedup = dur_a.as_secs_f64() / dur_b.as_secs_f64();
        println!("  B vs A speedup:            {:.2}x", speedup);
    }
}
