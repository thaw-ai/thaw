// thaw-bench: freeze/restore a GPU allocation and print throughput.
//
// Usage (on a machine with CUDA toolkit installed):
//   cargo run --release -p thaw-cli --features cuda -- [size_mb]
//
// Default size is 1024 MB (1 GB). On a 4090 with PCIe 4 x16 you
// should see ~12-13 GB/s for single-stream and the theoretical
// ceiling is ~24 GB/s with double-buffering (future).
//
// Without the `cuda` feature the binary prints an error and exits.

#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use thaw_core::RegionKind;
#[cfg(feature = "cuda")]
use thaw_runtime::{
    freeze, restore, restore_pipelined, CudaBackend, FreezeConfig, FreezeRequest,
    PipelineConfig, RealCuda,
};

#[cfg(feature = "cuda")]
fn format_throughput(bytes: u64, duration: std::time::Duration) -> String {
    let secs = duration.as_secs_f64();
    let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let gbps = gb / secs;
    format!("{:.3} s, {:.2} GB/s", secs, gbps)
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("thaw-bench requires the `cuda` feature. Build with:");
    eprintln!("  cargo run --release -p thaw-cli --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() {
    let size_mb: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let size_bytes = size_mb * 1024 * 1024;

    println!("thaw-bench");
    println!("==========");
    println!("region size: {} MB ({} bytes)", size_mb, size_bytes);
    println!();

    let backend = RealCuda::new();

    // --- Allocate and seed device memory ---
    println!("[1/6] Allocating device memory...");
    let src = backend
        .alloc_device_for_tests(size_bytes as usize)
        .expect("cudaMalloc failed -- is a GPU available?");
    let src_region = src.region();

    println!("[2/6] Seeding device memory with pattern...");
    let mut stage = backend
        .alloc_pinned(size_bytes as usize)
        .expect("cudaMallocHost failed");
    for i in 0..size_bytes as usize {
        stage.as_mut_slice()[i] = ((i as u64).wrapping_mul(13) & 0xFF) as u8;
    }
    backend
        .memcpy_h2d(&src_region, &stage)
        .expect("seed memcpy_h2d failed");

    // --- Raw DMA benchmark (no file format overhead) ---
    println!("[3/6] Raw memcpy_d2h (device -> pinned host)...");
    let t_d2h = Instant::now();
    backend.memcpy_d2h(&mut stage, &src_region).expect("d2h");
    let d2h_time = t_d2h.elapsed();
    println!("  d2h: {}", format_throughput(size_bytes, d2h_time));

    println!("[4/6] Raw memcpy_h2d (pinned host -> device)...");
    let dst = backend
        .alloc_device_for_tests(size_bytes as usize)
        .expect("cudaMalloc for dst failed");
    let dst_region = dst.region();
    let t_h2d = Instant::now();
    backend.memcpy_h2d(&dst_region, &stage).expect("h2d");
    let h2d_time = t_h2d.elapsed();
    println!("  h2d: {}", format_throughput(size_bytes, h2d_time));

    // --- Full pipeline benchmark (with file format) ---
    println!("[5/6] Full freeze/restore pipeline...");
    let mut file_buf: Vec<u8> = Vec::with_capacity(size_bytes as usize + 8192);

    let t_freeze = Instant::now();
    let written = freeze(
        &backend,
        &[FreezeRequest::new(RegionKind::Weights, 0, src_region)],
        &FreezeConfig::default(),
        &mut file_buf,
    )
    .expect("freeze failed");
    let freeze_time = t_freeze.elapsed();
    println!(
        "  freeze:  {} ({} bytes)",
        format_throughput(size_bytes, freeze_time),
        written
    );

    let t_restore = Instant::now();
    let stats = restore(&backend, &file_buf, |kind, _| {
        assert_eq!(kind, RegionKind::Weights);
        Some(dst_region)
    })
    .expect("restore failed");
    let restore_time = t_restore.elapsed();
    println!(
        "  restore: {} ({} regions)",
        format_throughput(size_bytes, restore_time),
        stats.regions_restored
    );

    // --- Pipelined restore benchmark ---
    println!("[6/8] Freezing to temp file for pipelined benchmark...");
    let tmp = {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().expect("create tempfile");
        tmp.write_all(&file_buf).expect("write tempfile");
        tmp.flush().expect("flush tempfile");
        tmp
    };

    println!("[7/8] Pipelined restore (double-buffered, O_DIRECT)...");
    // Zero the destination region so we know the restore did real work.
    {
        let zeros = backend
            .alloc_pinned(size_bytes as usize)
            .expect("alloc zeros");
        backend.memcpy_h2d(&dst_region, &zeros).expect("zero dst");
    }

    let t_pipelined = Instant::now();
    let pstats = restore_pipelined(
        &backend,
        tmp.path(),
        |kind, _| {
            assert_eq!(kind, RegionKind::Weights);
            Some(dst_region)
        },
        &PipelineConfig::default(),
    )
    .expect("pipelined restore failed");
    let pipelined_time = t_pipelined.elapsed();
    println!(
        "  pipelined: {} ({} regions)",
        format_throughput(size_bytes, pipelined_time),
        pstats.regions_restored
    );

    // --- Verify ---
    println!("[8/8] Verifying byte-exact round-trip...");
    {
        let mut check = backend
            .alloc_pinned(size_bytes as usize)
            .expect("alloc check buffer");
        backend
            .memcpy_d2h(&mut check, &dst_region)
            .expect("verify memcpy_d2h");

        let pass = check.as_slice().iter().enumerate().all(|(i, &b)| {
            b == ((i as u64).wrapping_mul(13) & 0xFF) as u8
        });
        if pass {
            println!("  PASS: all {} bytes match", size_bytes);
        } else {
            eprintln!("  FAIL: byte mismatch detected");
            std::process::exit(1);
        }
    }

    // --- Summary ---
    println!();
    println!("Summary");
    println!("-------");
    println!("  raw d2h:      {}", format_throughput(size_bytes, d2h_time));
    println!("  raw h2d:      {}", format_throughput(size_bytes, h2d_time));
    println!("  freeze:       {}", format_throughput(size_bytes, freeze_time));
    println!("  restore:      {}", format_throughput(size_bytes, restore_time));
    println!("  pipelined:    {}", format_throughput(size_bytes, pipelined_time));
    if pipelined_time < restore_time {
        let speedup = restore_time.as_secs_f64() / pipelined_time.as_secs_f64();
        println!("  pipeline vs sync: {:.1}x faster", speedup);
    }
    let total = freeze_time + restore_time;
    println!(
        "  round-trip:   {:.3} s ({} MB)",
        total.as_secs_f64(),
        size_mb
    );
}
