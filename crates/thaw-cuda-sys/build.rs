// build.rs for thaw-cuda-sys
//
// When the `cuda` feature is enabled, this script tells the linker
// where to find `cudart.lib` / `libcudart.so`. Without it, the
// `#[link(name = "cudart")]` attribute in lib.rs knows *what* to
// link but not *where* to find it, and the build fails with
// LNK1181 on Windows or "cannot find -lcudart" on Linux.
//
// The search order:
//   1. $CUDA_PATH/lib/x64 (Windows) or $CUDA_PATH/lib64 (Linux)
//      — this is the env var the CUDA toolkit installer sets.
//   2. The standard NVIDIA install path as a fallback.
//
// If neither exists the build still proceeds — the linker will
// try its own default search paths, which works on systems where
// cudart is on the system library path already (some Linux distros).

fn main() {
    if cfg!(feature = "cuda") {
        if let Some(cuda_path) = find_cuda_path() {
            let lib_dir = if cfg!(target_os = "windows") {
                format!("{}/lib/x64", cuda_path)
            } else {
                format!("{}/lib64", cuda_path)
            };
            println!("cargo:rustc-link-search=native={}", lib_dir);
        }
        println!("cargo:rustc-link-lib=dylib=cudart");
    }
}

fn find_cuda_path() -> Option<String> {
    // Try the standard env var first.
    if let Ok(path) = std::env::var("CUDA_PATH") {
        return Some(path);
    }

    // Fallback: common install locations.
    let candidates = if cfg!(target_os = "windows") {
        vec![
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0",
        ]
    } else {
        vec![
            "/usr/local/cuda",
            "/usr/local/cuda-13.0",
            "/usr/local/cuda-12.6",
            "/usr/local/cuda-12.4",
        ]
    };

    for c in candidates {
        if std::path::Path::new(c).exists() {
            return Some(c.to_string());
        }
    }

    None
}
