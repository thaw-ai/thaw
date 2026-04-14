# Setup — one page per machine

This is the minimum you need to run on each of your three dev machines. No more, no less. If a step fails, fix it before moving on — don't skip.

---

## Mac (M5 Pro) — primary dev loop for non-CUDA code

Your Mac is where you'll do the majority of Phase 1 work (file format, I/O layer, tests). No GPU required, no CUDA toolchain.

### Prerequisites

1. **Rust toolchain (stable, 1.75+):**
   ```bash
   rustc --version   # should print 1.75.0 or newer
   cargo --version
   ```
   If not installed:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Git** (you already have this).

### Verify the scaffold works

```bash
cd ~/Desktop/yc/thaw
cargo build -p thaw-core
cargo test -p thaw-core
```

You should see `test new_header_has_thaw_magic_bytes ... ok` and a green summary.

If you see anything else, stop and debug. The first test passing is the "foundation is solid" checkpoint for the entire project.

### Recommended: `cargo-watch` for a tight feedback loop

```bash
cargo install cargo-watch
cargo watch -x 'test -p thaw-core'
```

This re-runs the tests every time you save a file. It turns TDD from "type, switch to terminal, type, switch back" into a single screen.

---

## Laptop (Windows + RTX 4090) — GPU integration tests

The laptop is where tier-3 GPU tests live: you need real CUDA and a real GPU, but you don't need full-scale models.

### Prerequisites

1. **CUDA toolkit 12.x or newer:**
   ```cmd
   nvcc --version
   ```
   Install from https://developer.nvidia.com/cuda-downloads if missing.

2. **Rust (same as Mac).**

3. **A C++ compiler that `nvcc` recognizes.** On Windows, this means the MSVC toolchain — install "Desktop development with C++" via Visual Studio Installer.

4. **Git.**

### Verify the CUDA path works

Once `thaw-cuda-sys` exists (Phase 1 day 5ish), you'll run:
```bash
cargo test -p thaw-runtime --features cuda
```

Until then, the laptop just runs the same non-CUDA tests as the Mac:
```bash
cargo test -p thaw-core
```

### Disk: use NVMe, not the spinner

If your laptop has both an NVMe SSD and a hard disk, put the repo on the NVMe. The whole project is I/O-bound; running benchmarks from a spinner will give garbage numbers.

---

## Colab Pro (H100) — full benchmarks and demo runs

Colab is where the headline numbers come from. Keep Colab-specific config in a notebook under `tests/colab/` (doesn't exist yet — we'll create it during Phase 3).

### Session setup (every time)

Colab doesn't persist state across sessions, so you re-run setup each time. Budget ~2 minutes for this.

```python
# Cell 1: clone the repo
!git clone https://github.com/matteso1/thaw.git
%cd thaw

# Cell 2: install Rust (cached across sessions if you use a persistent disk,
# otherwise takes ~40s per cold start)
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
!source $HOME/.cargo/env

# Cell 3: verify CUDA
!nvcc --version
!nvidia-smi

# Cell 4: verify the scaffold
!cargo test -p thaw-core
```

### GPU quota watchdog

Colab Pro gives you "compute units" that burn while a GPU is attached. An H100 burns them fast. **Disconnect the runtime when you're not actively running a benchmark.** A 12-hour idle Colab session with an H100 attached will drain a month of compute units in one night.

### When Colab isn't enough

If Colab runs out of units, or you need a persistent H100 for more than a few hours:
- Lambda Labs: ~$2.49/hr for H100 on-demand
- Vast.ai: ~$1.50-2/hr, more variable quality
- RunPod: ~$2/hr, solid

Budget $100-200 for the entire sprint on rented H100 time if Colab falls short. It's not a lot.

---

## Environment variables (none yet, but here's where they'll go)

No env vars needed for Phase 1. As we grow we'll document any new ones here. Keep `.env` files out of git.

---

## Troubleshooting quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `cargo build` fails on Mac with "cannot find `cxx`" | You accidentally uncommented `thaw-cuda-sys` in workspace Cargo.toml before it was ready | Re-comment it |
| `nvcc --version` fails on Laptop | CUDA not on PATH | Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin` to PATH |
| Colab notebook can't find Rust after restart | `cargo` not on PATH in new shell | Re-run cell 2 (install Rust) |
| Tests hang forever | You added a `std::thread::sleep` or an infinite loop by accident | `cargo test -- --nocapture` to see output; kill and debug |
| `cargo test` passes on Mac but fails on Colab | Filesystem differences (case-sensitivity, permissions) | Always use lowercase filenames; check with `ls -la` |
