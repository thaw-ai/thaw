# Contributing to thaw

thaw is a small team moving fast. These are the rails that keep PRs flowing
without regressions or hours lost to footguns we've already paid for once.

## The loop

1. Branch off `main` (`git checkout -b area/short-description`).
2. Make the change. Keep the diff scoped — one concern per PR.
3. Run the checks below locally.
4. Open a PR. CI runs `lint`, `cargo test`, and `pytest`. Fill in the template.
5. Get a review. Address blocking + high items; land it.

Never push to `main` directly, and never tag a release without a GPU
validation pass — see [Release](#release).

## Checks before you push

```bash
# Project footgun linter (whole tree, fast, no deps)
python tools/thaw_lint.py

# Real-bug lint on the Python you touched
ruff check --select F,E9 <your changed .py files>

# Rust: format + lint + tests
cargo fmt --all
cargo clippy --workspace --exclude thaw-py --all-targets -- -D warnings
cargo test --workspace --exclude thaw-py

# Python tests (torch/vllm/sglang are mocked — no GPU needed)
pytest tests/ -q
```

### What CI gates

| Gate | Scope | Why scoped this way |
|------|-------|---------------------|
| `cargo clippy -D warnings` | whole workspace (minus `thaw-py`) | clean today — a regression guard |
| `python tools/thaw_lint.py` | whole tree | every ERROR rule is currently green |
| `ruff check --select F,E9` | **changed Python only** | the legacy tree has known debt; see the ratchet below |

**The ruff ratchet.** We do not gate the whole tree on ruff yet — there are
~36 pre-existing real-bug findings (mostly unused imports) that a one-shot
cleanup would land right on top of in-flight PRs. So ruff runs only on the
files your PR changes: new and touched code must be clean, legacy debt does
not block you. Once the open PRs land, a dedicated `ruff --fix` sweep will
clear the backlog and the gate widens to the whole tree. If you're already in
a legacy file, fixing its findings while you're there is welcome.

`cargo fmt --check` is **not** gated yet (a full `cargo fmt` touches ~16 files
and would conflict with open Rust PRs). Run `cargo fmt --all` before pushing
anyway; we'll turn the gate on once the tree is normalized.

## thaw-lint — the project-specific rules

`tools/thaw_lint.py` encodes the mistakes that have actually cost us time.
`ERROR` fails CI; `WARN` just surfaces. Run `python tools/thaw_lint.py --list`
to see them all. Current ERROR rules:

- **`import thaw_native`** → the `thaw-native` wheel installs the module
  `thaw`. Write `import thaw`. (This one cost real fresh-pod debugging hours.)
- **Personal `@icloud.com` email** in tracked files → the public contact is
  `nils@thaw.sh`.
- **`matteso1/thaw` URLs** → the repo lives at `thaw-ai/thaw`; the old one is
  archived and 404s clone scripts.
- **Leftover `breakpoint()` / `pdb.set_trace()`** in shipped Python.
- **Retracted speedup multipliers** (`17.2x`, `12.6x`, `9.7x`, `5.9x`) in the
  README or `site/` — see [receipt hygiene](#receipt--benchmark-hygiene).
- **Hard-setting `VLLM_ALLOW_INSECURE_SERIALIZATION`** → use
  `os.environ.setdefault` so a user can override a security-sensitive flag.

Sure a line is a false positive? Add `# thaw-lint: allow` to it. New footgun
worth guarding? Add a `Rule` to `tools/thaw_lint.py` with a test — keep ERROR
rules ones the tree is currently clean on, so they're regression guards, not a
backlog.

## Receipt & benchmark hygiene

Performance is the pitch, so the bar on numbers is high.

- **Only claim re-validated numbers.** Default to the most conservative
  defensible figure, not the most flattering one.
- **A perf claim ships with its receipt** — a JSON under `site/receipts/` from
  the run that produced it, in the same PR. Prose-only numbers rot.
- **Say what the benchmark actually measured.** If it injects synthetic
  latency, it measured latency-hiding, not bandwidth — don't let the README
  round that up to a throughput claim.
- **Scrub local paths** (`C:\Users\...`, `/home/you/...`) out of committed
  receipts.
- Retired numbers live in `docs/ARCHITECTURE.md` so they don't resurface.

## GPU vs. no-GPU

A lot of thaw runs without a GPU — the whole inspect/diff/log/rewind surface,
the format readers, the Rust core against `MockCuda`. Those are unit-tested in
CI. The GPU paths (real freeze/restore/fork, TP, KV cache) are validated on a
pod per release; `tests/gpu/` and `benchmarks/` hold those. Don't put a
GPU-requiring test in the default `pytest` path — mock the engine
(`tests/conftest.py` shows how) or gate it under `tests/gpu/`.

## Ownership boundary (format vs. transport)

To keep two people out of each other's way, work splits at the file format
line:

- **Above the line — format, semantics, the verb surface.** The `.thaw` /
  `.thawkv` schema, `handle.json`, cross-engine restore, `verify`, lineage,
  the fork API contract. Changes here define *what the bytes mean*.
- **Below the line — byte transport & throughput.** Restore DMA, freeze
  pipelining, multi-NVMe/TP write paths, parallel cloud fetch. Changes here
  move *the same bytes faster*.

`python/thaw_common/cloud.py` is the one file both sides touch — partition it:
manifest parse/resolve above the line, the transfer executor below it, with a
typed boundary between. When a change spans the line (e.g. shard-at-freeze),
freeze the on-disk schema + golden fixtures **first**, then build transport
against bytes-on-disk rather than against in-flight code.

## Release

The `thaw-native` version string lives in **two** independent places that must
match: `Cargo.toml` `[workspace.package]` and **`crates/thaw-py/pyproject.toml`**
(the authoritative file maturin reads — the other crate `Cargo.toml`s inherit
via `version.workspace = true`). Desyncing them has failed a release before.
Bump both, validate on a GPU pod, then tag.
