<!-- Keep the diff scoped to one concern. See CONTRIBUTING.md. -->

## What & why

<!-- One or two sentences: what changes, and the problem it solves. -->

## How it was verified

<!-- Commands you ran, or the pod/receipt for a GPU/perf change. -->

## Checklist

- [ ] `python tools/thaw_lint.py` is clean
- [ ] `ruff check --select F,E9` passes on the Python I touched
- [ ] `cargo clippy --workspace --exclude thaw-py --all-targets -- -D warnings` is clean (if Rust changed)
- [ ] Tests pass (`pytest tests/ -q` and/or `cargo test`), and new behavior has a test
- [ ] Diff is scoped — no unrelated files
- [ ] **Perf claims ship their receipt** (`site/receipts/*.json` from the run), state what was actually measured, and use a re-validated number — no retracted multipliers, no local paths in receipts
