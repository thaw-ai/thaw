# thaw validation runbook

End-to-end procedure for producing defensible benchmark numbers for thaw.
Designed to be runnable by someone outside the team (a YC partner, a
skeptical investor, a reviewer on a vLLM PR). Every number in public
marketing copy should come from an artifact produced by this runbook.

---

## Goal

Generate, on real hardware, with a clean protocol, for multiple model
sizes, a report containing:

1. **Speedup** (normal cold start ÷ thaw cold NVMe) — the headline.
2. **Restore throughput** (GB/s) — the technical claim.
3. **Hot-swap steady-state** (seconds / GB/s after one-time pin).
4. **Bit-identity** — SHA256 of every parameter, pre-freeze vs post-restore.
5. **Variance** — median + CoV over N≥3 runs. CoV > 10% flags the number.

Each claim in README.md / site should map to a specific row in
the aggregate JSON. Nothing is published that hasn't been run ≥3 times
on the stated hardware.

---

## Hardware matrix

| Purpose | Instance | ~Cost/hr | Why |
|---|---|---|---|
| Multi-size TP=1 sweep | RunPod 1× H100 SXM 80GB | ~$2.69 | Headline path. Fits 8B / 7B / 14B comfortably. |
| **TP=2 historic re-measure** | **RunPod 2× A100 80GB** | **~$3.38** | **Same arch as the original 17× Llama-70B claim — apples-to-apples.** |
| TP=2 best-case | RunPod 2× H100 SXM 80GB | ~$5.38 | New number, not a re-measurement. Run after A100 pass if budget. |
| Consumer corroboration | RunPod RTX A6000 | ~$0.49 | Shows consumer-GPU path works. |
| CI smoke | RunPod RTX 3090 | ~$0.22 | Fastest cheapest sanity check. |

**Storage: 1 TB container** for TP=2 on 70B. Llama-3-70B fp16 is ~140 GB
downloaded + ~140 GB snapshot + rank-suffixed siblings + HF cache +
weight-hash snapshots. Going under 500 GB risks disk-full mid-sprint.

**Total budget** for full TP=1 + TP=2 revalidation: ~$40–55 across ~8 hrs
of pod time. Cheaper to re-run than to publish a flaky number.

### TP=2 notes

- `--tp 2` plumbed through `vllm_demo.py`, `run_validation.py`, `weight_hash.py`.
  Under TP>1:
  - Freeze/restore dispatch via `collective_rpc` → writes `weights.thaw` +
    `weights.rank1.thaw` (and higher ranks if applicable) as siblings.
  - Page-cache eviction hits every sibling file, not just the base.
  - Phase 4 (pre-staged RAM) is **skipped** — `restore_model_from_ram` has no
    TP dispatcher wired yet. Documented gap, recorded as `skipped: true` in
    JSON, not a failure.
  - bit-identity hashing uses `collective_rpc` so each rank's shard is hashed
    separately (keyed `rank0:param`, `rank1:param` to prevent collisions).
- Bump `--gpu-mem-util` to `0.85` for 70B TP=2 — default `0.25` won't fit.
- Set `NCCL_P2P_DISABLE=1` on A100s if you hit NCCL init hangs
  (documented vLLM workaround, not a thaw issue).

---

## Pre-flight

Gated Hugging Face models (Llama, Mistral variants) require auth:

```bash
pip install huggingface_hub
huggingface-cli login   # paste your HF token
```

Verify the pod has nvidia-smi, a working CUDA 12+ driver, and fast
NVMe (`/tmp` should be on local SSD, not network storage — check with
`df -T /tmp`).

---

## Pod setup (copy-paste as ONE block)

Uses the vLLM image so we don't fight torch installs. Installs thaw-native
from PyPI (manylinux wheel, no Rust toolchain needed).

```bash
set -euo pipefail
pip install --upgrade pip
pip install thaw-native thaw-vllm huggingface_hub
# vmtouch gives us HARD page cache eviction; fadvise is advisory and the
# kernel can ignore it on a fresh pod with abundant RAM, which silently
# invalidates every "cold restore" number.
apt-get update && apt-get install -y vmtouch || echo "vmtouch install failed — see notes"
git clone https://github.com/thaw-ai/thaw.git ~/thaw
cd ~/thaw
python -c "import thaw, thaw_vllm; print('thaw', thaw.__version__); print('thaw_vllm OK')"
```

If `vmtouch` won't install (ancient distro), cold-cache Phase 3 numbers
are advisory only — report the fallback message from Phase 3 output and
treat warm-cache + pre-staged-RAM as the reliable figures.

---

## Step 0 — TP=2 historic re-measurement (on 2× A100 80GB)

The existing site/README cites "17.2× on Llama-3-70B 2× A100 TP=2" from a
single measurement. Nils's standing rule (`feedback_benchmark_claims`) is
to only publish numbers re-measured across configs. This step replaces
that bare 17× with a median + CoV over N≥3 runs.

```bash
# on 2× A100 80GB pod:
export NCCL_P2P_DISABLE=1   # avoid known vLLM NCCL hangs on A100
cd ~/thaw
OUT=/tmp/tp2_$(date +%Y%m%d_%H%M)

python benchmarks/run_validation.py \
  --models meta-llama/Meta-Llama-3-70B \
  --runs 3 --tp 2 --gpu-mem-util 0.85 \
  --snapshot /tmp/llama70b.thaw \
  --out-dir $OUT \
  --cov-threshold 0.10

python benchmarks/weight_hash.py \
  --model meta-llama/Meta-Llama-3-70B \
  --tp 2 --gpu-mem-util 0.85 \
  --snapshot /tmp/llama70b_wh.thaw \
  --json-out $OUT/weight_hash_70b_tp2.json

cat $OUT/report.md
```

**Acceptance:**
- `thaw_cold_nvme.output_match_ref == True` on every run (greedy-match
  across both ranks).
- `weight_hash.diff.num_mismatches == 0` on rank 0 AND rank 1.
- `summary.speedup_cold_nvme` median ≥ 10× (revise copy to this number,
  not the one-off 17×).
- CoV ≤ 10% on restore throughput.

If any of these fail, **do not publish a TP=2 70B number at all** until
investigated. The scaffolding records the failure in JSON; attach that
file when filing an issue.

Also smaller-model TP=2 corroboration (cheaper, faster):

```bash
python benchmarks/run_validation.py \
  --models meta-llama/Meta-Llama-3-8B Qwen/Qwen2.5-14B \
  --runs 3 --tp 2 --gpu-mem-util 0.50 \
  --snapshot /tmp/thaw_tp2.thaw \
  --out-dir $OUT/tp2_small
```

---

## Step 1 — Multi-size cold-start validation (TP=1)

For each of 8B / 7B-mistral / 14B-qwen, runs 3× with fresh page cache:

```bash
cd ~/thaw
mkdir -p /tmp/val_$(date +%Y%m%d_%H%M)
OUT=/tmp/val_$(date +%Y%m%d_%H%M)

python benchmarks/run_validation.py \
  --models meta-llama/Meta-Llama-3-8B mistralai/Mistral-7B-v0.3 Qwen/Qwen2.5-14B \
  --runs 3 \
  --snapshot /tmp/thaw_val.thaw \
  --out-dir $OUT \
  --cov-threshold 0.10

# Outputs:
#   $OUT/aggregate.json   — machine-readable, schema_version=1
#   $OUT/report.md        — human table with median / min / max / CoV
#   $OUT/runs/*.json      — per-run reports (one per model per iteration)
#   $OUT/runs/*.log       — stdout/stderr of each run for postmortem
```

**Acceptance criteria (H100 SXM 80GB, float16, `--tp 1`):**

| Model | normal cold | thaw cold NVMe | restore GB/s | speedup | bit-identical |
|---|---|---|---|---|---|
| Llama-3-8B | 15–25 s | 2.5–4.5 s | 6–11 | 4–8× | PASS |
| Mistral-7B-v0.3 | 12–22 s | 2.0–4.0 s | 6–11 | 4–8× | PASS |
| Qwen2.5-14B | 25–40 s | 4.5–7.0 s | 6–11 | 5–9× | PASS |

Any metric with CoV > 10% is marked in `report.md` with a `:warning:`.
**Do not publish a flagged number.** Bump `--runs` to 5 and re-run that
model only; if still flagged, investigate before publishing.

---

## Step 2 — Bit-identity proof (per model)

Greedy-output match can pass with tiny drift. To prove bit-identity,
hash every parameter before freeze and after restore:

```bash
python benchmarks/weight_hash.py \
  --model meta-llama/Meta-Llama-3-8B \
  --snapshot /tmp/wh_llama.thaw \
  --json-out /tmp/wh_llama.json

python benchmarks/weight_hash.py \
  --model mistralai/Mistral-7B-v0.3 \
  --snapshot /tmp/wh_mistral.thaw \
  --json-out /tmp/wh_mistral.json

# ... one per model in the matrix
```

**Acceptance: every run exits 0 and `diff.num_mismatches == 0`.**

A single mismatched tensor means the freeze/restore pipeline is
corrupting weights on that layout — stop and investigate before
publishing any speedup number for that model.

---

## Step 3 — Hot-swap steady-state

First freeze N same-architecture snapshots (fine-tunes of the same base,
or the same model multiple times from different seeds). Then validate
steady-state swap performance:

```bash
# Produce a few same-arch snapshots to cycle through
thaw freeze --model meta-llama/Meta-Llama-3-8B --output /tmp/a.thaw
thaw freeze --model meta-llama/Meta-Llama-3-8B --output /tmp/b.thaw
thaw freeze --model meta-llama/Meta-Llama-3-8B --output /tmp/c.thaw

python benchmarks/validate_hotswap.py \
  --base-model meta-llama/Meta-Llama-3-8B \
  --models a:/tmp/a.thaw b:/tmp/b.thaw c:/tmp/c.thaw \
  --iterations 6 \
  --json-out /tmp/hotswap_8b.json
```

**Acceptance (H100 SXM, 8B, one-slot):**

- warmup (load 0) typically 5–10 s (one-time cudaHostRegister pin).
- steady-state (loads 1..N-1) median ≤ 0.5 s per swap.
- steady-state median throughput ≥ 30 GB/s.
- CoV on swap time ≤ 10% (flagged in report otherwise).

If the warmup time approaches the steady-state time, the pin isn't being
amortized — check that `VLLM_ENABLE_V1_MULTIPROCESSING=0` is set and the
slot is being reused (look for `backend=thaw_native_slot_pinned` in the
per-iteration output).

---

## Step 4 — What to publish

Only numbers that survive this pipeline end up in public copy:

1. Take the **median** (not mean) from aggregate.json.
2. Round to two significant figures — "9.7×" not "9.74235×".
3. Pair the number with the hardware it was measured on, the model, and
   the snapshot size in the same sentence. "9.7× on Llama-3-8B, H100 SXM
   80GB" — never a bare "9.7× faster."
4. Preserve `$OUT/` under `benchmarks/archive/YYYY-MM-DD_host/` so the
   artifact is reproducible when challenged.

**Threshold for "publishable":** median over ≥3 runs, CoV ≤ 10%,
bit-identity PASS, hardware fingerprint captured in the JSON.

---

## Failure-mode triage

| Symptom | Likely cause | Fix |
|---|---|---|
| "fadvise hint sent ... advisory" on every Phase 3 | No vmtouch | Install vmtouch; otherwise Phase 3 is warm-cache reading. |
| Restore GB/s < 3 on NVMe | Network storage or slow SSD | `df -T /tmp`; re-rent pod with local NVMe. |
| bit-identity FAIL on dtype={bfloat16,...} | Tensor layout/stride edge | Open an issue with the JSON diff attached; do NOT publish that model's number. |
| CoV > 10% on one metric | Neighboring workload on shared GPU / thermal | Re-run on a dedicated instance; flag the number until resolved. |
| Hot-swap warmup > 30s for 8B | cudaHostRegister hitting swap | `free -h` → needs ≥ 2× snapshot-size free RAM. |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` error | vLLM 0.19 msgspec IPC | `export VLLM_ALLOW_INSECURE_SERIALIZATION=1` — thaw_vllm does this on import, but a prior `import vllm` can lock it. |
| "gated model" HTTP 401 | Missing HF auth | `huggingface-cli login` on the pod. |

---

## One-line "I'm an investor, prove it" script

This is the demo you leave running on a shared screen. Runs the minimum
to produce an aggregate report + bit-identity proof in ~25 minutes on
H100 SXM.

```bash
cd ~/thaw
OUT=/tmp/prove_$(date +%Y%m%d_%H%M) && mkdir -p $OUT
python benchmarks/run_validation.py \
  --models meta-llama/Meta-Llama-3-8B \
  --runs 3 --out-dir $OUT --snapshot /tmp/prove.thaw && \
python benchmarks/weight_hash.py \
  --model meta-llama/Meta-Llama-3-8B \
  --snapshot /tmp/wh.thaw --json-out $OUT/weight_hash.json && \
echo "---" && cat $OUT/report.md
```

If any step exits non-zero or the report has a `:warning:` flag, the
demo has not passed — say so, don't paper over it.
