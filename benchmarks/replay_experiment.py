"""Flight-recorder replay experiments E1-E3.

Pre-registered design: private/plans/2026-06-12_flight-recorder-experiment-plan.md.

Hypothesis under test: the per-step kernel-facing batch SHAPE (recorded
by thaw_vllm.recorder, ~1 integer pair per request per step) plus static
engine config is sufficient to retroactively reproduce a past generation
bit-exactly — neighbor token CONTENT is irrelevant (Cankaya 2606.00279).

Stages (each emits PASS/FAIL + a JSON receipt):

  E1  neighbor-content irrelevance: target request R co-batched with
      junk neighbors A, then junk neighbors B (same lengths, different
      content). Control: recorded shape traces must match. Result: R's
      per-token logprobs must be bit-identical across runs.
      Kill criterion K1: content leaks into R -> kilobyte certificates
      are impossible; project re-scopes.

  E2  record + dummy-padded replay: R generated under junk load; the
      replay reconstructs the load FROM THE TRACE ALONE (dummy_spec:
      prompt_len + num_sampled per neighbor, fresh junk content),
      re-runs, and compares R bit-for-bit. This is the certificate
      claim: the trace suffices to recreate the shapes.

  E3  closed-loop sampled replay: same as E2 but R samples at
      temperature with a fixed seed. Replay must re-emit R's exact
      token sequence end-to-end, not just matching teacher-forced
      logprobs.

Run on a pod (single GPU, V1 in-process):

  VLLM_ENABLE_V1_MULTIPROCESSING=0 python benchmarks/replay_experiment.py \
      --stage all --out results/replay_e123.json

Notes:
  - enforce_eager=True throughout: CUDA-graph padding quantizes batch
    shapes to capture sizes; eager mode keeps the recorded shape equal
    to the kernel-facing shape. Graph-mode support = later work, and
    the receipt records the setting.
  - All requests in a stage are submitted in one llm.generate() call;
    staggered arrival is a pod-side extension (E4 territory).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import time

# Must be set before vLLM import anywhere in the process.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

TOPK = 5
JUNK_TOKEN_LOW = 1000
JUNK_TOKEN_HIGH = 5000


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def f64_bits(x) -> str:
    """Exact bit pattern of a float for unambiguous receipts."""
    if x is None:
        return "none"
    return struct.pack("<d", float(x)).hex()


def token_logprob_rows(request_output, max_topk: int = TOPK):
    """[(token_id, logprob, [(tid, lp), ...top-k]), ...] for output 0."""
    from thaw_vllm.rewind import extract_token_logprobs

    comp = request_output.outputs[0]
    rows = []
    for t in extract_token_logprobs(comp, max_topk=max_topk):
        topk = [(c["token_id"], c["logprob"]) for c in (t.get("topk") or [])]
        rows.append((t["token_id"], t["logprob"], topk))
    return rows


def compare_target(out_a, out_b) -> dict:
    """Bitwise comparison of one request's outputs across two runs."""
    rows_a = token_logprob_rows(out_a)
    rows_b = token_logprob_rows(out_b)
    ids_a = [r[0] for r in rows_a]
    ids_b = [r[0] for r in rows_b]
    result: dict = {
        "n_tokens_a": len(rows_a),
        "n_tokens_b": len(rows_b),
        "token_ids_identical": ids_a == ids_b,
    }
    n = min(len(rows_a), len(rows_b))
    mismatches = []
    max_abs_delta = 0.0
    for i in range(n):
        lp_a, lp_b = rows_a[i][1], rows_b[i][1]
        if lp_a is None or lp_b is None:
            continue
        if lp_a != lp_b:
            max_abs_delta = max(max_abs_delta, abs(lp_a - lp_b))
            if len(mismatches) < 10:
                mismatches.append({
                    "pos": i,
                    "token_a": rows_a[i][0],
                    "token_b": rows_b[i][0],
                    "lp_a_bits": f64_bits(lp_a),
                    "lp_b_bits": f64_bits(lp_b),
                    "abs_delta": abs(lp_a - lp_b),
                })
    result["logprobs_bit_identical"] = (
        result["token_ids_identical"] and not mismatches
        and len(rows_a) == len(rows_b)
    )
    result["n_logprob_mismatches"] = len(mismatches)
    result["max_abs_logprob_delta"] = max_abs_delta
    result["first_mismatches"] = mismatches
    return result


# ---------------------------------------------------------------------------
# Load construction
# ---------------------------------------------------------------------------


def junk_prompt(rng: random.Random, length: int) -> list[int]:
    return [rng.randrange(JUNK_TOKEN_LOW, JUNK_TOKEN_HIGH) for _ in range(length)]


def neighbor_lengths(rng: random.Random, n: int, args) -> list[tuple[int, int]]:
    """(prompt_len, decode_len) per neighbor — drawn once per stage so
    runs A and B share lengths while content differs."""
    return [
        (
            rng.randrange(args.junk_prompt_min, args.junk_prompt_max + 1),
            rng.randrange(args.junk_decode_min, args.junk_decode_max + 1),
        )
        for _ in range(n)
    ]


def build_neighbors(content_rng: random.Random, spec: list[tuple[int, int]],
                    sampling_params_cls):
    """Junk requests from (prompt_len, decode_len) specs. ignore_eos +
    fixed max_tokens pins every neighbor's decode length so the shape
    trace is reproducible; per-request seeds pin their sampling."""
    prompts, sps = [], []
    for i, (plen, dlen) in enumerate(spec):
        prompts.append({"prompt_token_ids": junk_prompt(content_rng, plen)})
        sps.append(sampling_params_cls(
            temperature=1.0,
            seed=10_000 + i,
            max_tokens=dlen,
            ignore_eos=True,
        ))
    return prompts, sps


def reset_cache(llm) -> None:
    ok = llm.reset_prefix_cache()
    if ok is False:
        raise RuntimeError(
            "reset_prefix_cache() returned False — a replay pass would "
            "silently cache-hit the original run's blocks"
        )


def run_recorded(llm, prompts, sps):
    """One generate call under a fresh Recorder. Returns (outputs, trace)."""
    from thaw_vllm.recorder import Recorder

    rec = Recorder(llm)
    with rec:
        outs = llm.generate(prompts, sps)
    return outs, rec.trace


def trace_control(trace_a, trace_b, target_a: str, target_b: str) -> dict:
    """Shape-trace equality check — the experimental control. The target
    request is tagged so its slot in the shape is pinned across runs."""
    from thaw_vllm.recorder import shape_signature

    sig_a = shape_signature(trace_a, target=target_a)
    sig_b = shape_signature(trace_b, target=target_b)
    equal = sig_a == sig_b
    detail: dict = {
        "traces_shape_identical": equal,
        "n_steps_a": len(sig_a),
        "n_steps_b": len(sig_b),
    }
    if not equal:
        diverge_at = next(
            (i for i, (a, b) in enumerate(zip(sig_a, sig_b)) if a != b),
            min(len(sig_a), len(sig_b)),
        )
        detail["first_divergent_step"] = diverge_at
    return detail


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def target_request(args, tok, sampled: bool):
    from vllm import SamplingParams

    msgs = [{"role": "user", "content": args.target_prompt}]
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
    if sampled:
        sp = SamplingParams(temperature=0.8, top_p=0.95, seed=args.target_seed,
                            max_tokens=args.target_tokens, logprobs=TOPK)
    else:
        sp = SamplingParams(temperature=0.0, max_tokens=args.target_tokens,
                            logprobs=TOPK)
    return prompt, sp


def stage_e1(llm, tok, args) -> dict:
    """Neighbor-content irrelevance: same shapes, different junk."""
    from vllm import SamplingParams

    r_prompt, r_sp = target_request(args, tok, sampled=False)
    lengths = neighbor_lengths(random.Random(args.length_seed),
                               args.n_neighbors, args)

    runs = {}
    for tag, content_seed in (("a", 1), ("b", 2)):
        reset_cache(llm)
        n_prompts, n_sps = build_neighbors(random.Random(content_seed),
                                           lengths, SamplingParams)
        outs, trace = run_recorded(llm, [r_prompt] + n_prompts,
                                   [r_sp] + n_sps)
        runs[tag] = {"target_out": outs[0],
                     "target_id": outs[0].request_id,
                     "trace": trace}

    control = trace_control(runs["a"]["trace"], runs["b"]["trace"],
                            runs["a"]["target_id"], runs["b"]["target_id"])
    cmp = compare_target(runs["a"]["target_out"], runs["b"]["target_out"])
    return {
        "stage": "E1",
        "control": control,
        "comparison": cmp,
        "pass": bool(control["traces_shape_identical"]
                     and cmp["logprobs_bit_identical"]),
        "kill_criterion_K1_fired": bool(control["traces_shape_identical"]
                                        and not cmp["logprobs_bit_identical"]),
    }


def _record_then_replay(llm, tok, args, sampled: bool) -> dict:
    """Shared body of E2/E3: record under junk load, reconstruct the
    load from the trace alone, replay, compare the target bitwise."""
    from vllm import SamplingParams

    r_prompt, r_sp = target_request(args, tok, sampled=sampled)
    lengths = neighbor_lengths(random.Random(args.length_seed + 7),
                               args.n_neighbors, args)

    # Original run.
    reset_cache(llm)
    n_prompts, n_sps = build_neighbors(random.Random(11), lengths,
                                       SamplingParams)
    outs_orig, trace_orig = run_recorded(llm, [r_prompt] + n_prompts,
                                         [r_sp] + n_sps)
    target_id = outs_orig[0].request_id
    cert_bytes = trace_orig.certificate_bytes()

    # Replay: the ONLY inputs are the target request and the trace.
    spec = trace_orig.dummy_spec(exclude={target_id})
    replay_spec = [(s["prompt_len"], s["num_sampled"]) for s in spec]
    reset_cache(llm)
    d_prompts, d_sps = build_neighbors(random.Random(99), replay_spec,
                                       SamplingParams)
    outs_replay, trace_replay = run_recorded(llm, [r_prompt] + d_prompts,
                                             [r_sp] + d_sps)

    control = trace_control(trace_orig, trace_replay,
                            target_id, outs_replay[0].request_id)
    cmp = compare_target(outs_orig[0], outs_replay[0])
    return {
        "control": control,
        "comparison": cmp,
        "certificate_bytes": cert_bytes,
        "n_trace_steps": len(trace_orig.steps),
        "derived_dummy_spec": spec,
        "pass": bool(control["traces_shape_identical"]
                     and cmp["logprobs_bit_identical"]),
    }


def stage_e2(llm, tok, args) -> dict:
    out = _record_then_replay(llm, tok, args, sampled=False)
    out["stage"] = "E2"
    return out


def stage_e3(llm, tok, args) -> dict:
    out = _record_then_replay(llm, tok, args, sampled=True)
    out["stage"] = "E3"
    # E3's pass additionally requires the sampled token sequence itself
    # to be re-emitted — already implied by token_ids_identical, but
    # spelled out because it is the stage's headline claim.
    out["sampled_sequence_reemitted"] = out["comparison"]["token_ids_identical"]
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


STAGES = {"e1": stage_e1, "e2": stage_e2, "e3": stage_e3}


def run(args) -> None:
    from vllm import LLM

    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        enable_prefix_caching=True,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=0,
    )
    tok = llm.get_tokenizer()

    wanted = list(STAGES) if args.stage == "all" else [args.stage]
    results = []
    for name in wanted:
        t_stage = time.time()
        res = STAGES[name](llm, tok, args)
        res["seconds"] = round(time.time() - t_stage, 1)
        results.append(res)
        print(f"[replay] {res['stage']}: "
              f"{'PASS' if res['pass'] else 'FAIL'} "
              f"({res['seconds']}s)", flush=True)

    payload = {
        "experiment": "flight-recorder-replay-e1-e3",
        "prereg": "private/plans/2026-06-12_flight-recorder-experiment-plan.md",
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "enforce_eager": True,
            "n_neighbors": args.n_neighbors,
            "target_tokens": args.target_tokens,
            "target_seed": args.target_seed,
            "junk_prompt_range": [args.junk_prompt_min, args.junk_prompt_max],
            "junk_decode_range": [args.junk_decode_min, args.junk_decode_max],
            "max_model_len": args.max_model_len,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
        "stages": results,
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, default=str)
    os.replace(tmp, args.out)
    print(f"[replay] receipt -> {args.out}", flush=True)

    if not all(r["pass"] for r in results):
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    p.add_argument("--n-neighbors", type=int, default=7)
    p.add_argument("--target-prompt",
                   default="Explain why floating-point addition is not "
                           "associative, with a concrete example.")
    p.add_argument("--target-tokens", type=int, default=200)
    p.add_argument("--target-seed", type=int, default=1234)
    p.add_argument("--length-seed", type=int, default=42)
    p.add_argument("--junk-prompt-min", type=int, default=64)
    p.add_argument("--junk-prompt-max", type=int, default=768)
    p.add_argument("--junk-decode-min", type=int, default=16)
    p.add_argument("--junk-decode-max", type=int, default=160)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--out", default="results/replay_e123.json")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
