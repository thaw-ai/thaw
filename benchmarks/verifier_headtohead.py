#!/usr/bin/env python3
"""verifier_headtohead.py — re-feed vs true-KV replay as an inference verifier.

Replay-based verifiers (DiFR arXiv:2511.20621, TOPLOC arXiv:2501.16007, SVIP,
VeriLLM) check an LLM output by recomputing its per-token signature and comparing
to a commitment. They reconstruct state by RE-FEEDING the transcript as one
prefill pass. The committed signature, though, is the DECODE-TIME distribution —
produced one token at a time. Re-feed (bulk prefill) does not reproduce
decode-time exactly, so the verifier's reconstruction drifts from the commitment
at exactly the low-margin tokens that matter.

A thaw verifier instead restores the true decode-time KV from the `.thawkv`
receipt and replays from it, reproducing the committed distribution bit-identically
(validated separately by thaw's bit-identical receipts) — so its reconstruction
error is ~0 by construction. This script measures the gap a re-feed verifier pays
against that ground truth, end to end, on GREEDY generations (the regime verifiers
target), across models.

Metrics (per model; greedy decode + a single re-feed prefill):
  drift                 |logit_refeed - logit_decode| per generated token.
  false_reject (logit)  a token is rejected if its reconstructed logit is > tau
                        from the commitment. Re-feed false-rejects genuine output;
                        true-KV: 0.
  argmax verifier       accept token j iff the re-derived argmax == the claimed
                        token. Two error directions, reported at LOW-MARGIN tokens
                        with Wilson 95% intervals:
                          false_reject  re-feed argmax != decode argmax (a genuine
                                        token wrongly rejected)
                          forgery_accept re-feed argmax == the decode RUNNER-UP (a
                                        single-token swap to rank-2 would be the
                                        re-feed argmax and pass) -- the security hole
  gap_masking           drift >= the top1-top2 gap: the swap is smaller than the
                        verifier's own re-feed noise. true-KV: 0.

Honest scope: the true-KV arm is sound by construction (decode-time distribution =
the commitment; `.thawkv` restores it bit-identically per separate receipts); this
run measures the re-feed verifier against that ground truth. A head-to-head against
a deployed DiFR/TOPLOC codebase is a further step.

Usage:
  python benchmarks/verifier_headtohead.py \
      --models Qwen/Qwen2.5-7B-Instruct,microsoft/Phi-3.5-mini-instruct \
      --n-prompts 150 --json-out site/receipts/<date>_verifier_headtohead_v2.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import List, Optional, Tuple

# Fallback reasoning prompts if the gsm8k dataset is unavailable.
_FALLBACK_PROMPTS: List[str] = [
    "A store had 124 apples. It sold 47 then 38. How many are left?",
    "Tom reads 23 pages a day for 9 days, then 15 a day for 4 days. Total pages?",
    "A train goes 60 mph for 2.5 h then 45 mph for 1.5 h. How far?",
    "Sarah has $250, buys 3 books at $19 and a $42 bag. How much remains?",
    "If 7 workers build a wall in 12 days, how long for 14 workers?",
    "A recipe needs 3 eggs for 12 cookies. How many eggs for 30 cookies?",
    "A tank holds 480 L and drains 16 L/min. How long to empty?",
    "A shirt costs $40, discounted 35%. Sale price?",
    "Ben is 3x his sister's age; in 6 years he'll be 2x. How old is he now?",
    "A ladder's base is 6 ft from a wall and reaches 8 ft up. Length?",
]

INSTRUCTION = ("\n\nReason step by step. End with one line exactly:\nAnswer: <number>")


def load_prompts(n: int) -> List[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [ds[i]["question"] + INSTRUCTION for i in range(min(n, len(ds)))]
    except Exception as e:  # noqa: BLE001 — dataset offline; fall back
        print(f"[verifier] gsm8k unavailable ({type(e).__name__}); using fallback prompts")
        base = _FALLBACK_PROMPTS
        return [(base[i % len(base)] + INSTRUCTION) for i in range(n)]


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score interval for a binomial rate (point, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (round(p, 4), round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def _top1_id(step: Optional[dict]) -> Optional[int]:
    if not step:
        return None
    return max(step.items(), key=lambda kv: kv[1].logprob)[0]


def _rank2_id(step: Optional[dict], top1: int) -> Optional[int]:
    if not step:
        return None
    others = [(k, v.logprob) for k, v in step.items() if k != top1]
    return max(others, key=lambda kv: kv[1])[0] if others else None


def _lp(step: Optional[dict], tid: int) -> Optional[float]:
    if not step:
        return None
    e = step.get(tid)
    return float(e.logprob) if e is not None else None


def run_model(model: str, prompts: List[str], max_tokens: int, margin_nats: float, seed: int) -> dict:
    from vllm import LLM, SamplingParams

    # Prefix caching OFF so re-feed is a clean fresh prefill (the verifier
    # condition), never a silent reuse of the live decode-time KV.
    llm = LLM(model=model, dtype="bfloat16", enable_prefix_caching=False,
              gpu_memory_utilization=0.9, max_model_len=4096, seed=seed)
    tok = llm.get_tokenizer()
    chats = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True)
             for p in prompts]

    # Greedy generation; capture decode-time top-8 per position.
    gen = llm.generate(chats, SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=8))
    transcripts, metas = [], []
    for g in gen:
        comp = g.outputs[0]
        if not comp.token_ids:
            continue
        transcripts.append({"prompt_token_ids": list(g.prompt_token_ids) + list(comp.token_ids)})
        metas.append((len(g.prompt_token_ids), comp))

    # Re-feed each full transcript as ONE prefill; prompt_logprobs=8 gives the
    # re-derived top-8 (hence argmax) at every position.
    refed = llm.generate(transcripts, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=8))

    drifts: List[float] = []
    am_elig = am_fr = am_fa = 0            # argmax: eligible, false-reject, forgery-accept (low-margin)
    mask_elig = mask_hit = 0              # gap-masking (low-margin)
    for (plen, comp), rf in zip(metas, refed):
        pl = rf.prompt_logprobs
        for j, tid in enumerate(comp.token_ids):
            dstep = comp.logprobs[j] if comp.logprobs and j < len(comp.logprobs) else None
            rstep = pl[plen + j] if pl and plen + j < len(pl) else None
            if not dstep or not rstep:
                continue
            d_top1, d_lp = _top1_id(dstep), _lp(dstep, tid)
            r_lp = _lp(rstep, tid)
            if d_top1 is None or d_lp is None or r_lp is None:
                continue
            drifts.append(abs(r_lp - d_lp))
            d_top2 = _rank2_id(dstep, d_top1)
            gap = (d_lp - _lp(dstep, d_top2)) if d_top2 is not None and _lp(dstep, d_top2) is not None else None
            if gap is None:
                continue
            low = gap < margin_nats
            if low:
                r_top1 = _top1_id(rstep)
                am_elig += 1
                if r_top1 != d_top1:
                    am_fr += 1                 # genuine argmax wrongly rejected
                if r_top1 == d_top2:
                    am_fa += 1                 # runner-up forgery would be accepted
                mask_elig += 1
                if abs(r_lp - d_lp) >= gap:
                    mask_hit += 1

    n = len(drifts)
    taus = [0.01, 0.02, 0.05, 0.1, 0.2]
    return {
        "model": model,
        "n_tokens": n,
        "n_low_margin": am_elig,
        "drift_nats": {
            "median": round(statistics.median(drifts), 4) if n else None,
            "p90": round(sorted(drifts)[int(0.9 * n)], 4) if n else None,
            "mean": round(statistics.fmean(drifts), 4) if n else None,
        },
        "false_reject_logit_sweep": [
            {"tau_nats": t, "refeed": round(sum(1 for x in drifts if x > t) / n, 4) if n else None,
             "true_kv": 0.0} for t in taus
        ],
        "argmax_verifier_low_margin": {
            "false_reject": {"refeed": wilson(am_fr, am_elig), "true_kv": 0.0},
            "forgery_accept": {"refeed": wilson(am_fa, am_elig), "true_kv": 0.0},
            "note": "Wilson 95% CI (point, lo, hi). false_reject = re-feed argmax != "
                    "decode argmax. forgery_accept = re-feed argmax == decode runner-up.",
        },
        "gap_masking_low_margin": {"refeed": wilson(mask_hit, mask_elig), "true_kv": 0.0},
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="Qwen/Qwen2.5-7B-Instruct",
                    help="comma-separated model ids")
    ap.add_argument("--model", help="single model (back-compat; overrides --models)")
    ap.add_argument("--n-prompts", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--margin-nats", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out")
    ns = ap.parse_args(argv)

    models = [ns.model] if ns.model else [m.strip() for m in ns.models.split(",") if m.strip()]
    prompts = load_prompts(ns.n_prompts)
    results = [run_model(m, prompts, ns.max_tokens, ns.margin_nats, ns.seed) for m in models]
    out = {"experiment": "verifier head-to-head (re-feed vs true-KV replay)",
           "n_prompts": len(prompts), "max_tokens": ns.max_tokens,
           "margin_nats": ns.margin_nats, "models": results}

    print(json.dumps(out, indent=2))
    if ns.json_out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(ns.json_out)), exist_ok=True)
        with open(ns.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {ns.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
