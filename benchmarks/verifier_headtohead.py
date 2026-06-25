#!/usr/bin/env python3
"""verifier_headtohead.py — re-feed vs true-KV replay as an inference verifier.

Replay-based verifiers (DiFR arXiv:2511.20621, TOPLOC arXiv:2501.16007, SVIP,
VeriLLM) check an LLM output by recomputing its per-token signature and comparing
to a commitment. They reconstruct state by RE-FEEDING the transcript as one prefill
pass. The committed signature, though, is the DECODE-TIME logprob — produced one
token at a time. Re-feed (bulk prefill) does not reproduce decode-time exactly, so
the verifier's reconstruction drifts from the commitment at exactly the low-margin
tokens that matter.

A thaw verifier instead restores the true decode-time KV from the `.thawkv` receipt
and replays from it, reproducing the committed logprob bit-identically (validated
separately by thaw's bit-identical receipts) — so its reconstruction error is ~0 by
construction. This script measures the gap the re-feed verifier pays against that
ground truth, end to end, on fresh generations.

Two results, both from one generate + one re-feed pass (GPU, ~vLLM only):

  A. false-reject on genuine output
     A genuine token should be ACCEPTED. The verifier accepts token j if the
     reconstructed logprob is within tolerance tau of the committed one. Re-feed
     drift d_j = |L_refeed_j - L_decode_j| exceeds tau at some genuine tokens, so a
     re-feed verifier wrongly rejects them. We sweep tau and report the rate. The
     true-KV verifier's rate is 0 (d_j = 0 by construction).

  B. forgery masking (the security edge)
     To keep genuine false-rejects low, a re-feed verifier must set tau wide. But a
     token swap (replace the generated token with the runner-up) only shifts the
     logprob by the top1-top2 gap g_j. Wherever the re-feed drift d_j >= g_j, the
     swap is *smaller than the verifier's own noise*: it cannot be told from a
     genuine token. We report how often re-feed drift masks a single-token swap,
     overall and at low-margin tokens — a window a true-KV verifier (d_j ~ 0) does
     not have.

Honest scope: the true-KV / thaw arm is sound by construction (decode-time logprob =
the commitment, and `.thawkv` restores it bit-identically per separate receipts);
this run measures the re-feed verifier against that ground truth. Fresh generations,
not the arXiv:2606.15621 data.

Usage:
  python benchmarks/verifier_headtohead.py --model Qwen/Qwen2.5-7B-Instruct \
      --json-out site/receipts/<date>_verifier_headtohead.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import List, Optional

# Reasoning prompts (math word problems) — the regime with many low-margin
# decision tokens. Embedded so the pod run needs no dataset download.
PROMPTS: List[str] = [
    "A store had 124 apples. It sold 47 in the morning and 38 in the afternoon. How many are left?",
    "Tom reads 23 pages a day for 9 days, then 15 pages a day for 4 days. How many pages total?",
    "A train travels 60 mph for 2.5 hours, then 45 mph for 1.5 hours. How far did it go?",
    "Sarah has $250. She buys 3 books at $19 each and a bag for $42. How much remains?",
    "A rectangle is 14 cm by 9 cm. What is its perimeter and area?",
    "If 7 workers build a wall in 12 days, how long for 14 workers at the same rate?",
    "A recipe needs 3 eggs for 12 cookies. How many eggs for 30 cookies?",
    "Jack saves $35 a week. How many weeks to save $560?",
    "A tank holds 480 liters and drains at 16 liters per minute. How long to empty?",
    "Mia scored 88, 92, and 79 on three tests. What is her average?",
    "A car uses 8 liters per 100 km. How much fuel for a 350 km trip?",
    "There are 18 rows of 24 chairs. How many chairs are there?",
    "A shirt costs $40 and is discounted 35%. What is the sale price?",
    "Ben is 3 times as old as his sister. In 6 years he will be twice her age. How old is he now?",
    "A pizza is cut into 8 slices. If 3 people each eat 2 slices, what fraction is left?",
    "A pool fills at 25 gallons per minute and holds 1500 gallons. How long to fill?",
    "If a number tripled and increased by 7 equals 34, what is the number?",
    "A farmer has 96 meters of fence for a square pen. What is the side length and area?",
    "Lucy buys 5 notebooks at $3.50 and 2 pens at $1.25. What is the total?",
    "A bus leaves at 9:15 and arrives at 11:40. How long is the trip in minutes?",
    "A factory makes 1,250 widgets a day. How many in a 6-day work week?",
    "Half of a class of 32 students play sports; a quarter of those also play music. How many play both?",
    "A ladder leans so its base is 6 ft from a wall and reaches 8 ft up. How long is the ladder?",
    "An investment of $2,000 grows 5% in a year. What is it worth after one year?",
]


def _lp_of(d: Optional[dict], token_id: int) -> Optional[float]:
    """The logprob a vLLM logprob-dict assigns to a specific token (or None)."""
    if not d:
        return None
    entry = d.get(token_id)
    return float(entry.logprob) if entry is not None else None


def _top1_top2_gap(step: Optional[dict], chosen_id: int, chosen_lp: float) -> Optional[float]:
    """Logprob gap between the chosen token and the best alternative at this step.

    A single-token swap to the runner-up shifts the signature by ~this much; if the
    verifier's re-feed drift exceeds it, the swap hides inside the drift.
    """
    if not step:
        return None
    others = [float(v.logprob) for k, v in step.items() if k != chosen_id]
    if not others:
        return None
    return abs(chosen_lp - max(others))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin-nats", type=float, default=0.3,
                    help="top1-top2 gap below this = a low-margin (decision) token")
    ap.add_argument("--json-out")
    ns = ap.parse_args(argv)

    from vllm import LLM, SamplingParams

    # Prefix caching OFF: re-feed must be a clean fresh prefill (the verifier
    # condition), never a silent reuse of the live decode-time KV.
    llm = LLM(model=ns.model, dtype="bfloat16", enable_prefix_caching=False,
              gpu_memory_utilization=0.92, max_model_len=4096, seed=ns.seed)
    tok = llm.get_tokenizer()

    prompts = [
        tok.apply_chat_template([{"role": "user", "content": p}],
                                tokenize=False, add_generation_prompt=True)
        for p in PROMPTS
    ]

    # Phase 1: generate, capturing the DECODE-TIME (committed) logprobs + top-5.
    gen_sp = SamplingParams(temperature=ns.temp, top_p=ns.top_p,
                            max_tokens=ns.max_tokens, logprobs=5, seed=ns.seed)
    gens = llm.generate(prompts, gen_sp)

    # Phase 2: re-feed each full transcript as ONE prefill; read prompt_logprobs.
    refeed_sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    transcripts, metas = [], []
    for g in gens:
        p_ids = list(g.prompt_token_ids)
        comp = g.outputs[0]
        c_ids = list(comp.token_ids)
        if not c_ids:
            continue
        transcripts.append({"prompt_token_ids": p_ids + c_ids})
        metas.append({"prompt_len": len(p_ids), "comp": comp})
    refeeds = llm.generate(transcripts, refeed_sp)

    # Phase 3: per-token drift, margin, and swap gap.
    drifts: List[float] = []        # |L_refeed - L_decode| per genuine token
    masked: List[bool] = []         # re-feed drift >= top1-top2 gap (swap hidden)
    margins: List[float] = []       # top1-top2 gap per token
    for meta, rf in zip(metas, refeeds):
        comp = meta["comp"]
        plen = meta["prompt_len"]
        c_ids = list(comp.token_ids)
        pl = rf.prompt_logprobs  # list over the full transcript; [i] may be None
        for j, tid in enumerate(c_ids):
            step = comp.logprobs[j] if comp.logprobs and j < len(comp.logprobs) else None
            l_decode = _lp_of(step, tid)
            l_refeed = _lp_of(pl[plen + j] if pl and plen + j < len(pl) else None, tid)
            if l_decode is None or l_refeed is None:
                continue
            d = abs(l_refeed - l_decode)
            gap = _top1_top2_gap(step, tid, l_decode)
            drifts.append(d)
            margins.append(gap if gap is not None else float("inf"))
            if gap is not None:
                masked.append(d >= gap)

    n = len(drifts)
    taus = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    false_reject = [
        {"tau_nats": t,
         "refeed_false_reject_rate": round(sum(1 for d in drifts if d > t) / n, 4),
         "true_kv_rate": 0.0}
        for t in taus
    ]
    lo = [m for m in margins if m < ns.margin_nats]
    lo_masked = [msk for msk, m in zip(masked, margins) if m < ns.margin_nats]
    out = {
        "model": ns.model,
        "n_tokens": n,
        "n_low_margin": len(lo),
        "drift_nats": {
            "median": round(statistics.median(drifts), 4),
            "p90": round(sorted(drifts)[int(0.9 * n)], 4),
            "mean": round(statistics.fmean(drifts), 4),
        },
        "false_reject_sweep": false_reject,
        "forgery_masking": {
            "all": round(sum(masked) / len(masked), 4) if masked else None,
            f"low_margin_lt_{ns.margin_nats}": (
                round(sum(lo_masked) / len(lo_masked), 4) if lo_masked else None),
            "note": "fraction of tokens where re-feed drift >= the top1-top2 gap, "
                    "i.e. a swap to the runner-up is smaller than the verifier's "
                    "own re-feed noise. true-KV verifier: 0 (drift ~ 0).",
        },
    }

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
