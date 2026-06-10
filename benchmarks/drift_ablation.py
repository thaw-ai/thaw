"""Re-feed vs exact-KV drift ablation.

Pre-registered experiment (private/plans/2026-06-10_drift-ablation-prereg.md).

Question: for per-token counterfactual credit estimates on LLM rollouts, how
often does the universal "re-feed the transcript" shortcut disagree IN SIGN
with the same estimate computed from the exact decode-time KV state?

Design (per problem):
  1. Generate N rollouts from a GSM8K problem with top-k logprobs captured.
     The decode-time KV blocks for every rollout are now in the prefix cache.
  2. Select low-margin pivot tokens near 16-token block boundaries.
  3. EXACT pass 1: for each pivot, continue from trunk[:p] + forced token
     (actual and alt arms, K seeded continuations each). The trunk prefix
     cache-hits the decode-time blocks: continuations resume from the exact
     state the rollout actually passed through.
  4. EXACT pass 2: identical request batch, cache still warm. This is the
     noise-floor replica (continuous-batching numerics only).
  5. reset_prefix_cache(), then RE-FEED pass: identical request batch, but the
     trunk is now freshly prefilled. The only changed variable vs pass 1/2 is
     the prefill path -- the thing every published credit method relies on.
  6. Grade all continuations by GSM8K exact-match; A_t = mean(R|actual) -
     mean(R|alt) per (pivot, pass). Primary metric: sign disagreement rate of
     exact-pass-1 vs re-feed, reported against the exact1-vs-exact2 floor.

Single GPU, no engine internals touched; thaw_vllm.rewind supplies the
logprob extraction. Run:

  python benchmarks/drift_ablation.py --out results/drift.json          # full
  python benchmarks/drift_ablation.py --smoke --out results/smoke.json  # tiny
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import zlib

BLOCK_SIZE = 16

# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"answer\s*[:=]?\s*\$?\s*(-?[\d][\d,]*(?:\.\d+)?)", re.I)
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _to_num(s: str):
    try:
        return float(s.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def extract_answer(text: str):
    """Last 'Answer: <num>' match wins; fall back to the last number in text."""
    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        return _to_num(matches[-1].group(1))
    nums = _NUMBER_RE.findall(text)
    if nums:
        return _to_num(nums[-1])
    return None


def gold_answer(answer_field: str):
    """GSM8K gold answers end with '#### <num>'."""
    tail = answer_field.rsplit("####", 1)[-1].strip()
    return _to_num(tail.replace(",", ""))


def grade(text: str, gold) -> int:
    pred = extract_answer(text)
    if pred is None or gold is None:
        return 0
    return int(abs(pred - gold) < 1e-4)


# ---------------------------------------------------------------------------
# Pivot selection (locked filter from the prereg)
# ---------------------------------------------------------------------------


def select_pivots(tokens, prompt_len, *, margin_lt, boundary_slack, max_pivots,
                  skip_token_ids):
    """tokens: rewind-style list of {token_id, logprob, topk}. Returns a list of
    {gen_index, abs_pos, actual_id, alt_id, margin} dicts.

    abs_pos is the absolute position of the pivot token: the continuation
    prompt is full_ids[:abs_pos] + [forced_token]. Locked filter: top1-top2
    gap < margin_lt, abs_pos within `boundary_slack` tokens past a block
    boundary, earliest-first, capped at max_pivots."""
    pivots = []
    for i, t in enumerate(tokens):
        topk = t.get("topk") or []
        if len(topk) < 2:
            continue
        lp0, lp1 = topk[0].get("logprob"), topk[1].get("logprob")
        if lp0 is None or lp1 is None:
            continue
        margin = lp0 - lp1
        if margin >= margin_lt:
            continue
        abs_pos = prompt_len + i
        if abs_pos % BLOCK_SIZE > boundary_slack:
            continue
        actual_id = t["token_id"]
        alt = next((c for c in topk if c["token_id"] != actual_id), None)
        if alt is None or alt["token_id"] in skip_token_ids:
            continue
        pivots.append({
            "gen_index": i,
            "abs_pos": abs_pos,
            "actual_id": int(actual_id),
            "alt_id": int(alt["token_id"]),
            "margin": float(margin),
        })
        if len(pivots) >= max_pivots:
            break
    return pivots


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def aggregate(pivot_records, margin_split=0.3):
    """Compute the pre-registered metrics from per-pivot records."""
    def flip_rate(records, a, b):
        eligible = [r for r in records
                    if sign(r[a]) != 0 or sign(r[b]) != 0]
        flips = [r for r in eligible if sign(r[a]) != sign(r[b])]
        return {
            "eligible": len(eligible),
            "flips": len(flips),
            "rate": (len(flips) / len(eligible)) if eligible else None,
        }

    out = {
        "n_pivots": len(pivot_records),
        "primary_exact_vs_refeed": flip_rate(pivot_records, "A_exact1", "A_refeed"),
        "floor_exact_vs_exact": flip_rate(pivot_records, "A_exact1", "A_exact2"),
    }
    for name, pred in (
        (f"margin_lt_{margin_split}", lambda r: r["margin"] < margin_split),
        (f"margin_ge_{margin_split}", lambda r: r["margin"] >= margin_split),
    ):
        sub = [r for r in pivot_records if pred(r)]
        out[name] = {
            "exact_vs_refeed": flip_rate(sub, "A_exact1", "A_refeed"),
            "floor": flip_rate(sub, "A_exact1", "A_exact2"),
        }
    deltas = [abs(r["A_exact1"] - r["A_refeed"]) for r in pivot_records]
    out["mean_abs_A_delta_exact_vs_refeed"] = (
        sum(deltas) / len(deltas) if deltas else None
    )
    g_div = [r for r in pivot_records if r.get("greedy_divergence_refeed") is not None]
    if g_div:
        out["greedy_probe"] = {
            "n": len(g_div),
            "refeed_divergence_rate": sum(
                1 for r in g_div if r["greedy_divergence_refeed"]) / len(g_div),
            "exact2_divergence_rate": sum(
                1 for r in g_div if r.get("greedy_divergence_exact2")) / len(g_div),
        }
    return out


# ---------------------------------------------------------------------------
# GPU run
# ---------------------------------------------------------------------------


def run(args):
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    from thaw_vllm.rewind import extract_token_logprobs

    t_start = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    scan_limit = args.scan_limit if args.screen_mixed else args.problems
    candidates = [
        {"id": i, "question": ds[i]["question"], "gold": gold_answer(ds[i]["answer"])}
        for i in range(min(scan_limit, len(ds)))
    ]

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.92,
        max_model_len=4096,
        seed=0,
    )
    tok = llm.get_tokenizer()
    skip_ids = {tid for tid in (tok.eos_token_id,) if tid is not None}
    skip_ids |= set(getattr(tok, "all_special_ids", []) or [])

    def reset_cache():
        if hasattr(llm, "reset_prefix_cache"):
            llm.reset_prefix_cache()
        else:
            llm.llm_engine.reset_prefix_cache()

    instruction = (
        "\n\nReason step by step. End your response with one final line in"
        " exactly this format:\nAnswer: <number>"
    )

    trunk_sp = SamplingParams(
        n=args.rollouts, temperature=args.temp, top_p=args.top_p,
        max_tokens=args.max_trunk_tokens, logprobs=5,
    )

    all_pivot_records = []
    per_problem = []
    accepted = 0

    for prob in candidates:
        if accepted >= args.problems:
            break
        t_prob = time.time()
        reset_cache()

        msgs = [{"role": "user", "content": prob["question"] + instruction}]
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

        # Phase 1: trunk rollouts (decode-time KV lands in the prefix cache).
        sp = trunk_sp.clone() if hasattr(trunk_sp, "clone") else trunk_sp
        sp.seed = stable_seed(prob["id"], "trunk")
        out = llm.generate([prompt], sp)[0]
        prompt_ids = list(out.prompt_token_ids)
        prompt_len = len(prompt_ids)

        # Amendment 1 screen: trunk accuracy must be mixed (uses only trunk
        # rollouts, computed before any arm continuations exist).
        if args.screen_mixed:
            trunk_acc = sum(
                grade(c.text, prob["gold"]) for c in out.outputs
            ) / max(len(out.outputs), 1)
            if trunk_acc in (0.0, 1.0):
                print(f"[drift] problem {prob['id']}: screened out "
                      f"(trunk_acc={trunk_acc})", flush=True)
                continue
        accepted += 1

        # Phase 2: pivots.
        requests = []  # (pivot_key, arm, k, prompt_ids, seed)
        pivot_meta = {}
        for r_idx, comp in enumerate(out.outputs):
            gen_ids = list(comp.token_ids)
            if len(gen_ids) < BLOCK_SIZE:
                continue
            tokens = extract_token_logprobs(comp, max_topk=5)
            pivots = select_pivots(
                tokens, prompt_len,
                margin_lt=args.margin, boundary_slack=args.boundary_slack,
                max_pivots=args.max_pivots, skip_token_ids=skip_ids,
            )
            full_ids = prompt_ids + gen_ids
            roll_correct = grade(comp.text, prob["gold"])
            for piv in pivots:
                key = (prob["id"], r_idx, piv["abs_pos"])
                pivot_meta[key] = {
                    "problem_id": prob["id"], "rollout": r_idx,
                    "abs_pos": piv["abs_pos"], "gen_index": piv["gen_index"],
                    "margin": piv["margin"], "rollout_correct": roll_correct,
                }
                prefix = full_ids[: piv["abs_pos"]]
                for arm, forced in (("actual", piv["actual_id"]),
                                    ("alt", piv["alt_id"])):
                    arm_ids = prefix + [forced]
                    for k in range(args.k):
                        seed = stable_seed(prob["id"], r_idx, piv["abs_pos"], arm, k)
                        requests.append((key, arm, k, arm_ids, seed))
                    # Greedy probe: numerics channel, one per arm.
                    requests.append((key, arm, "greedy", arm_ids, None))

        if not requests:
            per_problem.append({"problem_id": prob["id"], "pivots": 0})
            continue

        def run_pass(tag, fresh):
            if fresh:
                reset_cache()
            sps, prompts = [], []
            for (_, arm, k, ids, seed) in requests:
                if k == "greedy":
                    sps.append(SamplingParams(
                        temperature=0.0, max_tokens=args.greedy_probe_tokens,
                        logprobs=2,
                    ))
                else:
                    sps.append(SamplingParams(
                        temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_cont_tokens, seed=seed,
                    ))
                prompts.append({"prompt_token_ids": list(ids)})
            t0 = time.time()
            outs = llm.generate(prompts, sps)
            return outs, time.time() - t0

        exact1, t1 = run_pass("exact1", fresh=False)
        exact2, t2 = run_pass("exact2", fresh=False)
        refeed, t3 = run_pass("refeed", fresh=True)

        # Phase 3: grade and assemble per-pivot records.
        def collect(outs):
            scores = {}   # key -> {arm -> [0/1 per k]}
            greedy = {}   # key -> {arm -> {token_ids, first_logprob}}
            for (key, arm, k, _ids, _seed), o in zip(requests, outs):
                comp = o.outputs[0]
                if k == "greedy":
                    first_lp = None
                    if comp.logprobs:
                        step = comp.logprobs[0]
                        chosen = step.get(comp.token_ids[0]) if comp.token_ids else None
                        if chosen is not None:
                            first_lp = float(chosen.logprob)
                    greedy.setdefault(key, {})[arm] = {
                        "token_ids": list(comp.token_ids),
                        "first_logprob": first_lp,
                    }
                else:
                    scores.setdefault(key, {}).setdefault(arm, []).append(
                        grade(comp.text, prob["gold"])
                    )
            return scores, greedy

        s1, g1 = collect(exact1)
        s2, g2 = collect(exact2)
        s3, g3 = collect(refeed)

        def a_t(scores, key):
            arms = scores.get(key, {})
            act, alt = arms.get("actual", []), arms.get("alt", [])
            if not act or not alt:
                return None
            return sum(act) / len(act) - sum(alt) / len(alt)

        n_recorded = 0
        for key, meta in pivot_meta.items():
            a1, a2, a3 = a_t(s1, key), a_t(s2, key), a_t(s3, key)
            if a1 is None or a2 is None or a3 is None:
                continue
            rec = dict(meta)
            rec.update({"A_exact1": a1, "A_exact2": a2, "A_refeed": a3})
            for tag, g in (("exact2", g2), ("refeed", g3)):
                div = None
                lp_delta = None
                base = g1.get(key, {})
                other = g.get(key, {})
                if base and other:
                    div = any(
                        base.get(arm, {}).get("token_ids")
                        != other.get(arm, {}).get("token_ids")
                        for arm in ("actual", "alt")
                    )
                    lps = [
                        abs(base[arm]["first_logprob"] - other[arm]["first_logprob"])
                        for arm in ("actual", "alt")
                        if base.get(arm, {}).get("first_logprob") is not None
                        and other.get(arm, {}).get("first_logprob") is not None
                    ]
                    lp_delta = max(lps) if lps else None
                rec[f"greedy_divergence_{tag}"] = div
                rec[f"greedy_first_logprob_delta_{tag}"] = lp_delta
            all_pivot_records.append(rec)
            n_recorded += 1

        per_problem.append({
            "problem_id": prob["id"], "pivots": n_recorded,
            "rollout_accuracy": sum(
                grade(c.text, prob["gold"]) for c in out.outputs
            ) / max(len(out.outputs), 1),
            "pass_seconds": [round(t1, 1), round(t2, 1), round(t3, 1)],
            "total_seconds": round(time.time() - t_prob, 1),
        })

        # Crash-safe incremental dump.
        _dump(args, all_pivot_records, per_problem, t_start, final=False)
        print(f"[drift] problem {accepted}/{args.problems} (id={prob['id']}): "
              f"{n_recorded} pivots, {per_problem[-1]['total_seconds']}s",
              flush=True)

    _dump(args, all_pivot_records, per_problem, t_start, final=True)
    agg = aggregate(all_pivot_records)
    print(json.dumps(agg, indent=2))


def _dump(args, pivot_records, per_problem, t_start, *, final):
    payload = {
        "experiment": "refeed-vs-exact-kv-drift-ablation",
        "prereg": "private/plans/2026-06-10_drift-ablation-prereg.md",
        "config": {
            "model": args.model, "problems": args.problems,
            "rollouts": args.rollouts, "temperature": args.temp,
            "top_p": args.top_p, "k": args.k, "margin_lt": args.margin,
            "boundary_slack": args.boundary_slack,
            "max_pivots_per_rollout": args.max_pivots,
            "max_trunk_tokens": args.max_trunk_tokens,
            "max_cont_tokens": args.max_cont_tokens,
            "screen_mixed": args.screen_mixed,
            "scan_limit": args.scan_limit,
        },
        "final": final,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "aggregate": aggregate(pivot_records),
        "per_problem": per_problem,
        "pivots": pivot_records,
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, args.out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--problems", type=int, default=20)
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--boundary-slack", type=int, default=2)
    p.add_argument("--max-pivots", type=int, default=5)
    p.add_argument("--max-trunk-tokens", type=int, default=512)
    p.add_argument("--max-cont-tokens", type=int, default=400)
    p.add_argument("--greedy-probe-tokens", type=int, default=32)
    p.add_argument("--screen-mixed", action="store_true",
                   help="Amendment 1 cohort: keep only problems whose trunk "
                        "rollouts have mixed accuracy (scan until --problems "
                        "are collected)")
    p.add_argument("--scan-limit", type=int, default=200,
                   help="max GSM8K problems to scan when --screen-mixed")
    p.add_argument("--out", default="results/drift_ablation.json")
    p.add_argument("--smoke", action="store_true",
                   help="2 problems, 4 rollouts, K=2, 2 pivots/rollout")
    p.add_argument("--analyze", metavar="JSON",
                   help="recompute aggregates from an existing results file")
    args = p.parse_args()

    if args.analyze:
        with open(args.analyze) as f:
            data = json.load(f)
        print(json.dumps(aggregate(data["pivots"]), indent=2))
        return

    if args.smoke:
        args.problems, args.rollouts, args.k, args.max_pivots = 2, 4, 2, 2

    run(args)


if __name__ == "__main__":
    main()
