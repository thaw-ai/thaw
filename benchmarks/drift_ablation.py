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
     trunk is now freshly prefilled. Relative to passes 1/2 this changes the
     re-feed procedure as actually run: the recomputed prefix state plus the
     step-level batch dynamics that heavy chunked prefill induces.
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

_BOXED_RE = re.compile(r"\\boxed\{\s*\$?\s*(-?[\d][\d,]*(?:\.\d+)?)\s*\}")
_ANSWER_RE = re.compile(r"answer\s*[:=]?\s*\$?\s*(-?[\d][\d,]*(?:\.\d+)?)", re.I)
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _to_num(s: str):
    try:
        return float(s.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def extract_answer(text: str):
    """\\boxed{} wins (RL math models), then 'Answer: <num>', then last number."""
    boxed = list(_BOXED_RE.finditer(text))
    if boxed:
        return _to_num(boxed[-1].group(1))
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
                  skip_token_ids, block_size=BLOCK_SIZE):
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
        if abs_pos % block_size > boundary_slack:
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




def dedupe(records):
    """Drop duplicate (problem_id, rollout, abs_pos) keys, keeping the first.
    Needed when pooling legacy cohorts that scanned overlapping problem ranges."""
    seen, out = set(), []
    for r in records:
        k = (r["problem_id"], r["rollout"], r["abs_pos"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def full_report(records, margin_split=0.3):
    """Every metric the paper reports, from per-pivot records, in one place.

    Conditioning sets:
      - common: any of the three passes gives A_t != 0 (the prereg metric's
        denominator, shared by both comparisons)
      - exact1: A_exact1 != 0 (treatment-independent conditioning)
    Decomposition: zero-crossing (one side zero) vs polarity (signs strictly
    opposite). Clustered inference: per-problem flip-rate difference t-test +
    sign test. Downstream demo: Jaccard overlap of critical-token sets.
    """
    import math
    import statistics

    def rates(sub, b_key):
        n = len(sub)
        if n == 0:
            return {"n": 0}
        flips = [r for r in sub if sign(r["A_exact1"]) != sign(r[b_key])]
        polarity = [r for r in flips
                    if sign(r["A_exact1"]) != 0 and sign(r[b_key]) != 0]
        return {
            "n": n,
            "flips": len(flips),
            "flip_rate": len(flips) / n,
            "zero_crossings": len(flips) - len(polarity),
            "polarity_reversals": len(polarity),
            "polarity_rate": len(polarity) / n,
        }

    common = [r for r in records
              if sign(r["A_exact1"]) or sign(r["A_exact2"]) or sign(r["A_refeed"])]
    exact1_set = [r for r in records if sign(r["A_exact1"]) != 0]

    out = {
        "n_records": len(records),
        "n_problems": len({r["problem_id"] for r in records}),
        "common_set": {
            "refeed": rates(common, "A_refeed"),
            "floor": rates(common, "A_exact2"),
        },
        "exact1_conditioned": {
            "refeed": rates(exact1_set, "A_refeed"),
            "floor": rates(exact1_set, "A_exact2"),
        },
    }

    # Eligibility-source decomposition: which pass made each common pivot
    # eligible (re-feed creating its own eligibility inflates its flip count).
    out["eligibility_sources"] = {
        "exact1_nonzero": sum(1 for r in common if sign(r["A_exact1"])),
        "only_refeed_nonzero": sum(
            1 for r in common
            if not sign(r["A_exact1"]) and not sign(r["A_exact2"])
            and sign(r["A_refeed"])),
        "only_exact2_nonzero": sum(
            1 for r in common
            if not sign(r["A_exact1"]) and not sign(r["A_refeed"])
            and sign(r["A_exact2"])),
    }

    # Bias: is the re-feed perturbation zero-mean?
    if common:
        diffs = [r["A_refeed"] - r["A_exact1"] for r in common]
        m = statistics.mean(diffs)
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        out["bias"] = {
            "mean_diff": m,
            "t": (m / (sd / math.sqrt(len(diffs)))) if sd > 0 else 0.0,
            "n": len(diffs),
        }

    # McNemar on the common set (unclustered; reported alongside clustered).
    b = sum(1 for r in common
            if sign(r["A_exact1"]) != sign(r["A_refeed"])
            and sign(r["A_exact1"]) == sign(r["A_exact2"]))
    c = sum(1 for r in common
            if sign(r["A_exact1"]) == sign(r["A_refeed"])
            and sign(r["A_exact1"]) != sign(r["A_exact2"]))
    out["mcnemar"] = {
        "b_refeed_only": b, "c_floor_only": c,
        "z": ((b - c) / math.sqrt(b + c)) if (b + c) else 0.0,
    }

    # Problem-clustered inference: per-problem (refeed flip rate - floor flip
    # rate) on that problem's common pivots; t over problems + sign test.
    by_prob = {}
    for r in common:
        by_prob.setdefault(r["problem_id"], []).append(r)
    per_prob = []
    for pid, sub in by_prob.items():
        rf = sum(1 for r in sub if sign(r["A_exact1"]) != sign(r["A_refeed"]))
        fl = sum(1 for r in sub if sign(r["A_exact1"]) != sign(r["A_exact2"]))
        per_prob.append((pid, (rf - fl) / len(sub)))
    diffs = [d for _, d in per_prob]
    if len(diffs) > 1:
        m = statistics.mean(diffs)
        sd = statistics.stdev(diffs)
        out["clustered"] = {
            "n_problems": len(diffs),
            "mean_excess": m,
            "t": (m / (sd / math.sqrt(len(diffs)))) if sd > 0 else 0.0,
            "problems_refeed_worse": sum(1 for d in diffs if d > 0),
            "problems_floor_worse": sum(1 for d in diffs if d < 0),
            "problems_equal": sum(1 for d in diffs if d == 0),
        }

    # Downstream demo: critical-token selection overlap. A pivot is "critical"
    # under a method if |A_t| >= threshold; Jaccard against the exact-resume
    # selection, with the replica as the attainable ceiling.
    out["critical_token_jaccard"] = {}
    for thr in (0.5, 0.75):
        sel = {}
        for key in ("A_exact1", "A_exact2", "A_refeed"):
            sel[key] = {(r["problem_id"], r["rollout"], r["abs_pos"])
                        for r in records if abs(r[key]) >= thr}
        def jac(a, b):
            u = sel[a] | sel[b]
            return (len(sel[a] & sel[b]) / len(u)) if u else None
        out["critical_token_jaccard"][f"thr_{thr}"] = {
            "n_exact1": len(sel["A_exact1"]),
            "refeed_vs_exact1": jac("A_refeed", "A_exact1"),
            "replica_vs_exact1": jac("A_exact2", "A_exact1"),
        }

    # Margin buckets on the common set.
    for name, pred in ((f"margin_lt_{margin_split}",
                        lambda r: r["margin"] < margin_split),
                       (f"margin_ge_{margin_split}",
                        lambda r: r["margin"] >= margin_split)):
        sub = [r for r in common if pred(r)]
        out[name] = {"refeed": rates(sub, "A_refeed"),
                     "floor": rates(sub, "A_exact2")}

    # Greedy probes: divergence rates + bit-exactness of the per-pivot
    # max-over-arms first-token logprob delta.
    g = [r for r in records if r.get("greedy_divergence_refeed") is not None]
    if g:
        probe = {"n_pivots": len(g)}
        for tag in ("refeed", "exact2"):
            div = sum(1 for r in g if r.get(f"greedy_divergence_{tag}"))
            ds_ = [r[f"greedy_first_logprob_delta_{tag}"] for r in g
                   if r.get(f"greedy_first_logprob_delta_{tag}") is not None]
            nz = sorted(d for d in ds_ if d > 0)
            probe[tag] = {
                "divergence_rate": div / len(g),
                "bit_exact_rate": (sum(1 for d in ds_ if d == 0.0) / len(ds_))
                                  if ds_ else None,
                "median_nonzero_delta": nz[len(nz) // 2] if nz else None,
                "p90_delta": sorted(ds_)[int(0.9 * len(ds_))] if ds_ else None,
            }
        out["greedy_probe"] = probe

    # Cache instrumentation summary (present only on instrumented runs).
    instr = [r for r in records if r.get("cache_ok_exact1") is not None]
    if instr:
        out["cache_instrumentation"] = {
            "n_instrumented": len(instr),
            "exact1_ok_rate": sum(1 for r in instr if r["cache_ok_exact1"]) / len(instr),
            "exact2_ok_rate": sum(1 for r in instr if r["cache_ok_exact2"]) / len(instr),
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
    scan_start = args.scan_start
    candidates = [
        {"id": i, "question": ds[i]["question"], "gold": gold_answer(ds[i]["answer"])}
        for i in range(scan_start, min(scan_start + scan_limit, len(ds)))
    ]

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.92,
        max_model_len=4096,
        seed=0,
    )
    cont_temp = args.cont_temp if args.cont_temp is not None else args.temp
    try:
        block_size = llm.llm_engine.cache_config.block_size
    except AttributeError:
        block_size = llm.llm_engine.vllm_config.cache_config.block_size
    print(f"[drift] engine block_size={block_size}", flush=True)
    tok = llm.get_tokenizer()
    skip_ids = {tid for tid in (tok.eos_token_id,) if tid is not None}
    skip_ids |= set(getattr(tok, "all_special_ids", []) or [])

    def reset_cache():
        if hasattr(llm, "reset_prefix_cache"):
            ok = llm.reset_prefix_cache()
        else:
            ok = llm.llm_engine.reset_prefix_cache()
        if ok is False:
            raise RuntimeError("reset_prefix_cache() returned False; blocks "
                               "still in use -- a re-feed pass would silently "
                               "run as an exact pass")

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
            if len(gen_ids) < block_size:
                continue
            tokens = extract_token_logprobs(comp, max_topk=5)
            pivots = select_pivots(
                tokens, prompt_len,
                margin_lt=args.margin, boundary_slack=args.boundary_slack,
                max_pivots=args.max_pivots, skip_token_ids=skip_ids,
                block_size=block_size,
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
                        temperature=cont_temp, top_p=args.top_p,
                        max_tokens=args.max_cont_tokens, seed=seed,
                    ))
                prompts.append({"prompt_token_ids": list(ids)})
            t0 = time.time()
            outs = llm.generate(prompts, sps)
            cached = [getattr(o, "num_cached_tokens", None) for o in outs]
            return outs, time.time() - t0, cached

        exact1, t1, cached1 = run_pass("exact1", fresh=False)
        exact2, t2, cached2 = run_pass("exact2", fresh=False)
        refeed, t3, cached3 = run_pass("refeed", fresh=True)

        # Cache-hit instrumentation: for the exact passes, every request's
        # prompt must have hit at least the block-aligned trunk prefix, else
        # the "exact" arm silently degraded to a partial re-feed (eviction /
        # preemption). Recorded per pivot, summarized per problem.
        def cache_ok_by_key(cached):
            ok, mins = {}, {}
            for (key, _arm, _k, _ids, _seed), c in zip(requests, cached):
                if c is None:
                    continue
                expected = (pivot_meta[key]["abs_pos"] // block_size) * block_size
                mins[key] = min(mins.get(key, 1 << 30), c)
                this_ok = c >= expected
                ok[key] = ok.get(key, True) and this_ok
            return ok, mins

        ok1, _min1 = cache_ok_by_key(cached1)
        ok2, _min2 = cache_ok_by_key(cached2)
        viol1 = sum(1 for v in ok1.values() if not v)
        viol2 = sum(1 for v in ok2.values() if not v)
        if viol1 or viol2:
            print(f"[drift] WARNING problem {prob['id']}: cache-hit violations "
                  f"exact1={viol1} exact2={viol2} of {len(pivot_meta)} pivots",
                  flush=True)

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
            rec.update({"A_exact1": a1, "A_exact2": a2, "A_refeed": a3,
                        "cache_ok_exact1": ok1.get(key),
                        "cache_ok_exact2": ok2.get(key)})
            for tag, g in (("exact1", g1), ("exact2", g2), ("refeed", g3)):
                arms = g.get(key, {})
                rec[f"greedy_first_{tag}"] = {
                    arm: [d["token_ids"][0] if d["token_ids"] else None,
                          d["first_logprob"]]
                    for arm, d in arms.items()
                }
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
            "cache_violations": {"exact1": viol1, "exact2": viol2},
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
            "cont_temp": args.cont_temp,
            "screen_mixed": args.screen_mixed,
            "scan_limit": args.scan_limit,
            "scan_start": args.scan_start,
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
    p.add_argument("--cont-temp", type=float, default=None,
                   help="continuation temperature (default: --temp); lets the "
                        "temp-0 secondary keep trunk diversity at 0.7")
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
    p.add_argument("--scan-start", type=int, default=0,
                   help="first GSM8K test index to scan; use 20+ for screened "
                        "cohorts so they never overlap the primary cohort")
    p.add_argument("--out", default="results/drift_ablation.json")
    p.add_argument("--smoke", action="store_true",
                   help="2 problems, 4 rollouts, K=2, 2 pivots/rollout")
    p.add_argument("--analyze", metavar="JSON",
                   help="recompute aggregates from an existing results file")
    args = p.parse_args()

    if args.analyze:
        recs = []
        for path in args.analyze.split(","):
            with open(path.strip()) as f:
                recs.extend(json.load(f)["pivots"])
        recs = dedupe(recs)
        print(json.dumps(full_report(recs), indent=2))
        return

    if args.smoke:
        args.problems, args.rollouts, args.k, args.max_pivots = 2, 4, 2, 2

    run(args)


if __name__ == "__main__":
    main()
