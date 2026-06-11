# Pre-registration: Re-feed vs Exact-KV Drift Ablation

Date locked: 2026-06-10, BEFORE any GPU run. This file is the commitment device.
Per the 2026-06-08 verification memo: the sign-flip rate is highly sensitive to
temperature, K, grading, and token filter, so all four are fixed here and may not
be changed after looking at results. Any deviation must be reported as a deviation.

## Question

For per-token counterfactual credit estimates on LLM rollouts, how often does the
universal "re-feed the transcript" shortcut disagree IN SIGN with the same estimate
computed from a bit-exact KV fork of the live state?

## Decision rule (locked)

- Sign disagreement > 5% on the pre-registered token set: thaw is load-bearing.
  Proceed toward the Causal Token Audit paper.
- Sign disagreement < 2%: re-feed is good enough. Ship as an honest negative
  result (blog + repo), do not force a paper.
- 2-5%: gray band. Decide by where flips concentrate (low-margin tokens at the
  temperature people actually train at = lean paper; uniform noise = lean blog).

## Locked settings

- Model: Qwen/Qwen2.5-7B-Instruct (not gated, validated with the thaw stack).
  Note: a GRPO-trained checkpoint is the follow-up if results land in the gray
  band; this run measures drift on best-of-N style rollouts, which is the
  estimator construction every cited credit paper shares.
- Data: first 20 problems of GSM8K test split (deterministic slice, no cherry-pick).
- Rollouts per problem: 8, sampled at temperature 0.7, top_p 0.95, max 512 new tokens.
- Pivot selection (locked filter): generated tokens where the top-1/top-2 logprob
  gap < 1.0 (rewind low-margin flag), restricted to pivots landing within 2 tokens
  of a 16-token KV block boundary (thaw forks at block granularity; this keeps
  "exact" honest). Cap: 5 pivots per rollout, earliest-first.
- Credit estimate at each pivot: force the top-2 (counterfactual) token, sample
  K=4 continuations per arm at temperature 0.7, grade by GSM8K exact-match on the
  final answer. A_t = mean(R | actual token) - mean(R | forced alt token).
- Computed twice per pivot:
  1. EXACT arm: thaw fork of the live KV state at the block boundary, continuation
     resumes from bit-identical cached state.
  2. RE-FEED arm: transcript[:pivot] re-submitted as a fresh prompt (fresh prefill,
     whatever chunked-prefill/batching the engine does on a fresh request), same
     forced token, same K, same sampling seeds.
- Seeds: per-(pivot, arm, k) seeds fixed as hash(problem_id, rollout_id, pivot_pos,
  arm, k) so the two methods get identical sampling randomness. The ONLY uncontrolled
  difference between methods is the prefill path.
- NOISE FLOOR CONTROL (added 2026-06-10 before any GPU run): the exact arm is
  run TWICE (two identical batched passes from the same warm cache, same seeds).
  The exact-vs-exact replica sign-flip rate is the noise floor from batching /
  continuous-batching numerics. Re-feed drift only counts as real to the extent
  the exact-vs-refeed flip rate exceeds the exact-vs-exact floor.
- Alt token definition: the highest-logprob token in the captured top-k that is
  not the actually-sampled token. Pivots where that token is the EOS/special
  token are skipped.
- Primary metric: % of pivots where sign(A_t_exact) != sign(A_t_refeed), among
  pivots where at least one method gives a nonzero A_t. Reported alongside the
  exact-vs-exact floor on the same pivots.
- Secondary (reported, not decision-bearing): same rate broken out by margin bucket
  (<0.3, 0.3-1.0); mean |A_t| difference; rate at which the forced-token logprob
  itself differs by > 1e-3 between methods; greedy-continuation token divergence rate.

## Amendment 1 (2026-06-10, after smoke run, before any full-run results)

Smoke run (2 problems, 4 rollouts, K=2) exposed a POWER problem, not a drift
problem: Qwen2.5-7B-Instruct solves easy GSM8K problems in every continuation,
so A_t = 0 in both arms at almost every pivot (1/16 eligible) and the primary
metric is undecidable on the original cohort. The smoke run also validated the
instrument: exact-vs-exact replica first-token logprobs were bit-identical at
13/16 pivots (floor ~= 0) while re-feed shifted the logprob at 16/16 pivots
(up to 0.12). The drift mechanism is confirmed; the grading signal saturates.

Amendment, declared before looking at any full-run primary results:
- Keep the original cohort (first 20 GSM8K test problems) as specified.
- Add a SCREENED cohort: scanning GSM8K test from index 0, keep problems whose
  8 TRUNK rollouts have mixed accuracy (neither 0/8 nor 8/8 correct), until 20
  problems are collected. The screen uses only trunk accuracy, computed before
  any arm continuations run, and is blind to drift direction.
- The >5% / <2% decision rule applies to eligible pivots POOLED across both
  cohorts. Both cohorts' rates are also reported separately.

## Hardware / cost

- 1x A100 80GB, RunPod. Budget cap $20 of the $29 in the account. Stop the pod
  immediately after the results JSON is pulled off.

## Honesty constraints

- Report the result whichever way it goes. A null is a publishable negative.
- "Exact" claims are exact to the 16-token block boundary, stated plainly.
- No re-running with different settings to move the headline number. Sensitivity
  analyses (temp 0.0, K=8) are allowed only as clearly-labeled secondary runs and
  cannot replace the primary.

---

## Provenance note (added when published, 2026-06-11)

This file was written in a private planning directory before any GPU run on
2026-06-10 (settings locked first; Amendment 1 added after the smoke run and
before any full-run results). It is committed to the public repository now,
which means its git timestamp postdates the runs it governs. The
pre-registration is therefore self-attested rather than externally
timestamped, and the paper describes it as such. Future experiments by this
project will be registered with an external timestamp before execution.
