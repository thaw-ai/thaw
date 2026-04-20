"""
Pad a trunk prompt to a target token count using realistic filler.

Used by demos to run the fork primitive against large contexts. A
300-token trunk undersells the fork — prefill savings scale with
context length, so we inflate the trunk with domain-appropriate filler
to show the real win.

Filler is designed to look like plausible prior context so the model
doesn't veer off topic. Not adversarial.
"""
from __future__ import annotations

# Realistic filler: prior-reasoning transcript for a math tutor. Reads
# like an agent that has been reasoning at length about similar problems.
_REASONING_FILLER = """\
Earlier I worked through a warmup problem to establish the framework.
Given a tiered pricing structure with fixed base costs, variable per-unit
rates, and a compounding surcharge, the total cost T on input x is:

    T(x) = (1 + tax) * ((1 + surcharge) * (base + rate * max(0, x - threshold)))

This decomposes cleanly. Let me verify by expanding one level: the raw
shipping cost S is base + rate * excess, where excess is whatever exceeds
the first threshold pounds. Then the fuel-surcharged subtotal is
S * (1 + fuel_rate). Then the tax-inclusive total is that times (1 + tax).

For the warmup, a 12-pound package had base 5 flat, rate 1.50 per extra
pound, fuel 10%, tax 7%. So excess = 7, raw = 5 + 7*1.50 = 15.50, surcharged
= 15.50 * 1.10 = 17.05, taxed = 17.05 * 1.07 = 18.2435 → $18.24.

Checking the structure: doubling the excess roughly doubles the raw, and
since fuel and tax are multiplicative they pass through untouched. So a
24-pound package at the same rates would raw to 5 + 19*1.50 = 33.50,
surcharged to 36.85, taxed to 39.4295 → $39.43. Plausible.

Key invariants to preserve:
1. The first threshold pounds are a flat charge, not a per-pound charge.
2. Fuel surcharge applies to shipping only, not to tax.
3. Tax applies to the shipping + fuel amount, not separately.
4. Percentages compound multiplicatively, so the order matters only if
   there's a cap; here there's no cap so order is (1+s) * (1+t) = 1 + s + t + s*t.

Now returning to the current problem with fresh rates and weight:
"""

# Realistic filler: prior-turn transcript for a coding agent. Reads like
# an engineer walking through API design before writing the implementation.
_CODING_FILLER = """\
Before committing to an approach, let me enumerate the requirements and
the corners where implementations usually break.

Requirement recap:
- Bounded size: max_size entries, LRU eviction when over.
- Per-entry TTL: expired entries don't count toward max_size, don't
  appear in get(), don't count toward len().
- Thread safety: multiple producers and consumers with threading.RLock.
- API: __init__, get, put, __len__, clear.

Corner cases I've seen break implementations:
- A lazy expirer that only expires on access will misreport len() when
  the cache has sat idle; the spec says len() reflects unexpired entries,
  so we need to expire *before* returning a length.
- If we use OrderedDict.move_to_end() for LRU, we need to do it on get()
  as well as put() — forgetting get() means get-heavy workloads LRU-evict
  recently-read entries.
- On put() of an existing key, the spec says update-value-and-reset-TTL.
  Skipping the TTL reset is a common bug.
- Eviction counts: max_size is a count of live entries, so if we evict
  by "until len < max_size", lazy expiration interleaved with puts can
  leave us with a split eviction behavior. Safer: prune expired first,
  then LRU-evict.
- Locking: RLock is required if get() calls put() internally (for TTL
  refresh patterns). A plain Lock would deadlock.

Having laid those out, the implementation choice is between:
(a) OrderedDict + sidecar deadline dict, prune lazily;
(b) dict + doubly-linked list + per-node deadline, prune on access;
(c) dict + min-heap-of-deadlines, prune via heap pop;
(d) subclass dict (not recommended — API drift);
(e) composition over two OrderedDicts, one for LRU one for TTL.

I'll write down the trade-offs before picking. Each has different
performance characteristics under different workloads.
"""


def pad_trunk(
    tokenizer,
    base_trunk: str,
    target_tokens: int | None,
    kind: str = "reasoning",
) -> tuple[str, int]:
    """
    Pad ``base_trunk`` to ``target_tokens`` tokens by prepending filler.

    If ``target_tokens`` is None or the base trunk already exceeds it,
    returns the base trunk unchanged plus its current token count.

    Filler is prepended (not appended) so the base trunk's ending stays
    the natural pivot point for branching.

    Returns (padded_text, actual_token_count).
    """
    base_ids = tokenizer.encode(base_trunk, add_special_tokens=False)
    base_len = len(base_ids)

    if target_tokens is None or base_len >= target_tokens:
        return base_trunk, base_len

    filler = _CODING_FILLER if kind == "coding" else _REASONING_FILLER
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)

    needed = target_tokens - base_len
    reps = (needed + len(filler_ids) - 1) // len(filler_ids)
    padded_filler_ids = (filler_ids * reps)[:needed]

    # Rebuild text from tokens so the final token count is exact.
    padded_filler = tokenizer.decode(padded_filler_ids, skip_special_tokens=True)
    padded_text = padded_filler + "\n\n" + base_trunk

    actual_ids = tokenizer.encode(padded_text, add_special_tokens=False)
    return padded_text, len(actual_ids)
