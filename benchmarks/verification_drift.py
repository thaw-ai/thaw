#!/usr/bin/env python3
"""verification_drift.py — re-frame the re-feed-drift data as inference-verification error.

The verifiable-inference stack (DiFR arXiv:2511.20621, TOPLOC arXiv:2501.16007,
SVIP, VeriLLM) checks an LLM output by REPLAYING it: recompute the token-level
distribution and compare to a claimed signature, accept if they match. Every one
of them reconstructs the state by RE-FEEDING the transcript as a fresh prompt.

Our re-feed-drift result (arXiv:2606.15621) measured that re-feeding does not
reproduce the live decode-time state. This script reads that experiment's raw
per-pivot records and asks the verification question directly: at the low-margin
tokens that matter, how often does a re-feed-based verifier DISAGREE with what the
model actually generated — and how much of that disagreement is real (re-feed
error) vs the irreducible exact-replica noise floor?

Two metrics, both computed from existing data (no GPU):

  argmax disagreement
    `greedy_divergence_refeed` = the argmax next-token recomputed under re-feed
    differs from the true decode-time argmax (exact1). That is a hard verification
    failure: the verifier would compute a different token than was generated.
    Floor = `greedy_divergence_exact2` (a second exact-KV replica vs the first).

  threshold false-reject sweep
    A replay verifier accepts the generated token if the recomputed first-token
    logprob is within tolerance tau of the claimed one. Re-feed inflates that
    delta, so it falsely rejects genuine tokens. We sweep tau and report the
    re-feed reject rate vs the exact-replica floor.

The honest scope: this re-frames thaw's OWN 3-pass data (Qwen2.5-7B, the arXiv
2606.15621 run), not a head-to-head against a deployed DiFR/TOPLOC verifier — that
is the next experiment. But it quantifies, GPU-free, the divergence the entire
replay-verification class silently inherits, and which only thaw can remove by
replaying from the true decode-time KV.

Usage:
  python benchmarks/verification_drift.py benchmarks/results/2026-06-10_drift_ablation/primary.json
  python benchmarks/verification_drift.py <run.json> --json-out site/receipts/<date>_verification_drift.json
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional


def _rate(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


def argmax_disagreement(pivots: List[dict], margin_lt: Optional[float] = None) -> dict:
    """How often the recomputed argmax differs from the true decode-time argmax."""
    elig = [
        p for p in pivots
        if p.get("greedy_divergence_refeed") is not None
        and p.get("greedy_divergence_exact2") is not None
        and (margin_lt is None or p["margin"] < margin_lt)
    ]
    refeed = sum(1 for p in elig if p["greedy_divergence_refeed"])
    floor = sum(1 for p in elig if p["greedy_divergence_exact2"])
    r_refeed = _rate(refeed, len(elig))
    r_floor = _rate(floor, len(elig))
    return {
        "eligible": len(elig),
        "refeed_disagreement_rate": r_refeed,
        "replica_floor_rate": r_floor,
        "excess_pp": (None if r_refeed is None or r_floor is None
                      else round((r_refeed - r_floor) * 100, 2)),
    }


def false_reject_sweep(pivots: List[dict], taus: List[float]) -> List[dict]:
    """Reject = recomputed first-token logprob delta exceeds tolerance tau.

    The token was genuinely generated, so a reject is a FALSE reject. We compare
    the re-feed verifier's false-reject rate to the exact-replica floor at each tau.
    """
    elig = [
        p for p in pivots
        if p.get("greedy_first_logprob_delta_refeed") is not None
        and p.get("greedy_first_logprob_delta_exact2") is not None
    ]
    rows = []
    for tau in taus:
        refeed = sum(1 for p in elig if p["greedy_first_logprob_delta_refeed"] > tau)
        floor = sum(1 for p in elig if p["greedy_first_logprob_delta_exact2"] > tau)
        rr, rf = _rate(refeed, len(elig)), _rate(floor, len(elig))
        rows.append({
            "tau_nats": tau,
            "refeed_false_reject_rate": rr,
            "replica_floor_rate": rf,
            "excess_pp": (None if rr is None or rf is None
                          else round((rr - rf) * 100, 2)),
        })
    return rows


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


def credit_sign_flip(pivots: List[dict], margin_lt: Optional[float] = None) -> dict:
    """The original paper metric: re-feed flips the SIGN of the token's credit."""
    elig = [
        p for p in pivots
        if (margin_lt is None or p["margin"] < margin_lt)
        and (sign(p["A_exact1"]) != 0 or sign(p["A_refeed"]) != 0)
    ]
    floor_elig = [
        p for p in pivots
        if (margin_lt is None or p["margin"] < margin_lt)
        and (sign(p["A_exact1"]) != 0 or sign(p["A_exact2"]) != 0)
    ]
    refeed = sum(1 for p in elig if sign(p["A_exact1"]) != sign(p["A_refeed"]))
    floor = sum(1 for p in floor_elig if sign(p["A_exact1"]) != sign(p["A_exact2"]))
    rr, rf = _rate(refeed, len(elig)), _rate(floor, len(floor_elig))
    return {
        "eligible": len(elig),
        "refeed_flip_rate": rr,
        "replica_floor_rate": rf,
        "excess_pp": (None if rr is None or rf is None
                      else round((rr - rf) * 100, 2)),
    }


def analyze(run: dict, margin_split: float = 0.3) -> dict:
    pivots = run["pivots"]
    taus = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    return {
        "model": run.get("config", {}).get("model"),
        "n_pivots": len(pivots),
        "argmax_disagreement": {
            "all": argmax_disagreement(pivots),
            f"margin_lt_{margin_split}": argmax_disagreement(pivots, margin_split),
        },
        "credit_sign_flip": {
            "all": credit_sign_flip(pivots),
            f"margin_lt_{margin_split}": credit_sign_flip(pivots, margin_split),
        },
        "false_reject_sweep": false_reject_sweep(pivots, taus),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_json", help="a drift_ablation result json (e.g. primary.json)")
    ap.add_argument("--json-out", help="write the verification receipt here")
    ap.add_argument("--margin-split", type=float, default=0.3)
    ns = ap.parse_args(argv)

    with open(ns.run_json) as fh:
        run = json.load(fh)
    out = analyze(run, ns.margin_split)
    out["source"] = ns.run_json

    am = out["argmax_disagreement"]
    lo = am[f"margin_lt_{ns.margin_split}"]
    print(f"model: {out['model']}   pivots: {out['n_pivots']}")
    print("\nargmax disagreement (a re-feed verifier recomputes a DIFFERENT token "
          "than was generated):")
    print(f"  all pivots:        refeed {am['all']['refeed_disagreement_rate']:.1%}  "
          f"vs floor {am['all']['replica_floor_rate']:.1%}  "
          f"= +{am['all']['excess_pp']}pp")
    print(f"  low-margin (<{ns.margin_split}): refeed {lo['refeed_disagreement_rate']:.1%}  "
          f"vs floor {lo['replica_floor_rate']:.1%}  = +{lo['excess_pp']}pp")
    print("\nfalse-reject sweep (genuine token wrongly rejected by tolerance tau):")
    for row in out["false_reject_sweep"]:
        if row["refeed_false_reject_rate"] is None:
            continue
        print(f"  tau={row['tau_nats']:<5} refeed {row['refeed_false_reject_rate']:.1%}  "
              f"floor {row['replica_floor_rate']:.1%}  = +{row['excess_pp']}pp")

    if ns.json_out:
        with open(ns.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {ns.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
