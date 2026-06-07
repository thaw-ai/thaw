"""thaw_vllm.rewind — laptop-side inspection of RL / tree-search rollouts.

Where ``agentfs`` reads a frozen *session* (KV blocks + lineage), ``rewind``
reads a *rollout*: the tokens a fork generated, each with the logprob the
model assigned it. That is the missing piece for RL pivot resampling — you
fork a session at a pivot, branch N continuations, and want to see *where
the trajectories diverge in logprob space* and which branch the model was
most confident in.

A rollout is a plain ``rollout.json`` file (schema below). Reading, diffing,
and ranking rollouts needs no GPU, no torch, no vLLM — same discipline as
``agentfs``: this module is stdlib-only. The capture step (turning a vLLM
``CompletionOutput`` into a rollout) is the only GPU-adjacent part and lives
in ``fork.capture_rollouts``; the pure extraction helper here is duck-typed
so importing this module never pulls the engine in.

rollout.json schema (version 1)::

    {
      "version": 1,
      "rollout_id": "hex",
      "parent_id": "<fork handle id this branched from> | null",
      "label": "branch-a",
      "model_id": "meta-llama/Llama-3.1-8B-Instruct",
      "prompt_token_ids": [...],   # the shared trunk up to the fork point
      "fork_index": 195,           # len(prompt_token_ids); where gen starts
      "sampling": {"temperature": 0.8, "top_p": 0.95, "seed": 42},
      "text": "the full decoded continuation",
      "tokens": [
        {"token_id": 791, "text": " The", "logprob": -0.20,
         "topk": [{"token_id": 791, "text": " The", "logprob": -0.20},
                  {"token_id": 362, "text": " A",   "logprob": -1.90}]},
        ...
      ],
      "created_at": 1781234567.0
    }
"""

from __future__ import annotations

import json
import math
import os
import uuid
from typing import Optional

from thaw_vllm.agentfs import _fmt_int, _fmt_time

ROLLOUT_FILENAME = "rollout.json"
ROLLOUT_VERSION = 1


# ---------------------------------------------------------------------------
# capture: vLLM CompletionOutput -> plain token/logprob records (duck-typed)
# ---------------------------------------------------------------------------


def _logprob_value(obj) -> Optional[float]:
    """A vLLM ``Logprob`` exposes ``.logprob``; tolerate a bare float too."""
    if obj is None:
        return None
    val = getattr(obj, "logprob", obj)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def extract_token_logprobs(completion_output, max_topk: int = 5) -> list[dict]:
    """Turn a vLLM ``CompletionOutput`` into the rollout ``tokens`` list.

    Duck-typed: reads ``.token_ids`` and ``.logprobs`` (the per-step
    ``dict[token_id -> Logprob]`` vLLM fills when ``SamplingParams.logprobs``
    is set). No torch/vLLM import, so this stays laptop-safe; it just walks
    attributes. Tokens with no logprob data still record their ``token_id``.
    """
    token_ids = list(getattr(completion_output, "token_ids", []) or [])
    step_logprobs = getattr(completion_output, "logprobs", None) or []
    tokens: list[dict] = []
    for i, tid in enumerate(token_ids):
        tid = int(tid)
        entry: dict = {"token_id": tid, "text": None, "logprob": None, "topk": []}
        if i < len(step_logprobs) and step_logprobs[i]:
            step = step_logprobs[i]
            chosen = step.get(tid)
            if chosen is not None:
                entry["logprob"] = _logprob_value(chosen)
                entry["text"] = getattr(chosen, "decoded_token", None)
            ranked = sorted(
                step.items(), key=lambda kv: _logprob_value(kv[1]) or -1e30, reverse=True
            )[:max_topk]
            entry["topk"] = [
                {
                    "token_id": int(k),
                    "text": getattr(v, "decoded_token", None),
                    "logprob": _logprob_value(v),
                }
                for k, v in ranked
            ]
        tokens.append(entry)
    return tokens


def build_rollout(
    *,
    model_id: str,
    prompt_token_ids: list,
    tokens: list,
    fork_index: Optional[int] = None,
    parent_id: Optional[str] = None,
    label: Optional[str] = None,
    sampling: Optional[dict] = None,
    text: Optional[str] = None,
    rollout_id: Optional[str] = None,
    created_at: Optional[float] = None,
) -> dict:
    """Assemble a rollout record dict from plain data (no engine needed)."""
    prompt_token_ids = list(prompt_token_ids or [])
    if text is None:
        text = "".join(t.get("text") or "" for t in tokens)
    return {
        "version": ROLLOUT_VERSION,
        "rollout_id": rollout_id or uuid.uuid4().hex,
        "parent_id": parent_id,
        "label": label,
        "model_id": model_id,
        "prompt_token_ids": prompt_token_ids,
        "fork_index": fork_index if fork_index is not None else len(prompt_token_ids),
        "sampling": sampling or {},
        "text": text,
        "tokens": tokens,
        "created_at": created_at if created_at is not None else 0.0,
    }


def write_rollout(record: dict, target_dir: str) -> str:
    """Write ``record`` to ``target_dir/rollout.json``. Returns the path."""
    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, ROLLOUT_FILENAME)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def _resolve_rollout(path: str) -> str:
    path = os.path.abspath(path)
    if os.path.isdir(path):
        candidate = os.path.join(path, ROLLOUT_FILENAME)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"No {ROLLOUT_FILENAME} in {path}")
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"{path!r} is not a rollout (no {ROLLOUT_FILENAME})")


def summarize_rollout(path: str) -> dict:
    """Parse a rollout.json and compute its scores. GPU-free; stdlib only."""
    rp = _resolve_rollout(path)
    with open(rp) as f:
        r = json.load(f)

    tokens = r.get("tokens", []) or []
    lps = [t["logprob"] for t in tokens if t.get("logprob") is not None]
    seq_logprob = sum(lps) if lps else None
    mean_logprob = (seq_logprob / len(lps)) if lps else None
    perplexity = math.exp(-mean_logprob) if mean_logprob is not None else None

    return {
        "path": os.path.dirname(rp),
        "name": os.path.basename(os.path.dirname(rp)) or os.path.basename(rp),
        "rollout_id": r.get("rollout_id", ""),
        "parent_id": r.get("parent_id"),
        "label": r.get("label"),
        "model_id": r.get("model_id", "?"),
        "prompt_token_ids": list(r.get("prompt_token_ids", []) or []),
        "fork_index": r.get("fork_index", 0),
        "sampling": r.get("sampling", {}) or {},
        "text": r.get("text", "") or "",
        "tokens": tokens,
        "n_tokens": len(tokens),
        "n_scored": len(lps),
        "seq_logprob": seq_logprob,
        "mean_logprob": mean_logprob,
        "perplexity": perplexity,
        "created_at": r.get("created_at", 0.0),
    }


def _fmt_lp(x) -> str:
    return f"{x:.2f}" if isinstance(x, (int, float)) else "n/a"


def _fmt_sampling(s: dict) -> str:
    if not s:
        return "unknown"
    bits = []
    for k in ("temperature", "top_p", "top_k", "seed"):
        if k in s and s[k] is not None:
            short = {"temperature": "temp", "top_p": "top_p", "top_k": "top_k", "seed": "seed"}[k]
            bits.append(f"{short} {s[k]}")
    return " · ".join(bits) if bits else "unknown"


# ---------------------------------------------------------------------------
# render: inspect
# ---------------------------------------------------------------------------


def inspect_rollout(path: str) -> str:
    s = summarize_rollout(path)
    lines = [f"thaw rollout  {s['label'] or s['name']}"]

    def row(label, value):
        lines.append(f"  {label:<13} {value}")

    row("model", s["model_id"])
    if s["rollout_id"]:
        row("id", s["rollout_id"][:12])
    if s["parent_id"]:
        row("forked from", s["parent_id"][:12])
    row("trunk", f"{_fmt_int(len(s['prompt_token_ids']))} prompt tokens")
    row("generated", f"{_fmt_int(s['n_tokens'])} tokens")
    if s["seq_logprob"] is not None:
        row("seq logprob", _fmt_lp(s["seq_logprob"]))
        row("mean logprob", f"{s['mean_logprob']:.3f}")
        row("perplexity", f"{s['perplexity']:.2f}")
    else:
        row("logprobs", "not captured (re-run capture with logprobs enabled)")
    row("sampling", _fmt_sampling(s["sampling"]))
    if s["created_at"]:
        row("created", _fmt_time(s["created_at"]))
    if s["text"]:
        preview = s["text"][:220].replace("\n", " ")
        row("text", f"\"{preview}\"")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# render: diff  (the pivot — where two rollouts split, in logprob space)
# ---------------------------------------------------------------------------


def _first_divergence(ta: list, tb: list) -> int:
    """Index of the first generated token where chosen ids differ; len if none."""
    n = min(len(ta), len(tb))
    for i in range(n):
        if ta[i].get("token_id") != tb[i].get("token_id"):
            return i
    return n


def _topk_lookup(token_entry: dict, token_id: int):
    """Return (rank, logprob) for ``token_id`` in this step's top-k, or None."""
    for rank, alt in enumerate(token_entry.get("topk", [])):
        if alt.get("token_id") == token_id:
            return rank, alt.get("logprob")
    return None


def _shared_tail(tokens: list, upto: int, n: int = 6) -> str:
    start = max(0, upto - n)
    return "".join(t.get("text") or f"⟨{t.get('token_id')}⟩" for t in tokens[start:upto])


def diff_rollouts(path_a: str, path_b: str) -> str:
    a = summarize_rollout(path_a)
    b = summarize_rollout(path_b)
    ta, tb = a["tokens"], b["tokens"]

    lines = [
        "thaw rewind diff",
        f"  A  {a['label'] or a['name']}  ({(a['rollout_id'] or '')[:8]})",
        f"  B  {b['label'] or b['name']}  ({(b['rollout_id'] or '')[:8]})",
        "",
    ]

    def row(label, value):
        lines.append(f"  {label:<13} {value}")

    # model + trunk
    if a["model_id"] == b["model_id"]:
        row("model", f"same  ({a['model_id']})")
    else:
        row("model", f"DIFFER  A={a['model_id']}  B={b['model_id']}")

    if a["prompt_token_ids"] == b["prompt_token_ids"]:
        row("trunk", f"same  ({_fmt_int(len(a['prompt_token_ids']))} shared prompt tokens)")
    else:
        common = _first_divergence(
            [{"token_id": t} for t in a["prompt_token_ids"]],
            [{"token_id": t} for t in b["prompt_token_ids"]],
        )
        row("trunk", f"DIFFER  branches grew from different prompts (first {_fmt_int(common)} prompt tokens shared)")

    # scores
    row("length", f"A {_fmt_int(a['n_tokens'])} tok · B {_fmt_int(b['n_tokens'])} tok")
    if a["seq_logprob"] is not None and b["seq_logprob"] is not None:
        gap = a["seq_logprob"] - b["seq_logprob"]
        winner = "A" if gap > 0 else "B"
        row("seq logprob", f"A {_fmt_lp(a['seq_logprob'])} · B {_fmt_lp(b['seq_logprob'])}   ({winner} higher by {abs(gap):.2f})")
        row("perplexity", f"A {_fmt_lp(a['perplexity'])} · B {_fmt_lp(b['perplexity'])}")
    else:
        row("seq logprob", "unavailable (one or both rollouts have no logprobs)")

    # the pivot
    piv = _first_divergence(ta, tb)
    lines.append("")
    if piv >= min(len(ta), len(tb)):
        if len(ta) == len(tb):
            row("pivot", "none — identical token sequence")
        else:
            row("pivot", f"shared prefix, then one continues ({_fmt_int(abs(len(ta) - len(tb)))} tokens longer)")
        return "\n".join(lines)

    row("pivot", f"generated token {_fmt_int(piv)}  (after {_fmt_int(piv)} identical tokens)")
    tail = _shared_tail(ta, piv)
    if tail:
        lines.append(f"    shared   …{tail.replace(chr(10), ' ')}")

    at, bt = ta[piv], tb[piv]
    a_tok = (at.get("text") or f"⟨{at.get('token_id')}⟩").replace("\n", "\\n")
    b_tok = (bt.get("text") or f"⟨{bt.get('token_id')}⟩").replace("\n", "\\n")

    # counterfactual: what logprob did each branch assign the other's choice?
    a_cf = _topk_lookup(at, bt.get("token_id"))
    b_cf = _topk_lookup(bt, at.get("token_id"))
    a_extra = f"   (B ranked this #{b_cf[0] + 1} @ {_fmt_lp(b_cf[1])})" if b_cf else ""
    b_extra = f"   (A ranked this #{a_cf[0] + 1} @ {_fmt_lp(a_cf[1])})" if a_cf else ""

    lines.append(f"  - A  \"{a_tok}\"   logprob {_fmt_lp(at.get('logprob'))}{a_extra}")
    lines.append(f"  + B  \"{b_tok}\"   logprob {_fmt_lp(bt.get('logprob'))}{b_extra}")

    # how decisive was the split — gap between the two choices on A's own scale
    if at.get("logprob") is not None and a_cf and a_cf[1] is not None:
        margin = at["logprob"] - a_cf[1]
        verdict = "a close call" if margin < 1.0 else "a confident split" if margin < 3.0 else "a decisive split"
        row("decisiveness", f"A preferred its token by {margin:.2f} logprob — {verdict}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# render: pivot  (across N rollouts sharing a trunk — find the fork + winner)
# ---------------------------------------------------------------------------


def _collect_rollout_dirs(root: str) -> list:
    root = os.path.abspath(root)
    if os.path.isfile(os.path.join(root, ROLLOUT_FILENAME)):
        return [root]
    out = []
    if os.path.isdir(root):
        for entry in sorted(os.listdir(root)):
            p = os.path.join(root, entry)
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, ROLLOUT_FILENAME)):
                out.append(p)
    return out


def pivot_rollouts(root: str) -> str:
    """Across N rollouts sharing a trunk, find the earliest token where they
    stop agreeing, group branches by what they chose there, and rank by
    sequence logprob. This is 'fork a rollout, keep the winner', made
    inspectable. GPU-free.
    """
    dirs = _collect_rollout_dirs(root)
    if not dirs:
        return f"thaw rewind pivot  {os.path.abspath(root)}\n  (no rollouts found)"

    rs = []
    for d in dirs:
        try:
            rs.append(summarize_rollout(d))
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
    if len(rs) < 2:
        return (
            f"thaw rewind pivot  {os.path.abspath(root)}\n"
            f"  need at least 2 rollouts to find a pivot (found {len(rs)})"
        )

    model = rs[0]["model_id"]
    trunk_len = len(rs[0]["prompt_token_ids"])
    lines = [
        f"thaw rewind pivot  {os.path.abspath(root)}",
        f"  {_fmt_int(len(rs))} rollouts · trunk {_fmt_int(trunk_len)} tokens · {model}",
    ]

    # earliest generated index where the chosen tokens are not unanimous
    shortest = min(r["n_tokens"] for r in rs)
    pivot = shortest
    for i in range(shortest):
        ids = {r["tokens"][i].get("token_id") for r in rs}
        if len(ids) > 1:
            pivot = i
            break

    if pivot >= shortest:
        lines.append(f"  no divergence within the first {_fmt_int(shortest)} shared tokens")
    else:
        lines.append(f"  first divergence at generated token {_fmt_int(pivot)}:")
        groups: dict = {}
        for r in rs:
            tok = r["tokens"][pivot]
            key = tok.get("token_id")
            groups.setdefault(key, {"text": tok.get("text"), "members": []})
            groups[key]["members"].append(r)
        # order groups by best member score
        def grp_best(g):
            scored = [m["seq_logprob"] for m in g["members"] if m["seq_logprob"] is not None]
            return max(scored) if scored else -1e30

        for key, g in sorted(groups.items(), key=lambda kv: grp_best(kv[1]), reverse=True):
            tok_text = (g["text"] or f"⟨{key}⟩").replace("\n", "\\n")
            names = ", ".join((m["label"] or m["name"]) for m in g["members"])
            scores = ", ".join(_fmt_lp(m["seq_logprob"]) for m in g["members"])
            lines.append(f"    \"{tok_text}\"  →  {names}   (seq logprob {scores})")

    # winner overall
    scored = [r for r in rs if r["seq_logprob"] is not None]
    if scored:
        best = max(scored, key=lambda r: r["seq_logprob"])
        lines.append("")
        lines.append(
            f"  best branch: {best['label'] or best['name']}  "
            f"(seq logprob {_fmt_lp(best['seq_logprob'])}, perplexity {_fmt_lp(best['perplexity'])})"
        )
    else:
        lines.append("")
        lines.append("  ranking unavailable — rollouts carry no logprobs")
    return "\n".join(lines)
