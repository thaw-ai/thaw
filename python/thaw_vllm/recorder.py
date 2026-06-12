"""
thaw_vllm.recorder — shape-trace recorder for retroactive replay.

The flight-recorder hypothesis (private/plans/2026-06-12_flight-recorder-
experiment-plan.md): the kernel-facing batch SHAPE at each engine step,
plus static engine config, is sufficient to retroactively reproduce a
past generation bit-exactly. Token identities of co-batched neighbors
do not matter (Cankaya, arXiv 2606.00279, Section 3); only the shapes
they induce do.

This module records exactly that and nothing else:

  - per engine step: for each scheduled request, (tokens scheduled this
    step, tokens computed before this step) — both integers
  - per request, once: its prompt length
  - one static header: engine version, model, dtype, TP size, block
    size, device name

No token content, no activations, no text ever enters the trace. A
1k-token generation under load records on the order of kilobytes.

Usage::

    from thaw_vllm.recorder import Recorder

    rec = Recorder(llm)
    with rec:
        outs = llm.generate(prompts, sampling_params)
    rec.trace.save("trace.json")

    spec = rec.trace.dummy_spec(exclude={outs[0].request_id})
    # -> [(prompt_len, num_sampled), ...] for every other request:
    #    enough to rebuild shape-identical dummy load for replay.

Requires V1 in-process mode (VLLM_ENABLE_V1_MULTIPROCESSING=0) — the
recorder wraps ``scheduler.schedule()`` on the in-process EngineCore,
the same access path as kv_snapshot. Hook overhead is a few dict
writes per engine step.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

__all__ = ["Recorder", "ShapeTrace", "shape_signature"]


# ---------------------------------------------------------------------------
# Trace container
# ---------------------------------------------------------------------------


class ShapeTrace:
    """A recorded shape trace: static header + one record per engine step.

    ``steps`` is a list of dicts ``{"reqs": {req_id: [scheduled,
    computed_before]}, "total": int}``. ``req_meta`` maps req_id ->
    ``{"prompt_len": int, "first_step": int}``. Both are plain JSON
    types throughout so save/load is trivial and the certificate size
    is honest (``certificate_bytes`` is just the serialized length).
    """

    def __init__(self, header: Optional[dict] = None) -> None:
        self.header: dict = header or {}
        self.steps: list[dict] = []
        self.req_meta: dict[str, dict] = {}

    # -- recording ----------------------------------------------------------

    def add_step(self, reqs: dict[str, list[int]], total: int) -> None:
        self.steps.append({"reqs": reqs, "total": total})

    def note_request(self, req_id: str, prompt_len: int) -> None:
        if req_id not in self.req_meta:
            self.req_meta[req_id] = {
                "prompt_len": int(prompt_len),
                "first_step": len(self.steps),
            }

    # -- queries ------------------------------------------------------------

    def req_ids(self) -> list[str]:
        return list(self.req_meta.keys())

    def steps_for(self, req_id: str) -> list[tuple[int, int, int]]:
        """[(step_index, scheduled, computed_before), ...] for one request."""
        out = []
        for i, step in enumerate(self.steps):
            entry = step["reqs"].get(req_id)
            if entry is not None:
                out.append((i, entry[0], entry[1]))
        return out

    def num_sampled(self, req_id: str) -> int:
        """Number of tokens the engine sampled for this request.

        Every scheduled step that REACHES the end of the known prompt
        samples one token: each decode step (scheduled == 1 past the
        prompt) and the final prefill chunk. Equivalently: count steps
        where computed_before + scheduled > prompt_len - 1 is not quite
        right under chunked prefill, so we count directly: a step
        samples iff computed_before + scheduled >= prompt_len.
        """
        meta = self.req_meta.get(req_id)
        if meta is None:
            return 0
        prompt_len = meta["prompt_len"]
        n = 0
        for _, scheduled, computed in self.steps_for(req_id):
            if computed >= 0:
                if computed + scheduled >= prompt_len:
                    n += 1
            elif scheduled == 1:
                # Fallback when this vLLM version did not expose
                # computed-token counts for cached requests: count
                # plain decode steps. Misses only the first-token
                # sample of the final prefill chunk, which the caller
                # can see as an off-by-one in trace-vs-trace compare.
                n += 1
        return n

    def dummy_spec(self, exclude: set[str] | None = None) -> list[dict]:
        """Reconstruction spec for shape-identical dummy load.

        For every request not in ``exclude``, returns
        ``{"prompt_len": int, "num_sampled": int}``. Submitting dummy
        requests with junk prompts of ``prompt_len`` tokens and
        ``max_tokens=num_sampled`` (with ignore_eos) alongside the
        target request reproduces the recorded shape trace, assuming
        deterministic scheduling for identical lengths and arrival
        order — which is exactly what the replay experiment verifies
        by re-recording and comparing signatures.
        """
        exclude = exclude or set()
        spec = []
        for req_id in self.req_ids():
            if req_id in exclude:
                continue
            spec.append({
                "req_id": req_id,
                "prompt_len": self.req_meta[req_id]["prompt_len"],
                "num_sampled": self.num_sampled(req_id),
            })
        return spec

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": "thaw-shape-trace-v0",
            "header": self.header,
            "req_meta": self.req_meta,
            "steps": self.steps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ShapeTrace":
        t = cls(header=d.get("header") or {})
        t.req_meta = d.get("req_meta") or {}
        t.steps = d.get("steps") or []
        return t

    def save(self, path: str) -> int:
        """Write the trace as JSON; returns the certificate size in bytes."""
        data = json.dumps(self.to_dict(), separators=(",", ":"))
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(data)
        os.replace(tmp, path)
        return len(data)

    @classmethod
    def load(cls, path: str) -> "ShapeTrace":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def certificate_bytes(self) -> int:
        return len(json.dumps(self.to_dict(), separators=(",", ":")))


def shape_signature(trace: ShapeTrace, target: Optional[str] = None) -> list:
    """Content-free, req-id-free signature of a trace for comparison.

    Request ids are assigned by a monotonically increasing engine
    counter, so identical loads recorded in two different runs carry
    different ids. The signature erases naming: per step, the sorted
    multiset of (scheduled, computed_before) pairs plus the total. If
    ``target`` is given, that request's pair is tagged so its position
    in the shape is pinned, not just present.
    """
    sig = []
    for step in trace.steps:
        anon = []
        for req_id, (scheduled, computed) in step["reqs"].items():
            tag = "T" if (target is not None and req_id == target) else ""
            anon.append((scheduled, computed, tag))
        sig.append((sorted(anon), step["total"]))
    return sig


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class Recorder:
    """Wraps ``scheduler.schedule()`` on a live LLM and records shapes.

    Context manager: entering installs the wrapper, exiting restores
    the original method. The wrapper reads only scalar fields off the
    SchedulerOutput the scheduler already produced — no extra GPU work,
    no token content.
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self._scheduler = self._find_scheduler(llm)
        self._original_schedule = None
        self.trace = ShapeTrace(header=self._build_header(llm))

    # -- wiring -------------------------------------------------------------

    @staticmethod
    def _find_scheduler(llm: Any):
        # Same navigation as kv_snapshot._get_engine_core, inlined so
        # this module stays importable without torch installed (the
        # trace tooling runs laptop-side on saved traces).
        ec = llm.llm_engine.engine_core
        if hasattr(ec, "engine_core"):
            # InprocClient wraps the real EngineCore.
            ec = ec.engine_core
        scheduler = getattr(ec, "scheduler", None)
        if scheduler is None or not callable(
            getattr(scheduler, "schedule", None)
        ):
            raise RuntimeError(
                "Recorder needs the in-process V1 EngineCore scheduler "
                "(set VLLM_ENABLE_V1_MULTIPROCESSING=0 before engine "
                "construction); could not find scheduler.schedule() on "
                f"{type(ec).__name__}"
            )
        return scheduler

    @staticmethod
    def _build_header(llm: Any) -> dict:
        """Best-effort static half of the certificate. Every field is
        getattr-guarded: a missing field is recorded as None, never an
        error — the header informs, the steps decide."""
        header: dict = {}
        try:
            import vllm

            header["vllm_version"] = getattr(vllm, "__version__", None)
        except Exception:
            header["vllm_version"] = None
        try:
            import torch

            header["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                header["device_name"] = torch.cuda.get_device_name(0)
                header["cuda_version"] = torch.version.cuda
        except Exception:
            pass
        for attr_path, key in (
            ("model_config.model", "model"),
            ("model_config.dtype", "dtype"),
            ("cache_config.block_size", "block_size"),
            ("parallel_config.tensor_parallel_size", "tensor_parallel_size"),
        ):
            obj = getattr(llm, "llm_engine", None)
            cfg = getattr(obj, "vllm_config", obj)
            for part in attr_path.split("."):
                cfg = getattr(cfg, part, None)
                if cfg is None:
                    break
            header[key] = str(cfg) if cfg is not None else None
        return header

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "Recorder":
        if self._original_schedule is not None:
            raise RuntimeError("Recorder already started")
        original = self._scheduler.schedule
        trace = self.trace

        def recording_schedule(*args, **kwargs):
            output = original(*args, **kwargs)
            _record_step(trace, output)
            return output

        self._scheduler.schedule = recording_schedule
        self._original_schedule = original
        return self

    def stop(self) -> ShapeTrace:
        if self._original_schedule is not None:
            cls_method = getattr(type(self._scheduler), "schedule", None)
            if (
                cls_method is not None
                and getattr(self._original_schedule, "__func__", None)
                is cls_method
            ):
                # The original was the plain class method: deleting our
                # instance-level override restores pristine state, with
                # no lingering instance attribute shadowing the class.
                try:
                    delattr(self._scheduler, "schedule")
                except AttributeError:
                    self._scheduler.schedule = self._original_schedule
            else:
                self._scheduler.schedule = self._original_schedule
            self._original_schedule = None
        return self.trace

    def __enter__(self) -> "Recorder":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _record_step(trace: ShapeTrace, output: Any) -> None:
    """Extract the shape record from one SchedulerOutput.

    Field access is getattr-guarded against vLLM version drift: the
    essential record (num_scheduled_tokens + total) comes from fields
    that have been stable across V1; prompt lengths and computed-token
    counts come from the new/cached request metadata where available.
    An empty step (idle schedule call) is not recorded.
    """
    num_scheduled = getattr(output, "num_scheduled_tokens", None)
    if not num_scheduled:
        return
    total = getattr(output, "total_num_scheduled_tokens", None)
    if total is None:
        total = sum(num_scheduled.values())

    # Computed-before counts, where the metadata exposes them.
    computed: dict[str, int] = {}
    for new_req in getattr(output, "scheduled_new_reqs", None) or []:
        req_id = getattr(new_req, "req_id", None)
        if req_id is None:
            continue
        computed[req_id] = int(getattr(new_req, "num_computed_tokens", 0) or 0)
        prompt_ids = getattr(new_req, "prompt_token_ids", None)
        if prompt_ids is not None:
            # Length only — token content never enters the trace.
            trace.note_request(req_id, len(prompt_ids))
    cached = getattr(output, "scheduled_cached_reqs", None)
    if cached is not None:
        cached_ids = getattr(cached, "req_ids", None) or []
        cached_computed = getattr(cached, "num_computed_tokens", None) or []
        for req_id, n in zip(cached_ids, cached_computed):
            computed[req_id] = int(n)

    reqs = {
        str(req_id): [int(n), computed.get(req_id, -1)]
        for req_id, n in num_scheduled.items()
    }
    trace.add_step(reqs, int(total))
