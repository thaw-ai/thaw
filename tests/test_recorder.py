"""
Tests for thaw_vllm.recorder — shape-trace recorder.

CPU-only: the recorder is exercised against a fake EngineCore/scheduler
tree built from SimpleNamespace, mirroring the V1 SchedulerOutput field
layout (num_scheduled_tokens, total_num_scheduled_tokens,
scheduled_new_reqs, scheduled_cached_reqs). GPU behavior is exercised
by benchmarks/replay_experiment.py on a pod.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from thaw_vllm.recorder import Recorder, ShapeTrace, shape_signature  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def fake_output(num_scheduled, new_reqs=(), cached=None, total=None):
    return SimpleNamespace(
        num_scheduled_tokens=dict(num_scheduled),
        total_num_scheduled_tokens=(
            total if total is not None else sum(num_scheduled.values())
        ),
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=cached,
    )


def new_req(req_id, prompt_len, computed=0):
    return SimpleNamespace(
        req_id=req_id,
        num_computed_tokens=computed,
        prompt_token_ids=list(range(prompt_len)),
    )


def cached_reqs(pairs):
    """pairs: [(req_id, num_computed_tokens), ...]"""
    return SimpleNamespace(
        req_ids=[p[0] for p in pairs],
        num_computed_tokens=[p[1] for p in pairs],
    )


class FakeScheduler:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0

    def schedule(self):
        out = self._outputs[self.calls % len(self._outputs)]
        self.calls += 1
        return out


def fake_llm(scheduler):
    # llm.llm_engine.engine_core (InprocClient) .engine_core (real) .scheduler
    real_core = SimpleNamespace(scheduler=scheduler)
    inproc_client = SimpleNamespace(engine_core=real_core)
    return SimpleNamespace(llm_engine=SimpleNamespace(engine_core=inproc_client))


def drive(llm_obj, scheduler, n_steps):
    rec = Recorder(llm_obj)
    with rec:
        for _ in range(n_steps):
            scheduler.schedule()
    return rec.trace


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_start_stop_restores_original_schedule():
    sched = FakeScheduler([fake_output({"0": 1}, cached=cached_reqs([("0", 5)]))])
    llm = fake_llm(sched)
    original = sched.schedule
    rec = Recorder(llm)
    rec.start()
    assert sched.schedule != original
    rec.stop()
    # Bound methods compare by (__self__, __func__); identity can never
    # hold because each attribute access builds a fresh bound method.
    assert sched.schedule == original
    # And the override must be fully gone, not an instance attribute
    # shadowing the class method with equivalent behavior.
    assert "schedule" not in vars(sched)


def test_double_start_raises():
    sched = FakeScheduler([fake_output({"0": 1})])
    llm = fake_llm(sched)
    rec = Recorder(llm).start()
    try:
        import pytest

        with pytest.raises(RuntimeError):
            rec.start()
    finally:
        rec.stop()


def test_missing_scheduler_raises_with_guidance():
    import pytest

    bare = SimpleNamespace(
        llm_engine=SimpleNamespace(engine_core=SimpleNamespace())
    )
    with pytest.raises(RuntimeError, match="VLLM_ENABLE_V1_MULTIPROCESSING"):
        Recorder(bare)


# ---------------------------------------------------------------------------
# Recording semantics
# ---------------------------------------------------------------------------


def test_records_steps_prompt_lens_and_computed():
    # Step 1: new request "7" with a 100-token prompt, all scheduled.
    # Step 2: "7" decodes (computed=100).
    outputs = [
        fake_output({"7": 100}, new_reqs=[new_req("7", 100)]),
        fake_output({"7": 1}, cached=cached_reqs([("7", 100)])),
    ]
    sched = FakeScheduler(outputs)
    trace = drive(fake_llm(sched), sched, 2)

    assert len(trace.steps) == 2
    assert trace.req_meta["7"]["prompt_len"] == 100
    assert trace.steps[0]["reqs"]["7"] == [100, 0]
    assert trace.steps[0]["total"] == 100
    assert trace.steps[1]["reqs"]["7"] == [1, 100]


def test_idle_schedule_calls_not_recorded():
    sched = FakeScheduler([
        fake_output({"0": 1}, cached=cached_reqs([("0", 9)])),
        fake_output({}),
    ])
    trace = drive(fake_llm(sched), sched, 4)  # alternates busy/idle
    assert len(trace.steps) == 2


def test_token_content_never_recorded():
    sched = FakeScheduler([fake_output({"3": 8}, new_reqs=[new_req("3", 8)])])
    trace = drive(fake_llm(sched), sched, 1)
    flat = str(trace.to_dict())
    # The fake prompt is list(range(8)); its identity must not appear —
    # only its LENGTH. Spot-check that no token list survived.
    assert "[0, 1, 2" not in flat
    assert trace.req_meta["3"]["prompt_len"] == 8


# ---------------------------------------------------------------------------
# num_sampled / dummy_spec
# ---------------------------------------------------------------------------


def chunked_prefill_trace():
    """Request 'r': 1024-token prompt in two 512 chunks, then 3 decodes.
    Sampled tokens = 1 (final prefill chunk) + 3 (decodes) = 4."""
    t = ShapeTrace()
    t.note_request("r", 1024)
    t.add_step({"r": [512, 0]}, 512)
    t.add_step({"r": [512, 512]}, 512)
    for i in range(3):
        t.add_step({"r": [1, 1024 + i]}, 1)
    return t


def test_num_sampled_counts_final_chunk_and_decodes():
    assert chunked_prefill_trace().num_sampled("r") == 4


def test_num_sampled_fallback_without_computed_counts():
    t = ShapeTrace()
    t.note_request("r", 64)
    t.add_step({"r": [64, -1]}, 64)
    for _ in range(5):
        t.add_step({"r": [1, -1]}, 1)
    # Fallback counts plain decode steps only (documented off-by-one on
    # the final prefill chunk when computed counts are unavailable).
    assert t.num_sampled("r") == 5


def test_dummy_spec_excludes_target_and_derives_lengths():
    t = ShapeTrace()
    t.note_request("target", 50)
    t.note_request("n1", 200)
    t.add_step({"target": [50, 0], "n1": [200, 0]}, 250)
    t.add_step({"target": [1, 50], "n1": [1, 200]}, 2)
    t.add_step({"n1": [1, 201]}, 1)

    spec = t.dummy_spec(exclude={"target"})
    assert spec == [{"req_id": "n1", "prompt_len": 200, "num_sampled": 3}]


# ---------------------------------------------------------------------------
# Signature + serialization
# ---------------------------------------------------------------------------


def test_signature_invariant_to_req_id_renaming():
    a = ShapeTrace()
    a.add_step({"0": [50, 0], "1": [200, 0]}, 250)
    a.add_step({"0": [1, 50], "1": [1, 200]}, 2)
    b = ShapeTrace()
    b.add_step({"14": [200, 0], "15": [50, 0]}, 250)
    b.add_step({"15": [1, 50], "14": [1, 200]}, 2)
    assert shape_signature(a) == shape_signature(b)


def test_signature_pins_target_slot():
    a = ShapeTrace()
    a.add_step({"t": [50, 0], "n": [200, 0]}, 250)
    b = ShapeTrace()
    b.add_step({"t": [200, 0], "n": [50, 0]}, 250)
    # Untagged: same multiset, equal.
    assert shape_signature(a) == shape_signature(b)
    # Tagged: the target occupies a different shape slot — unequal.
    assert shape_signature(a, target="t") != shape_signature(b, target="t")


def test_save_load_round_trip(tmp_path):
    t = chunked_prefill_trace()
    t.header = {"model": "fake", "vllm_version": None}
    path = str(tmp_path / "trace.json")
    n_bytes = t.save(path)
    assert n_bytes == t.certificate_bytes()

    loaded = ShapeTrace.load(path)
    assert loaded.header == t.header
    assert loaded.steps == t.steps
    assert loaded.req_meta == t.req_meta
    assert shape_signature(loaded) == shape_signature(t)


def test_certificate_is_kilobyte_scale_for_1k_tokens():
    """The headline size claim, pinned: a 1k-decode-step trace with 8
    co-resident requests serializes to tens of KB, not MB."""
    t = ShapeTrace()
    for r in range(8):
        t.note_request(str(r), 512)
    for i in range(1000):
        t.add_step({str(r): [1, 512 + i] for r in range(8)}, 8)
    assert t.certificate_bytes() < 200_000
