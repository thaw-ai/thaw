"""
Tests for thaw_vllm.sleep_mode — vLLM sleep/wake backend wrapper.

CPU-only, mocked. Real GPU behavior is exercised by
demos/sleep_mode_demo.py on an H100 pod — that run produces the
bit-identity receipt cited in the vLLM RFC #34303 comment.

Coverage
--------
  - sleep() composes freeze_model_tp with vLLM's native llm.sleep(level=2).
  - strict=True raises SleepModeUnavailableError when enable_sleep_mode=False.
  - strict=False runs as freeze-only (snapshot lands, GPU not freed).
  - wake_up() composes llm.wake_up() with restore_model_tp.
  - Stats dict carries sleep/wake path + GPU-delta bookkeeping.
  - sleep_error is captured in stats if llm.sleep() raises.
"""
from unittest.mock import MagicMock, patch

import pytest


def _make_llm(enable_sleep_mode: bool = True):
    """Build a fake vLLM instance with a plausible config path."""
    llm = MagicMock()
    llm.llm_engine.vllm_config.parallel_config.tensor_parallel_size = 1
    llm.llm_engine.vllm_config.model_config.enable_sleep_mode = enable_sleep_mode
    llm.sleep = MagicMock()
    llm.wake_up = MagicMock()
    return llm


@pytest.fixture
def fake_llm_sleep_enabled():
    return _make_llm(enable_sleep_mode=True)


@pytest.fixture
def fake_llm_sleep_disabled():
    return _make_llm(enable_sleep_mode=False)


def test_sleep_composes_freeze_with_vllm_sleep(fake_llm_sleep_enabled):
    from thaw_vllm import sleep_mode

    fake_stats = {
        "num_regions": 195,
        "total_bytes": 16_060_000_000,
        "elapsed_s": 1.7,
        "throughput_gb_s": 9.44,
        "tensor_parallel_size": 1,
        "per_rank": [{"rank": 0}],
    }

    with patch("thaw_vllm.sleep_mode.freeze_model_tp",
               return_value=fake_stats) as mock_freeze:
        stats = sleep_mode.sleep(fake_llm_sleep_enabled, "/tmp/test.thaw")

    mock_freeze.assert_called_once_with(fake_llm_sleep_enabled, "/tmp/test.thaw")
    fake_llm_sleep_enabled.sleep.assert_called_once_with(level=2)
    assert stats["sleep_path"] == "/tmp/test.thaw"
    assert stats["sleep_level"] == 2
    assert stats["freed_gpu_memory"] is True
    assert stats["num_regions"] == 195
    assert "gpu_bytes_before_sleep" in stats
    assert "gpu_bytes_after_sleep" in stats
    assert "gpu_bytes_freed" in stats


def test_sleep_strict_raises_when_sleep_mode_disabled(fake_llm_sleep_disabled):
    from thaw_vllm import sleep_mode

    with patch("thaw_vllm.sleep_mode.freeze_model_tp") as mock_freeze:
        with pytest.raises(sleep_mode.SleepModeUnavailableError):
            sleep_mode.sleep(fake_llm_sleep_disabled, "/tmp/test.thaw")

    mock_freeze.assert_not_called()
    fake_llm_sleep_disabled.sleep.assert_not_called()


def test_sleep_non_strict_runs_freeze_only(fake_llm_sleep_disabled):
    """strict=False means we snapshot the weights but skip llm.sleep() — the
    caller gets a durable checkpoint without the GPU memory being freed."""
    from thaw_vllm import sleep_mode

    fake_stats = {
        "num_regions": 1, "total_bytes": 100,
        "elapsed_s": 0.1, "throughput_gb_s": 1.0,
        "tensor_parallel_size": 1, "per_rank": [],
    }

    with patch("thaw_vllm.sleep_mode.freeze_model_tp",
               return_value=fake_stats) as mock_freeze:
        stats = sleep_mode.sleep(
            fake_llm_sleep_disabled, "/tmp/x.thaw", strict=False,
        )

    mock_freeze.assert_called_once()
    fake_llm_sleep_disabled.sleep.assert_not_called()
    assert stats["freed_gpu_memory"] is False
    assert stats["sleep_path"] == "/tmp/x.thaw"


def test_sleep_captures_vllm_sleep_error(fake_llm_sleep_enabled):
    """If llm.sleep() raises, the freeze already wrote the snapshot — the
    error is attached to stats but we don't re-raise."""
    from thaw_vllm import sleep_mode

    fake_llm_sleep_enabled.sleep.side_effect = RuntimeError("nope")

    with patch("thaw_vllm.sleep_mode.freeze_model_tp",
               return_value={"num_regions": 1, "total_bytes": 100,
                             "elapsed_s": 0.1, "throughput_gb_s": 1.0,
                             "tensor_parallel_size": 1, "per_rank": []}):
        stats = sleep_mode.sleep(fake_llm_sleep_enabled, "/tmp/err.thaw")

    assert stats["freed_gpu_memory"] is False
    assert "sleep_error" in stats
    assert "nope" in stats["sleep_error"]
    assert stats["sleep_path"] == "/tmp/err.thaw"


def test_sleep_custom_level(fake_llm_sleep_enabled):
    from thaw_vllm import sleep_mode

    with patch("thaw_vllm.sleep_mode.freeze_model_tp",
               return_value={"num_regions": 0, "total_bytes": 0,
                             "elapsed_s": 0, "throughput_gb_s": 0,
                             "tensor_parallel_size": 1, "per_rank": []}):
        stats = sleep_mode.sleep(fake_llm_sleep_enabled, "/tmp/y.thaw", level=1)

    fake_llm_sleep_enabled.sleep.assert_called_once_with(level=1)
    assert stats["sleep_level"] == 1


def test_wake_up_composes_vllm_wake_with_restore(fake_llm_sleep_enabled):
    from thaw_vllm import sleep_mode

    fake_stats = {
        "num_regions": 195,
        "total_bytes": 16_060_000_000,
        "elapsed_s": 2.6,
        "throughput_gb_s": 6.18,
        "tensor_parallel_size": 1,
        "per_rank": [{"rank": 0}],
    }

    with patch("thaw_vllm.sleep_mode.restore_model_tp",
               return_value=fake_stats) as mock_restore:
        stats = sleep_mode.wake_up(fake_llm_sleep_enabled, "/tmp/test.thaw")

    fake_llm_sleep_enabled.wake_up.assert_called_once_with()
    mock_restore.assert_called_once_with(
        fake_llm_sleep_enabled, "/tmp/test.thaw", chunk_size_mb=64,
    )
    assert stats["wake_path"] == "/tmp/test.thaw"
    assert stats["vllm_wake_up_called"] is True
    assert "gpu_bytes_before_wake" in stats
    assert "gpu_bytes_after_wake" in stats


def test_wake_up_skips_vllm_wake_when_sleep_mode_disabled(fake_llm_sleep_disabled):
    """Freeze-only sleep means the GPU tensors were never freed — we go
    straight to restore_model_tp without calling llm.wake_up()."""
    from thaw_vllm import sleep_mode

    with patch("thaw_vllm.sleep_mode.restore_model_tp",
               return_value={"num_regions": 0, "total_bytes": 0,
                             "elapsed_s": 0, "throughput_gb_s": 0,
                             "tensor_parallel_size": 1, "per_rank": []}):
        stats = sleep_mode.wake_up(fake_llm_sleep_disabled, "/tmp/z.thaw")

    fake_llm_sleep_disabled.wake_up.assert_not_called()
    assert stats["vllm_wake_up_called"] is False


def test_wake_up_swallows_vllm_wake_error(fake_llm_sleep_enabled):
    """If the engine wasn't actually asleep, wake_up() raises. We don't
    want that to block the restore — the GPU tensors may just be in a
    weird shape that the restore can still overwrite."""
    from thaw_vllm import sleep_mode

    fake_llm_sleep_enabled.wake_up.side_effect = RuntimeError("not asleep")

    with patch("thaw_vllm.sleep_mode.restore_model_tp",
               return_value={"num_regions": 0, "total_bytes": 0,
                             "elapsed_s": 0, "throughput_gb_s": 0,
                             "tensor_parallel_size": 1, "per_rank": []}):
        stats = sleep_mode.wake_up(fake_llm_sleep_enabled, "/tmp/w.thaw")

    assert stats["vllm_wake_up_called"] is False
    assert stats["wake_path"] == "/tmp/w.thaw"
