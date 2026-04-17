"""
Tests for thaw_sglang — SGLang ModelLoader integration.

All tests run on Mac without GPU by mocking SGLang and thaw_common.
Tests cover:
  - Loader init extracts snapshot path from extra config
  - ValueError on missing snapshot path
  - download_model is a no-op
  - load_model calls _initialize_model + restore fallback chain
  - TP rank detection and per-rank paths
  - FileNotFoundError for missing per-rank snapshots
  - thaw_sglang.load() wires up Engine correctly
  - thaw_sglang.freeze() wires up Engine with FreezeLoader
  - FreezeLoader delegates to default loader then freezes
  - Public API re-exports
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_load_config(snapshot_path=None):
    """Create a mock LoadConfig with model_loader_extra_config."""
    config = MagicMock()
    if snapshot_path is not None:
        config.model_loader_extra_config = {"snapshot": snapshot_path}
    else:
        config.model_loader_extra_config = {}
    return config


def _make_load_config_none():
    """Create a mock LoadConfig with None extra config."""
    config = MagicMock()
    config.model_loader_extra_config = None
    return config


_FAKE_STATS = {
    "num_regions": 10,
    "total_bytes": 1_000_000_000,
    "elapsed_s": 0.5,
    "throughput_gb_s": 2.0,
}


# ---------------------------------------------------------------------------
# ThawSGLangModelLoader tests
# ---------------------------------------------------------------------------


class TestThawSGLangModelLoader:

    def test_init_extracts_snapshot(self):
        from thaw_sglang.loader import ThawSGLangModelLoader
        config = _make_load_config("/path/to/weights.thaw")
        loader = ThawSGLangModelLoader(config)
        assert loader.snapshot_path == "/path/to/weights.thaw"

    def test_init_missing_snapshot_raises(self):
        from thaw_sglang.loader import ThawSGLangModelLoader
        config = _make_load_config()  # empty extra config
        with pytest.raises(ValueError, match="snapshot"):
            ThawSGLangModelLoader(config)

    def test_init_none_extra_config_raises(self):
        from thaw_sglang.loader import ThawSGLangModelLoader
        config = _make_load_config_none()
        with pytest.raises(ValueError, match="snapshot"):
            ThawSGLangModelLoader(config)

    def test_download_model_noop(self):
        from thaw_sglang.loader import ThawSGLangModelLoader
        config = _make_load_config("/path/to/weights.thaw")
        loader = ThawSGLangModelLoader(config)
        # Should not raise — it's a no-op
        loader.download_model(MagicMock())

    def test_load_model_calls_restore(self):
        """load_model creates skeleton via _initialize_model, then restores."""
        from thaw_sglang.loader import ThawSGLangModelLoader

        with tempfile.NamedTemporaryFile(suffix=".thaw", delete=False) as f:
            f.write(b"THAW" + b"\x00" * 100)
            snap_path = f.name

        try:
            config = _make_load_config(snap_path)
            loader = ThawSGLangModelLoader(config)

            mock_model = MagicMock()
            mock_init = MagicMock(return_value=mock_model)

            with patch("thaw_sglang.loader._initialize_model", mock_init, create=True), \
                 patch.dict(sys.modules["sglang.srt.model_loader.loader"].__dict__,
                           {"_initialize_model": mock_init}), \
                 patch("thaw_sglang.loader.restore_model_from_ram",
                       return_value=dict(_FAKE_STATS)) as mock_restore, \
                 patch("thaw_sglang.loader._get_tp_rank", return_value=0), \
                 patch("thaw_sglang.loader._get_tp_size", return_value=1):

                # Patch the import inside load_model
                original_load_model = loader.load_model

                def patched_load_model(**kwargs):
                    import thaw_sglang.loader as loader_mod
                    # Patch _initialize_model at module level for the import
                    with patch.object(
                        sys.modules["sglang.srt.model_loader.loader"],
                        "_initialize_model", mock_init
                    ):
                        return original_load_model(**kwargs)

                result = patched_load_model(
                    model_config=MagicMock(),
                    device_config=MagicMock(),
                )

                assert result is mock_model
                assert mock_restore.called
        finally:
            os.unlink(snap_path)

    def test_load_model_fallback_chain(self):
        """If restore_model_from_ram fails, falls back to pipelined, then pure."""
        from thaw_sglang.loader import ThawSGLangModelLoader

        with tempfile.NamedTemporaryFile(suffix=".thaw", delete=False) as f:
            f.write(b"THAW" + b"\x00" * 100)
            snap_path = f.name

        try:
            config = _make_load_config(snap_path)
            loader = ThawSGLangModelLoader(config)

            mock_model = MagicMock()

            with patch.object(
                     sys.modules["sglang.srt.model_loader.loader"],
                     "_initialize_model", MagicMock(return_value=mock_model)), \
                 patch("thaw_sglang.loader.restore_model_from_ram",
                       side_effect=RuntimeError("no rust ext")), \
                 patch("thaw_sglang.loader.restore_model_pipelined",
                       side_effect=RuntimeError("no rust ext")), \
                 patch("thaw_sglang.loader.restore_model",
                       return_value=dict(_FAKE_STATS)) as mock_pure, \
                 patch("thaw_sglang.loader._get_tp_rank", return_value=0), \
                 patch("thaw_sglang.loader._get_tp_size", return_value=1):

                result = loader.load_model(
                    model_config=MagicMock(),
                    device_config=MagicMock(),
                )

                assert result is mock_model
                assert mock_pure.called
        finally:
            os.unlink(snap_path)


# ---------------------------------------------------------------------------
# TP (tensor parallel) tests
# ---------------------------------------------------------------------------


class TestTPSupport:

    def test_rank_0_uses_base_path(self):
        """Rank 0 loads from the base snapshot path."""
        from thaw_sglang.loader import ThawSGLangModelLoader

        with tempfile.NamedTemporaryFile(suffix=".thaw", delete=False) as f:
            f.write(b"THAW" + b"\x00" * 100)
            snap_path = f.name

        try:
            config = _make_load_config(snap_path)
            loader = ThawSGLangModelLoader(config)

            with patch.object(
                     sys.modules["sglang.srt.model_loader.loader"],
                     "_initialize_model", MagicMock(return_value=MagicMock())), \
                 patch("thaw_sglang.loader.restore_model_from_ram",
                       return_value=dict(_FAKE_STATS)) as mock_restore, \
                 patch("thaw_sglang.loader._get_tp_rank", return_value=0), \
                 patch("thaw_sglang.loader._get_tp_size", return_value=4):

                loader.load_model(
                    model_config=MagicMock(),
                    device_config=MagicMock(),
                )

                # Should use the base path for rank 0
                restore_path = mock_restore.call_args[0][1]
                assert restore_path == snap_path
        finally:
            os.unlink(snap_path)

    def test_rank_nonzero_uses_rank_path(self):
        """Non-zero ranks load from rank-specific paths."""
        from thaw_sglang.loader import ThawSGLangModelLoader

        tmpdir = tempfile.mkdtemp(prefix="thaw_test_")
        snap_path = os.path.join(tmpdir, "weights.thaw")
        rank_path = os.path.join(tmpdir, "weights.rank2.thaw")

        # Create both files
        for p in [snap_path, rank_path]:
            with open(p, "wb") as f:
                f.write(b"THAW" + b"\x00" * 100)

        try:
            config = _make_load_config(snap_path)
            loader = ThawSGLangModelLoader(config)

            with patch.object(
                     sys.modules["sglang.srt.model_loader.loader"],
                     "_initialize_model", MagicMock(return_value=MagicMock())), \
                 patch("thaw_sglang.loader.restore_model_from_ram",
                       return_value=dict(_FAKE_STATS)) as mock_restore, \
                 patch("thaw_sglang.loader._get_tp_rank", return_value=2), \
                 patch("thaw_sglang.loader._get_tp_size", return_value=4):

                loader.load_model(
                    model_config=MagicMock(),
                    device_config=MagicMock(),
                )

                restore_path = mock_restore.call_args[0][1]
                assert restore_path == rank_path
        finally:
            for p in [snap_path, rank_path]:
                if os.path.exists(p):
                    os.unlink(p)
            os.rmdir(tmpdir)

    def test_missing_rank_file_raises(self):
        """TP > 1 with missing per-rank file raises FileNotFoundError."""
        from thaw_sglang.loader import ThawSGLangModelLoader

        with tempfile.NamedTemporaryFile(suffix=".thaw", delete=False) as f:
            f.write(b"THAW" + b"\x00" * 100)
            snap_path = f.name

        try:
            config = _make_load_config(snap_path)
            loader = ThawSGLangModelLoader(config)

            with patch.object(
                     sys.modules["sglang.srt.model_loader.loader"],
                     "_initialize_model", MagicMock(return_value=MagicMock())), \
                 patch("thaw_sglang.loader._get_tp_rank", return_value=1), \
                 patch("thaw_sglang.loader._get_tp_size", return_value=2):

                with pytest.raises(FileNotFoundError, match="Per-rank snapshot"):
                    loader.load_model(
                        model_config=MagicMock(),
                        device_config=MagicMock(),
                    )
        finally:
            os.unlink(snap_path)

    def test_single_gpu_no_rank_check(self):
        """With tp_size=1, missing rank file doesn't matter (uses base path)."""
        from thaw_sglang.loader import ThawSGLangModelLoader

        with tempfile.NamedTemporaryFile(suffix=".thaw", delete=False) as f:
            f.write(b"THAW" + b"\x00" * 100)
            snap_path = f.name

        try:
            config = _make_load_config(snap_path)
            loader = ThawSGLangModelLoader(config)

            with patch.object(
                     sys.modules["sglang.srt.model_loader.loader"],
                     "_initialize_model", MagicMock(return_value=MagicMock())), \
                 patch("thaw_sglang.loader.restore_model_from_ram",
                       return_value=dict(_FAKE_STATS)), \
                 patch("thaw_sglang.loader._get_tp_rank", return_value=0), \
                 patch("thaw_sglang.loader._get_tp_size", return_value=1):

                # Should not raise — single GPU doesn't check for rank files
                result = loader.load_model(
                    model_config=MagicMock(),
                    device_config=MagicMock(),
                )
                assert result is not None
        finally:
            os.unlink(snap_path)


# ---------------------------------------------------------------------------
# thaw_sglang.load() convenience function
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ThawSGLangFreezeLoader tests
# ---------------------------------------------------------------------------


class TestThawSGLangFreezeLoader:

    def test_init_extracts_snapshot(self):
        from thaw_sglang.loader import ThawSGLangFreezeLoader
        config = _make_load_config("/path/to/output.thaw")
        loader = ThawSGLangFreezeLoader(config)
        assert loader.snapshot_path == "/path/to/output.thaw"

    def test_init_missing_snapshot_raises(self):
        from thaw_sglang.loader import ThawSGLangFreezeLoader
        config = _make_load_config()
        with pytest.raises(ValueError, match="snapshot"):
            ThawSGLangFreezeLoader(config)

    def test_load_model_freezes_after_loading(self):
        """FreezeLoader loads via default loader, then freezes."""
        from thaw_sglang.loader import ThawSGLangFreezeLoader

        with tempfile.NamedTemporaryFile(suffix=".thaw", delete=False) as f:
            snap_path = f.name

        try:
            config = _make_load_config(snap_path)
            loader = ThawSGLangFreezeLoader(config)

            mock_model = MagicMock()
            mock_default_loader = MagicMock()
            mock_default_loader.load_model.return_value = mock_model

            with patch("thaw_sglang.loader.get_model_loader",
                       return_value=mock_default_loader), \
                 patch("thaw_sglang.loader.freeze_model_pipelined",
                       return_value=dict(_FAKE_STATS)) as mock_freeze, \
                 patch("thaw_sglang.loader._get_tp_rank", return_value=0), \
                 patch("thaw_sglang.loader._get_tp_size", return_value=1):

                result = loader.load_model(
                    model_config=MagicMock(),
                    device_config=MagicMock(),
                )

                assert result is mock_model
                mock_default_loader.load_model.assert_called_once()
                mock_freeze.assert_called_once()
                assert mock_freeze.call_args[0][0] is mock_model
        finally:
            if os.path.exists(snap_path):
                os.unlink(snap_path)

    def test_load_model_freeze_fallback(self):
        """If pipelined freeze fails, falls back to basic freeze."""
        from thaw_sglang.loader import ThawSGLangFreezeLoader

        config = _make_load_config("/tmp/out.thaw")
        loader = ThawSGLangFreezeLoader(config)

        mock_model = MagicMock()
        mock_default_loader = MagicMock()
        mock_default_loader.load_model.return_value = mock_model

        with patch("thaw_sglang.loader.get_model_loader",
                   return_value=mock_default_loader), \
             patch("thaw_sglang.loader.freeze_model_pipelined",
                   side_effect=RuntimeError("no rust ext")), \
             patch("thaw_sglang.loader.freeze_model",
                   return_value=dict(_FAKE_STATS)) as mock_basic, \
             patch("thaw_sglang.loader._get_tp_rank", return_value=0), \
             patch("thaw_sglang.loader._get_tp_size", return_value=1):

            result = loader.load_model(
                model_config=MagicMock(),
                device_config=MagicMock(),
            )

            assert result is mock_model
            mock_basic.assert_called_once()

    def test_tp_rank_uses_rank_path(self):
        """FreezeLoader writes to per-rank path for non-zero ranks."""
        from thaw_sglang.loader import ThawSGLangFreezeLoader

        config = _make_load_config("/tmp/weights.thaw")
        loader = ThawSGLangFreezeLoader(config)

        mock_default_loader = MagicMock()
        mock_default_loader.load_model.return_value = MagicMock()

        with patch("thaw_sglang.loader.get_model_loader",
                   return_value=mock_default_loader), \
             patch("thaw_sglang.loader.freeze_model_pipelined",
                   return_value=dict(_FAKE_STATS)) as mock_freeze, \
             patch("thaw_sglang.loader._get_tp_rank", return_value=2), \
             patch("thaw_sglang.loader._get_tp_size", return_value=4):

            loader.load_model(
                model_config=MagicMock(),
                device_config=MagicMock(),
            )

            freeze_path = mock_freeze.call_args[0][1]
            assert freeze_path == "/tmp/weights.rank2.thaw"


# ---------------------------------------------------------------------------
# thaw_sglang.freeze() convenience function
# ---------------------------------------------------------------------------


class TestFreezeFunction:

    def test_freeze_calls_engine_and_shutdown(self):
        """thaw_sglang.freeze() creates Engine with FreezeLoader, then shuts down."""
        import thaw_sglang

        mock_engine = MagicMock()
        with patch("sglang.Engine", return_value=mock_engine) as mock_cls:
            thaw_sglang.freeze(
                "meta-llama/Meta-Llama-3-8B",
                "/path/to/output.thaw",
            )

            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["model_path"] == "meta-llama/Meta-Llama-3-8B"
            assert call_kwargs["model_loader_extra_config"] == {
                "snapshot": "/path/to/output.thaw"
            }
            mock_engine.shutdown.assert_called_once()

    def test_freeze_passes_tp_size(self):
        """Extra kwargs like tp_size are forwarded."""
        import thaw_sglang

        mock_engine = MagicMock()
        with patch("sglang.Engine", return_value=mock_engine) as mock_cls:
            thaw_sglang.freeze("model", "/out.thaw", tp_size=4)

            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["tp_size"] == 4


# ---------------------------------------------------------------------------
# thaw_sglang.load() convenience function
# ---------------------------------------------------------------------------


class TestLoadFunction:

    def test_load_calls_engine(self):
        """thaw_sglang.load() passes correct args to sglang.Engine."""
        import thaw_sglang

        mock_engine = MagicMock()
        with patch("sglang.Engine", return_value=mock_engine) as mock_cls:
            result = thaw_sglang.load(
                "meta-llama/Meta-Llama-3-8B",
                "/path/to/weights.thaw",
            )

            assert result is mock_engine
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["model_path"] == "meta-llama/Meta-Llama-3-8B"
            assert call_kwargs["model_loader_extra_config"] == {
                "snapshot": "/path/to/weights.thaw"
            }
            assert call_kwargs["dtype"] == "float16"

    def test_load_passes_kwargs(self):
        """Extra kwargs are forwarded to sglang.Engine."""
        import thaw_sglang

        mock_engine = MagicMock()
        with patch("sglang.Engine", return_value=mock_engine) as mock_cls:
            thaw_sglang.load(
                "model", "/snap.thaw",
                tp_size=4, dtype="bfloat16",
            )

            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["tp_size"] == 4
            # Explicit dtype overrides the default
            assert call_kwargs["dtype"] == "bfloat16"

    def test_load_default_dtype(self):
        """dtype defaults to float16 if not specified."""
        import thaw_sglang

        with patch("sglang.Engine", return_value=MagicMock()) as mock_cls:
            thaw_sglang.load("model", "/snap.thaw")
            assert mock_cls.call_args[1]["dtype"] == "float16"


# ---------------------------------------------------------------------------
# Public API / re-exports
# ---------------------------------------------------------------------------


class TestPublicAPI:

    def test_exports_freeze_model(self):
        import thaw_sglang
        assert hasattr(thaw_sglang, "freeze_model")

    def test_exports_restore_model(self):
        import thaw_sglang
        assert hasattr(thaw_sglang, "restore_model")

    def test_exports_restore_from_ram(self):
        import thaw_sglang
        assert hasattr(thaw_sglang, "restore_model_from_ram")

    def test_exports_loader_class(self):
        import thaw_sglang
        assert thaw_sglang.ThawSGLangModelLoader is not None

    def test_exports_freeze_loader_class(self):
        import thaw_sglang
        assert thaw_sglang.ThawSGLangFreezeLoader is not None

    def test_exports_load_function(self):
        import thaw_sglang
        assert callable(thaw_sglang.load)

    def test_exports_freeze_function(self):
        import thaw_sglang
        assert callable(thaw_sglang.freeze)

    def test_all_list(self):
        import thaw_sglang
        expected = [
            "freeze_model", "freeze_model_pipelined",
            "restore_model", "restore_model_from_ram", "restore_model_pipelined",
            "freeze", "load", "ThawSGLangModelLoader", "ThawSGLangFreezeLoader",
        ]
        for name in expected:
            assert name in thaw_sglang.__all__, f"{name} missing from __all__"
