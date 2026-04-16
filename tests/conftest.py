"""
conftest.py — Mock heavy dependencies (torch, vllm, sglang) so tests
run locally without GPU/CUDA.
"""

import sys
from unittest.mock import MagicMock

# Mock torch before any thaw imports
mock_torch = MagicMock()
mock_torch.nn = MagicMock()
mock_torch.nn.Module = type("Module", (), {})
mock_torch.uint8 = "uint8"
sys.modules.setdefault("torch", mock_torch)
sys.modules.setdefault("torch.nn", mock_torch.nn)

# Mock vllm
mock_vllm = MagicMock()
mock_vllm.SamplingParams = MagicMock(return_value=MagicMock())
sys.modules.setdefault("vllm", mock_vllm)
sys.modules.setdefault("vllm.config", MagicMock())
sys.modules.setdefault("vllm.model_executor", MagicMock())
sys.modules.setdefault("vllm.model_executor.model_loader", MagicMock())
sys.modules.setdefault("vllm.model_executor.model_loader.base_loader", MagicMock())

# Mock sglang
mock_sglang = MagicMock()
sys.modules.setdefault("sglang", mock_sglang)
sys.modules.setdefault("sglang.srt", MagicMock())
sys.modules.setdefault("sglang.srt.configs", MagicMock())
sys.modules.setdefault("sglang.srt.configs.load_config", MagicMock())
sys.modules.setdefault("sglang.srt.model_loader", MagicMock())

# SGLang's BaseModelLoader — create a real base class so our loader
# can inherit from it and super().__init__ works.
class _MockSGLangBaseModelLoader:
    def __init__(self, load_config):
        self.load_config = load_config

mock_sglang_loader_module = MagicMock()
mock_sglang_loader_module.BaseModelLoader = _MockSGLangBaseModelLoader
sys.modules.setdefault("sglang.srt.model_loader.loader", mock_sglang_loader_module)

# SGLang distributed — mock TP rank/size helpers
mock_sglang_distributed = MagicMock()
mock_sglang_distributed.get_tensor_model_parallel_rank = MagicMock(return_value=0)
mock_sglang_distributed.get_tensor_model_parallel_world_size = MagicMock(return_value=1)
sys.modules.setdefault("sglang.srt.distributed", mock_sglang_distributed)

# Mock transformers — AutoTokenizer.from_pretrained must fail so
# _get_tokenizer falls back to None (MagicMock isn't JSON-serializable).
mock_transformers = MagicMock()
mock_transformers.AutoTokenizer.from_pretrained.side_effect = OSError("mock")
sys.modules.setdefault("transformers", mock_transformers)
