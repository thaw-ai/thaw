"""
conftest.py — Mock heavy dependencies (torch, vllm) so pool tests
run locally without GPU/CUDA.
"""

import sys
from unittest.mock import MagicMock

# Mock torch before any thaw_vllm imports
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

# Mock transformers — AutoTokenizer.from_pretrained must fail so
# _get_tokenizer falls back to None (MagicMock isn't JSON-serializable).
mock_transformers = MagicMock()
mock_transformers.AutoTokenizer.from_pretrained.side_effect = OSError("mock")
sys.modules.setdefault("transformers", mock_transformers)
