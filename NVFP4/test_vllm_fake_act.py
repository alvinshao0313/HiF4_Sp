import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "3rdparty" / "vllm"))

from NVFP4.torch_fake import fake_quant_nvfp4_activation  # noqa: E402
from vllm.model_executor.layers.linear import UnquantizedLinearMethod  # noqa: E402
import vllm.config.vllm as vllm_config_module  # noqa: E402


class Nvfp4VllmFakeActTest(unittest.TestCase):
    def setUp(self):
        UnquantizedLinearMethod._NVF4_ACTIVATION_SCALE_CACHE.clear()

    def _make_method(
        self,
        scales_path: str,
        *,
        hif4_fake_act: bool = False,
        nvf4_fake_act: bool = True,
    ) -> UnquantizedLinearMethod:
        old_config = vllm_config_module._current_vllm_config
        vllm_config_module._current_vllm_config = types.SimpleNamespace(
            additional_config={
                "hif4_fake_act": hif4_fake_act,
                "nvf4_fake_act": nvf4_fake_act,
                "nvf4_activation_scales_path": scales_path,
            }
        )
        try:
            return UnquantizedLinearMethod()
        finally:
            vllm_config_module._current_vllm_config = old_config

    def _linear(self, prefix: str) -> torch.nn.Linear:
        torch.manual_seed(123)
        layer = torch.nn.Linear(16, 3, bias=True)
        layer.prefix = prefix
        return layer

    def _bind_scale(
        self,
        method: UnquantizedLinearMethod,
        layer: torch.nn.Linear,
    ) -> torch.nn.Linear:
        method._setup_nvf4_fake_act(layer)
        return layer

    def test_plain_linear_uses_sidecar_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            scales_path = Path(tmp) / "nvfp4_activation_scales.safetensors"
            scale = torch.tensor([8.0], dtype=torch.float32)
            save_file(
                {
                    "model.language_model.layers.0.mlp.down_proj.input_global_scale": scale,
                },
                scales_path,
            )
            method = self._make_method(str(scales_path))
            layer = self._bind_scale(
                method,
                self._linear("model.layers.0.mlp.down_proj"),
            )
            x = torch.randn(2, 16, dtype=torch.float32)

            actual = method.apply(layer, x, layer.bias)
            expected_x = fake_quant_nvfp4_activation(x, scale, output_dtype=x.dtype)
            expected = F.linear(expected_x, layer.weight, layer.bias)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_fused_qkv_linear_requires_equal_sidecar_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            scales_path = Path(tmp) / "nvfp4_activation_scales.safetensors"
            scale = torch.tensor([8.0], dtype=torch.float32)
            save_file(
                {
                    "model.language_model.layers.0.self_attn.q_proj.input_global_scale": scale.clone(),
                    "model.language_model.layers.0.self_attn.k_proj.input_global_scale": scale.clone(),
                    "model.language_model.layers.0.self_attn.v_proj.input_global_scale": scale.clone(),
                },
                scales_path,
            )
            method = self._make_method(str(scales_path))
            layer = self._bind_scale(
                method,
                self._linear("model.layers.0.self_attn.qkv_proj"),
            )
            x = torch.randn(2, 16, dtype=torch.float32)

            actual = method.apply(layer, x, layer.bias)
            expected_x = fake_quant_nvfp4_activation(x, scale, output_dtype=x.dtype)
            expected = F.linear(expected_x, layer.weight, layer.bias)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_apply_uses_bound_scale_without_sidecar_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            scales_path = Path(tmp) / "nvfp4_activation_scales.safetensors"
            scale = torch.tensor([8.0], dtype=torch.float32)
            save_file(
                {
                    "model.language_model.layers.0.mlp.down_proj.input_global_scale": scale,
                },
                scales_path,
            )
            method = self._make_method(str(scales_path))
            layer = self._bind_scale(
                method,
                self._linear("model.layers.0.mlp.down_proj"),
            )
            method.nvf4_activation_scales_path = str(Path(tmp) / "missing.safetensors")
            UnquantizedLinearMethod._NVF4_ACTIVATION_SCALE_CACHE.clear()
            x = torch.randn(2, 16, dtype=torch.float32)

            actual = method.apply(layer, x, layer.bias)
            expected_x = fake_quant_nvfp4_activation(x, scale, output_dtype=x.dtype)
            expected = F.linear(expected_x, layer.weight, layer.bias)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_linear_attention_layers_without_sidecar_scale_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            scales_path = Path(tmp) / "nvfp4_activation_scales.safetensors"
            save_file(
                {
                    "model.language_model.layers.0.mlp.down_proj.input_global_scale": torch.tensor(
                        [8.0], dtype=torch.float32
                    ),
                },
                scales_path,
            )
            method = self._make_method(str(scales_path))
            layer = self._bind_scale(
                method,
                self._linear("model.layers.0.linear_attn.conv1d"),
            )
            x = torch.randn(2, 16, dtype=torch.float32)

            actual = method.apply(layer, x, layer.bias)
            expected = F.linear(x, layer.weight, layer.bias)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_missing_sidecar_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                self._make_method(str(Path(tmp) / "missing.safetensors"))

    def test_missing_linear_scale_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            scales_path = Path(tmp) / "nvfp4_activation_scales.safetensors"
            save_file(
                {
                    "model.language_model.layers.0.mlp.up_proj.input_global_scale": torch.tensor(
                        [8.0], dtype=torch.float32
                    ),
                },
                scales_path,
            )
            method = self._make_method(str(scales_path))
            layer = self._linear("model.layers.0.mlp.down_proj")

            with self.assertRaisesRegex(ValueError, "Missing NVFP4 activation scale"):
                self._bind_scale(method, layer)

    def test_fused_scale_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            scales_path = Path(tmp) / "nvfp4_activation_scales.safetensors"
            save_file(
                {
                    "model.language_model.layers.0.self_attn.q_proj.input_global_scale": torch.tensor(
                        [8.0], dtype=torch.float32
                    ),
                    "model.language_model.layers.0.self_attn.k_proj.input_global_scale": torch.tensor(
                        [9.0], dtype=torch.float32
                    ),
                    "model.language_model.layers.0.self_attn.v_proj.input_global_scale": torch.tensor(
                        [8.0], dtype=torch.float32
                    ),
                },
                scales_path,
            )
            method = self._make_method(str(scales_path))
            layer = self._linear("model.layers.0.self_attn.qkv_proj")

            with self.assertRaisesRegex(ValueError, "Fused NVFP4 activation scales"):
                self._bind_scale(method, layer)

    def test_hif4_and_nvf4_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            scales_path = Path(tmp) / "nvfp4_activation_scales.safetensors"
            save_file(
                {
                    "model.language_model.layers.0.mlp.down_proj.input_global_scale": torch.tensor(
                        [8.0], dtype=torch.float32
                    ),
                },
                scales_path,
            )
            with self.assertRaisesRegex(ValueError, "cannot both be enabled"):
                self._make_method(str(scales_path), hif4_fake_act=True)


if __name__ == "__main__":
    unittest.main()
