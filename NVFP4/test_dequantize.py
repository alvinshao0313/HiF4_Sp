import json
import tempfile
import unittest
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from NVFP4.dequantize import (
    ACTIVATION_SCALES_FILE,
    INDEX_FILE,
    convert_nvfp4_checkpoint_to_bf16,
    dequantize_nvfp4_weight,
)


class Nvfp4DequantizeTest(unittest.TestCase):
    def test_dequantize_nvfp4_weight(self):
        weight_packed = torch.tensor([[0x21, 0xB7]], dtype=torch.uint8)
        weight_scale = torch.tensor([[2.0, 4.0]], dtype=torch.float8_e4m3fn)
        weight_global_scale = torch.tensor([2.0], dtype=torch.float32)

        out = dequantize_nvfp4_weight(
            weight_packed,
            weight_scale,
            weight_global_scale,
            dtype=torch.float32,
            group_size=2,
        )

        expected = torch.tensor([[0.5, 1.0, 12.0, -3.0]], dtype=torch.float32)
        torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)

    def test_convert_checkpoint_to_bf16_keeps_activation_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            config = {
                "model_type": "tiny",
                "dtype": "bfloat16",
                "quantization_config": {"format": "nvfp4-pack-quantized"},
            }
            (input_dir / "config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            (input_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

            shard_1 = {
                "dense.weight": torch.tensor(
                    [[1.0, 2.0]], dtype=torch.bfloat16
                ),
                "layer.weight_packed": torch.tensor(
                    [[0x21, 0xB7, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8
                ),
            }
            shard_2 = {
                "layer.weight_scale": torch.tensor(
                    [[2.0]], dtype=torch.float8_e4m3fn
                ),
                "layer.weight_global_scale": torch.tensor([2.0], dtype=torch.float32),
                "layer.input_global_scale": torch.tensor([8.0], dtype=torch.float32),
            }
            save_file(shard_1, input_dir / "model-00001-of-00002.safetensors")
            save_file(shard_2, input_dir / "model-00002-of-00002.safetensors")
            index = {
                "metadata": {"total_size": 0},
                "weight_map": {
                    **{
                        name: "model-00001-of-00002.safetensors"
                        for name in shard_1
                    },
                    **{
                        name: "model-00002-of-00002.safetensors"
                        for name in shard_2
                    },
                },
            }
            (input_dir / INDEX_FILE).write_text(
                json.dumps(index), encoding="utf-8"
            )

            convert_nvfp4_checkpoint_to_bf16(
                input_dir,
                output_dir,
                keep_activation_scales=True,
                max_shard_size=1024 * 1024,
            )

            with (output_dir / "config.json").open("r", encoding="utf-8") as handle:
                out_config = json.load(handle)
            self.assertNotIn("quantization_config", out_config)
            self.assertEqual(out_config["dtype"], "bfloat16")

            with (output_dir / INDEX_FILE).open("r", encoding="utf-8") as handle:
                out_index = json.load(handle)
            out_weight_map = out_index["weight_map"]
            self.assertIn("layer.weight", out_weight_map)
            self.assertIn("dense.weight", out_weight_map)
            self.assertNotIn("layer.weight_packed", out_weight_map)
            self.assertNotIn("layer.weight_scale", out_weight_map)
            self.assertNotIn("layer.weight_global_scale", out_weight_map)
            self.assertNotIn("layer.input_global_scale", out_weight_map)

            shard = output_dir / out_weight_map["layer.weight"]
            with safe_open(shard, framework="pt", device="cpu") as handle:
                converted = handle.get_tensor("layer.weight")
                self.assertEqual(converted.dtype, torch.bfloat16)
                torch.testing.assert_close(
                    converted.to(torch.float32),
                    torch.tensor(
                        [[0.5, 1.0, 6.0, -1.5] + [0.0] * 12],
                        dtype=torch.float32,
                    ),
                    rtol=0.0,
                    atol=0.0,
                )

            with safe_open(
                output_dir / ACTIVATION_SCALES_FILE,
                framework="pt",
                device="cpu",
            ) as handle:
                self.assertIn("layer.input_global_scale", handle.keys())
                torch.testing.assert_close(
                    handle.get_tensor("layer.input_global_scale"),
                    torch.tensor([8.0], dtype=torch.float32),
                )


if __name__ == "__main__":
    unittest.main()
