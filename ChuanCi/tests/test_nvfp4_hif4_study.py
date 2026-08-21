import copy
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

study = importlib.import_module("nvfp4_hif4_study")


class SyntheticStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = study.run_synthetic_study(
            study.StudyConfig(
                seed=31,
                samples_per_repeat=640,
                repeats=2,
                distributions=("gaussian",),
            ),
            device=torch.device("cpu"),
        )

    def test_schema_contains_all_main_experiments(self) -> None:
        self.assertEqual(self.result["schema_version"], 2)
        gaussian = self.result["synthetic"]["distributions"]["gaussian"]
        self.assertIn("same_source_format", gaussian)
        self.assertIn("native_conversion", gaussian)
        self.assertIn("pts_fp32", gaussian["native_conversion"]["variants"])
        self.assertIn("pts_bf16", gaussian["native_conversion"]["variants"])
        self.assertIn("pts_bf16_projection", gaussian["native_conversion"]["variants"])
        self.assertIn("pts_bf16_carrier_decomposition", gaussian["native_conversion"])
        self.assertIn("payload_ablation", gaussian)
        self.assertIn("micro_exponent_ablation", gaussian)
        self.assertIn("top_scale_ablation", gaussian)
        self.assertIn("group_size_ablation", gaussian)

    def test_group_size_ablation_retains_complete_three_level_hierarchy(self) -> None:
        gaussian = self.result["synthetic"]["distributions"]["gaussian"]
        for source_name in ("bf16_source", "nvfp4_source"):
            group = gaussian["group_size_ablation"][source_name]
            self.assertEqual(set(group["variants"]), {"g16", "g32", "g64"})
            for size in (16, 32, 64):
                variant = group["variants"][f"g{size}"]
                self.assertEqual(variant["group_size"], size)
                self.assertTrue(variant["full_three_level_hierarchy"])
                self.assertEqual(variant["config"]["payload_format"], "s1p2")
                self.assertEqual(variant["config"]["hierarchy_format"], "s1p2")
                self.assertTrue(variant["config"]["enable_exp8"])
                self.assertTrue(variant["config"]["enable_exp4"])

            nmse16 = group["variants"]["g16"]["metrics"]["nmse"]
            nmse32 = group["variants"]["g32"]["metrics"]["nmse"]
            nmse64 = group["variants"]["g64"]["metrics"]["nmse"]
            self.assertAlmostEqual(
                group["comparisons"]["nmse_drop_64_to_32"],
                nmse64 - nmse32,
            )
            self.assertAlmostEqual(
                group["comparisons"]["nmse_drop_32_to_16"],
                nmse32 - nmse16,
            )

    def test_payload_ablation_contains_native_and_upper_bound_paths(self) -> None:
        payload = self.result["synthetic"]["distributions"]["gaussian"]["payload_ablation"]
        for source_name in ("bf16_source", "nvfp4_source"):
            variants = payload[source_name]["variants"]
            self.assertIn("s1p2_native", variants)
            self.assertIn("e2m1_native", variants)
            self.assertIn("e2m1_fixed", variants)
            self.assertIn("bf16_range_matched", variants)
            self.assertIn("bf16_unclipped", variants)
            self.assertIn("e2m1_minus_s1p2_nmse", payload[source_name]["comparisons"])

    def test_error_decompositions_close_the_energy_identity(self) -> None:
        gaussian = self.result["synthetic"]["distributions"]["gaussian"]
        native_residual = gaussian["native_conversion"]["bf16_carrier_decomposition"]["identity_residual"]
        payload_residual = gaussian["payload_ablation"]["bf16_source"]["s1p2_vs_bf16_decomposition"]["identity_residual"]
        self.assertLess(abs(native_residual), 1e-8)
        self.assertLess(abs(payload_residual), 1e-8)

    def test_report_is_offline_and_written_for_general_readers(self) -> None:
        html_text = study.render_html_report(self.result)
        self.assertIn("为什么要做这些实验", html_text)
        self.assertIn("结论先行", html_text)
        self.assertIn("种合成分布", html_text)
        self.assertIn("S1P2、E2M1与BF16", html_text)
        self.assertIn("NVFP4转成HiF4", html_text)
        self.assertIn("PTS-FP32", html_text)
        self.assertIn("研究问题与实验假设", html_text)
        self.assertIn("HiF4的误差从哪里产生", html_text)
        self.assertIn("完整三级量化", html_text)
        self.assertIn("边际收益", html_text)
        self.assertIn("误差来源综合排序", html_text)
        self.assertIn("对后续HiF4量化算法的指导", html_text)
        self.assertIn("有效性威胁与结论边界", html_text)
        self.assertNotIn("https://", html_text)
        self.assertNotIn("http://", html_text)


class RealPackedStudyTests(unittest.TestCase):
    def test_real_packed_study_filters_layers_and_reports_main_paths(self) -> None:
        try:
            import safetensors.torch
        except ImportError:
            self.skipTest("safetensors unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            state = {}
            for layer in (3, 31):
                module_name = f"model.layers.{layer}.self_attn.q_proj"
                state[f"{module_name}.weight_packed"] = torch.full((2, 32), 0x11, dtype=torch.uint8)
                state[f"{module_name}.weight_scale"] = torch.full(
                    (2, 4), 2.0, dtype=torch.float8_e4m3fn
                )
                state[f"{module_name}.weight_global_scale"] = torch.tensor([2.0], dtype=torch.float32)
            safetensors.torch.save_file(state, checkpoint / "model.safetensors")

            result = study.run_real_packed_study(
                checkpoint,
                layers=(3,),
                device=torch.device("cpu"),
                chunk_groups=2,
            )

        self.assertEqual(result["tensor_count"], 1)
        self.assertEqual(result["layers"], [3])
        self.assertIn("model.layers.3.self_attn.q_proj.weight", result["tensors"])
        global_result = result["global"]
        self.assertIn("native_conversion", global_result)
        self.assertIn("payload_ablation", global_result)
        self.assertIn("micro_exponent_ablation", global_result)
        self.assertIn("top_scale_ablation", global_result)
        self.assertIn("group_size_ablation", global_result)
        self.assertIn("e2m1_native", global_result["payload_ablation"]["variants"])
        native_variants = global_result["native_conversion"]["variants"]
        self.assertIn("pts_fp32", native_variants)
        self.assertIn("pts_bf16", native_variants)
        self.assertIn("pts_bf16_projection", native_variants)
        self.assertIn("H11_full", global_result["micro_exponent_ablation"]["variants"])
        self.assertIn("hardware", global_result["top_scale_ablation"]["variants"])
        group = global_result["group_size_ablation"]
        self.assertEqual(set(group["variants"]), {"g16", "g32", "g64"})
        self.assertTrue(group["variants"]["g16"]["full_three_level_hierarchy"])
        self.assertIn("layer_results", result)
        self.assertIn("3", result["layer_results"])
        residual = global_result["native_conversion"]["bf16_carrier_decomposition"]["identity_residual"]
        pts_residual = global_result["native_conversion"]["pts_bf16_carrier_decomposition"]["identity_residual"]
        self.assertLess(abs(residual), 1e-8)
        self.assertLess(abs(pts_residual), 1e-8)

    def test_report_renders_real_packed_section(self) -> None:
        result = dict(SyntheticStudyTests.result)
        gaussian = SyntheticStudyTests.result["synthetic"]["distributions"]["gaussian"]
        native = copy.deepcopy(gaussian["native_conversion"])
        native["variants"]["pts_fp32"] = copy.deepcopy(native["variants"]["fp32_carrier"])
        native["variants"]["pts_bf16"] = copy.deepcopy(native["variants"]["bf16_carrier"])
        native["variants"]["pts_bf16_projection"] = copy.deepcopy(native["variants"]["bf16_projection"])
        native["pts_bf16_carrier_decomposition"] = copy.deepcopy(native["bf16_carrier_decomposition"])
        result["real_packed"] = {
            "checkpoint": "example",
            "layers": [3],
            "tensor_count": 1,
            "global": {
                "native_conversion": native,
                "payload_ablation": gaussian["payload_ablation"]["nvfp4_source"],
                "micro_exponent_ablation": gaussian["micro_exponent_ablation"]["nvfp4_source"],
                "top_scale_ablation": gaussian["top_scale_ablation"]["nvfp4_source"],
                "group_size_ablation": gaussian["group_size_ablation"]["nvfp4_source"],
            },
            "categories": {},
            "layer_results": {},
            "tensors": {},
        }
        html_text = study.render_html_report(result)
        self.assertIn("真实packed NVFP4权重", html_text)
        self.assertIn("保留全局FP32 scale", html_text)
        self.assertIn("PTS路径结论", html_text)
        self.assertIn("真实权重：两级指数位消融", html_text)
        self.assertIn("真实权重：顶层S0消融", html_text)
        self.assertIn("真实权重：顶层group共享范围", html_text)


if __name__ == "__main__":
    unittest.main()
