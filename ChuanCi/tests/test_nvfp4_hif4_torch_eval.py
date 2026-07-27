import importlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

module = importlib.import_module("nvfp4_hif4_torch")


def _assert_error_sums_close(
    testcase: unittest.TestCase,
    left: module.ErrorSums | dict,
    right: module.ErrorSums | dict,
    *,
    tol: float = 1e-10,
) -> None:
    if isinstance(left, dict):
        left = module.ErrorSums(**left)
    if isinstance(right, dict):
        right = module.ErrorSums(**right)
    testcase.assertEqual(left.numel, right.numel)
    for field in (
        "reference_energy",
        "approximation_energy",
        "error_energy",
        "dot",
        "absolute_error_sum",
        "max_absolute_error",
    ):
        testcase.assertLessEqual(
            abs(getattr(left, field) - getattr(right, field)),
            tol,
            msg=field,
        )


class NativeReferenceTests(unittest.TestCase):
    def test_nvfp4_paths_share_same_reference(self) -> None:
        reference = torch.tensor(
            [[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0] * 8],
            dtype=torch.float32,
        )
        result = module.evaluate_nvfp4_fake_weight(
            reference,
            pts_scale=0.25,
            return_reconstructions=True,
        )
        torch.testing.assert_close(
            result["reference"],
            reference,
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            set(result["paths"]),
            {"direct", "pts_fp32", "pts_bf16"},
        )

    def test_bf16_uses_post_cast_reference(self) -> None:
        fp32 = torch.tensor(
            [[1.001, -0.997, 0.333, -0.125] * 16],
            dtype=torch.float32,
        )
        expected = fp32.to(torch.bfloat16).to(torch.float32)
        result = module.evaluate_bf16_weight(
            fp32,
            return_reconstruction=True,
        )
        torch.testing.assert_close(
            result["reference"],
            expected,
            rtol=0,
            atol=0,
        )

    def test_missing_pts_never_gets_inferred(self) -> None:
        weight = torch.linspace(-1, 1, 64).reshape(1, 64)
        result = module.evaluate_nvfp4_fake_weight(weight)
        self.assertIsNotNone(result["paths"]["direct"])
        self.assertIsNone(result["paths"]["pts_fp32"])
        self.assertIsNone(result["paths"]["pts_bf16"])
        self.assertEqual(result["pts_status"], "not_provided")


class MetricTests(unittest.TestCase):
    def test_error_sums_and_metrics_match_manual_values(self) -> None:
        reference = torch.tensor([1.0, 2.0])
        approximation = torch.tensor([1.0, 1.0])
        sums = module.compute_error_sums(reference, approximation)
        metrics = module.finalize_error_metrics(sums)
        self.assertEqual(sums.numel, 2)
        self.assertEqual(sums.reference_energy, 5.0)
        self.assertEqual(sums.approximation_energy, 2.0)
        self.assertEqual(sums.error_energy, 1.0)
        self.assertEqual(sums.dot, 3.0)
        self.assertEqual(sums.absolute_error_sum, 1.0)
        self.assertEqual(sums.max_absolute_error, 1.0)
        self.assertAlmostEqual(metrics["nmse"], 0.2)
        self.assertAlmostEqual(metrics["nrmse"], math.sqrt(0.2))
        self.assertAlmostEqual(metrics["mae"], 0.5)

    def test_merge_chunks_matches_full(self) -> None:
        reference = torch.randn(128, generator=torch.Generator().manual_seed(21))
        approximation = torch.randn(128, generator=torch.Generator().manual_seed(22))
        full = module.compute_error_sums(reference, approximation)

        merged = module.ErrorSums()
        chunk_size = 64
        for start in range(0, reference.numel(), chunk_size):
            end = start + chunk_size
            merged = module.merge_error_sums(
                merged,
                module.compute_error_sums(reference[start:end], approximation[start:end]),
            )

        _assert_error_sums_close(self, merged, full, tol=1e-12)

    def test_zero_metrics_json_safe(self) -> None:
        zero_ref = torch.zeros(8)
        zero_approx = torch.zeros(8)
        sums = module.compute_error_sums(zero_ref, zero_approx)
        metrics = module.finalize_error_metrics(sums)
        json.dumps(metrics, allow_nan=False)


class TensorEvaluationTests(unittest.TestCase):
    def test_nvfp4_api_builds_three_paths_without_requantizing_reference(self) -> None:
        reference = torch.linspace(-2, 2, 64).reshape(1, 64)
        result = module.evaluate_nvfp4_fake_weight(
            reference,
            pts_scale=torch.tensor(0.25),
            return_reconstructions=True,
        )
        torch.testing.assert_close(result["reference"], reference, rtol=0, atol=0)
        for name in ("direct", "pts_fp32", "pts_bf16"):
            self.assertIn("metrics", result["paths"][name])
            self.assertEqual(
                result["paths"][name]["sums"]["reference_energy"],
                result["paths"]["direct"]["sums"]["reference_energy"],
            )

    def test_pts_bf16_path_matches_explicit_formula(self) -> None:
        reference = torch.randn(2, 64, generator=torch.Generator().manual_seed(6))
        scale = torch.tensor(0.137, dtype=torch.float32)
        result = module.evaluate_nvfp4_fake_weight(
            reference,
            pts_scale=scale,
            return_reconstructions=True,
        )
        normalized = (reference.float() / scale).to(torch.bfloat16).to(torch.float32)
        expected = module.quantize_hif4(normalized).values * scale
        torch.testing.assert_close(
            result["paths"]["pts_bf16"]["reconstruction"],
            expected,
            rtol=0,
            atol=0,
        )

    def test_pts_scale_validation(self) -> None:
        reference = torch.randn(64)

        module.evaluate_nvfp4_fake_weight(reference, pts_scale=0.25)
        module.evaluate_nvfp4_fake_weight(reference, pts_scale=torch.tensor(0.25))
        module.evaluate_nvfp4_fake_weight(
            reference,
            pts_scale=torch.tensor([0.25, 0.25]).reshape(1, 2).expand(32, 2),
        )

        with self.assertRaises((ValueError, TypeError)):
            module.evaluate_nvfp4_fake_weight(reference, pts_scale=0.0)

        with self.assertRaises((ValueError, TypeError)):
            module.evaluate_nvfp4_fake_weight(reference, pts_scale=-0.1)

        with self.assertRaises((ValueError, TypeError)):
            module.evaluate_nvfp4_fake_weight(reference, pts_scale=float("nan"))

        with self.assertRaises((ValueError, TypeError)):
            module.evaluate_nvfp4_fake_weight(reference, pts_scale=float("inf"))

        with self.assertRaises((ValueError, TypeError)):
            module.evaluate_nvfp4_fake_weight(
                reference,
                pts_scale=torch.tensor([0.1, 0.2]),
            )

        with self.assertRaises((ValueError, TypeError)):
            module.evaluate_nvfp4_fake_weight(
                reference,
                pts_scale=torch.tensor(0.25 + 0.25j),
            )

        with self.assertRaises((ValueError, TypeError)):
            module.evaluate_nvfp4_fake_weight(
                reference,
                pts_scale=torch.tensor(1, dtype=torch.int32),
            )

    def test_bf16_api_does_not_include_fp32_to_bf16_loss(self) -> None:
        source = torch.tensor([[1.001, -0.997] * 32], dtype=torch.float32)
        result = module.evaluate_bf16_weight(
            source,
            return_reconstruction=True,
        )
        expected_reference = source.to(torch.bfloat16).to(torch.float32)
        torch.testing.assert_close(
            result["reference"],
            expected_reference,
            rtol=0,
            atol=0,
        )
        self.assertEqual(result["input_kind"], "bf16")
        serializable = {
            key: value
            for key, value in result.items()
            if key not in {"reference", "reconstruction"}
        }
        self.assertNotIn("nvfp4", json.dumps(serializable, default=str).lower())


class SimulationExperimentTests(unittest.TestCase):
    def test_distributions_are_deterministic_and_correct_size(self) -> None:
        for name in (
            "gaussian",
            "laplace",
            "student_t3",
            "outlier_0p1pct_20x",
        ):
            a = module.make_distribution(name, 6_400, seed=20260723)
            b = module.make_distribution(name, 6_400, seed=20260723)
            self.assertEqual(a.device.type, "cpu")
            self.assertEqual(a.dtype, torch.float32)
            self.assertEqual(a.numel(), 6_400)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.isfinite(a).all().item())

        base = module.make_distribution("gaussian", 6_400, seed=20260723)
        outlier = module.make_distribution("outlier_0p1pct_20x", 6_400, seed=20260723)
        ratio = outlier / base
        outlier_count = int((ratio > 19.9).sum().item())
        self.assertEqual(outlier_count, 6)

    def test_quick_simulation_contains_e1_native_paths(self) -> None:
        result = module.run_simulation(
            module.ExperimentConfig(
                samples_per_repeat=6_400,
                repeats=1,
                phase_points=17,
            ),
            device=torch.device("cpu"),
            quick=True,
        )
        e1 = result["experiments"]["e1_native_source"]
        for distribution in (
            "gaussian",
            "laplace",
            "student_t3",
            "outlier_0p1pct_20x",
        ):
            self.assertIn("nv_direct", e1[distribution])
            self.assertIn("nv_pts_fp32", e1[distribution])
            self.assertIn("nv_pts_bf16", e1[distribution])
            self.assertIn("bf16_native", e1[distribution])

    def test_e2_paired_reference_energy_and_structure(self) -> None:
        result = module.run_simulation(
            module.ExperimentConfig(
                samples_per_repeat=6_400,
                repeats=1,
                phase_points=17,
            ),
            device=torch.device("cpu"),
            quick=True,
        )
        e2 = result["experiments"]["e2_pts_pairing"]
        required_fields = {
            "delta_nmse",
            "relative_change",
            "paired_delta_mean",
            "paired_delta_std",
            "paired_delta_ci95_low",
            "paired_delta_ci95_high",
            "pts_bf16_win_count",
        }
        for distribution in e2.values():
            if not isinstance(distribution, dict):
                continue
            for key, repeat_result in distribution.items():
                if not isinstance(repeat_result, dict):
                    continue
                if not str(key).startswith("repeat_"):
                    continue
                for path_name in ("nv_direct", "nv_pts_fp32", "nv_pts_bf16"):
                    if path_name in repeat_result:
                        ref_energy = repeat_result[path_name]["sums"]["reference_energy"]
                        for other in ("nv_direct", "nv_pts_fp32", "nv_pts_bf16"):
                            if other in repeat_result:
                                self.assertEqual(
                                    repeat_result[other]["sums"]["reference_energy"],
                                    ref_energy,
                                )
                for field in required_fields:
                    self.assertIn(field, repeat_result)

    def test_e3_legal_codebook_exact_fraction(self) -> None:
        result = module.run_simulation(
            module.ExperimentConfig(
                samples_per_repeat=6_400,
                repeats=1,
                phase_points=17,
            ),
            device=torch.device("cpu"),
            quick=True,
        )
        legal = result["experiments"]["e3_bf16_carrier"]["legal_codebook_products"]
        self.assertEqual(legal["total_pairs"], 127 * 8)
        self.assertEqual(legal["exact_fraction"], 1.0)
        self.assertEqual(legal["max_absolute_error"], 0.0)

    def test_e4_phase_properties(self) -> None:
        result = module.run_simulation(
            module.ExperimentConfig(
                samples_per_repeat=6_400,
                repeats=1,
                phase_points=17,
            ),
            device=torch.device("cpu"),
            quick=True,
        )
        e4 = result["experiments"]["e4_phase_sweep"]
        phase = torch.tensor(e4["phase"], dtype=torch.float32)
        phase_points = 17
        self.assertEqual(phase.numel(), phase_points)
        self.assertEqual(phase[0].item(), 1.0)
        self.assertTrue(torch.all(phase < 2).item())
        self.assertTrue(torch.all(phase[1:] > phase[:-1]).item())

        phase_one = e4["points"][0]
        direct = phase_one["paths"]["direct"]["reconstruction"]
        pts_fp32 = phase_one["paths"]["pts_fp32"]["reconstruction"]
        torch.testing.assert_close(direct, pts_fp32, rtol=0, atol=0)

    def test_e5_has_four_scale_modes(self) -> None:
        result = module.run_simulation(
            module.ExperimentConfig(
                samples_per_repeat=6_400,
                repeats=1,
                phase_points=17,
            ),
            device=torch.device("cpu"),
            quick=True,
        )
        e5 = result["experiments"]["e5_scale_mode_decomposition"]
        expected_modes = {"continuous", "bf16_math", "e6m2_only", "hardware"}
        for source_block in e5.values():
            if not isinstance(source_block, dict):
                continue
            for distribution_block in source_block.values():
                if not isinstance(distribution_block, dict):
                    continue
                self.assertTrue(expected_modes.issubset(set(distribution_block.keys())))

    def test_e6_marks_nonstandard_group_sizes(self) -> None:
        result = module.run_simulation(
            module.ExperimentConfig(
                samples_per_repeat=6_400,
                repeats=1,
                phase_points=17,
            ),
            device=torch.device("cpu"),
            quick=True,
        )
        e6 = result["experiments"]["e6_group_size_ablation"]
        for source_block in e6.values():
            if not isinstance(source_block, dict):
                continue
            for entry in source_block.values():
                if not isinstance(entry, dict):
                    continue
                for item in entry.values():
                    if not isinstance(item, dict) or "group_size" not in item:
                        continue
                    if item["group_size"] == 64:
                        self.assertTrue(item["is_standard_hif4"])
                    elif item["group_size"] in (16, 32):
                        self.assertFalse(item["is_standard_hif4"])

    def test_e7_separates_storage_and_conversion(self) -> None:
        result = module.run_simulation(
            module.ExperimentConfig(
                samples_per_repeat=6_400,
                repeats=1,
                phase_points=17,
            ),
            device=torch.device("cpu"),
            quick=True,
        )
        e7 = result["experiments"]["e7_storage_dtype"]
        required_sections = {
            "storage_projection",
            "fp32_container_conversion",
            "bf16_container_conversion",
        }
        self.assertTrue(required_sections.issubset(set(e7.keys())))
        for section in required_sections:
            self.assertIn("metrics", e7[section])


class CheckpointLoadingTests(unittest.TestCase):
    def test_plain_state_dict_pt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            state = {
                "model.layers.0.self_attn.q_proj.weight": torch.randn(
                    64, 128, dtype=torch.bfloat16
                ),
                "model.layers.0.input_layernorm.weight": torch.ones(
                    128, dtype=torch.bfloat16
                ),
            }
            torch.save(state, path)
            items = list(module.iter_checkpoint_tensors(path))
            self.assertEqual(len(items), 2)
            self.assertEqual(
                [name for name, _ in items],
                sorted(state.keys()),
            )
            for _, tensor in items:
                self.assertEqual(tensor.device.type, "cpu")

    def test_wrapped_state_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrapped.pt"
            state = {
                "model.layers.0.self_attn.q_proj.weight": torch.randn(
                    64, 128, dtype=torch.bfloat16
                ),
            }
            for wrapper_key in ("state_dict", "model_state_dict"):
                torch.save({wrapper_key: state}, path)
                items = list(module.iter_checkpoint_tensors(path))
                self.assertEqual(len(items), 1)
                self.assertEqual(items[0][0], "model.layers.0.self_attn.q_proj.weight")

            bad_path = Path(tmp) / "bad.pt"
            torch.save({"optimizer_state_dict": {"step": 1}}, bad_path)
            with self.assertRaises(ValueError) as ctx:
                list(module.iter_checkpoint_tensors(bad_path))
            self.assertIn(str(bad_path), str(ctx.exception))

    def test_pts_json_loading_and_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            valid_path = Path(tmp) / "pts.json"
            valid_path.write_text(
                json.dumps(
                    {
                        "model.layers.0.self_attn.q_proj.weight": 0.001953125,
                        "model.layers.0.mlp.up_proj.weight": 0.00390625,
                    }
                ),
                encoding="utf-8",
            )
            loaded = module.load_pts_scales(valid_path)
            self.assertAlmostEqual(
                float(loaded["model.layers.0.self_attn.q_proj.weight"]),
                0.001953125,
            )

            invalid_cases = [
                {"model.layers.0.weight": 0},
                {"model.layers.0.weight": -0.1},
                {"model.layers.0.weight": "bad"},
                {"": 0.1},
            ]
            for payload in invalid_cases:
                bad_path = Path(tmp) / "bad.json"
                bad_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises((ValueError, TypeError)):
                    module.load_pts_scales(bad_path)

            pt_path = Path(tmp) / "pts.pt"
            torch.save({"model.layers.0.weight": torch.tensor(0.125)}, pt_path)
            pt_loaded = module.load_pts_scales(pt_path)
            self.assertAlmostEqual(float(pt_loaded["model.layers.0.weight"]), 0.125)

    def test_safetensors_lazy_or_skip(self) -> None:
        try:
            import safetensors.torch  # noqa: F401
        except ImportError:
            self.skipTest("safetensors unavailable for create")
            return

        import safetensors.torch

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            tensor = torch.randn(8, 64, dtype=torch.bfloat16)
            safetensors.torch.save_file(
                {"model.layers.0.self_attn.q_proj.weight": tensor},
                path,
            )
            items = list(module.iter_checkpoint_tensors(path))
            self.assertEqual(len(items), 1)
            name, loaded = items[0]
            self.assertEqual(name, "model.layers.0.self_attn.q_proj.weight")
            torch.testing.assert_close(loaded.to(torch.float32), tensor.to(torch.float32))


class PackedNVFP4CheckpointTests(unittest.TestCase):
    def _write_checkpoint(self, directory: Path) -> str:
        try:
            import safetensors.torch
        except ImportError:
            self.skipTest("safetensors unavailable for packed checkpoint test")

        selected_module = "model.layers.3.self_attn.q_proj"
        unselected_module = "model.layers.31.self_attn.q_proj"
        state = {
            f"{selected_module}.weight_packed": torch.full(
                (2, 32), 0x11, dtype=torch.uint8
            ),
            f"{selected_module}.weight_scale": torch.full(
                (2, 4), 2.0, dtype=torch.float8_e4m3fn
            ),
            f"{selected_module}.weight_global_scale": torch.tensor(
                [2.0], dtype=torch.float32
            ),
            f"{unselected_module}.weight_packed": torch.full(
                (2, 32), 0x11, dtype=torch.uint8
            ),
            f"{unselected_module}.weight_scale": torch.full(
                (2, 4), 2.0, dtype=torch.float8_e4m3fn
            ),
            # 若未在读取前筛选，这个非法 scale 会使测试失败。
            f"{unselected_module}.weight_global_scale": torch.tensor(
                [0.0], dtype=torch.float32
            ),
        }
        safetensors.torch.save_file(state, directory / "model.safetensors")
        return f"{selected_module}.weight"

    def test_decode_packed_weight_and_checkpoint_pts_scale(self) -> None:
        decoded, pts_scale = module.decode_nvfp4_packed_weight(
            torch.tensor([[0x21, 0xB7]], dtype=torch.uint8),
            torch.tensor([[2.0, 4.0]], dtype=torch.float8_e4m3fn),
            torch.tensor([2.0], dtype=torch.float32),
            group_size=2,
        )
        expected = torch.tensor([[0.5, 1.0, 12.0, -3.0]], dtype=torch.float32)
        torch.testing.assert_close(decoded, expected, rtol=0.0, atol=0.0)
        self.assertEqual(pts_scale.dtype, torch.float32)
        self.assertEqual(pts_scale.item(), 0.5)

    def test_selected_packed_weights_are_filtered_before_decode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            selected_name = self._write_checkpoint(checkpoint)
            items = list(
                module.iter_nvfp4_packed_weights(
                    checkpoint,
                    tensor_names=(selected_name,),
                )
            )
            self.assertEqual(len(items), 1)
            name, decoded, pts_scale = items[0]
            self.assertEqual(name, selected_name)
            self.assertEqual(decoded.shape, (2, 64))
            self.assertEqual(pts_scale.item(), 0.5)

    def test_packed_checkpoint_evaluation_uses_embedded_global_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            selected_name = self._write_checkpoint(checkpoint)
            result = module.evaluate_checkpoint(
                checkpoint_path=checkpoint,
                input_kind="nvfp4_packed",
                pts_scales_path=None,
                include_regex=None,
                exclude_regex=None,
                device=torch.device("cpu"),
                hif4_config=module.HiF4Config(group_size=64, group_dim=-1),
                tensor_names=(selected_name,),
                compute_group_summary=False,
            )
            self.assertEqual(result["input_kind"], "nvfp4_packed")
            self.assertEqual(set(result["tensors"]), {selected_name})
            entry = result["tensors"][selected_name]
            self.assertEqual(entry["pts_status"], "checkpoint_global_scale")
            self.assertIsNone(entry["group_summary"])
            for path_name in ("direct", "pts_fp32", "pts_bf16"):
                self.assertIsNotNone(entry["paths"][path_name])
                self.assertNotIn("reconstruction", entry["paths"][path_name])


class CheckpointEvaluationTests(unittest.TestCase):
    def _make_checkpoint(self, directory: Path) -> None:
        state = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(
                192, 128, dtype=torch.float32
            ),
            "model.layers.1.self_attn.q_proj.weight": torch.randn(
                192, 128, dtype=torch.float32
            ),
        }
        torch.save(state, directory / "model.pt")

    def test_chunk_matches_full_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "ckpt"
            checkpoint.mkdir()
            self._make_checkpoint(checkpoint)
            pts_path = Path(tmp) / "pts.json"
            pts_path.write_text(
                json.dumps(
                    {
                        "model.layers.0.self_attn.q_proj.weight": 0.25,
                    }
                ),
                encoding="utf-8",
            )
            config = module.HiF4Config(group_size=64, group_dim=-1)
            common = dict(
                checkpoint_path=checkpoint,
                input_kind="nvfp4_fake",
                pts_scales_path=pts_path,
                include_regex=None,
                exclude_regex=None,
                device=torch.device("cpu"),
                hif4_config=config,
                tensor_names=("model.layers.0.self_attn.q_proj.weight",),
            )
            full = module.evaluate_checkpoint(**common, chunk_groups=16_384)
            for chunk_groups in (1, 17, 16_384):
                chunked = module.evaluate_checkpoint(**common, chunk_groups=chunk_groups)
                for path_name in ("direct", "pts_fp32", "pts_bf16"):
                    if full["global"].get(path_name) is None:
                        continue
                    _assert_error_sums_close(
                        self,
                        chunked["global"][path_name]["sums"],
                        full["global"][path_name]["sums"],
                    )

    def test_global_aggregation_not_mean_of_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "ckpt"
            checkpoint.mkdir()
            w1 = torch.ones(64, 64)
            w2 = torch.ones(64, 64) * 1000.0
            state = {
                "model.layers.0.self_attn.q_proj.weight": w1,
                "model.layers.1.self_attn.up_proj.weight": w2,
            }
            torch.save(state, checkpoint / "model.pt")

            result = module.evaluate_checkpoint(
                checkpoint_path=checkpoint,
                input_kind="bf16",
                pts_scales_path=None,
                include_regex=None,
                exclude_regex=None,
                device=torch.device("cpu"),
                hif4_config=module.HiF4Config(group_size=64, group_dim=-1),
            )

            tensor_nmses = [
                result["tensors"][name]["paths"]["native"]["metrics"]["nmse"]
                for name in result["tensors"]
            ]
            global_nmse = result["global"]["native"]["metrics"]["nmse"]
            mean_tensor_nmse = sum(tensor_nmses) / len(tensor_nmses)
            self.assertNotAlmostEqual(global_nmse, mean_tensor_nmse, places=8)

            all_ref = torch.cat([w1.reshape(-1), w2.reshape(-1)])
            all_recon = torch.cat(
                [
                    result["tensors"]["model.layers.0.self_attn.q_proj.weight"]["paths"][
                        "native"
                    ]["reconstruction"].reshape(-1),
                    result["tensors"]["model.layers.1.self_attn.up_proj.weight"]["paths"][
                        "native"
                    ]["reconstruction"].reshape(-1),
                ]
            )
            expected_global = module.finalize_error_metrics(
                module.compute_error_sums(all_ref, all_recon)
            )
            self.assertAlmostEqual(global_nmse, expected_global["nmse"], places=12)

    def test_missing_pts_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "ckpt"
            checkpoint.mkdir()
            state = {
                "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            }
            torch.save(state, checkpoint / "model.pt")

            without_pts = module.evaluate_checkpoint(
                checkpoint_path=checkpoint,
                input_kind="nvfp4_fake",
                pts_scales_path=None,
                include_regex=None,
                exclude_regex=None,
                device=torch.device("cpu"),
                hif4_config=module.HiF4Config(group_size=64, group_dim=-1),
                require_pts=False,
            )
            self.assertIsNotNone(without_pts["global"]["direct"])
            self.assertIsNone(without_pts["global"].get("pts_fp32"))
            self.assertIsNone(without_pts["global"].get("pts_bf16"))

            with self.assertRaises((ValueError, RuntimeError)):
                module.evaluate_checkpoint(
                    checkpoint_path=checkpoint,
                    input_kind="nvfp4_fake",
                    pts_scales_path=None,
                    include_regex=None,
                    exclude_regex=None,
                    device=torch.device("cpu"),
                    hif4_config=module.HiF4Config(group_size=64, group_dim=-1),
                    require_pts=True,
                )

            pts_path = Path(tmp) / "pts.json"
            pts_path.write_text("{}", encoding="utf-8")
            with self.assertRaises((ValueError, TypeError)):
                module.evaluate_checkpoint(
                    checkpoint_path=checkpoint,
                    input_kind="bf16",
                    pts_scales_path=pts_path,
                    include_regex=None,
                    exclude_regex=None,
                    device=torch.device("cpu"),
                    hif4_config=module.HiF4Config(group_size=64, group_dim=-1),
                )


class OutputErrorTests(unittest.TestCase):
    def test_output_error_matches_full_matmul(self) -> None:
        x = torch.randn(13, 8, generator=torch.Generator().manual_seed(7))
        w = torch.randn(6, 8, generator=torch.Generator().manual_seed(8))
        w_hat = w + 0.01
        result = module.evaluate_output_error(
            x,
            w,
            w_hat,
            token_batch_size=4,
        )
        y = x.float() @ w.float().T
        y_hat = x.float() @ w_hat.float().T
        expected = module.finalize_error_metrics(
            module.compute_error_sums(y, y_hat)
        )
        self.assertAlmostEqual(result["nmse"], expected["nmse"], places=10)


class CLITests(unittest.TestCase):
    def test_simulate_quick_cli_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "sim_out"
            exit_code = module.main(
                [
                    "simulate",
                    "--quick",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            self.assertEqual(exit_code, 0)
            for filename in ("results.json", "results.csv", "report.md"):
                self.assertTrue((output_dir / filename).is_file())
            with (output_dir / "results.json").open(encoding="utf-8") as handle:
                payload = json.load(handle)
            json.dumps(payload, allow_nan=False)

    def test_evaluate_checkpoint_cli_bf16(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "ckpt"
            checkpoint.mkdir()
            state = {
                "model.layers.0.self_attn.q_proj.weight": torch.randn(
                    64, 64, dtype=torch.bfloat16
                ),
            }
            torch.save(state, checkpoint / "model.pt")
            output_dir = Path(tmp) / "eval_out"
            exit_code = module.main(
                [
                    "evaluate-checkpoint",
                    "--checkpoint",
                    str(checkpoint),
                    "--input-kind",
                    "bf16",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            self.assertEqual(exit_code, 0)
            for filename in ("results.json", "results.csv", "report.md"):
                self.assertTrue((output_dir / filename).is_file())
            report = (output_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("bf16", report.lower())
            self.assertNotIn("no_pts", report.lower())
            self.assertNotIn("without_pts", report.lower())
            self.assertNotIn("legacy", report.lower())
            with (output_dir / "results.json").open(encoding="utf-8") as handle:
                json.load(handle)


if __name__ == "__main__":
    unittest.main()
