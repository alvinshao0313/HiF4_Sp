import importlib
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

module = importlib.import_module("nvfp4_hif4_torch")

HIFLOAT4_GPU_PATH = Path("/home/shaoyuantian/program/HiF4_Sp/HiFloat4/hif4_gpu")
PROJECT_ROOT = Path("/home/shaoyuantian/program/HiF4_Sp")


def _try_import_hifloat4():
    if not HIFLOAT4_GPU_PATH.is_dir():
        return None
    if str(HIFLOAT4_GPU_PATH) not in sys.path:
        sys.path.insert(0, str(HIFLOAT4_GPU_PATH))
    try:
        from quant_cy import QType, quant_dequant_float  # noqa: F401
        from quant_cy.base.QFuncs.hifx import quant_hifx  # noqa: F401

        return QType, quant_dequant_float, quant_hifx
    except Exception:
        return None


def _try_import_nvfp4():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from NVFP4.torch_fake import cast_to_fp4_e2m1, cast_to_fp8_e4m3fn  # noqa: F401

        return cast_to_fp4_e2m1, cast_to_fp8_e4m3fn
    except Exception:
        return None


class CrossCheckHiF4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hifloat4 = _try_import_hifloat4()
        if cls.hifloat4 is None:
            raise unittest.SkipTest("HiFloat4 quant_cy imports unavailable")

    def test_hardware_hif4_matches_quant_hifx(self) -> None:
        QType, quant_dequant_float, quant_hifx = self.hifloat4
        x = torch.randn(4, 128, generator=torch.Generator().manual_seed(42))
        greenfield = module.quantize_hif4(
            x,
            config=module.HiF4Config(scale_mode="hardware", group_size=64, group_dim=-1),
        ).values

        qtype = QType("hifx4").dim(-1)
        try:
            reference = quant_dequant_float(x, qtype, force_py=True, force_fp32=True)
        except Exception:
            reference = quant_hifx(x, qtype, qdim=-1)

        close = torch.isclose(greenfield, reference, rtol=1e-5, atol=1e-4)
        match_fraction = close.float().mean().item()
        self.assertGreater(
            match_fraction,
            0.99,
            msg=f"only {match_fraction * 100:.2f}% elements matched HiFloat4 reference",
        )


class CrossCheckNVFP4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nvfp4 = _try_import_nvfp4()
        if cls.nvfp4 is None:
            raise unittest.SkipTest("NVFP4.torch_fake imports unavailable")

    def test_e2m1_magnitudes_match_existing(self) -> None:
        cast_to_fp4_e2m1, _ = self.nvfp4
        x = torch.randn(256, generator=torch.Generator().manual_seed(7)).abs()
        greenfield = module.quantize_e2m1_magnitude(x)
        existing = cast_to_fp4_e2m1(x.to(torch.float32)).abs()
        torch.testing.assert_close(greenfield, existing, rtol=0, atol=0)

    def test_e4m3fn_positive_rounding_matches_existing(self) -> None:
        _, cast_to_fp8_e4m3fn = self.nvfp4
        x = torch.rand(512, generator=torch.Generator().manual_seed(9)) * 400.0 + 1e-6
        values, codes = module.build_e4m3fn_codebook()
        greenfield = module.round_positive_to_codebook(x, values, codes)
        existing = cast_to_fp8_e4m3fn(x.to(torch.float32)).abs()
        torch.testing.assert_close(greenfield, existing, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
