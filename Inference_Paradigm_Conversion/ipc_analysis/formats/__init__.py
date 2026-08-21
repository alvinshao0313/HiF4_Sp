"""Format adapters: source weight fingerprint + NVFP4/HiF4/MXFP8 QDQ views."""

from Inference_Paradigm_Conversion.ipc_analysis.formats.fingerprint import (
    QuantizedTensorView,
    load_nvfp4_qat_dequant_weight,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
from Inference_Paradigm_Conversion.ipc_analysis.formats.mxfp8 import quantize_mxfp8_activation
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import (
    load_nvfp4_activation_scales,
    quantize_nvfp4_activation,
    resolve_nvfp4_scale_for_module,
)

__all__ = [
    "QuantizedTensorView",
    "load_nvfp4_qat_dequant_weight",
    "quantize_hif4_tensor",
    "quantize_mxfp8_activation",
    "load_nvfp4_activation_scales",
    "quantize_nvfp4_activation",
    "resolve_nvfp4_scale_for_module",
]
