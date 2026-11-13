"""CPU checks of real registration/config parsing, including a spawn worker."""
import copy
import json
import multiprocessing
import unittest

from . import entry  # Registration is deliberately at module import time.
from .artifacts import tensor_record
from ..qwen3_mxfp4_dualscale.convert import SOURCE
from ..qwen3_mxfp4_dualscale.format import FORMAT_VERSION, SCALE_RULE, VARIANTS, MARKER, group_size
from .runtime import Qwen3RoutedMXFP4Config
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config


def child_registration(connection):
    connection.send(get_quantization_config(MARKER).__name__)
    connection.close()


class IntegrationTests(unittest.TestCase):
    def test_both_modelopt_schemas_retain_native_config(self):
        sources = [json.loads((SOURCE / "config.json").read_text())["quantization_config"],
                   json.loads((SOURCE / "hf_quant_config.json").read_text())]
        for original in sources:
            native = ModelOptNvFp4Config.from_config(original)
            for variant in ("dualscale_g512",):
                config = copy.deepcopy(original)
                config[MARKER] = {"format": FORMAT_VERSION, "variant": variant, "group_size": 32,
                    "scale_rule": SCALE_RULE}
                with self.assertRaises(ValueError):
                    Qwen3RoutedMXFP4Config.from_config(config)
                config[MARKER].update(algorithm="DualScale MXFP4", outer_group_size=group_size(variant), outer_scale="FP32(absmax/6); zero=1", rotation="none")
                converted = Qwen3RoutedMXFP4Config.from_config(config)
                self.assertEqual(converted.group_size, 16)
                for name in ("kv_cache_quant_algo", "exclude_modules", "is_checkpoint_nvfp4_serialized", "LinearMethodCls", "KVCacheMethodCls"):
                    self.assertEqual(getattr(converted, name), getattr(native, name))
                self.assertEqual(converted.variant, variant)
                self.assertEqual(Qwen3RoutedMXFP4Config.override_quantization_method(config, None), MARKER)
                config[MARKER]["scale_rule"] = "even"
                with self.assertRaises(ValueError):
                    Qwen3RoutedMXFP4Config.from_config(config)
            self.assertIsNone(Qwen3RoutedMXFP4Config.override_quantization_method(original, None))

    def test_spawn_registration(self):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=child_registration, args=(child,))
        process.start()
        child.close()
        try:
            self.assertTrue(parent.poll(60), "Spawn registration did not finish")
            self.assertEqual(parent.recv(), "Qwen3RoutedMXFP4Config")
            process.join(10)
            self.assertEqual(process.exitcode, 0)
        finally:
            parent.close()
            if process.is_alive():
                process.terminate()  # Only this test's own temporary child.
                process.join()


if __name__ == "__main__":
    unittest.main()
