"""CPU checks of real registration/config parsing, including a spawn worker."""
import copy
import json
import multiprocessing
import unittest

from . import entry  # Registration is deliberately at module import time.
from .artifacts import tensor_record
from .convert import SOURCE
from .format import FORMAT_VERSION, SCALE_RULE
from .transform import r16x2_matrix
from .runtime import Qwen3RoutedMXFP4Config
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config


def child_registration(connection):
    connection.send(get_quantization_config("qwen3_routed_mxfp4_r16x2").__name__)
    connection.close()


class IntegrationTests(unittest.TestCase):
    def test_both_modelopt_schemas_retain_native_config(self):
        sources = [json.loads((SOURCE / "config.json").read_text())["quantization_config"],
                   json.loads((SOURCE / "hf_quant_config.json").read_text())]
        for original in sources:
            native = ModelOptNvFp4Config.from_config(original)
            for variant in ("r16x2",):
                config = copy.deepcopy(original)
                config["qwen3_routed_mxfp4_r16x2"] = {"format": FORMAT_VERSION, "variant": variant, "group_size": 32,
                    "scale_rule": SCALE_RULE}
                if variant == "r16x2":
                    with self.assertRaises(ValueError):
                        Qwen3RoutedMXFP4Config.from_config(config)
                    config["qwen3_routed_mxfp4_r16x2"]["r16x2_sha256"] = tensor_record(r16x2_matrix())["sha256"]
                converted = Qwen3RoutedMXFP4Config.from_config(config)
                self.assertEqual(converted.group_size, 16)
                for name in ("kv_cache_quant_algo", "exclude_modules", "is_checkpoint_nvfp4_serialized", "LinearMethodCls", "KVCacheMethodCls"):
                    self.assertEqual(getattr(converted, name), getattr(native, name))
                self.assertEqual(converted.variant, variant)
                self.assertEqual(Qwen3RoutedMXFP4Config.override_quantization_method(config, None), "qwen3_routed_mxfp4_r16x2")
                config["qwen3_routed_mxfp4_r16x2"]["scale_rule"] = "even"
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
