from types import SimpleNamespace

import pytest

from Native_NVFP4_HiF4_Linear_Puncture.experiments.progressive_error_cancellation.config import Config


def test_formal_protocol_is_frozen():
    cfg = Config(output_dir="/tmp/run", method="direct", lambda_direction=.1)
    cfg.validate()
    assert cfg.matrix_sharing == "group"
    assert cfg.router_loss == "top_mass"
    assert len(cfg.layer_ids) == 48
    assert cfg.calib_nsamples == 128 and cfg.calib_val_nsamples == 32 and cfg.calib_holdout_nsamples == 32


@pytest.mark.parametrize("method,lam", [("baseline", 0.), ("direct", .1), ("jvp", .3), ("shuffled", .3)])
def test_method_configuration(method, lam):
    Config(output_dir="/tmp/run", method=method, lambda_direction=lam).validate()


def test_baseline_rejects_direction_weight():
    with pytest.raises(ValueError):
        Config(output_dir="/tmp/run", method="baseline", lambda_direction=.1).validate()
