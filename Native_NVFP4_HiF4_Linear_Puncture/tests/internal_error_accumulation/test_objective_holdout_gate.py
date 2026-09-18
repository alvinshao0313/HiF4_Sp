"""Regressions for loss accounting, paired units, and actual Top-K selection."""
import pytest

from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.objective_holdout import paired_gain, repair_legacy_validation
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.topk_validate import _sequential_current_state_recapture, validate_topk_selection, require_structural_completion
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.selected_layer_objective_trainer import accumulate_validation_metrics
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.run_state import atomic_write_json, read_json


def test_loss_is_not_double_counted():
    totals = {}
    accumulate_validation_metrics(totals, .25, {'loss': .25, 'final_logit_kl': .25})
    accumulate_validation_metrics(totals, .75, {'loss': .75, 'final_logit_kl': .75})
    assert totals == {'loss': 1., 'final_logit_kl': 1.}


def test_legacy_correction_preserves_archive_and_is_idempotent(tmp_path):
    p = tmp_path / '60_objective/objective_candidates/L31/O2_M/val_metrics.json'
    original = {'loss_name':'O2_M', 'loss':.4, 'cumulative_residual_nmse':.2, 'router_causal_contribution':.8}
    atomic_write_json(p, original)
    repair_legacy_validation(tmp_path)
    repair_legacy_validation(tmp_path)
    assert read_json(p)['loss'] == .2
    assert read_json(p)['router_causal_contribution'] is None
    assert read_json(p.with_name('val_metrics.schema1.json')) == original


def test_topk_ranking_precedes_execution_order():
    protocol = _sequential_current_state_recapture(run_root=None, topk_layers=[41,43,39,31,19], k=4)
    assert protocol['ordered_layers'] == [31,39,41,43]


def test_invalid_k_does_not_create_smaller_candidate():
    with pytest.raises(ValueError):
        _sequential_current_state_recapture(run_root=None, topk_layers=[41,43,39,31], k=8)


def test_s5_rejects_unreviewed_gate():
    with pytest.raises(RuntimeError, match='ALLOW_TOPK'):
        validate_topk_selection({'status':'WAITING_REVIEW'}, {})


def _rows(value):
    return [{'calibration_sample_id':str(i), 'source':'wikitext2' if i<8 else 's1k_original',
             'prefix_length_j':j, 'kl':value} for i in range(16) for j in (8,32,64,128)]


def test_pairing_uses_samples_and_rejects_missing_state():
    result = paired_gain(_rows(2), _rows(1), 'kl')
    assert result['n'] == 16
    assert result['mean'] == 1
    assert result['stable']
    with pytest.raises(RuntimeError, match='missing/duplicate'):
        paired_gain(_rows(2), _rows(1)[:-1], 'kl')


def test_source_regression_cannot_be_hidden_by_pooled_mean():
    candidate = _rows(1)
    for row in candidate:
        if row['source'] == 's1k_original':
            row['kl'] = 2.1
    assert not paired_gain(_rows(2), candidate, 'kl')['stable']


def test_complete_claim_without_capture_is_rejected(tmp_path):
    result = {'variants': {'E1': {'status':'COMPLETE', 'n_states':64}}}
    root = tmp_path / '70_variant_validation'
    atomic_write_json(root / 'variant_validation_results.json', result)
    (root / 'E1_E4_STRUCTURAL_VALIDATION.md').write_text('COMPLETE')
    with pytest.raises(RuntimeError, match='no actual capture'):
        require_structural_completion(tmp_path, result, {'E1'})
