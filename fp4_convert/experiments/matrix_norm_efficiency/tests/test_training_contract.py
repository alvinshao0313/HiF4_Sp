"""CPU checks for diagnostic isolation and preservation of complete samples."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from ...e2e_diag_reconstruction.data.calibration import CalibrationSample
from ..artifacts import write_json
from ..config import BATCH_SIZE, CANDIDATES, LAYERS
from ..training import selected_samples, training_dir, training_spec


def samples(prefix):
    return [CalibrationSample(f"{prefix}-{length}", length, torch.arange(length),
                              torch.arange(length).remainder(2))
            for length in (3, 8, 5, 9, 2, 7)]


class TrainingContractTests(unittest.TestCase):
    def test_qualification_keeps_full_longest_samples(self):
        training, validation = samples("train"), samples("val")
        short_train, short_val = selected_samples(training, validation, qualification=True)
        for original, selected in ((training, short_train), (validation, short_val)):
            self.assertEqual(len(selected), BATCH_SIZE)
            self.assertEqual([s.input_ids.numel() for s in selected], [9, 8, 7, 5])
            for sample in selected:
                self.assertTrue(any(sample is source for source in original))
                self.assertTrue(torch.equal(sample.loss_mask, torch.arange(sample.input_ids.numel()).remainder(2)))

    def test_formal_selection_does_not_replace_fixed_validation(self):
        training, validation = samples("train"), samples("val")
        actual_train, actual_val = selected_samples(training, validation, qualification=False)
        self.assertIs(actual_train, training)
        self.assertIs(actual_val, validation)
        with self.assertRaises(ValueError):
            selected_samples(training[:3], validation, qualification=True)

    def test_completed_training_cannot_be_confused_with_qualification(self):
        candidate = next(iter(CANDIDATES))
        _, epochs, layers = training_spec(candidate, False)
        self.assertEqual(layers, LAYERS)
        self.assertEqual(training_spec(candidate, True)[1:], (2, 2))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "training" / candidate
            manifest = {"complete": True, "candidate": candidate, "qualification": False,
                        "epochs": epochs, "completed_layers": list(range(layers))}
            write_json(manifest, directory / "manifest.json")
            self.assertEqual(training_dir(root, candidate), directory)
            for changed in ({"qualification": True}, {"completed_layers": [0, 1]}, {"epochs": epochs + 1}):
                write_json({**manifest, **changed}, directory / "manifest.json")
                with self.assertRaises(ValueError):
                    training_dir(root, candidate)


if __name__ == "__main__":
    unittest.main()
