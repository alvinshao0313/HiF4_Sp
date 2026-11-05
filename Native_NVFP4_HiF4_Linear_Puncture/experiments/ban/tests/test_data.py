"""Checks for C4 window boundaries and isolation of the caller's RNG state."""

import random

import pytest
import torch

from ..data import _sample_windows


def test_window_sampling_is_reproducible_and_does_not_change_global_rng():
    stream = torch.arange(10).reshape(1, 10)
    before = random.getstate()
    first, starts = _sample_windows(stream, nsamples=6, seqlen=4, seed=0)
    second, repeated_starts = _sample_windows(stream, nsamples=6, seqlen=4, seed=0)
    assert random.getstate() == before
    assert starts == repeated_starts == [6, 3, 6, 3, 0, 2]
    assert torch.equal(first, second)
    assert first.shape == (6, 4)
    assert first.dtype == torch.long
    # The first window reaches the final token: randint's upper bound is inclusive.
    assert first[0].tolist() == [6, 7, 8, 9]
    assert first[4].tolist() == [0, 1, 2, 3]


def test_window_sampling_accepts_exact_length_and_rejects_short_stream():
    stream = torch.arange(4).reshape(1, 4)
    samples, starts = _sample_windows(stream, nsamples=2, seqlen=4, seed=0)
    assert starts == [0, 0]
    assert torch.equal(samples, stream.expand(2, -1))
    with pytest.raises(ValueError, match="shorter than seqlen"):
        _sample_windows(stream, nsamples=1, seqlen=5, seed=0)
