import numpy as np
import pytest

pytest.importorskip("librosa")

from tagodsp.analysis.voicing import classify_events  # noqa: E402

SR = 48000
HOP = 256


def test_voiced_vs_unvoiced():
    n = SR
    t = np.arange(n) / SR
    voiced_sig = 0.3 * np.sin(2 * np.pi * 150 * t)
    noise = np.random.default_rng(0).standard_normal(n) * 0.1
    x = np.concatenate([voiced_sig, noise])
    n_frames = len(x) // HOP + 1
    events = [(0, n), (n, 2 * n)]
    res = classify_events(x, events, SR, HOP, n_frames)
    assert res[0].voiced is True
    assert res[1].voiced is False


def test_empty_event():
    x = np.zeros(SR)
    res = classify_events(x, [(100, 100)], SR, HOP, SR // HOP)
    assert res[0].voiced is False and res[0].voiced_frac == 0.0
