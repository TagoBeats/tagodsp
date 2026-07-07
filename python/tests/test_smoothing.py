import numpy as np
import pytest

from tagodsp.utils.smoothing import OnePoleSmoother


def test_step_reaches_63_percent_after_tau():
    sr = 48000
    tau = 0.01
    sm = OnePoleSmoother(tau_s=tau, sr=sr, initial=0.0)
    n = int(tau * sr)
    y = sm.process(np.ones(n))
    assert np.isclose(y[-1], 1 - np.exp(-1), atol=1e-2)


def test_zero_tau_passes_through():
    sm = OnePoleSmoother(tau_s=0.0, sr=48000)
    x = np.array([1.0, -0.5, 0.25])
    np.testing.assert_allclose(sm.process(x), x)


def test_reset():
    sm = OnePoleSmoother(tau_s=0.01, sr=48000, initial=0.5)
    sm.process(np.ones(100))
    sm.reset()
    y = sm.process(np.array([0.5]))
    assert np.isclose(y[0], 0.5)


def test_monotone_rise_on_step():
    sm = OnePoleSmoother(tau_s=0.005, sr=48000)
    y = sm.process(np.ones(200))
    assert np.all(np.diff(y) > 0)
    assert y[-1] < 1.0


def test_invalid_args():
    with pytest.raises(ValueError):
        OnePoleSmoother(tau_s=-1, sr=48000)
    with pytest.raises(ValueError):
        OnePoleSmoother(tau_s=0.01, sr=0)
