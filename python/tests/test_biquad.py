import numpy as np
import pytest

from tagodsp.filters.biquad import Biquad, highpass, lowpass, magnitude_db, peaking
from tagodsp.utils.gain import rms

SR = 48000


def test_lowpass_minus_3db_at_cutoff():
    c = lowpass(1000, SR)
    assert np.isclose(magnitude_db(c, np.array([1000.0]), SR)[0], -3.0103, atol=0.01)


def test_lowpass_flat_passband_and_rolloff():
    c = lowpass(1000, SR)
    assert np.isclose(magnitude_db(c, np.array([20.0]), SR)[0], 0.0, atol=0.01)
    # 2nd order: ~ -12 dB/octave above cutoff (measured away from Nyquist,
    # where the double zero at z=-1 steepens the digital response)
    m2k, m4k = magnitude_db(c, np.array([2000.0, 4000.0]), SR)
    assert np.isclose(m4k - m2k, -12.0, atol=1.0)


def test_highpass_mirror():
    c = highpass(1000, SR)
    assert np.isclose(magnitude_db(c, np.array([1000.0]), SR)[0], -3.0103, atol=0.01)
    assert np.isclose(magnitude_db(c, np.array([20000.0]), SR)[0], 0.0, atol=0.05)


def test_peaking_gain_at_center():
    c = peaking(2000, SR, gain_db=6.0, q=2.0)
    assert np.isclose(magnitude_db(c, np.array([2000.0]), SR)[0], 6.0, atol=0.01)


def test_time_domain_matches_analytic_magnitude():
    f0, ftest = 1000.0, 4000.0
    c = lowpass(f0, SR)
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * ftest * t)
    y = Biquad(c).process(x)
    # skip transient, compare steady-state gain against analytic response
    gain_db = 20 * np.log10(rms(y[SR // 4 :]) / rms(x[SR // 4 :]))
    expected = magnitude_db(c, np.array([ftest]), SR)[0]
    assert np.isclose(gain_db, expected, atol=0.05)


def test_block_processing_equals_single_pass():
    c = highpass(500, SR)
    rng = np.random.default_rng(42)
    x = rng.standard_normal(4096)
    y_single = Biquad(c).process(x)
    bq = Biquad(c)
    y_blocks = np.concatenate([bq.process(b) for b in np.split(x, 8)])
    np.testing.assert_allclose(y_blocks, y_single, atol=1e-12)


def test_invalid_params():
    with pytest.raises(ValueError):
        lowpass(0, SR)
    with pytest.raises(ValueError):
        lowpass(30000, SR)
    with pytest.raises(ValueError):
        lowpass(1000, SR, q=0)
