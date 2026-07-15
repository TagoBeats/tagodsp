import numpy as np
import pytest

from tagodsp.stereo.mono_low import MonoLow
from tagodsp.utils.gain import rms_db

SR = 44100


def _sine(freq: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return np.sin(2 * np.pi * freq * t)


def _spectrum_db(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = np.hanning(len(x))
    spec = 20 * np.log10(np.abs(np.fft.rfft(x * w)) + 1e-12)
    return np.fft.rfftfreq(len(x), 1 / SR), spec


def test_mono_input_stays_flat_and_mono():
    # LR4 lp + hp must sum allpass-flat, so mono material passes unchanged
    # in magnitude and stays perfectly correlated.
    rng = np.random.default_rng(5)
    mono = rng.standard_normal(4 * SR)
    x = np.column_stack([mono, mono])
    y = MonoLow(freq=120.0, sr=SR).process(x)
    assert np.allclose(y[:, 0], y[:, 1], atol=1e-12)
    freqs, in_spec = _spectrum_db(mono)
    _, out_spec = _spectrum_db(y[:, 0])
    band = (freqs > 30) & (freqs < 18000)
    # smooth both spectra before comparing single noise bins
    kernel = np.ones(51) / 51
    diff = np.convolve(out_spec - in_spec, kernel, mode="same")
    assert np.max(np.abs(diff[band])) < 0.2


def test_antiphase_low_is_attenuated():
    # Side content below the cutoff leaks only through the LR4 highpass slope
    # (24 dB/oct): expect roughly 30 dB at 50 Hz and 45+ dB at 30 Hz for fc=120.
    for freq, min_drop in [(50.0, 25.0), (30.0, 40.0)]:
        s = _sine(freq)
        x = np.column_stack([s, -s])
        y = MonoLow(freq=120.0, sr=SR).process(x)
        assert rms_db(y) - rms_db(x) < -min_drop


def test_antiphase_high_is_preserved():
    s = _sine(5000.0)
    x = np.column_stack([s, -s])
    y = MonoLow(freq=120.0, sr=SR).process(x)
    assert abs(rms_db(y) - rms_db(x)) < 0.1


def test_side_energy_split_at_crossover():
    # wide (uncorrelated) noise: side energy below the cutoff dies,
    # side energy well above survives.
    rng = np.random.default_rng(9)
    x = rng.standard_normal((4 * SR, 2))
    y = MonoLow(freq=200.0, sr=SR).process(x)
    side = (y[:, 0] - y[:, 1]) / 2
    freqs, spec = _spectrum_db(side)
    _, in_spec = _spectrum_db((x[:, 0] - x[:, 1]) / 2)
    kernel = np.ones(201) / 201
    drop = np.convolve(spec - in_spec, kernel, mode="same")
    assert np.mean(drop[(freqs > 30) & (freqs < 60)]) < -30.0
    assert np.mean(drop[(freqs > 2000) & (freqs < 8000)]) > -0.5


def test_reset_makes_process_deterministic():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((SR, 2))
    ml = MonoLow(freq=120.0, sr=SR)
    a = ml.process(x)
    ml.reset()
    b = ml.process(x)
    assert np.array_equal(a, b)


def test_rejects_non_stereo():
    ml = MonoLow(sr=SR)
    with pytest.raises(ValueError):
        ml.process(np.zeros(100))
    with pytest.raises(ValueError):
        ml.process(np.zeros((100, 3)))
