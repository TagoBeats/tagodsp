import numpy as np
import pytest

from tagodsp.spectral.stft import Stft, band_rms


def test_perfect_reconstruction():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(48000)
    stft = Stft(n_fft=1024, hop=256)
    y = stft.inverse(stft.forward(x), len(x))
    err = np.max(np.abs(x - y))
    assert 20 * np.log10(err) < -120


def test_shapes_and_axes():
    sr = 48000
    stft = Stft(n_fft=1024, hop=256)
    S = stft.forward(np.zeros(sr))
    assert S.shape[0] == stft.n_bins == 513
    freqs = stft.freqs(sr)
    assert freqs[0] == 0 and np.isclose(freqs[-1], sr / 2)
    times = stft.times(S.shape[1], sr)
    assert np.isclose(times[1] - times[0], 256 / sr)


def test_sine_lands_in_correct_bin():
    sr = 48000
    stft = Stft(n_fft=1024, hop=256)
    f = 1500.0
    t = np.arange(sr) / sr
    S = np.abs(stft.forward(np.sin(2 * np.pi * f * t)))
    freqs = stft.freqs(sr)
    mean_mag = S[:, 10:-10].mean(axis=1)
    assert abs(freqs[np.argmax(mean_mag)] - f) < sr / 1024


def test_band_rms_isolates_band():
    sr = 48000
    stft = Stft()
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 8000 * t)
    S = np.abs(stft.forward(x))
    freqs = stft.freqs(sr)
    hi = band_rms(S, freqs, 6000, 13000).mean()
    lo = band_rms(S, freqs, 100, 4000).mean()
    assert 20 * np.log10(hi / lo) > 40


def test_invalid_params():
    with pytest.raises(ValueError):
        Stft(n_fft=1024, hop=2048)
