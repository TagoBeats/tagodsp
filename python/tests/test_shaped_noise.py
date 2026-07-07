import numpy as np

from tagodsp.spectral.stft import Stft, band_rms
from tagodsp.synthesis.shaped_noise import shaped_noise

SR = 48000


def _bandlimited_noise(lo, hi, n, seed=1):
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(seed)
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfilt(sos, rng.standard_normal(n))


def test_same_length():
    seg = _bandlimited_noise(6000, 12000, 4800)
    out = shaped_noise(seg, np.random.default_rng(2))
    assert len(out) == len(seg)


def test_preserves_spectral_envelope():
    seg = _bandlimited_noise(6000, 12000, SR)
    out = shaped_noise(seg, np.random.default_rng(2))
    stft = Stft()
    freqs = stft.freqs(SR)
    # absolute scaling is the caller's job — normalize, then compare the shape
    out = out * (np.sqrt(np.mean(seg**2)) / np.sqrt(np.mean(out**2)))
    S_in = np.abs(stft.forward(seg))
    S_out = np.abs(stft.forward(out))
    e_in = band_rms(S_in, freqs, 6000, 12000).mean()
    e_out = band_rms(S_out, freqs, 6000, 12000).mean()
    assert abs(20 * np.log10(e_out / e_in)) < 1.0
    # synthesis stays band-limited: out-of-band well below in-band
    oob = band_rms(S_out, freqs, 100, 3000).mean()
    assert 20 * np.log10(oob / e_out) < -20.0


def test_decorrelated_from_input():
    seg = _bandlimited_noise(6000, 12000, SR)
    out = shaped_noise(seg, np.random.default_rng(2))
    corr = np.corrcoef(seg, out)[0, 1]
    assert abs(corr) < 0.1


def test_short_segment_fallback():
    seg = _bandlimited_noise(6000, 12000, 600)  # shorter than n_fft=1024
    out = shaped_noise(seg, np.random.default_rng(2))
    assert len(out) == 600 and np.all(np.isfinite(out))
