import numpy as np

from tagodsp.spectral.band_ratio_gate import BandRatioGate
from tagodsp.spectral.soft_mask import SoftMaskSeparator
from tagodsp.spectral.stft import Stft, band_rms
from tagodsp.utils.gain import rms

SR = 48000


def _separate(x):
    det = BandRatioGate(sr=SR).process(x)
    sep = SoftMaskSeparator(sr=SR)
    return det, sep.separate(x, det.gate)


def test_perfect_reconstruction(sibilant_like_signal):
    x, _ = sibilant_like_signal
    _, res = _separate(x)
    assert res.peak_delta_dbfs < -100


def test_masked_stream_is_highband(sibilant_like_signal):
    x, _ = sibilant_like_signal
    _, res = _separate(x)
    stft = Stft()
    freqs = stft.freqs(SR)
    S = np.abs(stft.forward(res.x_masked))
    hi = band_rms(S, freqs, 6000, 13000).mean()
    lo = band_rms(S, freqs, 100, 4000).mean()
    assert 20 * np.log10(hi / lo) > 12


def test_tone_untouched_outside_events(sibilant_like_signal):
    x, spans = sibilant_like_signal
    _, res = _separate(x)
    # region far away from any burst: masked stream ~ silent, rest ~ input
    s, e = int(0.05 * SR), int(0.35 * SR)
    assert rms(res.x_masked[s:e]) < 1e-4
    assert np.allclose(res.x_rest[s:e], x[s:e], atol=1e-4)


def test_mask_range(sibilant_like_signal):
    x, _ = sibilant_like_signal
    _, res = _separate(x)
    assert res.mask.min() >= 0.0 and res.mask.max() <= 1.0
