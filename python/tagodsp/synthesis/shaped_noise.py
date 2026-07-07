"""Shaped-noise resynthesis.

Random-phase resynthesis of a segment: keeps the (frequency-smoothed)
time-frequency magnitude envelope, discards phase and fine structure.
Useful as an energy-correct synthetic fill for damaged or excessive signal
portions. Output scaling is the caller's job.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from tagodsp.spectral.stft import Stft


def shaped_noise(
    seg: np.ndarray,
    rng: np.random.Generator,
    n_fft: int = 1024,
    freq_smooth_bins: float = 2.0,
) -> np.ndarray:
    """Random-phase resynthesis of `seg`, same length as input."""
    seg = np.asarray(seg, dtype=np.float64)
    if seg.ndim != 1:
        raise ValueError("expected 1D input")
    use_fft = n_fft if len(seg) >= n_fft else 512
    stft = Stft(n_fft=use_fft, hop=use_fft // 4)
    S = stft.forward(seg)
    mag = np.abs(S)
    if freq_smooth_bins > 0:
        mag = gaussian_filter1d(mag, sigma=freq_smooth_bins, axis=0)
    phase = rng.uniform(-np.pi, np.pi, S.shape)
    return stft.inverse(mag * np.exp(1j * phase), len(seg))
