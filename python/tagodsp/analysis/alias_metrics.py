"""Spectral purity metrics for nonlinear processors.

A waveshaper driven by a sine produces harmonics at k*f0. Everything else in
the spectrum is unwanted: aliased images folded back from above Nyquist, plus
whatever noise the processing adds. The ratio of the two is the number that
says how clean a clipper is.

    alias_nmr_db = 10*log10( energy outside the harmonic bins / energy at f0 )

Lower is cleaner. The measurement uses a Blackman-Harris window because its
sidelobes sit near -92 dB, far below the aliasing levels of interest; a Hann
window would leak enough energy out of the harmonic bins to look like alias.
The main lobe is correspondingly wide, so a band of +-_TOL_BINS around every
harmonic is excluded.

Source: standard noise-to-mask style ratio, as used for antialiasing
comparisons in Parker et al., "Reducing the Aliasing of Nonlinear Waveshaping
Using Continuous-Time Convolution" (DAFx-16).
"""

import numpy as np
from scipy.signal.windows import blackmanharris

# Blackman-Harris has an 8-bin main lobe; 8 bins each side covers it with margin.
_TOL_BINS = 8


def _spectrum(x: np.ndarray) -> np.ndarray:
    """Power spectrum of a single channel under a Blackman-Harris window."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got {x.ndim}-D")
    w = blackmanharris(len(x))
    return np.abs(np.fft.rfft(x * w)) ** 2


def _bins_near(freqs: np.ndarray, target: float, tol_bins: int) -> np.ndarray:
    """Boolean mask of the bins within tol_bins of target."""
    idx = int(np.argmin(np.abs(freqs - target)))
    mask = np.zeros(freqs.shape, dtype=bool)
    lo = max(idx - tol_bins, 0)
    mask[lo : idx + tol_bins + 1] = True
    return mask


def alias_nmr_db(y: np.ndarray, sr: int, f0: float, tol_bins: int = _TOL_BINS) -> float:
    """Energy that is neither the fundamental nor one of its harmonics, in dB.

    Relative to the energy of the fundamental. DC is excluded as well, since an
    asymmetric nonlinearity can add an offset that is not aliasing.
    """
    power = _spectrum(y)
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    nyquist = sr / 2

    unwanted = np.ones(freqs.shape, dtype=bool)
    unwanted &= ~_bins_near(freqs, 0.0, tol_bins)
    k = 1
    while k * f0 < nyquist:
        unwanted &= ~_bins_near(freqs, k * f0, tol_bins)
        k += 1

    fundamental = float(np.sum(power[_bins_near(freqs, f0, tol_bins)]))
    if fundamental <= 0.0:
        raise ValueError("no energy at f0, cannot form a ratio")
    return float(10.0 * np.log10(np.sum(power[unwanted]) / fundamental + 1e-300))


def harmonic_level_db(y: np.ndarray, sr: int, freq: float, tol_bins: int = _TOL_BINS) -> float:
    """Level of the spectral component at freq, relative to the strongest bin."""
    power = _spectrum(y)
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    band = float(np.sum(power[_bins_near(freqs, freq, tol_bins)]))
    return float(10.0 * np.log10(band / np.max(power) + 1e-300))
