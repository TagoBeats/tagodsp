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


def _band(n_bins: int, hz_per_bin: float, target: float, tol_bins: int) -> slice:
    """The bins within tol_bins of target, as a slice into an rfft spectrum."""
    idx = int(round(target / hz_per_bin))
    return slice(max(idx - tol_bins, 0), min(idx + tol_bins + 1, n_bins))


def alias_nmr_db(y: np.ndarray, sr: int, f0: float, tol_bins: int = _TOL_BINS) -> float:
    """Energy that is neither the fundamental nor one of its harmonics, in dB.

    Relative to the energy of the fundamental. DC is excluded as well, since an
    asymmetric nonlinearity can add an offset that is not aliasing.
    """
    power = _spectrum(y)
    hz_per_bin = sr / len(y)
    nyquist = sr / 2

    # One mask, cleared in place per harmonic. Allocating a full boolean array
    # per harmonic costs O(harmonics * N), which bites at low fundamentals: at
    # 50 Hz that was 12.8 ms against 0.9 ms for the FFT itself.
    unwanted = np.ones(len(power), dtype=bool)
    unwanted[_band(len(power), hz_per_bin, 0.0, tol_bins)] = False
    k = 1
    while k * f0 < nyquist:
        unwanted[_band(len(power), hz_per_bin, k * f0, tol_bins)] = False
        k += 1

    fundamental = float(np.sum(power[_band(len(power), hz_per_bin, f0, tol_bins)]))
    if fundamental <= 0.0:
        raise ValueError("no energy at f0, cannot form a ratio")
    return float(10.0 * np.log10(np.sum(power[unwanted]) / fundamental + 1e-300))


def harmonic_level_db(y: np.ndarray, sr: int, freq: float, tol_bins: int = _TOL_BINS) -> float:
    """Level of the spectral component at freq, relative to the strongest bin."""
    power = _spectrum(y)
    band = float(np.sum(power[_band(len(power), sr / len(y), freq, tol_bins)]))
    return float(10.0 * np.log10(band / np.max(power) + 1e-300))
