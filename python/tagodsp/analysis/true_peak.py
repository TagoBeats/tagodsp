"""True peak and inter-sample peak measurement per ITU-R BS.1770-4.

A sample peak only sees the sampled values. The reconstructed band-limited
waveform can rise above them between samples, and that maximum is the true
peak, in dBTP. Worst case for a single tone: a sine at fs/4 sampled at 45
degrees phase has all samples at sin(pi/4) = 0.7071 (-3.01 dBFS) while the
reconstruction reaches 1.0 (0 dBTP).

Method: oversample, then take the peak of the oversampled signal. BS.1770-4
requires at least 4x. The default here is 16x because a 4x meter underestimates
the true maximum when it falls between two of its support points; the error
grows towards Nyquist. Resampling uses the same polyphase path as
distortion/clipper.py (Kaiser beta 12, roughly 115 dB stopband) so that the
measurement and the processing chain interpolate identically.

Source: ITU-R BS.1770-4, Annex 2 (true-peak level measurement).
Concept note: docs/concepts/true_peak.md
"""

import numpy as np
from scipy.signal import resample_poly

from tagodsp.utils.gain import DB_MIN, lin_to_db, peak_db

# Same resampling window as distortion/clipper.py, on purpose.
_RESAMPLE_WINDOW = ("kaiser", 12.0)

_VALID_OVERSAMPLE = (4, 8, 16, 32)

# Input samples discarded at each end before taking the maximum. The resampler
# assumes silence outside the buffer, so its first and last output samples
# reconstruct a step into the signal rather than the signal itself, and that
# step rings: a sine at fs/4 measures +0.11 dBTP at the edges where the analytic
# answer is exactly 0, while the interior lands on 1.0000006. scipy's filter
# reaches 10 input samples, 16 covers it with margin.
#
# The cost is a blind spot of 16 samples (0.36 ms at 44.1 kHz) at each end of
# the buffer. Buffers of 2*_EDGE_TRIM samples or shorter are measured whole and
# may read high. Padding instead of trimming does not work: odd reflection is
# only smooth for slowly varying signals and makes fs/4 content worse, not
# better (measured +0.14 dB).
_EDGE_TRIM = 16


def _measured_region(x: np.ndarray, factor: int) -> np.ndarray:
    """Drop the resampler's transient zone at both ends, scaled by `factor`.

    factor is 1 for the original rate and the oversampling ratio for the
    upsampled signal, so both views cover the same stretch of time.
    """
    if x.shape[-1] > 2 * _EDGE_TRIM * factor:
        trim = _EDGE_TRIM * factor
        return x[..., trim:-trim]
    return x


def true_peak(x: np.ndarray, oversample: int = 16) -> float:
    """Linear true-peak amplitude of x, measured on an oversampled copy.

    x is 1-D (mono) or 2-D with time along the last axis. Returns the maximum
    absolute value across all channels, in linear amplitude.
    """
    if oversample not in _VALID_OVERSAMPLE:
        raise ValueError(f"oversample must be one of {_VALID_OVERSAMPLE}, got {oversample}")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim not in (1, 2):
        raise ValueError(f"x must be 1-D or 2-D, got {x.ndim}-D")
    if x.size == 0:
        return 0.0
    up = resample_poly(x, oversample, 1, axis=-1, window=_RESAMPLE_WINDOW)
    return float(np.max(np.abs(_measured_region(up, oversample))))


def true_peak_db(x: np.ndarray, oversample: int = 16, floor_db: float = DB_MIN) -> float:
    """True-peak level in dBTP."""
    return float(lin_to_db(true_peak(x, oversample=oversample), floor_db=floor_db))


def isp_overshoot_db(x: np.ndarray, oversample: int = 16) -> float:
    """How far the true peak exceeds the sample peak, in dB.

    Zero for signals whose reconstruction stays below the sampled maxima,
    positive whenever inter-sample peaks exist. This is the number that decides
    whether a clipper needs a true-peak-aware output stage.

    Both peaks are taken over the same stretch of time: the sample peak also
    ignores the edge samples that true_peak cannot measure, otherwise a peak
    sitting in the trimmed zone would produce a negative overshoot.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.any(x):
        return 0.0
    return true_peak_db(x, oversample=oversample) - peak_db(_measured_region(x, 1))
