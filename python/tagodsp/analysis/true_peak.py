"""True peak and inter-sample peak measurement per ITU-R BS.1770-4.

A sample peak only sees the sampled values. The reconstructed band-limited
waveform can rise above them between samples, and that maximum is the true
peak, in dBTP. Worst case for a single tone: a sine at fs/4 sampled at 45
degrees phase has all samples at sin(pi/4) = 0.7071 (-3.01 dBFS) while the
reconstruction reaches 1.0 (0 dBTP).

Method: oversample, then take the peak of the oversampled signal. BS.1770-4
requires at least 4x. The default here is 16x because a 4x meter underestimates
the true maximum when it falls between two of its support points; the error
grows towards Nyquist. Resampling goes through utils/resampling, the same path
the processing chain uses, so measurement and processing interpolate alike.

Source: ITU-R BS.1770-4, Annex 2 (true-peak level measurement).
Concept note: docs/concepts/true_peak.md
"""

import numpy as np

from tagodsp.utils.gain import DB_MIN, lin_to_db
from tagodsp.utils.resampling import FILTER_HALF_LEN, check_factor, upsample

_VALID_OVERSAMPLE = (4, 8, 16, 32)

# Input samples discarded at each end before taking any maximum. The resampler
# assumes silence outside the buffer, so its first and last output samples
# reconstruct a step into the signal rather than the signal itself, and that
# step rings: a sine at fs/4 measures +0.11 dBTP at the edges where the analytic
# answer is exactly 0, while the interior lands on 1.0000006.
#
# The interpolation filter reaches FILTER_HALF_LEN input samples, so that is the
# floor; the rest is margin. The cost is a blind spot of this many samples
# (0.36 ms at 44.1 kHz) at each end. Shorter buffers are measured whole and may
# read high. Padding instead of trimming does not work: odd reflection is only
# smooth for slowly varying signals and makes fs/4 content worse, not better
# (measured +0.14 dB), and scipy's own pad types reach +0.29 dB.
_EDGE_TRIM = FILTER_HALF_LEN + 6


def _views(x: np.ndarray, oversample: int) -> tuple[np.ndarray, np.ndarray]:
    """The stretch that can honestly be measured, at both rates.

    Returned together rather than trimmed twice by the caller, so the base-rate
    and oversampled views cannot drift apart: comparing a peak taken over one
    span with a peak taken over another is how an overshoot turns negative.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim not in (1, 2):
        raise ValueError(f"x must be 1-D or 2-D, got {x.ndim}-D")
    up = upsample(x, check_factor(oversample, _VALID_OVERSAMPLE))
    if x.shape[-1] <= 2 * _EDGE_TRIM:
        return x, up
    trim = _EDGE_TRIM
    return x[..., trim:-trim], up[..., trim * oversample : -trim * oversample]


def peaks_db(
    x: np.ndarray, oversample: int = 16, floor_db: float = DB_MIN
) -> tuple[float, float]:
    """Sample peak and true peak of the same stretch, in dB.

    Both from one resampling pass, which is the expensive part.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float(floor_db), float(floor_db)
    base, up = _views(x, oversample)
    sample_peak = float(np.max(np.abs(base))) if base.size else 0.0
    return (
        float(lin_to_db(sample_peak, floor_db=floor_db)),
        float(lin_to_db(float(np.max(np.abs(up))), floor_db=floor_db)),
    )


def true_peak(x: np.ndarray, oversample: int = 16) -> float:
    """Linear true-peak amplitude. 1-D, or 2-D with time on the last axis."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(_views(x, oversample)[1])))


def true_peak_db(x: np.ndarray, oversample: int = 16, floor_db: float = DB_MIN) -> float:
    """True-peak level in dBTP."""
    return peaks_db(x, oversample=oversample, floor_db=floor_db)[1]


def isp_overshoot_db(x: np.ndarray, oversample: int = 16) -> float:
    """How far the true peak exceeds the sample peak, in dB.

    Zero for signals whose reconstruction stays below the sampled maxima,
    positive whenever inter-sample peaks exist. This is the number that decides
    whether a clipper needs a true-peak-aware output stage.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.any(x):
        return 0.0
    sample_peak_db, tp_db = peaks_db(x, oversample=oversample)
    return tp_db - sample_peak_db
