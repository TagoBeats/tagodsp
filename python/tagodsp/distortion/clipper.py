"""Clipping waveshapers with optional oversampling.

All curves are odd-symmetric, identity below a threshold t and saturate towards
a fixed ceiling of 1.0. Threshold moves the knee, never the ceiling.

fl_softclip is the exact transfer curve of FL Studio's Fruity Soft Clipper,
reverse-measured 2026-07-15 (ramp bounce, max deviation 4.7e-8 over a 42 s
test bounce, i.e. float32 quantization noise):

    y = x                                          for |x| <= t
    y = sign(x) * (1 - (1-t) * exp(-(|x|-t)/(1-t)))  otherwise

The original quantizes the threshold to integer steps t = p/128 with knob
default p = 100. The original applies the curve without oversampling, which
is exactly the weakness the Clipper class fixes here.
Source: measurement report in ~/Documents/TagoClip/measure/analysis/REPORT.md
"""

import numpy as np
from scipy.signal import resample_poly

from tagodsp.utils.gain import db_to_lin

FL_THRESHOLD_DEFAULT = 100.0 / 128.0

# Kaiser beta 12 gives roughly 115 dB stopband for the resampling filters.
_RESAMPLE_WINDOW = ("kaiser", 12.0)


def fl_softclip(x: np.ndarray, threshold: float = FL_THRESHOLD_DEFAULT) -> np.ndarray:
    """Fruity Soft Clipper curve: linear below threshold, exponential knee above.

    y = sign(x) * (1 - (1-t) * exp(-(|x|-t)/(1-t))) for |x| > t, identity below.
    C1-continuous at the knee, ceiling asymptotically 1.0.
    """
    t = float(threshold)
    if not 0.0 < t < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {t}")
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    knee = 1.0 - (1.0 - t) * np.exp(-(ax - t) / (1.0 - t))
    return np.sign(x) * np.where(ax <= t, ax, knee)


def hardclip(x: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Hard clip at +-threshold. In this family the ceiling equals the threshold."""
    x = np.asarray(x, dtype=np.float64)
    return np.clip(x, -threshold, threshold)


def tanh_clip(x: np.ndarray, threshold: float = FL_THRESHOLD_DEFAULT) -> np.ndarray:
    """Tanh knee with the same parametrization as fl_softclip.

    y = sign(x) * (t + (1-t) * tanh((|x|-t)/(1-t))) for |x| > t, identity below.
    Softer approach to the ceiling than the exponential knee.
    """
    t = float(threshold)
    if not 0.0 < t < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {t}")
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    knee = t + (1.0 - t) * np.tanh((ax - t) / (1.0 - t))
    return np.sign(x) * np.where(ax <= t, ax, knee)


CURVES = {
    "fl": fl_softclip,
    "hard": hardclip,
    "tanh": tanh_clip,
}


class Clipper:
    """Waveshaping clipper with polyphase oversampling around the nonlinearity.

    Upsample by `oversample`, apply the curve, downsample. The decimation filter
    removes harmonics above the original Nyquist before they can alias. With
    oversample=1 and curve="fl" this is bit-faithful to the Fruity Soft Clipper
    (minus its float32 rounding).

    Offline prototype: process() expects the full buffer, reset() clears nothing
    because there is no state yet. Block-based state (resampler tails) is a
    C++ promotion concern.
    """

    def __init__(
        self,
        curve: str = "fl",
        threshold: float = FL_THRESHOLD_DEFAULT,
        oversample: int = 1,
        drive_db: float = 0.0,
    ):
        if curve not in CURVES:
            raise ValueError(f"curve must be one of {sorted(CURVES)}, got {curve!r}")
        if oversample not in (1, 2, 4, 8, 16):
            raise ValueError(f"oversample must be 1, 2, 4, 8 or 16, got {oversample}")
        self.curve = curve
        self.threshold = float(threshold)
        self.oversample = int(oversample)
        self.drive_db = float(drive_db)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64) * db_to_lin(self.drive_db)
        shape = CURVES[self.curve]
        if self.oversample == 1:
            return shape(x, self.threshold)
        up = resample_poly(x, self.oversample, 1, window=_RESAMPLE_WINDOW)
        y = shape(up, self.threshold)
        down = resample_poly(y, 1, self.oversample, window=_RESAMPLE_WINDOW)
        return down[: len(x)]

    def reset(self) -> None:
        """No state in the offline prototype."""
