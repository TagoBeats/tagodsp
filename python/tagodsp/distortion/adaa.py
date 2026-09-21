"""First-order antiderivative antialiasing for the clipping curves.

Instead of oversampling around the nonlinearity, ADAA averages the transfer
curve across the interval between two consecutive samples by evaluating its
antiderivative F1:

    y[n] = (F1(x[n]) - F1(x[n-1])) / (x[n] - x[n-1])

which is what a piecewise-linear reconstruction of the input, pushed through
the curve and band-limited again, would produce. It costs half a sample of
delay. Where the difference approaches zero the quotient loses its leading
digits to cancellation, so below `eps` the curve is evaluated directly at the
midpoint of the interval, which is the limit of the quotient.

All three curves are odd, so F1 is even and is evaluated on |x|. With
a = 1 - t and u = (|x| - t) / a, and F1 = x^2/2 below the knee:

    hard: t*|x| - t^2/2
    fl:   t^2/2 + (|x| - t) - a^2 * (1 - exp(-u))
    tanh: t^2/2 + t*(|x| - t) + a^2 * ln cosh(u)

Source: Parker, Zavalishin, Le Bivic, "Reducing the Aliasing of Nonlinear
Waveshaping Using Continuous-Time Convolution", DAFx-16.
Concept note: docs/concepts/adaa_clipper.md
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import resample_poly

from tagodsp.distortion.clipper import CURVES, FL_THRESHOLD_DEFAULT
from tagodsp.utils.gain import db_to_lin

# Same resampling window as distortion/clipper.py, so the two are comparable.
_RESAMPLE_WINDOW = ("kaiser", 12.0)

_LN2 = float(np.log(2.0))


def _ln_cosh(u: np.ndarray) -> np.ndarray:
    """Numerically stable ln(cosh(u)) for u >= 0.

    cosh grows exponentially, so the direct form overflows at high drive.
    ln cosh(u) = u + log1p(exp(-2u)) - ln 2.
    """
    return u + np.log1p(np.exp(-2.0 * u)) - _LN2


def antiderivative_hard(x: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """F1 of hardclip. Ceiling equals the threshold in this curve family."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    t = float(threshold)
    return np.where(ax <= t, 0.5 * ax * ax, t * ax - 0.5 * t * t)


def antiderivative_fl(x: np.ndarray, threshold: float = FL_THRESHOLD_DEFAULT) -> np.ndarray:
    """F1 of the Fruity Soft Clipper curve."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    t = float(threshold)
    a = 1.0 - t
    u = np.maximum(ax - t, 0.0) / a
    above = 0.5 * t * t + (ax - t) - a * a * (1.0 - np.exp(-u))
    return np.where(ax <= t, 0.5 * ax * ax, above)


def antiderivative_tanh(x: np.ndarray, threshold: float = FL_THRESHOLD_DEFAULT) -> np.ndarray:
    """F1 of the tanh knee curve."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    t = float(threshold)
    a = 1.0 - t
    u = np.maximum(ax - t, 0.0) / a
    above = 0.5 * t * t + t * (ax - t) + a * a * _ln_cosh(u)
    return np.where(ax <= t, 0.5 * ax * ax, above)


ANTIDERIVATIVES = {
    "fl": antiderivative_fl,
    "hard": antiderivative_hard,
    "tanh": antiderivative_tanh,
}


@dataclass
class ADAAClipper:
    """Clipping waveshaper with first-order ADAA, optionally oversampled.

    Stateful across calls: the previous driven input sample carries over, so
    process() can be called block by block. That only holds at oversample=1;
    with oversampling the resampler has state of its own that this offline
    prototype does not model, exactly as in Clipper.
    """

    curve: str = "fl"
    threshold: float = FL_THRESHOLD_DEFAULT
    oversample: int = 1
    drive_db: float = 0.0
    # Where the two branches cross over in accuracy, measured on the fl curve at
    # x = 1.2: the quotient loses to cancellation as the difference shrinks
    # (1.9e-13 error at 1e-3, 8.2e-8 at 1e-9) while the midpoint gains as its
    # O(d^2) truncation shrinks (2.1e-8 at 1e-3, exact by 1e-9). They meet near
    # 1e-4. Anywhere in 1e-3 to 1e-9 the error stays below float32 resolution,
    # so this is a cheap choice rather than a critical one.
    eps: float = 1e-4
    _x_prev: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.curve not in ANTIDERIVATIVES:
            raise ValueError(
                f"curve must be one of {sorted(ANTIDERIVATIVES)}, got {self.curve!r}"
            )
        if self.oversample not in (1, 2, 4, 8, 16):
            raise ValueError(f"oversample must be 1, 2, 4, 8 or 16, got {self.oversample}")
        if self.curve != "hard" and not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {self.threshold}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}")

    def _shape(self, x: np.ndarray, x_prev: float) -> tuple[np.ndarray, float]:
        """Apply first-order ADAA to one contiguous run of samples."""
        f = CURVES[self.curve]
        f1 = ANTIDERIVATIVES[self.curve]
        prev = np.concatenate(([x_prev], x[:-1]))
        delta = x - prev

        # Guard the division itself, then discard those entries. Dividing by the
        # raw delta would raise a warning and produce inf before the where runs.
        safe = np.where(np.abs(delta) < self.eps, 1.0, delta)
        quotient = (f1(x, self.threshold) - f1(prev, self.threshold)) / safe
        midpoint = f(0.5 * (x + prev), self.threshold)

        y = np.where(np.abs(delta) < self.eps, midpoint, quotient)
        return y, float(x[-1])

    def process(self, x: np.ndarray) -> np.ndarray:
        """Shape a 1-D buffer. Drive is applied before the curve."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"x must be 1-D, got {x.ndim}-D")
        if x.size == 0:
            return x.copy()
        driven = x * db_to_lin(self.drive_db)

        if self.oversample == 1:
            y, self._x_prev = self._shape(driven, self._x_prev)
            return y

        up = resample_poly(driven, self.oversample, 1, window=_RESAMPLE_WINDOW)
        shaped, self._x_prev = self._shape(up, self._x_prev)
        down = resample_poly(shaped, 1, self.oversample, window=_RESAMPLE_WINDOW)
        return down[: len(x)]

    def reset(self) -> None:
        """Forget the previous sample, as if starting from silence."""
        self._x_prev = 0.0

    def fallback_rate(self, x: np.ndarray) -> float:
        """Fraction of samples that would take the midpoint branch.

        Diagnostic for choosing eps: near-silent or low-frequency material sits
        almost entirely in the fallback, which means ADAA does nothing there.
        """
        x = np.asarray(x, dtype=np.float64) * db_to_lin(self.drive_db)
        if x.size == 0:
            return 0.0
        prev = np.concatenate(([self._x_prev], x[:-1]))
        return float(np.mean(np.abs(x - prev) < self.eps))
