"""Look-ahead limiter that holds a ceiling in dBTP, not just in sample peak.

Chain: oversample so inter-sample peaks become visible, take the gain each
oversampled sample needs, run a minimum over the look-ahead window, then smooth
that envelope with a Hann kernel whose support is at most half the window.

The order matters and carries the guarantee. After the running minimum,
g_env[m] <= g_req[n] holds for every m within L/2 of n. The smoothing kernel
only reaches those neighbours and its weights sum to one, so the smoothed gain
cannot exceed g_req[n] either. Smoothing first and taking the minimum after
would lose exactly that property at the transient where it matters.

Source: standard look-ahead limiter construction; the min-then-smooth ordering
is what makes the bound provable rather than approximate.
Concept note: docs/concepts/true_peak_limiter.md
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d

from tagodsp.utils.gain import db_to_lin
from tagodsp.utils.resampling import check_factor, upsample

_VALID_OVERSAMPLE = (2, 4, 8, 16, 32)

# How far under the ceiling to aim, per detector oversampling factor.
#
# The guarantee above holds for what the detector sees, and a detector with
# finite oversampling does not see the maximum that falls between its own
# support points. That blind spot is what the margin buys back, so it belongs
# to the factor rather than being one constant for all of them. Measured on
# clipped noise against a 32x meter, ceiling -1 dBTP, the excess left by a bare
# detector was 0.52 dB at 2x, 0.23 at 4x, 0.08 at 8x and 0.02 at 16x, roughly a
# quarter per doubling. These values carry about a factor of two of headroom on
# that, which costs loudness nobody can hear.
_SAFETY_DB = {2: 0.70, 4: 0.32, 8: 0.12, 16: 0.05, 32: 0.02}


@dataclass
class TruePeakLimiter:
    """Hold the true peak of a buffer at or below `ceiling_db`.

    Offline: process() expects the whole buffer. Stereo is linked, the gain
    comes from the loudest channel so the image does not move.
    """

    sr: float
    ceiling_db: float = -1.0
    lookahead_ms: float = 1.5
    oversample: int = 16
    # None follows the detector, see _SAFETY_DB. Override only with a measurement.
    safety_db: float | None = None

    def __post_init__(self) -> None:
        check_factor(self.oversample, _VALID_OVERSAMPLE)
        if self.lookahead_ms <= 0.0:
            raise ValueError(f"lookahead_ms must be positive, got {self.lookahead_ms}")
        if self.sr <= 0.0:
            raise ValueError(f"sr must be positive, got {self.sr}")
        if self.safety_db is not None and self.safety_db < 0.0:
            raise ValueError(f"safety_db must not be negative, got {self.safety_db}")

    @property
    def margin_db(self) -> float:
        """The margin actually in use, whether chosen here or passed in."""
        return _SAFETY_DB[self.oversample] if self.safety_db is None else self.safety_db

    @property
    def latency_samples(self) -> int:
        """Half the look-ahead window, in samples at the base rate."""
        return self._window() // 2

    def _window(self) -> int:
        n = int(round(self.lookahead_ms * 1e-3 * self.sr))
        return max(n | 1, 3)

    def gain_envelope(self, x: np.ndarray) -> np.ndarray:
        """The gain that process() will apply, at the base rate."""
        x = np.asarray(x, dtype=np.float64)
        ceiling = db_to_lin(self.ceiling_db - self.margin_db)

        up = upsample(x, self.oversample)
        loudest = np.max(np.abs(up), axis=0) if up.ndim > 1 else np.abs(up)
        required = np.minimum(1.0, ceiling / np.maximum(loudest, 1e-12))

        # Back to the base rate conservatively: the smallest gain any of the
        # oversampled positions asked for wins, so nothing in between is missed.
        n_base = x.shape[-1]
        per_sample = required[: n_base * self.oversample].reshape(n_base, self.oversample).min(
            axis=1
        )

        window = self._window()
        envelope = minimum_filter1d(per_sample, size=window, mode="nearest")

        kernel = np.hanning(max(window // 2 | 1, 3))
        kernel /= kernel.sum()
        pad = len(kernel) // 2
        return np.convolve(np.pad(envelope, pad, mode="edge"), kernel, mode="valid")

    def process(self, x: np.ndarray) -> np.ndarray:
        """Apply the limiter. 1-D, or 2-D with time on the last axis."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim not in (1, 2):
            raise ValueError(f"x must be 1-D or 2-D, got {x.ndim}-D")
        if x.size == 0:
            return x.copy()
        return x * self.gain_envelope(x)
