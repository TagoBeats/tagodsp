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

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import resample_poly

from tagodsp.utils.gain import db_to_lin

_RESAMPLE_WINDOW = ("kaiser", 12.0)


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
    safety_db: float = 0.05
    _last_gain: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.oversample not in (2, 4, 8, 16, 32):
            raise ValueError(f"oversample must be 2, 4, 8, 16 or 32, got {self.oversample}")
        if self.lookahead_ms <= 0.0:
            raise ValueError(f"lookahead_ms must be positive, got {self.lookahead_ms}")
        if self.sr <= 0.0:
            raise ValueError(f"sr must be positive, got {self.sr}")
        if self.safety_db < 0.0:
            raise ValueError(f"safety_db must not be negative, got {self.safety_db}")

    @property
    def latency_samples(self) -> int:
        """Half the look-ahead window, in samples at the base rate."""
        return self._window() // 2

    def _window(self) -> int:
        n = int(round(self.lookahead_ms * 1e-3 * self.sr))
        return max(n | 1, 3)

    def gain_envelope(self, x: np.ndarray) -> np.ndarray:
        """The gain that will be applied, at the base rate. Exposed for plots."""
        x = np.asarray(x, dtype=np.float64)
        # Aim slightly under. A detector that oversamples by a finite factor
        # always underestimates the true maximum, and a finer meter finds the
        # rest: measured against a 32x meter, an 8x detector leaves 0.09 dB and
        # a 16x detector 0.02 dB above the target. Matching the detector to our
        # own meter would only be tuning to that one instrument, so the margin
        # buys the promise instead, at a cost in loudness nobody can hear.
        ceiling = db_to_lin(self.ceiling_db - self.safety_db)

        up = resample_poly(x, self.oversample, 1, axis=-1, window=_RESAMPLE_WINDOW)
        loudest = np.max(np.abs(up), axis=0) if up.ndim > 1 else np.abs(up)
        required = np.minimum(1.0, ceiling / np.maximum(loudest, 1e-12))

        # Back to the base rate conservatively: the smallest gain any of the
        # oversampled positions asked for wins, so nothing in between is missed.
        n_base = x.shape[-1]
        padded = np.pad(
            required, (0, max(0, n_base * self.oversample - len(required))), mode="edge"
        )[: n_base * self.oversample]
        per_sample = padded.reshape(n_base, self.oversample).min(axis=1)

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
        gain = self.gain_envelope(x)
        self._last_gain = gain
        return x * gain

    def reset(self) -> None:
        self._last_gain = None
