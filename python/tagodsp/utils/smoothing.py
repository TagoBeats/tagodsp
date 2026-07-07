"""Parameter smoothing.

One-pole exponential smoother, the standard tool against parameter zipper noise.
Coefficient from time constant: a = exp(-1 / (tau_s * sr)), where the smoother
reaches ~63.2% of a step after tau_s seconds (RC lowpass analogy, see JOS,
"Introduction to Digital Filters").
"""

import numpy as np


class OnePoleSmoother:
    """Exponential one-pole smoother for 1D control signals.

    y[n] = a * y[n-1] + (1 - a) * x[n]
    """

    def __init__(self, tau_s: float, sr: float, initial: float = 0.0):
        if tau_s < 0:
            raise ValueError("tau_s must be >= 0")
        if sr <= 0:
            raise ValueError("sr must be > 0")
        self._a = float(np.exp(-1.0 / (tau_s * sr))) if tau_s > 0 else 0.0
        self._initial = float(initial)
        self._y = float(initial)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Smooth a 1D block of control values."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("expected 1D input")
        out = np.empty_like(x)
        y = self._y
        a = self._a
        b = 1.0 - a
        for i in range(x.size):
            y = a * y + b * x[i]
            out[i] = y
        self._y = y
        return out

    def reset(self, value: float | None = None) -> None:
        self._y = self._initial if value is None else float(value)
