"""Mono low end: collapse stereo content below a crossover frequency to mono.

Classic 808/mixbus utility. Split each channel with a Linkwitz-Riley 4th order
crossover (two cascaded Butterworth 2nd order biquads per band, RBJ cookbook),
sum the low band to mid, keep the high band stereo:

    low_i  = LR4_lp(x_i),  high_i = LR4_hp(x_i)
    out_i  = (low_L + low_R) / 2 + high_i

LR4 sums allpass-flat and in phase, so mono material passes with flat magnitude
and only the crossover's allpass phase. Source: S. Linkwitz, "Active crossover
networks", JAES 1976; RBJ Audio EQ Cookbook for the biquad sections.
"""

import numpy as np

from tagodsp.filters.biquad import Biquad, highpass, lowpass


class _LR4Band:
    """One Linkwitz-Riley 4th order band: two cascaded Butterworth biquads."""

    def __init__(self, kind, freq: float, sr: float):
        self._stages = [Biquad(kind(freq, sr)), Biquad(kind(freq, sr))]

    def process(self, x: np.ndarray) -> np.ndarray:
        for stage in self._stages:
            x = stage.process(x)
        return x

    def reset(self) -> None:
        for stage in self._stages:
            stage.reset()


class MonoLow:
    """Stereo-in, stereo-out. Everything below `freq` is summed to mono."""

    def __init__(self, freq: float = 120.0, sr: float = 44100.0):
        self.freq = float(freq)
        self.sr = float(sr)
        self._lp = [_LR4Band(lowpass, freq, sr) for _ in range(2)]
        self._hp = [_LR4Band(highpass, freq, sr) for _ in range(2)]

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"expected stereo input of shape (n, 2), got {x.shape}")
        low = np.column_stack([self._lp[ch].process(x[:, ch]) for ch in range(2)])
        high = np.column_stack([self._hp[ch].process(x[:, ch]) for ch in range(2)])
        mono_low = np.mean(low, axis=1)
        return high + mono_low[:, None]

    def reset(self) -> None:
        for band in (*self._lp, *self._hp):
            band.reset()
