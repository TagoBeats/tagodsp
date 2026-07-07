"""Biquad filters after the RBJ Audio EQ Cookbook.

Source: Robert Bristow-Johnson, "Cookbook formulae for audio equalizer biquad
filter coefficients" (https://www.w3.org/TR/audio-eq-cookbook/).

Transfer function (normalized by a0):
    H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)

Processing uses transposed direct form II.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BiquadCoeffs:
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float


def _wq(f0: float, sr: float, q: float):
    if not 0 < f0 < sr / 2:
        raise ValueError("f0 must be between 0 and sr/2")
    if q <= 0:
        raise ValueError("q must be > 0")
    w0 = 2.0 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2.0 * q)
    return w0, alpha


def lowpass(f0: float, sr: float, q: float = 0.7071067811865476) -> BiquadCoeffs:
    """RBJ lowpass. Default q = 1/sqrt(2) (Butterworth)."""
    w0, alpha = _wq(f0, sr, q)
    cosw = np.cos(w0)
    a0 = 1.0 + alpha
    return BiquadCoeffs(
        b0=(1.0 - cosw) / 2.0 / a0,
        b1=(1.0 - cosw) / a0,
        b2=(1.0 - cosw) / 2.0 / a0,
        a1=-2.0 * cosw / a0,
        a2=(1.0 - alpha) / a0,
    )


def highpass(f0: float, sr: float, q: float = 0.7071067811865476) -> BiquadCoeffs:
    """RBJ highpass. Default q = 1/sqrt(2) (Butterworth)."""
    w0, alpha = _wq(f0, sr, q)
    cosw = np.cos(w0)
    a0 = 1.0 + alpha
    return BiquadCoeffs(
        b0=(1.0 + cosw) / 2.0 / a0,
        b1=-(1.0 + cosw) / a0,
        b2=(1.0 + cosw) / 2.0 / a0,
        a1=-2.0 * cosw / a0,
        a2=(1.0 - alpha) / a0,
    )


def peaking(f0: float, sr: float, gain_db: float, q: float = 1.0) -> BiquadCoeffs:
    """RBJ peaking EQ."""
    w0, alpha = _wq(f0, sr, q)
    cosw = np.cos(w0)
    A = 10.0 ** (gain_db / 40.0)
    a0 = 1.0 + alpha / A
    return BiquadCoeffs(
        b0=(1.0 + alpha * A) / a0,
        b1=-2.0 * cosw / a0,
        b2=(1.0 - alpha * A) / a0,
        a1=-2.0 * cosw / a0,
        a2=(1.0 - alpha / A) / a0,
    )


class Biquad:
    """Stateful biquad processor, transposed direct form II."""

    def __init__(self, coeffs: BiquadCoeffs):
        self.coeffs = coeffs
        self._z1 = 0.0
        self._z2 = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("expected 1D input")
        c = self.coeffs
        out = np.empty_like(x)
        z1, z2 = self._z1, self._z2
        for i in range(x.size):
            v = x[i]
            y = c.b0 * v + z1
            z1 = c.b1 * v - c.a1 * y + z2
            z2 = c.b2 * v - c.a2 * y
            out[i] = y
        self._z1, self._z2 = z1, z2
        return out

    def reset(self) -> None:
        self._z1 = 0.0
        self._z2 = 0.0


def magnitude_db(coeffs: BiquadCoeffs, freqs: np.ndarray, sr: float) -> np.ndarray:
    """Analytic magnitude response in dB at the given frequencies."""
    w = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64) / sr
    z = np.exp(-1j * w)
    num = coeffs.b0 + coeffs.b1 * z + coeffs.b2 * z * z
    den = 1.0 + coeffs.a1 * z + coeffs.a2 * z * z
    return 20.0 * np.log10(np.abs(num / den))
