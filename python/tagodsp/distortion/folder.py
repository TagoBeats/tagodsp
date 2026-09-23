"""Wavefolder as a fourth curve for the clipping family.

A clipper flattens the signal at the threshold, a folder reflects it: whatever
runs past the ceiling comes back down mirrored, and at high drive it does so
several times, eventually crossing zero and coming out with the opposite sign.
The harmonics therefore do not grow monotonically with level, they come and go
with the number of folds the momentary sample goes through, which is the
metallic, vocal-ish West Coast sound (Buchla, Serge).

Parametrised so it can sit in the same signal path as the clipping curves:
identity below the threshold, odd symmetric at skew 0, threshold moves the
knee. The ceiling equals the threshold, as it already does for hardclip in
this family, and unlike fl and tanh where it is fixed at 1.0:

    y = sign(x) * t * T(|x| / t)

with T the unit triangle wave of period 4, T(0) = 0, T(1) = 1, T(2) = 0,
T(3) = -1. Below the knee T is the identity, so the curve is, and the slope
stays 1 through the knee: a folder has no knee to round, only fold corners.

The alternative parametrisation, folding only inside the headroom above t and
keeping the ceiling at 1.0, was tried first and discarded: at the family's
default threshold of 100/128 it leaves the output rippling between 0.56 and
1.0, which is a texture on top of a clipper and not a folder. Fold depth has
to be the full range or the character the curve exists for is not there.

Skew means folding the two halves differently, which is what adds even
harmonics; a symmetric folder is an odd function and produces odd harmonics
only, and those pile up into mush at high drive. The asymmetry sits in the
threshold:

    t_pos = t,   t_neg = t * (1 - skew),   skew in [0, 1)

so the negative side folds earlier and more often, at a lower peak. Negative
skew is left out on purpose: it is the same curve with the polarity flipped
and would push the negative peak above the threshold, past the one level the
rest of the chain is allowed to assume.

The curve is piecewise linear, so F1 is piecewise quadratic and closed form.
With G the integral of T from 0 (period 4, zero mean, hence periodic too):

    F1(x) = t_side^2 * G(|x| / t_side)

F1 is even only when skew is 0, so unlike the three clipping curves it is
evaluated on signed x and picks its side itself. No transcendentals and no
division by a signal-dependent quantity anywhere, which matters for the gate:
the numerically delicate part of ADAA is then the difference quotient alone,
not the curve feeding it.

Concept note and gate: docs/concepts/wavefold.md
"""

import numpy as np

# Same default knee as the clipping family (FL's knob default, 100/128).
# Spelled out rather than imported to keep this module free of a dependency on
# clipper.py, which imports back from here to register the curve.
FOLD_THRESHOLD_DEFAULT = 100.0 / 128.0

# Starting point, not a tuned value: the negative side folds at 0.7 * t, so its
# fold points never coincide with the positive ones and the even harmonics stay
# present at every drive setting rather than only at some. The gate measures
# this value, it does not assume it.
SKEW_DEFAULT = 0.3


def _triangle(u: np.ndarray) -> np.ndarray:
    """Unit triangle wave, period 4: T(0)=0, T(1)=1, T(2)=0, T(3)=-1."""
    return 1.0 - np.abs(np.mod(u + 1.0, 4.0) - 2.0)


def _triangle_integral(u: np.ndarray) -> np.ndarray:
    """G(u) = integral of T from 0 to u.

    T has zero mean over its period, so G is periodic with period 4 as well and
    the reduction below is exact rather than an approximation. Piecewise:
    r^2/2 on [0,1], 2r - r^2/2 - 1 on [1,3], r^2/2 - 4r + 8 on [3,4].
    """
    r = np.mod(u, 4.0)
    return np.where(
        r <= 1.0,
        0.5 * r * r,
        np.where(r <= 3.0, 2.0 * r - 0.5 * r * r - 1.0, 0.5 * r * r - 4.0 * r + 8.0),
    )


def _side_threshold(x: np.ndarray, threshold: float, skew: float) -> np.ndarray:
    """Per-sample threshold: t above zero, t * (1 - skew) below."""
    t = float(threshold)
    if not 0.0 < t <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {t}")
    s = float(skew)
    if not 0.0 <= s < 1.0:
        raise ValueError(f"skew must be in [0, 1), got {s}")
    return np.where(x >= 0.0, t, t * (1.0 - s))


def wavefold(
    x: np.ndarray,
    threshold: float = FOLD_THRESHOLD_DEFAULT,
    skew: float = 0.0,
) -> np.ndarray:
    """Triangle wavefolder. Identity below the threshold, peaks at the threshold."""
    x = np.asarray(x, dtype=np.float64)
    t = _side_threshold(x, threshold, skew)
    return np.sign(x) * t * _triangle(np.abs(x) / t)


def antiderivative_fold(
    x: np.ndarray,
    threshold: float = FOLD_THRESHOLD_DEFAULT,
    skew: float = 0.0,
) -> np.ndarray:
    """F1 of wavefold. Even only for skew = 0, so it works on signed x."""
    x = np.asarray(x, dtype=np.float64)
    t = _side_threshold(x, threshold, skew)
    return t * t * _triangle_integral(np.abs(x) / t)


def sinefold(
    x: np.ndarray,
    threshold: float = FOLD_THRESHOLD_DEFAULT,
    skew: float = 0.0,
) -> np.ndarray:
    """Smooth wavefolder, y = t * sin(x / t). Same idea without the corners.

    The triangle folder above is piecewise linear, so every fold point is a
    corner, and a corner is a discontinuity in the first derivative that throws
    harmonics of every order at the band limit. This one is C-infinity: slope 1
    at the origin, peak exactly t at x = pi*t/2, folds with the same period in
    input level. Below the threshold it is not the identity but a soft knee,
    sin(1) = 0.841, so 1.5 dB of compression right before the first fold.

    It exists to separate two questions the gate would otherwise answer at
    once: whether wavefolding aliases, and whether corners alias.
    """
    x = np.asarray(x, dtype=np.float64)
    t = _side_threshold(x, threshold, skew)
    return t * np.sin(x / t)


def antiderivative_sinefold(
    x: np.ndarray,
    threshold: float = FOLD_THRESHOLD_DEFAULT,
    skew: float = 0.0,
) -> np.ndarray:
    """F1 of sinefold: t^2 * (1 - cos(x / t)), zero at the origin."""
    x = np.asarray(x, dtype=np.float64)
    t = _side_threshold(x, threshold, skew)
    return t * t * (1.0 - np.cos(x / t))


def fold(x: np.ndarray, threshold: float = FOLD_THRESHOLD_DEFAULT) -> np.ndarray:
    """Symmetric fold, odd harmonics only. Curve entry "fold"."""
    return wavefold(x, threshold, skew=0.0)


def skew_fold(x: np.ndarray, threshold: float = FOLD_THRESHOLD_DEFAULT) -> np.ndarray:
    """Asymmetric fold at SKEW_DEFAULT, adds even harmonics. Entry "skewfold"."""
    return wavefold(x, threshold, skew=SKEW_DEFAULT)


def antiderivative_symmetric(
    x: np.ndarray, threshold: float = FOLD_THRESHOLD_DEFAULT
) -> np.ndarray:
    """F1 of fold."""
    return antiderivative_fold(x, threshold, skew=0.0)


def antiderivative_skewed(
    x: np.ndarray, threshold: float = FOLD_THRESHOLD_DEFAULT
) -> np.ndarray:
    """F1 of skew_fold."""
    return antiderivative_fold(x, threshold, skew=SKEW_DEFAULT)


def sine_fold(x: np.ndarray, threshold: float = FOLD_THRESHOLD_DEFAULT) -> np.ndarray:
    """Symmetric smooth fold. Curve entry "sinefold"."""
    return sinefold(x, threshold, skew=0.0)


def skew_sine_fold(x: np.ndarray, threshold: float = FOLD_THRESHOLD_DEFAULT) -> np.ndarray:
    """Asymmetric smooth fold at SKEW_DEFAULT. Entry "skewsinefold"."""
    return sinefold(x, threshold, skew=SKEW_DEFAULT)


def antiderivative_sine_symmetric(
    x: np.ndarray, threshold: float = FOLD_THRESHOLD_DEFAULT
) -> np.ndarray:
    """F1 of sine_fold."""
    return antiderivative_sinefold(x, threshold, skew=0.0)


def antiderivative_sine_skewed(
    x: np.ndarray, threshold: float = FOLD_THRESHOLD_DEFAULT
) -> np.ndarray:
    """F1 of skew_sine_fold."""
    return antiderivative_sinefold(x, threshold, skew=SKEW_DEFAULT)


FOLD_CURVES = {
    "fold": fold,
    "skewfold": skew_fold,
    "sinefold": sine_fold,
    "skewsinefold": skew_sine_fold,
}

FOLD_ANTIDERIVATIVES = {
    "fold": antiderivative_symmetric,
    "skewfold": antiderivative_skewed,
    "sinefold": antiderivative_sine_symmetric,
    "skewsinefold": antiderivative_sine_skewed,
}
