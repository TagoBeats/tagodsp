"""The one resampling contract the whole clipping chain shares.

Several modules resample around a nonlinearity or to expose inter-sample
detail, and their results are only comparable if they all interpolate the same
way. That promise used to live in prose, repeated in four docstrings. It lives
here instead.

Kaiser beta 12 gives roughly 115 dB of stopband, which is far below any level
these measurements care about.
"""

from scipy.signal import resample_poly

RESAMPLE_WINDOW = ("kaiser", 12.0)

FACTORS = (1, 2, 4, 8, 16, 32)

# scipy's resample_poly designs its filter as 2 * _FILTER_HALF_LEN * factor + 1
# taps, so its impulse response reaches this many input samples either side.
# Anything that needs to know where the edge transient ends needs this number,
# which is why it is stated once here rather than rediscovered per caller.
FILTER_HALF_LEN = 10


def check_factor(factor: int, allowed=FACTORS) -> int:
    """Validate an oversampling factor against the shared whitelist."""
    if factor not in allowed:
        raise ValueError(f"oversample must be one of {allowed}, got {factor}")
    return factor


def upsample(x, factor: int, axis: int = -1):
    """Raise the sample rate by `factor` using the shared window."""
    if factor == 1:
        return x
    return resample_poly(x, factor, 1, axis=axis, window=RESAMPLE_WINDOW)


def downsample(x, factor: int, axis: int = -1):
    """Lower the sample rate by `factor` using the shared window."""
    if factor == 1:
        return x
    return resample_poly(x, 1, factor, axis=axis, window=RESAMPLE_WINDOW)
