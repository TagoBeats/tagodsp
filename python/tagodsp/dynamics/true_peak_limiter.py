"""Look-ahead limiter that holds a ceiling in dBTP, not just in sample peak.

Chain: a linear-phase lowpass that removes everything above 0.875x Nyquist
(this is part of the ceiling stage, not a separate step someone could switch
off), then oversample the lowpassed signal so inter-sample peaks become
visible, take the gain each oversampled sample needs, run a minimum over the
look-ahead window, then smooth that envelope with a Hann kernel whose support
is at most half the window. The gain is applied to the lowpassed signal, so
what gets detected is exactly what gets output.

The order after the lowpass matters and carries the guarantee. After the
running minimum, g_env[m] <= g_req[n] holds for every m within L/2 of n. The
smoothing kernel only reaches those neighbours and its weights sum to one, so
the smoothed gain cannot exceed g_req[n] either. Smoothing first and taking the
minimum after would lose exactly that property at the transient where it
matters.

Source: standard look-ahead limiter construction; the min-then-smooth ordering
is what makes the bound provable rather than approximate.
Concept note: docs/concepts/true_peak_limiter.md
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import firwin, resample_poly

from tagodsp.utils.gain import db_to_lin
from tagodsp.utils.resampling import check_factor

_VALID_OVERSAMPLE = (2, 4, 8, 16, 32)

# The ceiling-stage lowpass, in front of both detection and the output. Cutoff
# is a fraction of Nyquist, not a fixed Hz value, so these taps are identical
# at every sample rate (0.875 * Nyquist, 255 taps, Kaiser beta 8). Group delay
# is (255 - 1) / 2 = 127 samples, likewise sample-rate independent.
#
# This is why the detector below no longer has to resolve energy right up to
# Nyquist: after this filter there is essentially nothing left there to miss.
# Robin listened at 48k and 44.1k (24.09.2026): not audible in A/B. Keep
# exactly this design, it is validated, not a knob.
LOWPASS_REL_CUTOFF = 0.875
LOWPASS_NUM_TAPS = 255
LOWPASS_BETA = 8.0
LOWPASS_TAPS = firwin(LOWPASS_NUM_TAPS, LOWPASS_REL_CUTOFF, window=("kaiser", LOWPASS_BETA))
LOWPASS_GROUP_DELAY = (LOWPASS_NUM_TAPS - 1) // 2

# Detector: the only configuration below with a measured margin.
#
# What used to be here (a table of one margin per oversample factor, 0.70 dB
# down to 0.02 dB) was calibrated against a 32x verification meter that shared
# the detector's own defect: a finite resampling filter with its cutoff at
# Nyquist under-reads energy just below Nyquist, no matter how far you
# oversample, and lengthening the same filter design converges far too slowly
# to fix it (256 taps/phase still read +0.21 dB high at the worst point). Both
# the old default detector and the old verification meter used that same
# filter family, so the margins agreed with the wrong answer. Measured
# 23.09.2026 with the flawed instrument: -0.98 dBTP for a 16x detector against
# a target of -1.0. Measured 23./24.09.2026 with an ideal-reconstruction meter
# (steep Kaiser 14, 32x, 512 taps/phase, edges excluded) on real okayes stems
# and driven noise: the true worst case at 16x, no lowpass, was +1.04 dB over
# ceiling on stems and +3.4 dB on noise.
#
# The fix is the lowpass above plus this detector: 8x oversample, half length
# 16 (32 taps/phase), Kaiser beta 8, cutoff fixed at 1/oversample (no separate
# cutoff knob, see detector_cutoff below). Measured against the ideal meter,
# raw excess with no margin: real material <= +0.06 dB, noise <= +0.095 dB, at
# 44.1/48/96 kHz and ceilings -0.1/-1.0/-3.0 dBTP. Margin decided by Robin:
# 0.2 dB. This is the only margin this file claims; any other
# oversample/detector_half_len/detector_beta combination has no measurement
# behind it and needs an explicit safety_db, see margin_db below.
DETECTOR_OVERSAMPLE = 8
DETECTOR_HALF_LEN = 16
DETECTOR_BETA = 8.0
DETECTOR_MARGIN_DB = 0.2


def _apply_lowpass(x: np.ndarray) -> np.ndarray:
    """The ceiling-stage lowpass, centred (offline mode="same")."""
    if x.ndim == 1:
        return np.convolve(x, LOWPASS_TAPS, mode="same")
    return np.stack([np.convolve(ch, LOWPASS_TAPS, mode="same") for ch in x])


@dataclass
class TruePeakLimiter:
    """Hold the true peak of a buffer at or below `ceiling_db`.

    Offline: process() expects the whole buffer. Stereo is linked, the gain
    comes from the loudest channel so the image does not move.
    """

    sr: float
    ceiling_db: float = -1.0
    lookahead_ms: float = 1.5
    # Detector oversampling factor. DETECTOR_OVERSAMPLE (8) is the only value
    # with a measured margin, see margin_db. Others remain selectable for
    # experiments but need an explicit safety_db.
    oversample: int = DETECTOR_OVERSAMPLE
    # None follows the measured default, see margin_db and DETECTOR_MARGIN_DB.
    safety_db: float | None = None
    # Detector interpolation filter: half length in base-rate samples, and
    # Kaiser beta. Filter cutoff is always 1/oversample (fixed at the
    # base-rate Nyquist boundary after upsampling); there used to be a
    # separate detector_cutoff knob here, but shifting the cutoff above
    # Nyquist to widen the passband made the true-peak excess worse, not
    # better, so that knob is gone.
    detector_half_len: int = DETECTOR_HALF_LEN
    detector_beta: float = DETECTOR_BETA

    def __post_init__(self) -> None:
        check_factor(self.oversample, _VALID_OVERSAMPLE)
        if self.lookahead_ms <= 0.0:
            raise ValueError(f"lookahead_ms must be positive, got {self.lookahead_ms}")
        if self.sr <= 0.0:
            raise ValueError(f"sr must be positive, got {self.sr}")
        if self.safety_db is not None and self.safety_db < 0.0:
            raise ValueError(f"safety_db must not be negative, got {self.safety_db}")
        if self.detector_half_len <= 0:
            raise ValueError(f"detector_half_len must be positive, got {self.detector_half_len}")

    @property
    def margin_db(self) -> float:
        """The margin actually in use, whether measured or passed in.

        Only the (oversample, detector_half_len, detector_beta) combination
        this file was measured with carries a margin. Anything else raises,
        because a guessed margin is worse than an explicit one: it looks like
        a guarantee and is not.
        """
        if self.safety_db is not None:
            return self.safety_db
        if (self.oversample, self.detector_half_len, self.detector_beta) == (
            DETECTOR_OVERSAMPLE,
            DETECTOR_HALF_LEN,
            DETECTOR_BETA,
        ):
            return DETECTOR_MARGIN_DB
        raise ValueError(
            "no measured margin for oversample="
            f"{self.oversample}, detector_half_len={self.detector_half_len}, "
            f"detector_beta={self.detector_beta}; pass safety_db explicitly"
        )

    @property
    def latency_samples(self) -> int:
        """Total latency the streaming port carries, at the base rate.

        Four pieces, all in front of or inside the look-ahead stage: the
        lowpass group delay (LOWPASS_GROUP_DELAY, 127 samples, fixed by tap
        count and therefore the same at every sample rate), the detector's own
        interpolation delay (detector_half_len samples), half the look-ahead
        window (the running-minimum stage), and half the smoothing kernel. At
        48 kHz with the defaults this is 127 + 16 + 36 + 18 = 197 samples,
        matching the ~200 samples Robin accepted for this fix (23./24.09.2026).
        This method reports what the streaming port will cost; process() below
        stays offline and centred (mode="same"), so it does not carry this
        delay itself.
        """
        window = self._window()
        kernel_len = max(window // 2 | 1, 3)
        return LOWPASS_GROUP_DELAY + self.detector_half_len + window // 2 + kernel_len // 2

    def _window(self) -> int:
        n = int(round(self.lookahead_ms * 1e-3 * self.sr))
        return max(n | 1, 3)

    def _gain_envelope_from_lowpassed(self, x_lp: np.ndarray) -> np.ndarray:
        ceiling = db_to_lin(self.ceiling_db - self.margin_db)

        L = self.oversample
        h = firwin(
            2 * self.detector_half_len * L + 1,
            1.0 / L,
            window=("kaiser", self.detector_beta),
        )
        up = resample_poly(x_lp, L, 1, axis=-1, window=h)
        loudest = np.max(np.abs(up), axis=0) if up.ndim > 1 else np.abs(up)
        required = np.minimum(1.0, ceiling / np.maximum(loudest, 1e-12))

        # Back to the base rate conservatively: the smallest gain any of the
        # oversampled positions asked for wins, so nothing in between is missed.
        n_base = x_lp.shape[-1]
        per_sample = required[: n_base * L].reshape(n_base, L).min(axis=1)

        window = self._window()
        envelope = minimum_filter1d(per_sample, size=window, mode="nearest")

        kernel = np.hanning(max(window // 2 | 1, 3))
        kernel /= kernel.sum()
        pad = len(kernel) // 2
        return np.convolve(np.pad(envelope, pad, mode="edge"), kernel, mode="valid")

    def gain_envelope(self, x: np.ndarray) -> np.ndarray:
        """The gain that process() will apply, at the base rate.

        Computed on the lowpassed signal: detection sees exactly the audio
        process() outputs, not the raw input.
        """
        x = np.asarray(x, dtype=np.float64)
        return self._gain_envelope_from_lowpassed(_apply_lowpass(x))

    def process(self, x: np.ndarray) -> np.ndarray:
        """Apply the limiter. 1-D, or 2-D with time on the last axis."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim not in (1, 2):
            raise ValueError(f"x must be 1-D or 2-D, got {x.ndim}-D")
        if x.size == 0:
            return x.copy()
        x_lp = _apply_lowpass(x)
        return x_lp * self._gain_envelope_from_lowpassed(x_lp)
