import numpy as np
import pytest
from scipy.ndimage import minimum_filter1d
from scipy.signal import firwin, resample_poly

from tagodsp.analysis.alias_metrics import alias_nmr_db
from tagodsp.distortion.clipper import Clipper
from tagodsp.dynamics.true_peak_limiter import LOWPASS_GROUP_DELAY, TruePeakLimiter

SR = 44100
# The plan allowed 0.1 dB of slack above the ceiling. The limiter does better,
# so the tests demand better: not one measurement may exceed the number the user
# set.
#
# Meter (24.09.2026): steep Kaiser 14, 32x, edges excluded. This is the same
# class of instrument that found the true-peak bug this file regression-tests
# below, deliberately not tagodsp.analysis.true_peak's meter: that one shares
# its resampling filter (half length 10, beta 12) with the old, broken
# detector default, so it shares its blind spot near Nyquist too. It was
# tried here first and produced a false-positive ceiling breach on one of the
# 500 random-signal cases below (verify -0.93 dBTP where the steep meter and
# the design margin both say -1.16 dBTP) purely from its own filter's
# artifacts, not from anything the limiter did wrong. An instrument must not
# have the weakness it is checking for, so this file uses its own.
VERIFY_OVERSAMPLE = 32
_VERIFY_HALF_LEN = 64  # 64 * 32 = 2048 taps/phase; converged, 512 changes nothing past 1e-4 dB
_VERIFY_BETA = 14.0
_VERIFY_TAPS = firwin(
    2 * _VERIFY_HALF_LEN * VERIFY_OVERSAMPLE + 1,
    1.0 / VERIFY_OVERSAMPLE,
    window=("kaiser", _VERIFY_BETA),
)
TOLERANCE_DB = 0.0


def verify_db(x: np.ndarray) -> float:
    """Steep 32x true-peak meter, edges excluded."""
    up = resample_poly(x, VERIFY_OVERSAMPLE, 1, axis=-1, window=_VERIFY_TAPS)
    trim = _VERIFY_HALF_LEN * VERIFY_OVERSAMPLE
    seg = up[..., trim:-trim] if up.shape[-1] > 2 * trim else up
    return float(20 * np.log10(np.max(np.abs(seg)) + 1e-12))


def test_the_verification_meter_is_finer_than_the_detector():
    # Everything below rests on this. If someone raises the limiter's default
    # detector to match, the checks would quietly share its blind spot and pass
    # by agreeing with the instrument under test.
    assert VERIFY_OVERSAMPLE > TruePeakLimiter(sr=SR).oversample


def test_default_detector_holds_the_ceiling():
    # oversample=8 with the default detector_half_len/detector_beta plus the
    # ceiling-stage lowpass is the only configuration with a measured margin,
    # see DETECTOR_MARGIN_DB in true_peak_limiter.py.
    rng = np.random.default_rng(99)
    limiter = TruePeakLimiter(sr=SR, ceiling_db=-1.0)
    worst = -np.inf
    for _ in range(40):
        x = Clipper(curve="hard", threshold=1.0, oversample=1).process(
            rng.standard_normal(4096) * rng.uniform(1.0, 6.0)
        )
        worst = max(worst, verify_db(limiter.process(x)))
    assert worst <= -1.0, f"default detector left {worst:.3f} dBTP"


@pytest.mark.parametrize("oversample", [2, 4, 16])
def test_unmeasured_detector_config_needs_explicit_margin(oversample):
    # Only the default (oversample=8, half_len=16, beta=8) has a margin that
    # was ever checked against the ideal meter. Any other combination used to
    # silently reuse a table of margins that were measured with a meter that
    # shared the detector's own blind spot near Nyquist (see the
    # DETECTOR_MARGIN_DB comment). It must now raise instead of guessing.
    limiter = TruePeakLimiter(sr=SR, ceiling_db=-1.0, oversample=oversample)
    with pytest.raises(ValueError):
        limiter.process(np.ones(64) * 0.9)
    # An explicit safety_db unblocks it; the caller is then the one vouching
    # for the number.
    unblocked = TruePeakLimiter(sr=SR, ceiling_db=-1.0, oversample=oversample, safety_db=1.0)
    unblocked.process(np.ones(64) * 0.9)  # must not raise


def _sine(freq: float, amp: float = 1.0, n: int = 2**14, sr: int = SR, phase: float = 0.0):
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / sr + phase)


def worst_case_suite() -> dict[str, np.ndarray]:
    """Signals built to defeat a true-peak limiter."""
    rng = np.random.default_rng(4)
    clipped = Clipper(curve="hard", threshold=1.0, oversample=1).process(_sine(2000.0, amp=6.0))
    return {
        # Every sample at 0.7071 while the reconstruction reaches 1.0.
        "quarter_nyquist": _sine(SR / 4, n=2**14, phase=np.pi / 4),
        # Near Nyquist, where a coarse meter underestimates most.
        "near_nyquist": _sine(21000.0, amp=0.98),
        "square_ish": clipped,
        "clipped_noise": Clipper(curve="hard", threshold=1.0, oversample=1).process(
            rng.standard_normal(2**14) * 4.0
        ),
        "impulses": np.zeros(2**14),
        "dc_steps": np.repeat(rng.choice([-0.95, 0.95], 64), 256),
    }


def test_ceiling_holds_on_the_worst_case_suite():
    for name, x in worst_case_suite().items():
        if name == "impulses":
            x = x.copy()
            x[::777] = 0.99
        y = TruePeakLimiter(sr=SR, ceiling_db=-1.0).process(x)
        assert verify_db(y) <= -1.0 + TOLERANCE_DB, name


def test_ceiling_holds_over_five_hundred_random_signals():
    rng = np.random.default_rng(2026)
    limiter = TruePeakLimiter(sr=SR, ceiling_db=-1.0)
    worst = -np.inf
    for _ in range(500):
        kind = rng.integers(0, 3)
        n = 4096
        if kind == 0:
            x = rng.standard_normal(n) * rng.uniform(0.2, 4.0)
        elif kind == 1:
            x = _sine(float(rng.uniform(100, 21000)), amp=float(rng.uniform(0.5, 4.0)), n=n)
        else:
            x = Clipper(curve="fl", threshold=0.5, oversample=1).process(
                _sine(float(rng.uniform(100, 15000)), amp=float(rng.uniform(1.0, 8.0)), n=n)
            )
        worst = max(worst, verify_db(limiter.process(x)))
    assert worst <= -1.0 + TOLERANCE_DB, f"worst measured {worst:.3f} dBTP"


@pytest.mark.parametrize("ceiling_db", [-0.1, -1.0, -3.0, -6.0])
def test_every_ceiling_is_respected(ceiling_db):
    x = Clipper(curve="hard", threshold=1.0, oversample=1).process(_sine(3000.0, amp=5.0))
    y = TruePeakLimiter(sr=SR, ceiling_db=ceiling_db).process(x)
    assert verify_db(y) <= ceiling_db + TOLERANCE_DB


def test_signal_below_the_ceiling_is_not_gain_reduced():
    # A limiter that gain-reduces material it is not supposed to touch is
    # useless on a master bus, where it sits permanently. The ceiling-stage
    # lowpass (24.09.2026 fix) is now always applied, in front of detection
    # and at the output, so the output is no longer bit-identical to the
    # input even when nothing needs limiting; the invariant that still has to
    # hold is that the gain itself stays at unity.
    x = _sine(1000.0, amp=0.2)
    limiter = TruePeakLimiter(sr=SR, ceiling_db=-1.0)
    gain = limiter.gain_envelope(x)
    assert np.allclose(gain, 1.0, atol=1e-9)


def test_it_does_not_give_away_more_than_it_has_to():
    # The ceiling must be held, but the result should still land close to it
    # rather than ducking far below and throwing away loudness.
    x = Clipper(curve="hard", threshold=1.0, oversample=1).process(_sine(2000.0, amp=4.0))
    y = TruePeakLimiter(sr=SR, ceiling_db=-1.0).process(x)
    assert verify_db(y) > -1.0 - 1.5


def test_gain_modulation_stays_clean():
    # Limiting is a time varying gain, which sprays sidebands. Check they stay
    # far below the fundamental: a tone limited by a few dB must not turn into
    # an intermodulation mess.
    x = _sine(1000.0, amp=1.6, n=2**15)
    y = TruePeakLimiter(sr=SR, ceiling_db=-1.0).process(x)
    assert alias_nmr_db(y, SR, 1000.0) < -60.0


def test_stereo_gain_is_linked():
    # One loud channel must pull the other down by the same amount, otherwise
    # the stereo image shifts whenever the limiter engages.
    loud = Clipper(curve="hard", threshold=1.0, oversample=1).process(_sine(3000.0, amp=5.0))
    quiet = _sine(500.0, amp=0.1, n=len(loud))
    stereo = np.stack([loud, quiet])
    out = TruePeakLimiter(sr=SR, ceiling_db=-1.0).process(stereo)
    ratio = out[1] / np.where(np.abs(quiet) > 1e-9, quiet, np.nan)
    applied = ratio[np.isfinite(ratio)]
    assert applied.min() < 1.0 - 1e-6  # the quiet channel really was pulled down
    assert verify_db(out) <= -1.0 + TOLERANCE_DB


def test_latency_is_reported():
    # Total streaming latency, not just the look-ahead half-window: lowpass
    # group delay + detector interpolation delay + half window + half kernel.
    # See latency_samples' docstring for the breakdown and the 48 kHz numbers.
    limiter = TruePeakLimiter(sr=SR, lookahead_ms=1.5)
    window = max(int(round(1.5e-3 * SR)) | 1, 3)
    kernel_len = max(window // 2 | 1, 3)
    expected = LOWPASS_GROUP_DELAY + limiter.detector_half_len + window // 2 + kernel_len // 2
    assert limiter.latency_samples == expected


def test_latency_is_independent_of_sample_rate_for_the_lowpass_term():
    # The lowpass taps are fixed by count, not by cutoff in Hz, so their group
    # delay must be the same 127 samples at every sample rate. Rule 7 in the
    # TagoClipPro CLAUDE.md depends on the latency being constant per sample
    # rate; this is the piece of that guarantee that lives in this file.
    assert LOWPASS_GROUP_DELAY == 127


def test_invalid_args():
    with pytest.raises(ValueError):
        TruePeakLimiter(sr=SR, oversample=3)
    with pytest.raises(ValueError):
        TruePeakLimiter(sr=SR, lookahead_ms=0.0)
    with pytest.raises(ValueError):
        TruePeakLimiter(sr=0)
    with pytest.raises(ValueError):
        TruePeakLimiter(sr=SR).process(np.zeros((2, 2, 16)))


# --- Regression: the 23./24.09.2026 true-peak hole ------------------------
#
# Root cause: any finite detector filter with its cutoff at Nyquist under-reads
# energy just below Nyquist, and a longer filter of the same design converges
# far too slowly to fix it (256 taps/phase still read +0.21 dB high). Measured
# worst case with the old 16x default, no lowpass, against an ideal meter:
# +1.04 dB over ceiling on real okayes stems, +3.4 dB on synthetic noise.
#
# Verdict meter: verify_db, defined above, for the same reason given there.


def _old_process(x: np.ndarray, sr: float, ceiling_db: float = -1.0, lookahead_ms: float = 1.5,
                  oversample: int = 16) -> np.ndarray:
    """The pre-fix algorithm (no ceiling-stage lowpass, shared resampling
    filter for detection: half length 10, beta 12, the old per-factor
    _SAFETY_DB table). Kept only so this test can prove it is broken; the file
    under test no longer contains this code, this is the copy the task asked
    for so red-then-green does not depend on git surgery in the worktree."""
    from tagodsp.utils.resampling import upsample

    old_safety_db = {2: 0.70, 4: 0.32, 8: 0.12, 16: 0.05, 32: 0.02}
    x = np.asarray(x, dtype=np.float64)
    ceiling = 10 ** ((ceiling_db - old_safety_db[oversample]) / 20.0)
    up = upsample(x, oversample)
    required = np.minimum(1.0, ceiling / np.maximum(np.abs(up), 1e-12))
    n_base = x.shape[-1]
    per_sample = required[: n_base * oversample].reshape(n_base, oversample).min(axis=1)
    n = int(round(lookahead_ms * 1e-3 * sr))
    window = max(n | 1, 3)
    envelope = minimum_filter1d(per_sample, size=window, mode="nearest")
    kernel = np.hanning(max(window // 2 | 1, 3))
    kernel /= kernel.sum()
    pad = len(kernel) // 2
    gain = np.convolve(np.pad(envelope, pad, mode="edge"), kernel, mode="valid")
    return x * gain


def _driven_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    """Scratchpad noise recipe: seeded standard_normal * 0.6, then hard-clipped
    without anti-aliasing oversampling, so it carries broadband harmonics all
    the way to Nyquist, exactly like the render engine's drive stage does."""
    x = rng.standard_normal(n) * 0.6
    return Clipper(curve="hard", threshold=0.5, oversample=1).process(x)


def test_regression_true_peak_hole_is_closed():
    rng = np.random.default_rng(2024)
    x = _driven_noise(rng, 2**16)
    ceiling_db = -1.0

    # Red: the pre-fix algorithm really does miss the ceiling on this fixture.
    old_excess = verify_db(_old_process(x, SR, ceiling_db=ceiling_db)) - ceiling_db
    assert old_excess > 0.2, (
        "fixture stopped reproducing the bug, old detector only read "
        f"{old_excess:+.3f} dB over ceiling; strengthen the noise case"
    )

    # Green: the fixed limiter holds it, no margin left to give (TOLERANCE_DB).
    new_out = TruePeakLimiter(sr=SR, ceiling_db=ceiling_db).process(x)
    new_excess = verify_db(new_out) - ceiling_db
    assert new_excess <= TOLERANCE_DB, f"new detector left {new_excess:+.3f} dB over ceiling"
