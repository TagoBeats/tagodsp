import numpy as np
import pytest

from tagodsp.analysis.alias_metrics import alias_nmr_db
from tagodsp.analysis.true_peak import true_peak_db
from tagodsp.distortion.clipper import Clipper
from tagodsp.dynamics.true_peak_limiter import TruePeakLimiter

SR = 44100
# The plan allowed 0.1 dB of slack above the ceiling. The limiter does better,
# so the tests demand better: not one measurement may exceed the number the user
# set. Verification runs at 32x, finer than the limiter's own 16x detector, so
# this cannot pass by being measured with the same instrument it uses.
TOLERANCE_DB = 0.0
VERIFY_OVERSAMPLE = 32


def verify_db(x: np.ndarray) -> float:
    return true_peak_db(x, oversample=VERIFY_OVERSAMPLE)


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


def test_signal_below_the_ceiling_passes_through_untouched():
    # A limiter that colours material it is not supposed to touch is useless on
    # a master bus, where it sits permanently.
    x = _sine(1000.0, amp=0.2)
    y = TruePeakLimiter(sr=SR, ceiling_db=-1.0).process(x)
    assert np.allclose(y, x, atol=1e-12)


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
    limiter = TruePeakLimiter(sr=SR, lookahead_ms=1.5)
    assert limiter.latency_samples == int(round(1.5e-3 * SR)) // 2


def test_invalid_args():
    with pytest.raises(ValueError):
        TruePeakLimiter(sr=SR, oversample=3)
    with pytest.raises(ValueError):
        TruePeakLimiter(sr=SR, lookahead_ms=0.0)
    with pytest.raises(ValueError):
        TruePeakLimiter(sr=0)
    with pytest.raises(ValueError):
        TruePeakLimiter(sr=SR).process(np.zeros((2, 2, 16)))
