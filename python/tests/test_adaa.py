import numpy as np
import pytest

from tagodsp.analysis.alias_metrics import alias_nmr_db, harmonic_level_db
from tagodsp.distortion.adaa import ANTIDERIVATIVES, ADAAClipper
from tagodsp.distortion.clipper import CURVES, Clipper
from tagodsp.utils.gain import rms_db

SR = 44100
CURVE_NAMES = ("fl", "hard", "tanh")


def _sine(freq: float, amp: float = 1.0, n: int = 2**15, sr: int = SR) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / sr)


@pytest.mark.parametrize("name", CURVE_NAMES)
def test_antiderivative_differentiates_back_to_the_curve(name):
    # The one property that makes ADAA correct at all: dF1/dx must be f.
    # The knee is skipped: hardclip is only C0 there, its slope jumps from 1 to
    # 0, and a central difference straddling that kink averages both sides. The
    # error is 1e-5 exactly at +-t and 9e-12 everywhere else.
    f, f1 = CURVES[name], ANTIDERIVATIVES[name]
    t = 0.6
    x = np.linspace(-4.0, 4.0, 200001)
    h = x[1] - x[0]
    numeric = (f1(x[2:], t) - f1(x[:-2], t)) / (2 * h)
    inner = x[1:-1]
    away_from_knee = np.abs(np.abs(inner) - t) > 2 * h
    error = np.abs(numeric - f(inner, t))[away_from_knee]
    assert np.max(error) < 1e-9


@pytest.mark.parametrize("name", CURVE_NAMES)
def test_antiderivative_is_even_and_continuous_at_the_knee(name):
    f1 = ANTIDERIVATIVES[name]
    t = 0.6
    x = np.linspace(0.0, 5.0, 5001)
    assert np.allclose(f1(x, t), f1(-x, t), atol=1e-15)
    # Both branches meet at t^2/2.
    assert f1(np.array([t]), t)[0] == pytest.approx(0.5 * t * t, abs=1e-12)
    assert f1(np.array([t + 1e-9]), t)[0] == pytest.approx(0.5 * t * t, abs=1e-8)


@pytest.mark.parametrize("name", CURVE_NAMES)
def test_below_the_knee_adaa_is_exactly_a_two_point_average(name):
    # Below the threshold the curve is the identity, so F1 = x^2/2 and the
    # quotient collapses to (x[n] + x[n-1]) / 2 analytically. That is also
    # where the half sample of delay comes from.
    x = _sine(300.0, amp=0.2, n=4096)
    y = ADAAClipper(curve=name, threshold=0.9, eps=1e-12).process(x)
    expected = 0.5 * (x + np.concatenate(([0.0], x[:-1])))
    assert np.max(np.abs(y - expected)) < 1e-12


@pytest.mark.parametrize("name", CURVE_NAMES)
def test_adaa_alone_clears_the_kill_bar(name):
    # Minimum gate from the plan, fixed before the measurement: below 6 dB of
    # alias reduction the idea is dead, 10 dB or more would have been a clean
    # pass. Measured on 2026-09-21: 6.6 dB (fl), 7.5 dB (hard), 6.5 dB (tanh).
    # So ADAA on its own is alive but not sufficient, which is why the product
    # configuration pairs it with 2x oversampling, see the test below.
    x = _sine(5000.0)
    naive = Clipper(curve=name, threshold=67 / 128, oversample=1, drive_db=11.0).process(x)
    adaa = ADAAClipper(curve=name, threshold=67 / 128, oversample=1, drive_db=11.0).process(x)
    gain_db = alias_nmr_db(naive, SR, 5000.0) - alias_nmr_db(adaa, SR, 5000.0)
    assert gain_db > 6.0


@pytest.mark.parametrize("name", CURVE_NAMES)
@pytest.mark.parametrize("freq", [1000.0, 5000.0, 11000.0])
def test_adaa_at_4x_beats_plain_oversampling_at_8x(name, freq):
    # The target gate and the actual product claim: cleaner than what TagoClip
    # ships today at its highest setting, while resampling by 4 instead of 8,
    # which is half the filter arithmetic and half the curve evaluations.
    x = _sine(freq)
    shipping = Clipper(curve=name, threshold=67 / 128, oversample=8, drive_db=11.0).process(x)
    candidate = ADAAClipper(curve=name, threshold=67 / 128, oversample=4, drive_db=11.0).process(x)
    assert alias_nmr_db(candidate, SR, freq) < alias_nmr_db(shipping, SR, freq)


@pytest.mark.parametrize("name", CURVE_NAMES)
def test_adaa_at_2x_only_wins_below_the_top_octave(name):
    # Pins a limit that a single operating point hides: at 5 kHz ADAA 2x beats
    # plain 8x, at 11 kHz it loses by roughly 11 dB because two times headroom
    # is not enough for a fundamental that high. This is why the candidate runs
    # at 4x and not at 2x.
    for freq, adaa_should_win in ((5000.0, True), (11000.0, False)):
        x = _sine(freq)
        shipping = Clipper(curve=name, threshold=67 / 128, oversample=8, drive_db=11.0).process(x)
        two_x = ADAAClipper(curve=name, threshold=67 / 128, oversample=2, drive_db=11.0).process(x)
        wins = alias_nmr_db(two_x, SR, freq) < alias_nmr_db(shipping, SR, freq)
        assert wins is adaa_should_win


@pytest.mark.parametrize("name", CURVE_NAMES)
def test_the_legitimate_third_harmonic_survives(name):
    # Cleaning up alias must not quietly change the character of the curve.
    # Checked at the candidate setting; at 1x the same figure is -1.7 dB, which
    # is the rolloff pinned down in the test below.
    x = _sine(5000.0)
    reference = Clipper(curve=name, threshold=67 / 128, oversample=8, drive_db=11.0).process(x)
    candidate = ADAAClipper(curve=name, threshold=67 / 128, oversample=4, drive_db=11.0).process(x)
    h3_ref = harmonic_level_db(reference, SR, 15000.0)
    h3_adaa = harmonic_level_db(candidate, SR, 15000.0)
    assert abs(h3_ref - h3_adaa) < 1.0


def test_first_order_adaa_rolls_off_the_top_end():
    # Not a defect but the price of the method, and it has to stay pinned: below
    # the knee ADAA is the two-point average, so its magnitude response is
    # cos(pi*f/fs). At 15 kHz that is -6.35 dB at 1x and -1.30 dB at 2x, which
    # is why the candidate runs at 2x and why the ear, not this test, decides
    # whether 4x is needed.
    for oversample, expected_db in ((1, -6.35), (2, -1.30)):
        x = _sine(15000.0, amp=0.1, n=2**15)
        y = ADAAClipper(curve="fl", threshold=0.9, oversample=oversample, eps=1e-12).process(x)
        assert rms_db(y) - rms_db(x) == pytest.approx(expected_db, abs=0.1)


@pytest.mark.parametrize("name", CURVE_NAMES)
def test_low_frequency_level_is_untouched(name):
    # At 100 Hz there is nothing to antialias, so ADAA must not move the level.
    x = _sine(100.0, n=SR)
    static = Clipper(curve=name, threshold=0.5, oversample=1, drive_db=12.0).process(x)
    adaa = ADAAClipper(curve=name, threshold=0.5, oversample=1, drive_db=12.0).process(x)
    assert abs(rms_db(adaa) - rms_db(static)) < 0.1


@pytest.mark.parametrize("name", CURVE_NAMES)
def test_block_processing_is_bit_identical_to_one_buffer(name):
    # Prerequisite for the later golden tests in the plugin.
    x = _sine(3000.0, amp=2.0, n=8192)
    whole = ADAAClipper(curve=name, threshold=0.5).process(x)
    chunked = ADAAClipper(curve=name, threshold=0.5)
    pieces = [chunked.process(block) for block in np.split(x, [64, 700, 701, 4096])]
    assert np.array_equal(np.concatenate(pieces), whole)


def test_tanh_survives_extreme_drive():
    # ln cosh overflows in its direct form; the stable form must not.
    y = ADAAClipper(curve="tanh", threshold=0.5, drive_db=60.0).process(_sine(1000.0, n=4096))
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 1.0 + 1e-9


def test_reset_clears_the_carried_sample():
    clip = ADAAClipper(curve="fl", threshold=0.5)
    x = _sine(1000.0, n=512)
    first = clip.process(x)
    clip.reset()
    assert np.array_equal(clip.process(x), first)


def test_fallback_rate_reports_the_ill_conditioned_share():
    quiet = ADAAClipper(curve="fl", threshold=0.5, eps=1e-3)
    assert quiet.fallback_rate(_sine(20.0, amp=1e-4, n=4096)) > 0.9
    loud = ADAAClipper(curve="fl", threshold=0.5, eps=1e-9)
    assert loud.fallback_rate(_sine(5000.0, n=4096)) < 0.01


def test_invalid_args():
    with pytest.raises(ValueError):
        ADAAClipper(curve="fuzz")
    with pytest.raises(ValueError):
        ADAAClipper(oversample=3)
    with pytest.raises(ValueError):
        ADAAClipper(eps=0.0)
    with pytest.raises(ValueError):
        ADAAClipper(curve="fl", threshold=1.5)
    with pytest.raises(ValueError):
        ADAAClipper().process(np.zeros((2, 64)))
