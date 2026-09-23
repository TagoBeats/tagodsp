import numpy as np
import pytest

from tagodsp.analysis.alias_metrics import harmonic_level_db
from tagodsp.distortion.adaa import ANTIDERIVATIVES, ADAAClipper
from tagodsp.distortion.clipper import CURVES, Clipper
from tagodsp.distortion.folder import (
    SKEW_DEFAULT,
    antiderivative_fold,
    antiderivative_sinefold,
    sinefold,
    wavefold,
)

SR = 44100
TRIANGLE_NAMES = ("fold", "skewfold")
SINE_NAMES = ("sinefold", "skewsinefold")
FOLD_NAMES = (*TRIANGLE_NAMES, *SINE_NAMES)
SKEWS = (0.0, SKEW_DEFAULT, 0.6)
T = 0.6


def _sine(freq: float, amp: float = 1.0, n: int = 2**15, sr: int = SR) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / sr)


@pytest.mark.parametrize("skew", SKEWS)
def test_antiderivative_differentiates_back_to_the_curve(skew):
    # The property ADAA correctness rests on: dF1/dx must be the curve. The
    # fold corners are skipped, a central difference straddling a corner
    # averages both slopes and is wrong there by construction.
    x = np.linspace(-6.0, 6.0, 400001)
    h = x[1] - x[0]
    numeric = (antiderivative_fold(x[2:], T, skew) - antiderivative_fold(x[:-2], T, skew)) / (2 * h)
    inner = x[1:-1]

    # Corners sit at |x| = odd multiples of the side threshold. Distance is
    # converted back to x units before it is compared against the step size.
    t_side = np.where(inner >= 0.0, T, T * (1.0 - skew))
    u = np.abs(inner) / t_side
    distance = np.abs(np.mod(u, 2.0) - 1.0) * t_side
    away = distance > 4 * h

    error = np.abs(numeric - wavefold(inner, T, skew))[away]
    assert np.max(error) < 1e-9


@pytest.mark.parametrize("skew", SKEWS)
def test_identity_below_the_threshold_on_both_sides(skew):
    limit = T * (1.0 - skew)
    x = np.linspace(-limit, limit, 4001)
    np.testing.assert_allclose(wavefold(x, T, skew), x, atol=1e-15)


@pytest.mark.parametrize("skew", SKEWS)
def test_the_threshold_is_the_ceiling_at_every_drive(skew):
    # As with hardclip in this family the threshold is the ceiling, and skew
    # only ever lowers the negative peak. Whatever the drive, nothing leaves
    # the band the rest of the chain is allowed to assume.
    x = np.linspace(-40.0, 40.0, 400001)
    y = wavefold(x, T, skew)
    # Tolerance is the grid: the peaks sit at odd multiples of the side
    # threshold and the sweep only lands near them, not on them.
    assert np.max(np.abs(y)) <= T + 1e-12
    # Folding inverts, so a positive input reaches -t too and the overall range
    # is symmetric whatever the skew. What skew changes is the branch: the
    # negative half of the input never swings further than its own threshold.
    assert np.max(np.abs(y[x >= 0.0])) == pytest.approx(T, abs=1e-3)
    assert np.max(np.abs(y[x < 0.0])) == pytest.approx(T * (1.0 - skew), abs=1e-3)


@pytest.mark.parametrize("skew", SKEWS)
def test_the_curve_is_continuous(skew):
    # C0 everywhere including the corners: the folds reflect, they do not jump.
    x = np.linspace(-20.0, 20.0, 800001)
    step = np.max(np.abs(np.diff(wavefold(x, T, skew))))
    assert step < 2.0 * (x[1] - x[0])


def test_it_folds_rather_than_saturating():
    # What separates the candidate from the three clipping curves: past the
    # fold point the output comes back down, crosses zero, and at 3t a loud
    # sample comes out sign-inverted. No clipper can do that.
    assert wavefold(np.array([T]), T)[0] == pytest.approx(T)
    assert wavefold(np.array([2 * T]), T)[0] == pytest.approx(0.0, abs=1e-12)
    assert wavefold(np.array([3 * T]), T)[0] == pytest.approx(-T)
    assert wavefold(np.array([4 * T]), T)[0] == pytest.approx(0.0, abs=1e-12)


def test_the_slope_carries_through_the_fold_point():
    # A folder has no knee to round: below t the curve is the identity and the
    # first fold ramp arrives with slope 1 as well, so the only breaks in the
    # derivative are the corners themselves.
    h = 1e-7
    for x0 in (T - 1e-3, T - 1e-4):
        slope = (wavefold(np.array([x0 + h]), T)[0] - wavefold(np.array([x0 - h]), T)[0]) / (2 * h)
        assert slope == pytest.approx(1.0, abs=1e-6)


def test_symmetric_fold_is_odd_and_skewed_is_not():
    x = np.linspace(-8.0, 8.0, 40001)
    np.testing.assert_allclose(wavefold(x, T, 0.0), -wavefold(-x, T, 0.0), atol=1e-15)
    assert not np.allclose(
        wavefold(x, T, SKEW_DEFAULT), -wavefold(-x, T, SKEW_DEFAULT), atol=1e-6
    )


def test_skew_is_what_puts_even_harmonics_in_the_spectrum():
    # The reason skew exists at all. Measured on the second harmonic, which an
    # odd curve cannot produce and a skewed one can.
    x = _sine(1000.0)
    sym = Clipper(curve="fold", threshold=T, oversample=8, drive_db=6.0).process(x)
    skewed = Clipper(curve="skewfold", threshold=T, oversample=8, drive_db=6.0).process(x)
    h2_sym = harmonic_level_db(sym, SR, 2000.0)
    h2_skew = harmonic_level_db(skewed, SR, 2000.0)
    assert h2_sym < -80.0
    assert h2_skew > h2_sym + 40.0


@pytest.mark.parametrize("skew", SKEWS)
def test_antiderivative_is_continuous_and_zero_at_the_origin(skew):
    # F1 has to agree across the two sides where they meet, otherwise the ADAA
    # quotient sees a step every time the signal crosses zero.
    for sign in (1.0, -1.0):
        near = antiderivative_fold(np.array([sign * 1e-9]), T, skew)[0]
        assert near == pytest.approx(0.0, abs=1e-15)
    t_neg = T * (1.0 - skew)
    assert antiderivative_fold(np.array([T]), T, skew)[0] == pytest.approx(0.5 * T * T)
    assert antiderivative_fold(np.array([-t_neg]), T, skew)[0] == pytest.approx(0.5 * t_neg * t_neg)


def test_symmetric_antiderivative_is_even():
    x = np.linspace(0.0, 8.0, 8001)
    np.testing.assert_allclose(
        antiderivative_fold(x, T, 0.0), antiderivative_fold(-x, T, 0.0), atol=1e-15
    )


@pytest.mark.parametrize("name", FOLD_NAMES)
def test_the_curve_is_registered_for_the_shared_signal_path(name):
    # The architectural claim under gate: a fourth chip in a row that exists,
    # not a second product. Both dicts have to carry it or it is neither.
    assert name in CURVES
    assert name in ANTIDERIVATIVES
    x = _sine(500.0, amp=0.8, n=4096)
    assert Clipper(curve=name, oversample=4).process(x).shape == x.shape
    assert ADAAClipper(curve=name, oversample=4).process(x).shape == x.shape


@pytest.mark.parametrize("name", TRIANGLE_NAMES)
def test_below_the_knee_adaa_is_exactly_a_two_point_average(name):
    # Below the threshold the curve is the identity on both sides, so F1 is
    # x^2/2 and the quotient collapses to the mean of the two samples.
    x = _sine(300.0, amp=0.2, n=4096)
    y = ADAAClipper(curve=name, threshold=0.9, eps=1e-12).process(x)
    expected = 0.5 * (x + np.concatenate(([0.0], x[:-1])))
    assert np.max(np.abs(y - expected)) < 1e-12


@pytest.mark.parametrize("name", FOLD_NAMES)
def test_adaa_stays_inside_the_ceiling_at_every_drive(name):
    # A folder hits many more corners than a clipper hits knees, and the ADAA
    # quotient is at its most delicate near them. At 1x the output is an
    # average of curve values and cannot leave the band.
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 0.3, 2**15)
    for drive in (0.0, 6.0, 12.0, 24.0, 36.0):
        y = ADAAClipper(curve=name, threshold=T, drive_db=drive).process(x)
        assert np.all(np.isfinite(y))
        assert np.max(np.abs(y)) <= T + 1e-9


@pytest.mark.parametrize("name", TRIANGLE_NAMES)
def test_oversampled_overshoot_stays_bounded(name):
    # With oversampling the decimation filter rings around every corner and the
    # output does leave the band. That is the resampler, not the curve, and it
    # is what the true-peak stage downstream exists for. The bar is set by what
    # that stage budgets: more than 3 dB above the ceiling and the ceiling
    # control would have to give back more headroom than it has. Worst case
    # measured 2026-09-23 is 0.739 at threshold 0.6 and +12 dB, so 1.8 dB.
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 0.3, 2**15)
    peaks = [
        np.max(np.abs(ADAAClipper(curve=name, threshold=T, oversample=4, drive_db=d).process(x)))
        for d in (0.0, 6.0, 12.0, 24.0, 36.0)
    ]
    assert np.all(np.isfinite(peaks))
    assert max(peaks) < T * 1.41


@pytest.mark.parametrize("name", FOLD_NAMES)
def test_block_processing_is_bit_identical_to_one_buffer(name):
    x = _sine(700.0, amp=0.9, n=8192)
    whole = ADAAClipper(curve=name, threshold=T, drive_db=9.0).process(x)
    blocked = ADAAClipper(curve=name, threshold=T, drive_db=9.0)
    pieces = [blocked.process(block) for block in np.split(x, 8)]
    np.testing.assert_array_equal(whole, np.concatenate(pieces))


@pytest.mark.parametrize("skew", SKEWS)
def test_sinefold_antiderivative_differentiates_back_to_the_curve(skew):
    # No corners anywhere, so unlike the triangle version nothing has to be
    # masked out of the comparison. That is the whole point of this variant.
    # Looser bar than the triangle test for a reason that is the instrument and
    # not the curve: a central difference truncates at h^2 * y'''/6, and this
    # curve has a third derivative where a piecewise linear one has none. That
    # floor is 1.7e-9 here, so the bar sits above it.
    x = np.linspace(-6.0, 6.0, 200001)
    h = x[1] - x[0]
    numeric = (antiderivative_sinefold(x[2:], T, skew) - antiderivative_sinefold(x[:-2], T, skew)) / (
        2 * h
    )
    assert np.max(np.abs(numeric - sinefold(x[1:-1], T, skew))) < 1e-8


@pytest.mark.parametrize("skew", SKEWS)
def test_sinefold_peaks_at_the_threshold_and_stays_there(skew):
    x = np.linspace(-40.0, 40.0, 400001)
    y = sinefold(x, T, skew)
    assert np.max(np.abs(y)) <= T + 1e-12
    assert np.max(np.abs(y[x >= 0.0])) == pytest.approx(T, abs=1e-5)
    assert np.max(np.abs(y[x < 0.0])) == pytest.approx(T * (1.0 - skew), abs=1e-5)


def test_sinefold_has_unity_slope_at_the_origin_and_a_soft_knee():
    # Not the identity below the threshold like the other curves, but the
    # departure is a soft knee rather than a gain error: slope 1 at zero,
    # sin(1) = 0.841 at the threshold, so 1.5 dB just before the first fold.
    h = 1e-7
    slope = (sinefold(np.array([h]), T)[0] - sinefold(np.array([-h]), T)[0]) / (2 * h)
    assert slope == pytest.approx(1.0, abs=1e-9)
    assert sinefold(np.array([T]), T)[0] == pytest.approx(T * np.sin(1.0))


def test_sinefold_is_odd_at_zero_skew():
    x = np.linspace(-8.0, 8.0, 40001)
    np.testing.assert_allclose(sinefold(x, T, 0.0), -sinefold(-x, T, 0.0), atol=1e-15)


@pytest.mark.parametrize("name", SINE_NAMES)
def test_sinefold_adaa_stays_inside_the_ceiling(name):
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 0.3, 2**15)
    for drive in (0.0, 6.0, 12.0, 24.0, 36.0):
        y = ADAAClipper(curve=name, threshold=T, drive_db=drive).process(x)
        assert np.all(np.isfinite(y))
        assert np.max(np.abs(y)) <= T + 1e-9


def test_invalid_args():
    with pytest.raises(ValueError):
        wavefold(np.zeros(4), threshold=0.0)
    with pytest.raises(ValueError):
        wavefold(np.zeros(4), threshold=1.5)
    with pytest.raises(ValueError):
        wavefold(np.zeros(4), skew=1.0)
    with pytest.raises(ValueError):
        antiderivative_fold(np.zeros(4), skew=-0.1)
