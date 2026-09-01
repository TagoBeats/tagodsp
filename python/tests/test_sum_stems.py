"""Tests for sum-stem detection.

Every case here encodes a defect that actually happened while running the X-Ray
masking corpus over Robin's stem library, not a hypothetical one. See
tagodsp.analysis.sum_stems for the measured ground truth behind the thresholds.
"""

import numpy as np
import pytest

from tagodsp.analysis.sum_stems import (
    SUM_CORR_PARTNER_MIN,
    find_sum_stems,
    looks_like_sum_name,
)

SR = 48000
N = SR * 4  # 4 s is plenty; find_sum_stems reads at most SUM_CORR_SECONDS anyway


def noise(rng: np.random.Generator, n: int = N) -> np.ndarray:
    return rng.standard_normal(n)


def shifted(x: np.ndarray, lag: int) -> np.ndarray:
    """x delayed by `lag` samples, zero padded, same length."""
    out = np.zeros_like(x)
    out[lag:] = x[: len(x) - lag]
    return out


@pytest.fixture
def rng():
    return np.random.default_rng(20260901)


def names_of(verdicts):
    return [v.name for v in verdicts]


def test_sum_stem_is_detected(rng):
    """A stem that is the plain sum of the others is found, the others are not."""
    a, b, c = noise(rng), noise(rng), noise(rng)
    found = find_sum_stems(["a", "b", "c", "mixdown"], [a, b, c, a + b + c], SR)
    assert names_of(found) == ["mixdown"]
    assert found[0].corr > 0.99


def test_sum_stem_detected_despite_lag(rng):
    """Plugin delay compensation shifts a master bounce; the lag search covers it.

    Measured in the wild: fieber_Master sat 44.5 ms behind the sum of its stems
    and correlated at -0.12 at zero lag, +0.967 once the lag was searched.
    """
    a, b, c = noise(rng), noise(rng), noise(rng)
    lag = int(round(0.0445 * SR))
    found = find_sum_stems(["a", "b", "c", "master"], [a, b, c, shifted(a + b + c, lag)], SR)
    assert names_of(found) == ["master"]


def test_two_identical_sum_copies_both_detected(rng):
    """FL exports the mix twice ("_Master" and "_Current", bit identical).

    Both have to go. This only works because removal is iterative and one at a
    time: with both still in the pool, each copy sits inside the comparison sum of
    the other and covers for it.
    """
    a, b, c = noise(rng), noise(rng), noise(rng)
    total = a + b + c
    found = find_sum_stems(["a", "b", "c", "master", "current"], [a, b, c, total, total.copy()], SR)
    assert sorted(names_of(found)) == ["current", "master"]


def test_doubled_layer_survives(rng):
    """A doubled layer looks like a sum against the full comparison sum but is not.

    Measured in the wild: "sorry bout it [808]" correlated +0.775 against the sum
    of all other stems, but -0.663 once its one partner ("mainstream [808]") was
    removed. Nothing may be dropped here.
    """
    a, b, c = noise(rng), noise(rng), noise(rng)
    double = 0.9 * shifted(a, 120)  # same source, slightly scaled and delayed
    found = find_sum_stems(["a", "b", "c", "a_double"], [a, b, c, double], SR)
    assert found == []


def test_partner_test_skipped_when_too_few_stems(rng):
    """With under 3 stems left after removing the partner, only check (a) decides.

    Three stems total: dropping the candidate and its partner leaves one, so the
    partner test cannot run and the verdict records corr_without_partner=None.
    """
    a, b = noise(rng), noise(rng)
    found = find_sum_stems(["a", "b", "sum"], [a, b, a + b], SR)
    assert names_of(found) == ["sum"]
    assert found[0].corr_without_partner is None


def test_partner_test_threshold_is_not_tautological(rng):
    """A real sum stays well above the partner threshold, a double falls far below.

    Guards the gap the two thresholds live in: if this ever narrows, the filter
    starts eating real stems (which is exactly what happened before the partner
    test existed).
    """
    a, b, c, d = noise(rng), noise(rng), noise(rng), noise(rng)
    real_sum = find_sum_stems(["a", "b", "c", "d", "sum"], [a, b, c, d, a + b + c + d], SR)
    assert real_sum[0].corr_without_partner > SUM_CORR_PARTNER_MIN
    double = find_sum_stems(["a", "b", "c", "d", "dbl"], [a, b, c, d, 0.9 * a], SR)
    assert double == []


def test_silent_stem_is_not_a_sum(rng):
    """An all-zero stem correlates with nothing and must not be dropped."""
    a, b, c = noise(rng), noise(rng), noise(rng)
    found = find_sum_stems(["a", "b", "c", "silence"], [a, b, c, np.zeros(N)], SR)
    assert found == []


@pytest.mark.parametrize(
    "name",
    [
        "beat_Master",
        "beat_Current",
        "beat_Drum Bus",
        "beat_Loop Bus",
        "beat_mixdown",
        "Master",
        "night line_MASTER",
    ],
)
def test_name_prefilter_catches_sums(name):
    assert looks_like_sum_name(name)


@pytest.mark.parametrize(
    "name",
    [
        # source separation folders: "instruments" is a real stem, never a sum
        "song.mp3_instruments",
        "song.mp3_vocals",
        "beat_aug - nylex kick",
        "beat_Analog Lab V #2",
        "beat_Omnisphere",
        "beat_@prod.xtc - sorry bout it [808]",
    ],
)
def test_name_prefilter_keeps_real_stems(name):
    assert not looks_like_sum_name(name)
