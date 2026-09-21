import numpy as np
import pytest

from tagodsp.analysis.alias_metrics import alias_nmr_db, harmonic_level_db
from tagodsp.distortion.clipper import Clipper

SR = 44100
N = 2**15


def _sine(freq: float, amp: float = 1.0, n: int = N, sr: int = SR) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / sr)


def test_known_interferer_is_measured_at_its_actual_level():
    # Analytic case: one tone at f0 plus one non-harmonic tone 40 dB down.
    # The metric must report that -40 dB and nothing else.
    x = _sine(1000.0) + _sine(3700.0, amp=0.01)
    assert alias_nmr_db(x, SR, 1000.0) == pytest.approx(-40.0, abs=0.5)


def test_leakage_floor_is_far_below_the_levels_of_interest():
    # A pure sine has no alias at all, so what the metric reports here is its
    # own window leakage. It has to sit well below the -40 dB case above,
    # otherwise the instrument cannot resolve what it is meant to measure.
    assert alias_nmr_db(_sine(1000.0), SR, 1000.0) < -100.0


def test_harmonics_are_not_counted_as_alias():
    # A soft-clipped sine at a low frequency puts all its harmonics below
    # Nyquist, so nothing folds back and the metric must stay near the floor.
    y = Clipper(curve="fl", threshold=0.5, oversample=16, drive_db=12.0).process(_sine(200.0))
    assert alias_nmr_db(y, SR, 200.0) < -60.0


def test_clipping_without_oversampling_is_measurably_dirtier():
    naive = Clipper(curve="fl", threshold=67 / 128, oversample=1, drive_db=11.0)
    clean = Clipper(curve="fl", threshold=67 / 128, oversample=8, drive_db=11.0)
    x = _sine(5000.0)
    dirty_db = alias_nmr_db(naive.process(x), SR, 5000.0)
    clean_db = alias_nmr_db(clean.process(x), SR, 5000.0)
    assert dirty_db - clean_db > 20.0


def test_harmonic_level_finds_the_third_harmonic():
    y = Clipper(curve="hard", threshold=0.5, oversample=16, drive_db=6.0).process(_sine(1000.0))
    h3 = harmonic_level_db(y, SR, 3000.0)
    empty = harmonic_level_db(y, SR, 3500.0)
    assert h3 > -40.0
    assert h3 - empty > 40.0


def test_rejects_multichannel_and_silence():
    with pytest.raises(ValueError):
        alias_nmr_db(np.zeros((2, N)), SR, 1000.0)
    with pytest.raises(ValueError):
        alias_nmr_db(np.zeros(N), SR, 1000.0)
