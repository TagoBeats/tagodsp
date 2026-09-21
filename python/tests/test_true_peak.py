import numpy as np
import pytest

from tagodsp.analysis.true_peak import isp_overshoot_db, true_peak, true_peak_db
from tagodsp.distortion.clipper import Clipper
from tagodsp.utils.gain import peak_db


def _sine(freq: float, sr: int, n: int, phase: float = 0.0, amp: float = 1.0) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / sr + phase)


def test_quarter_nyquist_sine_is_the_textbook_3db_case():
    # Sine at fs/4 sampled at 45 degrees: every sample sits at sin(pi/4),
    # the reconstruction reaches 1.0. Analytic overshoot is exactly 3.01 dB.
    sr = 48000
    x = _sine(sr / 4, sr, 4096, phase=np.pi / 4)
    assert peak_db(x) == pytest.approx(-3.0103, abs=1e-6)
    assert true_peak_db(x) == pytest.approx(0.0, abs=0.05)
    assert isp_overshoot_db(x) == pytest.approx(3.0103, abs=0.05)


def test_low_frequency_sine_has_no_meaningful_overshoot():
    # 100 Hz at 44100: 441 samples per cycle, so a sample lands almost exactly
    # on the peak. Analytic shortfall is below 0.001 dB.
    sr = 44100
    x = _sine(100.0, sr, 10 * 441)
    assert isp_overshoot_db(x) < 0.01


def test_true_peak_is_never_below_sample_peak():
    rng = np.random.default_rng(7)
    for _ in range(20):
        x = rng.standard_normal(2048) * 0.3
        # Interpolation passes through the samples, so the maximum of the
        # reconstruction cannot fall below the sampled maximum. Tolerance
        # covers the filter not being an exact Nyquist filter.
        assert true_peak_db(x) > peak_db(x) - 0.01


def test_hard_clipping_creates_inter_sample_peaks():
    # Detection test, not a magnitude claim: the actual numbers per curve,
    # threshold and drive come from examples/isp_sweep.py. What must hold is
    # that clipping lifts the true peak above the sample peak by more than the
    # 0.1 dB that delivery specs still care about, while the same sine unclipped
    # stays flat. A clipped sine is not an ideal square wave, so the textbook
    # 9 percent Gibbs overshoot is an upper bound, not the expected value.
    sr = 44100
    x = _sine(1000.0, sr, 3 * sr, amp=4.0)
    clipped = Clipper(curve="hard", threshold=1.0, oversample=1).process(x)
    assert peak_db(clipped) == pytest.approx(0.0, abs=1e-9)
    assert isp_overshoot_db(clipped) > 0.1
    assert isp_overshoot_db(clipped) > 10 * isp_overshoot_db(_sine(1000.0, sr, 3 * sr))


def test_multichannel_takes_the_loudest_channel():
    sr = 48000
    quiet = _sine(1000.0, sr, 4096, amp=0.1)
    loud = _sine(sr / 4, sr, 4096, phase=np.pi / 4)
    stereo = np.stack([quiet, loud])
    assert true_peak_db(stereo) == pytest.approx(true_peak_db(loud), abs=1e-9)


def test_higher_oversampling_never_reports_less():
    # A coarser meter can only miss peaks, never invent them.
    sr = 44100
    x = Clipper(curve="hard", threshold=1.0, oversample=1).process(
        _sine(7000.0, sr, 2 * sr, amp=3.0)
    )
    tp4 = true_peak_db(x, oversample=4)
    tp16 = true_peak_db(x, oversample=16)
    assert tp16 >= tp4 - 1e-9


def test_peak_in_the_trimmed_edge_does_not_produce_negative_overshoot():
    # The loudest sample sits inside the zone true_peak cannot measure. Both
    # peaks must then ignore it, otherwise the overshoot comes out negative.
    sr = 44100
    x = _sine(1000.0, sr, sr, amp=0.5)
    x[3] = 0.99
    assert isp_overshoot_db(x) >= 0.0


def test_silence_and_empty():
    assert true_peak(np.zeros(128)) == 0.0
    assert isp_overshoot_db(np.zeros(128)) == 0.0
    assert true_peak(np.array([])) == 0.0
    assert isp_overshoot_db(np.array([])) == 0.0


def test_invalid_args():
    with pytest.raises(ValueError):
        true_peak(np.zeros(64), oversample=3)
    with pytest.raises(ValueError):
        true_peak(np.zeros((2, 2, 64)))
