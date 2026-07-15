import numpy as np
import pytest

from tagodsp.distortion.clipper import (
    FL_THRESHOLD_DEFAULT,
    Clipper,
    fl_softclip,
    hardclip,
    tanh_clip,
)

# Golden pairs straight out of the FL Studio measurement bounce (threshold knob
# at default, i.e. t = 100/128). Inputs and outputs are float32-quantized by the
# render, hence the 2e-6 tolerance. Source: TagoClip/measure, 2026-07-15.
FL_GOLDEN = [
    (1.9835741519927979, 0.9991027116775513),
    (-1.308665156364441, -0.9803733229637146),
    (-2.954784870147705, -0.99998939037323),
    (-3.4486207962036133, -0.9999988675117493),
    (-3.1523194313049316, -0.9999957084655762),
    (-1.0123636722564697, -0.9239485859870911),
    (-3.251086473464966, -0.9999972581863403),
    (0.9300577044487, 0.8892067670822144),
]


def test_fl_matches_fruity_measurement():
    x = np.array([p[0] for p in FL_GOLDEN])
    y_ref = np.array([p[1] for p in FL_GOLDEN])
    y = fl_softclip(x, FL_THRESHOLD_DEFAULT)
    assert np.max(np.abs(y - y_ref)) < 2e-6


@pytest.mark.parametrize("curve", [fl_softclip, tanh_clip])
def test_identity_below_threshold(curve):
    t = 0.7
    x = np.linspace(-t, t, 101)
    assert np.allclose(curve(x, t), x, atol=1e-15)


@pytest.mark.parametrize("curve", [fl_softclip, tanh_clip, hardclip])
def test_odd_symmetry_and_ceiling(curve):
    x = np.linspace(-10, 10, 2001)
    y = curve(x)
    assert np.allclose(y, -curve(-x), atol=1e-15)
    assert np.max(np.abs(y)) <= 1.0 + 1e-12
    # monotonically non-decreasing
    assert np.all(np.diff(y) >= -1e-15)


def test_knee_is_c1_continuous():
    t = 0.6
    eps = 1e-9
    for curve in (fl_softclip, tanh_clip):
        below = (curve(t, t) - curve(t - eps, t)) / eps
        above = (curve(t + eps, t) - curve(t, t)) / eps
        assert below == pytest.approx(1.0, abs=1e-6)
        assert above == pytest.approx(1.0, abs=1e-6)


def test_hardclip():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert np.array_equal(hardclip(x), np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))


def test_clipper_os1_equals_bare_curve():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(4096)
    clip = Clipper(curve="fl", threshold=0.6, oversample=1)
    assert np.array_equal(clip.process(x), fl_softclip(x, 0.6))


def test_clipper_drive_and_length():
    x = np.full(1000, 0.1)
    clip = Clipper(curve="hard", threshold=1.0, drive_db=20.0, oversample=4)
    y = clip.process(x)
    assert len(y) == len(x)
    # 0.1 driven by +20 dB is 1.0, hard-clipped to 1.0 (resampler ripple aside)
    assert np.median(y) == pytest.approx(1.0, abs=1e-3)


def _tone_level_db(x: np.ndarray, sr: int, freq: float) -> float:
    """Peak spectrum level near freq, in dB relative to the global maximum."""
    w = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * w))
    spec_db = 20 * np.log10(spec + 1e-12)
    spec_db -= spec_db.max()
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    i = np.argmin(np.abs(freqs - freq))
    return float(spec_db[max(i - 20, 0) : i + 20].max())

def test_oversampling_kills_alias():
    # 5 kHz sine at +11 dB drive, sr 44100: the 9th harmonic (45 kHz) aliases
    # to 900 Hz. Measured at -32 dB in the FL original (oversample=1).
    sr = 44100
    t = np.arange(3 * sr) / sr
    x = 3.63 * np.sin(2 * np.pi * 5000 * t)
    naive = Clipper(curve="fl", threshold=67 / 128, oversample=1).process(x)
    oversampled = Clipper(curve="fl", threshold=67 / 128, oversample=8).process(x)
    alias_naive = _tone_level_db(naive, sr, 900.0)
    alias_os = _tone_level_db(oversampled, sr, 900.0)
    assert alias_naive > -40.0  # sanity: the alias really is there without OS
    assert alias_naive - alias_os > 40.0
    # the legitimate 3rd harmonic must survive oversampling
    h3_naive = _tone_level_db(naive, sr, 15000.0)
    h3_os = _tone_level_db(oversampled, sr, 15000.0)
    assert abs(h3_naive - h3_os) < 1.0


def test_invalid_args():
    with pytest.raises(ValueError):
        fl_softclip(np.zeros(4), threshold=1.5)
    with pytest.raises(ValueError):
        Clipper(curve="fuzz")
    with pytest.raises(ValueError):
        Clipper(oversample=3)
