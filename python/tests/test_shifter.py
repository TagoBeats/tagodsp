"""Analytic tests for pitch.shifter.PitchShifter."""

import numpy as np
import pytest

from tagodsp.pitch.shifter import PitchShifter

SR = 48000


def sine(freq: float, dur_s: float = 2.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(dur_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_freq(x: np.ndarray, sr: int = SR) -> float:
    """Peak frequency of the windowed FFT, refined by parabolic interpolation."""
    w = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * w))
    k = int(np.argmax(spec))
    if 0 < k < len(spec) - 1:
        a, b, c = np.log(spec[k - 1] + 1e-12), np.log(spec[k] + 1e-12), np.log(spec[k + 1] + 1e-12)
        k += 0.5 * (a - c) / (a - 2 * b + c)
    return k * sr / len(x)


def test_pitch_up_seven_semitones_moves_fft_peak():
    x = sine(440.0)
    shifter = PitchShifter(SR, pitch_semitones=7.0)
    y = shifter.process(x)
    expected = 440.0 * 2 ** (7 / 12)
    # analyze the middle to skip transients at the edges
    n = len(y)
    measured = dominant_freq(y[n // 4 : 3 * n // 4])
    assert measured == pytest.approx(expected, rel=0.01)


def test_pitch_down_octave():
    x = sine(440.0)
    y = PitchShifter(SR, pitch_semitones=-12.0).process(x)
    n = len(y)
    measured = dominant_freq(y[n // 4 : 3 * n // 4])
    assert measured == pytest.approx(220.0, rel=0.01)


def test_mix_zero_is_exact_passthrough():
    x = sine(300.0, dur_s=0.5)
    y = PitchShifter(SR, pitch_semitones=5.0, mix=0.0).process(x)
    np.testing.assert_allclose(y, x, atol=1e-7)


def test_gain_db_scales_output_after_mix():
    x = sine(300.0, dur_s=0.5)
    y = PitchShifter(SR, pitch_semitones=5.0, mix=0.0, gain_db=-6.0).process(x)
    np.testing.assert_allclose(y, x * 10 ** (-6.0 / 20.0), atol=1e-7)


def test_output_shape_and_sanity_mono_and_stereo():
    x = sine(220.0, dur_s=1.0)
    shifter = PitchShifter(SR, pitch_semitones=3.0, formant_semitones=-2.0)
    y = shifter.process(x)
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) < 1.0  # 0.5 amp sine must not clip through the engine

    st = np.stack([x, sine(330.0, dur_s=1.0)])
    y2 = shifter.process(st)
    assert y2.shape == st.shape


def test_wet_output_is_time_aligned():
    """Engine latency is compensated: pitch=0 output correlates at ~zero lag."""
    rng = np.random.default_rng(3)
    x = (rng.standard_normal(SR) * 0.1).astype(np.float32)
    y = PitchShifter(SR, pitch_semitones=0.0).process(x)
    corr = np.correlate(y, x, mode="full")
    lag = int(np.argmax(np.abs(corr))) - (len(x) - 1)
    assert abs(lag) <= 128


def test_latency_reported():
    shifter = PitchShifter(SR)
    assert shifter.latency_samples > 0


def test_parameter_validation():
    with pytest.raises(ValueError):
        PitchShifter(0)
    with pytest.raises(ValueError):
        PitchShifter(SR, mix=1.5)
    with pytest.raises(ValueError):
        PitchShifter(SR).process(np.zeros((2, 2, 2), dtype=np.float32))
