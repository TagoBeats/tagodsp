import numpy as np
import pytest

SR = 48000


@pytest.fixture
def sibilant_like_signal():
    """1 kHz tone with three highband noise bursts, mimicking vowels + esses.

    Returns (x, burst_spans_samples).
    """
    rng = np.random.default_rng(3)
    dur = 3.0
    n = int(dur * SR)
    t = np.arange(n) / SR
    # tone quiet enough that bursts clearly exceed the anchor band (real esses
    # poke 5-15 dB over the vocal body; the ratio gate needs that contrast)
    x = 0.05 * np.sin(2 * np.pi * 1000 * t)

    from scipy.signal import butter, sosfilt

    sos = butter(4, [6000, 12000], btype="band", fs=SR, output="sos")
    spans = []
    for start_s, dur_s, gain in [(0.5, 0.12, 0.6), (1.4, 0.08, 0.5), (2.3, 0.1, 1.0)]:
        s = int(start_s * SR)
        e = s + int(dur_s * SR)
        burst = sosfilt(sos, rng.standard_normal(e - s))
        burst /= np.max(np.abs(burst))
        fade = np.minimum(1.0, np.minimum(np.arange(e - s), np.arange(e - s)[::-1]) / 96)
        x[s:e] += gain * burst * fade
        spans.append((s, e))
    return x, spans
