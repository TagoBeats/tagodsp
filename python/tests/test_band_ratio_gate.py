import numpy as np

from tagodsp.spectral.band_ratio_gate import BandRatioGate
from tagodsp.utils.events import gate_to_events

SR = 48000


def test_detects_bursts_not_tone(sibilant_like_signal):
    x, spans = sibilant_like_signal
    gate = BandRatioGate(sr=SR)
    res = gate.process(x)

    events = gate_to_events(res.gate, len(x), SR, gate.stft.hop, min_ms=30.0)
    assert len(events) == len(spans)
    # each detected event overlaps its burst
    for (s, e), (bs, be) in zip(events, spans):
        assert max(s, bs) < min(e, be)


def test_silence_produces_no_events():
    res = BandRatioGate(sr=SR).process(np.zeros(SR))
    assert res.n_events == 0
    assert res.active_pct == 0.0


def test_pure_tone_no_false_trigger():
    t = np.arange(2 * SR) / SR
    x = 0.5 * np.sin(2 * np.pi * 800 * t)
    res = BandRatioGate(sr=SR).process(x)
    assert res.active_pct < 1.0


def test_offset_controls_sensitivity(sibilant_like_signal):
    x, _ = sibilant_like_signal
    loose = BandRatioGate(sr=SR, offset_db=3.0).process(x)
    strict = BandRatioGate(sr=SR, offset_db=20.0).process(x)
    assert loose.active_pct > strict.active_pct
