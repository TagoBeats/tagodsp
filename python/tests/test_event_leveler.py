import numpy as np

from tagodsp.dynamics.event_leveler import EventLeveler
from tagodsp.spectral.band_ratio_gate import BandRatioGate
from tagodsp.spectral.soft_mask import SoftMaskSeparator
from tagodsp.utils.events import gate_to_events
from tagodsp.utils.gain import rms_db

SR = 48000


def _pipeline(x, **leveler_kwargs):
    det = BandRatioGate(sr=SR).process(x)
    sep = SoftMaskSeparator(sr=SR).separate(x, det.gate)
    events = gate_to_events(
        det.gate, len(x), SR, 256, min_ms=30.0, tail_samples=512
    )
    lev = EventLeveler(sr=SR, **leveler_kwargs)
    plan = lev.plan(sep.x_masked, events)
    lev.apply_vetoes(plan, det.S_mag, det.freqs, hop=256)
    out = lev.render(sep.x_masked, plan)
    return det, sep, plan, out


def test_levels_land_in_corridor(sibilant_like_signal):
    x, _ = sibilant_like_signal
    _, _, plan, out = _pipeline(x, threshold_db=-30.0, target_db=-35.0, replacement=0.5)
    active = [ev for ev in plan if not ev.vetoed]
    assert active
    for ev in active:
        post = rms_db(out[ev.s : ev.e])
        assert abs(post - ev.desired_db) < 1.0
        assert -36.0 < post < -29.0


def test_zero_replacement_is_pure_gain(sibilant_like_signal):
    x, _ = sibilant_like_signal
    _, sep, plan, out = _pipeline(x, replacement=0.0, severity_enabled=False)
    # no synthesis: output is input scaled per event, so zero where input is zero
    quiet = slice(0, int(0.3 * SR))
    np.testing.assert_allclose(out[quiet], sep.x_masked[quiet])


def test_veto_on_overlong_event():
    lev = EventLeveler(sr=SR, max_event_ms=200.0)
    x = np.random.default_rng(0).standard_normal(SR) * 0.1
    plan = lev.plan(x, [(0, int(0.5 * SR))])  # 500 ms event
    n = lev.apply_vetoes(plan, np.ones((513, 200)), np.linspace(0, SR / 2, 513), hop=256)
    assert n == 1 and plan[0].vetoed


def test_severity_scales_rate():
    lev = EventLeveler(sr=SR)
    rng = np.random.default_rng(1)
    soft = rng.standard_normal(4800) * 0.05
    spiky = soft.copy()
    spiky[2400] = 1.0  # single hard peak → large peak-excess
    p_soft = lev.plan(soft, [(0, 4800)])[0]
    p_spiky = lev.plan(spiky, [(0, 4800)])[0]
    assert p_spiky.r_effective > p_soft.r_effective


def test_voiced_rate_reduction():
    lev = EventLeveler(sr=SR, severity_enabled=False, replacement=0.8)
    x = np.random.default_rng(2).standard_normal(4800) * 0.1
    p_s = lev.plan(x, [(0, 4800)], voiced=[False])[0]
    p_z = lev.plan(x, [(0, 4800)], voiced=[True])[0]
    assert np.isclose(p_z.r_effective, p_s.r_effective * 0.5)


def test_corridor_swap_and_boost_clamp():
    lev = EventLeveler(sr=SR, threshold_db=-40.0, target_db=-20.0, max_boost_db=15.0)
    x = np.full(4800, 1e-4)  # -80 dBFS
    ev = lev.plan(x, [(0, 4800)])[0]
    # corridor swapped to [-40, -20]; boost clamped to level + 15
    assert ev.desired_db <= ev.level_db + 15.0 + 1e-9
