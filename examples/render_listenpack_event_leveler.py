"""Render the full event-leveler chain as a listening pack.

Chain: BandRatioGate -> SoftMaskSeparator -> EventLeveler -> recombine.
Input: synthetic "vocal" (tone + highband bursts at wildly different levels).

Run:  uv run python examples/render_listenpack_event_leveler.py [input.wav]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from listenpack import ListenPack

from tagodsp.dynamics.event_leveler import EventLeveler
from tagodsp.spectral.band_ratio_gate import BandRatioGate
from tagodsp.spectral.soft_mask import SoftMaskSeparator
from tagodsp.utils.events import gate_to_events
from tagodsp.utils.gain import rms_db

SR = 48000


def synthetic_vocal() -> np.ndarray:
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(11)
    n = int(6.0 * SR)
    t = np.arange(n) / SR
    # harmonic tone with energy in the anchor band (500-4k), like a vocal body
    x = np.zeros(n)
    for k in range(1, 12):
        x += (0.05 / k) * np.sin(2 * np.pi * 220 * k * t)
    x *= 1 + 0.3 * np.sin(2 * np.pi * 3 * t)
    sos = butter(4, [6000, 12000], btype="band", fs=SR, output="sos")
    for start_s, dur_s, gain in [
        (0.6, 0.12, 0.9),   # way too loud
        (1.8, 0.10, 0.5),
        (3.0, 0.08, 0.15),  # too quiet
        (4.4, 0.12, 0.7),
        (5.3, 0.09, 0.25),
    ]:
        s = int(start_s * SR)
        e = s + int(dur_s * SR)
        burst = sosfilt(sos, rng.standard_normal(e - s))
        burst /= np.max(np.abs(burst))
        fade = np.minimum(1.0, np.minimum(np.arange(e - s), np.arange(e - s)[::-1]) / 96)
        x[s:e] += gain * burst * fade
    return x


def main() -> None:
    if len(sys.argv) > 1:
        import soundfile as sf

        x, sr = sf.read(sys.argv[1], dtype="float64")
        if x.ndim > 1:
            x = x.mean(axis=1)
        assert sr == SR, f"expected {SR} Hz input"
    else:
        x = synthetic_vocal()

    det = BandRatioGate(sr=SR).process(x)
    print(f"detector: {det.n_events} events, {det.active_pct:.1f}% active")

    sep = SoftMaskSeparator(sr=SR).separate(x, det.gate)
    print(f"separation delta: {sep.peak_delta_dbfs:.1f} dBFS")

    events = gate_to_events(det.gate, len(x), SR, 256, min_ms=30.0, tail_samples=512)
    lev = EventLeveler(sr=SR, threshold_db=-30.0, target_db=-35.0, replacement=0.5)
    plan = lev.plan(sep.x_masked, events)
    n_veto = lev.apply_vetoes(plan, det.S_mag, det.freqs, hop=256)
    for ev in plan:
        tag = f"VETO ({ev.veto_reason})" if ev.vetoed else \
            f"{ev.level_db:6.1f} -> {ev.desired_db:6.1f} dB, r={ev.r_effective * 100:4.1f}%"
        print(f"  ev @ {ev.s / SR:6.3f}s  {tag}")
    print(f"vetoed: {n_veto}/{len(plan)}")

    x_masked_out = lev.render(sep.x_masked, plan)
    y = sep.x_rest + x_masked_out

    pack = ListenPack("event_leveler_demo")
    pack.add_pair("full_mix", x, y, SR, label_a="dry", label_b="leveled")
    pack.add_pair("masked_stream", sep.x_masked, x_masked_out, SR, label_a="dry", label_b="leveled")
    pack.spectrum_plot("full_mix_spectrum", {"dry": x, "leveled": y}, SR)

    for ev in plan:
        if not ev.vetoed:
            print(f"  post @ {ev.s / SR:6.3f}s: {rms_db(x_masked_out[ev.s:ev.e]):6.1f} dB "
                  f"(desired {ev.desired_db:6.1f})")
    print(f"\nPack rendered: {pack.dir}")


if __name__ == "__main__":
    main()
