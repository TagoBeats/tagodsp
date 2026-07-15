"""Render a mono-low listening pack: wide low end vs mono'd low end.

Material: an 808 with detuned L/R (phasey low end, the classic problem case)
and a mix of wide pad plus clean 808. A is untouched, B has MonoLow at 120 Hz.

Run:  uv run python examples/render_listenpack_mono_low.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from listenpack import ListenPack

from tagodsp.stereo.mono_low import MonoLow
from tagodsp.utils.gain import db_to_lin

SR = 44100
CUTOFF = 120.0


def detuned_808(seconds: float = 4.0) -> np.ndarray:
    """Four 808 hits with L/R detune: 54 vs 56 Hz, slow phase beating in the low end."""
    n_hit = int(0.9 * SR)
    t = np.arange(n_hit) / SR
    out = np.zeros((int(seconds * SR), 2))
    for ch, f0 in enumerate((54.0, 56.0)):
        freq = f0 + 25.0 * np.exp(-t * 18.0)
        phase = 2 * np.pi * np.cumsum(freq) / SR
        hit = np.sin(phase) * np.exp(-t * 3.5)
        for k in range(4):
            start = int(k * SR)
            out[start : start + n_hit, ch] += hit
    return out * db_to_lin(-3.0)


def pad_plus_808(seconds: float = 6.0, seed: int = 21) -> np.ndarray:
    """Wide detuned saw pad plus a clean mono 808 underneath."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    pad = np.zeros((n, 2))
    for ch in range(2):
        for f0 in (110.0, 164.8, 220.0):
            detune = 1.0 + rng.uniform(-0.004, 0.004)
            phase = 2 * np.pi * f0 * detune * t + rng.uniform(0, 2 * np.pi)
            saw = 2 * ((phase / (2 * np.pi)) % 1.0) - 1.0
            pad[:, ch] += saw
    pad /= np.max(np.abs(pad))
    pad *= db_to_lin(-16.0)

    n_hit = int(1.2 * SR)
    th = np.arange(n_hit) / SR
    freq = 55.0 + 25.0 * np.exp(-th * 18.0)
    hit = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-th * 2.5) * db_to_lin(-4.0)
    mix = pad.copy()
    for k in range(0, int(seconds), 2):
        start = int(k * SR)
        end = min(start + n_hit, n)
        mix[start:end, 0] += hit[: end - start]
        mix[start:end, 1] += hit[: end - start]
    return mix


def main() -> None:
    pack = ListenPack("mono_low")

    for name, x in {"808_detuned": detuned_808(), "pad_plus_808": pad_plus_808()}.items():
        ml = MonoLow(freq=CUTOFF, sr=SR)
        y = ml.process(x)
        pack.add_pair(name, x, y, SR, label_a="wide", label_b=f"mono_unter_{CUTOFF:.0f}Hz")
        side_in = (x[:, 0] - x[:, 1]) / 2
        side_out = (y[:, 0] - y[:, 1]) / 2
        pack.spectrum_plot(
            f"side_{name}", {"side vorher": side_in, "side nachher": side_out}, SR,
            title=f"{name}: Side-Kanal vor/nach MonoLow @ {CUTOFF:.0f} Hz",
        )

    page = pack.audition_page(title=f"MonoLow Prototyp, Crossover {CUTOFF:.0f} Hz (LR4)")
    print(f"Listenpack: {pack.dir}")
    print(f"Audition:   {page}")


if __name__ == "__main__":
    main()
