"""Render a biquad listening pack: pink noise through lowpass / highpass / peaking.

Run:  uv run python examples/render_listenpack_biquad.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from listenpack import ListenPack

from tagodsp.filters.biquad import Biquad, highpass, lowpass, magnitude_db, peaking
from tagodsp.utils.gain import db_to_lin, peak_db

SR = 48000
DUR_S = 4.0


def pink_noise(n: int, seed: int = 7) -> np.ndarray:
    """Pink noise via 1/f-shaped spectrum of white noise."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[1:] /= np.sqrt(f[1:])
    x = np.fft.irfft(spec, n)
    return x / np.max(np.abs(x)) * db_to_lin(-12.0)


def main() -> None:
    pack = ListenPack("biquad_demo")
    x = pink_noise(int(DUR_S * SR))

    cases = {
        "lowpass_1k": lowpass(1000, SR),
        "highpass_1k": highpass(1000, SR),
        "peaking_2k_+6dB": peaking(2000, SR, gain_db=6.0, q=2.0),
    }

    for name, coeffs in cases.items():
        y = Biquad(coeffs).process(x)
        pack.add_pair(name, x, y, SR, label_a="dry", label_b=name)
        pack.spectrum_plot(name, {"dry": x, name: y}, SR)
        print(f"{name}: peak {peak_db(y):+.1f} dBFS")

    # analytic responses in one overview plot
    import matplotlib.pyplot as plt

    freqs = np.geomspace(20, 20000, 512)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, coeffs in cases.items():
        ax.semilogx(freqs, magnitude_db(coeffs, freqs, SR), label=name)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title("Analytic magnitude responses (RBJ)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    pack.add_plot("magnitude_overview", fig)

    print(f"\nPack rendered: {pack.dir}")


if __name__ == "__main__":
    main()
