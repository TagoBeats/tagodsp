"""Render a clipper listening pack: FL curve vs tanh, naive vs oversampled.

Synth material (808, hats, 5 kHz sine) driven hot into the clipper, plus a
clone check of the FL mode against the real Fruity Soft Clipper bounces from
the TagoClip measurement session, if present.

Run:  uv run python examples/render_listenpack_clipper.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from listenpack import ListenPack

from tagodsp.distortion.clipper import FL_THRESHOLD_DEFAULT, Clipper, fl_softclip, tanh_clip
from tagodsp.filters.biquad import Biquad, highpass

SR = 44100

# Segment starts in the measurement bounce, seconds (see measure/exports/settings.txt)
FRUITY_SEGMENTS = {
    "sine_1k_hot": (17.12, 4.0, "full_thr_default", FL_THRESHOLD_DEFAULT),
    "sine_5k_hot": (30.83, 4.0, "full_thr_low30", 67 / 128),
    "transients": (37.69, 4.0, "full_thr_default", FL_THRESHOLD_DEFAULT),
}


def synth_808(seconds: float = 4.0) -> np.ndarray:
    """Four 808 hits: pitch glide 80 -> 55 Hz, exponential decay."""
    n_hit = int(0.9 * SR)
    t = np.arange(n_hit) / SR
    freq = 55.0 + 25.0 * np.exp(-t * 18.0)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    hit = np.sin(phase) * np.exp(-t * 3.5)
    out = np.zeros(int(seconds * SR))
    for k in range(4):
        start = int(k * SR)
        out[start : start + n_hit] += hit
    return out


def synth_hats(seconds: float = 4.0, seed: int = 11) -> np.ndarray:
    """Eighth-note noise bursts, highpassed at 6 kHz."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    out = np.zeros(n)
    burst_len = int(0.06 * SR)
    env = np.exp(-np.arange(burst_len) / SR * 90.0)
    step = int(0.25 * SR)
    for start in range(0, n - burst_len, step):
        out[start : start + burst_len] += rng.standard_normal(burst_len) * env
    out = Biquad(highpass(6000, SR)).process(out)
    return out / np.max(np.abs(out))


def sine_5k(seconds: float = 3.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    x = np.sin(2 * np.pi * 5000 * t)
    fade = int(0.01 * SR)
    x[:fade] *= np.linspace(0, 1, fade)
    x[-fade:] *= np.linspace(1, 0, fade)
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fruity-dir",
        type=Path,
        default=Path.home() / "Documents/TagoClip/measure",
        help="TagoClip measurement folder for the clone check (skipped if missing)",
    )
    args = parser.parse_args()

    pack = ListenPack("clipper")
    drive = 11.0

    signals = {"808": synth_808(), "hats": synth_hats(), "sine_5k": sine_5k()}
    for name, x in signals.items():
        naive = Clipper("fl", oversample=1, drive_db=drive).process(x)
        os8 = Clipper("fl", oversample=8, drive_db=drive).process(x)
        pack.add_pair(f"{name}_aliasing", naive, os8, SR, label_a="os1_wie_fruity", label_b="os8")
        pack.spectrum_plot(
            f"spectrum_{name}", {"os1 (wie Fruity)": naive, "os8": os8}, SR,
            title=f"{name}, FL-Kurve @ +{drive:.0f} dB Drive",
        )

    fl = Clipper("fl", oversample=8, drive_db=drive).process(signals["808"])
    th = Clipper("tanh", oversample=8, drive_db=drive).process(signals["808"])
    pack.add_pair("808_kurven", fl, th, SR, label_a="fl_kurve", label_b="tanh_kurve")

    # Transfer curve plot
    import matplotlib.pyplot as plt

    xi = np.linspace(-3, 3, 1000)
    fig, ax = plt.subplots(figsize=(7, 5))
    for t_val in (127 / 128, FL_THRESHOLD_DEFAULT, 67 / 128):
        ax.plot(xi, fl_softclip(xi, t_val), label=f"fl, t={t_val:.3f}")
    ax.plot(xi, tanh_clip(xi, FL_THRESHOLD_DEFAULT), "--", label="tanh, t=0.781")
    ax.set_xlabel("in")
    ax.set_ylabel("out")
    ax.grid(alpha=0.3)
    ax.legend()
    pack.add_plot("transfer_curves", fig)

    # Clone check against the real Fruity Soft Clipper renders
    if (args.fruity_dir / "exports/full_dry.wav").exists():
        import soundfile as sf

        dry = sf.read(args.fruity_dir / "exports/full_dry.wav")[0][:, 0]
        for name, (start, dur, render, t_val) in FRUITY_SEGMENTS.items():
            s0, s1 = int(start * SR), int((start + dur) * SR)
            fruity = sf.read(args.fruity_dir / f"exports/{render}.wav")[0][:, 0][s0:s1]
            ours = Clipper("fl", threshold=t_val, oversample=1).process(dry[s0:s1])
            pack.add_pair(f"clone_{name}", fruity, ours, SR,
                          label_a="fruity_original", label_b="tagoclip_fl_mode")
            print(f"clone {name}: max delta {np.max(np.abs(fruity - ours)):.2e}")
    else:
        print(f"Fruity renders not found under {args.fruity_dir}, clone check skipped")

    page = pack.audition_page(title="TagoClip Prototyp: FL-Kurve, Oversampling, Clone-Check")
    print(f"Listenpack: {pack.dir}")
    print(f"Audition:   {page}")


if __name__ == "__main__":
    main()
