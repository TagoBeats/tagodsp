"""Render an ADAA listening pack: is the measured win audible, and what does it cost.

The sweep says ADAA at 4x is cleaner than plain 8x at half the arithmetic, and
that first-order ADAA rolls off the top end by 6.35 dB at 15 kHz when run at 1x
and by 0.31 dB at 4x. Numbers decide nothing about sound, so this pack puts the
four relevant configurations next to each other:

    adaa4x_vs_plain8x   the candidate against what TagoClip ships today.
                        Expected to be indistinguishable. If it is not, the
                        cheaper configuration is not free after all.
    adaa2x_vs_plain8x   the cheaper fallback, rolloff -1.30 dB at 15 kHz and
                        worse alias above 10 kHz.
    adaa1x_vs_plain8x   what the rolloff sounds like when it is unmistakable.
                        Calibration for the ear, not a product option.
    adaa1x_vs_plain1x   ADAA on its own against the raw FL behaviour, which is
                        the comparison a "clean mode" claim would rest on.

Every pair is RMS matched to its A side, because the louder one wins an A/B for
the wrong reason. The applied gain is printed so it stays visible.

Run:  uv run python examples/render_listenpack_adaa.py
      uv run python examples/render_listenpack_adaa.py --audio ~/Documents/loop.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from listenpack import ListenPack
from render_listenpack_clipper import synth_808, synth_hats

from tagodsp.analysis.alias_metrics import alias_nmr_db
from tagodsp.distortion.adaa import ADAAClipper
from tagodsp.distortion.clipper import Clipper
from tagodsp.utils.gain import lin_to_db, rms

SR = 44100
THRESHOLD = 67 / 128
DRIVE_DB = 11.0


def match_rms(reference: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale x to the RMS of reference. Returns the signal and the gain in dB."""
    r_ref, r_x = rms(reference), rms(x)
    if r_x <= 0.0:
        return x, 0.0
    g = r_ref / r_x
    return x * g, float(lin_to_db(g))


def drums(seconds: float = 4.0) -> np.ndarray:
    """808 plus hats, the material the plugin actually sees."""
    low = synth_808(seconds)
    high = synth_hats(seconds) * 0.35
    mix = low + high
    return mix / np.max(np.abs(mix))


def bright_noise(seconds: float = 3.0, seed: int = 5) -> np.ndarray:
    """Sustained bright material, the worst case for a high frequency rolloff."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * SR))
    fade = int(0.05 * SR)
    x[:fade] *= np.linspace(0, 1, fade)
    x[-fade:] *= np.linspace(1, 0, fade)
    return x / np.max(np.abs(x)) * 0.7


def sine(freq: float, seconds: float = 3.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    x = np.sin(2 * np.pi * freq * t)
    fade = int(0.01 * SR)
    x[:fade] *= np.linspace(0, 1, fade)
    x[-fade:] *= np.linspace(1, 0, fade)
    return x


def plain(oversample: int, x: np.ndarray) -> np.ndarray:
    return Clipper(
        curve="fl", threshold=THRESHOLD, oversample=oversample, drive_db=DRIVE_DB
    ).process(x)


def adaa(oversample: int, x: np.ndarray) -> np.ndarray:
    return ADAAClipper(
        curve="fl", threshold=THRESHOLD, oversample=oversample, drive_db=DRIVE_DB
    ).process(x)


# (item suffix, A oversampling, A label, B oversampling, B label)
COMPARISONS = (
    ("adaa4x_vs_plain8x", 8, "plain_8x_heute", 4, "adaa_4x"),
    ("adaa2x_vs_plain8x", 8, "plain_8x_heute", 2, "adaa_2x"),
    ("adaa1x_vs_plain8x", 8, "plain_8x_heute", 1, "adaa_1x"),
    ("adaa1x_vs_plain1x", 1, "plain_1x_wie_fl", 1, "adaa_1x"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="optional real-world wav to run through the same comparisons",
    )
    args = parser.parse_args()

    signals = {
        "drums": drums(),
        "hats": synth_hats(),
        "bright_noise": bright_noise(),
        "sine_5k": sine(5000.0),
        "sine_11k": sine(11000.0),
    }

    if args.audio is not None:
        import soundfile as sf

        data, sr_in = sf.read(args.audio)
        if sr_in != SR:
            raise SystemExit(f"{args.audio} is at {sr_in} Hz, expected {SR}")
        mono = data[:, 0] if data.ndim > 1 else data
        signals[args.audio.stem[:20]] = mono[: 8 * SR] / max(np.max(np.abs(mono)), 1e-12) * 0.9

    pack = ListenPack("adaa")
    print(f"{'Vergleich':<34} {'Signal':<14} {'Pegelabgleich':>14}")
    for suffix, plain_os, label_a, adaa_os, label_b in COMPARISONS:
        for name, x in signals.items():
            a = plain(plain_os, x)
            b_raw = adaa(adaa_os, x)
            b, gain_db = match_rms(a, b_raw)
            pack.add_pair(f"{name}_{suffix}", a, b, SR, label_a=label_a, label_b=label_b)
            print(f"{suffix:<34} {name:<14} {gain_db:>+13.2f} dB")

    for name, x in signals.items():
        pack.spectrum_plot(
            f"spectrum_{name}",
            {
                "plain 1x (wie FL)": plain(1, x),
                "plain 8x (heute)": plain(8, x),
                "ADAA 1x": adaa(1, x),
                "ADAA 4x (Kandidat)": adaa(4, x),
            },
            SR,
            title=f"{name}, FL-Kurve bei +{DRIVE_DB:.0f} dB Drive",
        )

    print()
    print("Alias-Metrik der Testtoene, kleiner ist sauberer:")
    for name, freq in (("sine_5k", 5000.0), ("sine_11k", 11000.0)):
        x = signals[name]
        values = {
            "plain 1x": alias_nmr_db(plain(1, x), SR, freq),
            "plain 8x": alias_nmr_db(plain(8, x), SR, freq),
            "ADAA 1x": alias_nmr_db(adaa(1, x), SR, freq),
            "ADAA 4x": alias_nmr_db(adaa(4, x), SR, freq),
        }
        joined = "  ".join(f"{k} {v:+.1f}" for k, v in values.items())
        print(f"  {name}: {joined}")

    page = pack.audition_page(title="TagoClip Pro: ADAA gegen Oversampling")
    print()
    print(f"Listenpack: {pack.dir}")
    print(f"Audition:   {page}")
    print()
    print("Worauf es ankommt: 'adaa4x_vs_plain8x' muss ununterscheidbar sein.")
    print("'adaa1x_vs_plain8x' zeigt, wie der Hochton-Abfall klingt, wenn er deutlich ist.")


if __name__ == "__main__":
    main()
