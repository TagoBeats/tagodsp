"""Criterion 3 of the fold gate: Raphael Kim's demo recipe, rendered for the ear.

His recipe, verbatim from the mail of 2026-09-22: four bars, a saw bass mixed
with white noise, plus a kick. He expects that to be enough to judge a fold
curve, and that claim is testable.

The measurements in examples/fold_gate.py already decided the numeric part: of
the four candidates only the smooth symmetric folder clears the alias bar, and
even that one is 17 dB dirtier than the clipping curves at its worst operating
point (5 kHz, +24 dB). So these pairs are not there to confirm the winner, they
are there to answer the questions the numbers cannot:

    shape_at_12db      triangle against smooth at the same drive. If the
                       cheap-sounding one is the one that reads as a folder,
                       the measurement won a fight the product loses.
    skew_at_12db       smooth symmetric against smooth skewed. Skew is what
                       Raphael singled out, and it is also what costs 11 dB of
                       alias. This pair is what that trade sounds like.
    against_fl         the candidate against the curve the product ships as
                       default. If the difference is small, a fourth chip is a
                       fourth chip for nothing.
    worst_cell         the smooth folder at +24 dB against the same at +12.
                       The numbers say the top of the drive range falls apart.
    adaa_vs_plain8x    the shipping configuration against brute force, same
                       comparison the ADAA pack ran on the clipping curves.

Every pair is RMS matched to its A side, because the louder one wins an A/B for
the wrong reason. The applied gain is printed so it stays visible.

Run:  uv run python examples/render_listenpack_fold.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from listenpack import ListenPack

from tagodsp.distortion.adaa import ADAAClipper
from tagodsp.distortion.clipper import Clipper
from tagodsp.utils.gain import lin_to_db, rms

SR = 44100
THRESHOLD = 67 / 128
BPM = 140.0
BARS = 4
NOTES_HZ = (55.0, 55.0, 65.41, 49.0)  # A1, A1, C2, G1, one per bar


def band_limited_saw(freq: float, n: int, sr: int = SR) -> np.ndarray:
    """Additive saw with every harmonic below Nyquist and none above it.

    A naive saw is already aliased before it reaches the curve, and a folder
    would then be blamed for dirt it inherited. This is the one place in the
    pack where being pedantic is not optional.
    """
    t = np.arange(n) / sr
    y = np.zeros(n)
    for k in range(1, int((sr / 2) / freq)):
        y -= np.sin(2 * np.pi * k * freq * t) / k
    return (2.0 / np.pi) * y


def demo_material(seed: int = 2026) -> np.ndarray:
    """Four bars: saw bass with white noise, plus a kick on every beat."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / BPM
    bar_n = int(round(4 * beat * SR))
    n = bar_n * BARS

    bass = np.concatenate([band_limited_saw(f, bar_n) for f in NOTES_HZ])

    # Noise sits well under the bass: Raphael mixes it in to hear what the
    # curve does to a broadband bed, not to make a noise track.
    noise = rng.standard_normal(n) * 0.06

    kick = np.zeros(n)
    beat_n = int(round(beat * SR))
    for hit in range(4 * BARS):
        start = hit * beat_n
        tail = np.arange(min(beat_n, n - start)) / SR
        sweep = np.sin(2 * np.pi * (45.0 + 75.0 * np.exp(-tail * 55.0)) * tail)
        kick[start : start + len(tail)] += sweep * np.exp(-tail * 22.0)

    mix = 0.7 * bass + noise + 0.6 * kick
    return mix / np.max(np.abs(mix)) * 0.7


def match_rms(reference: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale x to the RMS of reference. Returns the signal and the gain in dB."""
    r_ref, r_x = rms(reference), rms(x)
    if r_x <= 0.0:
        return x, 0.0
    g = r_ref / r_x
    return x * g, float(lin_to_db(g))


def adaa(curve: str, drive_db: float, oversample: int = 4) -> ADAAClipper:
    return ADAAClipper(
        curve=curve, threshold=THRESHOLD, oversample=oversample, drive_db=drive_db
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drive", type=float, default=12.0, help="drive for the shape pairs")
    args = parser.parse_args()

    x = demo_material()
    d = args.drive
    pack = ListenPack("fold_gate")

    cases = {
        "shape_at_12db": (adaa("fold", d), adaa("sinefold", d), "triangle", "smooth"),
        "skew_at_12db": (adaa("sinefold", d), adaa("skewsinefold", d), "symmetric", "skewed"),
        "against_fl": (adaa("fl", d), adaa("sinefold", d), "fl", "sinefold"),
        "worst_cell": (adaa("sinefold", 12.0), adaa("sinefold", 24.0), "12dB", "24dB"),
        "adaa_vs_plain8x": (
            adaa("sinefold", d),
            Clipper(curve="sinefold", threshold=THRESHOLD, oversample=8, drive_db=d),
            "adaa4x",
            "plain8x",
        ),
    }

    print(f"{'case':<18} {'A':<12} {'B':<12} gain on B")
    for item, (proc_a, proc_b, label_a, label_b) in cases.items():
        a = proc_a.process(x)
        b, gain = match_rms(a, proc_b.process(x))
        pack.add_pair(item, a, b, SR, label_a=label_a, label_b=label_b)
        pack.spectrum_plot(item, {label_a: a, label_b: b}, SR, title=item)
        print(f"{item:<18} {label_a:<12} {label_b:<12} {gain:+.2f} dB")

    # The unprocessed recipe, so "that is just what the source sounds like" is
    # a check and not an argument.
    pack.add_pair("source", x, x, SR, label_a="dry", label_b="dry")

    page = pack.audition_page("Fold gate, Raphael Kim's demo recipe")
    print(f"\npack: {pack.dir}\naudition: {page}")


if __name__ == "__main__":
    main()
