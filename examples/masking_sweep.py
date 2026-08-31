"""Positive control for the masking detector: does it react to real masking at all?

Everything measured so far ran on one already mixed beat, so only false alarms
were visible, never false negatives. A detector that stays silent about
everything would have looked perfect there.

Two sweeps over synthetic signals with known ground truth, run for all three
scoring methods:

  level      target and masker share one band; the masker level is swept from
             well below to well above the target. Perceptually the target gets
             harder to hear the louder the masker is, and is fully buried at
             the top end. A detector that only fires near level parity has a
             blind spot exactly where masking is strongest.

  separation target level fixed, masker moved away in frequency. Masking should
             fall off as the two stop sharing a band.

Prints tables and writes plots. It states no verdict.

Run:  uv run python examples/masking_sweep.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import butter, sosfilt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from tagodsp.analysis.masking import MaskingDetector, Track  # noqa: E402
from tagodsp.spectral.stft import Stft  # noqa: E402

SR = 48000
DURATION = 4.0
WINDOW_SECONDS = 0.2222
SCORINGS = ("relative", "collision", "contention")
TARGET_HZ = 400.0
BANDWIDTH = 0.25  # +/- quarter octave around the center


def band_noise(center_hz: float, amp: float, seed: int) -> np.ndarray:
    """Band limited noise around center_hz, peak normalized then scaled to amp."""
    rng = np.random.default_rng(seed)
    n = int(DURATION * SR)
    lo = center_hz * 2**-BANDWIDTH
    hi = min(center_hz * 2**BANDWIDTH, SR / 2 * 0.99)
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    x = sosfilt(sos, rng.standard_normal(n))
    peak = np.max(np.abs(x))
    return (x / peak * amp) if peak > 0 else x


def score_of(target: np.ndarray, masker: np.ndarray, scoring: str) -> float:
    """Highest zone score the detector reports for this pair, 0.0 if it stays silent."""
    det = MaskingDetector(
        sr=SR, scoring=scoring, window_seconds=WINDOW_SECONDS, stft=Stft(4096, 2048)
    )
    res = det.analyze([Track("target", target), Track("masker", masker)])
    return max((z.score for z in res.zones), default=0.0)


def sweep_level(deltas: np.ndarray) -> dict[str, list[float]]:
    target = band_noise(TARGET_HZ, 0.2, seed=1)
    out = {s: [] for s in SCORINGS}
    for d in deltas:
        masker = band_noise(TARGET_HZ, 0.2 * 10 ** (d / 20.0), seed=2)
        for s in SCORINGS:
            out[s].append(score_of(target, masker, s))
    return out


def sweep_separation(octaves: np.ndarray) -> dict[str, list[float]]:
    target = band_noise(TARGET_HZ, 0.2, seed=1)
    out = {s: [] for s in SCORINGS}
    for o in octaves:
        masker = band_noise(TARGET_HZ * 2**o, 0.2, seed=2)
        for s in SCORINGS:
            out[s].append(score_of(target, masker, s))
    return out


def table(title: str, axis_label: str, axis: np.ndarray, data: dict[str, list[float]]) -> str:
    lines = [f"\n{title}", f"{axis_label:>12} " + "".join(f"{s:>13}" for s in SCORINGS)]
    lines.append("-" * (12 + 13 * len(SCORINGS)))
    for i, v in enumerate(axis):
        row = "".join(f"{data[s][i]:>13.3f}" for s in SCORINGS)
        lines.append(f"{v:>12.1f} {row}")
    return "\n".join(lines)


def plot(path: Path, axis, data, xlabel: str, title: str, note: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s in SCORINGS:
        ax.plot(axis, data[s], marker="o", label=s)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("highest zone score (0 = silent)")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.text(0.5, -0.02, note, ha="center", fontsize=8, color="#555")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Positive control sweeps for the masking detector")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "listenpacks" / "masking_sweep",
        help="folder for the plots",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    deltas = np.arange(-30.0, 31.0, 5.0)
    level = sweep_level(deltas)
    print(
        table(
            f"Level sweep, both at {TARGET_HZ:.0f} Hz, same time window",
            "masker dB",
            deltas,
            level,
        )
    )
    plot(
        args.out / "level_sweep.png",
        deltas,
        level,
        "masker level relative to target [dB]",
        f"Same band ({TARGET_HZ:.0f} Hz), masker level swept",
        "Right hand side is the fully buried target, the strongest masking case.",
    )

    octaves = np.arange(0.0, 3.01, 0.25)
    sep = sweep_separation(octaves)
    print(table("Separation sweep, equal level", "octaves", octaves, sep))
    plot(
        args.out / "separation_sweep.png",
        octaves,
        sep,
        "masker distance from target [octaves]",
        "Equal level, masker moved away in frequency",
        "Scores should fall to zero once the two no longer share a band.",
    )

    print(f"\nplots: {args.out}")


if __name__ == "__main__":
    main()
