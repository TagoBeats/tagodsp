"""Write the golden file the C++ masking port is verified against.

The C++ core in `cpp/include/tagodsp/masking.hpp` is a separate implementation,
not a binding, so the only thing that keeps the two honest is a value diff over
a fixture both sides can build. This script is the Python half: it generates the
fixture, runs `MaskingDetector` with scoring="contention", and writes every
stage of the pipeline as plain numbers. `cpp/tests/test_masking.cpp` regenerates
the same fixture from the spec below and compares.

The fixture is deliberately synthetic and short. It is a numeric fixture, not
audio: the gates are hard, so there are clicks, and that is fine. What it has to
do is exercise every stage (framing, band mapping, window aggregation, the
contention term, the BFS clustering, the frequency merge) with values that are
not all zero, in a few kilobytes of text.

Fixture spec, implemented identically on both sides:

  sr = 16000, n = 32000 samples (2.0 s)

  keys:  0.5 * sin(2*pi*255*t) + 0.5 * sin(2*pi*975*t),
         gated on for t in [0.0, 0.8) and [1.2, 2.0)
  pad:   0.4 * sin(2*pi*265*t) + 0.5 * sin(2*pi*985*t),
         gated on for t in [0.3, 1.5)
  noise: 0.05 * u[n], gated on for all t

Both conflict regions sit well above the low end on purpose. At n_fft = 1024 and
sr = 16000 a bin is 15.6 Hz wide, so a 70 Hz tone smears its main lobe across
three log-spaced bands and neither track ever contends cleanly with the other.
That is a property of the fixture, not of the detector, and it would have made
the golden depend on leakage instead of on the code under test.

  u[n] comes from a 64-bit LCG (Knuth's MMIX constants), so it is exactly
  reproducible in both languages without shipping a sample dump:

      state_0 = 12345
      state_{n+1} = (state_n * 6364136223846793005 + 1442695040888963407) mod 2^64
      u[n] = (state_{n+1} >> 11) * 2^-53 * 2 - 1

keys and pad overlap in time twice and share energy in two separated frequency
regions, around 260 Hz and around 980 Hz. That is on purpose: it makes the pair
break into two frequency-disjoint conflicts, so the summary layer's frequency
merge is exercised rather than assumed. noise is there so the run has a pair
that must stay silent.

The two implementations compute sin() through different libraries, so the
comparison runs against a relative tolerance rather than bit equality. Threshold
crossings are the one place where that could in principle flip a decision; on
this fixture the cell values sit far from the thresholds, and the test asserts
the zone and conflict counts as well as the values, so a flip would fail loudly
instead of passing quietly.
"""

import argparse
from pathlib import Path

import numpy as np

from tagodsp.analysis.masking import MaskingDetector, Track
from tagodsp.spectral.stft import Stft

SR = 16000.0
N_SAMPLES = 32000
N_FFT = 1024
HOP = 256
N_BANDS = 24
F_LO = 40.0
F_HI = 8000.0
WINDOW_SECONDS = 0.1
GAP = 2
MIN_LEN = 3
HIT_DB = -11.1
SCORE_DB = -9.75
LCG_SEED = 12345

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "cpp" / "tests" / "golden" / "masking.txt"


def lcg_noise(n: int, seed: int = LCG_SEED) -> np.ndarray:
    """Uniform noise in [-1, 1) from a 64-bit LCG, reproducible in C++."""
    mask = (1 << 64) - 1
    state = seed
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) & mask
        out[i] = ((state >> 11) * (2.0**-53)) * 2.0 - 1.0
    return out


def gate(t: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    g = np.zeros_like(t)
    for lo, hi in spans:
        g[(t >= lo) & (t < hi)] = 1.0
    return g


def fixture() -> list[Track]:
    t = np.arange(N_SAMPLES, dtype=np.float64) / SR
    keys = (0.5 * np.sin(2 * np.pi * 255.0 * t) + 0.5 * np.sin(2 * np.pi * 975.0 * t)) * gate(
        t, [(0.0, 0.8), (1.2, 2.0)]
    )
    pad = (0.4 * np.sin(2 * np.pi * 265.0 * t) + 0.5 * np.sin(2 * np.pi * 985.0 * t)) * gate(
        t, [(0.3, 1.5)]
    )
    noise = 0.05 * lcg_noise(N_SAMPLES)
    return [Track("keys", keys), Track("pad", pad), Track("noise", noise)]


def fmt(values) -> str:
    return " ".join(f"{float(v):.17g}" for v in np.asarray(values).ravel())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="golden file to write")
    args = parser.parse_args()

    tracks = fixture()
    detector = MaskingDetector(
        sr=SR,
        scoring="contention",
        n_bands=N_BANDS,
        f_lo=F_LO,
        f_hi=F_HI,
        window_seconds=WINDOW_SECONDS,
        gap=GAP,
        min_len=MIN_LEN,
        contention_hit_db=HIT_DB,
        contention_score_db=SCORE_DB,
        stft=Stft(n_fft=N_FFT, hop=HOP),
    )
    result = detector.analyze(tracks)

    lines: list[str] = []
    lines.append("# tagodsp masking golden, written by examples/masking_golden.py")
    lines.append("# see that script for the fixture spec the C++ test regenerates")
    for key, value in (
        ("sr", SR),
        ("n_samples", N_SAMPLES),
        ("n_fft", N_FFT),
        ("hop", HOP),
        ("n_bands", N_BANDS),
        ("f_lo", F_LO),
        ("f_hi", F_HI),
        ("window_seconds", WINDOW_SECONDS),
        ("gap", GAP),
        ("min_len", MIN_LEN),
        ("contention_hit_db", HIT_DB),
        ("contention_score_db", SCORE_DB),
        ("n_windows", result.n_windows),
    ):
        lines.append(f"{key} {value!r}" if isinstance(value, str) else f"{key} {value}")

    lines.append("tracks " + " ".join(t.name for t in tracks))
    lines.append("band_edges " + fmt(result.band_edges_hz))

    # Two raw STFT rows so a framing or FFT bug fails here instead of showing up
    # three stages later as a wrong band power.
    stft = Stft(n_fft=N_FFT, hop=HOP)
    power = np.abs(stft.forward(tracks[0].x)) ** 2  # (bins, frames)
    for frame in (0, 40):
        lines.append(f"stft_power_row keys {frame} " + fmt(power[:, frame]))

    for name, P in result.band_power.items():
        lines.append(f"band_power {name} " + fmt(P))
    for (a, b), C in result.cells.items():
        lines.append(f"cells {a} {b} " + fmt(C))

    lines.append(f"zones {len(result.zones)}")
    for z in result.zones:
        lines.append(
            "zone "
            + " ".join(
                str(v)
                for v in (
                    z.tracks[0],
                    z.tracks[1],
                    z.band,
                    f"{z.freq_lo_hz:.17g}",
                    f"{z.freq_hi_hz:.17g}",
                    z.windows[0],
                    z.windows[1],
                    f"{z.score:.17g}",
                )
            )
        )

    lines.append(f"conflicts {len(result.conflicts)}")
    for c in result.conflicts:
        lines.append(
            "conflict "
            + " ".join(
                str(v)
                for v in (
                    c.tracks[0],
                    c.tracks[1],
                    c.band,
                    f"{c.freq_lo_hz:.17g}",
                    f"{c.freq_hi_hz:.17g}",
                    c.windows[0],
                    c.windows[1],
                    c.active_windows,
                    c.occurrences,
                    f"{c.score:.17g}",
                )
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    print(
        f"n_windows {result.n_windows}, zones {len(result.zones)}, "
        f"conflicts {len(result.conflicts)}"
    )
    for c in result.conflicts:
        print(
            f"  {c.tracks[0]} x {c.tracks[1]} band {c.band} "
            f"{c.freq_lo_hz:.1f}-{c.freq_hi_hz:.1f} Hz "
            f"windows {c.windows[0]}-{c.windows[1]} "
            f"occurrences {c.occurrences} score {c.score:.3f}"
        )


if __name__ == "__main__":
    main()
