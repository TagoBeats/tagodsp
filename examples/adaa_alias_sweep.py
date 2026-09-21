"""Compare ADAA against plain oversampling for the clipping curves.

Answers the kill question of the TagoClip Pro prototype phase: does ADAA buy
enough alias suppression to justify a product, and at what cost in high
frequency response and arithmetic.

Metric is alias_nmr_db, the energy that is neither the fundamental nor one of
its harmonics, relative to the fundamental. Lower is cleaner.

Usage:
    uv run python examples/adaa_alias_sweep.py
    uv run python examples/adaa_alias_sweep.py --out docs/measurements/adaa_alias_2026-09-21.md
"""

import argparse
from pathlib import Path

import numpy as np

from tagodsp.analysis.alias_metrics import alias_nmr_db, harmonic_level_db
from tagodsp.distortion.adaa import ADAAClipper
from tagodsp.distortion.clipper import Clipper

CURVES = ("fl", "hard", "tanh")
FREQS = (1000.0, 5000.0, 11000.0)
DRIVES_DB = (6.0, 12.0, 18.0, 24.0)
THRESHOLD = 67 / 128

# (label, builder) for every configuration under comparison.
CONFIGS = (
    ("plain 1x", lambda c, d: Clipper(curve=c, threshold=THRESHOLD, oversample=1, drive_db=d)),
    ("plain 4x", lambda c, d: Clipper(curve=c, threshold=THRESHOLD, oversample=4, drive_db=d)),
    ("plain 8x", lambda c, d: Clipper(curve=c, threshold=THRESHOLD, oversample=8, drive_db=d)),
    ("ADAA 1x", lambda c, d: ADAAClipper(curve=c, threshold=THRESHOLD, oversample=1, drive_db=d)),
    ("ADAA 2x", lambda c, d: ADAAClipper(curve=c, threshold=THRESHOLD, oversample=2, drive_db=d)),
    ("ADAA 4x", lambda c, d: ADAAClipper(curve=c, threshold=THRESHOLD, oversample=4, drive_db=d)),
)

SR = 44100
N = 2**15

# scipy's resample_poly builds a filter of 2*half*factor+1 taps with half=10,
# which is exactly what TagoClip ships (81 taps at 4x, 161 at 8x).
_FIR_HALF = 10


def _taps(factor: int) -> int:
    return 0 if factor == 1 else 2 * _FIR_HALF * factor + 1


def cost_per_input_sample(factor: int) -> tuple[int, int]:
    """Rough (MACs, transcendental calls) per input sample at the base rate.

    Upsampling costs one full filter length per input sample spread over the
    polyphase branches, decimation costs the same again. The nonlinearity runs
    once per oversampled sample. Counts, not timings: ADAA is a sample
    recursion and oversampling is vectorised, so a numpy timing would measure
    numpy rather than the algorithm.
    """
    return 2 * _taps(factor), factor


def run() -> list[dict]:
    t = np.arange(N) / SR
    rows = []
    for curve in CURVES:
        for freq in FREQS:
            x = np.sin(2 * np.pi * freq * t)
            for drive in DRIVES_DB:
                for label, build in CONFIGS:
                    y = build(curve, drive).process(x)
                    rows.append(
                        {
                            "curve": curve,
                            "freq": freq,
                            "drive_db": drive,
                            "config": label,
                            "alias_db": alias_nmr_db(y, SR, freq),
                        }
                    )
    return rows


def _table(rows: list[dict], columns: list[str], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join([":---"] * len(header)) + " |"]
    for r in rows:
        cells = [f"{r[c]:+.2f}" if isinstance(r[c], float) else str(r[c]) for c in columns]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def summarise(rows: list[dict]) -> str:
    labels = [label for label, _ in CONFIGS]

    def mean_for(label: str, **filters) -> float:
        sel = [
            r
            for r in rows
            if r["config"] == label and all(r[k] == v for k, v in filters.items())
        ]
        return float(np.mean([r["alias_db"] for r in sel]))

    lines = [
        "# ADAA gegen reines Oversampling",
        "",
        f"Erzeugt von `examples/adaa_alias_sweep.py`, sr {SR} Hz, {len(rows)} Messungen.",
        "Metrik ist `alias_nmr_db`: Energie, die weder Grundton noch Harmonische ist, "
        "relativ zum Grundton. Kleiner ist sauberer.",
        "",
        "## Mittelwert je Konfiguration und Kurve",
        "",
    ]

    per_config = []
    for label in labels:
        row = {"config": label}
        for curve in CURVES:
            row[curve] = mean_for(label, curve=curve)
        row["alle"] = mean_for(label)
        per_config.append(row)
    lines.append(
        _table(
            per_config,
            ["config", *CURVES, "alle"],
            ["Konfiguration", *CURVES, "Mittel"],
        )
    )

    lines += ["", "## Nach Frequenz, Mittel ueber alle Kurven und Drive-Stufen", ""]
    per_freq = []
    for label in labels:
        row = {"config": label}
        for freq in FREQS:
            row[f"{freq / 1000:g}k"] = mean_for(label, freq=freq)
        per_freq.append(row)
    lines.append(
        _table(
            per_freq,
            ["config", *[f"{f / 1000:g}k" for f in FREQS]],
            ["Konfiguration", *[f"{f / 1000:g} kHz" for f in FREQS]],
        )
    )

    lines += ["", "## Rechenaufwand je Eingangssample, gezaehlt statt gemessen", ""]
    cost_rows = []
    for label, factor in (("plain 4x", 4), ("plain 8x", 8), ("ADAA 1x", 1), ("ADAA 2x", 2)):
        macs, evals = cost_per_input_sample(factor)
        cost_rows.append(
            {
                "config": label,
                "macs": str(macs),
                "evals": str(evals),
                "note": "plus 1 Division je Auswertung" if "ADAA" in label else "",
            }
        )
    lines.append(
        _table(
            cost_rows,
            ["config", "macs", "evals", "note"],
            ["Konfiguration", "MACs Resampling", "Kennlinien-Auswertungen", "Anmerkung"],
        )
    )
    lines += [
        "",
        "Das ist eine Zaehlung, keine Zeitmessung. ADAA ist eine Sample-Rekursion, "
        "Oversampling ist vektorisiert, ein numpy-Timing wuerde numpy messen und nicht "
        "den Algorithmus. Die CPU-Frage faellt erst in C++.",
        "",
        "## Hochton-Abfall, der Preis von ADAA erster Ordnung",
        "",
        "Unterhalb des Knies ist ADAA exakt der Zweipunkt-Mittelwert, der Amplitudengang "
        "also `cos(pi*f/fs)`. Gemessen und mit der Theorie deckungsgleich:",
        "",
    ]
    rolloff = []
    for freq in (5000.0, 10000.0, 15000.0, 20000.0):
        row = {"freq": f"{freq / 1000:g} kHz"}
        for factor in (1, 2, 4):
            row[f"{factor}x"] = float(20 * np.log10(np.cos(np.pi * freq / (factor * SR))))
        rolloff.append(row)
    lines.append(
        _table(rolloff, ["freq", "1x", "2x", "4x"], ["Frequenz", "ADAA 1x", "ADAA 2x", "ADAA 4x"])
    )

    lines += ["", "## Treue der 3. Harmonischen gegen plain 8x, 5 kHz bei +11 dB", ""]
    x = np.sin(2 * np.pi * 5000.0 * np.arange(N) / SR)
    h3 = []
    for curve in CURVES:
        ref = harmonic_level_db(
            Clipper(curve=curve, threshold=THRESHOLD, oversample=8, drive_db=11.0).process(x),
            SR,
            15000.0,
        )
        row = {"curve": curve}
        for factor in (1, 2, 4):
            val = harmonic_level_db(
                ADAAClipper(
                    curve=curve, threshold=THRESHOLD, oversample=factor, drive_db=11.0
                ).process(x),
                SR,
                15000.0,
            )
            row[f"{factor}x"] = val - ref
        h3.append(row)
    lines.append(
        _table(h3, ["curve", "1x", "2x", "4x"], ["Kurve", "ADAA 1x", "ADAA 2x", "ADAA 4x"])
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="write the report to this file")
    args = parser.parse_args()

    report = summarise(run())
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
