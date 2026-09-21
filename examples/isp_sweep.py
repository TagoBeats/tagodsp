"""Measure inter-sample peaks created by clipping.

Answers the first question of the TagoClip Pro prototype phase: after clipping,
how far does the reconstructed waveform rise above the sample peak, and does
oversampling alone already fix it. Run before touching ADAA, because it needs
no new DSP and can kill the product idea on its own.

The overshoot is a difference of two levels, so it needs no normalisation and
is directly comparable across settings.

Usage:
    uv run python examples/isp_sweep.py
    uv run python examples/isp_sweep.py --out docs/measurements/isp_sweep_2026-09-21.md
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from reporting import markdown_table
from testsignals import clipper_suite

from tagodsp.analysis.true_peak import peaks_db
from tagodsp.distortion.clipper import FL_THRESHOLD_DEFAULT, Clipper

CURVES = ("fl", "hard", "tanh")
THRESHOLDS = {"0.50": 0.5, "fl-default": FL_THRESHOLD_DEFAULT}
DRIVES_DB = (6.0, 12.0, 24.0)
OVERSAMPLE = (1, 4, 8)

# Delivery specs still care about a tenth of a dB, so that is the line between
# "measurable" and "matters".
RELEVANT_DB = 0.1


def run(sr: int, seconds: float) -> list[dict]:
    signals = clipper_suite(sr, seconds)
    rows = []
    for name, x in signals.items():
        for curve in CURVES:
            for t_label, threshold in THRESHOLDS.items():
                for drive in DRIVES_DB:
                    for os_factor in OVERSAMPLE:
                        y = Clipper(
                            curve=curve,
                            threshold=threshold,
                            oversample=os_factor,
                            drive_db=drive,
                        ).process(x)
                        # Nominal ceiling of the curve family: hardclip stops at
                        # its threshold, fl and tanh approach 1.0 asymptotically.
                        ceiling_db = 20 * np.log10(threshold) if curve == "hard" else 0.0
                        # Both peaks from one resampling pass. Asking for them
                        # separately resampled the same buffer twice, which was
                        # a third of this script's runtime.
                        sample_db, tp = peaks_db(y)
                        rows.append(
                            {
                                "signal": name,
                                "curve": curve,
                                "threshold": t_label,
                                "drive_db": drive,
                                "oversample": os_factor,
                                "peak_db": sample_db,
                                "true_peak_db": tp,
                                "overshoot_db": tp - sample_db,
                                "sample_over_ceiling_db": sample_db - ceiling_db,
                                "true_over_ceiling_db": tp - ceiling_db,
                            }
                        )
    return rows


def summarise(rows: list[dict], sr: int, seconds: float) -> str:
    worst = max(rows, key=lambda r: r["overshoot_db"])
    worst_ceiling = max(rows, key=lambda r: r["true_over_ceiling_db"])
    over_line = sum(1 for r in rows if r["overshoot_db"] > RELEVANT_DB)
    over_ceiling = sum(1 for r in rows if r["true_over_ceiling_db"] > RELEVANT_DB)

    lines = [
        "# ISP-Sweep: Inter-Sample-Peaks nach dem Clipping",
        "",
        f"Erzeugt von `examples/isp_sweep.py`, sr {sr} Hz, {seconds:g} s pro Signal, "
        f"{len(rows)} Kombinationen.",
        "True Peak gemessen mit 16x Oversampling nach BS.1770-4 "
        "(`analysis/true_peak.py`).",
        "",
        "Zwei verschiedene Zahlen, beide noetig:",
        "",
        "- **Ueberschuss** ist True Peak minus Sample-Peak. Er sagt, wie viel ein "
        "Sample-Peak-Meter dem Nutzer verschweigt.",
        "- **Ueber Decke** ist True Peak minus der nominalen Decke der Kurve "
        "(Threshold bei `hard`, 0 dBFS bei `fl` und `tanh`). Das ist die Zahl, die "
        "das Produktversprechen tragen muss, denn sie sagt, ob die Datei die "
        "Plattform-Pruefung besteht.",
        "",
        "## Antwort auf die Ausgangsfrage",
        "",
        f"- Groesster Ueberschuss ueber dem Sample-Peak: **{worst['overshoot_db']:+.2f} dB** "
        f"({worst['signal']}, Kurve {worst['curve']}, Threshold {worst['threshold']}, "
        f"Drive {worst['drive_db']:+.0f} dB, Oversampling {worst['oversample']}x)",
        f"- Groesste Ueberschreitung der Decke: "
        f"**{worst_ceiling['true_over_ceiling_db']:+.2f} dB** "
        f"({worst_ceiling['signal']}, Kurve {worst_ceiling['curve']}, "
        f"Threshold {worst_ceiling['threshold']}, Drive {worst_ceiling['drive_db']:+.0f} dB, "
        f"Oversampling {worst_ceiling['oversample']}x)",
        f"- Kombinationen mit Ueberschuss ueber {RELEVANT_DB} dB: "
        f"**{over_line} von {len(rows)}**",
        f"- Kombinationen mit Deckenueberschreitung ueber {RELEVANT_DB} dB: "
        f"**{over_ceiling} von {len(rows)}**",
        "",
        "## Je Kurve und Oversampling",
        "",
    ]

    grouped = []
    for curve in CURVES:
        for os_factor in OVERSAMPLE:
            sel = [r for r in rows if r["curve"] == curve and r["oversample"] == os_factor]
            grouped.append(
                {
                    "curve": curve,
                    "oversample": f"{os_factor}x",
                    "mean_db": float(np.mean([r["overshoot_db"] for r in sel])),
                    "max_db": float(np.max([r["overshoot_db"] for r in sel])),
                    "sample_ceil_max": float(np.max([r["sample_over_ceiling_db"] for r in sel])),
                    "true_ceil_max": float(np.max([r["true_over_ceiling_db"] for r in sel])),
                }
            )
    lines.append(
        markdown_table(
            grouped,
            [
                ("curve", "Kurve"),
                ("oversample", "Oversampling"),
                ("mean_db", "Ueberschuss Mittel"),
                ("max_db", "Ueberschuss Max"),
                ("sample_ceil_max", "Sample ueber Decke Max"),
                ("true_ceil_max", "True ueber Decke Max"),
            ],
        )
    )

    lines += ["", "## Worst Case je Signal, gemessen an der Deckenueberschreitung", ""]
    per_signal = []
    for name in dict.fromkeys(r["signal"] for r in rows):
        sel = [r for r in rows if r["signal"] == name]
        w = max(sel, key=lambda r: r["true_over_ceiling_db"])
        best_os = min(
            (r for r in sel if r["oversample"] == max(OVERSAMPLE)),
            key=lambda r: r["true_over_ceiling_db"],
        )
        per_signal.append(
            {
                "signal": name,
                "curve": w["curve"],
                "drive": f"{w['drive_db']:+.0f}",
                "oversample": f"{w['oversample']}x",
                "true_over_ceiling_db": w["true_over_ceiling_db"],
                "best_case_db": best_os["true_over_ceiling_db"],
            }
        )
    lines.append(
        markdown_table(
            per_signal,
            [
                ("signal", "Signal"),
                ("curve", "Kurve"),
                ("drive", "Drive"),
                ("oversample", "Oversampling"),
                ("true_over_ceiling_db", "True ueber Decke"),
                ("best_case_db", f"Bestfall bei {max(OVERSAMPLE)}x"),
            ],
        )
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sr", type=int, default=44100, help="sample rate in Hz")
    parser.add_argument("--seconds", type=float, default=1.0, help="length per test signal")
    parser.add_argument("--out", type=Path, default=None, help="write the report to this file")
    args = parser.parse_args()

    rows = run(args.sr, args.seconds)
    report = summarise(rows, args.sr, args.seconds)
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
