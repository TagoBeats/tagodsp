"""Does the whole chain hold the ceiling it promises?

Closes the loop on the first measurement of the prototype phase. That one showed
the true peak landing up to 5.8 dB above the ceiling after clipping, and that
oversampling does not fix it. This runs the same parameter space again, once
without and once with the true-peak limiter at the end, and reports how many
combinations still exceed the target.

Target is -1 dBTP, which is what a producer sets by hand today with a separate
limiter behind the clipper.

Usage:
    uv run python examples/chain_ceiling_check.py
    uv run python examples/chain_ceiling_check.py --out docs/measurements/chain_2026-09-21.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from reporting import markdown_table
from testsignals import clipper_suite

from tagodsp.analysis.true_peak import true_peak_db
from tagodsp.distortion.adaa import ADAAClipper
from tagodsp.distortion.clipper import FL_THRESHOLD_DEFAULT, Clipper
from tagodsp.dynamics.true_peak_limiter import TruePeakLimiter

CURVES = ("fl", "hard", "tanh")
THRESHOLDS = {"0.50": 0.5, "fl-default": FL_THRESHOLD_DEFAULT}
DRIVES_DB = (6.0, 12.0, 24.0)
CEILING_DB = -1.0
SR = 44100
# Verify with a finer meter than the limiter detects with, so the check cannot
# pass by sharing the blind spot of the instrument under test.
VERIFY_OVERSAMPLE = 32

# One instance, so the report quotes the margin the run actually used.
_LIMITER = TruePeakLimiter(sr=SR, ceiling_db=CEILING_DB)


# (label, builder) for the clipper stages under test
STAGES = (
    ("plain 1x", lambda c, t, d: Clipper(curve=c, threshold=t, oversample=1, drive_db=d)),
    ("plain 8x", lambda c, t, d: Clipper(curve=c, threshold=t, oversample=8, drive_db=d)),
    ("ADAA 4x", lambda c, t, d: ADAAClipper(curve=c, threshold=t, oversample=4, drive_db=d)),
)


def run(sr: int, seconds: float) -> list[dict]:
    signals = clipper_suite(sr, seconds)
    limiter = TruePeakLimiter(sr=sr, ceiling_db=CEILING_DB)
    rows = []
    for name, x in signals.items():
        for curve in CURVES:
            for t_label, threshold in THRESHOLDS.items():
                for drive in DRIVES_DB:
                    for stage, build in STAGES:
                        y = build(curve, threshold, drive).process(x)
                        rows.append(
                            {
                                "signal": name,
                                "curve": curve,
                                "threshold": t_label,
                                "drive_db": drive,
                                "stage": stage,
                                "bare_db": true_peak_db(y, oversample=VERIFY_OVERSAMPLE),
                                "limited_db": true_peak_db(
                                    limiter.process(y), oversample=VERIFY_OVERSAMPLE
                                ),
                            }
                        )
    return rows


def summarise(rows: list[dict], sr: int, seconds: float) -> str:
    def over(values: list[float]) -> int:
        return sum(1 for v in values if v > CEILING_DB)

    lines = [
        "# Haelt die Kette ihre Decke?",
        "",
        f"Erzeugt von `examples/chain_ceiling_check.py`, sr {sr} Hz, {seconds:g} s pro Signal, "
        f"{len(rows)} Kombinationen. Ziel ist {CEILING_DB:+.0f} dBTP, gegengeprueft mit "
        f"{VERIFY_OVERSAMPLE}x nach BS.1770-4 und damit feiner als der Detektor des Limiters "
        "selbst, sonst teilte die Pruefung dessen blinden Fleck.",
        "",
        "Die erste Messung der Prototyp-Phase hatte gezeigt, dass der True Peak nach dem "
        "Clipping bis zu 5,8 dB ueber der Decke liegt und Oversampling daran nichts aendert. "
        "Hier dieselbe Frage einmal ohne und einmal mit der Limiter-Stufe am Ende.",
        "",
        "## Ueberschreitungen der Decke",
        "",
    ]
    per_stage = []
    for stage, _ in STAGES:
        sel = [r for r in rows if r["stage"] == stage]
        bare = [r["bare_db"] for r in sel]
        lim = [r["limited_db"] for r in sel]
        per_stage.append(
            {
                "stage": stage,
                "bare": f"**{over(bare)} von {len(sel)}**",
                "limited": f"**{over(lim)} von {len(sel)}**",
                "worst_bare": f"{max(bare):+.2f} dBTP",
                "worst_limited": f"{max(lim):+.2f} dBTP",
            }
        )
    lines.append(
        markdown_table(
            per_stage,
            [
                ("stage", "Clipper-Stufe"),
                ("bare", "ohne Limiter"),
                ("limited", "mit Limiter"),
                ("worst_bare", "schlimmster Fall ohne"),
                ("worst_limited", "mit"),
            ],
        )
    )

    worst = max(rows, key=lambda r: r["limited_db"])
    lines += [
        "",
        "## Der schlimmste Fall mit Limiter",
        "",
        f"{worst['limited_db']:+.3f} dBTP bei {worst['signal']}, Kurve {worst['curve']}, "
        f"Threshold {worst['threshold']}, Drive {worst['drive_db']:+.0f} dB, "
        f"Stufe {worst['stage']}.",
        "",
        f"Der Plan hatte 0,1 dB Schlupf ueber der Decke erlaubt, also {CEILING_DB + 0.1:+.1f} "
        "dBTP. Gefordert wird hier die schaerfere Fassung: kein Wert ueber der gesetzten Zahl. "
        f"Moeglich macht das der Sicherheitsabstand von {_LIMITER.margin_db:.2f} dB, der dem "
        f"Detektor mit {_LIMITER.oversample}x folgt, siehe "
        "`docs/concepts/true_peak_limiter.md`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sr", type=int, default=SR)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = summarise(run(args.sr, args.seconds), args.sr, args.seconds)
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
