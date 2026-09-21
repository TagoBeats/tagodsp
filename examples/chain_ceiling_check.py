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
from pathlib import Path

import numpy as np

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


def build_signals(sr: int, seconds: float) -> dict[str, np.ndarray]:
    n = int(sr * seconds)
    t = np.arange(n) / sr
    rng = np.random.default_rng(2026)
    burst = np.sin(2 * np.pi * 60.0 * t) * np.exp(-t * 12.0)
    for hit in (0.25, 0.5, 0.75):
        start = int(hit * n)
        tail = np.arange(n - start) / sr
        burst[start:] += np.sin(2 * np.pi * 3000.0 * tail) * np.exp(-tail * 60.0)
    return {
        "sine_1k": np.sin(2 * np.pi * 1000.0 * t),
        "sine_5k": np.sin(2 * np.pi * 5000.0 * t),
        "sine_11k": np.sin(2 * np.pi * 11000.0 * t),
        "twotone_11k_12k": 0.5 * (np.sin(2 * np.pi * 11000 * t) + np.sin(2 * np.pi * 12000 * t)),
        "drum_ish": burst / np.max(np.abs(burst)),
        "noise": rng.standard_normal(n) * 0.25,
    }


# (label, builder) for the clipper stages under test
STAGES = (
    ("plain 1x", lambda c, t, d: Clipper(curve=c, threshold=t, oversample=1, drive_db=d)),
    ("plain 8x", lambda c, t, d: Clipper(curve=c, threshold=t, oversample=8, drive_db=d)),
    ("ADAA 4x", lambda c, t, d: ADAAClipper(curve=c, threshold=t, oversample=4, drive_db=d)),
)


def run(sr: int, seconds: float) -> list[dict]:
    signals = build_signals(sr, seconds)
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
        "| Clipper-Stufe | ohne Limiter | mit Limiter | schlimmster Fall ohne | mit |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for stage, _ in STAGES:
        sel = [r for r in rows if r["stage"] == stage]
        bare = [r["bare_db"] for r in sel]
        lim = [r["limited_db"] for r in sel]
        lines.append(
            f"| {stage} | **{over(bare)} von {len(sel)}** | **{over(lim)} von {len(sel)}** | "
            f"{max(bare):+.2f} dBTP | {max(lim):+.2f} dBTP |"
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
        "Moeglich macht das der Sicherheitsabstand von 0,05 dB, siehe "
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
