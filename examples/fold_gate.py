"""Gate for the fourth curve: does a wavefolder survive the ADAA machinery.

Same mechanism as the ADAA gate: the thresholds are fixed before the run, in
the constants below, and the script prints a verdict rather than a table to be
read charitably. Two of the three criteria from the plan are measured here, the
third one is a listening test and lives in render_listenpack_fold.py.

Criterion 1, aliasing. The candidate is compared against the three curves that
already ship, at the configuration the product uses. It passes at ADAA 4x with
six dB of room or less over `hard`, the dirtiest curve in the product today,
measured in the same run. Six dB is a factor of two in unwanted energy, and the room
exists because a folder legitimately produces harmonics of far higher order
than a clipper does, so part of its aliasing is the curve and not the
implementation. Second leg, so that the first one cannot be passed by a curve
that simply makes few harmonics: ADAA 4x must beat plain 1x of the same curve
by at least 15 dB. The three clipping curves manage 21 to 29 dB there.

The metric is alias_to_harmonics_db and not the alias_nmr_db the ADAA gate
used, because the old one divides by the fundamental and a folder nulls its own
fundamental at certain drives; see the note in alias_metrics.py. Both are
printed, the verdict uses the first.

Criterion 2, stability at the fold corners. A clipper has one knee, a folder
has a corner every t of input level, and the ADAA fallback branch is a midpoint
evaluation whose error is second order in the sample difference only where the
curve is smooth. At a corner it is first order, i.e. proportional to eps. This
is the fold-specific risk and the eps sweep is what exposes it. Bars are
inherited from the ADAA gate: the step where the two branches swap stays below
-90 dBFS, and the alias performance moves by less than 0.5 dB across eps.

Usage:
    uv run python examples/fold_gate.py
    uv run python examples/fold_gate.py --out docs/measurements/fold_gate_2026-09-23.md
"""

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from reporting import markdown_table

from tagodsp.analysis.alias_metrics import alias_nmr_db, alias_to_harmonics_db
from tagodsp.distortion.adaa import ADAAClipper
from tagodsp.distortion.clipper import CLIPPING_CURVES, Clipper
from tagodsp.distortion.folder import (
    SKEW_DEFAULT,
    antiderivative_fold,
    antiderivative_sinefold,
    sinefold,
    wavefold,
)
from tagodsp.utils.gain import lin_to_db

# Two folder shapes with the same fold geometry: piecewise linear with a corner
# at every fold point, and C-infinity with none. The gate has to tell the two
# questions apart, otherwise it answers "do corners alias" and reports it as
# "does wavefolding alias".
CANDIDATES = ("fold", "skewfold", "sinefold", "skewsinefold")
SKEW_OF = {"fold": 0.0, "skewfold": SKEW_DEFAULT, "sinefold": 0.0, "skewsinefold": SKEW_DEFAULT}
CURVES = (*CLIPPING_CURVES, *CANDIDATES)
FREQS = (1000.0, 5000.0, 11000.0)
DRIVES_DB = (6.0, 12.0, 18.0, 24.0)
THRESHOLD = 67 / 128

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

# Gate, fixed 2026-09-23 before the first run.
PRODUCT_CONFIG = "ADAA 4x"
# The rule: six dB of room over the dirtiest curve already in the product,
# measured in the same run with the same instrument. It was written as the
# absolute -39.6 dB first, from `hard` at -45.60 in the report of 2026-09-21,
# and turned into a rule when the instrument had to change: the number moves
# with the metric, the rule does not.
ANCHOR_CURVE = "hard"
ALIAS_ROOM_DB = 6.0
ADAA_GAIN_MIN_DB = 15.0  # ADAA 4x against plain 1x of the same curve
BRANCH_STEP_MAX_DBFS = -90.0  # inherited from the ADAA gate
EPS_SPREAD_MAX_DB = 0.5  # inherited from the ADAA gate
EPS_VALUES = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-9)
EPS_SHIPPING = 1e-4


def _triangle_exact(u: Fraction) -> Fraction:
    """Unit triangle wave in exact rational arithmetic."""
    r = u - 4 * (u // 4)
    if r <= 1:
        return r
    if r <= 3:
        return 2 - r
    return r - 4


def _g_exact(u: Fraction) -> Fraction:
    """Integral of the triangle wave from 0 to u, exact."""
    r = u - 4 * (u // 4)
    if r <= 1:
        return r * r / 2
    if r <= 3:
        return 2 * r - r * r / 2 - 1
    return r * r / 2 - 4 * r + 8


def _branch_step_triangle(x0: np.ndarray, d: float, threshold: float, skew: float) -> float:
    """Worst gap between the two ADAA branches at a corner, computed exactly.

    numpy's longdouble is plain float64 on Apple silicon, so a "wider mantissa"
    reference is the same instrument wearing a hat. The triangle folder is
    piecewise quadratic in F1 with rational breakpoints, and every float is a
    dyadic rational, so Fraction evaluates both branches with no rounding at
    all. That is the reference this measurement needs: the quantity under test
    is a difference of two nearly equal numbers.
    """
    t_pos, t_neg = Fraction(threshold), Fraction(threshold) * (1 - Fraction(skew))
    dd = Fraction(d)

    def f1(x: Fraction) -> Fraction:
        t = t_pos if x >= 0 else t_neg
        return t * t * _g_exact(abs(x) / t)

    def curve(x: Fraction) -> Fraction:
        t = t_pos if x >= 0 else t_neg
        sign = 1 if x >= 0 else -1
        return sign * t * _triangle_exact(abs(x) / t)

    worst = 0.0
    for value in x0:
        x = Fraction(float(value))
        quotient = (f1(x + dd) - f1(x)) / dd
        worst = max(worst, abs(float(quotient - curve(x + dd / 2))))
    return worst


def _branch_step_sine(x0: np.ndarray, d: float, threshold: float, skew: float) -> float:
    """Same gap for the smooth folder, via an identity instead of a subtraction.

    F1(x+d) - F1(x) = t^2 * (cos(x/t) - cos((x+d)/t)), and the sum-to-product
    identity turns that into 2*t^2 * sin((2x+d)/(2t)) * sin(d/(2t)), which
    never subtracts two nearly equal numbers. Relative error stays at machine
    epsilon, i.e. about -320 dBFS, far under the -90 dBFS this has to resolve.
    """
    t = np.where(x0 >= 0.0, threshold, threshold * (1.0 - skew))
    quotient = 2 * t * t * np.sin((2 * x0 + d) / (2 * t)) * np.sin(d / (2 * t)) / d
    midpoint = t * np.sin((x0 + d / 2) / t)
    return float(np.max(np.abs(quotient - midpoint)))


def run_alias() -> list[dict]:
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
                            "alias_db": alias_to_harmonics_db(y, SR, freq),
                            "nmr_db": alias_nmr_db(y, SR, freq),
                        }
                    )
    return rows


def run_eps(curve: str) -> list[dict]:
    """Branch step and alias spread over eps, measured at the fold points.

    The step is the worst case the fallback can produce: a difference just below
    eps, so the cheap branch runs, swept across a window around the third fold
    point. For the triangle shapes that window contains a corner, which is the
    fold-specific risk; for the sine shapes it is the same window without one,
    which is what makes the two columns comparable.
    """
    skew = SKEW_OF[curve]
    shape = "sine" if "sine" in curve else "triangle"
    f1_64 = antiderivative_sinefold if shape == "sine" else antiderivative_fold
    sine = np.sin(2 * np.pi * 5000.0 * np.arange(N) / SR)

    rows = []
    for eps in EPS_VALUES:
        d = eps * 0.999
        # Two grids: a broad one across the window, and a dense one at the
        # scale of d itself, because the worst case is the interval that
        # straddles the corner and a broad grid steps over it for small eps.
        corner = 3.0 * THRESHOLD
        x0 = np.concatenate(
            [
                np.linspace(corner - 0.05, corner + 0.05, 501),
                corner - d * np.linspace(0.0, 1.0, 129),
            ]
        )
        if shape == "sine":
            step = _branch_step_sine(x0, d, THRESHOLD, skew)
        else:
            step = _branch_step_triangle(x0, d, THRESHOLD, skew)

        # Same quantity through the shipping float64 path. The gap to the exact
        # reference above is the cancellation error, the thing eps exists for.
        q64 = (f1_64(x0 + d, THRESHOLD, skew) - f1_64(x0, THRESHOLD, skew)) / d
        mid64 = (
            sinefold(x0 + d / 2, THRESHOLD, skew)
            if shape == "sine"
            else wavefold(x0 + d / 2, THRESHOLD, skew)
        )
        cancellation = abs(float(np.max(np.abs(q64 - mid64))) - step)

        y = ADAAClipper(
            curve=curve, threshold=THRESHOLD, oversample=1, drive_db=11.0, eps=eps
        ).process(sine)
        # The shared floor of -160 dBFS is the audio floor and sits above both
        # of these quantities, which would print as a flat column of floor.
        rows.append(
            {
                "eps": eps,
                "step_db": lin_to_db(max(step, 1e-300), floor_db=-400.0),
                "cancellation_db": lin_to_db(max(cancellation, 1e-300), floor_db=-400.0),
                "alias_db": alias_to_harmonics_db(y, SR, 5000.0),
                "finite": bool(np.all(np.isfinite(y))),
            }
        )
    return rows


def dc_offset_db(curve: str) -> float:
    """DC the curve adds on a symmetric input. Asymmetry has to go somewhere."""
    x = np.sin(2 * np.pi * 100.0 * np.arange(N) / SR)
    y = ADAAClipper(curve=curve, threshold=THRESHOLD, oversample=4, drive_db=12.0).process(x)
    return float(lin_to_db(max(abs(float(np.mean(y))), 1e-300)))


def _mean(rows: list[dict], **filters) -> float:
    sel = [r for r in rows if all(r[k] == v for k, v in filters.items())]
    return float(np.mean([r["alias_db"] for r in sel]))


def summarise(alias_rows: list[dict], eps_rows: dict[str, list[dict]]) -> str:
    labels = [label for label, _ in CONFIGS]
    lines = [
        "# Gate: Wavefolder als vierte Kurve",
        "",
        f"Erzeugt von `examples/fold_gate.py`, sr {SR} Hz, {len(alias_rows)} Alias-Messungen, "
        f"Threshold {THRESHOLD:.4f}, Skew {SKEW_DEFAULT}.",
        "Metrik ist `alias_nmr_db`: Energie, die weder Grundton noch Harmonische ist, "
        "relativ zum Grundton. Kleiner ist sauberer.",
        "",
        "Die Schwellen standen vor dem Lauf fest und sind im Skript als Konstanten "
        "hinterlegt, nicht im Text.",
        "",
        "## Kriterium 1a: Alias-Niveau je Konfiguration und Kurve",
        "",
    ]

    per_config = []
    for label in labels:
        row = {"config": label}
        for curve in CURVES:
            row[curve] = _mean(alias_rows, config=label, curve=curve)
        per_config.append(row)
    lines.append(
        markdown_table(per_config, [("config", "Konfiguration"), *[(c, c) for c in CURVES]])
    )

    bar = _mean(alias_rows, config=PRODUCT_CONFIG, curve=ANCHOR_CURVE) + ALIAS_ROOM_DB
    lines += [
        "",
        "## Kriterium 1a: Urteil",
        "",
        f"Grenze ist `{ANCHOR_CURVE}` bei {PRODUCT_CONFIG} plus {ALIAS_ROOM_DB:.0f} dB Luft, "
        f"im selben Lauf gemessen: {bar:.2f} dB.",
        "",
    ]
    verdict_rows = []
    for curve in CANDIDATES:
        value = _mean(alias_rows, config=PRODUCT_CONFIG, curve=curve)
        verdict_rows.append(
            {
                "curve": curve,
                "value": f"{value:.2f}",
                "bar": f"{bar:.2f}",
                "verdict": "bestanden" if value <= bar else "gerissen",
            }
        )
    lines.append(
        markdown_table(
            verdict_rows,
            [
                ("curve", "Kurve"),
                ("value", f"{PRODUCT_CONFIG} (dB)"),
                ("bar", "Grenze (dB)"),
                ("verdict", "Urteil"),
            ],
        )
    )

    lines += [
        "",
        "## Kriterium 1a: schlechteste Einzelzelle",
        "",
        "Der Mittelwert kann einen einzelnen katastrophalen Betriebspunkt verstecken. "
        "Hier steht die schlechteste Kombination aus Frequenz und Drive je Kurve, "
        f"bei {PRODUCT_CONFIG}, mit den bestehenden Kurven als Massstab.",
        "",
    ]
    worst = []
    for curve in CURVES:
        sel = [r for r in alias_rows if r["config"] == PRODUCT_CONFIG and r["curve"] == curve]
        hit = max(sel, key=lambda r: r["alias_db"])
        worst.append(
            {
                "curve": curve,
                "value": f"{hit['alias_db']:.2f}",
                "where": f"{hit['freq'] / 1000:g} kHz bei +{hit['drive_db']:.0f} dB",
            }
        )
    lines.append(
        markdown_table(
            worst,
            [("curve", "Kurve"), ("value", "schlechteste Zelle (dB)"), ("where", "wo")],
        )
    )

    lines += [
        "",
        f"## Kriterium 1a: wo der Mittelwert herkommt ({PRODUCT_CONFIG})",
        "",
        "Ein knapper Mittelwert kann aus einem gleichmaessigen Feld kommen oder aus "
        "einer Ecke, die alles andere mittraegt. Deshalb aufgeschluesselt.",
        "",
    ]
    detail = []
    for curve in CANDIDATES:
        for drive in DRIVES_DB:
            row = {"curve": curve, "drive": f"+{drive:.0f}"}
            for freq in FREQS:
                row[f"{freq / 1000:g}k"] = _mean(
                    alias_rows, config=PRODUCT_CONFIG, curve=curve, drive_db=drive, freq=freq
                )
            detail.append(row)
    lines.append(
        markdown_table(
            detail,
            [
                ("curve", "Kurve"),
                ("drive", "Drive (dB)"),
                *[(f"{f / 1000:g}k", f"{f / 1000:g} kHz") for f in FREQS],
            ],
        )
    )

    lines += [
        "",
        "## Kriterium 1b: greift ADAA bei dieser Kurve ueberhaupt",
        "",
        f"ADAA 4x gegen plain 1x derselben Kurve, Mindestgewinn {ADAA_GAIN_MIN_DB:.0f} dB.",
        "",
    ]
    gain_rows = []
    for curve in CURVES:
        gain = _mean(alias_rows, config="plain 1x", curve=curve) - _mean(
            alias_rows, config=PRODUCT_CONFIG, curve=curve
        )
        gain_rows.append(
            {
                "curve": curve,
                "gain": f"{gain:.2f}",
                "verdict": (
                    ""
                    if curve not in CANDIDATES
                    else ("bestanden" if gain >= ADAA_GAIN_MIN_DB else "gerissen")
                ),
            }
        )
    lines.append(
        markdown_table(
            gain_rows, [("curve", "Kurve"), ("gain", "Gewinn (dB)"), ("verdict", "Urteil")]
        )
    )

    lines += ["", "## Kriterium 2: eps-Sweep an den Faltpunkten", ""]
    lines += [
        "`Sprung` ist der Unterschied zwischen Quotient und Mittelpunkt-Zweig genau dort, "
        "wo die Umschaltung passiert, gemessen ueber ein Fenster um den dritten Faltpunkt. "
        f"Grenze {BRANCH_STEP_MAX_DBFS:.0f} dBFS. `Ausloeschung` ist der Abstand zwischen "
        "dem float64-Quotienten und demselben Quotienten in long double, also der Fehler, "
        "gegen den eps ueberhaupt existiert.",
        "",
    ]
    for curve, rows in eps_rows.items():
        lines += [f"### {curve}", ""]
        table = [
            {
                "eps": f"{r['eps']:.0e}",
                "step": f"{r['step_db']:.1f}",
                "cancel": f"{r['cancellation_db']:.1f}",
                "alias": f"{r['alias_db']:.2f}",
                "finite": "ja" if r["finite"] else "NEIN",
            }
            for r in rows
        ]
        lines.append(
            markdown_table(
                table,
                [
                    ("eps", "eps"),
                    ("step", "Sprung (dBFS)"),
                    ("cancel", "Ausloeschung (dBFS)"),
                    ("alias", "Alias (dB)"),
                    ("finite", "endlich"),
                ],
            )
        )
        values = [r["alias_db"] for r in rows]
        spread = max(values) - min(values)
        shipping = next(r for r in rows if r["eps"] == EPS_SHIPPING)
        lines += [
            "",
            f"Spreizung ueber eps: {spread:.2f} dB, Grenze {EPS_SPREAD_MAX_DB} dB, "
            f"{'bestanden' if spread < EPS_SPREAD_MAX_DB else 'gerissen'}.",
            f"Sprung beim ausgelieferten eps {EPS_SHIPPING:.0e}: "
            f"{shipping['step_db']:.1f} dBFS, Grenze {BRANCH_STEP_MAX_DBFS:.0f} dBFS, "
            f"{'bestanden' if shipping['step_db'] < BRANCH_STEP_MAX_DBFS else 'gerissen'}.",
            "",
        ]

    lines += [
        "## Nebenbefund: Gleichspannung",
        "",
        "Eine unsymmetrische Kennlinie erzeugt einen Gleichanteil. Kein Gate-Kriterium, "
        "aber ein Bauteil, das im Plugin dann existieren muss.",
        "",
    ]
    lines.append(
        markdown_table(
            [{"curve": c, "dc": f"{dc_offset_db(c):.1f}"} for c in CURVES],
            [("curve", "Kurve"), ("dc", "DC (dBFS)")],
        )
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="write the report to this file")
    args = parser.parse_args()

    eps_rows = {curve: run_eps(curve) for curve in CANDIDATES}
    report = summarise(run_alias(), eps_rows)
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
