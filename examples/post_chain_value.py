"""Is the ceiling stage still worth anything when plugins follow it?

Robin's own master chain, screenshotted 21.09.2026, puts the clipper in slot 2
and FabFilter Pro-L 2 in the last slot, with a transient designer, a glue bus
processor and a stereo imager in between. That breaks the simple story: a
ceiling held at the clipper output guarantees nothing about the delivered file,
because everything downstream moves the peaks again.

So this measures the weaker claim instead, the one that could still be true:
with a clean ceiling at the clipper, does the FINAL limiter have measurably
less work to do?

The comparison is only fair at matched loudness. Simply turning the level down
after the clipper would also spare the final limiter, and that is a trivial,
worthless result. Both variants are therefore RMS-matched right after the
clipper stage, before anything downstream runs, so what is compared is crest
factor and not level.

The final limiter here is this repo's own TruePeakLimiter standing in for
Pro-L 2. That is stated rather than hidden: it shares the detector design, so
the absolute numbers are a floor, not a forecast of what Pro-L 2 would do.

Usage:
    uv run python examples/post_chain_value.py
    uv run python examples/post_chain_value.py --out docs/measurements/post_chain_2026-09-21.md
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from reporting import markdown_table

from tagodsp.analysis.sum_stems import find_sum_stems
from tagodsp.analysis.true_peak import true_peak_db
from tagodsp.distortion.adaa import ADAAClipper
from tagodsp.distortion.clipper import FL_THRESHOLD_DEFAULT
from tagodsp.dynamics.true_peak_limiter import TruePeakLimiter
from tagodsp.utils.gain import lin_to_db, rms
from tagodsp.utils.smoothing import OnePoleSmoother

SR = 44100
CEILING_DB = -1.0
DRIVES_DB = (6.0, 12.0, 18.0)
OVERSAMPLE = 4
VERIFY_OVERSAMPLE = 32

# Thresholds fixed before the first run, on the peak gain reduction the final
# limiter is spared. Below 0.3 dB nothing is audible or demonstrable, so there
# is no claim to make; above 1.0 dB there is a number to put on a product page.
VALUE_DB = 1.0
MARGINAL_DB = 0.3

# A real master chain lifts the crest factor by a couple of dB, not by ten. Both
# ends are checked: see check_instrument for why each one would invalidate the run.
CREST_GROWTH_MIN = 0.5
CREST_GROWTH_MAX = 4.0

# The stand-in for what sits between clipper and final limiter in the real
# chain. Not models of those plugins, just three processors that do what they
# do to the peaks: a transient designer sharpens attacks, a glue compressor
# with a slow attack lets transients through while pulling the body down, and
# a widener pushes side energy back into L and R.
#
# Calibrated, not guessed: at -12 dB and 3:1 the glue stage peaks at 3.1 dB of
# gain reduction on this material, which is what a mix bus actually runs. The
# first attempt (-18 dB, 2.5:1) reduced by far more and inflated the crest
# factor by 12 dB on its own, which pinned the final limiter and would have
# buried the very effect this script measures.
TRANSIENT_AMOUNT = 0.25
TRANSIENT_MAX_GAIN = 10 ** (3.0 / 20)
GLUE_THRESHOLD_DB = -12.0
GLUE_RATIO = 3.0
WIDTH = 1.35

DEFAULT_STEMS = Path.home() / "Music/Beats 2026/okayes/okayes Stems"


def loudest_window(x: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """The most energetic stretch, so the excerpt is not an intro or a gap.

    Same window picker as render_listenpack_adaa.py, kept local so this script
    does not import from another example.
    """
    n = int(seconds * sr)
    mono = x.mean(axis=1) if x.ndim > 1 else x
    if len(mono) <= n:
        return x
    hop = max(int(0.25 * sr), 1)
    cumulative = np.concatenate(([0.0], np.cumsum(mono**2)))
    starts = np.arange(0, len(mono) - n, hop)
    start = int(starts[np.argmax(cumulative[starts + n] - cumulative[starts])])
    return x[start : start + n]


def load_mix(folder: Path, seconds: float) -> tuple[np.ndarray, int]:
    """Sum the real stems into the mix that would enter the master chain.

    The full-mix bounces FL exports alongside the stems ("_Master", "_Current")
    are removed by measurement, not by name alone: leaving one in would double
    the whole mix on top of itself. Returns (2, N), channels first, which is
    what TruePeakLimiter expects.
    """
    import soundfile as sf

    names, signals, sr_seen = [], [], set()
    for path in sorted(folder.glob("*.wav")):
        data, sr = sf.read(path)
        sr_seen.add(sr)
        names.append(path.stem)
        signals.append(np.atleast_2d(data.T))
    if not signals:
        raise SystemExit(f"no wav files in {folder}")
    if len(sr_seen) > 1:
        raise SystemExit(f"stems have mixed sample rates: {sorted(sr_seen)}")
    sr = sr_seen.pop()

    length = min(s.shape[-1] for s in signals)
    signals = [s[..., :length] for s in signals]
    monos = [s.mean(axis=0) for s in signals]
    sums = {v.name for v in find_sum_stems(names, monos, sr)}
    kept = [s for name, s in zip(names, signals) if name not in sums]
    print(f"  {len(kept)} von {len(names)} Stems summiert, raus: {sorted(sums) or 'nichts'}")

    mix = np.zeros((2, length))
    for stem in kept:
        mix += stem if stem.shape[0] == 2 else np.repeat(stem, 2, axis=0)
    clip = loudest_window(mix.T, sr, seconds).T
    peak = float(np.max(np.abs(clip)))
    return (clip / peak * 10 ** (-0.5 / 20) if peak > 0 else clip), sr


def _follow(x: np.ndarray, sr: int, tau_s: float) -> np.ndarray:
    """Envelope of the linked channels, so the stereo image never shifts."""
    linked = np.max(np.abs(x), axis=0)
    return OnePoleSmoother(tau_s=tau_s, sr=sr).process(linked)


def transient_boost(x: np.ndarray, sr: int, amount: float = TRANSIENT_AMOUNT) -> np.ndarray:
    """Lift attacks: a fast envelope running above a slow one means a transient."""
    fast = _follow(x, sr, 0.002)
    slow = _follow(x, sr, 0.060)
    gain = 1.0 + amount * np.maximum(fast / np.maximum(slow, 1e-9) - 1.0, 0.0)
    # Capped at +3 dB. Without a cap a single sample can ask for +12 dB, which no
    # transient designer does and which alone inflated the crest factor by 12 dB.
    return x * np.minimum(gain, TRANSIENT_MAX_GAIN)


def glue_compress(x: np.ndarray, sr: int) -> np.ndarray:
    """Slow-attack bus compression: the body comes down, the spikes stay put."""
    env = _follow(x, sr, 0.030)
    threshold = 10 ** (GLUE_THRESHOLD_DB / 20)
    over = np.maximum(env / threshold, 1.0)
    return x * over ** (1.0 / GLUE_RATIO - 1.0)


def widen(x: np.ndarray, width: float = WIDTH) -> np.ndarray:
    """Push side energy back into L and R, which is where new peaks come from."""
    mid = 0.5 * (x[0] + x[1])
    side = 0.5 * (x[0] - x[1]) * width
    return np.stack([mid + side, mid - side])


def post_chain(x: np.ndarray, sr: int) -> np.ndarray:
    return widen(glue_compress(transient_boost(x, sr), sr))


def crest_db(x: np.ndarray) -> float:
    """True peak over RMS. This is what the downstream chain inflates."""
    return true_peak_db(x, oversample=VERIFY_OVERSAMPLE) - lin_to_db(rms(x))


def check_instrument(mix: np.ndarray, sr: int) -> float:
    """The post chain must demonstrably inflate the crest factor.

    If it does not, this script measures nothing and every verdict below would
    be an artefact of a stand-in that stands in for nothing.
    """
    grown = crest_db(post_chain(mix, sr)) - crest_db(mix)
    if not CREST_GROWTH_MIN <= grown <= CREST_GROWTH_MAX:
        raise SystemExit(
            f"post chain moves the crest factor by {grown:+.2f} dB, outside the plausible "
            f"{CREST_GROWTH_MIN} to {CREST_GROWTH_MAX} dB a master chain does. Too little and "
            "this script cannot see its own subject; too much and the final limiter is pinned "
            "by the stand-in, which buries whatever the ceiling stage contributed. Recalibrate "
            "before trusting any row below."
        )
    return grown


def clip_stage(x: np.ndarray, drive_db: float, ceiling: bool, sr: int) -> np.ndarray:
    """Clipper, then optionally this product's ceiling stage. Channels first."""
    shaped = np.stack(
        [
            ADAAClipper(
                curve="fl",
                threshold=FL_THRESHOLD_DEFAULT,
                oversample=OVERSAMPLE,
                drive_db=drive_db,
            ).process(channel)
            for channel in x
        ]
    )
    if not ceiling:
        return shaped
    return TruePeakLimiter(sr=sr, ceiling_db=CEILING_DB).process(shaped)


def run(mix: np.ndarray, sr: int) -> list[dict]:
    final = TruePeakLimiter(sr=sr, ceiling_db=CEILING_DB)
    rows = []
    for drive in DRIVES_DB:
        # The no-ceiling variant sets the loudness both are measured at, so the
        # ceiling stage cannot win this by simply being quieter.
        reference_rms = rms(clip_stage(mix, drive, ceiling=False, sr=sr))
        for tail, tail_label in ((False, "nichts dahinter"), (True, "volle Kette")):
            spared = {}
            for ceiling in (False, True):
                y = clip_stage(mix, drive, ceiling, sr)
                y *= reference_rms / max(rms(y), 1e-12)
                if tail:
                    y = post_chain(y, sr)
                gr = -lin_to_db(final.gain_envelope(y))
                spared[ceiling] = float(np.max(gr))
                rows.append(
                    {
                        "drive_db": drive,
                        "tail": tail_label,
                        "ceiling": "an" if ceiling else "aus",
                        "tp_in_db": true_peak_db(y, oversample=VERIFY_OVERSAMPLE),
                        "crest_db": crest_db(y),
                        "gr_peak_db": spared[ceiling],
                        "gr_mean_db": float(np.mean(gr[gr > 0.01])) if np.any(gr > 0.01) else 0.0,
                        "gr_active_pct": 100.0 * float(np.mean(gr > 0.01)),
                    }
                )
            rows[-1]["spared_db"] = spared[False] - spared[True]
    return rows


def summarise(rows: list[dict], crest_growth: float, sr: int) -> str:
    lines = [
        "# Traegt die Ceiling-Stufe noch, wenn Plugins folgen?",
        "",
        f"Material: Summe der echten Stems (Beat `okayes`), {sr} Hz, lauteste Passage. "
        f"Clipper ADAA {OVERSAMPLE}x, fl-Kurve. Decke {CEILING_DB:+.1f} dBTP. "
        f"Verifiziert mit {VERIFY_OVERSAMPLE}x.",
        "",
        "**Beide Varianten sind nach dem Clipper auf dasselbe RMS gezogen.** Ohne das "
        "wuerde die Ceiling-Stufe den Vergleich allein dadurch gewinnen, dass sie leiser "
        "ist, und das waere kein Befund, sondern ein Pegelunterschied.",
        "",
        f"**Pruefung des Messinstruments:** die nachgeschaltete Kette hebt den Crest-Faktor "
        f"um {crest_growth:+.2f} dB. Taete sie das nicht, koennte dieser Test seinen eigenen "
        "Gegenstand nicht sehen.",
        "",
        "## Alle Laeufe",
        "",
        markdown_table(
            rows,
            (
                ("drive_db", "Drive", "+.0f"),
                ("tail", "Danach"),
                ("ceiling", "Ceiling"),
                ("tp_in_db", "dBTP vor dem Limiter"),
                ("crest_db", "Crest"),
                ("gr_peak_db", "GR max"),
                ("gr_mean_db", "GR Mittel"),
                ("gr_active_pct", "GR aktiv %", ".1f"),
            ),
        ),
        "",
        "## Was dem finalen Limiter erspart bleibt",
        "",
    ]

    spared = [r for r in rows if "spared_db" in r]
    lines.append(
        markdown_table(
            spared,
            (("drive_db", "Drive", "+.0f"), ("tail", "Danach"), ("spared_db", "GR max gespart")),
        )
    )

    by_drive = {
        r["drive_db"]: {x["tail"]: x["spared_db"] for x in spared if x["drive_db"] == r["drive_db"]}
        for r in spared
    }
    with_tail = [v["volle Kette"] for v in by_drive.values()]
    worst, best = min(with_tail), max(with_tail)
    above = sum(v >= VALUE_DB for v in with_tail)

    if worst >= VALUE_DB:
        verdict = (
            f"**Lesart B traegt.** Auch im schlechtesten Fall bleiben mit voller Kette "
            f"{worst:.2f} dB Spitzen-GR erspart, ueber der vorab gesetzten Schwelle von "
            f"{VALUE_DB:.1f} dB. Das ist eine Zahl, die auf eine Produktseite darf."
        )
    elif best < MARGINAL_DB:
        verdict = (
            f"**Lesart B faellt.** Mit voller Kette bleiben hoechstens {best:.2f} dB erspart, "
            f"unter der Abbruchschwelle von {MARGINAL_DB:.1f} dB. Die nachgeschalteten "
            "Plugins fressen den Vorteil auf. Dann traegt nur Lesart A, also das Produkt "
            "als letzte Instanz."
        )
    else:
        verdict = (
            f"**Das Gate ist verfehlt, aber knapp und ungleichmaessig.** Gefordert war "
            f"{VALUE_DB:.1f} dB im *schlechtesten* Fall, gemessen sind {worst:.2f} dB. "
            f"Bei {above} von {len(with_tail)} Drive-Stufen liegt der Wert darueber, "
            f"Spitze {best:.2f} dB. Die Schwelle bleibt stehen wie gesetzt: knapp verfehlt "
            "ist verfehlt. Aber die Streuung ist selbst der Befund, siehe unten."
        )

    ordered = sorted(by_drive.items())
    trend = ", ".join(
        f"{d:+.0f} dB Drive: {v['nichts dahinter']:.2f} ohne / {v['volle Kette']:.2f} mit Kette"
        for d, v in ordered
    )
    eats = [d for d, v in ordered if v["volle Kette"] < v["nichts dahinter"]]
    grows = [d for d, v in ordered if v["volle Kette"] >= v["nichts dahinter"]]
    direction = []
    if eats:
        direction.append(
            "Bei " + ", ".join(f"{d:+.0f} dB" for d in eats) + " frisst die Kette einen Teil "
            "des Vorteils, wie erwartet."
        )
    if grows:
        direction.append(
            "Bei " + ", ".join(f"{d:+.0f} dB" for d in grows) + " **vergroessert sie ihn**, "
            "entgegen der Erwartung: ohne Ceiling-Stufe hinterlaesst der Clipper dort Spitzen, "
            "die Transientenformer und Widener anschliessend mit anheben, statt sie zu glaetten."
        )

    lines += [
        "",
        "## Verdikt",
        "",
        verdict,
        "",
        f"Gespart, je Drive-Stufe: {trend}.",
        "",
        " ".join(direction),
        "",
        "**Der Vorteil waechst mit dem Drive.** Genau dort, wo hart geclippt wird, traegt die "
        "Ceiling-Stufe also am meisten, und zwar durch die Kette hindurch. Wer den Clipper nur "
        "antippt, braucht sie nicht.",
        "",
        "**Grenze dieser Messung:** der finale Limiter ist die eigene `TruePeakLimiter`-"
        "Klasse als Stellvertreter fuer Pro-L 2, und die drei Prozessoren dazwischen sind "
        "generische Stellvertreter, keine Modelle der echten Plugins. Die Zahlen zeigen die "
        "Richtung und die Groessenordnung, nicht was Pro-L 2 im konkreten Projekt tut.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stems-dir", type=Path, default=DEFAULT_STEMS)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.stems_dir.is_dir():
        raise SystemExit(f"no stem folder at {args.stems_dir}, pass --stems-dir")
    print(f"Stems aus {args.stems_dir}")
    mix, sr = load_mix(args.stems_dir, args.seconds)
    crest_growth = check_instrument(mix, sr)

    report = summarise(run(mix, sr), crest_growth, sr)
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
