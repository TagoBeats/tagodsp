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
from tagodsp.utils.gain import db_to_lin, lin_to_db, rms

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


# Drive per case. Hot where a clipper actually gets used, moderate on a finished
# master, because +11 dB into an already maximised mix is mush and nobody can
# judge a subtle difference through it. Differences between the configurations
# grow with drive, so the hot cases are the sensitive ones: indistinguishable
# there means indistinguishable everywhere below.
DRIVE_PER_CASE = {"master": 4.0}


def drive_for(name: str) -> float:
    return DRIVE_PER_CASE.get(name, DRIVE_DB)


def _per_channel(make, x: np.ndarray) -> np.ndarray:
    """Run a fresh processor over each channel. Time is the first axis."""
    if x.ndim == 1:
        return make().process(x)
    return np.stack([make().process(x[:, ch]) for ch in range(x.shape[1])], axis=1)


def plain(oversample: int, x: np.ndarray, drive_db: float = DRIVE_DB) -> np.ndarray:
    return _per_channel(
        lambda: Clipper(
            curve="fl", threshold=THRESHOLD, oversample=oversample, drive_db=drive_db
        ),
        x,
    )


def adaa(oversample: int, x: np.ndarray, drive_db: float = DRIVE_DB) -> np.ndarray:
    return _per_channel(
        lambda: ADAAClipper(
            curve="fl", threshold=THRESHOLD, oversample=oversample, drive_db=drive_db
        ),
        x,
    )


# (item suffix, A oversampling, A label, B oversampling, B label)
COMPARISONS = (
    ("adaa4x_vs_plain8x", 8, "plain_8x_heute", 4, "adaa_4x"),
    ("adaa2x_vs_plain8x", 8, "plain_8x_heute", 2, "adaa_2x"),
    ("adaa1x_vs_plain8x", 8, "plain_8x_heute", 1, "adaa_1x"),
    ("adaa1x_vs_plain1x", 1, "plain_1x_wie_fl", 1, "adaa_1x"),
)

# Real stems: (case name, fragment of the file name). One beat so the four cases
# stay musically coherent, and the fragments survive FL's project-name prefix.
STEM_CASES = (
    ("808", "PV3 808 4"),
    ("drum_bus", "Drum Bus"),
    ("hats", "hi hat"),
    ("master", "Master"),
)

# Which comparison to render for which case, so the page stays clickable.
# Everything gets the decision; the rolloff and the alias grit are checked where
# they actually show up.
FOCUS = {
    "adaa4x_vs_plain8x": None,
    "adaa2x_vs_plain8x": ("hats", "master"),
    "adaa1x_vs_plain8x": ("hats", "master"),
    "adaa1x_vs_plain1x": ("hats",),
}


def loudest_window(x: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """The most energetic stretch, so the excerpt is not an intro or a gap."""
    n = int(seconds * sr)
    mono = x.mean(axis=1) if x.ndim > 1 else x
    if len(mono) <= n:
        return x
    hop = max(int(0.25 * sr), 1)
    energy = [(float(np.sum(mono[s : s + n] ** 2)), s) for s in range(0, len(mono) - n, hop)]
    start = max(energy)[1]
    return x[start : start + n]


def load_stems(folder: Path, seconds: float) -> tuple[dict[str, np.ndarray], int]:
    """Load the named stem cases, trimmed to their loudest window.

    Each stem is peak normalised to -0.5 dBFS before the drive is applied,
    otherwise a bus sitting at -30 dBFS RMS would never reach the knee and the
    comparison would be between two untouched files.
    """
    import soundfile as sf

    signals: dict[str, np.ndarray] = {}
    sr_seen: set[int] = set()
    for case, fragment in STEM_CASES:
        matches = [p for p in sorted(folder.glob("*.wav")) if fragment.lower() in p.name.lower()]
        if not matches:
            print(f"  {case}: no file containing {fragment!r}, skipped")
            continue
        data, sr = sf.read(matches[0])
        sr_seen.add(sr)
        clip = loudest_window(data, sr, seconds)
        peak = float(np.max(np.abs(clip)))
        clip = clip / peak * 10 ** (-0.5 / 20) if peak > 0 else clip
        signals[case] = clip
        print(f"  {case}: {matches[0].name[:60]} at {sr} Hz")
    if len(sr_seen) > 1:
        raise SystemExit(f"stems have mixed sample rates: {sorted(sr_seen)}")
    return signals, sr_seen.pop() if sr_seen else SR


DEFAULT_STEMS = Path.home() / "Music/Beats 2026/okayes/okayes Stems"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stems-dir",
        type=Path,
        default=DEFAULT_STEMS,
        help="folder of real stems; falls back to synthetic material if missing",
    )
    parser.add_argument(
        "--seconds", type=float, default=10.0, help="excerpt length per stem"
    )
    args = parser.parse_args()

    if args.stems_dir.is_dir():
        print(f"Stems aus {args.stems_dir}")
        signals, sr = load_stems(args.stems_dir, args.seconds)
        tone_checks: tuple[tuple[str, float], ...] = ()
    else:
        print(f"{args.stems_dir} nicht gefunden, synthetisches Material")
        sr = SR
        signals = {
            "drums": drums(),
            "hats": synth_hats(),
            "bright_noise": bright_noise(),
            "sine_5k": sine(5000.0),
            "sine_11k": sine(11000.0),
        }
        tone_checks = (("sine_5k", 5000.0), ("sine_11k", 11000.0))

    if not signals:
        raise SystemExit("no material to render")

    print()
    print("Wie hart greift der Clipper an? Reduktion gegen das reine Anheben um den Drive:")
    for name, x in signals.items():
        d = drive_for(name)
        driven_only = rms(x) * db_to_lin(d)
        reduction = rms(plain(8, x, d)) / max(driven_only, 1e-12)
        print(f"  {name:<12} Drive +{d:>4.1f} dB  ->  {lin_to_db(reduction):+6.2f} dB geclippt")

    pack = ListenPack("adaa")
    print()
    print(f"{'Vergleich':<22} {'Signal':<12} {'Pegelabgleich':>14}")
    for suffix, plain_os, label_a, adaa_os, label_b in COMPARISONS:
        only = FOCUS.get(suffix)
        for name, x in signals.items():
            if only is not None and name not in only:
                continue
            d = drive_for(name)
            a = plain(plain_os, x, d)
            b, gain_db = match_rms(a, adaa(adaa_os, x, d))
            pack.add_pair(f"{name}_{suffix}", a, b, sr, label_a=label_a, label_b=label_b)
            print(f"{suffix:<22} {name:<12} {gain_db:>+13.2f} dB")

    for name, x in signals.items():
        mono = x.mean(axis=1) if x.ndim > 1 else x
        d = drive_for(name)
        pack.spectrum_plot(
            f"spectrum_{name}",
            {
                "plain 1x (wie FL)": plain(1, mono, d),
                "plain 8x (heute)": plain(8, mono, d),
                "ADAA 1x": adaa(1, mono, d),
                "ADAA 4x (Kandidat)": adaa(4, mono, d),
            },
            sr,
            title=f"{name}, FL-Kurve bei +{d:.0f} dB Drive",
        )

    if tone_checks:
        print()
        print("Alias-Metrik der Testtoene, kleiner ist sauberer:")
        for name, freq in tone_checks:
            x = signals[name]
            values = {
                "plain 1x": alias_nmr_db(plain(1, x), sr, freq),
                "plain 8x": alias_nmr_db(plain(8, x), sr, freq),
                "ADAA 1x": alias_nmr_db(adaa(1, x), sr, freq),
                "ADAA 4x": alias_nmr_db(adaa(4, x), sr, freq),
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
