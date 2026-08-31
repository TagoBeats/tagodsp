"""Render A/B listening pairs for detected masking zones.

A conflict zone is a claim: "these two tracks cover each other here". The only
honest check is to hear the mix once untouched and once with a targeted dip in
exactly that zone. If nothing changes audibly, the zone was not one.

This script renders those pairs. It makes no judgement and recommends no
scoring method. Ears decide.

Zone selection deliberately covers the cases where the scoring methods
disagree, because a plain top-list of one method would not separate them.

Usage:
    python render_listenpack_masking.py /path/to/stems --bpm 135 \
        --exclude master,drum_bus --merge rim:rim1,rim2wav
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from listenpack import ListenPack

from tagodsp.analysis.masking import MaskingDetector, Track, Zone
from tagodsp.filters.biquad import Biquad, peaking
from tagodsp.spectral.stft import Stft

WINDOWS_PER_BAR = 8
PAD_SECONDS = 1.0
MIN_EXCERPT_S = 4.0
MAX_EXCERPT_S = 12.0
LOW_END_MAX_HZ = 200.0
SCORINGS = ("relative", "collision", "contention")


def load_stems(stems_dir: Path, exclude: set[str], merges: dict[str, list[str]]):
    """Return (sr, {name: (n, ch) float array}) with all tracks trimmed to equal length."""
    wavs = {w.stem: w for w in sorted(stems_dir.glob("*.wav"))}
    if not wavs:
        raise SystemExit(f"no wav files in {stems_dir}")

    def read(name: str):
        x, sr = sf.read(wavs[name], dtype="float64", always_2d=True)
        return sr, x

    tracks: dict[str, np.ndarray] = {}
    sr = None
    for target, sources in merges.items():
        parts = []
        for s in sources:
            if s not in wavs:
                raise SystemExit(f"merge source not found: {s}")
            rate, x = read(s)
            sr = sr or rate
            if rate != sr:
                raise SystemExit(f"sample rate mismatch in merge {target}")
            parts.append(x)
        n = min(p.shape[0] for p in parts)
        tracks[target] = sum(p[:n] for p in parts)
    for name in wavs:
        if name in exclude:
            continue
        rate, x = read(name)
        sr = sr or rate
        if rate != sr:
            raise SystemExit(f"sample rate mismatch on {name}")
        tracks[name] = x

    n = min(x.shape[0] for x in tracks.values())
    return sr, {k: v[:n] for k, v in sorted(tracks.items())}


def analyze_pairwise(tracks: dict[str, np.ndarray], sr: int, window_seconds: float):
    """Run every scoring over every pair in isolation.

    Pairs are analyzed alone on purpose: the reference inside the scoring is the
    pair itself, and the target product is a plugin with one sidechain input
    that never sees more than two signals.
    """
    results: dict[str, dict[tuple[str, str], object]] = {s: {} for s in SCORINGS}
    for a, b in itertools.combinations(sorted(tracks), 2):
        for scoring in SCORINGS:
            det = MaskingDetector(
                sr=sr,
                scoring=scoring,
                window_seconds=window_seconds,
                stft=Stft(4096, 2048),
            )
            results[scoring][(a, b)] = det.analyze(
                [Track(a, tracks[a]), Track(b, tracks[b])]
            )
    return results


def all_zones(res_by_pair) -> list[Zone]:
    zones: list[Zone] = []
    for res in res_by_pair.values():
        zones.extend(res.zones)
    zones.sort(key=lambda z: z.score, reverse=True)
    return zones


def overlaps(x: Zone, y: Zone) -> bool:
    """Same pair, and both the window range and the frequency range overlap."""
    if set(x.tracks) != set(y.tracks):
        return False
    w_hit = x.windows[0] <= y.windows[1] and y.windows[0] <= x.windows[1]
    f_hit = x.freq_lo_hz < y.freq_hi_hz and y.freq_lo_hz < x.freq_hi_hz
    return w_hit and f_hit


def select_zones(by_scoring) -> list[tuple[str, Zone]]:
    """Pick the positions the scorings disagree about. Returns (rule, zone)."""
    contention = all_zones(by_scoring["contention"])
    relative = all_zones(by_scoring["relative"])
    picked: list[tuple[str, Zone]] = []

    def take(rule: str, zone: Zone) -> None:
        if any(overlaps(zone, z) for _, z in picked):
            return
        if len(picked) < 8:
            picked.append((rule, zone))

    for z in contention[:3]:
        take("contention top 3", z)
    for z in contention:
        if z.freq_lo_hz < LOW_END_MAX_HZ and z.freq_hi_hz <= LOW_END_MAX_HZ:
            take("contention low end", z)
            break
    n_before = len(picked)
    for z in relative:
        if len(picked) - n_before >= 2:
            break
        if not any(overlaps(z, c) for c in contention):
            take("relative top, contention silent", z)
    return picked


def zone_bands(zone: Zone, edges: np.ndarray, n_bands: int) -> list[int]:
    return [
        k
        for k in range(n_bands)
        if edges[k] >= zone.freq_lo_hz - 1e-6 and edges[k + 1] <= zone.freq_hi_hz + 1e-6
    ]


def louder_track(zone: Zone, res, n_bands: int) -> str:
    """The track occupying the space. It gets the dip so the other one can through."""
    ks = zone_bands(zone, res.band_edges_hz, n_bands)
    w0, w1 = zone.windows
    a, b = zone.tracks
    pa = res.band_power[a][w0 : w1 + 1, ks].mean()
    pb = res.band_power[b][w0 : w1 + 1, ks].mean()
    return a if pa >= pb else b


def excerpt_bounds(zone: Zone, window_seconds: float, n_samples: int, sr: int):
    start = zone.windows[0] * window_seconds - PAD_SECONDS
    end = (zone.windows[1] + 1) * window_seconds + PAD_SECONDS
    total = n_samples / sr
    if end - start > MAX_EXCERPT_S:
        mid = 0.5 * (start + end)
        start, end = mid - MAX_EXCERPT_S / 2, mid + MAX_EXCERPT_S / 2
    if end - start < MIN_EXCERPT_S:
        mid = 0.5 * (start + end)
        start, end = mid - MIN_EXCERPT_S / 2, mid + MIN_EXCERPT_S / 2
    start = max(0.0, start)
    end = min(total, end)
    return int(start * sr), int(end * sr)


def apply_dip(x: np.ndarray, zone: Zone, sr: int, dip_db: float) -> np.ndarray:
    """Peaking cut across the zone band. Center is the geometric mean because the
    band grid is logarithmic; Q is set so the bandwidth matches the zone."""
    f_center = float(np.sqrt(zone.freq_lo_hz * zone.freq_hi_hz))
    bandwidth = max(zone.freq_hi_hz - zone.freq_lo_hz, 1e-6)
    coeffs = peaking(f_center, sr, -abs(dip_db), q=f_center / bandwidth)
    out = np.empty_like(x)
    for ch in range(x.shape[1]):
        out[:, ch] = Biquad(coeffs).process(x[:, ch])
    return out


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def match_and_guard(a: np.ndarray, b: np.ndarray):
    """Match b's RMS to a, then keep both below full scale with one shared factor.

    Without the RMS match the louder rendering always wins the comparison and
    the test says nothing. The clip guard uses the same factor on both so the
    relationship between them survives. Returns (a, b, g) so the solo reference
    can be put into the same gain frame instead of getting its own.
    """
    ra, rb = rms(a), rms(b)
    b = b * (ra / rb) if rb > 0 else b
    peak = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-12)
    g = 0.999 / peak if peak > 0.999 else 1.0
    return a * g, b * g, g


def slug(text: str) -> str:
    """Filename-safe token. Commas and underscores would break the item naming."""
    for bad in (",", "_", " "):
        text = text.replace(bad, "-")
    return text.strip("-")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render A/B listening pairs for masking zones")
    ap.add_argument("stems_dir", type=Path, help="folder with one wav per track")
    ap.add_argument("--bpm", type=float, required=True, help="tempo for the time grid")
    ap.add_argument("--exclude", default="", help="comma list of stems that are not tracks")
    ap.add_argument("--merge", action="append", default=[], help="target:source1,source2")
    ap.add_argument("--name", default="masking", help="listening pack name")
    ap.add_argument("--dip-db", type=float, default=6.0, help="depth of the cut in dB")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "listenpacks",
        help="root folder for the pack",
    )
    args = ap.parse_args()

    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    merges: dict[str, list[str]] = {}
    for m in args.merge:
        target, sources = m.split(":")
        merges[target] = [s.strip() for s in sources.split(",")]
        exclude.update(merges[target])

    sr, tracks = load_stems(args.stems_dir, exclude, merges)
    window_seconds = 60.0 / args.bpm * 4.0 / WINDOWS_PER_BAR
    n_samples = next(iter(tracks.values())).shape[0]
    print(f"{len(tracks)} tracks at {sr} Hz, window {window_seconds:.4f} s")

    by_scoring = analyze_pairwise(tracks, sr, window_seconds)
    for scoring in SCORINGS:
        print(f"  {scoring:<11} {len(all_zones(by_scoring[scoring])):>4} zones")

    picked = select_zones(by_scoring)
    if not picked:
        raise SystemExit("no zones selected")

    full_mix = sum(tracks.values())
    n_bands = MaskingDetector(sr=sr).n_bands
    pack = ListenPack(args.name, root=args.root)
    rows = []

    for i, (rule, zone) in enumerate(picked, start=1):
        a, b = zone.tracks
        pair = (a, b) if (a, b) in by_scoring["contention"] else (b, a)
        res = by_scoring["contention"][pair]
        dipped = louder_track(zone, res, n_bands)
        s0, s1 = excerpt_bounds(zone, window_seconds, n_samples, sr)

        masked = b if dipped == a else a

        mix_a = full_mix[s0:s1].copy()
        mix_b = (full_mix - tracks[dipped])[s0:s1] + apply_dip(
            tracks[dipped], zone, sr, args.dip_db
        )[s0:s1]
        mix_a, mix_b, g = match_and_guard(mix_a, mix_b)
        # The track the zone claims is buried, at its own level inside the mix and
        # in the same gain frame as A and B. It names what to listen for, so
        # "heard no difference" can be told apart from "was audible all along".
        solo = tracks[masked][s0:s1] * g

        item = (
            f"{i:02d}_{slug(rule)}_{slug(a)}-x-{slug(b)}_"
            f"{zone.freq_lo_hz:.0f}-{zone.freq_hi_hz:.0f}Hz"
        )
        pack.add_pair(
            item,
            mix_a,
            mix_b,
            sr,
            label_a="mix",
            label_b="dipped",
            c=solo,
            label_c=f"solo-{slug(masked)}",
        )
        pack.spectrum_plot(
            item,
            {"mix": mix_a.mean(axis=1), "dipped": mix_b.mean(axis=1)},
            sr,
            title=f"{a} x {b}, {zone.freq_lo_hz:.0f}-{zone.freq_hi_hz:.0f} Hz, dip on {dipped}",
        )

        scores = {}
        for scoring in SCORINGS:
            hits = [z for z in all_zones(by_scoring[scoring]) if overlaps(zone, z)]
            scores[scoring] = f"{max(z.score for z in hits):.3f}" if hits else ""
        rows.append(
            {
                "item": item,
                "pair": f"{a} x {b}",
                "hz": f"{zone.freq_lo_hz:.0f}-{zone.freq_hi_hz:.0f}",
                "win": f"{zone.windows[0]}-{zone.windows[1]}",
                "dipped": dipped,
                "masked": masked,
                "rule": rule,
                "rms_delta_db": 20 * np.log10(rms(mix_b) / rms(mix_a)),
                **scores,
            }
        )

    pack.audition_page(title=f"Masking zones: {args.name}")

    head = (
        "| # | pair | Hz | windows | dipped | masked (C) | relative | collision | "
        "contention | rule |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {r['item'].split('_')[0]} | {r['pair']} | {r['hz']} | {r['win']} | "
        f"{r['dipped']} | {r['masked']} | {r['relative']} | {r['collision']} | "
        f"{r['contention']} | {r['rule']} |\n"
        for r in rows
    )
    readme = (
        f"# Masking zones: {args.name}\n\n"
        f"Source: `{args.stems_dir}`, {args.bpm:g} BPM, dip {args.dip_db:g} dB.\n\n"
        "This does not test any audio output. X-Ray processes nothing. A zone is a claim,\n"
        "and masking has no automatic ground truth, so the ear is the only oracle available.\n"
        "The cut is a measuring tool, not a feature.\n\n"
        "Three renderings per position, all in the same gain frame:\n\n"
        "- **A mix**: every track, untouched\n"
        "- **B dipped**: same mix, with a peaking cut across the zone band on the louder of\n"
        "  the two tracks, RMS matched to A so the louder rendering cannot win by loudness\n"
        "- **C solo**: the track the zone claims is buried, at its own level in the mix.\n"
        "  This names what to listen for\n\n"
        "Empty score cells mean that scoring method does not report the zone at all.\n\n"
        + head
        + body
        + "\nOpen `audition.html`. Switching keeps the playback position.\n\n"
        "## How to read a position\n\n"
        "Listen to C first, then A, then B.\n\n"
        "- C is hard to find inside A, and B uncovers it: the zone was real and worth acting on\n"
        "- C is plainly audible in A already: false alarm, whatever B does\n"
        "- C is buried in A and B does not uncover it: the zone may be real but the cut was\n"
        "  the wrong move, so the position says nothing either way\n\n"
        "Write the verdict down per position before moving on.\n"
    )
    (pack.dir / "README.md").write_text(readme)

    print(f"\npack: {pack.dir}")
    print(f"audition: {pack.dir / 'audition.html'}\n")
    print(head + body)
    print("RMS match (dB, should be ~0):")
    for r in rows:
        print(f"  {r['item']}: {r['rms_delta_db']:+.6f}")


if __name__ == "__main__":
    main()
