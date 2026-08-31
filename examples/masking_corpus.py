"""Negative control at scale: how loud is the detector on finished mixes?

Runs every scoring over a library of stem folders. The material is Robin's own
released beats, so it is mixed and considered clean. A detector meant to point
at real problems has to be quiet here. Whatever it reports on this corpus is a
false alarm rate, not a finding.

This is only half the picture. The positive control (does it react to real
masking at all) lives in masking_sweep.py. Neither script states a verdict.

One analyze() call per beat over all stems is used instead of looping pairs.
That is not an approximation: since the scoring reference is the pair itself,
both give bit identical zones, verified on real stems.

Run:  uv run python examples/masking_corpus.py ~/Music --limit 10
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from tagodsp.analysis.masking import MaskingDetector, Track  # noqa: E402
from tagodsp.spectral.stft import Stft  # noqa: E402

SCORINGS = ("relative", "collision", "contention")
WINDOW_SECONDS = 0.2  # fixed grid: one variable less across beats of unknown tempo
MIN_STEMS = 4
MAX_STEMS = 24

# Names that are a sum of other stems rather than a track of their own. Kept
# conservative on purpose; every exclusion is printed so it stays auditable.
SUM_PATTERNS = re.compile(
    r"^(master|mixdown|mix|full|premaster|pre-master|beat|instrumental)\b"
    r"|(drum[\s_-]*bus|full[\s_-]*mix|sum)",
    re.IGNORECASE,
)


def find_stem_folders(root: Path, pattern: str) -> list[Path]:
    """Folders holding at least MIN_STEMS wav files directly (not in subfolders).

    The path has to match `pattern` somewhere, so a DAW's rendered-clips folder
    does not get mistaken for a stem export of a finished mix.
    """
    rx = re.compile(pattern, re.IGNORECASE)
    out = []
    for d in sorted({p.parent for p in root.rglob("*.wav")}):
        if not rx.search(str(d.relative_to(root))):
            continue
        n = sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
        if n >= MIN_STEMS:
            out.append(d)
    return out


def load_beat(folder: Path, max_seconds: float):
    """Return (sr, [Track]) or None if the folder is unusable."""
    wavs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    kept, dropped = [], []
    for p in wavs:
        (dropped if SUM_PATTERNS.search(p.stem) else kept).append(p)
    if not (MIN_STEMS <= len(kept) <= MAX_STEMS):
        return None, dropped, f"{len(kept)} usable stems"

    tracks, sr = [], None
    for p in kept:
        try:
            info = sf.info(p)
        except Exception as exc:  # unreadable file, skip the whole beat
            return None, dropped, f"unreadable {p.name}: {exc}"
        if sr is None:
            sr = info.samplerate
        elif info.samplerate != sr:
            return None, dropped, "mixed sample rates"
        x, _ = sf.read(p, frames=int(max_seconds * sr), dtype="float64", always_2d=True)
        tracks.append(Track(p.stem, x))

    n = min(t.x.shape[0] for t in tracks)
    if n < sr * 2:
        return None, dropped, "shorter than 2 s"
    for t in tracks:
        t.x = t.x[:n]
    return (sr, tracks), dropped, None


def analyze_beat(sr: int, tracks: list[Track]) -> dict[str, dict]:
    out = {}
    n_pairs = len(tracks) * (len(tracks) - 1) // 2
    for scoring in SCORINGS:
        det = MaskingDetector(
            sr=sr, scoring=scoring, window_seconds=WINDOW_SECONDS, stft=Stft(4096, 2048)
        )
        zones = det.analyze(tracks).zones
        noisy = {tuple(sorted(z.tracks)) for z in zones}
        out[scoring] = {
            "zones": len(zones),
            "pairs_flagged": len(noisy),
            "pairs_total": n_pairs,
            "top_score": round(max((z.score for z in zones), default=0.0), 3),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the masking detector over a stem library")
    ap.add_argument("root", type=Path, help="library root to scan for stem folders")
    ap.add_argument("--limit", type=int, default=0, help="stop after N beats (0 = all)")
    ap.add_argument("--max-seconds", type=float, default=60.0, help="analyze the first N seconds")
    ap.add_argument(
        "--folder-pattern",
        default="stem",
        help="only scan folders whose path matches this (case insensitive)",
    )
    ap.add_argument("--out", type=Path, default=None, help="write results as json")
    args = ap.parse_args()

    folders = find_stem_folders(args.root, args.folder_pattern)
    print(f"{len(folders)} candidate folders under {args.root} matching {args.folder_pattern!r}")

    results, skipped = [], []
    for folder in folders:
        if args.limit and len(results) >= args.limit:
            break
        loaded, dropped, why = load_beat(folder, args.max_seconds)
        if loaded is None:
            skipped.append((folder, why))
            continue
        sr, tracks = loaded
        t0 = time.time()
        stats = analyze_beat(sr, tracks)
        name = folder.relative_to(args.root).as_posix()
        results.append({"beat": name, "stems": len(tracks), "stats": stats})
        print(
            f"  {name[:52]:<52} {len(tracks):>3} stems  {time.time() - t0:5.1f}s  "
            + "  ".join(f"{s[:4]}={stats[s]['zones']:>4}" for s in SCORINGS)
        )
        if dropped:
            print(f"      excluded as sums: {', '.join(p.stem for p in dropped)}")

    if not results:
        raise SystemExit("no usable beats found")

    print(f"\n{len(results)} beats analyzed, {len(skipped)} skipped")
    print(f"\n{'scoring':<12} {'zones/beat':>12} {'median':>8} {'quiet beats':>13} "
          f"{'pairs flagged':>15}")
    print("-" * 64)
    summary = {}
    for s in SCORINGS:
        z = np.array([r["stats"][s]["zones"] for r in results], dtype=float)
        flagged = sum(r["stats"][s]["pairs_flagged"] for r in results)
        total = sum(r["stats"][s]["pairs_total"] for r in results)
        summary[s] = {
            "mean_zones": float(z.mean()),
            "median_zones": float(np.median(z)),
            "silent_beats": int((z == 0).sum()),
            "beats": len(z),
            "pairs_flagged_pct": 100.0 * flagged / total if total else 0.0,
        }
        print(
            f"{s:<12} {z.mean():>12.1f} {np.median(z):>8.0f} "
            f"{int((z == 0).sum()):>6} / {len(z):<4} {100.0 * flagged / total:>14.1f}%"
        )
    print("\n'quiet beats' counts beats with zero zones. On finished mixes, higher is better.")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "window_seconds": WINDOW_SECONDS,
                    "max_seconds": args.max_seconds,
                    "summary": summary,
                    "beats": results,
                    "skipped": [[str(f), w] for f, w in skipped],
                },
                indent=2,
            )
        )
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
