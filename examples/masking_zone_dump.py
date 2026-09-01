"""Dump every contention zone of the corpus with all its fields, for value diffs.

masking_corpus.py only reports counts and a top score per pair. That is too
coarse to prove a refactor changed nothing: a zone could move to a different
band or a different time range and the counts would not budge. This dumps the
full zone list so two revisions can be diffed field by field.

Zone.score is reported both as stored and in dB, because a change to the
display normalization is expected to move `score` while the underlying dB value
and the zone set itself must not move.

Run:  uv run python examples/masking_zone_dump.py ~/Music --out zones.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from masking_corpus import WINDOW_SECONDS, find_stem_folders, load_beat  # noqa: E402

from tagodsp.analysis.masking import MaskingDetector  # noqa: E402
from tagodsp.spectral.stft import Stft  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="library root to scan for stem folders")
    ap.add_argument("--limit", type=int, default=0, help="stop after N beats (0 = all)")
    ap.add_argument("--max-seconds", type=float, default=60.0, help="analyze the first N seconds")
    ap.add_argument("--pattern", default="stem", help="path must match this (case insensitive)")
    ap.add_argument("--scoring", default="contention")
    ap.add_argument("--out", type=Path, required=True, help="write the dump as json")
    args = ap.parse_args()

    folders = find_stem_folders(args.root, args.pattern)
    if args.limit:
        folders = folders[: args.limit]

    out = []
    for folder in folders:
        loaded, _excluded, why = load_beat(folder, args.max_seconds)
        if loaded is None:
            print(f"  skip {folder.name}: {why}")
            continue
        sr, tracks = loaded
        det = MaskingDetector(
            sr=sr, scoring=args.scoring, window_seconds=WINDOW_SECONDS, stft=Stft(4096, 2048)
        )
        res = det.analyze(tracks)
        # cells carry the raw value clustering ran on; for contention that is dB
        zones = [
            {
                "tracks": sorted(z.tracks),
                "band": z.band,
                "freq_lo_hz": round(z.freq_lo_hz, 4),
                "freq_hi_hz": round(z.freq_hi_hz, 4),
                "w0": z.windows[0],
                "w1": z.windows[1],
                "score": round(z.score, 6),
            }
            for z in res.zones
        ]
        zones.sort(key=lambda z: (z["tracks"], z["band"], z["w0"], z["w1"]))
        out.append({"beat": folder.name, "folder": str(folder.resolve()), "zones": zones})
        print(f"  {folder.name[:48]:<48} {len(zones):>4} zones")

    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n{sum(len(b['zones']) for b in out)} zones over {len(out)} beats -> {args.out}")


if __name__ == "__main__":
    main()
