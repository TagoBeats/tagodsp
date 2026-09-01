"""Where does scoring="contention" actually live on its own scale?

The cell value is d = 10*log10(min(a, b) / mix_bb) in dB, upper bound -3.01 dB
(min(a, b) is at most half of mix_bb, reached when the pair puts its entire
energy into one band). The reported Zone.score normalizes d over
[contention_score_db, CONTENTION_CEIL_DB].

This script measures what d really does over the corpus. Two questions:

  1. Is the -3 dB ceiling reachable at all, or is it pure theory? Measured
     2026-09-01 over 51 beats: the highest cell hits -3.13 dB, so it is real
     material. The ceiling stays where it is.
  2. Where does the reported population sit? Zones span -9.75 to -3.10 dB, a
     window of 6.65 dB. The old normalization ran from -30 dB, an unvalidated
     placeholder, which pressed every reported zone into the top quarter of
     [0, 1] and left no room for a display threshold. That is why the display
     floor is now the reporting threshold.

d decomposes exactly into two factors:

    r = 2*min(a, b) / (a + b)      # contention ratio in the band, 1.0 at parity
    s = (a + b) / mix_bb           # the band's share of the pair's broadband power
    min(a, b) / mix_bb = r * s / 2

r is the thing the detector is about; s is how concentrated the pair's energy
is. The theoretical ceiling assumes s = 1. Both are reported separately,
because a ceiling picked without knowing the s distribution would be a guess.

States no verdict and changes no defaults.

Run:  uv run python examples/masking_scale.py ~/Music --out scale.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from masking_corpus import WINDOW_SECONDS, find_stem_folders, load_beat  # noqa: E402

from tagodsp.analysis.masking import CONTENTION_CEIL_DB, MaskingDetector  # noqa: E402
from tagodsp.spectral.stft import Stft  # noqa: E402

# Fixed histogram grid so percentiles are exact to the bin width without
# holding every cell of the corpus in memory (51 beats are ~10^8 cells).
DB_LO, DB_HI, DB_STEP = -120.0, 0.0, 0.05
EDGES = np.arange(DB_LO, DB_HI + DB_STEP, DB_STEP)
CENTERS = EDGES[:-1] + DB_STEP / 2


def hist(values: np.ndarray) -> np.ndarray:
    return np.histogram(values, bins=EDGES)[0]


def pct(counts: np.ndarray, q: float) -> float:
    """Percentile from the histogram (bin center), q in percent."""
    total = counts.sum()
    if total == 0:
        return float("nan")
    return float(CENTERS[np.searchsorted(np.cumsum(counts), total * q / 100.0)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="library root to scan for stem folders")
    ap.add_argument("--limit", type=int, default=0, help="stop after N beats (0 = all)")
    ap.add_argument("--max-seconds", type=float, default=60.0, help="analyze the first N seconds")
    ap.add_argument("--pattern", default="stem", help="path must match this (case insensitive)")
    ap.add_argument("--out", type=Path, default=None, help="write results as json")
    args = ap.parse_args()

    defaults = MaskingDetector(sr=48000, scoring="contention")
    hit_db = defaults.contention_hit_db
    score_db = defaults.contention_score_db
    span = CONTENTION_CEIL_DB - score_db  # the display scale

    folders = find_stem_folders(args.root, args.pattern)
    if args.limit:
        folders = folders[: args.limit]
    print(f"{len(folders)} stem folders, hit at {hit_db:.2f} dB, score at {score_db:.2f} dB\n")

    h_all = np.zeros(len(CENTERS), dtype=np.int64)  # d over every cell with energy
    h_hit = np.zeros(len(CENTERS), dtype=np.int64)  # d over cells at or above hit_db
    h_parity = np.zeros(len(CENTERS), dtype=np.int64)  # d where r >= 0.95 (band at parity)
    h_s = np.zeros(len(CENTERS), dtype=np.int64)  # s in dB, all cells with energy
    zone_db: list[float] = []  # zone scores (p90 of the region) back in dB
    beats = 0
    d_max = -np.inf

    for folder in folders:
        loaded, _excluded, why = load_beat(folder, args.max_seconds)
        if loaded is None:
            print(f"  skip {folder.name}: {why}")
            continue
        sr, tracks = loaded
        det = MaskingDetector(
            sr=sr, scoring="contention", window_seconds=WINDOW_SECONDS, stft=Stft(4096, 2048)
        )
        res = det.analyze(tracks)
        bp = res.band_power

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a, b = bp[tracks[i].name], bp[tracks[j].name]
                mix_bb = a.sum(axis=1) + b.sum(axis=1)
                nz = mix_bb > 0
                if not nz.any():
                    continue
                min_ab = np.minimum(a, b)[nz]
                tot = (a + b)[nz]
                share = min_ab / mix_bb[nz, None]
                live = share > 0
                if not live.any():
                    continue
                d = 10.0 * np.log10(share[live])
                s = tot[live] / mix_bb[nz, None].repeat(a.shape[1], axis=1)[live]
                r = 2.0 * min_ab[live] / tot[live]
                h_all += hist(d)
                h_hit += hist(d[d >= hit_db])
                h_parity += hist(d[r >= 0.95])
                h_s += hist(10.0 * np.log10(np.maximum(s, 1e-12)))
                d_max = max(d_max, float(d.max()))

        # zone score is a p90 over cells, so it is not recoverable from the
        # histograms; invert the stored normalized value instead
        for z in res.zones:
            zone_db.append(score_db + z.score * span)
        beats += 1
        print(f"  {folder.name[:48]:<48} {len(tracks):>3} stems  {len(res.zones):>4} zones")

    if beats == 0:
        raise SystemExit("no usable beats")

    zone_arr = np.array(zone_db) if zone_db else np.zeros(0)
    clipped = int((zone_arr >= CONTENTION_CEIL_DB - 1e-9).sum())

    report = {
        "beats": beats,
        "window_seconds": WINDOW_SECONDS,
        "max_seconds": args.max_seconds,
        "contention_ceil_db": CONTENTION_CEIL_DB,
        "hit_db": hit_db,
        "score_db": score_db,
        "d_max_observed": d_max,
        "cells": {f"p{q}": pct(h_all, q) for q in (50, 90, 99, 99.9, 99.99, 100)},
        "cells_above_hit": {f"p{q}": pct(h_hit, q) for q in (50, 90, 99, 99.9, 100)},
        "cells_at_parity_r095": {f"p{q}": pct(h_parity, q) for q in (50, 90, 99, 99.9, 100)},
        "band_share_s_db": {f"p{q}": pct(h_s, q) for q in (50, 90, 99, 99.9, 100)},
        "zones": {
            "n": len(zone_db),
            "min": float(zone_arr.min()) if len(zone_arr) else None,
            "median": float(np.median(zone_arr)) if len(zone_arr) else None,
            "p90": float(np.percentile(zone_arr, 90)) if len(zone_arr) else None,
            "p99": float(np.percentile(zone_arr, 99)) if len(zone_arr) else None,
            "max": float(zone_arr.max()) if len(zone_arr) else None,
            "clipped_at_ceiling": clipped,
        },
    }

    print(f"\n{beats} beats, {len(zone_db)} contention zones")
    print(
        f"highest d anywhere in the corpus: {d_max:.2f} dB (theoretical bound {CONTENTION_CEIL_DB})"
    )
    for key in ("cells", "cells_above_hit", "cells_at_parity_r095", "band_share_s_db", "zones"):
        print(f"\n{key}:")
        for k, v in report[key].items():
            print(f"  {k:>20} {v if not isinstance(v, float) else f'{v:8.2f}'}")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
