"""Frequency masking (conflict) detection between multiple tracks.

Finds zones where two tracks occupy the same frequency band at the same time
and therefore mask each other. Pipeline (shared by all scoring methods):

  1. mono-sum each track (mean over channels if stereo)
  2. STFT via `tagodsp.spectral.stft.Stft` (default n_fft=4096, hop=1024)
  3. power spectrum |S|^2
  4. map bins to `n_bands` log-spaced bands between f_lo and f_hi
     (edges = geomspace(f_lo, f_hi, n_bands + 1)); power per band is the sum
     of its bins
  5. aggregate frames into time windows of `window_seconds`; the value per
     (window, band) is the mean of the frames that fall into it
  6. `n_windows` is shared across all tracks, sized to the longest track;
     shorter tracks are zero-padded so every band-power matrix has the same
     shape (n_windows, n_bands), holding absolute (non-normalized) power

Three scoring methods turn a pair of band-power matrices into a per-cell
"conflict amount". "relative" and "collision" produce cells in [0, 1] and are
thresholded there; "contention" produces cells in dB and is thresholded in dB,
with the [0, 1] normalization applied once per surviving zone (see below). The
clustering itself is the same code for all three.

Scoring "relative" (the X-Ray prototype's current, shipped behavior):

    peak = P.max()
    db   = 10 * log10(max(P / peak, 1e-9))       # -90 dB everywhere if peak <= 0
    norm = clip((db - db_floor) / -db_floor, 0, 1)
    cell = min(norm_a, norm_b)

Each track is normalized independently against its own loudest band. This
means two tracks can score a full conflict even if one is 30 dB quieter than
the other, because both are loud relative to themselves.

Scoring "collision" (proposed alternative, not yet measured):

    c = 2 * min(a, b) / (a + b)                  # 1.0 at parity, -> 0 as one
                                                  # track dominates; 0 if a+b == 0
    audible = mix_bb[w] >= 10 ** (abs_floor_db / 10)
              and (a + b) >= 10 ** (band_floor_db / 10) * mix_bb[w]
    cell = c if audible else 0.0

`mix_bb[w]` is the broadband power of the two tracks being compared (not the
whole mix), summed over every band, in window w (full-scale reference is
1.0). The reference is the pair rather than the full mix because the target
product is a plugin with a sidechain input: it only ever sees two signals
and never the whole mix, so a reference the product cannot compute would
not transfer. Measured on real stems, a full-mix reference made a pair's
zones depend on how many other tracks were playing at the same time
(closed_hat x rim: 2 zones among 8 stems, 11 zones for the same pair
analyzed alone); "relative" has no such dependency. Accepted cost of the
pair reference: two quiet tracks contending with each other still score
even if both are buried under the rest of the mix. Working on absolute
power instead of a per-track peak keeps the level difference between
tracks: a track 30 dB below another does not register as a collision. The
gate is evaluated per window (not against a whole-file peak) so the
detector stays streamable.

Scoring "contention" (replaces the ratio-plus-gate with one term):

    mix_bb[w] = band_power[a][w, k] + band_power[b][w, k], summed over bands,
                for the two tracks being compared
    share     = min(a, b) / mix_bb[w]            # 0.0 if mix_bb[w] <= 0
    cell      = 10 * log10(max(share, 1e-12))    # dB, NOT normalized to [0, 1]

`mix_bb` is the same pair-broadband sum used by "collision" (shared via the
`_pair_broadband` helper), not the whole mix; see the rationale above the
"collision" formula (sidechain-input product, per-pair reference, accepted
cost on buried-quiet pairs). `share` measures min(a, b) as a fraction of
what the two tracks being compared are doing, not as a fraction of a and
b's own sum the way "collision"'s 2*min(a,b)/(a+b) does; this makes the
term scale-aware, unlike "collision"'s ratio, which is blind to overall
loudness.

CONTENTION_CEIL_DB = -3.0: min(a, b) can be at most half of mix_bb, and with
the pair as the reference this bound is exact, not just an upper bound: two
tracks equally loud with their entire energy in one band makes
min(a, b) / mix_bb equal to exactly 0.5, i.e. -3.01 dB. The scale is
normalized against -3 dB, not 0 dB, because 0 dB would never be reachable.
Measured over the corpus (51 beats, examples/masking_scale.py, 2026-09-01) the
bound is not just theoretical: the highest cell anywhere reaches -3.13 dB, so
the top of the scale is real material and not headroom.

Thresholds and display scale are separate for "contention", and this is the
one thing that differs from the other two scorings:

  - clustering decides on the dB value directly, via contention_hit_db and
    contention_score_db, NOT on a [0, 1] cell. Physical thresholds survive a
    change of the display scale and port to C++ without a hidden mapping.
  - the reported Zone.score is normalized afterwards, over
    [contention_score_db, CONTENTION_CEIL_DB]. Everything that gets reported
    at all lives in that window, so the displayed score uses its full range.

The defaults -11.1 dB (hit) and -9.75 dB (score) are the exact dB equivalents
of the hit_min=0.7 / score_min=0.75 pair on the old -30 dB / -3 dB scale, so
the set of reported zones is unchanged from the detector that Phase 0
validated by ear. Only the number attached to a zone changed. The old
contention_floor_db = -30.0 was documented as an unvalidated placeholder and
is gone: measured over the corpus, reported zones span -9.75 to -3.10 dB, a
window of 6.65 dB. Normalizing that over 27 dB pressed every zone into the top
quarter of [0, 1] (median 0.83, p99 0.96) and made a display threshold near
0.95 meaningless. A zone sitting exactly on the reporting threshold now
displays as 0.0: it is the least bad thing still worth showing, not "nothing".

There is no separate audibility gate. min(a, b) does that job by itself:
  - one track much quieter than the other -> min(a, b) small -> share small
  - both tracks quiet (even at ratio 1.0) -> min(a, b) small -> share small
  - one track alone is loud, the other silent -> min(a, b) = 0 -> share = 0

hit_min and score_min apply to "relative" and "collision" only, and are tuned
for "relative" (the X-Ray prototype's shipped detector). "contention" uses its
own dB thresholds; see above.

Clustering (identical for all methods), ported from the X-Ray prototype's
`detect_conflicts()` (~/Documents/project-xray/mockup/analysis/analyze_stems.py):
cells at or above the hit threshold are grouped by BFS into regions (time gaps
up to `gap` windows, band adjacency of 1), short regions (< min_len windows)
are dropped, and the region value is the 90th percentile over its cells;
regions below the score threshold are dropped. The surviving region value is
mapped to Zone.score, which is always in [0, 1]: identity for "relative" and
"collision", the dB normalization described above for "contention".

Source: ~/Documents/project-xray/mockup/analysis/analyze_stems.py,
`analyze_streams()` and `detect_conflicts()`.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from tagodsp.spectral.stft import Stft

# Ceiling for scoring="contention": with the pair as broadband reference,
# min(a, b) / mix_bb reaches exactly 0.5 when the two tracks are equally
# loud (-3.01 dB). See module docstring for the derivation; measured over the
# corpus the highest cell reaches -3.13 dB, so the bound is not just theory.
CONTENTION_CEIL_DB = -3.0


@dataclass
class Track:
    name: str
    x: np.ndarray  # mono (n,) or stereo (n, 2)


@dataclass
class Zone:
    tracks: tuple[str, str]
    band: int
    freq_lo_hz: float
    freq_hi_hz: float
    windows: tuple[int, int]
    score: float


@dataclass
class MaskingResult:
    zones: list[Zone]
    band_edges_hz: np.ndarray
    n_windows: int
    window_seconds: float
    scoring: str
    band_power: dict[str, np.ndarray]  # per track, (n_windows, n_bands), absolute
    # per pair, (n_windows, n_bands), the values clustering ran on. Unit depends
    # on the scoring: [0, 1] for "relative" and "collision", dB for "contention".
    cells: dict[tuple[str, str], np.ndarray]


def _normalize_relative(P: np.ndarray, db_floor: float) -> np.ndarray:
    """Per-track relative normalization used by scoring "relative".

    peak = P.max(); db = 10*log10(max(P/peak, 1e-9)), or -90 dB everywhere if
    the track is silent (peak <= 0); scaled to [0, 1] via
    (db - db_floor) / -db_floor.
    """
    peak = P.max()
    if peak > 0:
        db = 10.0 * np.log10(np.maximum(P / peak, 1e-9))
    else:
        db = np.full_like(P, -90.0)
    return np.clip((db - db_floor) / -db_floor, 0.0, 1.0)


@dataclass
class MaskingDetector:
    """Detects frequency-masking zones between multiple tracks.

    See module docstring for the shared pipeline and the three scoring
    formulas ("relative" vs "collision" vs "contention"). hit_min/score_min
    apply to "relative" and "collision" only and are tuned for "relative";
    "contention" thresholds in dB via contention_hit_db/contention_score_db.
    """

    sr: float
    scoring: str = "collision"  # "collision" | "relative" | "contention"
    n_bands: int = 30
    f_lo: float = 20.0
    f_hi: float = 20000.0
    window_seconds: float = 0.1
    hit_min: float = 0.7  # not used by scoring="contention"
    score_min: float = 0.75  # not used by scoring="contention"
    gap: int = 2
    min_len: int = 3
    db_floor: float = -36.0  # only used by scoring="relative"
    band_floor_db: float = -35.0  # only used by scoring="collision"
    abs_floor_db: float = -80.0  # only used by scoring="collision"
    # only used by scoring="contention". The defaults are the exact dB
    # equivalents of hit_min=0.7 / score_min=0.75 on the old -30 dB scale, so
    # the reported zones are the ones Phase 0 validated. See module docstring.
    contention_hit_db: float = -11.1
    contention_score_db: float = -9.75
    stft: Stft = field(default_factory=lambda: Stft(n_fft=4096, hop=1024))

    def __post_init__(self) -> None:
        if self.scoring not in ("collision", "relative", "contention"):
            raise ValueError(
                f"scoring must be 'collision', 'relative', or 'contention', got {self.scoring!r}"
            )
        if self.contention_score_db >= CONTENTION_CEIL_DB:
            raise ValueError(
                f"contention_score_db must be < {CONTENTION_CEIL_DB}, "
                f"got {self.contention_score_db!r}"
            )
        if self.contention_hit_db > self.contention_score_db:
            raise ValueError(
                "contention_hit_db must be <= contention_score_db, otherwise a region "
                "could never reach a score its own cells cannot have; got "
                f"{self.contention_hit_db!r} > {self.contention_score_db!r}"
            )
        if self.n_bands < 2:
            raise ValueError("n_bands must be >= 2")
        if not (0 < self.f_lo < self.f_hi):
            raise ValueError("need 0 < f_lo < f_hi")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if self.gap < 0:
            raise ValueError("gap must be >= 0")
        if self.min_len < 1:
            raise ValueError("min_len must be >= 1")
        if not (0 <= self.hit_min <= 1):
            raise ValueError("hit_min must be in [0, 1]")
        if not (0 <= self.score_min <= 1):
            raise ValueError("score_min must be in [0, 1]")

    def band_edges(self) -> np.ndarray:
        return np.geomspace(self.f_lo, self.f_hi, self.n_bands + 1)

    def _mono(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            return x.mean(axis=1)
        if x.ndim != 1:
            raise ValueError("expected mono (n,) or stereo (n, 2) input")
        return x

    def _stft_power(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mono = self._mono(x)
        S = self.stft.forward(mono)
        power = np.abs(S) ** 2
        freqs = self.stft.freqs(self.sr)
        times = self.stft.times(S.shape[1], self.sr)
        return power, freqs, times

    def _band_indices(self, freqs: np.ndarray) -> np.ndarray:
        edges = self.band_edges()
        return np.digitize(freqs, edges) - 1

    def _pair_broadband(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Broadband power of the two tracks being compared, per window (full-scale ref 1.0).

        Not the whole mix: the reference is the pair, since the target product (a
        plugin with a sidechain input) only ever sees two signals. Shared by
        scoring="collision" (audibility gate) and scoring="contention" (share
        denominator).
        """
        return a.sum(axis=1) + b.sum(axis=1)

    def _aggregate_bands(
        self, power: np.ndarray, band_idx: np.ndarray, times: np.ndarray, n_windows: int
    ) -> np.ndarray:
        P = np.zeros((n_windows, self.n_bands))
        if len(times) == 0:
            return P
        time_idx = np.minimum((times / self.window_seconds).astype(int), n_windows - 1)
        for b in range(self.n_bands):
            rows = band_idx == b
            if not rows.any():
                continue
            col = power[rows].sum(axis=0)
            for w in range(n_windows):
                sel = time_idx == w
                if sel.any():
                    P[w, b] = col[sel].mean()
        return P

    def band_power(self, x: np.ndarray, n_windows: int | None = None) -> np.ndarray:
        power, freqs, times = self._stft_power(x)
        band_idx = self._band_indices(freqs)
        if n_windows is None:
            longest = float(times[-1]) if len(times) else 0.0
            n_windows = max(1, int(round(longest / self.window_seconds)))
        return self._aggregate_bands(power, band_idx, times, n_windows)

    def _zone_score(self, value: float) -> float:
        """Map a surviving region value to the reported Zone.score in [0, 1].

        Identity for "relative"/"collision", whose cells already are [0, 1].
        For "contention" the cells are dB, normalized here over
        [contention_score_db, CONTENTION_CEIL_DB]: the reporting threshold
        displays as 0.0, the exact upper bound of the measure as 1.0.
        """
        if self.scoring != "contention":
            return value
        span = CONTENTION_CEIL_DB - self.contention_score_db
        return float(np.clip((value - self.contention_score_db) / span, 0.0, 1.0))

    def _cluster(
        self,
        cell: np.ndarray,
        edges: np.ndarray,
        name_i: str,
        name_j: str,
        hit_min: float,
        score_min: float,
    ) -> list[Zone]:
        """Group cells at or above `hit_min` into zones, keeping regions >= `score_min`.

        Both thresholds are in the unit of `cell`, which the caller picks: [0, 1]
        for "relative"/"collision", dB for "contention".
        """
        n_windows, n_bands = cell.shape
        hit = cell >= hit_min
        seen = np.zeros_like(hit, dtype=bool)
        zones: list[Zone] = []
        for w0, k0 in np.argwhere(hit):
            w0, k0 = int(w0), int(k0)
            if seen[w0, k0]:
                continue
            seen[w0, k0] = True
            queue = deque([(w0, k0)])
            cells: list[tuple[int, int]] = []
            while queue:
                w, k = queue.popleft()
                cells.append((w, k))
                for dw in range(-self.gap, self.gap + 1):
                    for dk in (-1, 0, 1):
                        w2, k2 = w + dw, k + dk
                        if (
                            0 <= w2 < n_windows
                            and 0 <= k2 < n_bands
                            and hit[w2, k2]
                            and not seen[w2, k2]
                        ):
                            seen[w2, k2] = True
                            queue.append((w2, k2))
            ws = [w for w, _ in cells]
            if max(ws) - min(ws) + 1 < self.min_len:
                continue
            vals = np.sort([cell[w, k] for w, k in cells])
            idx = min(int(len(vals) * 0.9), len(vals) - 1)
            value = float(vals[idx])
            if value < score_min:
                continue
            score = self._zone_score(value)
            # Dominant band: the one contributing the most cells to the region,
            # ties broken by their summed strength. The X-Ray prototype summed
            # the [0, 1] cell values, where every cell of a region sits between
            # hit_min and 1.0 and the sum is therefore a cell count with a weak
            # strength modulation. Made explicit here so the choice no longer
            # depends on where the zero of the cell scale happens to sit, which
            # would silently change once the cells are dB.
            band_count: dict[int, int] = {}
            band_strength: dict[int, float] = {}
            for w, k in cells:
                band_count[k] = band_count.get(k, 0) + 1
                band_strength[k] = band_strength.get(k, 0.0) + float(cell[w, k])
            ks = [k for _, k in cells]
            band = max(band_count, key=lambda k: (band_count[k], band_strength[k]))
            zones.append(
                Zone(
                    tracks=(name_i, name_j),
                    band=band,
                    freq_lo_hz=float(edges[min(ks)]),
                    freq_hi_hz=float(edges[max(ks) + 1]),
                    windows=(min(ws), max(ws)),
                    score=score,
                )
            )
        return zones

    def analyze(self, tracks: list[Track]) -> MaskingResult:
        edges = self.band_edges()

        raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        max_time = 0.0
        for t in tracks:
            power, freqs, times = self._stft_power(t.x)
            raw[t.name] = (power, freqs, times)
            if len(times):
                max_time = max(max_time, float(times[-1]))
        n_windows = max(1, int(round(max_time / self.window_seconds)))

        band_power: dict[str, np.ndarray] = {}
        for t in tracks:
            power, freqs, times = raw[t.name]
            band_idx = self._band_indices(freqs)
            band_power[t.name] = self._aggregate_bands(power, band_idx, times, n_windows)

        cells: dict[tuple[str, str], np.ndarray] = {}
        zones: list[Zone] = []

        if self.scoring == "relative":
            norm = {name: _normalize_relative(P, self.db_floor) for name, P in band_power.items()}
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    name_i, name_j = tracks[i].name, tracks[j].name
                    cell = np.minimum(norm[name_i], norm[name_j])
                    cells[(name_i, name_j)] = cell
                    zones.extend(
                        self._cluster(cell, edges, name_i, name_j, self.hit_min, self.score_min)
                    )
        elif self.scoring == "collision":
            abs_floor = 10 ** (self.abs_floor_db / 10)
            band_floor = 10 ** (self.band_floor_db / 10)
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    name_i, name_j = tracks[i].name, tracks[j].name
                    a = band_power[name_i]
                    b = band_power[name_j]
                    mix_bb = self._pair_broadband(a, b)
                    denom = a + b
                    c = np.zeros_like(a)
                    nz = denom > 0
                    c[nz] = 2.0 * np.minimum(a, b)[nz] / denom[nz]
                    gate_abs = mix_bb >= abs_floor
                    gate_band = denom >= band_floor * mix_bb[:, None]
                    cell = np.where(gate_abs[:, None] & gate_band, c, 0.0)
                    cells[(name_i, name_j)] = cell
                    zones.extend(
                        self._cluster(cell, edges, name_i, name_j, self.hit_min, self.score_min)
                    )
        else:  # contention
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    name_i, name_j = tracks[i].name, tracks[j].name
                    a = band_power[name_i]
                    b = band_power[name_j]
                    mix_bb = self._pair_broadband(a, b)
                    min_ab = np.minimum(a, b)
                    share = np.zeros_like(a)
                    nz = mix_bb > 0
                    share[nz, :] = min_ab[nz, :] / mix_bb[nz, None]
                    # cells stay in dB here; clustering thresholds are dB and the
                    # normalization to [0, 1] happens once per surviving zone
                    cell = 10.0 * np.log10(np.maximum(share, 1e-12))
                    cells[(name_i, name_j)] = cell
                    zones.extend(
                        self._cluster(
                            cell,
                            edges,
                            name_i,
                            name_j,
                            self.contention_hit_db,
                            self.contention_score_db,
                        )
                    )

        zones.sort(key=lambda z: z.score, reverse=True)

        return MaskingResult(
            zones=zones,
            band_edges_hz=edges,
            n_windows=n_windows,
            window_seconds=self.window_seconds,
            scoring=self.scoring,
            band_power=band_power,
            cells=cells,
        )
