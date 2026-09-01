"""Detect stems that are sums of other, still-kept stems, not tracks of their own.

Why this exists: FL Studio prefixes every exported stem with the project name, so a
master-bus export is called e.g. "night line_Master", not "Master". A pattern
anchored at the start of the file name therefore never matches; the marker word has
to be looked for in the last underscore-separated segment instead
(`looks_like_sum_name`). "Master" and "Current" bounces of the same project are
bit-identical copies of the full mix, so leaving either one in a stem folder means
whatever runs on top of these stems (MaskingDetector included) would end up
comparing the mix against itself.

A name match alone is not proof, only a cheap pre-filter. The measured check
(`find_sum_stems`) correlates each remaining stem against the sum of the others,
with lag search to cover plugin delay compensation on a master bus (44.5 ms
measured on a real project). That alone is not enough either: a doubled layer (the
same part bounced twice, e.g. under two plugin chains) also correlates highly
against the sum of the others, because it is still in there once. The partner test
tells the two apart: a genuine sum contains many tracks and survives losing any one
of them, while a doubled layer is a single track wearing two names and collapses
once its one partner is removed from the comparison sum.

Measured ground truth behind the thresholds:
  - full-mix copy ("Master"/"Current"): corr 0.967 against the sum of the other
    stems, at 44.5 ms lag (plugin delay compensation on the master bus)
  - doubled layer (one part exported under two chains): corr 0.851 against the sum
    of the others, but -0.663 once its one partner is removed
  - genuine submix (several tracks routed into one bus): corr 0.969 against the sum
    of the others, and still 0.522 once its most-correlated partner is removed

This module takes signal arrays, not file paths, and depends only on numpy: no
soundfile, no MaskingDetector. That is what makes it testable with synthetic
signals.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Words that mark a stem as a sum of other stems rather than a track of its own,
# checked against the last underscore-separated segment of the file name (see
# looks_like_sum_name). Kept conservative on purpose: "instruments" is never in
# here, it is a real stem in source-separation folders (vocals/drums/bass/instruments).
SUM_NAME_WORDS = frozenset(
    {"master", "current", "mixdown", "premaster", "pre-master", "mix", "mixed", "sum"}
)

# Cross-correlation above which a stem measures as a sum of the other, still kept
# stems (see find_sum_stems). Measured ground truth: a full-mix copy correlates at
# +0.967 with the sum of the other stems in its project. 0.8 leaves clear margin
# below that while still being well above what two genuinely different but related
# stems (e.g. doubled melody lines) show. Fixed at 0.8, do not retune this against
# a desired result.
SUM_CORR_MAX = 0.8

# Corr threshold for the partner test in find_sum_stems: with the single highest
# correlated partner removed from the comparison sum, a genuine sum has to stay
# above this value. A sum contains many tracks, so removing just one of them
# cannot collapse it; a doubled layer is exactly one track wearing two names, so
# removing its one partner takes almost everything with it. Measured ground truth:
# a doubled 808 layer correlates at +0.851 against the sum of all other stems, but
# only -0.663 once its one partner is removed; a genuine submix stays at 0.522
# with its most-correlated partner removed. 0.5 sits with a large margin below
# where real sums land and well above where doubles fall. Fixed at 0.5, do not
# retune this against a desired result.
SUM_CORR_PARTNER_MIN = 0.5

# How much lag to search when correlating a candidate sum stem against the sum of
# the others. Plugin delay compensation on a master bus was measured at 44.5 ms
# for a real project; 1 s is generous headroom above that.
SUM_CORR_MAX_LAG_SECONDS = 1.0

# How much audio to use for the measured sum check. Enough to get a stable
# correlation estimate without reading and FFT-ing entire files.
SUM_CORR_SECONDS = 20.0


@dataclass
class SumStemVerdict:
    """One stem measured to be a sum of the other, still-kept stems."""

    name: str
    reason: str  # always "measured": name-based filtering is looks_like_sum_name, not this
    corr: float
    corr_without_partner: float | None  # None when the partner test was skipped


def looks_like_sum_name(name: str) -> bool:
    """True if a file name marks a sum of other stems rather than a stem of its own.

    FL Studio prefixes every exported stem with the project name, so the file is
    called e.g. "night line_Master", not "Master": the marker word sits in the
    last underscore-separated segment, not at the string start. Also treats any
    last segment ending in "bus" as a sum ("Drum Bus", "Master Bus"), a bus is by
    definition a sum of the tracks routed into it.
    """
    last = name.rsplit("_", 1)[-1].strip().lower()
    return last in SUM_NAME_WORDS or last.endswith("bus")


def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def best_lag_correlation(a: np.ndarray, b: np.ndarray, sr: float, max_lag_seconds: float) -> float:
    """Max normalized cross-correlation between mono a and b over +-max_lag_seconds.

    FFT based (np.fft.rfft/irfft), not an O(n*m) loop. Normalization is the usual
    cc / (norm(a) * norm(b)) after mean removal. a and b must be the same length.
    """
    a = a - a.mean()
    b = b - b.mean()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    fa = np.fft.rfft(a, nfft)
    fb = np.fft.rfft(b, nfft)
    cc = np.fft.irfft(fa * np.conj(fb), nfft)
    max_lag = int(round(max_lag_seconds * sr))
    lags = np.arange(-max_lag, max_lag + 1)
    idx = lags % nfft  # negative lags wrap to the end of the circular result
    vals = cc[idx] / (na * nb)
    return float(np.max(vals))


def find_sum_stems(
    names: Sequence[str],
    signals: Sequence[np.ndarray],
    sr: float,
    corr_max: float = SUM_CORR_MAX,
    partner_min: float = SUM_CORR_PARTNER_MIN,
    max_lag_seconds: float = SUM_CORR_MAX_LAG_SECONDS,
    corr_seconds: float = SUM_CORR_SECONDS,
) -> list[SumStemVerdict]:
    """Iteratively find stems that measure as a sum of the other, still-kept stems.

    For each remaining stem, computes the maximum normalized cross-correlation
    between it and the sum of all other remaining stems (lag search over
    +-max_lag_seconds, to cover plugin delay compensation on a master bus). A stem
    only counts as a sum if it clears both:

      (a) corr against the sum of the others > corr_max
      (b) partner test: find the single other stem j most correlated with the
          candidate, remove ONLY j from the comparison sum, and correlate again.
          A genuine sum contains many tracks and stays high once one partner is
          gone; a doubled layer is one track under two names and collapses (see
          SUM_CORR_PARTNER_MIN). Skipped if fewer than 3 stems would remain in the
          comparison sum after removing j, in which case only (a) decides and
          that is recorded as corr_without_partner=None in the returned verdict.

    Candidates are walked in descending order of (a); a candidate that fails the
    partner test is left in the pool (it is a double, not a sum) and the next
    candidate is tried, since the pool is unchanged and its own corr against (a)
    would not change on a retry. As soon as one candidate clears both checks it is
    dropped and the whole search restarts on what remains, one at a time on
    purpose: a duplicated master bounce (e.g. "Master" and "Current" both bit
    identical to the mix) each covers for the other if both are still in the pool,
    so removing them together would undercount.

    Runs on the first corr_seconds of each signal (or the available length if
    shorter), which is enough to measure the correlation and keeps this fast.

    Returns the found sums as a list of SumStemVerdict, in removal order.
    """
    n_max = int(round(corr_seconds * sr))
    clips = {
        name: _mono(np.asarray(x)[:n_max]).astype(np.float64)
        for name, x in zip(names, signals, strict=True)
    }
    remaining = list(names)
    found: list[SumStemVerdict] = []

    while len(remaining) >= 2:
        n = min(clips[name].shape[0] for name in remaining)
        arrs = {name: clips[name][:n] for name in remaining}
        total = sum(arrs.values())

        scored = []
        for name in remaining:
            others_sum = total - arrs[name]
            corr = best_lag_correlation(arrs[name], others_sum, sr, max_lag_seconds)
            scored.append((name, corr))
        scored.sort(key=lambda t: -t[1])

        dropped_name = None
        for name, corr in scored:
            if corr <= corr_max:
                break  # sorted descending: nothing further down clears (a) either
            others = [x for x in remaining if x != name]

            partner, partner_corr = None, -1.0
            for other in others:
                pc = best_lag_correlation(arrs[name], arrs[other], sr, max_lag_seconds)
                if pc > partner_corr:
                    partner, partner_corr = other, pc

            comparison_without_partner = [x for x in others if x != partner]
            if len(comparison_without_partner) >= 3:
                sum_without_partner = total - arrs[name] - arrs[partner]
                corr_without_partner = best_lag_correlation(
                    arrs[name], sum_without_partner, sr, max_lag_seconds
                )
                is_sum = corr_without_partner > partner_min
            else:
                corr_without_partner = None
                is_sum = True  # too few stems left to run the partner test, (a) decides

            if is_sum:
                found.append(SumStemVerdict(name, "measured", corr, corr_without_partner))
                dropped_name = name
                break
            # doubled layer, not a sum: leave it in the pool, try the next candidate

        if dropped_name is None:
            break
        remaining = [x for x in remaining if x != dropped_name]

    return found
