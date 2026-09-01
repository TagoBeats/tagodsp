# Zone fragmentation, measured and summarized (2026-09-01)

Last open display blocker on `contention` before the C++ port: a reported pair
does not come out as one finding but as a pile of zones. Fixed by a summary
layer above the clustering, not by touching the clustering.

Corpus: 51 stem folders under `~/Music`, first 60 s, `window_seconds = 0.2`,
`Stft(4096, 2048)`, scoring `contention`. Same corpus and settings as
`masking_scale_2026-09-01.md`.

Scripts: `examples/masking_zone_dump.py` (now dumps `conflicts` alongside
`zones`, so one pass proves both things).

## The shape of the problem

1222 zones over 51 beats, 242 reported pairs.

| | |
| --- | --- |
| zones per reported pair | median 3, mean 5.0, p90 11, max 64 |
| pairs that are a single zone | 90 of 242 (37.2 %) |

**The noted "median 4, maximum 44" is off.** The measured maximum is 64
(`hedge_Addictive Keys x hedge_Insert 5`); 44 is the second worst pair
(`Ice T_Insert 4 x Ice T_Pattern 1`). Median is 3, not 4.

## Where the split runs

All 8269 zone-vs-zone relations inside a reported pair:

| relation | count | share |
| --- | --- | --- |
| overlap in both axes | 47 | 0.6 % |
| frequency overlap, apart in time | 4864 | 58.8 % |
| time overlap, apart in frequency | 158 | 1.9 % |
| apart in both | 3200 | 38.7 % |

Median time gap between frequency-overlapping fragments: **66 windows = 13.2 s**
(p90 186, max 294).

Read: the fragments are not chopped-up neighbours that a larger clustering `gap`
could have bridged. They sit a third of a minute apart. A pair fights in one
frequency region and does it again every time the section comes back. A genuine
second region at the same time is 1.9 % of the relations.

That rules out tuning `gap` as the fix. Bridging 66 windows would mean a gap of
66, which would also swallow every real pause and make the zone's time extent
meaningless. The right move is a layer that groups over frequency and turns time
into a count.

## The summary layer

`summarize_conflicts()` in `analysis/masking.py`, reported as
`MaskingResult.conflicts`. Groups a pair's zones by overlapping frequency range
(interval merge, low to high), collapses time.

| | |
| --- | --- |
| 1222 zones | -> 289 conflicts (0.24x) |
| conflicts per reported pair | median 1, mean 1.2, p90 2, max 3 |
| pairs down to a single conflict | 201 of 242 (83.1 %) |
| occurrences per conflict | median 2, p90 10, max 44 |
| active windows per conflict | median 9, p90 79, max 298 |
| conflicts per beat | median 5, max 14 |

Worst offenders, now readable as one line each:

```
 44x  active 291w  score 0.77   159-1002 Hz  Ice T_Insert 4      x Ice T_Pattern 1
 33x  active 157w  score 0.62   100-126  Hz  hedge_Addictive Ke  x hedge_Insert 5
 32x  active 293w  score 0.75   200-796  Hz  letter_Analog Lab   x letter_Loop Auto
 29x  active 298w  score 0.79   126-502  Hz  Oh cherry_Insert 8  x Oh cherry_Pattern
 24x  active 116w  score 0.43    50-126  Hz  lo siento_KSHMR Po  x lo siento_Kontakt
```

The 64-zone pair becomes three conflicts (33x, 22x, 9x), which is the honest
answer: that pair really does fight in more than one place.

### The chaining question

The interval merge is transitive, so A 100-200, B 190-400, C 390-800 land in one
group although A and C do not overlap. Measured over the 289 groups:

| | |
| --- | --- |
| groups with no overlap common to all members | 33 (11.4 %) |
| widening over the widest member | median 0.00 oct, p90 0.00, **max 0.66** |
| group span | median 0.33 oct (one band), p90 1.00, max 2.66 |

Worst case is two bands wider than the widest member. Accepted: the alternative
is splitting one dispute into two entries whose ranges touch.

### Why `score` is the maximum

The conflict score is `max` over its members, not the mean. Two reasons:

1. It is the worst moment of the dispute, which is what a warning should carry.
2. It is the only aggregate that does not move when the same material happens to
   fragment into more or fewer zones. A mean sinks as the fragment count rises,
   which would smuggle the fragmentation back into the number.

Persistence stays a separate axis, reported as `occurrences` and
`active_windows`, so a display can rank by severity or by how much of the track
is affected without the two being mixed. `active_windows` counts covered windows
only; the silence between occurrences is not counted.

## The zone set is untouched

The layer is additive. Value diff of `masking_zone_dump.py` before and after,
keyed on the stem folder (not the folder name, three beats share a name):

- 51 beats before and after, same set
- 1222 zones before and after, 0 beats with a different zone count
- **0 deviations** in `tracks`, `band`, `freq_lo_hz`, `freq_hi_hz`, `w0`, `w1`
  and `score`

So the zone set Phase 0 validated by ear is the same one the summary groups.
