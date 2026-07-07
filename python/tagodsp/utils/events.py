"""Boolean gate to event conversion."""

import numpy as np


def gate_edges(gate: np.ndarray) -> list[tuple[int, int]]:
    """Rising/falling edges of a boolean gate as (start, end) frame pairs."""
    g = np.asarray(gate).astype(np.int8)
    edges = np.diff(g)
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if g.size and g[0]:
        starts = [0] + starts
    if g.size and g[-1]:
        ends = ends + [len(g)]
    return list(zip(starts, ends))


def gate_to_events(
    gate: np.ndarray,
    n_samples: int,
    sr: float,
    hop: int,
    min_ms: float = 0.0,
    tail_samples: int = 0,
) -> list[tuple[int, int]]:
    """Frame-domain gate to sample-domain (start, end) events.

    tail_samples extends each event (e.g. n_fft//2 to cover the analysis
    window tail). Events shorter than min_ms are dropped.
    """
    min_len = int(min_ms * 1e-3 * sr)
    events = []
    for fs, fe in gate_edges(gate):
        s = min(fs * hop, n_samples)
        e = min(fe * hop + tail_samples, n_samples)
        if e - s >= min_len:
            events.append((s, e))
    return events
