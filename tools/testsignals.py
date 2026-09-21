"""Test material shared by the clipper measurement scripts.

isp_sweep and chain_ceiling_check answer the same question, once without and
once with the limiter, and their reports are only comparable while the material
is identical. Keeping the generator in one place is what makes that true rather
than merely intended.
"""

import numpy as np


def clipper_suite(sr: int, seconds: float, seed: int = 2026) -> dict[str, np.ndarray]:
    """Six signals spanning the cases where inter-sample peaks differ."""
    n = int(sr * seconds)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)

    burst = np.sin(2 * np.pi * 60.0 * t) * np.exp(-t * 12.0)
    for hit in (0.25, 0.5, 0.75):
        start = int(hit * n)
        tail = np.arange(n - start) / sr
        burst[start:] += np.sin(2 * np.pi * 3000.0 * tail) * np.exp(-tail * 60.0)

    return {
        "sine_1k": np.sin(2 * np.pi * 1000.0 * t),
        "sine_5k": np.sin(2 * np.pi * 5000.0 * t),
        "sine_11k": np.sin(2 * np.pi * 11000.0 * t),
        "twotone_11k_12k": 0.5 * (np.sin(2 * np.pi * 11000 * t) + np.sin(2 * np.pi * 12000 * t)),
        "drum_ish": burst / np.max(np.abs(burst)),
        "noise": rng.standard_normal(n) * 0.25,
    }
