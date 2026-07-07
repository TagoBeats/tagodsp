"""Per-event voiced/voiceless classification via pYIN.

Wraps librosa.pyin (Mauch & Dixon 2014, "pYIN: A fundamental frequency
estimator using probabilistic threshold distributions"). librosa is an
optional dependency: install with `tagodsp[voicing]`.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class VoicingResult:
    voiced: bool
    voiced_frac: float


def voiced_flag(
    x: np.ndarray,
    sr: float,
    hop: int,
    n_frames: int,
    fmin: float = 70.0,
    fmax: float = 400.0,
) -> np.ndarray:
    """Boolean voiced flag per frame, length-aligned to n_frames."""
    import librosa

    _, flag, _ = librosa.pyin(
        np.asarray(x, dtype=np.float32), fmin=fmin, fmax=fmax, sr=sr, hop_length=hop
    )
    flag = np.asarray(flag, dtype=bool)
    if len(flag) < n_frames:
        flag = np.pad(flag, (0, n_frames - len(flag)))
    return flag[:n_frames]


def classify_events(
    x: np.ndarray,
    events: list[tuple[int, int]],
    sr: float,
    hop: int,
    n_frames: int,
    threshold: float = 0.5,
    fmin: float = 70.0,
    fmax: float = 400.0,
) -> list[VoicingResult]:
    """Classify sample-domain (start, end) events as voiced/voiceless.

    An event is voiced when the mean voiced-flag fraction over its frame
    span reaches `threshold`.
    """
    flag = voiced_flag(x, sr, hop, n_frames, fmin=fmin, fmax=fmax)
    results = []
    for s, e in events:
        s_fr = s // hop
        e_fr = min(e // hop + 1, len(flag))
        if e_fr <= s_fr:
            results.append(VoicingResult(voiced=False, voiced_frac=0.0))
            continue
        frac = float(flag[s_fr:e_fr].mean())
        results.append(VoicingResult(voiced=frac >= threshold, voiced_frac=frac))
    return results
