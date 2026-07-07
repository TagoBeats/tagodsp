"""Envelope followers with asymmetric attack/release.

One-pole follower: env += coeff * (target - env), where coeff is the attack
coefficient while rising and the release coefficient while falling.
coeff = 1 - exp(-1 / (t_ms * 1e-3 * rate)). Classic peak/RMS detector
topology (see Giannoulis, Massberg & Reiss 2012, "Digital Dynamic Range
Compressor Design").
"""

import numpy as np


def _coeff(time_ms: float, rate: float) -> float:
    return 1.0 - float(np.exp(-1.0 / max(time_ms * 1e-3 * rate, 1e-9)))


class EnvelopeFollower:
    """Asymmetric peak or RMS envelope follower.

    `rate` is the update rate in Hz: the audio sample rate for sample-domain
    signals, or sr/hop for frame-domain signals (e.g. smoothing a detector
    gate into a 0..1 envelope).
    """

    def __init__(
        self,
        attack_ms: float,
        release_ms: float,
        rate: float,
        mode: str = "peak",
    ):
        if rate <= 0:
            raise ValueError("rate must be > 0")
        if mode not in ("peak", "rms"):
            raise ValueError("mode must be 'peak' or 'rms'")
        self.mode = mode
        self._a_atk = _coeff(attack_ms, rate)
        self._a_rel = _coeff(release_ms, rate)
        self._env = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("expected 1D input")
        target = x * x if self.mode == "rms" else np.abs(x)
        out = np.empty_like(target)
        env = self._env
        a_atk, a_rel = self._a_atk, self._a_rel
        for i in range(target.size):
            t = target[i]
            env += (a_atk if t > env else a_rel) * (t - env)
            out[i] = env
        self._env = env
        return np.sqrt(out) if self.mode == "rms" else out

    def reset(self) -> None:
        self._env = 0.0
