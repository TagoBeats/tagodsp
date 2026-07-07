"""Sigmoid soft-mask separation with perfect reconstruction.

Splits a signal into a "masked" stream (bins whose energy exceeds a per-frame
anchor reference) and the complementary "rest" stream:

  mask(f, t) = sigmoid((excess_db(f, t) - threshold_db) / temperature_db)
  X_masked = X * mask,  X_rest = X * (1 - mask)

mask + (1 - mask) = 1, so rest + masked reconstructs the input exactly (up to
STFT round-trip error). A detector gate, smoothed by an asymmetric envelope,
confines the mask to detected events so everything else passes untouched.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from tagodsp.analysis.envelope import EnvelopeFollower
from tagodsp.spectral.stft import Stft, band_rms


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


@dataclass
class SeparationResult:
    x_rest: np.ndarray
    x_masked: np.ndarray
    mask: np.ndarray
    peak_delta_dbfs: float   # reconstruction error, should be deeply negative


@dataclass
class SoftMaskSeparator:
    sr: float
    anchor_band: tuple[float, float] = (500.0, 4000.0)
    mask_band: tuple[float, float] = (4000.0, 20000.0)
    threshold_db: float = 0.0
    temperature_db: float = 3.0
    freq_smooth_bins: float = 3.0
    time_smooth_frames: float = 2.0
    gate_attack_ms: float = 1.0
    gate_release_ms: float = 50.0
    stft: Stft = field(default_factory=lambda: Stft(n_fft=1024, hop=256))

    def separate(self, x: np.ndarray, gate: np.ndarray) -> SeparationResult:
        x = np.asarray(x, dtype=np.float64)
        S = self.stft.forward(x)
        S_mag = np.abs(S)
        freqs = self.stft.freqs(self.sr)
        n_frames = S.shape[1]

        gate = np.asarray(gate, dtype=np.float64)
        if len(gate) < n_frames:
            gate = np.pad(gate, (0, n_frames - len(gate)))
        gate = gate[:n_frames]
        frame_rate = self.sr / self.stft.hop
        follower = EnvelopeFollower(
            self.gate_attack_ms, self.gate_release_ms, rate=frame_rate, mode="peak"
        )
        gate_env = follower.process(gate)

        mask = self._build_mask(S_mag, freqs, gate_env)

        x_masked = self.stft.inverse(S * mask, len(x))
        x_rest = self.stft.inverse(S * (1.0 - mask), len(x))

        delta = x - (x_rest + x_masked)
        peak = float(np.max(np.abs(delta))) if delta.size else 0.0
        peak_dbfs = 20.0 * np.log10(peak) if peak > 0 else -np.inf

        return SeparationResult(
            x_rest=x_rest, x_masked=x_masked, mask=mask, peak_delta_dbfs=peak_dbfs
        )

    def _build_mask(
        self, S_mag: np.ndarray, freqs: np.ndarray, gate_env: np.ndarray
    ) -> np.ndarray:
        anchor_rms = band_rms(S_mag, freqs, *self.anchor_band)
        anchor_db = 20.0 * np.log10(anchor_rms)
        bin_db = 20.0 * np.log10(S_mag + 1e-24)
        excess = bin_db - anchor_db[np.newaxis, :]

        mask = _sigmoid((excess - self.threshold_db) / max(self.temperature_db, 0.1))

        lo_bin = int(np.searchsorted(freqs, self.mask_band[0]))
        hi_bin = int(np.searchsorted(freqs, min(self.mask_band[1], freqs[-1])))
        mask[:lo_bin, :] = 0.0
        if hi_bin < mask.shape[0]:
            mask[hi_bin:, :] = 0.0

        if self.freq_smooth_bins > 0 or self.time_smooth_frames > 0:
            mask = gaussian_filter(
                mask,
                sigma=(self.freq_smooth_bins, self.time_smooth_frames),
                mode="constant",
                cval=0.0,
            )

        mask = mask * gate_env[np.newaxis, :]
        np.clip(mask, 0.0, 1.0, out=mask)
        return mask
