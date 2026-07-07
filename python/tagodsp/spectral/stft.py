"""STFT wrapper with perfect reconstruction.

Forward: Hann-windowed frames, rfft, center padding (frame k centered on
sample k*hop). Inverse: weighted overlap-add normalized by the summed squared
window (see Griffin & Lim 1984; JOS, "Spectral Audio Signal Processing").
Hann with hop = n_fft/4 satisfies COLA.
"""

import numpy as np


class Stft:
    def __init__(self, n_fft: int = 1024, hop: int = 256):
        if n_fft <= 0 or hop <= 0 or hop > n_fft:
            raise ValueError("need 0 < hop <= n_fft")
        self.n_fft = n_fft
        self.hop = hop
        self.window = np.hanning(n_fft + 1)[:-1].astype(np.float64)  # periodic Hann

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    def freqs(self, sr: float) -> np.ndarray:
        """Center frequency of each bin in Hz."""
        return np.fft.rfftfreq(self.n_fft, 1.0 / sr)

    def times(self, n_frames: int, sr: float) -> np.ndarray:
        """Center time of each frame in seconds."""
        return np.arange(n_frames) * self.hop / sr

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Complex STFT of a 1D signal, shape (n_bins, n_frames)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("expected 1D input")
        pad = self.n_fft // 2
        xp = np.pad(x, (pad, pad))
        n_frames = 1 + (len(xp) - self.n_fft) // self.hop
        idx = np.arange(self.n_fft)[None, :] + self.hop * np.arange(n_frames)[:, None]
        frames = xp[idx] * self.window[None, :]
        return np.fft.rfft(frames, axis=1).T

    def inverse(self, S: np.ndarray, length: int) -> np.ndarray:
        """Inverse STFT via weighted overlap-add, trimmed to `length` samples."""
        S = np.asarray(S)
        n_frames = S.shape[1]
        frames = np.fft.irfft(S.T, n=self.n_fft, axis=1)
        out_len = self.n_fft + self.hop * (n_frames - 1)
        y = np.zeros(out_len)
        norm = np.zeros(out_len)
        w = self.window
        w2 = w * w
        for k in range(n_frames):
            s = k * self.hop
            y[s : s + self.n_fft] += frames[k] * w
            norm[s : s + self.n_fft] += w2
        y = y / np.maximum(norm, 1e-12)
        pad = self.n_fft // 2
        return y[pad : pad + length]


def band_rms(S_mag: np.ndarray, freqs: np.ndarray, lo_hz: float, hi_hz: float) -> np.ndarray:
    """Per-frame RMS of magnitude bins within [lo_hz, hi_hz)."""
    lo = int(np.searchsorted(freqs, lo_hz))
    hi = int(np.searchsorted(freqs, hi_hz))
    if hi <= lo:
        raise ValueError("empty band")
    return np.sqrt(np.mean(np.abs(S_mag[lo:hi, :]) ** 2, axis=0) + 1e-24)
