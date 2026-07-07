"""Band-ratio gate: spectral event detector via band energy ratio.

Detection logic (developed for consonant/sibilant detection, generic for any
"target band pops out over anchor band" problem):

  anchor_db(t) = level of the anchor band (e.g. vocal body), smoothed
  target_db(t) = level of the target band (e.g. sibilance zone)
  gate(t)      = (target_db - anchor_db > offset_db) & (anchor_db > floor)

Morphological closing fills micro-gaps shorter than close_ms.
Offline detector (non-causal Gaussian smoothing).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter1d

from tagodsp.spectral.stft import Stft, band_rms


@dataclass
class GateResult:
    gate: np.ndarray          # bool per frame
    excess_db: np.ndarray
    anchor_db: np.ndarray
    target_db: np.ndarray
    times: np.ndarray
    S_mag: np.ndarray
    freqs: np.ndarray
    n_events: int
    active_pct: float


@dataclass
class BandRatioGate:
    sr: float
    anchor_band: tuple[float, float] = (500.0, 4000.0)
    target_band: tuple[float, float] = (6000.0, 13000.0)
    offset_db: float = 3.0
    smooth_ms: float = 20.0
    anchor_floor_db: float = -50.0
    close_ms: float = 20.0
    stft: Stft = field(default_factory=lambda: Stft(n_fft=1024, hop=256))

    def process(self, x: np.ndarray) -> GateResult:
        S_mag = np.abs(self.stft.forward(x))
        freqs = self.stft.freqs(self.sr)
        times = self.stft.times(S_mag.shape[1], self.sr)

        anchor = band_rms(S_mag, freqs, *self.anchor_band)
        target = band_rms(S_mag, freqs, *self.target_band)
        anchor_db_raw = 20.0 * np.log10(anchor)
        target_db = 20.0 * np.log10(target)

        smooth_frames = max(1.0, self.smooth_ms * 1e-3 * self.sr / self.stft.hop)
        anchor_db = gaussian_filter1d(anchor_db_raw, sigma=smooth_frames / 2)

        excess_db = target_db - anchor_db
        gate_raw = (excess_db > self.offset_db) & (anchor_db > self.anchor_floor_db)
        close_fr = max(1, int(round(self.close_ms * 1e-3 * self.sr / self.stft.hop)))
        gate = binary_closing(gate_raw, structure=np.ones(close_fr))

        return GateResult(
            gate=gate,
            excess_db=excess_db,
            anchor_db=anchor_db,
            target_db=target_db,
            times=times,
            S_mag=S_mag,
            freqs=freqs,
            n_events=int(np.sum(np.diff(gate.astype(int)) > 0) + int(gate[0])),
            active_pct=100.0 * float(gate.mean()),
        )
