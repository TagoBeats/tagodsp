"""Event-based gain leveling with optional synthetic replacement.

Levels every detected event of a (separated) stream into the corridor
[target_db ... threshold_db] and optionally replaces part of its content with
shaped-noise synthesis:

  1. measure event level L (RMS, dBFS)
  2. desired level D = clamp(L, target, threshold)
  3. keep original scaled to  D * (1 - r)           (r = replacement rate)
  4. add shaped noise scaled to D * sqrt(1-(1-r)^2)  (energy-correct fill)

Extras:
  - severity-adaptive rate: peak-excess (peak - rms) through a sigmoid, so
    spiky events get near-base rate and soft events stay largely intact
  - scope vetoes: events that are too long or whose target/anchor energy
    ratio is implausible bypass processing entirely
  - voiced events (see tagodsp.analysis.voicing) get a scaled-down rate to
    preserve harmonic structure
"""

from dataclasses import dataclass, field

import numpy as np

from tagodsp.spectral.stft import band_rms
from tagodsp.synthesis.shaped_noise import shaped_noise
from tagodsp.utils.gain import db_to_lin, rms_db


@dataclass
class Event:
    s: int
    e: int
    level_db: float
    desired_db: float
    r_effective: float
    peak_excess_db: float = float("nan")
    severity: float = float("nan")
    voiced: bool = False
    vetoed: bool = False
    veto_reason: str = ""
    energy_ratio: float = float("nan")


@dataclass
class EventLeveler:
    sr: float
    threshold_db: float = -30.0
    target_db: float = -35.0
    replacement: float = 0.5          # base replacement rate 0..1
    max_boost_db: float = 15.0
    edge_ramp_ms: float = 5.0
    silence_db: float = -90.0
    # severity-adaptive rate
    severity_enabled: bool = True
    sev_min_db: float = 3.0
    sev_max_db: float = 15.0
    sev_k: float = 5.0
    # scope vetoes
    vetoes_enabled: bool = True
    max_event_ms: float = 200.0
    energy_ratio_min: float = 0.05
    energy_ratio_max: float = 10.0
    anchor_band: tuple[float, float] = (500.0, 4000.0)
    target_band: tuple[float, float] = (6000.0, 13000.0)
    # voiced path
    voiced_rate_factor: float = 0.5
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(1337))

    def plan(
        self,
        x_stream: np.ndarray,
        events: list[tuple[int, int]],
        voiced: list[bool] | None = None,
    ) -> list[Event]:
        """Build the leveling plan for sample-domain (start, end) events."""
        target_db, threshold_db = self.target_db, self.threshold_db
        if target_db > threshold_db:
            target_db, threshold_db = threshold_db, target_db
        r_base = float(np.clip(self.replacement, 0.0, 1.0))
        voiced = voiced or [False] * len(events)

        plan = []
        for (s, e), is_voiced in zip(events, voiced):
            seg = x_stream[s:e]
            lvl = rms_db(seg)
            if lvl <= self.silence_db:
                continue
            desired = float(np.clip(lvl, target_db, threshold_db))
            desired = min(desired, lvl + self.max_boost_db)

            if self.severity_enabled:
                peak_exc, sev_norm, r_eff = self._severity_rate(seg, lvl, r_base)
            else:
                peak_exc, sev_norm, r_eff = float("nan"), float("nan"), r_base
            if is_voiced:
                r_eff *= self.voiced_rate_factor

            plan.append(
                Event(
                    s=s, e=e, level_db=lvl, desired_db=desired,
                    r_effective=r_eff, peak_excess_db=peak_exc,
                    severity=sev_norm, voiced=is_voiced,
                )
            )
        return plan

    def apply_vetoes(
        self,
        plan: list[Event],
        S_mag: np.ndarray,
        freqs: np.ndarray,
        hop: int,
    ) -> int:
        """Mark implausible events for bypass. Returns number of vetoed events."""
        if not self.vetoes_enabled:
            return 0
        n_frames = S_mag.shape[1]
        n_vetoed = 0
        for ev in plan:
            duration_ms = (ev.e - ev.s) / self.sr * 1000.0
            if duration_ms > self.max_event_ms:
                ev.vetoed = True
                ev.veto_reason = f"duration {duration_ms:.0f}ms > {self.max_event_ms:.0f}"
                n_vetoed += 1
                continue
            s_fr = ev.s // hop
            e_fr = min(ev.e // hop + 1, n_frames)
            if e_fr <= s_fr:
                ev.vetoed = True
                ev.veto_reason = "empty frame range"
                n_vetoed += 1
                continue
            S_slice = S_mag[:, s_fr:e_fr]
            anchor = band_rms(S_slice, freqs, *self.anchor_band)
            target = band_rms(S_slice, freqs, *self.target_band)
            ratio = float(target.mean() / (anchor.mean() + 1e-24))
            ev.energy_ratio = ratio
            if ratio < self.energy_ratio_min:
                ev.vetoed = True
                ev.veto_reason = f"weak ratio {ratio:.3f} < {self.energy_ratio_min}"
                n_vetoed += 1
            elif ratio > self.energy_ratio_max:
                ev.vetoed = True
                ev.veto_reason = f"extreme ratio {ratio:.1f} > {self.energy_ratio_max:.0f}"
                n_vetoed += 1
        return n_vetoed

    def render(self, x_stream: np.ndarray, plan: list[Event]) -> np.ndarray:
        """Apply the plan: per-event gain plus energy-correct synthetic fill."""
        x_stream = np.asarray(x_stream, dtype=np.float64)
        n = len(x_stream)
        gain_env = np.ones(n)
        synth_sum = np.zeros(n)
        for ev in plan:
            if ev.vetoed:
                continue
            s, e = ev.s, ev.e
            seg = x_stream[s:e]
            keep_frac = 1.0 - ev.r_effective
            synth_frac = float(np.sqrt(max(0.0, 1.0 - keep_frac**2)))
            ev_rms = float(db_to_lin(ev.level_db))
            desired_rms = float(db_to_lin(ev.desired_db))

            keep_gain = (desired_rms * keep_frac) / max(ev_rms, 1e-12)
            w = self._edge_ramp(e - s)
            gain_env[s:e] = 1.0 + w * (keep_gain - 1.0)

            if synth_frac > 0.0:
                synth = shaped_noise(seg, self.rng)
                s_rms = float(np.sqrt(np.mean(synth**2) + 1e-24))
                synth *= (desired_rms * synth_frac) / max(s_rms, 1e-12)
                synth_sum[s:e] += synth * w

        return x_stream * gain_env + synth_sum

    def _severity_rate(self, seg: np.ndarray, level_db: float, base_rate: float):
        peak_db = 20.0 * np.log10(np.max(np.abs(seg)) + 1e-12)
        peak_excess = peak_db - level_db
        sev_norm = float(
            np.clip((peak_excess - self.sev_min_db) / (self.sev_max_db - self.sev_min_db), 0.0, 1.0)
        )
        sigmoid = 1.0 / (1.0 + np.exp(-self.sev_k * (sev_norm - 0.5)))
        return float(peak_excess), sev_norm, base_rate * float(sigmoid)

    def _edge_ramp(self, n: int) -> np.ndarray:
        """Raised-cosine fade-in/out envelope of length n."""
        ramp_n = min(int(self.edge_ramp_ms * 1e-3 * self.sr), max(1, n // 4))
        env = np.ones(n)
        if ramp_n > 1:
            t = np.linspace(0.0, np.pi / 2, ramp_n)
            env[:ramp_n] = np.sin(t) ** 2
            env[-ramp_n:] = np.cos(t) ** 2
        return env
