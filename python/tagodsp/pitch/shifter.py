"""Pitch shifting with independent formant control via signalsmith-stretch.

Source: Signalsmith Stretch (https://github.com/Signalsmith-Audio/signalsmith-stretch),
design writeup: https://signalsmith-audio.co.uk/writing/2023/stretch-design/.
Bindings: vendored python-stretch (third_party/python-stretch), patched to expose
the formant methods. Concept note: docs/concepts/pitch_shift_formant_signalsmith.md.

The engine's process() is offline: it compensates its own input/output latency
(seek + flush) and resets internal state after every call, so one call renders
one complete, time-aligned signal. That alignment is what makes the dry/wet mix
a plain crossfade here. The realtime plugin port will use the C++ streaming API
instead.
"""

import numpy as np

from tagodsp.utils.gain import db_to_lin

try:
    import python_stretch
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "python-stretch not installed. Run `uv sync` (vendored in third_party/python-stretch)."
    ) from e

# Wet level loss per semitone, measured on the two v1 test vocals (2026-07-13,
# RMS wet vs dry, mean of male hook and Toms Diner). Index 0 = -12 st, step 1.
# Compensating the wet path keeps all transpositions at roughly equal loudness;
# exactly 0 dB at pitch 0 so the neutral setting stays transparent.
_LEVEL_COMP_DB = (
    -0.9, -0.2, 0.2, 0.3, 0.6, 0.6, 1.1, 1.3, 1.5, 1.6, 1.6, 1.0, 0.0,
    1.3, 2.4, 3.1, 3.6, 4.1, 4.6, 4.6, 5.4, 5.8, 6.2, 5.9, 5.4,
)


def _level_comp_db(pitch_semitones: float) -> float:
    """Linear interpolation into the measured table, clamped to +-12."""
    p = float(np.clip(pitch_semitones, -12.0, 12.0))
    return float(np.interp(p, np.arange(-12, 13), _LEVEL_COMP_DB))


class PitchShifter:
    """Monophonic/stereo vocal pitcher: pitch, formant, mix (Little-AlterBoy-style).

    Parameters
    ----------
    sr : sample rate in Hz, explicit as everywhere in tagodsp.
    pitch_semitones : transposition, typically -12..+12.
    formant_semitones : formant shift on top of (independent of) the pitch shift.
    mix : dry/wet crossfade 0..1 (1 = fully wet).
    gain_db : output gain in dB, applied after the mix (plugin range is +-18).
    preserve_formants : keep the spectral envelope in place when transposing.
        True is the AlterBoy behavior (pitch and formant independent); False
        lets formants follow the pitch (classic transpose sound).
    tonality_limit_hz : upper limit for harmonic mapping inside the engine.
        Signalsmith suggests ~8 kHz for voice, but listening tests on sung
        vocals (2026-07-13, up5/up12 male hook) picked 12 kHz: audibly less
        grainy, no regressions on female material or downshifts.
    formant_base_hz : anchor for the engine's envelope estimate. 0 = automatic;
        setting it near the singer's fundamental can improve formant tracking.
    level_compensation : compensate the measured wet level loss per semitone
        (see _LEVEL_COMP_DB) so all transpositions sit at similar loudness.
        The table was measured with the default 120 ms block; treat it as an
        approximation for custom block sizes.
    block_s / interval_s : STFT block and hop in seconds. None = engine preset
        (presetDefault: block 0.12 s, interval 0.03 s). Shorter blocks smear
        transients less and gurgle less on dense material, at the cost of
        rougher low end. Set both or neither.
    """

    def __init__(
        self,
        sr: float,
        pitch_semitones: float = 0.0,
        formant_semitones: float = 0.0,
        mix: float = 1.0,
        gain_db: float = 0.0,
        preserve_formants: bool = True,
        tonality_limit_hz: float = 12000.0,
        formant_base_hz: float = 0.0,
        level_compensation: bool = True,
        block_s: float | None = None,
        interval_s: float | None = None,
    ):
        if sr <= 0:
            raise ValueError("sr must be > 0")
        if (block_s is None) != (interval_s is None):
            raise ValueError("set block_s and interval_s together or neither")
        self.sr = float(sr)
        self.pitch_semitones = float(pitch_semitones)
        self.formant_semitones = float(formant_semitones)
        self.mix = mix
        self.gain_db = float(gain_db)
        self.preserve_formants = bool(preserve_formants)
        self.tonality_limit_hz = float(tonality_limit_hz)
        self.formant_base_hz = float(formant_base_hz)
        self.level_compensation = bool(level_compensation)
        self.block_s = None if block_s is None else float(block_s)
        self.interval_s = None if interval_s is None else float(interval_s)
        self._engine = python_stretch.Signalsmith.Stretch()
        self._channels = 0

    @property
    def mix(self) -> float:
        return self._mix

    @mix.setter
    def mix(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("mix must be in [0, 1]")
        self._mix = float(value)

    @property
    def latency_samples(self) -> int:
        """Total engine latency (input + output) in samples.

        The offline process() already compensates it; this is the number the
        realtime plugin will have to report to the host.
        """
        self._prepare(self._channels or 1)
        return int(self._engine.inputLatency() + self._engine.outputLatency())

    def _prepare(self, channels: int) -> None:
        if channels != self._channels:
            if self.block_s is None:
                self._engine.preset(channels, self.sr)
            else:
                self._engine.configure(
                    channels,
                    int(round(self.block_s * self.sr)),
                    int(round(self.interval_s * self.sr)),
                )
            self._channels = channels

    def reset(self) -> None:
        """Reset engine state. process() also resets after every call."""
        if self._channels:
            self._engine.reset()

    def process(self, x: np.ndarray) -> np.ndarray:
        """Render one complete signal, same shape and length as the input.

        Accepts mono (n,) or multichannel (channels, n). Output is float32,
        time-aligned with the input (engine latency is compensated internally).
        """
        x = np.asarray(x, dtype=np.float32)
        mono = x.ndim == 1
        buf = x[np.newaxis, :] if mono else x
        if buf.ndim != 2:
            raise ValueError("expected (n,) or (channels, n)")

        self._prepare(buf.shape[0])
        self._engine.setTransposeSemitones(
            self.pitch_semitones, self.tonality_limit_hz / self.sr
        )
        self._engine.setFormantSemitones(self.formant_semitones, self.preserve_formants)
        self._engine.setFormantBase(self.formant_base_hz)

        wet = self._engine.process(np.ascontiguousarray(buf))
        n = buf.shape[1]
        if wet.shape[1] < n:  # guard against off-by-one from the engine's rounding
            wet = np.pad(wet, ((0, 0), (0, n - wet.shape[1])))
        wet = wet[:, :n]

        if self.level_compensation:
            wet = wet * db_to_lin(_level_comp_db(self.pitch_semitones))

        y = (1.0 - self._mix) * buf + self._mix * wet
        y = (y * db_to_lin(self.gain_db)).astype(np.float32)
        return y[0] if mono else y
