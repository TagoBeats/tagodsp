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

try:
    import python_stretch
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "python-stretch not installed. Run `uv sync` (vendored in third_party/python-stretch)."
    ) from e


class PitchShifter:
    """Monophonic/stereo vocal pitcher: pitch, formant, mix (Little-AlterBoy-style).

    Parameters
    ----------
    sr : sample rate in Hz, explicit as everywhere in tagodsp.
    pitch_semitones : transposition, typically -12..+12.
    formant_semitones : formant shift on top of (independent of) the pitch shift.
    mix : dry/wet crossfade 0..1 (1 = fully wet).
    preserve_formants : keep the spectral envelope in place when transposing.
        True is the AlterBoy behavior (pitch and formant independent); False
        lets formants follow the pitch (classic transpose sound).
    tonality_limit_hz : upper limit for harmonic mapping inside the engine,
        Signalsmith recommends ~8 kHz for voice.
    formant_base_hz : anchor for the engine's envelope estimate. 0 = automatic;
        setting it near the singer's fundamental can improve formant tracking.
    """

    def __init__(
        self,
        sr: float,
        pitch_semitones: float = 0.0,
        formant_semitones: float = 0.0,
        mix: float = 1.0,
        preserve_formants: bool = True,
        tonality_limit_hz: float = 8000.0,
        formant_base_hz: float = 0.0,
    ):
        if sr <= 0:
            raise ValueError("sr must be > 0")
        self.sr = float(sr)
        self.pitch_semitones = float(pitch_semitones)
        self.formant_semitones = float(formant_semitones)
        self.mix = mix
        self.preserve_formants = bool(preserve_formants)
        self.tonality_limit_hz = float(tonality_limit_hz)
        self.formant_base_hz = float(formant_base_hz)
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
            self._engine.preset(channels, self.sr)
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

        y = (1.0 - self._mix) * buf + self._mix * wet
        y = y.astype(np.float32)
        return y[0] if mono else y
