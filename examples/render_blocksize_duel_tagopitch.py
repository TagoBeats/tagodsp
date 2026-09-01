"""TagoPitch block-size duel: 120 ms preset (A) vs 60 ms blocks (B).

Same pitch settings through the Python reference engine, only the STFT block
changes (interval stays at block/4 like the preset). Hypothesis: shorter
blocks smear transients less and reduce the watery gurgle on dense material.

All bounces are RMS-matched to the dry input so level differences don't skew
the A/B. No deltas: two independently pitched renders never null against each
other, the difference file would just contain both signals.

Run:  uv run python examples/render_blocksize_duel_tagopitch.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from listenpack import ListenPack

from tagodsp.pitch.shifter import PitchShifter

TEST_AUDIO = Path("/Users/admin/Documents/robinbusse.dev/TestAudio")
MAX_DUR_S = 12.0

# (source name, file, pitch semitones)
ITEMS = [
    ("vocal_up3", "T_VA_130_vocal_hook_loop_ride_male_Dmin.wav", 3.0),
    ("vocal_up12", "T_VA_130_vocal_hook_loop_ride_male_Dmin.wav", 12.0),
    ("vocal_down12", "T_VA_130_vocal_hook_loop_ride_male_Dmin.wav", -12.0),
    ("dense_mix_up3", "OS_LLS_130_A#m_Fall_In_Love.wav", 3.0),
    ("dense_mix_up12", "OS_LLS_130_A#m_Fall_In_Love.wav", 12.0),
]

# label -> (block_s, interval_s); None = presetDefault (0.12 / 0.03)
CONFIGS = {"block120": (None, None), "block60": (0.06, 0.015)}


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))) + 1e-20)


def render(x: np.ndarray, sr: int, pitch: float, block_s, interval_s) -> np.ndarray:
    shifter = PitchShifter(
        sr=sr, pitch_semitones=pitch, mix=1.0, block_s=block_s, interval_s=interval_s
    )
    return shifter.process(x)


def spectrogram_stack(pack: ListenPack, name: str, signals: dict[str, np.ndarray], sr: int) -> None:
    fig, axes = plt.subplots(len(signals), 1, figsize=(10, 3.2 * len(signals)), sharex=True)
    for ax, (label, x) in zip(axes, signals.items()):
        ax.specgram(
            x.mean(axis=0), NFFT=2048, Fs=sr, noverlap=1536, cmap="magma", vmin=-140, vmax=-30
        )
        ax.set_ylabel("Hz")
        ax.set_title(label, fontsize=10, loc="left")
        ax.set_ylim(0, 16000)
    axes[-1].set_xlabel("Zeit [s]")
    fig.suptitle(f"{name}: Block-Duell 120 ms vs 60 ms")
    pack.add_plot(f"{name}_spectrogram", fig)


def main() -> None:
    pack = ListenPack("tagopitch_blocksize_duel")
    settings = [
        "TagoPitch Block-Size-Duell (Python-Referenz-Engine, mix 100 %, formant 0)",
        "A = block120: presetDefault (block 0.12 s, interval 0.03 s)",
        "B = block60:  configure(block 0.06 s, interval 0.015 s)",
        "Alle Bounces RMS-gematcht auf das Dry-Input-Level.",
        "",
    ]

    for name, fname, pitch in ITEMS:
        x, sr = sf.read(TEST_AUDIO / fname, dtype="float32", always_2d=True)
        x = np.ascontiguousarray(x[: int(MAX_DUR_S * sr)].T)  # (channels, n)
        dry_rms = rms(x)

        rendered = {}
        gains_db = {}
        for label, (block_s, interval_s) in CONFIGS.items():
            y = render(x, sr, pitch, block_s, interval_s)
            g = dry_rms / rms(y)
            rendered[label] = (y * g).astype(np.float32)
            gains_db[label] = 20 * np.log10(g)

        a, b = rendered["block120"], rendered["block60"]
        pack.add_pair(name, a.T, b.T, sr, label_a="block120", label_b="block60")
        spectrogram_stack(pack, name, {"dry": x, "block120": a, "block60": b}, sr)

        line = (
            f"{name}: {fname} ({x.shape[1] / sr:.1f} s), pitch {pitch:+.0f} st | "
            f"RMS-Match block120 {gains_db['block120']:+.1f} dB, "
            f"block60 {gains_db['block60']:+.1f} dB"
        )
        settings.append(line)
        print(line)

    (pack.dir / "settings.txt").write_text("\n".join(settings) + "\n")
    page = pack.audition_page("TagoPitch Block-Duell: 120 ms (A) vs 60 ms (B)")
    print(f"\npack: {pack.dir.resolve()}")
    print(f"audition: {page.resolve()}")


if __name__ == "__main__":
    main()
