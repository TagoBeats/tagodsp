"""TagoPitch anti-gurgle knobs: block60 baseline (A) vs overlap/tonality variants (B).

Follow-up to the block-size duel (block60 won / at least tied). Cheap knobs
against phase-vocoder gurgle, all reachable through the Python binding:

    overlap8    - interval block/8 instead of block/4 (smoother phase steps)
    tonality8k  - tonality limit 8 kHz instead of 12 kHz
    combo       - both at once

All bounces RMS-matched to the dry input. No deltas (pitched renders never
null against each other).

Run:  uv run python examples/render_antigurgle_tagopitch.py
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
BLOCK_S = 0.06

# (source name, file, pitch semitones)
ITEMS = [
    ("dense_mix_up12", "OS_LLS_130_A#m_Fall_In_Love.wav", 12.0),
    ("vocal_up12", "T_VA_130_vocal_hook_loop_ride_male_Dmin.wav", 12.0),
    ("vocal_down12", "T_VA_130_vocal_hook_loop_ride_male_Dmin.wav", -12.0),
]

# label -> (interval_s, tonality_limit_hz); baseline = (block/4, 12k)
BASELINE = ("block60", BLOCK_S / 4, 12000.0)
VARIANTS = {
    "overlap8": (BLOCK_S / 8, 12000.0),
    "tonality8k": (BLOCK_S / 4, 8000.0),
    "combo": (BLOCK_S / 8, 8000.0),
}


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))) + 1e-20)


def render(x, sr, pitch, interval_s, tonality_hz):
    shifter = PitchShifter(
        sr=sr, pitch_semitones=pitch, mix=1.0,
        tonality_limit_hz=tonality_hz, block_s=BLOCK_S, interval_s=interval_s,
    )
    y = shifter.process(x)
    g = rms(x) / rms(y)
    return (y * g).astype(np.float32), 20 * np.log10(g)


def main() -> None:
    pack = ListenPack("tagopitch_antigurgle")
    base_label, base_interval, base_tonality = BASELINE
    settings = [
        "TagoPitch Anti-Gurgle-Runde (Python-Referenz-Engine, mix 100 %, formant 0)",
        f"A = {base_label}: block {BLOCK_S:.3f} s, interval {base_interval:.4f} s, tonality 12 kHz",
        "B-Varianten: " + ", ".join(
            f"{k} (interval {v[0]:.4f} s, tonality {v[1] / 1000:.0f} kHz)"
            for k, v in VARIANTS.items()
        ),
        "Alle Bounces RMS-gematcht auf das Dry-Input-Level.",
        "",
    ]

    for name, fname, pitch in ITEMS:
        x, sr = sf.read(TEST_AUDIO / fname, dtype="float32", always_2d=True)
        x = np.ascontiguousarray(x[: int(MAX_DUR_S * sr)].T)

        base, g_base = render(x, sr, pitch, base_interval, base_tonality)
        gains = [f"{base_label} {g_base:+.1f} dB"]
        spectro = {"dry": x, base_label: base}

        for label, (interval_s, tonality_hz) in VARIANTS.items():
            y, g = render(x, sr, pitch, interval_s, tonality_hz)
            pack.add_pair(f"{name}_{label}", base.T, y.T, sr, label_a=base_label, label_b=label)
            spectro[label] = y
            gains.append(f"{label} {g:+.1f} dB")

        fig, axes = plt.subplots(len(spectro), 1, figsize=(10, 2.8 * len(spectro)), sharex=True)
        for ax, (label, s) in zip(axes, spectro.items()):
            ax.specgram(
                s.mean(axis=0), NFFT=2048, Fs=sr, noverlap=1536, cmap="magma", vmin=-140, vmax=-30
            )
            ax.set_ylabel("Hz")
            ax.set_title(label, fontsize=10, loc="left")
            ax.set_ylim(0, 16000)
        axes[-1].set_xlabel("Zeit [s]")
        fig.suptitle(f"{name}: Anti-Gurgle-Varianten (pitch {pitch:+.0f} st)")
        pack.add_plot(f"{name}_spectrogram", fig)

        line = f"{name}: {fname}, pitch {pitch:+.0f} st | RMS-Match " + ", ".join(gains)
        settings.append(line)
        print(line)

    (pack.dir / "settings.txt").write_text("\n".join(settings) + "\n")
    page = pack.audition_page("TagoPitch Anti-Gurgle: block60 (A) vs Variante (B)")
    print(f"\npack: {pack.dir.resolve()}")
    print(f"audition: {page.resolve()}")


if __name__ == "__main__":
    main()
