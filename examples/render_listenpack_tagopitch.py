"""Render the TagoPitch listening pack: signalsmith-stretch on real vocals.

Variants per source: pitch +-3/+-12 st (formant-preserved), formant-only +-4 st,
octave down classic vs formant-corrected (deep voice), +12 at 50% mix.
Writes an audition.html into the pack for quick A/B in the browser; the
adjacent wav pairs still work for Finder Quick-Look A/B.

Run:  uv run python examples/render_listenpack_tagopitch.py [sources ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from listenpack import ListenPack

from tagodsp.pitch.shifter import PitchShifter

DEFAULT_SOURCES = [
    Path("/Users/admin/Documents/robinbusse.dev/TestAudio/T_VA_130_vocal_hook_loop_ride_male_Dmin.wav"),
    Path("/Users/admin/Documents/robinbusse.dev/TestAudio/toms_diner_segment_0.wav"),
]
MAX_DUR_S = 12.0

# name -> (pitch_st, formant_st, mix, preserve_formants)
VARIANTS = {
    "up3": (3.0, 0.0, 1.0, True),
    "down3": (-3.0, 0.0, 1.0, True),
    "up12": (12.0, 0.0, 1.0, True),
    "down12_deepvoice": (-12.0, 0.0, 1.0, True),
    "down12_classic": (-12.0, 0.0, 1.0, False),
    "formant_up4": (0.0, 4.0, 1.0, True),
    "formant_down4": (0.0, -4.0, 1.0, True),
    "up12_mix50": (12.0, 0.0, 0.5, True),
}


def load(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    n = min(len(x), int(MAX_DUR_S * sr))
    return np.ascontiguousarray(x[:n].T), sr  # (channels, n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "sources", nargs="*", type=Path, default=DEFAULT_SOURCES,
        help="wav files or directories with dry vocals (default: robinbusse.dev TestAudio picks)",
    )
    sources = parser.parse_args().sources
    wavs = []
    for src in sources:
        wavs += sorted(src.glob("*.wav")) if src.is_dir() else [src]

    pack = ListenPack("tagopitch")
    items = []
    for wav in wavs:
        x, sr = load(wav)
        tag = wav.stem.split("_")[0].lower()
        for name, (pitch, formant, mix, preserve) in VARIANTS.items():
            shifter = PitchShifter(
                sr, pitch_semitones=pitch, formant_semitones=formant,
                mix=mix, preserve_formants=preserve,
            )
            y = shifter.process(x)
            peak = np.max(np.abs(y))
            trim = ""
            if peak > 10 ** (-1 / 20):  # keep playback headroom, attenuate only
                y = y * (10 ** (-1 / 20) / peak)
                trim = f" (trimmed {-(20 * np.log10(peak) + 1.0):+.1f} dB)"
            item = f"{tag}_{name}"
            pack.add_pair(item, x.T, y.T, sr)
            items.append(item)
            print(f"{item}: peak {20 * np.log10(peak + 1e-12):+.1f} dBFS{trim}, "
                  f"engine latency {shifter.latency_samples} samples")
        pack.spectrum_plot(
            f"{tag}_formants_octave_down",
            {"dry": x[0],
             "deep_voice (preserved)": PitchShifter(sr, -12.0).process(x)[0],
             "classic": PitchShifter(sr, -12.0, preserve_formants=False).process(x)[0]},
            sr,
            title=f"{tag}: octave down, formant-preserved vs classic",
        )

    page = pack.audition_page("TagoPitch Listenpack")
    print(f"\nPack rendered: {pack.dir}")
    print(f"Audition UI:   open '{page}'")


if __name__ == "__main__":
    main()
