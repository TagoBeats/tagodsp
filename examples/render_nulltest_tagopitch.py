"""TagoPitch null test: pitch 0 / mix 100 through the C++ engine vs dry.

Renders each source through the TagoPitchRender CLI (stream mode = exact
plugin signal path, latency trimmed), builds the residual wet - dry and
normalizes it to a comfortable listening level. The residual is exactly the
"noise" the engine adds at neutral settings.

Sources: one vocal, one dense mix. Run:
    uv run python examples/render_nulltest_tagopitch.py
"""

import argparse
import html as html_mod
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from listenpack import ListenPack

RENDER_BIN = Path(
    "/Users/admin/Documents/TagoPitch/build/TagoPitchRender_artefacts/Release/TagoPitchRender"
)
TEST_AUDIO = Path("/Users/admin/Documents/robinbusse.dev/TestAudio")
SOURCES = {
    "vocal": TEST_AUDIO / "T_VA_130_vocal_hook_loop_ride_male_Dmin.wav",
    "dense_mix": TEST_AUDIO / "OS_LLS_130_A#m_Fall_In_Love.wav",
}
MAX_DUR_S = 12.0
RESIDUAL_TARGET_PEAK_DB = -12.0


def render_engine(dry_path: Path, out_path: Path, pitch: float) -> None:
    """Stream mode, formant 0 / mix 100 / gain 0."""
    subprocess.run(
        [str(RENDER_BIN), "stream", str(dry_path), str(out_path), str(pitch), "0", "100", "0"],
        check=True,
        capture_output=True,
    )


def spectrogram_stack(
    pack: ListenPack, name: str, signals: dict[str, np.ndarray], sr: int, subtitle: str
) -> None:
    fig, axes = plt.subplots(len(signals), 1, figsize=(10, 3.2 * len(signals)), sharex=True)
    for ax, (label, x) in zip(axes, signals.items()):
        mono = x.mean(axis=1)
        ax.specgram(mono, NFFT=2048, Fs=sr, noverlap=1536, cmap="magma", vmin=-140, vmax=-30)
        ax.set_ylabel("Hz")
        ax.set_title(label, fontsize=10, loc="left")
        ax.set_ylim(0, 16000)
    axes[-1].set_xlabel("Zeit [s]")
    fig.suptitle(f"{name}: {subtitle} (dry vs engine vs residual)")
    pack.add_plot(f"{name}_spectrogram", fig)


def audition_page_with_deltas(pack: ListenPack) -> Path:
    """ListenPack audition page plus one solo-button row per residual in deltas/."""
    out = pack.audition_page("TagoPitch: Dry (A) vs Engine (B)")
    extra = "\n".join(
        f'<div class="row"><span>{html_mod.escape(f.stem)}</span>'
        f'<button data-src="deltas/{html_mod.escape(f.name)}">Residual</button></div>'
        for f in sorted((pack.dir / "deltas").glob("*.wav"))
    )
    page = out.read_text()
    page = page.replace("<script>", f"<h1>Residuals (normalisiert)</h1>\n{extra}\n<script>")
    out.write_text(page)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pitch", type=float, default=0.0, help="Pitch in Halbtoenen")
    parser.add_argument(
        "--pack-dir", type=Path, default=None,
        help="Bestehenden Pack-Ordner wiederverwenden statt neuen anzulegen",
    )
    args = parser.parse_args()

    pack = ListenPack("tagopitch_nulltest")
    if args.pack_dir is not None:
        pack.dir = args.pack_dir
        pack.audio_dir = pack.dir / "audio"
        pack.plot_dir = pack.dir / "plots"
    delta_dir = pack.dir / "deltas"
    for d in (pack.audio_dir, pack.plot_dir, delta_dir):
        d.mkdir(parents=True, exist_ok=True)

    pitch_tag = f"pitch{args.pitch:+.0f}".replace("+", "up").replace("-", "down")
    settings_file = pack.dir / "settings.txt"
    settings = [settings_file.read_text().rstrip()] if settings_file.exists() else [
        "TagoPitch Null-Test",
        "Engine: TagoPitchRender stream mode (exakter Plugin-Pfad, Latenz getrimmt)",
    ]
    settings.append(f"\nRun {pitch_tag}: formant 0 st, mix 100 %, gain 0 dB")

    for name, src in SOURCES.items():
        name = f"{name}_{pitch_tag}"
        x, sr = sf.read(src, dtype="float32", always_2d=True)
        x = x[: int(MAX_DUR_S * sr)]

        with tempfile.TemporaryDirectory() as tmp:
            dry_path = Path(tmp) / "dry.wav"
            wet_path = Path(tmp) / "wet.wav"
            sf.write(dry_path, x, sr)
            render_engine(dry_path, wet_path, args.pitch)
            y, _ = sf.read(wet_path, dtype="float32", always_2d=True)

        n = min(len(x), len(y))
        dry, wet = x[:n], y[:n]
        residual = wet - dry

        # Report levels relative to dry, then normalize residual for listening.
        def rms_db(s: np.ndarray) -> float:
            return 20 * np.log10(np.sqrt(np.mean(np.square(s.astype(np.float64)))) + 1e-20)

        res_rel_db = rms_db(residual) - rms_db(dry)
        peak = np.max(np.abs(residual)) + 1e-20
        norm_gain_db = RESIDUAL_TARGET_PEAK_DB - 20 * np.log10(peak)
        residual_norm = residual * (10 ** (norm_gain_db / 20))

        pack.add_pair(name, dry, wet, sr, label_a="dry", label_b="engine")
        sf.write(delta_dir / f"{name}_residual_norm.wav", residual_norm, sr)

        spectrogram_stack(
            pack, name,
            {"dry": dry, f"engine {pitch_tag} / mix 100": wet,
             f"residual (roh, {res_rel_db:+.1f} dB rel. dry)": residual},
            sr,
            subtitle=f"{pitch_tag} / mix 100",
        )
        pack.spectrum_plot(
            f"{name}_spectrum",
            {
                "dry": dry.mean(axis=1),
                "engine": wet.mean(axis=1),
                "residual": residual.mean(axis=1),
            },
            sr,
            title=f"{name}: Spektrum dry vs engine vs residual ({pitch_tag}, mix 100)",
        )
        settings.append(
            f"{name}: {src.name} ({n / sr:.1f} s) | Residual-RMS {res_rel_db:+.1f} dB rel. dry"
            f" | Residual-File normalisiert mit {norm_gain_db:+.1f} dB "
            f"(Peak -> {RESIDUAL_TARGET_PEAK_DB:.0f} dBFS)"
        )
        print(settings[-1])

    settings_file.write_text("\n".join(settings) + "\n")
    page = audition_page_with_deltas(pack)
    print(f"\npack: {pack.dir.resolve()}")
    print(f"audition: {page.resolve()}")


if __name__ == "__main__":
    main()
