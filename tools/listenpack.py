"""Render listening packs for A/B comparison by ear.

Layout per pack (timestamped folder, subfolders per media type):

    listenpacks/<YYYY-MM-DD_HHMM_name>/
        audio/   <item>_A_<label_a>.wav, <item>_B_<label_b>.wav  (adjacent for
                 Finder Quick-Look A/B with arrow keys)
        plots/   comparison PNGs

Ears have the final word; plots are supporting material.
"""

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


class ListenPack:
    def __init__(self, name: str, root: str | Path = "listenpacks"):
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        self.dir = Path(root) / f"{stamp}_{name}"
        self.audio_dir = self.dir / "audio"
        self.plot_dir = self.dir / "plots"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def add_pair(
        self,
        item: str,
        a: np.ndarray,
        b: np.ndarray,
        sr: int,
        label_a: str = "dry",
        label_b: str = "wet",
    ) -> None:
        """Write an adjacent A/B wav pair."""
        sf.write(self.audio_dir / f"{item}_A_{label_a}.wav", np.asarray(a, np.float32), sr)
        sf.write(self.audio_dir / f"{item}_B_{label_b}.wav", np.asarray(b, np.float32), sr)

    def add_plot(self, name: str, fig: plt.Figure) -> None:
        fig.savefig(self.plot_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def spectrum_plot(
        self,
        name: str,
        signals: dict[str, np.ndarray],
        sr: int,
        title: str | None = None,
    ) -> None:
        """Welch-style magnitude spectrum comparison of named signals."""
        from scipy.signal import welch

        fig, ax = plt.subplots(figsize=(9, 4.5))
        for label, x in signals.items():
            f, pxx = welch(np.asarray(x, np.float64), fs=sr, nperseg=4096)
            ax.semilogx(f[1:], 10 * np.log10(pxx[1:] + 1e-20), label=label)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD [dB]")
        ax.set_title(title or name)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        self.add_plot(name, fig)
