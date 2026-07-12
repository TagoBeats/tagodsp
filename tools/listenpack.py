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

    def audition_page(self, title: str = "Listenpack") -> Path:
        """Write an audition.html web A/B player into the pack folder.

        One row per pair in audio/; switching A/B keeps the playback position.
        Robin's preferred way to run listening tests (2026-07-11).
        """
        import html as html_mod
        import re

        pairs = {}
        for f in sorted(self.audio_dir.glob("*_A_*.wav")):
            item = re.match(r"(.+)_A_", f.name).group(1)
            b = list(self.audio_dir.glob(f"{item}_B_*.wav"))
            if b:
                pairs[item] = (f.name, b[0].name)

        rows = "\n".join(
            f'<div class="row"><span>{html_mod.escape(item)}</span>'
            f'<button data-src="audio/{html_mod.escape(a)}">A</button>'
            f'<button data-src="audio/{html_mod.escape(b)}">B</button></div>'
            for item, (a, b) in pairs.items()
        )
        page = f"""<!doctype html>
<meta charset="utf-8">
<title>{html_mod.escape(title)}</title>
<style>
  body {{ font: 15px/1.5 -apple-system, sans-serif; max-width: 640px;
         margin: 3rem auto; color: #ddd; background: #16181c; }}
  h1 {{ font-size: 1.1rem; }} p {{ color: #888; }}
  .row {{ display: flex; gap: .6rem; align-items: center; padding: .35rem 0;
          border-bottom: 1px solid #24272d; }}
  .row span {{ flex: 1; }}
  button {{ background: #24272d; color: #ddd; border: 0; border-radius: 4px;
            padding: .3rem .9rem; cursor: pointer; }}
  button.playing {{ background: #3a7bd5; color: #fff; }}
</style>
<h1>{html_mod.escape(title)}</h1>
<p>A/B wechselt an gleicher Stelle im File. Nochmal klicken stoppt.</p>
{rows}
<script>
  const player = new Audio();
  let active = null;
  document.querySelectorAll("button").forEach(btn => {{
    btn.addEventListener("click", () => {{
      if (active === btn) {{ player.pause(); mark(null); return; }}
      const sameItem = active && active.closest(".row") === btn.closest(".row");
      const t = sameItem ? player.currentTime : 0;
      player.src = btn.dataset.src;
      player.currentTime = t;
      player.play();
      mark(btn);
    }});
  }});
  player.addEventListener("ended", () => mark(null));
  function mark(btn) {{
    document.querySelectorAll("button.playing").forEach(b => b.classList.remove("playing"));
    if (btn) btn.classList.add("playing");
    active = btn;
  }}
</script>
"""
        out = self.dir / "audition.html"
        out.write_text(page)
        return out
