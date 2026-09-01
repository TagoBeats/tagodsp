"""Blind A/B listening pack: mud (parity) vs burial, does contour loss differ.

On 2026-08-31 X-Ray was scoped to report "mud" (two similarly loud elements sharing a
band) and deliberately not "burial" (one element buried under a much louder one), on
the assumption that total burial is audible without a tool anyway. That assumption
carries the whole product and had never been tested. A prior listening test
(render_listenpack_masking_pairs.py) already contradicted it.

This script asks exactly one question: does an element lose more contour when it
collides with an equally loud track, or when it sits under a much louder one.

Selection is by measured level ratio only, never by what a detector reports. For every
beat, one MaskingDetector.analyze() run (scoring="relative", any scoring works here
since band_power does not depend on it) gives band_power per track. For every ordered
pair (target, masker):

  1. k_star = the target's own loudest band (argmax over bands of the target's summed
     band power)
  2. delta_db = 10*log10(masker energy at k_star / target energy at k_star); positive
     means the masker is louder than the target in the target's own band
  3. cell: "parity" if abs(delta_db) <= PARITY_MAX_ABS_DB, "buried" if delta_db is
     between BURIED_MIN_DB and BURIED_MAX_DB, otherwise not a candidate
  4. both tracks must be audible (broadband RMS >= AUDIBLE_FLOOR_DB)
  5. "parity" is symmetric enough that both orderings of a pair can qualify; when they
     do, only the ordering with the quieter track as target is kept

The excerpt is centered on the window where min(target, masker) energy at k_star peaks,
then delta_db is recomputed over just that excerpt and the candidate is dropped if it
no longer falls in its own cell's range: the screening pass covers the whole beat, but
only the excerpt is what gets listened to.

6 "parity" and 6 "buried" positions are built, matched pairwise into 6 groups by
frequency (|log2(f_parity / f_buried)| <= MATCH_MAX_OCTAVES over k_star's band center),
preferring a match from the same beat. A parity candidate with no buried match within
the cap is dropped together with the slot it would have used, not backfilled from a 7th
candidate.

Rendering is identical to render_listenpack_masking_pairs.py, reused from there:
  A = target track alone
  B = target track plus masker track (plain sum)
There is no RMS match between A and B, only one shared clip-guard gain
(g = 0.999 / max(peak(A), peak(B)) if that peak > 0.999, else 1.0), so the target is
bit identical in A and B. 10 ms linear fades at both edges.

The pack is blind: positions are shuffled with a fixed seed and named pos_01..pos_12.
No cell, match group, beat, track name, or frequency appears in any filename,
audition.html, or the README. There are no plots.

Usage:
    uv run python examples/render_listenpack_mud_vs_burial.py \
        --corpus /tmp/masking_corpus_v3.json [--per-cell 6] [--name mud_vs_burial] \
        [--root <repo>/listenpacks] [--seed 20260901]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from listenpack import ListenPack  # noqa: E402
from masking_corpus import load_beat  # noqa: E402
from render_listenpack_masking_pairs import (  # noqa: E402
    excerpt_bounds,
    fade,
    load_pair,
    rms_dbfs,
)

from tagodsp.analysis.masking import MaskingDetector  # noqa: E402
from tagodsp.spectral.stft import Stft  # noqa: E402

CELLS = ("parity", "buried")
PARITY_MAX_ABS_DB = 4.0
BURIED_MIN_DB = 12.0
BURIED_MAX_DB = 20.0
MATCH_MAX_OCTAVES = 1.0
AUDIBLE_FLOOR_DB = -50.0
EXCERPT_SECONDS = 7.0
FADE_MS = 10.0


def compute_delta_db(masker_energy: float, target_energy: float) -> float | None:
    """10*log10(masker_energy / target_energy), or None if either side is silent.

    Silence on either side makes the ratio meaningless rather than extreme; treating
    it as "no candidate" (via classify_cell returning None for a None input) matches
    the spec's "sonst kein Kandidat" for a masker with no energy in the target's band.
    """
    if target_energy <= 0.0 or masker_energy <= 0.0:
        return None
    return float(10.0 * np.log10(masker_energy / target_energy))


def classify_cell(delta_db: float | None) -> str | None:
    if delta_db is None:
        return None
    if abs(delta_db) <= PARITY_MAX_ABS_DB:
        return "parity"
    if BURIED_MIN_DB <= delta_db <= BURIED_MAX_DB:
        return "buried"
    return None


def _screen_ordered_pair(
    beat_name: str,
    folder: Path,
    sr: int,
    target: str,
    masker: str,
    band_power: dict[str, np.ndarray],
    band_sums: dict[str, np.ndarray],
    edges: np.ndarray,
    audible_db: dict[str, float],
    window_seconds: float,
    n_windows: int,
    total_s: float,
    pair_scores: dict[tuple[str, str], dict],
) -> tuple[dict | None, str | None]:
    """Screen one ordered (target, masker) pair.

    Returns (candidate, raw_cell). raw_cell is set whenever cell assignment succeeded
    on the whole-beat screening pass, regardless of whether the excerpt re-check below
    then confirmed it; the caller uses it to count "screened" candidates separately
    from "verified" ones. candidate is None if the pair never qualified, or if it
    qualified on the full beat but the excerpt re-check moved delta_db out of its cell.
    """
    if audible_db[target] < AUDIBLE_FLOOR_DB or audible_db[masker] < AUDIBLE_FLOOR_DB:
        return None, None

    sums_t = band_sums[target]
    k_star = int(np.argmax(sums_t))
    target_energy = float(sums_t[k_star])
    masker_energy = float(band_sums[masker][k_star])
    delta_db_screening = compute_delta_db(masker_energy, target_energy)
    raw_cell = classify_cell(delta_db_screening)
    if raw_cell is None:
        return None, None

    shared = np.minimum(band_power[target][:, k_star], band_power[masker][:, k_star])
    w_center = int(np.argmax(shared))
    center_s = (w_center + 0.5) * window_seconds
    start_s, end_s = excerpt_bounds(center_s, total_s, EXCERPT_SECONDS)
    w_lo = max(0, min(int(np.floor(start_s / window_seconds)), n_windows - 1))
    w_hi = max(w_lo, min(int(np.floor((end_s - 1e-9) / window_seconds)), n_windows - 1))

    target_excerpt_energy = float(band_power[target][w_lo : w_hi + 1, k_star].sum())
    masker_excerpt_energy = float(band_power[masker][w_lo : w_hi + 1, k_star].sum())
    delta_db_excerpt = compute_delta_db(masker_excerpt_energy, target_excerpt_energy)
    if classify_cell(delta_db_excerpt) != raw_cell:
        return None, raw_cell  # screened, but the excerpt re-check rejected it

    band_center_hz = float(np.sqrt(edges[k_star] * edges[k_star + 1]))
    key = tuple(sorted((target, masker)))
    scores = pair_scores.get(key, {"relative": None, "collision": None, "contention": None})

    candidate = {
        "beat": beat_name,
        "folder": folder,
        "sr": sr,
        "a": key[0],
        "b": key[1],
        "target": target,
        "masker": masker,
        "cell": raw_cell,
        "k_star": k_star,
        "band_center_hz": band_center_hz,
        "delta_db_screening": delta_db_screening,
        "delta_db_excerpt": delta_db_excerpt,
        "rms_target_dbfs": audible_db[target],
        "rms_masker_dbfs": audible_db[masker],
        "excerpt_start_s": start_s,
        "excerpt_end_s": end_s,
        "target_excerpt_energy": target_excerpt_energy,
        "top_scores": scores,
    }
    return candidate, raw_cell


def screen_corpus(corpus: dict) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """Screen every beat in the corpus JSON for parity/buried candidates.

    One MaskingDetector.analyze() call per beat (scoring="relative"; band_power does
    not depend on the scoring method, see the module docstring). Audio is loaded via
    masking_corpus.load_beat, which already applies the sum-stem exclusion, and is
    discarded again once the beat has been screened.
    """
    window_seconds = corpus["window_seconds"]
    max_seconds = corpus["max_seconds"]
    n_fft = corpus["stft"]["n_fft"]
    hop = corpus["stft"]["hop"]

    candidates: dict[str, list[dict]] = {c: [] for c in CELLS}
    counts = {
        "parity_screened": 0,
        "buried_screened": 0,
        "parity_verified": 0,
        "buried_verified": 0,
    }

    for beat in corpus["beats"]:
        folder = Path(beat["folder"])
        sr = beat["sr"]
        loaded, _dropped, why = load_beat(folder, max_seconds)
        if loaded is None:
            print(f"  skip {beat['beat']}: {why}")
            continue
        beat_sr, tracks = loaded
        if beat_sr != sr:
            print(f"  skip {beat['beat']}: samplerate changed since the corpus scan")
            continue

        det = MaskingDetector(
            sr=sr, scoring="relative", window_seconds=window_seconds, stft=Stft(n_fft, hop)
        )
        result = det.analyze(tracks)
        band_power = result.band_power
        edges = det.band_edges()
        n_windows = result.n_windows
        total_s = tracks[0].x.shape[0] / sr

        pair_scores = {}
        for p in beat["pairs"]:
            key = tuple(sorted((p["a"], p["b"])))
            pair_scores[key] = {
                "relative": p["relative"]["top_score"],
                "collision": p["collision"]["top_score"],
                "contention": p["contention"]["top_score"],
            }

        names = [t.name for t in tracks]
        audible_db = {t.name: rms_dbfs(t.x) for t in tracks}
        band_sums = {name: band_power[name].sum(axis=0) for name in names}

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                cand_ab, raw_ab = _screen_ordered_pair(
                    beat["beat"],
                    folder,
                    sr,
                    a,
                    b,
                    band_power,
                    band_sums,
                    edges,
                    audible_db,
                    window_seconds,
                    n_windows,
                    total_s,
                    pair_scores,
                )
                cand_ba, raw_ba = _screen_ordered_pair(
                    beat["beat"],
                    folder,
                    sr,
                    b,
                    a,
                    band_power,
                    band_sums,
                    edges,
                    audible_db,
                    window_seconds,
                    n_windows,
                    total_s,
                    pair_scores,
                )
                for raw_cell in (raw_ab, raw_ba):
                    if raw_cell is not None:
                        counts[f"{raw_cell}_screened"] += 1

                chosen: list[dict] = []
                both_parity = bool(
                    cand_ab
                    and cand_ab["cell"] == "parity"
                    and cand_ba
                    and cand_ba["cell"] == "parity"
                )
                if both_parity:
                    # both orderings qualify: keep the quieter track as target
                    if audible_db[a] < audible_db[b] or (audible_db[a] == audible_db[b] and a < b):
                        chosen.append(cand_ab)
                    else:
                        chosen.append(cand_ba)
                else:
                    if cand_ab is not None:
                        chosen.append(cand_ab)
                    if cand_ba is not None:
                        chosen.append(cand_ba)

                for c in chosen:
                    candidates[c["cell"]].append(c)
                    counts[f"{c['cell']}_verified"] += 1

        print(f"  screened {beat['beat']}: {len(tracks)} tracks")

    return candidates, counts


def match_groups(candidates: dict[str, list[dict]], per_cell: int) -> list[tuple[dict, dict]]:
    """Match up to per_cell "parity" candidates to a frequency-close "buried" one each.

    Parity candidates are ranked descending by the target's k_star energy over the
    excerpt (the positions where the element is most clearly present), one attempt per
    beat. The top per_cell distinct-beat candidates are fixed as the attempt set before
    any matching happens, so a failed match is dropped rather than replaced by a 7th
    candidate ("nicht auffuellen").

    For each attempt, a buried partner is looked for within MATCH_MAX_OCTAVES of the
    parity candidate's band center: same beat first, else the closest candidate from a
    beat not already claimed by another position in this pack. Beats are claimed as
    soon as a parity attempt is fixed (whether or not it later finds a match) and again
    whenever a cross-beat buried match is made.
    """
    parity_sorted = sorted(candidates["parity"], key=lambda c: -c["target_excerpt_energy"])
    buried = candidates["buried"]

    def buried_key(b: dict) -> tuple[str, str, str]:
        return (b["beat"], b["target"], b["masker"])

    beats_used: set[str] = set()
    used_buried: set[tuple[str, str, str]] = set()
    groups: list[tuple[dict, dict]] = []
    dropped = 0

    # Walk the full ranked candidate list (one attempt per beat) until per_cell groups
    # are built or the pool runs out. "nicht auffuellen" means a failed attempt is not
    # compensated by relaxing a constraint (octave cap, beat exclusivity, reusing a
    # buried candidate), not that the walk stops after the first per_cell candidates.
    for cand in parity_sorted:
        if len(groups) >= per_cell:
            break
        beat = cand["beat"]
        if beat in beats_used:
            continue
        beats_used.add(beat)  # at most one parity attempt per beat, win or lose

        f_z = cand["band_center_hz"]

        def distance(b: dict, f_z: float = f_z) -> float:
            return abs(float(np.log2(b["band_center_hz"] / f_z)))

        same_pool = sorted(
            (b for b in buried if b["beat"] == beat and buried_key(b) not in used_buried),
            key=distance,
        )
        foreign_pool = sorted(
            (
                b
                for b in buried
                if b["beat"] != beat
                and b["beat"] not in beats_used
                and buried_key(b) not in used_buried
            ),
            key=distance,
        )

        match, same_flag = None, None
        if same_pool and distance(same_pool[0]) <= MATCH_MAX_OCTAVES:
            match, same_flag = same_pool[0], True
        elif foreign_pool and distance(foreign_pool[0]) <= MATCH_MAX_OCTAVES:
            match, same_flag = foreign_pool[0], False

        if match is None:
            dropped += 1
            print(
                f"WARNING: parity position from beat {beat!r} "
                f"({cand['target']} x {cand['masker']}) dropped, no buried match within "
                f"{MATCH_MAX_OCTAVES} octave(s)"
            )
            continue

        group_id = f"g{len(groups) + 1}"
        dist_val = distance(match)
        for pos in (cand, match):
            pos["match_group"] = group_id
            pos["match_distance_octaves"] = dist_val
            pos["same_beat_as_match"] = same_flag
        used_buried.add(buried_key(match))
        if not same_flag:
            beats_used.add(match["beat"])
        groups.append((cand, match))

    if len(groups) < per_cell:
        print(
            f"WARNING: only {len(groups)}/{per_cell} match groups built "
            f"({dropped} parity candidate(s) dropped for lack of a partner, not backfilled)"
        )

    return groups


def assert_not_excluded(pos: dict, excluded_by_beat: dict[str, set[str]]) -> None:
    """Hard stop if a position uses a stem masking_corpus.load_beat excluded as a sum.

    load_beat already filters sum stems out before any track reaches screening, so an
    excluded stem should never survive into a position. If one does, the exclusion
    filter and this script have drifted apart, which should raise, not build quietly.
    """
    excluded = excluded_by_beat.get(pos["beat"], set())
    for role in ("target", "masker"):
        name = pos[role]
        if name in excluded:
            raise AssertionError(
                f"{pos['beat']}: {role} stem {name!r} is in the excluded list for this "
                "beat but reached position building"
            )


def finalize_position(pos: dict, max_seconds: float) -> dict:
    """Load the pair fresh and render A (target alone) / B (target plus masker).

    Reuses the excerpt_start_s/excerpt_end_s already fixed and verified during
    screening; band_power (and hence those bounds) is per track and does not depend on
    which other tracks were loaded alongside it, so re-deriving it here is unnecessary.
    """
    loaded = load_pair(pos["folder"], pos["target"], pos["masker"], max_seconds)
    if loaded is None:
        raise AssertionError(
            f"{pos['beat']}: could not reload {pos['target']!r}/{pos['masker']!r} for "
            "final rendering, even though screening already loaded them"
        )
    sr, x_target, x_masker = loaded
    s0 = int(round(pos["excerpt_start_s"] * sr))
    s1 = min(int(round(pos["excerpt_end_s"] * sr)), x_target.shape[0], x_masker.shape[0])
    if s1 <= s0:
        raise AssertionError(f"{pos['beat']}: empty excerpt for {pos['target']!r}")

    A = x_target[s0:s1].copy()
    B = x_target[s0:s1] + x_masker[s0:s1]
    peak = max(float(np.max(np.abs(A))), float(np.max(np.abs(B))), 1e-12)
    g = 0.999 / peak if peak > 0.999 else 1.0
    A = fade(A * g, sr, FADE_MS)
    B = fade(B * g, sr, FADE_MS)

    pos["sr"] = sr
    pos["gain_g"] = g
    pos["audio_a"] = A
    pos["audio_b"] = B
    pos["duration_s"] = (s1 - s0) / sr
    return pos


_AUDITION_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Mud vs Burial Hoertest</title>
<style>
  body { font: 15px/1.5 -apple-system, sans-serif; max-width: 760px;
         margin: 3rem auto; color: #ddd; background: #16181c; }
  h1 { font-size: 1.2rem; }
  p { color: #888; }
  .intro { background: #1c1f26; border-radius: 6px; padding: 1rem 1.2rem;
           margin-bottom: 1.5rem; }
  .row { display: flex; gap: .6rem; align-items: center; padding: .5rem 0;
         border-bottom: 1px solid #24272d; flex-wrap: wrap; }
  .row .label { flex: 0 0 60px; }
  .row .ab { display: flex; gap: .4rem; }
  .row .judge { display: flex; gap: .4rem; margin-left: auto; flex-wrap: wrap; }
  button { background: #24272d; color: #ddd; border: 0; border-radius: 4px;
           padding: .35rem .9rem; cursor: pointer; }
  button.playing { background: #3a7bd5; color: #fff; }
  button.j.active { background: #d5573a; color: #fff; }
  .toolbar { margin: 1.5rem 0; display: flex; align-items: center; gap: .8rem; }
  #copy-status { color: #6c6; }
</style>
<h1>Mud vs Burial Hoertest</h1>
<div class="intro">
  <p>A ist ein Element allein, B ist dasselbe Element plus eine zweite Spur. Das
  Element ist in A und B exakt gleich laut, es kommt nur die zweite Spur dazu.</p>
  <p>In manchen Positionen ist die zweite Spur deutlich lauter, dann wird B insgesamt
  lauter. Das gehoert dazu, urteile ueber das Element und nicht ueber die
  Gesamtlautstaerke.</p>
  <p>Vier Urteile stehen zur Wahl:</p>
  <p><b>bleibt klar</b>: das Element steht in B genauso da wie in A.<br>
  <b>verschwimmt</b>: die Kontur geht verloren, das Element ist aber noch klar da.<br>
  <b>geht unter</b>: das Element ist kaum noch herauszuhoeren.<br>
  <b>unsicher</b>: du kannst dich nicht entscheiden.</p>
  <p>Der Unterschied zwischen "verschwimmt" und "geht unter" ist der ganze Punkt hier,
  also nimm dir die Zeit dafuer. Stell dir pro Position eine angenehme Lautstaerke ein,
  die Positionen sind untereinander nicht pegelangeglichen.</p>
</div>
<div class="toolbar">
  <button id="copy-btn">Ergebnis kopieren</button>
  <span id="copy-status"></span>
</div>
__ROWS__
<script>
  const STORAGE_KEY = __STORAGE_KEY__;
  const player = new Audio();
  player.loop = true;
  let active = null;

  document.querySelectorAll(".ab button[data-src]").forEach(btn => {
    btn.addEventListener("click", () => {
      if (active === btn) { player.pause(); mark(null); return; }
      const sameRow = active && active.closest(".row") === btn.closest(".row");
      const t = sameRow ? player.currentTime : 0;
      player.src = btn.dataset.src;
      player.currentTime = t;
      player.play();
      mark(btn);
    });
  });
  function mark(btn) {
    document.querySelectorAll("button.playing").forEach(b => b.classList.remove("playing"));
    if (btn) btn.classList.add("playing");
    active = btn;
  }

  function loadJudgments() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; }
    catch (e) { return {}; }
  }
  function saveJudgments(data) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  }
  const judgments = loadJudgments();

  document.querySelectorAll(".row").forEach(row => {
    const pos = row.dataset.pos;
    const buttons = row.querySelectorAll(".j");
    function refresh() {
      buttons.forEach(b => b.classList.toggle("active", judgments[pos] === b.dataset.val));
    }
    buttons.forEach(b => {
      b.addEventListener("click", () => {
        judgments[pos] = b.dataset.val;
        saveJudgments(judgments);
        refresh();
      });
    });
    refresh();
  });

  document.getElementById("copy-btn").addEventListener("click", () => {
    const order = [...document.querySelectorAll(".row")].map(r => r.dataset.pos);
    const text = order.map(pos => `${pos}: ${judgments[pos] || ""}`).join("\\n");
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e) { /* clipboard fallback only */ }
    document.body.removeChild(ta);
    const status = document.getElementById("copy-status");
    status.textContent = "kopiert";
    setTimeout(() => { status.textContent = ""; }, 1500);
  });
</script>
"""


def _position_row(pos: dict) -> str:
    pid = pos["id"]
    return (
        f'<div class="row" data-pos="{pid}">\n'
        f'  <span class="label">{pid}</span>\n'
        f'  <span class="ab">\n'
        f'    <button data-src="audio/{pid}_A_solo.wav">A</button>\n'
        f'    <button data-src="audio/{pid}_B_pair.wav">B</button>\n'
        f"  </span>\n"
        f'  <span class="judge">\n'
        f'    <button class="j" data-val="bleibt klar">bleibt klar</button>\n'
        f'    <button class="j" data-val="verschwimmt">verschwimmt</button>\n'
        f'    <button class="j" data-val="geht unter">geht unter</button>\n'
        f'    <button class="j" data-val="unsicher">unsicher</button>\n'
        f"  </span>\n"
        f"</div>"
    )


def write_audition_html(pack: ListenPack, positions: list[dict]) -> Path:
    storage_key = json.dumps(f"mud_vs_burial_judgments_{pack.dir.name}")
    rows_html = "\n".join(_position_row(pos) for pos in positions)
    page = _AUDITION_TEMPLATE.replace("__ROWS__", rows_html).replace("__STORAGE_KEY__", storage_key)
    out = pack.dir / "audition.html"
    out.write_text(page)
    return out


def write_readme(pack: ListenPack, positions: list[dict]) -> Path:
    table_rows = "\n".join(f"| {pos['id']} | |" for pos in positions)
    readme = (
        "# Mud vs Burial Hoertest\n\n"
        "Blind gemischter Hoertest: klaert, ob ein Element mehr Kontur verliert, wenn "
        "es mit\neiner aehnlich lauten Spur kollidiert, oder wenn es unter einer viel "
        "lauteren begraben\nliegt. Pro Position hoerst du nur die zwei Spuren, die auch "
        "das Plugin sehen wuerde.\n\n"
        "## Testdesign\n\n"
        "Pro Position zwei Files:\n\n"
        "- **A**: das Element allein\n"
        "- **B**: dasselbe Element plus eine zweite Spur (einfache Summe, beide auf "
        "ihrem\n  Pegel im Mix)\n\n"
        "Es gibt keinen RMS-Match zwischen A und B. Auf beide Files zusammen wird nur "
        "ein\ngemeinsamer Gain-Faktor als Clip-Guard angewendet. Damit ist das Element "
        "in A und B\nbitidentisch, es kommt ausschliesslich die zweite Spur dazu.\n\n"
        "## Urteile\n\n"
        "Vier Stufen stehen zur Wahl:\n\n"
        "- **bleibt klar**: das Element steht in B genauso da wie in A\n"
        "- **verschwimmt**: die Kontur geht verloren, das Element ist aber noch klar "
        "da\n"
        "- **geht unter**: das Element ist kaum noch herauszuhoeren\n"
        "- **unsicher**: keine klare Entscheidung moeglich\n\n"
        'Der Unterschied zwischen "verschwimmt" und "geht unter" ist der Kern des '
        "Tests.\n\n"
        "## Warum keine Plots\n\n"
        "Ein Spektrum-Plot wuerde sofort zeigen, wo die zweite Spur liegt, und damit "
        "die\nBlindheit des Tests zerstoeren. Deshalb gibt es hier bewusst keine.\n\n"
        "## Warum key/ erst nach dem Hoeren\n\n"
        "key/key.json loest jede Position vollstaendig auf (Herkunft, Zielspur, "
        "zweite Spur,\nPegelverhaeltnis, Frequenz). Wer vorher reinschaut hoert nicht "
        "mehr blind. Erst\nurteilen, dann aufloesen.\n\n"
        "## Urteile\n\n"
        "| Position | Urteil |\n"
        "|---|---|\n"
        f"{table_rows}\n\n"
        "Urteile: bleibt klar, verschwimmt, geht unter, unsicher. Eintragen hier oder "
        "direkt\nin audition.html, der Kopieren-Button dort gibt dir den Block in "
        "diesem Format.\n"
    )
    out = pack.dir / "README.md"
    out.write_text(readme)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a blind A/B listening pack: mud (parity) vs burial"
    )
    ap.add_argument("--corpus", type=Path, default=Path("/tmp/masking_corpus_v3.json"))
    ap.add_argument("--per-cell", type=int, default=6)
    ap.add_argument("--name", default="mud_vs_burial")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "listenpacks",
        help="root folder for the pack",
    )
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    corpus = json.loads(args.corpus.read_text())
    max_seconds = corpus["max_seconds"]

    print("screening candidates...")
    candidates, counts = screen_corpus(corpus)
    for cell in CELLS:
        print(
            f"  {cell:<8} {counts[f'{cell}_screened']:>5} screened  "
            f"{counts[f'{cell}_verified']:>5} verified after the excerpt re-check  "
            f"{len(candidates[cell]):>5} final"
        )
    excerpt_rejected = {
        cell: counts[f"{cell}_screened"] - counts[f"{cell}_verified"] for cell in CELLS
    }

    print("matching parity to buried...")
    groups = match_groups(candidates, args.per_cell)
    if not groups:
        raise SystemExit("no matched groups could be built")

    excluded_by_beat = {
        b["beat"]: {e["name"] for e in b.get("excluded", [])} for b in corpus["beats"]
    }

    print("rendering...")
    all_positions: list[dict] = []
    for parity_pos, buried_pos in groups:
        for pos in (parity_pos, buried_pos):
            assert_not_excluded(pos, excluded_by_beat)
            finalize_position(pos, max_seconds)
            all_positions.append(pos)

    rng = random.Random(args.seed)
    rng.shuffle(all_positions)
    width = max(2, len(str(len(all_positions))))
    for i, pos in enumerate(all_positions, start=1):
        pos["id"] = f"pos_{i:0{width}d}"

    pack = ListenPack(args.name, root=args.root)
    for pos in all_positions:
        pack.add_pair(
            pos["id"], pos["audio_a"], pos["audio_b"], pos["sr"], label_a="solo", label_b="pair"
        )

    key_dir = pack.dir / "key"
    key_dir.mkdir(parents=True, exist_ok=True)
    key_data = {
        pos["id"]: {
            "cell": pos["cell"],
            "match_group": pos["match_group"],
            "beat": pos["beat"],
            "folder": str(pos["folder"]),
            "a": pos["a"],
            "b": pos["b"],
            "target": pos["target"],
            "masker": pos["masker"],
            "k_star": pos["k_star"],
            "band_center_hz": pos["band_center_hz"],
            "delta_db_screening": pos["delta_db_screening"],
            "delta_db_excerpt": pos["delta_db_excerpt"],
            "rms_target_dbfs": pos["rms_target_dbfs"],
            "rms_masker_dbfs": pos["rms_masker_dbfs"],
            "excerpt_start_s": pos["excerpt_start_s"],
            "excerpt_end_s": pos["excerpt_end_s"],
            "gain_g": pos["gain_g"],
            "match_distance_octaves": pos["match_distance_octaves"],
            "same_beat_as_match": pos["same_beat_as_match"],
            "scores": pos["top_scores"],
        }
        for pos in all_positions
    }
    (key_dir / "key.json").write_text(json.dumps(key_data, indent=2))

    write_audition_html(pack, all_positions)
    write_readme(pack, all_positions)

    print(f"\npack: {pack.dir}")
    print(f"audition: {pack.dir / 'audition.html'}\n")
    print(f"{len(groups)} match group(s) built")
    for cell in CELLS:
        print(f"excerpt re-check rejected {excerpt_rejected[cell]} {cell} candidate(s)")
    print(f"\n{'position':<12}{'duration_s':>12}")
    for pos in all_positions:
        print(f"{pos['id']:<12}{pos['duration_s']:>12.2f}")


if __name__ == "__main__":
    main()
