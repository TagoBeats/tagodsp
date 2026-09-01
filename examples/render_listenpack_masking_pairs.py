"""Blind A/B listening pack: does "contention" match a human impression on real music?

Reads the corpus JSON from masking_corpus.py and picks up to 4 * per_flagged_arm
positions (12 by default) across three arms: pairs where only "relative" reports a
zone, pairs where only "contention" reports a zone, and a silent control arm where
neither reports anything. "collision" is ignored for arm assignment, it only rides
along in the key for reference.

The control is needed because two tracks played together always sound different from
one track alone, so "a bit blurred" is not readable without it. It also has to sit in
the same part of the spectrum as what it is a control for, otherwise a judgment
difference could just as well mean "contention found something" as "keys blur more
than kick and 808 regardless". So every flagged position (relative_only or
contention_only) gets its own frequency-matched silent control instead of drawing the
silent arm from the globally most active quiet pairs: the control's "contested band"
(the band with the highest shared, i.e. min(a, b), power) has to land within 1.0
octave of the flagged zone's band center, preferring a control from the same beat
(same material, mix, and production) when one qualifies. A flagged position without a
qualifying control is dropped rather than rendered ungated, see select_positions.

Per position, two files:

  A = the target track alone
  B = the target track plus the masking track (plain sum, both at their level in
      the mix)

There is no RMS match between A and B. The only processing applied is one shared
gain factor, used purely as a clip guard:

    g = 0.999 / max(peak(A), peak(B))   if that peak > 0.999, else g = 1.0

so the target is bit-identical in A and B; only the masker is added in B. Matching
B's RMS to A would make the target quieter in B and manufacture exactly the effect
this test is supposed to measure, so it is deliberately not done.

Everything is re-derived fresh at render time from just the two stems named in the
corpus JSON (audio, detector run, zone, excerpt), nothing is taken over from the
corpus numbers except which pair to look at and how to rank candidates.

The pack is blind: positions are shuffled with a fixed seed and named pos_01..pos_12,
no arm, beat, track name, frequency, or match pairing appears in audition.html, the
README, or any filename. The full resolution, including which position is whose
control, lives in key/key.json, meant to be opened only after judging. There are no
plots: a spectrum plot would show exactly where the masker sits and break the
blindness.

Usage:
    python render_listenpack_masking_pairs.py --corpus /tmp/masking_corpus.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from listenpack import ListenPack  # noqa: E402
from masking_corpus import load_beat  # noqa: E402

from tagodsp.analysis.masking import MaskingDetector, Track  # noqa: E402
from tagodsp.spectral.stft import Stft  # noqa: E402

ARMS = ("relative_only", "contention_only", "silent")
FLAGGED_ARMS = ("relative_only", "contention_only")
AUDIBLE_FLOOR_DB = -50.0
EXCERPT_SECONDS = 7.0
FADE_MS = 10.0

# Cap for the control matching in find_control: a silent candidate's contested band
# has to sit within this many octaves of the flagged position's zone band center to
# count as a match, same-beat or not. Positions that cannot clear this within the
# corpus are dropped rather than rendered with a mismatched control.
MATCH_MAX_OCTAVES = 1.0


def mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(mono(x)))))


def rms_dbfs(x: np.ndarray) -> float:
    r = rms(x)
    return 20.0 * np.log10(r) if r > 0 else float("-inf")


def fade(x: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    """Linear fade in/out at the excerpt edges to avoid clicks."""
    n = x.shape[0]
    length = min(int(round(fade_ms / 1000.0 * sr)), n // 2)
    if length <= 0:
        return x
    ramp = np.linspace(0.0, 1.0, length)
    out = x.copy()
    out[:length] *= ramp[:, None]
    out[-length:] *= ramp[::-1, None]
    return out


def excerpt_bounds(center_s: float, total_s: float, length_s: float) -> tuple[float, float]:
    """Center a length_s window on center_s, clamped to [0, total_s]."""
    start = center_s - length_s / 2
    end = center_s + length_s / 2
    if start < 0:
        end -= start
        start = 0.0
    if end > total_s:
        start -= end - total_s
        end = total_s
    return max(0.0, start), min(total_s, end)


def classify_pair(p: dict) -> str | None:
    """Arm for a pair entry from the corpus JSON, or None if it belongs to no arm.

    "collision" is ignored on purpose, only relative/contention decide the arm.
    """
    has_rel = p["relative"]["zones"] > 0
    has_con = p["contention"]["zones"] > 0
    if has_rel and has_con:
        return None
    if has_rel:
        return "relative_only"
    if has_con:
        return "contention_only"
    return "silent"


def screen_corpus(corpus: dict, floor_db: float) -> dict[str, list[dict]]:
    """Build sorted candidate lists per arm from the corpus JSON.

    Loads each beat once (via masking_corpus.load_beat, the same loader the corpus
    scan used) to apply the audibility filter and, for the silent arm, to get a cheap
    screening estimate of each candidate's contested band (the band with the highest
    shared, i.e. min(a, b), power, from a single beat-level detector run over all
    tracks). That estimate ("screen_band_center_hz") only decides which silent
    candidates are worth building; the final band center used for matching and the
    key comes from a fresh pair-only rebuild in try_build_silent_position, same as
    the flagged arms already re-derive their zone from a pair-only run. Audio is
    discarded again once a beat has been screened.
    """
    window_seconds = corpus["window_seconds"]
    max_seconds = corpus["max_seconds"]
    n_fft = corpus["stft"]["n_fft"]
    hop = corpus["stft"]["hop"]
    candidates: dict[str, list[dict]] = {a: [] for a in ARMS}

    for beat in corpus["beats"]:
        folder = Path(beat["folder"])
        sr = beat["sr"]
        by_arm: dict[str, list[dict]] = {a: [] for a in ARMS}
        for p in beat["pairs"]:
            arm = classify_pair(p)
            if arm is not None:
                by_arm[arm].append(p)
        if not any(by_arm.values()):
            continue

        loaded, _dropped, why = load_beat(folder, max_seconds)
        if loaded is None:
            print(f"  skip {beat['beat']}: {why}")
            continue
        beat_sr, tracks = loaded
        if beat_sr != sr:
            print(f"  skip {beat['beat']}: samplerate changed since the corpus scan")
            continue
        by_name = {t.name: t.x for t in tracks}
        audible_db = {name: rms_dbfs(x) for name, x in by_name.items()}

        band_power = None
        edges = None
        if by_arm["silent"]:
            det = MaskingDetector(
                sr=sr, scoring="relative", window_seconds=window_seconds, stft=Stft(n_fft, hop)
            )
            result = det.analyze(tracks)
            band_power = result.band_power
            edges = det.band_edges()

        for arm, pairs in by_arm.items():
            for p in pairs:
                a, b = p["a"], p["b"]
                if a not in audible_db or b not in audible_db:
                    continue
                if audible_db[a] < floor_db or audible_db[b] < floor_db:
                    continue

                entry = {
                    "beat": beat["beat"],
                    "folder": folder,
                    "sr": sr,
                    "a": a,
                    "b": b,
                    "json_scores": {
                        "relative": p["relative"]["top_score"],
                        "collision": p["collision"]["top_score"],
                        "contention": p["contention"]["top_score"],
                    },
                }
                if arm == "relative_only":
                    entry["score"] = p["relative"]["top_score"]
                elif arm == "contention_only":
                    entry["score"] = p["contention"]["top_score"]
                else:
                    shared = np.minimum(band_power[a], band_power[b])
                    band_totals = shared.sum(axis=0)
                    if not np.any(band_totals > 0):
                        continue  # nothing shared at all, not a usable control
                    k = int(np.argmax(band_totals))
                    entry["score"] = float(band_totals[k])
                    entry["screen_band_center_hz"] = float(np.sqrt(edges[k] * edges[k + 1]))
                candidates[arm].append(entry)
        print(f"  screened {beat['beat']}: {len(tracks)} tracks")

    for arm in ARMS:
        candidates[arm].sort(key=lambda c: -c["score"])
    return candidates


def find_wav(folder: Path, name: str) -> Path | None:
    direct = folder / f"{name}.wav"
    if direct.exists():
        return direct
    for p in folder.glob(f"{name}.*"):
        if p.suffix.lower() == ".wav":
            return p
    return None


def load_pair(folder: Path, a: str, b: str, max_seconds: float):
    """Load just the two named stems fresh, trimmed to their common length."""
    pa, pb = find_wav(folder, a), find_wav(folder, b)
    if pa is None or pb is None:
        return None
    sr = sf.info(pa).samplerate
    xa, _ = sf.read(pa, frames=int(max_seconds * sr), dtype="float64", always_2d=True)
    xb, _ = sf.read(pb, frames=int(max_seconds * sr), dtype="float64", always_2d=True)
    n = min(xa.shape[0], xb.shape[0])
    return sr, xa[:n], xb[:n]


def _finalize_position(
    arm: str,
    candidate: dict,
    sr: int,
    xa: np.ndarray,
    xb: np.ndarray,
    total_s: float,
    center_s: float,
    zone_info: dict,
) -> dict | None:
    """Shared tail of position building: pick the excerpt around center_s, decide
    target vs masker by which is quieter, apply the shared clip-guard gain, and
    fade. Returns None if the excerpt would be empty (center_s outside the audio).

    Shared by try_build_flagged_position and try_build_silent_position, which only
    differ in how they arrive at center_s and zone_info.
    """
    a, b = candidate["a"], candidate["b"]
    start_s, end_s = excerpt_bounds(center_s, total_s, EXCERPT_SECONDS)
    s0 = int(round(start_s * sr))
    s1 = min(int(round(end_s * sr)), xa.shape[0])
    if s1 <= s0:
        return None
    ea, eb = xa[s0:s1], xb[s0:s1]

    db = {a: rms_dbfs(ea), b: rms_dbfs(eb)}
    r = {a: rms(ea), b: rms(eb)}
    ex = {a: ea, b: eb}
    target_name = a if (r[a] < r[b] or (r[a] == r[b] and a < b)) else b
    masker_name = b if target_name == a else a

    A = ex[target_name].copy()
    B = ex[target_name] + ex[masker_name]
    peak = max(float(np.max(np.abs(A))), float(np.max(np.abs(B))), 1e-12)
    g = 0.999 / peak if peak > 0.999 else 1.0
    A = fade(A * g, sr, FADE_MS)
    B = fade(B * g, sr, FADE_MS)

    return {
        "arm": arm,
        "beat": candidate["beat"],
        "folder": str(candidate["folder"]),
        "a": a,
        "b": b,
        "sr": sr,
        "target": target_name,
        "masker": masker_name,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": (s1 - s0) / sr,
        "zone": zone_info,
        "top_scores": candidate["json_scores"],
        "rms_target_dbfs": db[target_name],
        "rms_masker_dbfs": db[masker_name],
        "gain_g": g,
        "audio_a": A,
        "audio_b": B,
    }


def try_build_flagged_position(
    candidate: dict, arm: str, window_seconds: float, n_fft: int, hop: int, max_seconds: float
) -> dict | None:
    """Re-derive a relative_only/contention_only position from scratch: load the
    pair, re-run the detector on it alone, center on its top zone, and render A/B.
    Returns None if the pair can no longer be used (file missing, or the zone did
    not survive the fresh, pair-only run).

    Sets "band_center_hz" to f_z = sqrt(freq_lo_hz * freq_hi_hz), the target this
    position's control has to be frequency-matched against (see find_control).
    """
    folder, a, b = candidate["folder"], candidate["a"], candidate["b"]
    loaded = load_pair(folder, a, b, max_seconds)
    if loaded is None:
        return None
    sr, xa, xb = loaded
    total_s = xa.shape[0] / sr

    scoring = "relative" if arm == "relative_only" else "contention"
    det = MaskingDetector(
        sr=sr, scoring=scoring, window_seconds=window_seconds, stft=Stft(n_fft, hop)
    )
    result = det.analyze([Track(a, xa), Track(b, xb)])
    if not result.zones:
        return None
    top = result.zones[0]
    center_s = ((top.windows[0] + top.windows[1] + 1) / 2) * window_seconds
    zone_info = {
        "windows": list(top.windows),
        "freq_lo_hz": top.freq_lo_hz,
        "freq_hi_hz": top.freq_hi_hz,
    }

    built = _finalize_position(arm, candidate, sr, xa, xb, total_s, center_s, zone_info)
    if built is None:
        return None
    built["band_center_hz"] = float(np.sqrt(top.freq_lo_hz * top.freq_hi_hz))
    return built


def try_build_silent_position(
    candidate: dict, window_seconds: float, n_fft: int, hop: int, max_seconds: float
) -> dict | None:
    """Re-derive a silent control from scratch: load the pair, re-run the detector
    on it alone, and center on its contested band rather than the old broadband
    peak.

    shared[w, k] = min(band_power_a[w, k], band_power_b[w, k]) is summed over all
    windows to find the contested band k (the band the two tracks share the most),
    then the window w within that band where shared[w, k] peaks becomes the excerpt
    center. This replaces the previous rule of centering on the window with the
    largest min(broadband_a, broadband_b), because the excerpt now has to represent
    the same band the matching flagged position's zone sits in, not just the
    loudest shared moment overall.

    Sets "band_center_hz" to f_s = sqrt(edge[k] * edge[k+1]), the value find_control
    compares against the flagged position's f_z. Returns None if the pair shares
    nothing (file missing, or every band's shared power is zero).
    """
    folder, a, b = candidate["folder"], candidate["a"], candidate["b"]
    loaded = load_pair(folder, a, b, max_seconds)
    if loaded is None:
        return None
    sr, xa, xb = loaded
    total_s = xa.shape[0] / sr

    det = MaskingDetector(
        sr=sr, scoring="relative", window_seconds=window_seconds, stft=Stft(n_fft, hop)
    )
    result = det.analyze([Track(a, xa), Track(b, xb)])
    shared = np.minimum(result.band_power[a], result.band_power[b])
    band_totals = shared.sum(axis=0)
    if not np.any(band_totals > 0):
        return None
    k = int(np.argmax(band_totals))
    w = int(np.argmax(shared[:, k]))
    edges = det.band_edges()
    center_s = (w + 0.5) * window_seconds
    zone_info = {
        "windows": [w, w],
        "freq_lo_hz": float(edges[k]),
        "freq_hi_hz": float(edges[k + 1]),
    }

    built = _finalize_position("silent", candidate, sr, xa, xb, total_s, center_s, zone_info)
    if built is None:
        return None
    built["band_center_hz"] = float(np.sqrt(edges[k] * edges[k + 1]))
    return built


def assert_not_excluded(built: dict, excluded_by_beat: dict[str, set[str]]) -> None:
    """Hard stop if a built position uses a stem masking_corpus.load_beat excluded.

    load_beat (name based plus measured sum detection) already ran when the corpus
    JSON was produced, and screen_corpus/try_build_flagged_position/
    try_build_silent_position re-run load_beat and the detector on the same stems,
    so an excluded stem should never survive into a position. If one does, the
    exclusion filter and the render path have drifted apart, which is exactly the
    kind of silent bug this test is meant to catch, not reproduce: raise instead of
    building the position quietly.
    """
    excluded = excluded_by_beat.get(built["beat"], set())
    for role in ("target", "masker"):
        name = built[role]
        if name in excluded:
            raise AssertionError(
                f"{built['beat']}: {role} stem {name!r} is in the excluded list for this "
                "beat but reached position building; load_beat filter is not being respected"
            )


def find_control(
    flagged: dict,
    silent_candidates: list[dict],
    used_silent: set[tuple[str, str, str]],
    beats_used: set[str],
    window_seconds: float,
    n_fft: int,
    hop: int,
    max_seconds: float,
) -> dict | None:
    """Find and build a frequency-matched silent control for one flagged position.

    Prefers the closest candidate from the SAME beat as flagged (same material, mix,
    production), accepted only if its distance is within MATCH_MAX_OCTAVES. Failing
    that, falls back to the closest candidate from a beat that has not been used by
    any position yet (flagged or already-matched control), also capped at
    MATCH_MAX_OCTAVES. Candidates are screened by screen_band_center_hz (cheap,
    beat-level estimate) but only counted once a fresh pair-only rebuild confirms
    the final band_center_hz still clears the cap. Returns None if nothing in the
    corpus qualifies; the caller drops the flagged position in that case rather than
    rendering it without a control.

    Mutates used_silent and beats_used on a successful match: a silent candidate is
    used at most once, and a foreign-beat match occupies that beat for every later
    position (same-beat matches don't need this, the beat is already occupied by
    the flagged position they belong to).
    """
    f_z = flagged["band_center_hz"]
    beat = flagged["beat"]

    def cand_key(c: dict) -> tuple[str, str, str]:
        return (c["beat"], c["a"], c["b"])

    def screen_distance(c: dict) -> float:
        return abs(float(np.log2(c["screen_band_center_hz"] / f_z)))

    def try_candidates(pool: list[dict]) -> dict | None:
        for c in sorted(pool, key=screen_distance):
            if screen_distance(c) > MATCH_MAX_OCTAVES:
                break  # sorted ascending: nothing further down clears the cap either
            built = try_build_silent_position(c, window_seconds, n_fft, hop, max_seconds)
            if built is None:
                continue
            final_distance = abs(float(np.log2(built["band_center_hz"] / f_z)))
            if final_distance > MATCH_MAX_OCTAVES:
                continue
            built["match_distance_octaves"] = final_distance
            return built, c
        return None

    same_beat = [
        c for c in silent_candidates if c["beat"] == beat and cand_key(c) not in used_silent
    ]
    result = try_candidates(same_beat)
    if result is not None:
        built, c = result
        built["same_beat_as_match"] = True
        used_silent.add(cand_key(c))
        return built

    foreign = [
        c
        for c in silent_candidates
        if c["beat"] != beat and c["beat"] not in beats_used and cand_key(c) not in used_silent
    ]
    result = try_candidates(foreign)
    if result is not None:
        built, c = result
        built["same_beat_as_match"] = False
        used_silent.add(cand_key(c))
        beats_used.add(c["beat"])
        return built

    return None


def select_positions(
    candidates: dict[str, list[dict]],
    per_flagged_arm: int,
    window_seconds: float,
    n_fft: int,
    hop: int,
    max_seconds: float,
    excluded_by_beat: dict[str, set[str]],
) -> list[dict]:
    """Build up to per_flagged_arm flagged positions per flagged arm (at most one
    flagged position per beat, arms not backfilled from each other), then match
    each to a frequency-matched silent control via find_control.

    A flagged position without a qualifying control is dropped rather than
    rendered ungated: 10 clean positions beat 12 with one mismatched pair. Returns
    a flat list of built positions (flagged and control interleaved by match
    group), not organized by arm, since the silent arm no longer has a selection
    of its own independent of what it is matched to.
    """
    beats_used: set[str] = set()
    flagged_positions: list[dict] = []
    for arm in FLAGGED_ARMS:
        built_count = 0
        for c in candidates[arm]:
            if built_count >= per_flagged_arm:
                break
            if c["beat"] in beats_used:
                continue
            built = try_build_flagged_position(c, arm, window_seconds, n_fft, hop, max_seconds)
            if built is None:
                continue
            assert_not_excluded(built, excluded_by_beat)
            flagged_positions.append(built)
            beats_used.add(c["beat"])
            built_count += 1
        if built_count < per_flagged_arm:
            print(
                f"WARNING: arm {arm!r} only yielded {built_count}/{per_flagged_arm} "
                "flagged positions after filtering, not backfilled from another arm"
            )

    used_silent: set[tuple[str, str, str]] = set()
    positions: list[dict] = []
    for i, flagged in enumerate(flagged_positions, start=1):
        group = f"g{i}"
        control = find_control(
            flagged,
            candidates["silent"],
            used_silent,
            beats_used,
            window_seconds,
            n_fft,
            hop,
            max_seconds,
        )
        if control is None:
            print(
                f"WARNING: dropping flagged position {flagged['beat']} "
                f"({flagged['a']} x {flagged['b']}, arm={flagged['arm']}): no silent "
                f"control within {MATCH_MAX_OCTAVES} octave(s) found in the corpus"
            )
            continue
        assert_not_excluded(control, excluded_by_beat)
        flagged["match_group"] = group
        flagged["match_role"] = "flagged"
        flagged["same_beat_as_match"] = control["same_beat_as_match"]
        control["match_group"] = group
        control["match_role"] = "control"
        positions.append(flagged)
        positions.append(control)
    return positions


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
        f'    <button class="j" data-val="verschwimmt">verschwimmt</button>\n'
        f'    <button class="j" data-val="bleibt klar">bleibt klar</button>\n'
        f'    <button class="j" data-val="unsicher">unsicher</button>\n'
        f"  </span>\n"
        f"</div>"
    )


_AUDITION_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Masking Pairs Hoertest</title>
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
  .row .judge { display: flex; gap: .4rem; margin-left: auto; }
  button { background: #24272d; color: #ddd; border: 0; border-radius: 4px;
           padding: .35rem .9rem; cursor: pointer; }
  button.playing { background: #3a7bd5; color: #fff; }
  button.j.active { background: #d5573a; color: #fff; }
  .toolbar { margin: 1.5rem 0; display: flex; align-items: center; gap: .8rem; }
  #copy-status { color: #6c6; }
</style>
<h1>Masking Pairs Hoertest</h1>
<div class="intro">
  <p>A ist das Zielelement allein, B ist dasselbe Element plus eine zweite Spur.
  Das Ziel ist in A und B exakt gleich laut, in B kommt nur die zweite Spur dazu.</p>
  <p>Frage pro Position: verliert das Element in B an Kontur, oder bleibt es genauso
  klar wie in A. Hoer erst A, dann B, gerne mehrfach, B laeuft in Schleife.</p>
  <p>Die Reihenfolge der zwoelf Positionen ist gemischt, es gibt bewusst keine
  Zusatzinfo dazu welche Spur oder welches Frequenzband das ist.</p>
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


def write_audition_html(pack: ListenPack, positions: list[dict]) -> Path:
    """Write a blind A/B judgment page.

    This is a dedicated page, not ListenPack.audition_page: that shared page has no
    loop playback and cannot record a judgment, both required here.
    """
    storage_key = json.dumps(f"masking_pairs_judgments_{pack.dir.name}")
    rows_html = "\n".join(_position_row(pos) for pos in positions)
    page = _AUDITION_TEMPLATE.replace("__ROWS__", rows_html).replace("__STORAGE_KEY__", storage_key)
    out = pack.dir / "audition.html"
    out.write_text(page)
    return out


def write_readme(pack: ListenPack, positions: list[dict]) -> Path:
    table_rows = "\n".join(f"| {pos['id']} | |" for pos in positions)
    readme = (
        "# Masking Pairs Hoertest\n\n"
        "Blind gemischter Hoertest: klaert, ob contention-Treffer auf echter Musik "
        "einem\nmenschlichen Eindruck entsprechen. Pro Position hoerst du nur die "
        "zwei Spuren, die\nauch das Plugin sieht.\n\n"
        "## Testdesign\n\n"
        "Pro Position zwei Files:\n\n"
        "- **A**: das Zielelement allein\n"
        "- **B**: dasselbe Zielelement plus eine zweite Spur (einfache Summe, beide "
        "auf\n  ihrem Pegel im Mix)\n\n"
        "Es gibt keinen RMS-Match zwischen A und B. Auf beide Files zusammen wird "
        "nur ein\ngemeinsamer Gain-Faktor als Clip-Guard angewendet. Damit ist das "
        "Zielelement in A\nund B bitidentisch, es kommt ausschliesslich die zweite "
        "Spur dazu. Ein RMS-Match\nwuerde das Ziel in B leiser machen und genau den "
        "Effekt herstellen, den der Test\neigentlich messen soll.\n\n"
        "Frage pro Position: verschwimmt das Ziel in B, oder bleibt es klar.\n\n"
        "## Warum keine Plots\n\n"
        "Ein Spektrum-Plot wuerde sofort zeigen, wo die zweite Spur liegt, und damit "
        "die\nBlindheit des Tests zerstoeren. Deshalb gibt es hier bewusst keine.\n\n"
        "## Warum key/ erst nach dem Hoeren\n\n"
        "key/key.json loest jede Position vollstaendig auf (Herkunft, Zielspur, "
        "Maskiererspur,\nFrequenzen). Wer vorher reinschaut hoert nicht mehr blind. "
        "Erst urteilen, dann\naufloesen.\n\n"
        "## Urteile\n\n"
        "| Position | Urteil |\n"
        "|---|---|\n"
        f"{table_rows}\n\n"
        "Urteile: verschwimmt, bleibt klar, unsicher. Eintragen hier oder direkt in "
        "audition.html,\nder Kopieren-Button dort gibt dir den Block in diesem "
        "Format.\n"
    )
    out = pack.dir / "README.md"
    out.write_text(readme)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a blind A/B listening pack for masking pairs")
    ap.add_argument("--corpus", type=Path, default=Path("/tmp/masking_corpus.json"))
    ap.add_argument("--name", default="masking_pairs")
    ap.add_argument("--per-flagged-arm", type=int, default=3)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "listenpacks",
        help="root folder for the pack",
    )
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    corpus = json.loads(args.corpus.read_text())
    window_seconds = corpus["window_seconds"]
    max_seconds = corpus["max_seconds"]
    n_fft = corpus["stft"]["n_fft"]
    hop = corpus["stft"]["hop"]

    print("screening candidates...")
    candidates = screen_corpus(corpus, AUDIBLE_FLOOR_DB)
    for arm in ARMS:
        print(f"  {arm:<16} {len(candidates[arm]):>5} candidates after the audibility filter")

    excluded_by_beat = {
        b["beat"]: {e["name"] for e in b.get("excluded", [])} for b in corpus["beats"]
    }

    print("building positions...")
    all_positions = select_positions(
        candidates,
        args.per_flagged_arm,
        window_seconds,
        n_fft,
        hop,
        max_seconds,
        excluded_by_beat,
    )
    if not all_positions:
        raise SystemExit("no positions could be built")

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
            "arm": pos["arm"],
            "beat": pos["beat"],
            "folder": pos["folder"],
            "a": pos["a"],
            "b": pos["b"],
            "target": pos["target"],
            "masker": pos["masker"],
            "zone": pos["zone"],
            "excerpt_start_s": pos["start_s"],
            "excerpt_end_s": pos["end_s"],
            "top_scores": pos["top_scores"],
            "rms_target_dbfs": pos["rms_target_dbfs"],
            "rms_masker_dbfs": pos["rms_masker_dbfs"],
            "gain_g": pos["gain_g"],
            "match_group": pos["match_group"],
            "match_role": pos["match_role"],
            "band_center_hz": pos["band_center_hz"],
            "match_distance_octaves": pos.get("match_distance_octaves"),
            "same_beat_as_match": pos["same_beat_as_match"],
        }
        for pos in all_positions
    }
    (key_dir / "key.json").write_text(json.dumps(key_data, indent=2))

    write_audition_html(pack, all_positions)
    write_readme(pack, all_positions)

    print(f"\npack: {pack.dir}")
    print(f"audition: {pack.dir / 'audition.html'}\n")
    print(f"{'position':<12}{'duration_s':>12}")
    for pos in all_positions:
        print(f"{pos['id']:<12}{pos['duration_s']:>12.2f}")


if __name__ == "__main__":
    main()
