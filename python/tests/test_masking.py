import numpy as np
import pytest

from tagodsp.analysis.masking import MaskingDetector, Track, _normalize_relative

SR = 48000


def _tone(freq_hz: float, dur_s: float, sr: float, amp: float = 0.5, start_s: float = 0.0,
          total_s: float | None = None) -> np.ndarray:
    """Sine tone active in [start_s, start_s + dur_s), zero elsewhere, over total_s."""
    total_s = total_s if total_s is not None else start_s + dur_s
    n_total = int(round(total_s * sr))
    x = np.zeros(n_total)
    s = int(round(start_s * sr))
    e = s + int(round(dur_s * sr))
    t = np.arange(e - s) / sr
    x[s:e] = amp * np.sin(2 * np.pi * freq_hz * t)
    return x


def _best_zone(zones):
    assert zones, "expected at least one zone"
    return zones[0]  # analyze() sorts descending by score


def test_collision_detected_both_scorings():
    # Two 200 Hz tones at equal level, overlapping for well over min_len windows.
    a = _tone(200.0, 1.5, SR, amp=0.5)
    b = _tone(200.0, 1.5, SR, amp=0.5)
    tracks = [Track("a", a), Track("b", b)]

    for scoring in ("collision", "relative", "contention"):
        det = MaskingDetector(sr=SR, scoring=scoring)
        result = det.analyze(tracks)
        assert len(result.zones) >= 1
        best = _best_zone(result.zones)
        assert best.freq_lo_hz <= 200.0 < best.freq_hi_hz


def test_level_mismatch_collision_silent_relative_loud():
    # Same setup, but track b is 30 dB quieter than track a.
    # This is the central case the spec calls out: scoring="collision" works
    # on absolute power, so a track 30 dB down does not register as a
    # collision. scoring="relative" normalizes each track against its own
    # peak, so it stays loud relative to itself and still reports a zone.
    # This is not a bug in the test: it is the documented defect of
    # scoring="relative" that a later measurement is meant to resolve.
    a = _tone(200.0, 1.5, SR, amp=0.5)
    b = _tone(200.0, 1.5, SR, amp=0.5 * 10 ** (-30.0 / 20.0))
    tracks = [Track("a", a), Track("b", b)]

    collision = MaskingDetector(sr=SR, scoring="collision").analyze(tracks)
    relative = MaskingDetector(sr=SR, scoring="relative").analyze(tracks)

    assert collision.zones == []
    assert len(relative.zones) >= 1


def test_contention_silent_on_level_mismatch():
    # Same setup as test_level_mismatch_collision_silent_relative_loud: track
    # b is 30 dB quieter than track a. scoring="contention" works on
    # absolute power like scoring="collision", so it must stay silent here
    # too.
    a = _tone(200.0, 1.5, SR, amp=0.5)
    b = _tone(200.0, 1.5, SR, amp=0.5 * 10 ** (-30.0 / 20.0))
    tracks = [Track("a", a), Track("b", b)]

    contention = MaskingDetector(sr=SR, scoring="contention").analyze(tracks)

    assert contention.zones == []


def test_contention_silent_on_quiet_band():
    # Both tracks are the sum of a loud 3000 Hz tone and a very quiet 200 Hz
    # tone (~35 dB below the 3000 Hz tone), with each component equal-level
    # between the two tracks, both active over the full duration.
    #
    # At 200 Hz the two tracks are equally loud, so their ratio is 1.0:
    # scoring="collision" reports a zone there because 2*min(a,b)/(a+b) only
    # sees the ratio, not that the mix is dominated by the 3000 Hz tone.
    # scoring="contention" measures min(a, b) against the whole window's
    # broadband power, so the quiet band falls through even at ratio 1.0.
    # This is the exact measured defect of scoring="collision" the spec
    # calls out: 75% of its reported zones sit 25-35 dB under the window's
    # broadband power.
    # 34 dB, not exactly 35 dB: at 35 dB this synthetic two-tone signal sits
    # right on scoring="collision"'s own band_floor_db=-35 dB gate, and STFT
    # windowing pushes it just under the threshold, so collision reports 0
    # zones too and the test would not show the defect it is meant to show.
    # See final report for the measured boundary.
    loud = _tone(3000.0, 1.5, SR, amp=0.5)
    quiet = _tone(200.0, 1.5, SR, amp=0.5 * 10 ** (-34.0 / 20.0))
    a = loud + quiet
    b = a.copy()
    tracks = [Track("a", a), Track("b", b)]

    collision = MaskingDetector(sr=SR, scoring="collision").analyze(tracks)
    contention = MaskingDetector(sr=SR, scoring="contention").analyze(tracks)

    def in_200hz_band(zones):
        return [z for z in zones if z.freq_lo_hz <= 200.0 < z.freq_hi_hz]

    collision_200 = in_200hz_band(collision.zones)
    contention_200 = in_200hz_band(contention.zones)

    assert len(collision_200) >= 1
    assert contention_200 == []


def test_pair_reference_is_context_free():
    # The broadband reference (mix_bb) used by "collision" and "contention"
    # is the pair being compared, not the full mix. This is the whole point
    # of the pair-broadband change: the target product is a plugin with a
    # sidechain input that only ever sees two signals, never the full mix,
    # so a pair's zones must not depend on how many other tracks are
    # analyzed alongside it. "relative" never depended on the mix either
    # (each track is normalized against its own peak), so it is included
    # here only as a contrast/control, not as something this change affects.
    a = _tone(200.0, 1.5, SR, amp=0.5)
    b = _tone(200.0, 1.5, SR, amp=0.5)
    c = _tone(8000.0, 1.5, SR, amp=5.0)  # much louder, disjoint band

    def ab_zones(zones):
        return [z for z in zones if z.tracks == ("a", "b")]

    for scoring in ("collision", "relative", "contention"):
        det = MaskingDetector(sr=SR, scoring=scoring)
        alone = ab_zones(det.analyze([Track("a", a), Track("b", b)]).zones)
        with_extra = ab_zones(det.analyze([Track("a", a), Track("b", b), Track("c", c)]).zones)

        assert len(alone) == len(with_extra) >= 1
        for z1, z2 in zip(alone, with_extra, strict=True):
            assert z1.band == z2.band
            assert z1.windows == z2.windows
            assert np.isclose(z1.score, z2.score)


def test_disjoint_bands_no_zones():
    a = _tone(100.0, 1.5, SR, amp=0.5)
    b = _tone(8000.0, 1.5, SR, amp=0.5)
    tracks = [Track("a", a), Track("b", b)]

    for scoring in ("collision", "relative", "contention"):
        result = MaskingDetector(sr=SR, scoring=scoring).analyze(tracks)
        assert result.zones == []


def test_disjoint_time_no_zones():
    # a active in [0, 1.0), b active in [2.0, 3.0), well beyond STFT smearing.
    a = _tone(200.0, 1.0, SR, amp=0.5, start_s=0.0, total_s=3.0)
    b = _tone(200.0, 1.0, SR, amp=0.5, start_s=2.0, total_s=3.0)
    tracks = [Track("a", a), Track("b", b)]

    for scoring in ("collision", "relative", "contention"):
        result = MaskingDetector(sr=SR, scoring=scoring).analyze(tracks)
        assert result.zones == []


def test_short_overlap_below_min_len_no_zones():
    # Overlap shorter than min_len (default 3) windows of window_seconds=0.1,
    # i.e. well under 0.3s: should be clustered but then dropped by the
    # min_len check, not by the score check.
    a = _tone(200.0, 0.15, SR, amp=0.5)
    b = _tone(200.0, 0.15, SR, amp=0.5)
    tracks = [Track("a", a), Track("b", b)]

    for scoring in ("collision", "relative", "contention"):
        result = MaskingDetector(sr=SR, scoring=scoring).analyze(tracks)
        assert result.zones == []


def test_normalize_relative_peak_and_floor():
    db_floor = -36.0
    peak = 2.0
    floor_val = peak * (10 ** (db_floor / 10))
    P = np.array([[peak, floor_val, 0.0]])
    norm = _normalize_relative(P, db_floor)
    assert np.isclose(norm[0, 0], 1.0)
    assert np.isclose(norm[0, 1], 0.0, atol=1e-9)
    assert norm[0, 2] == 0.0


def test_relative_normalization_reaches_max_via_band_power():
    x = _tone(200.0, 1.0, SR, amp=0.7)
    det = MaskingDetector(sr=SR)
    P = det.band_power(x)
    norm = _normalize_relative(P, det.db_floor)
    assert np.isclose(norm.max(), 1.0)


def test_invalid_scoring_raises():
    with pytest.raises(ValueError):
        MaskingDetector(sr=SR, scoring="nonsense")


def test_contention_floor_validation():
    # contention_floor_db must be < CONTENTION_CEIL_DB (-3.0).
    with pytest.raises(ValueError):
        MaskingDetector(sr=SR, contention_floor_db=0.0)


def test_invalid_window_seconds_raises():
    with pytest.raises(ValueError):
        MaskingDetector(sr=SR, window_seconds=0.0)
