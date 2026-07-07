import numpy as np

from tagodsp.utils.gain import db_to_lin, lin_to_db, peak_db, rms, rms_db


def test_db_lin_roundtrip():
    db = np.array([-60.0, -6.0, 0.0, 6.0])
    np.testing.assert_allclose(lin_to_db(db_to_lin(db)), db, atol=1e-12)


def test_known_values():
    assert np.isclose(db_to_lin(0.0), 1.0)
    assert np.isclose(db_to_lin(-20.0), 0.1)
    assert np.isclose(lin_to_db(2.0), 6.0205999, atol=1e-6)


def test_lin_to_db_floor():
    assert lin_to_db(0.0) == -160.0
    assert lin_to_db(0.0, floor_db=-100.0) == -100.0


def test_rms_of_sine():
    sr = 48000
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 1000 * t)
    assert np.isclose(rms(x), 1 / np.sqrt(2), atol=1e-4)
    assert np.isclose(rms_db(x), -3.0103, atol=1e-3)


def test_peak_db():
    x = np.array([0.0, -0.5, 0.25])
    assert np.isclose(peak_db(x), lin_to_db(0.5))
