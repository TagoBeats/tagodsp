"""Gain and level conversion utilities.

Conventions: dB values are relative full scale unless stated otherwise.
Formulas: lin = 10^(dB/20), dB = 20*log10(lin).
"""

import numpy as np

DB_MIN = -160.0


def db_to_lin(db):
    """Convert decibels to linear amplitude. lin = 10^(dB/20)."""
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 20.0)


def lin_to_db(lin, floor_db: float = DB_MIN):
    """Convert linear amplitude to decibels, clamped at floor_db to avoid -inf."""
    lin = np.abs(np.asarray(lin, dtype=np.float64))
    floor_lin = db_to_lin(floor_db)
    return 20.0 * np.log10(np.maximum(lin, floor_lin))


def rms(x: np.ndarray) -> float:
    """Root mean square of a signal."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(x))))


def rms_db(x: np.ndarray, floor_db: float = DB_MIN) -> float:
    """RMS level in dBFS."""
    return float(lin_to_db(rms(x), floor_db=floor_db))


def peak_db(x: np.ndarray, floor_db: float = DB_MIN) -> float:
    """Sample peak level in dBFS."""
    x = np.asarray(x, dtype=np.float64)
    return float(lin_to_db(np.max(np.abs(x)) if x.size else 0.0, floor_db=floor_db))
