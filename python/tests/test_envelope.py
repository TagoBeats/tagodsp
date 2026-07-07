import numpy as np
import pytest

from tagodsp.analysis.envelope import EnvelopeFollower

SR = 48000


def test_attack_time_constant():
    atk_ms = 10.0
    ef = EnvelopeFollower(attack_ms=atk_ms, release_ms=100.0, rate=SR)
    n = int(atk_ms * 1e-3 * SR)
    env = ef.process(np.ones(2 * n))
    assert np.isclose(env[n - 1], 1 - np.exp(-1), atol=2e-2)


def test_release_slower_than_attack():
    ef = EnvelopeFollower(attack_ms=1.0, release_ms=100.0, rate=SR)
    x = np.concatenate([np.ones(4800), np.zeros(4800)])
    env = ef.process(x)
    assert env[4799] > 0.99
    assert env[-1] > 0.3  # 100 ms release: still decaying after 100 ms


def test_rms_mode_on_sine():
    ef = EnvelopeFollower(attack_ms=50.0, release_ms=50.0, rate=SR, mode="rms")
    t = np.arange(SR) / SR
    env = ef.process(np.sin(2 * np.pi * 1000 * t))
    assert np.isclose(env[-1], 1 / np.sqrt(2), atol=0.02)


def test_frame_rate_gate_smoothing():
    frame_rate = SR / 256
    ef = EnvelopeFollower(attack_ms=1.0, release_ms=50.0, rate=frame_rate)
    gate = np.concatenate([np.zeros(20), np.ones(20), np.zeros(40)])
    env = ef.process(gate)
    assert env.max() <= 1.0
    assert env[39] > 0.9
    assert 0.0 < env[50] < env[39]  # smooth release, no hard cut


def test_reset_and_invalid():
    ef = EnvelopeFollower(1.0, 10.0, rate=SR)
    ef.process(np.ones(100))
    ef.reset()
    assert ef.process(np.zeros(1))[0] == 0.0
    with pytest.raises(ValueError):
        EnvelopeFollower(1.0, 10.0, rate=SR, mode="avg")
