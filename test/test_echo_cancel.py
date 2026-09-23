"""Echo canceller backends on a simulated room (real Piper speech)."""

import os

import numpy as np
import pytest

from realtime_api.audio_utils import StreamResampler, read_wav_mono, to_int16
from realtime_api.echo_cancel import (
    available_backends,
    erle_db,
    estimate_delay,
    make_echo_canceller,
    run_offline,
)

DATA = os.path.join(os.path.dirname(__file__), "data")
FS = 16000
AVAILABLE = [name for name, status in available_backends().items() if status == "ok" and name != "none"]
# echo reduction required after convergence (dB); loose bounds, real runs are ~10 dB better
MIN_ERLE = {"webrtc": 30.0, "speex": 15.0, "nlms": 20.0}


def _room(seed: int = 0):
    """Robot voice through a synthetic reverberant room with a slightly
    distorting speaker, plus a user speaking over it (double talk)."""
    from scipy.signal import fftconvolve

    far, _ = read_wav_mono(os.path.join(DATA, "robot_voice_16k.wav"), FS)
    near, _ = read_wav_mono(os.path.join(DATA, "user_voice_16k.wav"), FS)
    far = np.concatenate([far, far])
    rng = np.random.default_rng(seed)
    n = int(0.15 * FS)
    t = np.arange(n) / FS
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / 0.12)
    ir[: int(0.04 * FS)] = 0.0  # 40 ms: the reference leads the echo
    ir[int(0.04 * FS)] = 3.0
    ir *= 1.2 / np.sqrt(np.sum(ir ** 2))
    echo = np.tanh(2.0 * fftconvolve(far / 32768.0, ir)[: far.size]) / 2.0
    user = np.zeros(far.size)
    start = 16 * FS
    user[start:start + near.size] = near / 32768.0 * 0.5
    noise = 10 ** (-60 / 20) * rng.standard_normal(far.size)
    return far, to_int16(echo + user + noise), to_int16(user), slice(8 * FS, 15 * FS), slice(start, start + near.size)


@pytest.mark.parametrize("backend", AVAILABLE)
def test_backend_removes_echo(backend):
    far, mic, _, echo_only, _ = _room()
    aec = make_echo_canceller(backend, FS)
    out = run_offline(aec, mic, far)
    aec.close()
    assert erle_db(mic[echo_only], out[echo_only]) > MIN_ERLE[backend]


@pytest.mark.parametrize("backend", [b for b in AVAILABLE if b in ("speex", "nlms")])
def test_linear_backends_keep_user_speech(backend):
    far, mic, user, _, talk = _room()
    out = run_offline(make_echo_canceller(backend, FS), mic, far)
    o = out[talk].astype(float)
    u = user[talk].astype(float)
    corr = max(np.dot(o[k:], u[: u.size - k]) / np.sqrt(np.dot(o, o) * np.dot(u, u) + 1e-9)
               for k in range(0, 480, 16))
    assert corr > 0.6


def test_none_backend_is_passthrough():
    x = (np.arange(160) * 7).astype(np.int16)
    assert np.array_equal(make_echo_canceller("none").process(x, x), x)


def test_auto_picks_a_backend():
    assert make_echo_canceller("auto", FS).name in ("webrtc", "speex", "nlms")


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        make_echo_canceller("magic")


@pytest.mark.parametrize("shift_ms", [-30, 0, 25, 120])
def test_estimate_delay_sign_and_value(shift_ms):
    rng = np.random.default_rng(3)
    ref = to_int16(rng.standard_normal(FS * 2) * 0.1)
    k = int(shift_ms * FS / 1000)
    mic = np.roll(ref, k)  # mic lags ref by k samples (negative: leads)
    lag, conf = estimate_delay(ref, mic, FS, max_delay_s=0.3)
    assert abs(lag - k) <= 1
    assert conf > 8


def test_stream_resampler_is_continuous():
    t = np.arange(48000) / 48000.0
    tone = to_int16(0.5 * np.sin(2 * np.pi * 440 * t))
    rs = StreamResampler(48000, 16000)
    out = np.concatenate([rs.process(tone[i:i + 480]) for i in range(0, tone.size, 480)])
    assert abs(out.size - 16000) <= 2
    # no clicks at block borders: sample-to-sample steps stay small
    assert np.max(np.abs(np.diff(out[100:].astype(int)))) < 0.5 * 32767 * 2 * np.pi * 440 / 16000 * 1.3
