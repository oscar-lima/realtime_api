"""Reference timeline, playback queue and echo gate (no audio hardware)."""

import numpy as np

from realtime_api.duplex_audio import FRAME, PROC_RATE, EchoGate, PlaybackQueue, ReferenceTimeline, speaker_envelope


def test_timeline_read_back_by_time():
    tl = ReferenceTimeline(PROC_RATE, seconds=2.0)
    x = np.arange(1, 1601, dtype=np.int16)
    tl.write(x, 10.0)
    assert np.array_equal(tl.read(10.0, 1600), x)
    assert np.array_equal(tl.read(10.05, 10), x[800:810])
    # before the first write and after the end: silence
    assert not tl.read(9.0, 100).any()
    assert not tl.read(10.2, 100).any()
    # straddling the end: known part copied, rest zero
    part = tl.read(10.09, 320)
    assert np.array_equal(part[:160], x[1440:1600]) and not part[160:].any()
    assert abs(tl.end_time() - 10.1) < 1e-9


def test_timeline_ignores_timestamp_jitter_but_follows_real_jumps():
    tl = ReferenceTimeline(PROC_RATE, seconds=5.0)
    rng = np.random.default_rng(0)
    t = 1.0
    for k in range(300):  # 3 s of 10 ms blocks with +-3 ms timestamp jitter
        tl.write(np.full(160, k % 100 + 1, dtype=np.int16), t + rng.uniform(-0.003, 0.003))
        t += 0.01
    before = tl.reanchors
    # after settling, a block written at time t reads back at time t within 0.5 ms
    block = tl.read(t - 0.5, 160)
    assert np.count_nonzero(block == block[80]) >= 160 - 8
    assert np.count_nonzero(tl.read(t - 2.0, 2 * PROC_RATE) == 0) < 20  # (almost) no holes
    tl.write(np.full(160, 7, dtype=np.int16), t + 0.3)  # dropout: real gap
    assert tl.reanchors == before + 1
    assert tl.read(t + 0.3, 1)[0] == 7
    assert not tl.read(t + 0.1, 100).any()


def test_timeline_ring_wraps():
    tl = ReferenceTimeline(PROC_RATE, seconds=0.1)  # 1600 samples
    for k in range(30):
        tl.write(np.full(160, k, dtype=np.int16), k * 0.01)
    assert tl.read(0.29, 1)[0] == 29
    assert not tl.read(0.0, 10).any()  # overwritten long ago -> treated as unknown


def test_playback_queue_counts_played_audio_and_clears():
    q = PlaybackQueue(24000)
    q.put(np.ones(2400, dtype=np.int16), "item_a")
    q.put(np.ones(2400, dtype=np.int16), "item_b")
    assert q.pull(3000).sum() == 3000
    assert q.played_ms("item_a") == 100
    assert q.played_ms("item_b") == 25
    assert q.clear() == "item_b"
    assert q.pending_samples() == 0
    assert not q.pull(100).any()


def _frame(level_dbfs, rng):
    amp = 32768 * 10 ** (level_dbfs / 20) * np.sqrt(2)
    return (amp * np.sin(np.linspace(0, 40 * np.pi, FRAME)) + rng.standard_normal(FRAME)).astype(np.int16)


def test_gate_modes_without_echo_pass_everything():
    rng = np.random.default_rng(0)
    for mode in ("full", "smart", "half"):
        g = EchoGate(mode)
        f = _frame(-30, rng)
        assert np.array_equal(np.concatenate(g.process(f, echo_active=False)), f)


def test_half_gate_mutes_while_robot_speaks():
    g = EchoGate("half")
    f = _frame(-20, np.random.default_rng(0))
    assert not np.concatenate(g.process(f, echo_active=True, ref_level_db=-20)).any()


def test_smart_gate_blocks_residual_echo_and_opens_for_user():
    rng = np.random.default_rng(1)
    g = EchoGate("smart")
    sent_during_echo = 0
    for _ in range(300):  # 3 s of robot speech; residual echo ~35 dB under the speaker
        ref_db = -17 + rng.uniform(-6, 3)
        out = g.process(_frame(ref_db - 35 + rng.uniform(-3, 3), rng), True, ref_db)
        sent_during_echo += int(np.concatenate(out).any())
    assert sent_during_echo <= 6  # <= 2 %

    opened_at = None
    released = 0
    for i in range(50):  # user starts talking over the robot, as loud as the echo
        out = g.process(_frame(-26, rng), True, -17)
        released += len(out)
        if opened_at is None and np.concatenate(out).any():
            opened_at = i
    assert opened_at is not None and opened_at <= 5
    assert released > 50  # pre-roll frames were released when it opened


def test_speaker_envelope_takes_loudest_frame():
    x = np.zeros(FRAME * 5, dtype=np.int16)
    x[FRAME * 3:FRAME * 4] = 1000
    assert abs(speaker_envelope(x) - 1000) < 1
