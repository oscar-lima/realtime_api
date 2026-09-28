"""Barge-in in a crowded room and push-to-talk (#190), with real speech mixed into babble (no audio hardware)."""

import base64
import os
import wave

import numpy as np
import pytest

from realtime_api.realtime_client import build_session
from realtime_api.voice_session import RoomLevel, TalkGate, VoiceSession
from test_realtime_client import FakeEngine, make_client

_DATA = os.path.join(os.path.dirname(__file__), "data")


def voice(name):
    with wave.open(os.path.join(_DATA, name)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)


def at_dbfs(x, db):
    return x * (10 ** (db / 20.0) * 32768.0 / np.sqrt(np.mean(x ** 2)))


def babble(seconds, db, seed=0):
    """Several people talking at once, further away: four shifted voices summed."""
    rng = np.random.default_rng(seed)
    src = np.concatenate([voice("user_voice_16k.wav"), voice("robot_voice_16k.wav")])
    n = int(seconds * 16000)
    mix = sum(np.roll(np.resize(src, n), int(rng.integers(0, src.size))) for _ in range(4))
    return at_dbfs(mix + rng.standard_normal(n) * 30.0, db)


def frames(x):
    x = np.clip(x, -32768, 32767).astype(np.int16)
    return [x[i:i + 160] for i in range(0, x.size - 159, 160)]


def feed(room, x):
    for f in frames(x):
        room.add(f)


SPEECH = voice("user_voice_16k.wav")[: 16000]  # the first second: speech onset


def test_close_voice_over_babble_is_loud_talk_further_away_is_not():
    room = RoomLevel()
    feed(room, babble(12.0, -38.0))
    for db, loud in [(-22.0, True), (-33.0, False)]:  # at the mic / one more person 2 m away
        test = RoomLevel()
        test._history.extend(room._history)
        feed(test, babble(1.0, -38.0, seed=1) + at_dbfs(SPEECH, db))
        assert test.loud() is loud, (db, test.above_floor_db())


def test_in_a_quiet_room_every_voice_is_loud():
    room = RoomLevel()
    feed(room, np.random.default_rng(2).standard_normal(16000 * 5) * 20.0)  # about -64 dBFS of fan noise
    feed(room, at_dbfs(SPEECH, -40.0))
    assert room.loud()


def test_without_room_history_any_voice_may_interrupt():
    room = RoomLevel()
    feed(room, at_dbfs(SPEECH, -40.0)[:8000])
    assert room.loud()


def test_gate_silence_does_not_lower_the_room_floor():
    room = RoomLevel()
    feed(room, babble(5.0, -38.0))
    floor = room.floor_db()
    feed(room, np.zeros(16000 * 10))  # the echo gate sent silence while the robot talked
    assert abs(room.floor_db() - floor) < 0.5
    assert room.voice_db() < -100.0  # but it is recent silence, not a voice


def crowded_session(barge_in):
    client = make_client()
    client.session_ready.set()
    engine = FakeEngine()
    session = VoiceSession(client, engine, speak_only=True, barge_in=barge_in)
    for f in frames(babble(8.0, -38.0)):
        engine.callbacks[0](f, {})
    engine.play(np.zeros(10, dtype=np.int16), 24000, "say_1")
    return client, engine, session


def test_room_talk_does_not_cut_the_robot_off_but_the_person_at_the_mic_does():
    client, engine, session = crowded_session("loud")
    for f in frames(babble(0.3, -38.0, seed=3) + at_dbfs(voice("user_voice_16k.wav")[:4800], -33.0)):
        engine.callbacks[0](f, {})
    client._dispatch({"type": "input_audio_buffer.speech_started"})
    assert engine.playing and session.interruptions == 0
    client._dispatch({"type": "input_audio_buffer.speech_stopped"})
    client._dispatch({"type": "input_audio_buffer.speech_started"})  # the person at the mic starts softly ...
    for f in frames(at_dbfs(SPEECH, -22.0)):  # ... and gets loud within the check window
        engine.callbacks[0](f, {})
    assert not engine.playing and session.interruptions == 1


def test_barge_in_stop_ignores_voices_until_the_stop_word():
    client, engine, session = crowded_session("stop")
    for f in frames(at_dbfs(SPEECH, -15.0)):
        engine.callbacks[0](f, {})
    client._dispatch({"type": "input_audio_buffer.speech_started"})
    assert engine.playing
    assert session.interrupt("stop word") and not engine.playing
    assert not session.interrupt("stop word")  # nothing playing any more


def test_the_server_interrupts_answers_only_for_barge_in_any():
    turn = build_session("x", interrupt_response=False)["audio"]["input"]["turn_detection"]
    assert turn["interrupt_response"] is False
    turn = build_session("x", vad="server_vad")["audio"]["input"]["turn_detection"]
    assert turn["interrupt_response"] is True and turn["prefix_padding_ms"] == 300
    with pytest.raises(ValueError):
        VoiceSession(make_client(), None, barge_in="sometimes")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_talk_gate_sends_silence_until_pressed_then_the_moment_before_too():
    clock = Clock()
    gate = TalkGate(preroll_ms=300, hangover_ms=500, timeout_s=15.0, clock=clock)
    ones = [np.full(160, k + 1, dtype=np.int16) for k in range(100)]
    out = [gate.process(f) for f in ones[:40]]
    assert out[:30] == [[]] * 30  # held back as pre-roll first
    assert all(len(o) == 1 and not o[0].any() for o in out[30:])  # then silence, one per frame
    gate.set(True)
    first = gate.process(ones[40])
    assert len(first) == 31 and first[0][0] == 11 and first[-1][0] == 41  # 300 ms before the press + now
    gate.set(False)
    assert gate.process(ones[41])[0][0] == 42  # hangover: the last word still goes out
    clock.t = 0.6
    assert gate.process(ones[42]) == []
    gate.set(True)
    clock.t = 16.0  # a forgotten switch closes by itself
    assert not gate.is_open


def test_push_to_talk_session_sends_the_mic_only_while_talking():
    client = make_client()
    client.session_ready.set()
    engine = FakeEngine()
    session = VoiceSession(client, engine, push_to_talk=True)
    loud = np.full(160, 3000, dtype=np.int16)
    for _ in range(80):
        engine.callbacks[0](loud, {})
    heard = [np.frombuffer(base64.b64decode(m["audio"]), np.int16) for m in client._ws.sent
             if m["type"] == "input_audio_buffer.append"]
    assert heard and not any(chunk.any() for chunk in heard)
    session.set_talk(True)
    client._ws.sent.clear()
    engine.callbacks[0](loud, {})
    heard = np.concatenate([np.frombuffer(base64.b64decode(m["audio"]), np.int16) for m in client._ws.sent])
    assert heard.size >= 0.3 * 24000 * 0.9 and np.abs(heard).max() > 2000  # the pre-roll went out at once
