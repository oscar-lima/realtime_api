"""Realtime client protocol handling and VoiceSession glue, with fakes."""

import base64
import json
import time
import types

import numpy as np
import pytest

from realtime_api.realtime_client import RealtimeClient, build_session, ga_to_beta_session, resolve_endpoint
from realtime_api.voice_session import ROBOT_COMMAND_TOOL, VoiceSession


class FakeWs:
    def __init__(self):
        self.sent = []

    def send(self, data):
        self.sent.append(json.loads(data))


def make_client():
    client = RealtimeClient("ws://test", "token")
    client._ws = FakeWs()
    return client


class FakePlayback:
    def __init__(self):
        self.played = {}

    def played_ms(self, item):
        return self.played.get(item, 0)


class FakeEngine:
    def __init__(self):
        self.playback = FakePlayback()
        self.queued = []
        self.callbacks = []
        self.playing = False

    def add_frame_callback(self, cb):
        self.callbacks.append(cb)

    def play(self, pcm, rate, item_id=""):
        self.queued.append((pcm, rate, item_id))
        self.playing = True

    def is_playing(self):
        return self.playing

    def stop_playback(self):
        self.playing = False
        return self.queued[-1][2] if self.queued else None


def test_resolve_endpoint_litellm_and_openai(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "master")
    url, key = resolve_endpoint("litellm", "gpt-realtime-2.1-mini", "http://host.docker.internal:4000/v1/")
    assert url == "ws://host.docker.internal:4000/v1/realtime?model=gpt-realtime-2.1-mini"
    assert key == "master"
    url, _ = resolve_endpoint("litellm", "m", "https://proxy.example")
    assert url == "wss://proxy.example/v1/realtime?model=m"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    url, key = resolve_endpoint("openai", "gpt-realtime-2.1")
    assert url.startswith("wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1") and key == "sk-test"


def test_session_shapes():
    ga = build_session("hi", tools=[ROBOT_COMMAND_TOOL])
    assert ga["type"] == "realtime"
    assert ga["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert ga["audio"]["input"]["turn_detection"]["interrupt_response"] is True
    beta = ga_to_beta_session(ga)
    assert beta["input_audio_format"] == "pcm16" and beta["voice"] == "cedar"
    assert beta["tools"][0]["name"] == "send_robot_command"
    assert "audio" not in build_session("hi", audio=False)


def test_session_sent_after_created_matching_protocol():
    client = make_client()
    client._session = build_session("hi")
    client._dispatch({"type": "session.created", "session": {"type": "realtime", "model": "x"}})
    assert client.protocol == "ga"
    assert client._ws.sent[-1]["session"]["type"] == "realtime"
    client._dispatch({"type": "session.created", "session": {"object": "realtime.session"}})
    assert client.protocol == "beta"
    assert client._ws.sent[-1]["session"]["input_audio_format"] == "pcm16"
    client._dispatch({"type": "session.updated", "session": {}})
    assert client.wait_ready(0.1)


def test_beta_event_names_are_mapped():
    client = make_client()
    seen = []
    client.on("response.output_audio.delta", lambda e: seen.append(e["type"]))
    client._dispatch({"type": "response.audio.delta", "delta": ""})
    assert seen == ["response.output_audio.delta"]


def test_audio_flows_both_ways():
    client = make_client()
    client.session_ready.set()
    engine = FakeEngine()
    VoiceSession(client, engine)
    pcm = (np.arange(480) % 100).astype(np.int16)
    client._dispatch({"type": "response.output_audio.delta", "item_id": "item_1",
                      "delta": base64.b64encode(pcm.tobytes()).decode()})
    got, rate, item = engine.queued[0]
    assert rate == 24000 and item == "item_1" and np.array_equal(got, pcm)
    for _ in range(4):  # four 10 ms frames at 16 kHz -> one 40 ms chunk at 24 kHz
        engine.callbacks[0](np.ones(160, dtype=np.int16), {})
    sent = client._ws.sent[-1]
    assert sent["type"] == "input_audio_buffer.append"
    assert abs(len(base64.b64decode(sent["audio"])) // 2 - 960) <= 2


def test_barge_in_stops_playback_and_truncates():
    client = make_client()
    engine = FakeEngine()
    session = VoiceSession(client, engine)
    engine.play(np.zeros(10, dtype=np.int16), 24000, "item_7")
    engine.playback.played["item_7"] = 1234
    client._dispatch({"type": "input_audio_buffer.speech_started"})
    assert not engine.playing
    assert session.interruptions == 1
    assert client._ws.sent[-1] == {"type": "conversation.item.truncate", "item_id": "item_7", "content_index": 0,
                                   "audio_end_ms": 1234, "event_id": client._ws.sent[-1]["event_id"]}
    # audio of the interrupted answer still in flight is dropped
    client._dispatch({"type": "response.output_audio.delta", "item_id": "item_7",
                      "delta": base64.b64encode(np.ones(10, dtype=np.int16).tobytes()).decode()})
    assert len(engine.queued) == 1 and not engine.playing
    # user speech while the robot is silent is not a barge-in
    client._dispatch({"type": "input_audio_buffer.speech_started"})
    assert session.interruptions == 1


def test_tool_call_is_answered_once_response_is_done():
    client = make_client()
    calls = []
    VoiceSession(client, None, on_tool_call=lambda name, args: calls.append((name, args)) or {"ok": True})
    client._dispatch({"type": "response.done", "response": {"output": [
        {"type": "function_call", "name": "send_robot_command", "call_id": "c1",
         "arguments": json.dumps({"command": "bring the cup"})},
    ]}})
    assert calls == [("send_robot_command", {"command": "bring the cup"})]
    kinds = [e["type"] for e in client._ws.sent]
    assert kinds == ["conversation.item.create", "response.create"]
    assert client._ws.sent[0]["item"] == {"type": "function_call_output", "call_id": "c1", "output": '{"ok": true}'}


def test_user_and_assistant_text_callbacks():
    client = make_client()
    got = []
    VoiceSession(client, None, on_user_text=lambda t: got.append(("user", t)),
                 on_assistant_text=lambda t: got.append(("robot", t)))
    client._dispatch({"type": "conversation.item.input_audio_transcription.completed", "transcript": " hello "})
    client._dispatch({"type": "response.output_audio_transcript.done", "transcript": "Hi there."})
    assert got == [("user", "hello"), ("robot", "Hi there.")]


def test_benign_server_errors_do_not_count_as_failures():
    client = make_client()
    client._dispatch({"type": "error", "error": {"code": "conversation_already_has_active_response", "message": "x"}})
    assert client.last_error is None
    client._dispatch({"type": "error", "error": {"code": "insufficient_quota", "message": "x"}})
    assert client.last_error["code"] == "insufficient_quota"


def test_speak_only_says_out_of_band_and_mutes_unrequested_answers():
    client = make_client()
    client.session_ready.set()
    engine = FakeEngine()
    session = VoiceSession(client, engine, speak_only=True)
    session.say("I picked the coke.")
    request = client._ws.sent[-1]["response"]
    assert request["conversation"] == "none" and request["metadata"] == {"source": "say"}
    assert "I picked the coke." in request["instructions"]
    client._dispatch({"type": "response.created", "response": {"id": "r1", "metadata": {"source": "say"}}})
    client._dispatch({"type": "response.done", "response": {"id": "r1", "output": []}})
    # an answer nobody asked for (e.g. to what the person said) is cancelled and never played
    client._dispatch({"type": "response.created", "response": {"id": "r2", "metadata": None}})
    assert client._ws.sent[-1]["type"] == "response.cancel" and client._ws.sent[-1]["response_id"] == "r2"
    client._dispatch({"type": "response.output_audio.delta", "response_id": "r2", "item_id": "item_9",
                      "delta": base64.b64encode(np.ones(10, dtype=np.int16).tobytes()).decode()})
    assert engine.queued == []


class FakeServer:
    """A fake ``websocket`` module: every connection greets, confirms session.update, and can expire."""

    def __init__(self):
        self.connections = []

    def create_connection(self, url, header=None, timeout=None):
        conn = FakeConnection()
        self.connections.append(conn)
        return conn


class FakeConnection:
    def __init__(self):
        import queue

        self.incoming = queue.Queue()
        self.sent = []
        self.incoming.put({"type": "session.created", "session": {"type": "realtime", "model": "fake"}})

    def settimeout(self, value):
        pass

    def recv(self):
        event = self.incoming.get(timeout=5)
        return "" if event is None else json.dumps(event)

    def send(self, data):
        event = json.loads(data)
        self.sent.append(event)
        if event["type"] == "session.update":
            self.incoming.put({"type": "session.updated", "session": event["session"]})

    def expire(self):
        """What the API does after 60 minutes."""
        self.incoming.put({"type": "error", "error": {"type": "invalid_request_error", "code": "session_expired",
                                                      "message": "Your session hit the maximum duration of 60 minutes."}})
        self.incoming.put(None)

    def close(self):
        self.incoming.put(None)


def test_reconnect_opens_a_new_session_with_the_same_configuration_and_handlers(monkeypatch):
    import sys

    server = FakeServer()
    monkeypatch.setitem(sys.modules, "websocket", server)
    client = RealtimeClient("ws://test", "token")
    engine = FakeEngine()
    session = VoiceSession(client, engine, speak_only=True)
    closed = []
    client.on("connection.closed", closed.append)
    config = build_session("x", language="en")
    client.connect(config)
    assert client.wait_ready(2)
    server.connections[0].expire()
    assert client.closed.wait(2) and client.last_error["code"] == "session_expired"
    session.say("Sorry, I was away.")  # waits for the next session instead of being lost
    client.reconnect()
    assert client.wait_ready(2) and not client.closed.is_set() and len(closed) == 1
    second = server.connections[1].sent
    assert second[0] == {"type": "session.update", "session": config, "event_id": second[0]["event_id"]}
    said = lambda: any(e["type"] == "response.create" and "Sorry, I was away." in e["response"]["instructions"]  # noqa: E731
                       for e in list(second))
    deadline = time.monotonic() + 2
    while not said() and time.monotonic() < deadline:  # flushed by the session.updated handler (reader thread)
        time.sleep(0.01)
    assert said()
    client.close()


def test_a_failed_reconnect_leaves_the_client_closed(monkeypatch):
    import sys

    def refuse(url, header=None, timeout=None):
        raise ConnectionRefusedError("proxy down")

    monkeypatch.setitem(sys.modules, "websocket", types.SimpleNamespace(create_connection=refuse))
    client = RealtimeClient("ws://test", "token")
    client._session = {"type": "realtime"}
    with pytest.raises(ConnectionRefusedError):
        client.reconnect()
    assert client.closed.is_set()
    client.send({"type": "input_audio_buffer.append"})  # silently dropped, no exception

