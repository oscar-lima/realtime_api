"""Glue between the duplex audio engine and the Realtime API client.

Kept free of ROS so the standalone demo and a later ROS node / mobipick_gpt
agent share it: the demo passes print callbacks, a ROS node would publish
the same callbacks on topics and turn ``on_tool_call`` into a request to the
mobipick_gpt agents.
"""

from __future__ import annotations

import collections
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from realtime_api.audio_utils import StreamResampler
from realtime_api.duplex_audio import PROC_RATE
from realtime_api.realtime_client import API_RATE, RealtimeClient

_LOG = logging.getLogger(__name__)

ToolCall = Callable[[str, Dict[str, Any]], Any]
TextCallback = Callable[[str], None]

DEFAULT_INSTRUCTIONS = (
    "You are Mobipick, a friendly mobile manipulation robot (a mobile base with a UR arm and a "
    "gripper) working in a lab. Talk like a helpful colleague: short spoken answers, one or two "
    "sentences, no lists or markdown. When the person asks you to do something physical (pick, "
    "place, bring, move, look for, go somewhere), call send_robot_command once with a short, "
    "complete imperative instruction in English, then briefly confirm what you will do. If the "
    "request is ambiguous, ask one short clarifying question before calling the tool. You only "
    "hear the person through a microphone; ignore faint fragments of your own voice."
)

ROBOT_COMMAND_TOOL = {
    "type": "function",
    "name": "send_robot_command",
    "description": (
        "Forward a task for the robot to the Mobipick task planner, which executes it with the "
        "robot's skills (navigation, perception, pick and place)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Complete imperative instruction, e.g. "
                        "'bring the red cup from the kitchen table to the person'."},
        },
        "required": ["command"],
    },
}


# --barge-in: what may stop the robot's speech. loud: a voice clearly above the room (the person at the
# microphone, not talk further away in a crowded room); any: any speech the server detects; stop: only a stop
# word (the agent stops the speech when its transcript arrives).
BARGE_IN_MODES = ("loud", "any", "stop")
BARGE_IN_CHECK_S = 1.5  # after the server hears speech over the robot, it may still grow loud this long


def _power_db(power: float) -> float:
    return 10.0 * float(np.log10(power / 32768.0 ** 2 + 1e-12))


class RoomLevel:
    """How far the voice at the microphone stands out of the room, for barge-in.

    The floor is the median 10 ms level over the last ``history_s`` (frames the
    echo gate replaced with silence do not count); the voice is the loudest
    ``window_ms`` of the last ``recent_s``. Speech mixed into babble
    (test_barge_in.py) reads 3-7 dB above its RMS difference to the babble,
    and babble alone up to 5 dB: 15 dB lets a voice about 10 dB above the
    crowd (the person at the microphone) through, not talk further away. In a
    quiet room every voice is far above the floor.
    """

    def __init__(self, margin_db: float = 15.0, history_s: float = 30.0, recent_s: float = 1.0,
                 window_ms: int = 200, min_history_s: float = 3.0) -> None:
        self.margin_db = float(margin_db)
        self._history: "collections.deque[float]" = collections.deque(maxlen=int(history_s * 100))
        self._recent: "collections.deque[float]" = collections.deque(maxlen=int(recent_s * 100))
        self._window = max(1, int(window_ms) // 10)
        self._min_history = int(min_history_s * 100)
        self._lock = threading.Lock()  # frames come from the mic thread, questions from the websocket thread

    def add(self, frame: np.ndarray) -> None:
        """One 10 ms frame of the cleaned mic signal."""
        power = float(np.mean(frame.astype(np.float64) ** 2)) if frame.size else 0.0
        with self._lock:
            self._recent.append(power)
            if power > 0.0:  # exact zeros: silence the echo gate sent instead of the mic
                self._history.append(power)

    def floor_db(self) -> float:
        with self._lock:
            history = np.array(self._history, dtype=float)
        return _power_db(float(np.median(history))) if history.size else -120.0

    def voice_db(self) -> float:
        with self._lock:
            recent = np.array(self._recent, dtype=float)
        if recent.size == 0:
            return -120.0
        n = min(self._window, recent.size)
        return _power_db(float(np.max(np.convolve(recent, np.ones(n) / n, mode="valid"))))

    def above_floor_db(self) -> float:
        return self.voice_db() - self.floor_db()

    def loud(self) -> bool:
        with self._lock:
            known = len(self._history)
        if known < self._min_history:
            return True  # too little known about the room yet: any voice may interrupt
        return self.above_floor_db() >= self.margin_db


class TalkGate:
    """Push-to-talk: the mic reaches the server only while the talk switch is on.

    ``set(True)`` opens it for at most ``timeout_s`` (a forgotten switch closes
    by itself), ``set(False)`` closes it after ``hangover_ms`` so the last word
    is not cut. While closed, frames are held back ``preroll_ms`` and then sent
    as silence: opening releases the moment before the press too, so the first
    syllable is kept, and the stream keeps its length for the server's VAD.
    """

    def __init__(self, preroll_ms: int = 300, hangover_ms: int = 500, timeout_s: float = 15.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._hold: "collections.deque[np.ndarray]" = collections.deque()
        self._preroll = max(1, int(preroll_ms) // 10)
        self._hangover_s = hangover_ms / 1000.0
        self.timeout_s = float(timeout_s)
        self._open_until = 0.0
        self._clock = clock

    @property
    def is_open(self) -> bool:
        return self._clock() < self._open_until

    def set(self, on: bool) -> None:
        now = self._clock()
        if on:
            self._open_until = now + self.timeout_s
        elif self._open_until > now:
            self._open_until = now + self._hangover_s

    def process(self, frame: np.ndarray) -> List[np.ndarray]:
        """The frames to send for this mic frame: none, one, or the pre-roll at opening."""
        if self.is_open:
            out = list(self._hold) + [frame]
            self._hold.clear()
            return out
        self._hold.append(frame)
        if len(self._hold) > self._preroll:
            return [np.zeros_like(self._hold.popleft())]
        return []


class VoiceSession:
    """Streams cleaned mic audio up, plays the answer, handles barge-in."""

    def __init__(
        self,
        client: RealtimeClient,
        engine: Optional[Any] = None,
        on_user_text: Optional[TextCallback] = None,
        on_assistant_text: Optional[TextCallback] = None,
        on_assistant_delta: Optional[TextCallback] = None,
        on_tool_call: Optional[ToolCall] = None,
        send_chunk_ms: int = 40,
        speak_only: bool = False,
        barge_in: str = "any",
        barge_in_margin_db: float = 15.0,
        push_to_talk: bool = False,
        talk_timeout_s: float = 15.0,
    ) -> None:
        """``speak_only``: the model never answers on its own; ``say`` runs out of
        band (without the conversation, so it cannot drift into replying to
        what the person said) and any response it did not request is
        cancelled and muted. ``barge_in``: one of ``BARGE_IN_MODES``.
        ``push_to_talk``: the mic is sent only while ``set_talk(True)``."""
        if barge_in not in BARGE_IN_MODES:
            raise ValueError(f"unknown barge-in mode {barge_in!r}")
        self.client = client
        self.engine = engine
        self.on_user_text = on_user_text
        self.on_assistant_text = on_assistant_text
        self.on_assistant_delta = on_assistant_delta
        self.on_tool_call = on_tool_call
        self._up = StreamResampler(PROC_RATE, API_RATE)
        self._chunk: List[np.ndarray] = []
        self._chunk_frames = max(1, send_chunk_ms // 10)
        self._lock = threading.Lock()
        self.response_active = False
        self.interruptions = 0
        self._interrupted: "set[str]" = set()  # items whose remaining audio must not play
        self._say_queue: "collections.deque[str]" = collections.deque()
        self.speak_only = speak_only
        self._rejected: "set[str]" = set()  # responses nobody asked for (speak_only)
        # speak_only: the person's language; say() translates into it when needed
        self.language = "English"
        self.barge_in = barge_in
        self.room = RoomLevel(barge_in_margin_db)
        self._barge_check_until = 0.0
        self.talk: Optional[TalkGate] = TalkGate(timeout_s=talk_timeout_s) if push_to_talk else None

        client.on("response.created", self._on_response_created)
        client.on("response.output_audio.delta", self._on_audio_delta)
        client.on("response.output_audio_transcript.delta", self._on_transcript_delta)
        client.on("response.output_audio_transcript.done", self._on_transcript_done)
        client.on("response.output_text.delta", self._on_transcript_delta)
        client.on("response.output_text.done", self._on_text_done)
        client.on("conversation.item.input_audio_transcription.completed", self._on_user_transcript)
        client.on("input_audio_buffer.speech_started", self._on_speech_started)
        client.on("input_audio_buffer.speech_stopped", self._on_speech_stopped)
        client.on("response.done", self._on_response_done)
        client.on("connection.closed", self._on_connection_closed)
        client.on("session.updated", lambda event: self._flush_say())  # also a new session after a reconnect
        if engine is not None:
            engine.add_frame_callback(self._on_mic_frame)

    # ------------------------------------------------------------ upstream

    def _on_mic_frame(self, frame: np.ndarray, info: Dict[str, object]) -> None:
        self.room.add(frame)
        if self._barge_check_until:
            self._check_barge_in()
        frames = [frame] if self.talk is None else self.talk.process(frame)
        if not self.client.session_ready.is_set() or self.client.closed.is_set():
            return
        for out in frames:
            with self._lock:
                self._chunk.append(out)
                if len(self._chunk) < self._chunk_frames:
                    continue
                pcm16k = np.concatenate(self._chunk)
                self._chunk = []
                pcm24k = self._up.process(pcm16k)
            self.client.append_audio(pcm24k)

    def set_talk(self, on: bool) -> None:
        """Push-to-talk switch (no effect without ``push_to_talk``)."""
        if self.talk is not None:
            self.talk.set(on)

    # ------------------------------------------------------------ downstream

    def _on_response_created(self, event: Dict[str, Any]) -> None:
        self.response_active = True
        response = event.get("response", {})
        if self.speak_only and (response.get("metadata") or {}).get("source") != "say":
            self._rejected.add(response.get("id", ""))
            self.client.cancel_response(response.get("id", ""))  # not the sentence being said
            _LOG.info("cancelled a response nobody asked for")

    def _muted(self, event: Dict[str, Any]) -> bool:
        return event.get("response_id", "") in self._rejected or event.get("item_id", "") in self._interrupted

    # ------------------------------------------------------------ robot side

    def say(self, text: str) -> None:
        """Speak ``text`` verbatim in the session's voice (e.g. from /speak).

        Queued while an answer is streaming so it never cuts the model off.
        The sentence becomes part of the conversation, so the model knows
        what the robot said.
        """
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._say_queue.append(text)
            busy = self.response_active
        if not busy:
            self._flush_say()

    def note(self, text: str) -> None:
        """Add silent context (robot status) the model can use later."""
        self.client.send({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "system", "content": [{"type": "input_text", "text": text}]},
        })

    def _on_connection_closed(self, event: Dict[str, Any]) -> None:
        """No answer streams any more; sentences to say wait for the next session (reconnect)."""
        with self._lock:
            self.response_active = False
            self._chunk = []

    def _flush_say(self) -> None:
        with self._lock:
            if self.response_active or not self._say_queue or not self.client.session_ready.is_set() \
                    or self.client.closed.is_set():
                return  # a sentence waits for the next session instead of going into a closed socket
            text = self._say_queue.popleft()
            self.response_active = True  # until response.created/done arrive
        if self.speak_only:
            self.client.send({
                "type": "response.create",
                "response": {
                    "conversation": "none",
                    "input": [],
                    "metadata": {"source": "say"},
                    "instructions": self._say_instructions(text),
                    "tool_choice": "none",
                },
            })
            return
        self.client.send({
            "type": "response.create",
            "response": {
                "instructions": ("Read the following text aloud exactly as written, in the first person, "
                                 "without adding, removing or commenting on anything:\n" + text),
                "tool_choice": "none",
            },
        })

    def _say_instructions(self, text: str) -> str:
        if self.language == "English":
            # a lead-in like "Alright, let me think this through out loud." slipped in
            # with a looser wording: act as a text-to-speech engine, nothing else
            return ("You are a text-to-speech engine now. Say exactly the text between <text> and </text>, "
                    "word for word, and nothing else: no lead-in, no remark, no answer to it, "
                    "even when it asks a question or says what to do next.\n<text>" + text + "</text>")
        return (f"Speak {self.language}. Say the following text aloud in {self.language}, in the first person: "
                f"parts already in {self.language} exactly as written, anything else translated faithfully into "
                f"{self.language}, keeping names and numbers. You are a robot arm on a mobile base: pick means "
                "grasp an object (Spanish agarrar or recoger, German greifen or aufnehmen), place means put down "
                "(colocar, abstellen), insert means put into a box (meter, einlegen), perceive means look at "
                "(percibir, erfassen), table means mesa or Tisch. Do not add, remove or comment on anything, no lead-in and no answer to it:\n<text>" + text + "</text>")

    def _on_audio_delta(self, event: Dict[str, Any]) -> None:
        # deltas of an interrupted answer can still be in flight: drop them
        if self.engine is not None and not self._muted(event):
            self.engine.play(RealtimeClient.decode_audio(event), API_RATE, event.get("item_id", ""))

    def _on_transcript_delta(self, event: Dict[str, Any]) -> None:
        if self.on_assistant_delta and not self._muted(event):
            self.on_assistant_delta(event.get("delta", ""))

    def _on_transcript_done(self, event: Dict[str, Any]) -> None:
        if self.on_assistant_text and not self._muted(event):
            self.on_assistant_text(event.get("transcript", ""))

    def _on_text_done(self, event: Dict[str, Any]) -> None:
        if self.on_assistant_text:
            self.on_assistant_text(event.get("text", ""))

    def _on_user_transcript(self, event: Dict[str, Any]) -> None:
        if self.on_user_text:
            self.on_user_text(event.get("transcript", "").strip())

    def _on_speech_started(self, event: Dict[str, Any]) -> None:
        """Barge-in: the person talks while the robot speaks -> stop talking (see ``BARGE_IN_MODES``)."""
        if self.engine is None or not self.engine.is_playing() or self.barge_in == "stop":
            return
        if self.barge_in == "any":
            self.interrupt()
            return
        # logged for tuning --barge-in-margin-db with the next crowd
        _LOG.info("speech over the robot: %.0f dB above the room (barge-in from %.0f dB)",
                  self.room.above_floor_db(), self.room.margin_db)
        if self.room.loud():
            self.interrupt("loud voice")
        else:
            self._barge_check_until = time.monotonic() + BARGE_IN_CHECK_S

    def _on_speech_stopped(self, event: Dict[str, Any]) -> None:
        self._barge_check_until = 0.0

    def _check_barge_in(self) -> None:
        """Speech the server heard over the robot: stop the robot once it gets loud (mic thread)."""
        if time.monotonic() > self._barge_check_until or self.engine is None or not self.engine.is_playing():
            self._barge_check_until = 0.0
        elif self.room.loud():
            self._barge_check_until = 0.0
            self.interrupt("loud voice")

    def interrupt(self, reason: str = "") -> bool:
        """Stop the robot's speech now; False if it was not speaking."""
        if self.engine is None or not self.engine.is_playing():
            return False
        item = self.engine.stop_playback()
        self.interruptions += 1
        if item:
            self._interrupted.add(item)
        if item and item != "calibration":
            played = self.engine.playback.played_ms(item)
            if not self.speak_only:  # out-of-band items are not in the conversation
                self.client.truncate(item, played)
                if self.barge_in != "any":
                    self.client.cancel_response()  # the server interrupts it only for "any"
            _LOG.info("barge-in: stopped robot speech after %d ms%s", played, f" ({reason})" if reason else "")
        return True

    def _on_response_done(self, event: Dict[str, Any]) -> None:
        self.response_active = False
        response = event.get("response", {})
        calls = [o for o in response.get("output", []) if o.get("type") == "function_call"]
        if not calls:
            self._flush_say()
            return
        for call in calls:
            name = call.get("name", "")
            try:
                args = json.loads(call.get("arguments") or "{}")
            except ValueError:
                args = {"raw": call.get("arguments")}
            if self.on_tool_call is None:
                result: Any = {"error": f"no handler for tool {name}"}
            else:
                try:
                    result = self.on_tool_call(name, args)
                except Exception as exc:  # report tool failures to the model
                    result = {"error": str(exc)}
            self.client.send_function_output(call.get("call_id", ""), result, respond=False)
        self.client.send({"type": "response.create"})
