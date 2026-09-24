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
    ) -> None:
        """``speak_only``: the model never answers on its own; ``say`` runs out of
        band (without the conversation, so it cannot drift into replying to
        what the person said) and any response it did not request is
        cancelled and muted."""
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

        client.on("response.created", self._on_response_created)
        client.on("response.output_audio.delta", self._on_audio_delta)
        client.on("response.output_audio_transcript.delta", self._on_transcript_delta)
        client.on("response.output_audio_transcript.done", self._on_transcript_done)
        client.on("response.output_text.delta", self._on_transcript_delta)
        client.on("response.output_text.done", self._on_text_done)
        client.on("conversation.item.input_audio_transcription.completed", self._on_user_transcript)
        client.on("input_audio_buffer.speech_started", self._on_speech_started)
        client.on("response.done", self._on_response_done)
        if engine is not None:
            engine.add_frame_callback(self._on_mic_frame)

    # ------------------------------------------------------------ upstream

    def _on_mic_frame(self, frame: np.ndarray, info: Dict[str, object]) -> None:
        if not self.client.session_ready.is_set() or self.client.closed.is_set():
            return
        with self._lock:
            self._chunk.append(frame)
            if len(self._chunk) < self._chunk_frames:
                return
            pcm16k = np.concatenate(self._chunk)
            self._chunk = []
            pcm24k = self._up.process(pcm16k)
        self.client.append_audio(pcm24k)

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

    def _flush_say(self) -> None:
        with self._lock:
            if self.response_active or not self._say_queue:
                return
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
        """Barge-in: the user talks while the robot speaks -> stop talking."""
        if self.engine is None or not self.engine.is_playing():
            return
        item = self.engine.stop_playback()
        self.interruptions += 1
        if item:
            self._interrupted.add(item)
        if item and item != "calibration":
            played = self.engine.playback.played_ms(item)
            if not self.speak_only:  # out-of-band items are not in the conversation
                self.client.truncate(item, played)
            _LOG.info("barge-in: stopped robot speech after %d ms", played)

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
