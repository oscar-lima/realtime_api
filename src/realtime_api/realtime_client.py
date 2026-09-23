"""Minimal OpenAI Realtime API client over a WebSocket.

Runs on Python 3.8 (``websocket-client``, already in the Mobipick Noetic
image) and speaks the GA protocol (``session.type = "realtime"``) that the
LiteLLM proxy bridges by default. Beta event names (``response.audio.delta``
etc.) are accepted as well, so an older proxy or a forced
``OpenAI-Beta: realtime=v1`` header keeps working.

Endpoints:
  LiteLLM proxy  ws://<host>:4000/v1/realtime?model=<alias>   (Bearer = LiteLLM master key)
  OpenAI direct  wss://api.openai.com/v1/realtime?model=<model> (Bearer = OpenAI key)
"""

from __future__ import annotations

import base64
import itertools
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

import numpy as np

_LOG = logging.getLogger(__name__)

API_RATE = 24000  # the Realtime API speaks 24 kHz mono PCM16

EventHandler = Callable[[Dict[str, Any]], None]

_BENIGN_ERRORS = {"conversation_already_has_active_response", "response_cancel_not_active"}

# GA name -> beta alias, so handlers can subscribe with GA names only
_BETA_ALIASES = {
    "response.audio.delta": "response.output_audio.delta",
    "response.audio.done": "response.output_audio.done",
    "response.audio_transcript.delta": "response.output_audio_transcript.delta",
    "response.audio_transcript.done": "response.output_audio_transcript.done",
    "response.text.delta": "response.output_text.delta",
    "response.text.done": "response.output_text.done",
}


def read_key_file(path: str) -> str:
    try:
        with open(os.path.expanduser(path)) as handle:
            return handle.readline().strip()
    except OSError:
        return ""


def resolve_endpoint(
    backend: str,
    model: str,
    litellm_url: str = "",
    api_key: str = "",
) -> "tuple[str, str]":
    """Return (websocket url, bearer token) for ``litellm`` or ``openai``.

    Keys come from ``api_key``, else the environment (``LITELLM_API_KEY`` /
    ``OPENAI_API_KEY``), else the usual key files next to the workspace.
    """
    src = os.path.expanduser("~/ros1_ws/amenable_ws/src")
    if backend == "litellm":
        base = litellm_url or os.environ.get("LITELLM_BASE_URL", "http://127.0.0.1:4000/v1")
        base = base.rstrip("/")
        if base.startswith("http"):
            base = "ws" + base[4:]
        if not base.endswith("/v1"):
            base += "/v1"
        key = api_key or os.environ.get("LITELLM_API_KEY") or read_key_file(os.path.join(src, "litellm_master_key"))
        return f"{base}/realtime?model={quote(model)}", key
    if backend == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY") or read_key_file(os.path.join(src, "openai_api_key"))
        return f"wss://api.openai.com/v1/realtime?model={quote(model)}", key
    raise ValueError(f"unknown backend {backend!r} (litellm or openai)")


def build_session(
    instructions: str,
    voice: str = "marin",
    tools: Optional[List[Dict[str, Any]]] = None,
    audio: bool = True,
    vad: str = "semantic_vad",
    vad_eagerness: str = "auto",
    server_vad_threshold: float = 0.6,
    transcription_model: str = "gpt-4o-mini-transcribe",
    noise_reduction: Optional[str] = "far_field",
    language: str = "en",
) -> Dict[str, Any]:
    """GA ``session`` object for ``session.update``."""
    session: Dict[str, Any] = {
        "type": "realtime",
        "instructions": instructions,
        "output_modalities": ["audio"] if audio else ["text"],
    }
    if audio:
        if vad == "semantic_vad":
            turn = {"type": "semantic_vad", "eagerness": vad_eagerness,
                    "create_response": True, "interrupt_response": True}
        elif vad == "server_vad":
            turn = {"type": "server_vad", "threshold": server_vad_threshold, "prefix_padding_ms": 300,
                    "silence_duration_ms": 500, "create_response": True, "interrupt_response": True}
        else:
            turn = None
        audio_in: Dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": API_RATE},
            "turn_detection": turn,
            "transcription": ({"model": transcription_model, "language": language} if language
                              else {"model": transcription_model}) if transcription_model else None,
        }
        if noise_reduction:
            audio_in["noise_reduction"] = {"type": noise_reduction}
        session["audio"] = {
            "input": audio_in,
            "output": {"format": {"type": "audio/pcm", "rate": API_RATE}, "voice": voice},
        }
    if tools:
        session["tools"] = tools
        session["tool_choice"] = "auto"
    return session


def ga_to_beta_session(session: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a GA session object to the beta shape (legacy proxies)."""
    beta: Dict[str, Any] = {
        "instructions": session.get("instructions", ""),
        "modalities": ["text", "audio"] if session.get("output_modalities") == ["audio"] else ["text"],
    }
    audio = session.get("audio") or {}
    if audio:
        beta["input_audio_format"] = "pcm16"
        beta["output_audio_format"] = "pcm16"
        beta["voice"] = audio.get("output", {}).get("voice", "alloy")
        audio_in = audio.get("input", {})
        beta["turn_detection"] = audio_in.get("turn_detection")
        beta["input_audio_transcription"] = audio_in.get("transcription")
        if audio_in.get("noise_reduction"):
            beta["input_audio_noise_reduction"] = audio_in["noise_reduction"]
    if session.get("tools"):
        beta["tools"] = session["tools"]
        beta["tool_choice"] = session.get("tool_choice", "auto")
    return beta


class RealtimeClient:
    """Thread-based Realtime API connection with event callbacks."""

    def __init__(self, url: str, token: str, extra_headers: Optional[List[str]] = None, timeout: float = 15.0) -> None:
        self.url = url
        self._headers = [f"Authorization: Bearer {token}"] + list(extra_headers or [])
        self._timeout = timeout
        self._ws = None
        self._send_lock = threading.Lock()
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._reader: Optional[threading.Thread] = None
        self._ids = itertools.count(1)
        self.connected = threading.Event()
        self.closed = threading.Event()
        self.session_ready = threading.Event()
        self.protocol = "ga"
        self.last_error: Optional[Dict[str, Any]] = None
        self._session: Optional[Dict[str, Any]] = None

    # -------------------------------------------------------------- events

    def on(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to an event type (GA names; '*' = every event)."""
        self._handlers.setdefault(event_type, []).append(handler)

    def _dispatch(self, event: Dict[str, Any]) -> None:
        etype = event.get("type", "")
        etype = _BETA_ALIASES.get(etype, etype)
        event["type"] = etype
        if etype == "session.created":
            session = event.get("session", {})
            self.protocol = "ga" if session.get("type") == "realtime" else "beta"
            _LOG.info("realtime session created (%s protocol, model %s)", self.protocol, session.get("model"))
            if self._session is not None:
                self._send_session()
        elif etype == "session.updated":
            self.session_ready.set()
        elif etype == "error":
            error = event.get("error", event)
            if error.get("code") in _BENIGN_ERRORS:
                # e.g. server VAD saw another turn end while an answer is still
                # streaming; the server just skips the extra response
                _LOG.debug("realtime: %s", error.get("message"))
            else:
                self.last_error = error
                _LOG.error("realtime error: %s", error)
        for handler in self._handlers.get(etype, []) + self._handlers.get("*", []):
            try:
                handler(event)
            except Exception:
                _LOG.exception("handler for %s failed", etype)

    # -------------------------------------------------------------- connection

    def connect(self, session: Optional[Dict[str, Any]] = None) -> None:
        """Open the socket; ``session`` is sent once the server greets us."""
        import websocket  # websocket-client

        self._session = session
        self._ws = websocket.create_connection(self.url, header=self._headers, timeout=self._timeout)
        self._ws.settimeout(None)
        self.connected.set()
        self._reader = threading.Thread(target=self._read_loop, name="realtime-rx", daemon=True)
        self._reader.start()

    def wait_ready(self, timeout: float = 15.0) -> bool:
        """Wait for ``session.updated``; False on timeout, error or close."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.session_ready.wait(0.05):
                return True
            if self.closed.is_set() or self.last_error is not None:
                return False
        return False

    def _read_loop(self) -> None:
        try:
            while True:
                raw = self._ws.recv()
                if not raw:
                    break
                try:
                    event = json.loads(raw)
                except ValueError:
                    _LOG.warning("non-JSON message: %r", raw[:200])
                    continue
                self._dispatch(event)
        except Exception as exc:
            if not self.closed.is_set():
                _LOG.warning("realtime connection closed: %s", exc)
        finally:
            self.closed.set()
            self._dispatch({"type": "connection.closed"})

    def close(self) -> None:
        self.closed.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass

    def send(self, event: Dict[str, Any]) -> None:
        if self._ws is None or self.closed.is_set():
            return
        event.setdefault("event_id", f"evt_client_{next(self._ids)}")
        data = json.dumps(event)
        with self._send_lock:
            self._ws.send(data)

    def _send_session(self) -> None:
        session = self._session or {}
        if self.protocol == "beta":
            session = ga_to_beta_session(session)
        self.send({"type": "session.update", "session": session})

    def update_session(self, session: Dict[str, Any]) -> None:
        self._session = session
        self._send_session()

    # -------------------------------------------------------------- helpers

    def append_audio(self, pcm24k: np.ndarray) -> None:
        self.send({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm24k.astype(np.int16).tobytes()).decode("ascii"),
        })

    def send_text(self, text: str, respond: bool = True) -> None:
        self.send({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
        })
        if respond:
            self.send({"type": "response.create"})

    def truncate(self, item_id: str, audio_end_ms: int) -> None:
        """Tell the server how much of an assistant audio item was heard."""
        self.send({"type": "conversation.item.truncate", "item_id": item_id,
                   "content_index": 0, "audio_end_ms": int(max(0, audio_end_ms))})

    def cancel_response(self) -> None:
        self.send({"type": "response.cancel"})

    def send_function_output(self, call_id: str, output: Any, respond: bool = True) -> None:
        self.send({
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": call_id,
                     "output": output if isinstance(output, str) else json.dumps(output)},
        })
        if respond:
            self.send({"type": "response.create"})

    @staticmethod
    def decode_audio(event: Dict[str, Any]) -> np.ndarray:
        return np.frombuffer(base64.b64decode(event.get("delta", "")), dtype=np.int16)
