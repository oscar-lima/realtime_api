"""Command-line plumbing shared by the demo and the ROS voice agent."""

from __future__ import annotations

import argparse
import json
import os
import threading

import numpy as np

from realtime_api.audio_utils import write_wav_mono
from realtime_api.duplex_audio import PROC_RATE, DuplexAudioEngine
from realtime_api.realtime_client import RealtimeClient

# Robot voices: cedar (default) and echo, both male. The API has more
# (alloy, ash, ballad, coral, sage, shimmer, verse, marin), deliberately not offered.
VOICES = ["cedar", "echo"]

DELAY_CACHE = os.path.expanduser("~/.cache/realtime_api/echo_delay.json")


def add_api_args(p: argparse.ArgumentParser) -> None:
    api = p.add_argument_group("api")
    api.add_argument("--backend", choices=["litellm", "openai"], default="litellm")
    api.add_argument("--model", default=os.environ.get("REALTIME_MODEL", "gpt-realtime-2.1-mini"),
                     help="LiteLLM alias or OpenAI model id")
    api.add_argument("--litellm-url", default="", help="default $LITELLM_BASE_URL or http://127.0.0.1:4000/v1")
    api.add_argument("--api-key", default="", help="default $LITELLM_API_KEY / $OPENAI_API_KEY or key files")
    api.add_argument("--voice", default=os.environ.get("REALTIME_VOICE", "cedar"), choices=VOICES,
                     help="robot voice: cedar (default) or echo")
    api.add_argument("--language", default="",
                     help="language hint for the user transcript ('' = auto-detect)")
    api.add_argument("--vad", choices=["semantic_vad", "server_vad"], default="semantic_vad")
    api.add_argument("--vad-eagerness", choices=["low", "medium", "high", "auto"], default="low",
                     help="low waits for complete sentences (fewer misheard fragments)")
    api.add_argument("--instructions-file", default="")
    api.add_argument("--transcription-model", default=os.environ.get("REALTIME_TRANSCRIPTION_MODEL",
                                                                     "gpt-4o-transcribe"),
                     help="model that turns your speech into text ('' disables transcripts)")
    api.add_argument("--no-tools", action="store_true", help="do not offer send_robot_command")


def add_audio_args(p: argparse.ArgumentParser) -> None:
    audio = p.add_argument_group("audio")
    audio.add_argument("--input", default=os.environ.get("REALTIME_INPUT_DEVICE"),
                       help="mic index or name substring (e.g. ALU1)")
    audio.add_argument("--output", default=os.environ.get("REALTIME_OUTPUT_DEVICE"),
                       help="speaker index or name substring (e.g. Pebble)")
    audio.add_argument("--aec", default="auto", choices=["auto", "webrtc", "speex", "nlms", "none"])
    audio.add_argument("--gate", default="smart", choices=["smart", "half", "full"],
                       help="smart: pass user speech over the robot (barge-in), half: mute mic while "
                            "the robot talks, full: trust the AEC completely")
    audio.add_argument("--echo-delay-ms", type=float, default=None,
                       help="default: the value measured last time for this mic/speaker pair, else 0")
    audio.add_argument("--lead-ms", type=float, default=40.0)
    audio.add_argument("--calibrate", action="store_true",
                       help="measure the echo delay with short noise bursts at startup (normally not needed: "
                            "it is cached and tracked during robot speech)")
    audio.add_argument("--output-gain", type=float, default=1.0)
    audio.add_argument("--mic-gain", type=float, default=1.0)
    audio.add_argument("--debug-dir", default="", help="save mic/reference/aec/sent wavs here on exit")
    audio.add_argument("--meter", action="store_true", help="print audio levels every 2 s")


def _delay_cache_key(args: argparse.Namespace) -> str:
    return f"{args.input or 'default'}|{args.output or 'default'}"


def load_echo_delay(args: argparse.Namespace) -> float:
    try:
        with open(DELAY_CACHE) as handle:
            return float(json.load(handle).get(_delay_cache_key(args), 0.0))
    except (OSError, ValueError, TypeError, AttributeError):
        return 0.0


def save_echo_delay(args: argparse.Namespace, engine: DuplexAudioEngine) -> None:
    """Remember the echo delay so the next start needs no calibration bursts."""
    try:
        cache = {}
        if os.path.exists(DELAY_CACHE):
            with open(DELAY_CACHE) as handle:
                cache = json.load(handle)
        cache[_delay_cache_key(args)] = round(engine.echo_delay_s * 1000.0, 1)
        os.makedirs(os.path.dirname(DELAY_CACHE), exist_ok=True)
        with open(DELAY_CACHE, "w") as handle:
            json.dump(cache, handle, indent=1)
    except (OSError, ValueError):
        pass


def make_engine(args: argparse.Namespace) -> DuplexAudioEngine:
    delay_ms = args.echo_delay_ms if args.echo_delay_ms is not None else load_echo_delay(args)
    engine = DuplexAudioEngine(
        input_device=args.input, output_device=args.output, aec_backend=args.aec, gate_mode=args.gate,
        echo_delay_ms=delay_ms, ref_lead_ms=args.lead_ms, output_gain=args.output_gain,
        mic_gain=args.mic_gain, record_debug=bool(args.debug_dir),
    )
    engine.start()
    if args.calibrate:
        print("calibrating echo path (short noise bursts) ...", flush=True)
        lag, conf = engine.calibrate()
        print(f"  echo delay {engine.echo_delay_s * 1000:.0f} ms (lag {lag:.0f} ms, confidence {conf:.0f})"
              + ("" if conf >= 8 else "  <- weak echo; is the speaker on and loud enough?"), flush=True)
        if conf >= 8:
            save_echo_delay(args, engine)
    else:
        print(f"echo delay {delay_ms:.0f} ms (cached; tracked while the robot speaks)", flush=True)
    return engine


def save_debug(engine: DuplexAudioEngine, directory: str) -> None:
    if not directory or not engine.debug_mic:
        return
    os.makedirs(directory, exist_ok=True)
    for name, frames in (("mic", engine.debug_mic), ("reference", engine.debug_ref),
                         ("aec_out", engine.debug_out), ("sent", engine.debug_sent)):
        write_wav_mono(os.path.join(directory, f"{name}.wav"), np.concatenate(frames), PROC_RATE)
    print(f"debug wavs written to {directory}")


def start_meter(engine: DuplexAudioEngine, stop: threading.Event) -> None:
    def loop() -> None:
        while not stop.wait(2.0):
            s = engine.stats
            delay = engine.last_delay
            print(f"  [mic {s['mic_db']:6.1f} dBFS | aec out {s['out_db']:6.1f} | ref {s['ref_db']:6.1f} | "
                  f"ERLE {s['erle_db']:5.1f} dB | gate {'open' if s.get('gate_open') else 'shut'}"
                  + (f" | echo lag {delay[0]:.0f} ms" if delay else "") + "]", flush=True)
    threading.Thread(target=loop, daemon=True).start()


def connect(client: RealtimeClient, url: str, session: dict) -> bool:
    print(f"connecting to {url} ...", flush=True)
    try:
        client.connect(session)
    except Exception as exc:  # handshake refused, proxy down, bad key, ...
        print(f"could not connect: {exc}".split(" -+-+- ")[0])
        if "401" in str(exc) or "403" in str(exc):
            print("check the key: LiteLLM needs its master key ($LITELLM_API_KEY or "
                  "amenable_ws/src/litellm_master_key), OpenAI direct needs $OPENAI_API_KEY")
        return False
    if not client.wait_ready(15):
        print(f"session not ready: {client.last_error or 'connection closed'}")
        client.close()
        return False
    return True
