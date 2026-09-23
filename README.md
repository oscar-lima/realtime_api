# realtime_api

Talk to Mobipick through the OpenAI Realtime API (speech in, speech out),
routed through LiteLLM. Its main job is to stop the robot's own voice from
getting back into the microphone. Without that, the Realtime API hears the
robot, takes it for the user, and interrupts itself or answers itself.

Status: a standalone dry-run demo with no ROS. Everything was tested except
a live conversation: the OpenAI project has hit its spend limit
(`project_spend_limit_exceeded`). That error comes back correctly through
LiteLLM, so the route itself works. The `send_robot_command` tool
only prints the command. Plugging it into `mobipick_gpt` as an optional agent
is the next step (see [Next steps](#next-steps)).

## How the echo is removed

One process plays the robot's voice and also captures the microphone, so it
knows exactly which samples went to the speaker and when.

```
Realtime API audio (24 kHz) ─► playback queue ─► OutputStream ─► speaker
                                                     │ DAC timestamp
                                                     ▼
                                        ReferenceTimeline (16 kHz)
                                                     │ read(t − echo_delay + lead)
microphone ─► InputStream ─(ADC timestamp t)─► AEC(mic, reference) ─► EchoGate ─► Realtime API
```

1. **Timestamp alignment.** Every speaker sample is stored with its DAC time
   and every mic frame gets its ADC time. The mic and speaker can be different
   USB devices (ALU1 mic, Pebble speaker).
2. **Calibration.** At startup, a few short noise bursts (about 1 s each)
   measure where the echo really lands. Device-reported latencies are often
   wrong: PipeWire was 30–35 ms off. A background GCC-PHAT tracker keeps
   checking this during robot speech and corrects drift.
3. **Acoustic echo canceller** (`--aec`):
   - `webrtc`: WebRTC AEC3 via `livekit` (`pip install livekit`, Python ≥ 3.9). Best.
   - `speex`: SpeexDSP through ctypes on `libspeexdsp.so.1`. Already in the
     Mobipick Noetic image, works on Python 3.8.
   - `nlms`: a pure numpy frequency-domain NLMS with two-path double-talk
     handling and residual echo suppression. Always available.
   - `auto` (the default) picks the first one that loads.
4. **Echo gate** (`--gate`):
   - `smart` (default): while the robot talks, a frame is forwarded only if
     it is clearly louder, relative to what the speaker is playing, than the
     residual echo learned so far. That means a person is talking over the
     robot, which is barge-in. Otherwise the gate sends silence.
   - `half`: mute the mic while the robot talks.
   - `full`: trust the AEC completely.
5. **Barge-in.** When the server VAD reports user speech while the robot is
   talking, playback stops at once and the assistant message is truncated to
   what was actually played.

### Measured on real hardware

Setup: a Pebble V3 speaker (the robot's model) and a laptop's built-in mic,
with a loud fan in the room. Each run plays 12.6 s of Piper robot speech with
the gate in smart mode (`realtime_voice_demo --echo-test`). The ranges come
from 3–6 runs per backend with the final code. The room and the mic will
change these numbers, so run the same test on the robot.

| backend | echo removed (ERLE) | robot-speech frames sent to API |
|---------|--------------------:|---------------------------------:|
| webrtc  | 26–47 dB | 0 % in 5 of 6 runs, 9 % once |
| speex   | 14–30 dB | 0–10 % |
| nlms    | 12–30 dB | 6–21 % |
| none    | 0 dB     | 8 % (the smart gate alone), 98 % with no gate |

Speex and NLMS need about 3 s of robot speech to converge. After that they
stay converged for the whole session.

What leaks through with WebRTC is around −70 dBFS, well below anything
server VAD should react to. That has not been checked against the real API
yet (see Status above). If the robot still interrupts itself, use
`--gate half`: nothing is sent while the robot talks, at the cost of
barge-in.

This table shows how much user speech gets through while the robot talks.
The recording was replayed with a user voice mixed in digitally, the
cancellers were already converged, and the gate used its final defaults:

| backend | echo frames leaked | user as loud as echo | user −6 dB | user −12 dB |
|---------|------:|------:|------:|------:|
| webrtc  | 0.9 % | 93 % | 62 % | 41 % |
| speex   | 6.9 % | 98 % | 95 % | 94 % |
| nlms    | 0.9 % | 98 % | 97 % | 89 % |

WebRTC removes the most echo, but it also turns down a user who speaks
much more quietly than the robot.

PipeWire's system-wide `module-echo-cancel` (`aec_method=webrtc`, PipeWire
1.0.5) removed only about 8 dB in the same setup. The robot's voice should
therefore be played through this engine and not through `aplay` or a second
process.

## Install

```bash
cd ~/ros1_ws/amenable_ws/src/realtime_api
pip install -r requirements.txt          # numpy scipy sounddevice websocket-client (+ livekit on py>=3.9)
sudo apt install libportaudio2 libspeexdsp1   # if missing
```

On the host (Ubuntu 24.04, Python 3.12), all three AEC backends are
available. In the Noetic container (Python 3.8) you also need
`pip3 install sounddevice`. Speex and NLMS are available there, WebRTC is not.

## Run

### 1. Check the echo cancellation (no API, no cost)

```bash
scripts/realtime_voice_demo --list-devices
scripts/realtime_voice_demo --echo-test --input ALU1 --output Pebble
```

This plays a robot sentence and reports how much echo was removed and how
many frames would have reached the API. The target is close to 0 %. To
compare all backends on the same recording and check barge-in, run:

```bash
scripts/aec_loopback_test --input ALU1 --output Pebble --save-dir /tmp/aec_test
scripts/aec_loopback_test --replay /tmp/aec_test --near-db -6    # re-analyse, user 6 dB quieter than echo
```

If it warns that the echo is barely above the room noise, raise the speaker
volume or pass `--output-gain`.

### 2. Start LiteLLM with a realtime model

This uses the same pinned image, OpenAI key file and master key as
`mobipick_gpt`. It listens on port 4001 so it does not clash with the main
proxy on 4000:

```bash
scripts/run_litellm_realtime.sh
```

To serve realtime from the main proxy instead, copy the entries from
`config/litellm_realtime.yaml` into `mobipick_gpt/config/litellm_config.yaml`
and use `--litellm-url http://127.0.0.1:4000/v1`.

### 3. Talk

```bash
scripts/realtime_voice_demo --text --litellm-url http://127.0.0.1:4001/v1      # typed chat, checks the route
scripts/realtime_voice_demo --litellm-url http://127.0.0.1:4001/v1 --input ALU1 --output Pebble --meter
```

Useful flags:

- `--aec webrtc|speex|nlms|none` and `--gate smart|half|full`
- `--vad semantic_vad|server_vad` and `--vad-eagerness low|medium|high|auto`
- `--model gpt-realtime-2.1` for the full model (default `gpt-realtime-2.1-mini`)
- `--backend openai` to skip LiteLLM
- `--debug-dir DIR` saves `mic.wav`, `reference.wav`, `aec_out.wav` and
  `sent.wav` on exit. `sent.wav` is exactly what the API heard.
- `--meter` prints mic, AEC and reference levels, ERLE, gate state and the
  echo lag every 2 s.

The session asks the API for `far_field` input noise reduction and
`gpt-4o-mini-transcribe` user transcripts. Semantic VAD with
`interrupt_response` handles turn-taking.

## Tests

```bash
python3 -m pytest          # 29 tests, no audio hardware or network needed
```

The tests simulate a reverberant room with real Piper speech
(`test/data/*.wav`) and fake the websocket. They pass on Python 3.12 (host)
and Python 3.8 (Mobipick Noetic image).

## Layout

```
src/realtime_api/
  echo_cancel.py      AEC backends (webrtc/speex/nlms), GCC-PHAT delay estimate, ERLE
  duplex_audio.py     DuplexAudioEngine, ReferenceTimeline, PlaybackQueue, EchoGate
  realtime_client.py  websocket-client Realtime API client (GA + beta events), LiteLLM/OpenAI endpoints
  voice_session.py    glue: mic -> API, API audio -> speaker, barge-in, tool calls
  audio_utils.py      resampling, framing, wav io
scripts/
  realtime_voice_demo     standalone demo (voice / --text / --echo-test)
  aec_loopback_test       hardware AEC benchmark across backends
  run_litellm_realtime.sh LiteLLM proxy with the realtime aliases
config/litellm_realtime.yaml
```

## Next steps

- **mobipick_gpt agent.** `VoiceSession` has no ROS dependency. A ROS node
  (or a Pi harness tool) would pass an `on_tool_call` that forwards
  `send_robot_command(command)` to the mobipick_gpt router, the way
  `human_says` does today. It would publish the user and robot transcripts
  and report task progress back with `client.send_text(..., respond=True)`.
- **Other voices on the same speaker.** If `piper_tts_node` also speaks,
  send its audio through `DuplexAudioEngine.play()` so the canceller knows
  it. As a fallback, call `engine.set_external_speaking(True)` while it
  talks, which treats that time like `half` mode.
- **WebRTC in the robot container.** livekit needs Python ≥ 3.9. Run the
  voice process on the host or in a small 22.04+ container next to the
  Noetic one, or use `--aec speex` or `--aec nlms` inside Noetic.
