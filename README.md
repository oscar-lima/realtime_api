# realtime_api

Talk to Mobipick through the OpenAI Realtime API (speech in, speech out),
routed through LiteLLM. Its main job is to stop the robot's own voice from
getting back into the microphone. Without that, the Realtime API hears the
robot, takes it for the user, and interrupts itself or answers itself.

Two ways to use it:

- **Mobipick voice agent** (`scripts/realtime_voice_agent`): an optional
  speech front end of the mobipick_gpt agents. Everything you say goes to
  `/recognized_speech` and whatever the agents say on `/speak` is spoken in
  the realtime voice. Tested live in simulation: you talk and the robot moves.
- **Standalone demo** (`scripts/realtime_voice_demo`): a conversation without
  ROS, a typed-chat check of the API route, and an echo test that needs no
  API at all.

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
2. **Echo delay.** Device-reported latencies are often wrong (PipeWire was
   30–35 ms off), so the real echo delay is measured and cached per mic and
   speaker pair in `~/.cache/realtime_api/echo_delay.json`. A background
   GCC-PHAT tracker checks it during robot speech, corrects drift, and the
   corrected value is saved on exit. `--calibrate` measures it at startup
   with a few short noise bursts instead; it is rarely needed.
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

On the host (Ubuntu 24.04, Python 3.12), once:

```bash
cd ~/ros1_ws/amenable_ws/src/realtime_api
scripts/install_host.sh        # .venv with numpy scipy sounddevice websocket-client livekit roslibpy
sudo apt install libportaudio2 libspeexdsp1   # only if the script reports them missing
```

All three echo cancellers work there. In the Noetic container (Python 3.8),
`pip3 install sounddevice roslibpy` is enough; Speex and NLMS work there,
WebRTC does not.

## Mobipick voice agent

In the Mobipick Labs GUI, choose the voice in the `voice_agent` dropdown
(`off` is the classic text workflow, `cedar` is the default robot voice,
`echo` the alternative) and press Auto Launch, or press **Voice Agent** once
GPT Robot Demo and LiteLLM are running. By hand:

```bash
scripts/run_voice_agent.sh                       # cedar, laptop mic and speaker
scripts/run_voice_agent.sh --voice echo --input ALU1 --output Pebble
scripts/voice_sampler                            # hear cedar and echo
```

`run_voice_agent.sh` uses the venv, the LiteLLM master key and LiteLLM on
:4000 (`gpt-realtime-2.1-mini`, or `--model gpt-realtime-2.1`). It finds
GPT Robot Demo's rosbridge on the `mobipick` Docker network by itself and
waits up to 30 s for it.

| topic | type | direction |
|---|---|---|
| `/recognized_speech` | String | out: your utterances (requests, events, answers) |
| `/speak`, `/realtime/say` | String | in: spoken verbatim in the realtime voice |
| `/mobipick_gpt/gpt_debug` | String | in: action progress, silent context ("what are you doing?") |
| `/mobipick_gpt/busy` | Bool | in: an order runs, utterances go to it as events |
| `/realtime/is_speaking` | Bool | out: robot voice audible (mobipick_gpt `listen` waits for it) |
| `/realtime/user_transcript`, `/realtime/robot_transcript` | String | out: transcripts |

Two modes (`--mode`, or `REALTIME_MODE`):

- **bridge** (default): a pure speech bridge. Every recognized utterance goes
  verbatim to `/recognized_speech`, unfiltered, and the mobipick_gpt router,
  chatbot and planner decide what to do and what to say. The realtime model
  never replies on its own (`create_response: false`); it only speaks
  `/speak` and `/realtime/say`. Transcription uses `gpt-4o-transcribe` with a
  hint listing the robot's objects and places (`--transcription-model`).
- **agent**: the realtime model is Mobipick's persona
  (`mobipick_gpt/config/prompts/realtime_voice.txt`, with the static facts of
  `chatbot.txt` filled in by `src/realtime_api/mobipick_prompt.py`). It
  answers small talk itself, speaks in the first person, and calls its only
  tool, `execute`, for orders and live-state questions.

In both modes, while an order runs every utterance also reaches that order as
an event: mobipick_api buffers `/recognized_speech`, and the planner reads it
with `check_for_events()` (a "cancel", "stop" or new information) or takes it
as the answer to its question in `listen()`. Noise transcribed as filler or
non-Latin text is dropped. Turn detection is semantic VAD with `low`
eagerness, which waits for complete sentences.

## Standalone demo

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

### 2. LiteLLM

The main Mobipick proxy (GUI button LiteLLM, port 4000) already serves
`gpt-realtime-2.1-mini` and `gpt-realtime-2.1`. For a standalone proxy with
only these aliases, run `scripts/run_litellm_realtime.sh` (port 4001,
`config/litellm_realtime.yaml`) and add `--litellm-url http://127.0.0.1:4001/v1`.

### 3. Talk

```bash
scripts/realtime_voice_demo --text                                   # typed chat, checks the route
scripts/realtime_voice_demo --input ALU1 --output Pebble --meter
```

Useful flags (demo and voice agent):

- `--voice cedar|echo`
- `--aec webrtc|speex|nlms|none` and `--gate smart|half|full`
- `--vad semantic_vad|server_vad` and `--vad-eagerness low|medium|high|auto`
- `--model gpt-realtime-2.1` for the full model (default `gpt-realtime-2.1-mini`)
- `--backend openai` to skip LiteLLM
- `--language ''` to auto-detect the transcript language (default `en`)
- `--debug-dir DIR` saves `mic.wav`, `reference.wav`, `aec_out.wav` and
  `sent.wav` on exit. `sent.wav` is exactly what the API heard.
- `--meter` prints mic, AEC and reference levels, ERLE, gate state and the
  echo lag every 2 s.

The session asks the API for `far_field` input noise reduction and
`gpt-4o-mini-transcribe` user transcripts. Semantic VAD with
`interrupt_response` handles turn-taking. Barge-in stops playback and
truncates the answer to what was actually heard.

## Tests

```bash
python3 -m pytest          # 30 tests, no audio hardware or network needed
```

The tests simulate a reverberant room with real Piper speech
(`test/data/*.wav`) and fake the websocket. They pass on Python 3.12 (host)
and Python 3.8 (Mobipick Noetic image).

## Layout

```
src/realtime_api/
  echo_cancel.py      AEC backends (webrtc/speex/nlms), GCC-PHAT delay estimate, ERLE
  duplex_audio.py     DuplexAudioEngine, ReferenceTimeline, ClockTracker, PlaybackQueue, EchoGate
  realtime_client.py  websocket-client Realtime API client (GA + beta events), LiteLLM/OpenAI endpoints
  voice_session.py    glue: mic -> API, API audio -> speaker, barge-in, tool calls, say() and note()
  ros_bridge.py       std_msgs String/Bool topics over rosbridge (roslibpy)
  mobipick_prompt.py  voice prompt with the mobipick_gpt chatbot facts filled in
  app.py              shared command line, engine setup, echo delay cache
  audio_utils.py      resampling, framing, wav io
scripts/
  realtime_voice_agent    Mobipick voice agent (ROS over rosbridge)
  run_voice_agent.sh      host launcher used by the GUI button (venv, keys, rosbridge discovery)
  install_host.sh         one-time host venv
  voice_sampler           hear the robot voices
  realtime_voice_demo     standalone demo (voice / --text / --echo-test)
  aec_loopback_test       hardware AEC benchmark across backends
  run_litellm_realtime.sh standalone LiteLLM proxy with the realtime aliases
config/litellm_realtime.yaml
```

## Notes

- **Other voices on the same speaker.** Audio played by another process
  (for example `piper_tts_node` through `aplay`) is not in the canceller's
  reference. Send it through `/realtime/say` or `DuplexAudioEngine.play()`
  instead. As a fallback, `engine.set_external_speaking(True)` treats that
  time like `half` mode.
- **WebRTC in the robot container.** livekit needs Python ≥ 3.9. Run the
  voice agent on the host or in a small 22.04+ container next to the Noetic
  one, or use `--aec speex` or `--aec nlms` inside Noetic.
- **While an order runs**, mobipick_gpt starts no new workflow; your words
  reach the running order as events, so "cancel" works when the planner
  checks for events between actions.
