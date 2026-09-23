"""Full-duplex audio engine: play the robot voice, capture the mic, remove echo.

Why this works where the old whisper_ros pipeline struggled: this engine is
the *only* thing that plays the robot voice, so it knows exactly which
samples went to the loudspeaker and when. Every speaker sample is written
into a ``ReferenceTimeline`` stamped with its DAC time; every mic frame is
stamped with its ADC time. For a mic frame captured at time t the echo
canceller receives the speaker signal that was played ``echo_delay`` before
t, shifted by ``ref_lead_ms`` so the reference always precedes its echo (an
adaptive filter cannot look into the future). It removes the linear echo,
suppresses the residue, and an
``EchoGate`` finally decides whether what is left may be sent upstream.

Mic and speaker may be different USB devices (Mobipick: ALU1 mic + Pebble
speaker). Their timestamps come from the same PortAudio clock, so the
alignment survives separate devices; an optional chirp calibration and a
background GCC-PHAT delay tracker correct wrongly reported latencies and
slow clock drift.

Data flow::

    playback queue (device rate) --> OutputStream --> loudspeaker
            |                         '--> ReferenceTimeline (16 kHz, DAC time)
    microphone --> InputStream --(ADC time)--> processing thread:
            mic frame(t) + timeline.read(t - echo_delay + lead) --> AEC --> EchoGate
            --> on_frame(cleaned 16 kHz int16, info)
"""

from __future__ import annotations

import collections
import logging
import math
import queue
import threading
import time
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from realtime_api.audio_utils import FrameChunker, StreamResampler, dbfs, rms, to_int16
from realtime_api.echo_cancel import EchoCanceller, estimate_delay, make_echo_canceller

_LOG = logging.getLogger(__name__)

PROC_RATE = 16000
FRAME = PROC_RATE // 100  # 10 ms


ECHO_ACTIVE_RMS = 30.0  # about -61 dBFS: quieter speaker output is ignored
ECHO_TAIL_S = 0.4  # how long after the speaker stops the room may still ring


def speaker_envelope(recent: np.ndarray) -> float:
    """Loudest 10 ms RMS in a stretch of speaker signal."""
    n = (recent.size // FRAME) * FRAME
    if n == 0:
        return 0.0
    frames = recent[:n].reshape(-1, FRAME).astype(np.float64)
    return float(np.sqrt(np.max(np.mean(frames ** 2, axis=1))))


# --------------------------------------------------------------------------
# reference timeline
# --------------------------------------------------------------------------


class ClockTracker:
    """Map a device's continuous sample count onto stream time.

    Callback timestamps jitter by several milliseconds (PipeWire especially)
    while the sample clock only drifts by a few ppm. The tracked time
    therefore advances purely by sample count; a linear echo canceller
    cannot follow even one-sample wobbles of the alignment. The jitter is
    averaged, and only when the averaged offset exceeds ``step_ms`` (latency
    really changed, or accumulated clock drift) is it applied, as one rare
    discrete step. During the first ``settle_s`` (streams just opened,
    PipeWire still renegotiating latency) the offset is followed freely,
    and dropouts beyond ``jump_ms`` re-anchor immediately.
    """

    def __init__(self, jump_ms: float = 40.0, step_ms: float = 3.0, settle_updates: int = 100) -> None:
        self.jump_s = jump_ms / 1000.0
        self.step_s = step_ms / 1000.0
        self.settle_updates = int(settle_updates)
        self.t: Optional[float] = None  # stream time of the next sample
        self._offset = 0.0  # averaged (reported - tracked)
        self._n = 0
        self.reanchors = 0
        self.steps = 0

    def update(self, t_reported: float) -> float:
        """Feed the timestamp reported for the next sample; returns the time to use."""
        if self.t is None or abs(t_reported - self.t) > self.jump_s:
            if self.t is not None:
                self.reanchors += 1
            self.t = t_reported
            self._offset = 0.0
            self._n = 0
            return self.t
        self._n += 1
        err = t_reported - self.t
        if self._n <= self.settle_updates:
            self.t += err / (self._n + 1)  # running mean while settling
            return self.t
        self._offset += 0.02 * (err - self._offset)
        if abs(self._offset) > self.step_s:
            self.t += self._offset
            self._offset = 0.0
            self.steps += 1
        return self.t

    def advance(self, n: int, rate: int) -> None:
        if self.t is not None:
            self.t += n / float(rate)


class ReferenceTimeline:
    """Ring buffer of loudspeaker samples addressed by absolute stream time."""

    def __init__(self, rate: int = PROC_RATE, seconds: float = 20.0, jump_ms: float = 40.0) -> None:
        self.rate = int(rate)
        self._size = int(self.rate * seconds)
        self._buf = np.zeros(self._size, dtype=np.int16)
        self._next: Optional[int] = None  # absolute index of the next sample to write
        self._clock = ClockTracker(jump_ms)
        self._lock = threading.Lock()

    @property
    def reanchors(self) -> int:
        return self._clock.reanchors

    def write(self, x: np.ndarray, t_start: float) -> None:
        """Store speaker samples whose first sample is heard at ``t_start``."""
        if x.size == 0:
            return
        with self._lock:
            idx = int(round(self._clock.update(t_start) * self.rate))
            if self._next is not None and idx > self._next:
                # moved forward (dropout or drift): nothing known was played there
                self._put(self._next, np.zeros(min(idx - self._next, self._size), dtype=np.int16))
            self._next = idx
            self._put(self._next, x)
            self._next += x.size
            self._clock.advance(x.size, self.rate)

    def _put(self, start: int, x: np.ndarray) -> None:
        x = x[-self._size:]
        pos = start % self._size
        first = min(x.size, self._size - pos)
        self._buf[pos:pos + first] = x[:first]
        if first < x.size:
            self._buf[: x.size - first] = x[first:]

    def end_time(self) -> Optional[float]:
        """Stream time up to which speaker samples are known."""
        with self._lock:
            return None if self._next is None else self._next / float(self.rate)

    def read(self, t_start: float, n: int) -> np.ndarray:
        idx = int(round(t_start * self.rate))
        out = np.zeros(n, dtype=np.int16)
        with self._lock:
            if self._next is None:
                return out
            lo = max(idx, self._next - self._size)
            hi = min(idx + n, self._next)
            if hi <= lo:
                return out
            pos = lo % self._size
            cnt = hi - lo
            first = min(cnt, self._size - pos)
            seg = np.concatenate([self._buf[pos:pos + first], self._buf[: cnt - first]])
            out[lo - idx: hi - idx] = seg
        return out


# --------------------------------------------------------------------------
# playback queue
# --------------------------------------------------------------------------


class PlaybackQueue:
    """Thread-safe FIFO of device-rate samples tagged with an item id.

    Keeps count of how much of each item actually reached the device, which
    the Realtime API needs to truncate the assistant message on barge-in.
    """

    def __init__(self, rate: int) -> None:
        self.rate = int(rate)
        self._chunks: Deque[Tuple[str, np.ndarray]] = collections.deque()
        self._lock = threading.Lock()
        self._played: Dict[str, int] = {}
        self.current_item: Optional[str] = None

    def put(self, x: np.ndarray, item_id: str = "") -> None:
        if x.size:
            with self._lock:
                self._chunks.append((item_id, x))

    def pull(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.int16)
        filled = 0
        with self._lock:
            while filled < n and self._chunks:
                item, chunk = self._chunks[0]
                take = min(n - filled, chunk.size)
                out[filled:filled + take] = chunk[:take]
                filled += take
                self._played[item] = self._played.get(item, 0) + take
                self.current_item = item
                if take == chunk.size:
                    self._chunks.popleft()
                else:
                    self._chunks[0] = (item, chunk[take:])
        return out

    def pending_samples(self) -> int:
        with self._lock:
            return sum(c.size for _, c in self._chunks)

    def played_ms(self, item_id: str) -> int:
        with self._lock:
            return int(1000 * self._played.get(item_id, 0) / self.rate)

    def clear(self) -> Optional[str]:
        """Drop everything not yet played; return the item that was playing."""
        with self._lock:
            self._chunks.clear()
            return self.current_item


# --------------------------------------------------------------------------
# echo gate
# --------------------------------------------------------------------------


class EchoGate:
    """Decide which cleaned mic frames may go upstream.

    modes
      ``full``  always forward (trust the AEC, best barge-in)
      ``smart`` while the speaker is (or just was) active, forward only when
                the AEC output is clearly louder, *relative to what the
                speaker is currently playing*, than the residual echo seen
                so far -> a human is talking over the robot. Otherwise
                forward silence. Default.
      ``half``  forward silence whenever the speaker is active (walkie-talkie,
                bulletproof, no barge-in)

    The smart gate learns the residual coupling ``aec_out_dB - speaker_dB``
    while it is shut and opens when a frame exceeds its upper percentile by
    ``margin_db`` for ``attack_frames`` frames in a row. Silence (zeros) is
    sent instead of dropping frames so the server-side VAD sees a continuous
    stream; the frames buffered while shut are released as pre-roll when the
    gate opens so the start of the user's word is not cut off.
    """

    def __init__(
        self,
        mode: str = "smart",
        margin_db: float = 6.0,
        min_level_dbfs: float = -50.0,
        attack_frames: int = 4,
        hangover_ms: int = 400,
        preroll_ms: int = 200,
        history_s: float = 6.0,
        percentile: float = 90.0,
        initial_coupling_db: float = -20.0,
    ) -> None:
        if mode not in ("full", "smart", "half"):
            raise ValueError(f"unknown gate mode {mode!r}")
        self.mode = mode
        self.margin_db = float(margin_db)
        self.min_level = float(min_level_dbfs)
        self.attack = int(attack_frames)
        self.hangover = max(1, hangover_ms // 10)
        self.percentile = float(percentile)
        self.initial_coupling = float(initial_coupling_db)
        self._preroll: Deque[np.ndarray] = collections.deque(maxlen=max(1, preroll_ms // 10))
        self._coupling: Deque[float] = collections.deque(maxlen=int(history_s * 100))
        self._above = 0
        self._open_left = 0
        self.is_open = False

    def residual_coupling_db(self) -> float:
        if len(self._coupling) < 50:
            return self.initial_coupling
        return float(np.percentile(np.fromiter(self._coupling, float), self.percentile))

    def process(self, frame: np.ndarray, echo_active: bool, ref_level_db: float = -90.0) -> List[np.ndarray]:
        """Return the frames to send for this input frame (usually one).

        ``ref_level_db``: recent loudness envelope of the speaker signal
        aligned to this frame (dBFS), used by the smart mode.
        """
        if self.mode == "full" or not echo_active:
            self.is_open = True
            self._above = 0
            self._open_left = 0
            self._preroll.clear()
            return [frame]
        silence = np.zeros_like(frame)
        if self.mode == "half":
            self.is_open = False
            return [silence]

        level = dbfs(frame)
        coupling = level - max(ref_level_db, -70.0)
        threshold = self.residual_coupling_db() + self.margin_db
        if coupling > threshold and level > self.min_level:
            self._above += 1
        else:
            self._above = 0
            if not self._open_left:
                self._coupling.append(coupling)  # learn how loud the leftover echo is
        if self._above >= self.attack:
            self._open_left = self.hangover
            if not self.is_open:
                self.is_open = True
                out = list(self._preroll) + [frame]
                self._preroll.clear()
                return out
        if self._open_left:
            self._open_left -= 1
            self.is_open = True
            return [frame]
        self.is_open = False
        self._preroll.append(frame)
        return [silence]


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------


FrameCallback = Callable[[np.ndarray, Dict[str, object]], None]


def find_device(name_or_index: object, kind: str) -> Optional[int]:
    """Resolve a device by index or case-insensitive name substring."""
    import sounddevice as sd

    if name_or_index is None or name_or_index == "" or name_or_index == -1:
        return None
    if isinstance(name_or_index, int) or str(name_or_index).lstrip("-").isdigit():
        return int(name_or_index)
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    needle = str(name_or_index).lower()
    for i, dev in enumerate(sd.query_devices()):
        if dev.get(key, 0) > 0 and needle in dev["name"].lower():
            return i
    raise ValueError(f"no {kind} device matching {name_or_index!r}; run with --list-devices")


def pick_rate(device: Optional[int], kind: str, preferred: List[int]) -> int:
    import sounddevice as sd

    check = sd.check_input_settings if kind == "input" else sd.check_output_settings
    for rate in preferred:
        try:
            check(device=device, samplerate=rate, channels=1, dtype="int16")
            return rate
        except Exception:
            continue
    info = sd.query_devices(device, kind)
    return int(info["default_samplerate"])


class DuplexAudioEngine:
    """Owns speaker output, mic input, alignment, AEC and the echo gate."""

    def __init__(
        self,
        input_device: object = None,
        output_device: object = None,
        aec_backend: str = "auto",
        gate_mode: str = "smart",
        echo_delay_ms: float = 0.0,
        ref_lead_ms: float = 40.0,
        output_gain: float = 1.0,
        mic_gain: float = 1.0,
        track_delay: bool = True,
        aec_kwargs: Optional[Dict[str, object]] = None,
        gate_kwargs: Optional[Dict[str, object]] = None,
        record_debug: bool = False,
    ) -> None:
        self.input_device = find_device(input_device, "input")
        self.output_device = find_device(output_device, "output")
        self.in_rate = pick_rate(self.input_device, "input", [PROC_RATE, 48000, 44100])
        self.out_rate = pick_rate(self.output_device, "output", [24000, 48000, 44100, PROC_RATE])
        self.aec: EchoCanceller = make_echo_canceller(aec_backend, PROC_RATE, **(aec_kwargs or {}))
        self.aec.set_delay_ms(int(ref_lead_ms))
        self.gate = EchoGate(gate_mode, **(gate_kwargs or {}))
        # echo_delay: where the echo really sits relative to the timestamps
        # (wrongly reported device latencies + sound travel), measured by
        # calibrate() and the delay tracker. ref_lead: safety margin by which
        # the reference handed to the AEC precedes the echo.
        self.echo_delay_s = float(echo_delay_ms) / 1000.0
        self.ref_lead_s = float(ref_lead_ms) / 1000.0
        self.output_gain = float(output_gain)
        self.mic_gain = float(mic_gain)
        self.track_delay = bool(track_delay)

        self.timeline = ReferenceTimeline(PROC_RATE)
        self.playback = PlaybackQueue(self.out_rate)
        self._ref_resampler = StreamResampler(self.out_rate, PROC_RATE)
        self._mic_resampler = StreamResampler(self.in_rate, PROC_RATE)
        self._mic_q: "queue.Queue[Tuple[np.ndarray, float]]" = queue.Queue(maxsize=500)
        self._chunker = FrameChunker(FRAME)
        self._mic_t: Optional[float] = None  # stream time of the next 16 kHz mic frame
        self._mic_clock = ClockTracker()
        self._pending: Deque[Tuple[np.ndarray, float]] = collections.deque()
        self._calib_frames: Optional[List[Tuple[np.ndarray, np.ndarray, float]]] = None
        self._aec_lock = threading.Lock()  # the AEC state is not thread safe
        self._started_at = time.monotonic()
        self._callbacks: List[FrameCallback] = []
        self._external_speaking = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._in_stream = None
        self._out_stream = None
        self._out_resamplers: Dict[int, StreamResampler] = {}

        # delay tracker history (raw mic and aligned reference, 16 kHz)
        self._hist_s = 4.0
        self._hist_mic: Deque[np.ndarray] = collections.deque(maxlen=int(self._hist_s * 100))
        self._hist_ref: Deque[np.ndarray] = collections.deque(maxlen=int(self._hist_s * 100))
        self._frames_since_track = 0
        self.last_delay: Optional[Tuple[float, float]] = None
        self._suspect_lag: Optional[float] = None

        # stats
        self.stats: Dict[str, float] = {"mic_db": -90.0, "out_db": -90.0, "ref_db": -90.0, "erle_db": 0.0}
        self._erle_acc = [0.0, 0.0]
        self.record_debug = record_debug
        self.debug_mic: List[np.ndarray] = []
        self.debug_ref: List[np.ndarray] = []
        self.debug_out: List[np.ndarray] = []
        self.debug_sent: List[np.ndarray] = []

    # ------------------------------------------------------------ public api

    def add_frame_callback(self, cb: FrameCallback) -> None:
        self._callbacks.append(cb)

    def play(self, pcm: np.ndarray, rate: int, item_id: str = "") -> None:
        """Queue mono int16 audio (any rate) for playback."""
        if rate not in self._out_resamplers:
            self._out_resamplers[rate] = StreamResampler(rate, self.out_rate)
        self.playback.put(self._out_resamplers[rate].process(pcm), item_id)

    def stop_playback(self) -> Optional[str]:
        """Drop queued audio immediately (barge-in). Returns the item id."""
        return self.playback.clear()

    def is_playing(self) -> bool:
        return self.playback.pending_samples() > 0

    def set_external_speaking(self, speaking: bool) -> None:
        """Another process (e.g. piper_tts_node) is using the speaker."""
        self._external_speaking = bool(speaking)

    def start(self) -> None:
        import sounddevice as sd

        self._running = True
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._process_loop, name="aec", daemon=True)
        self._thread.start()
        self._out_stream = sd.OutputStream(
            device=self.output_device, samplerate=self.out_rate, channels=1, dtype="int16",
            blocksize=self.out_rate // 100, latency="low", callback=self._out_cb,
        )
        self._in_stream = sd.InputStream(
            device=self.input_device, samplerate=self.in_rate, channels=1, dtype="int16",
            blocksize=self.in_rate // 100, latency="low", callback=self._in_cb,
        )
        self._out_stream.start()
        self._in_stream.start()
        _LOG.info(
            "audio engine: in=%s@%d out=%s@%d aec=%s gate=%s echo_delay=%.0fms lead=%.0fms "
            "latency in=%.1fms out=%.1fms",
            self.input_device, self.in_rate, self.output_device, self.out_rate, self.aec.name,
            self.gate.mode, self.echo_delay_s * 1000, self.ref_lead_s * 1000, self._in_stream.latency * 1000,
            self._out_stream.latency * 1000,
        )

    def stop(self) -> None:
        self._running = False
        for stream in (self._in_stream, self._out_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.aec.close()

    # ------------------------------------------------------------ callbacks

    def _out_cb(self, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            _LOG.debug("output status: %s", status)
        x = self.playback.pull(frames)
        if self.output_gain != 1.0:
            x = to_int16(x.astype(np.float32) * self.output_gain / 32768.0)
        outdata[:, 0] = x
        t_dac = time_info.outputBufferDacTime or (self._out_stream.time + self._out_stream.latency)
        self.timeline.write(self._ref_resampler.process(x), t_dac)

    def _in_cb(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            _LOG.debug("input status: %s", status)
        t_adc = time_info.inputBufferAdcTime or (self._in_stream.time - self._in_stream.latency)
        try:
            self._mic_q.put_nowait((indata[:, 0].copy(), t_adc))
        except queue.Full:
            _LOG.warning("mic queue full, dropping audio")

    # ------------------------------------------------------------ processing

    def _process_loop(self) -> None:
        while self._running:
            try:
                block, t_adc = self._mic_q.get(timeout=0.2)
            except queue.Empty:
                continue
            x16 = self._mic_resampler.process(block)
            if self.mic_gain != 1.0:
                x16 = to_int16(x16.astype(np.float32) * self.mic_gain / 32768.0)
            # continuous 16 kHz sample clock, smoothed against timestamp jitter
            pending = self._chunker._buf.size  # samples buffered but not yet framed
            self._mic_t = self._mic_clock.update(t_adc) - pending / PROC_RATE
            self._mic_clock.advance(x16.size, PROC_RATE)
            for frame in self._chunker.push(x16):
                self._pending.append((frame, self._mic_t))
                self._mic_t += FRAME / PROC_RATE
            self._drain_pending()

    def _ref_time(self, t: float) -> float:
        """Stream time of the speaker samples to pair with mic time t."""
        return t - self.echo_delay_s + self.ref_lead_s

    def _drain_pending(self) -> None:
        # With a small or negative echo delay the reference for a mic frame
        # may not be written yet; wait for the output callback, but never
        # hold audio for more than 300 ms.
        while self._pending:
            frame, t = self._pending[0]
            end = self.timeline.end_time()
            ready = end is not None and end >= self._ref_time(t) + FRAME / PROC_RATE
            if not ready and len(self._pending) < 30:
                return
            self._pending.popleft()
            self._process_frame(frame, t)

    def process_frame_offline(self, mic: np.ndarray, ref: np.ndarray) -> List[np.ndarray]:
        """Run AEC + gate on already aligned frames (tests, file replay)."""
        return self._run_chain(mic, ref, echo_active=None)

    def _process_frame(self, mic: np.ndarray, t: float) -> None:
        ref = self.timeline.read(self._ref_time(t), FRAME)
        if self._calib_frames is not None:
            self._calib_frames.append((mic, ref, t))
        self._run_chain(mic, ref, echo_active=None, t=t)

    def _echo_state(self, t: Optional[float], ref: np.ndarray) -> Tuple[bool, float]:
        """(speaker may still be audible, speaker loudness envelope in dBFS)."""
        if t is None:
            recent = ref
        else:
            recent = self.timeline.read(self._ref_time(t) - ECHO_TAIL_S, int(ECHO_TAIL_S * PROC_RATE) + FRAME)
        env = speaker_envelope(recent)
        env_db = 20.0 * math.log10(max(env, 1e-3) / 32768.0)
        return self._external_speaking or env > ECHO_ACTIVE_RMS, env_db

    def _run_chain(self, mic: np.ndarray, ref: np.ndarray, echo_active: Optional[bool], t: Optional[float] = None) -> List[np.ndarray]:
        with self._aec_lock:
            out = self.aec.process(mic, ref)
        active, ref_env_db = self._echo_state(t, ref)
        if echo_active is not None:
            active = echo_active
        sent = self.gate.process(out, active, ref_env_db)

        a = 0.95
        self.stats["mic_db"] = a * self.stats["mic_db"] + (1 - a) * dbfs(mic)
        self.stats["out_db"] = a * self.stats["out_db"] + (1 - a) * dbfs(out)
        self.stats["ref_db"] = a * self.stats["ref_db"] + (1 - a) * dbfs(ref)
        if rms(ref) > 100.0:
            self._erle_acc[0] = 0.98 * self._erle_acc[0] + float(np.mean(mic.astype(np.float64) ** 2))
            self._erle_acc[1] = 0.98 * self._erle_acc[1] + float(np.mean(out.astype(np.float64) ** 2))
            self.stats["erle_db"] = 10 * math.log10((self._erle_acc[0] + 1) / (self._erle_acc[1] + 1))
        self.stats["gate_open"] = float(self.gate.is_open)
        self.stats["echo_active"] = float(active)

        if self.record_debug:
            self.debug_mic.append(mic)
            self.debug_ref.append(ref)
            self.debug_out.append(out)
            self.debug_sent.append(np.concatenate(sent)[-FRAME:])

        if self.track_delay and t is not None:
            self._track_delay(mic, ref)

        info = {"echo_active": active, "gate_open": self.gate.is_open, "t": t}
        for frame in sent:
            for cb in self._callbacks:
                try:
                    cb(frame, info)
                except Exception:
                    _LOG.exception("frame callback failed")
        return sent

    # ------------------------------------------------------------ delay

    def _track_delay(self, mic: np.ndarray, ref: np.ndarray) -> None:
        """Every ~3 s of robot speech, verify the reference leads the echo.

        Windows containing double talk (gate open: a person speaks over the
        robot) are skipped because they give unreliable estimates, and the
        delay only moves after two consistent, confident measurements.
        """
        if self.gate.is_open and self.gate.mode == "smart":
            self._hist_mic.clear()  # restart the window after double talk
            self._hist_ref.clear()
            return
        self._hist_mic.append(mic)
        self._hist_ref.append(ref)
        if rms(ref) < 100.0:
            return
        self._frames_since_track += 1
        if self._frames_since_track < 300 or len(self._hist_ref) < self._hist_ref.maxlen:
            return
        self._frames_since_track = 0
        r = np.concatenate(self._hist_ref)
        m = np.concatenate(self._hist_mic)
        lag, conf = estimate_delay(r, m, PROC_RATE, max_delay_s=0.5)
        lag_ms = 1000.0 * lag / PROC_RATE
        self.last_delay = (lag_ms, conf)
        lead_ms = self.ref_lead_s * 1000.0
        suspect = conf >= 20.0 and abs(lag_ms - lead_ms) > 25.0
        previous, self._suspect_lag = self._suspect_lag, (lag_ms if suspect else None)
        if suspect and previous is not None and abs(previous - lag_ms) < 5.0:
            self._suspect_lag = None
            self._set_echo_delay(self.echo_delay_s + (lag_ms - lead_ms) / 1000.0)
            _LOG.warning(
                "echo lags the reference by %.0f ms instead of %.0f ms (confidence %.0f); "
                "echo delay is now %.0f ms", lag_ms, lead_ms, conf, self.echo_delay_s * 1000,
            )

    def _set_echo_delay(self, seconds: float) -> None:
        self.echo_delay_s = min(max(seconds, -0.2), 0.5)

    def calibrate(
        self, seconds: float = 1.0, level_dbfs: float = -14.0, max_rounds: int = 4, warmup_s: float = 1.5,
    ) -> Tuple[float, float]:
        """Play short noise bursts, measure where their echo lands and set
        ``echo_delay`` so the reference leads the echo by ``ref_lead``.

        Device latencies (PipeWire especially) settle during the first
        second after opening a stream, so this waits ``warmup_s`` and repeats
        until two rounds agree within 3 ms. The recorded bursts are finally
        replayed through the echo canceller so it starts out converged.
        Engine must be running. Returns (final echo lag behind the
        reference in ms, should be ~ref_lead; confidence)."""
        wait = warmup_s - (time.monotonic() - self._started_at)
        if wait > 0:
            time.sleep(wait)
        lag_ms, conf, frames = float("nan"), 0.0, []
        prev_delay: Optional[float] = None
        for _ in range(max_rounds):
            lag_ms, conf, frames = self._calibration_round(seconds, level_dbfs)
            if conf < 8.0:
                continue
            self._set_echo_delay(self.echo_delay_s + (lag_ms - self.ref_lead_s * 1000.0) / 1000.0)
            if prev_delay is not None and abs(prev_delay - self.echo_delay_s) < 0.003:
                break
            prev_delay = self.echo_delay_s
        with self._aec_lock:
            self.aec.set_delay_ms(int(self.ref_lead_s * 1000))
            for mic, t in frames:  # pre-train with the correct alignment
                self.aec.process(mic, self.timeline.read(self._ref_time(t), FRAME))
        return lag_ms, conf

    def _calibration_round(self, seconds: float, level_dbfs: float) -> Tuple[float, float, List[Tuple[np.ndarray, float]]]:
        n = int(seconds * self.out_rate)
        rng = np.random.default_rng()
        burst = rng.standard_normal(n) * (10 ** (level_dbfs / 20.0))
        fade = np.minimum(1.0, np.minimum(np.arange(n), np.arange(n)[::-1]) / (0.02 * self.out_rate))
        was_tracking, self.track_delay = self.track_delay, False
        self._calib_frames = []
        self.play(to_int16(burst * fade), self.out_rate, "calibration")
        time.sleep(seconds + 0.6)
        frames, self._calib_frames = self._calib_frames, None
        self.track_delay = was_tracking
        if len(frames) < 50:
            return float("nan"), 0.0, frames
        mic = np.concatenate([m for m, _, _ in frames])
        ref = np.concatenate([r for _, r, _ in frames])
        lag, conf = estimate_delay(ref, mic, PROC_RATE, max_delay_s=0.5)
        return 1000.0 * lag / PROC_RATE, conf, [(m, t) for m, _, t in frames]
