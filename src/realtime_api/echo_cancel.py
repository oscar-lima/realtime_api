"""Acoustic echo cancellation (AEC) backends.

The robot plays its own voice through a loudspeaker and the microphone picks
it up again. Because this process both plays the robot voice and captures the
microphone, it knows the exact far-end (reference) signal and can remove it
from the microphone signal before anything is sent to the Realtime API.

Backends, best first (``make_echo_canceller("auto")`` picks the first that
loads):

``webrtc``  WebRTC AEC3 through ``livekit.rtc.AudioProcessingModule``
            (``pip install livekit``, Python >= 3.9). Same canceller Chrome
            and most conferencing tools use; handles delay search, nonlinear
            residual suppression and double talk internally.
``speex``   SpeexDSP MDF echo canceller + residual echo suppressor through
            ctypes on ``libspeexdsp.so.1`` (``apt install libspeexdsp1``,
            already present in the Mobipick Noetic image, Python 3.8 ok).
``nlms``    Pure numpy partitioned-block frequency-domain NLMS with a
            two-path (foreground/background) double-talk safe update and a
            spectral residual echo suppressor. Always available.
``none``    Pass-through, for A/B comparisons.

All backends take and return 10 ms mono int16 frames (``frame_size``
samples at ``sample_rate``). The reference frame passed with a microphone
frame must be the loudspeaker signal time-aligned to that microphone frame
(see ``duplex_audio.ReferenceTimeline``), possibly leading it by a few tens
of milliseconds; it must never lag it.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

_LOG = logging.getLogger(__name__)

BACKENDS = ("webrtc", "speex", "nlms", "none")


class EchoCanceller:
    """Interface: process one mic frame together with its reference frame."""

    name = "none"

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 10) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_size = int(self.sample_rate * frame_ms // 1000)

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        return mic

    def set_delay_ms(self, delay_ms: int) -> None:
        """Hint of how far the reference leads the echo (webrtc only)."""

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------
# WebRTC AEC3 (livekit)
# --------------------------------------------------------------------------


class WebRtcEchoCanceller(EchoCanceller):
    name = "webrtc"

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 10,
        noise_suppression: bool = True,
        high_pass_filter: bool = True,
        auto_gain_control: bool = False,
        delay_ms: int = 0,
        **_: object,
    ) -> None:
        if frame_ms != 10:
            raise ValueError("WebRTC APM requires 10 ms frames")
        super().__init__(sample_rate, frame_ms)
        from livekit import rtc  # noqa: WPS433 - optional dependency

        self._rtc = rtc
        self._apm = rtc.AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=noise_suppression,
            high_pass_filter=high_pass_filter,
            auto_gain_control=auto_gain_control,
        )
        self._delay_ms = int(delay_ms)

    def set_delay_ms(self, delay_ms: int) -> None:
        self._delay_ms = max(0, int(delay_ms))

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        n = self.frame_size
        ref_frame = self._rtc.AudioFrame(bytearray(ref.astype(np.int16).tobytes()), self.sample_rate, 1, n)
        self._apm.process_reverse_stream(ref_frame)
        mic_frame = self._rtc.AudioFrame(bytearray(mic.astype(np.int16).tobytes()), self.sample_rate, 1, n)
        self._apm.set_stream_delay_ms(self._delay_ms)
        self._apm.process_stream(mic_frame)
        return np.frombuffer(bytes(mic_frame.data), dtype=np.int16).copy()


# --------------------------------------------------------------------------
# SpeexDSP (ctypes)
# --------------------------------------------------------------------------

# speex_echo.h / speex_preprocess.h request codes
_SPEEX_ECHO_SET_SAMPLING_RATE = 24
_SPEEX_PREPROCESS_SET_DENOISE = 0
_SPEEX_PREPROCESS_SET_NOISE_SUPPRESS = 18
_SPEEX_PREPROCESS_SET_ECHO_SUPPRESS = 20
_SPEEX_PREPROCESS_SET_ECHO_SUPPRESS_ACTIVE = 22
_SPEEX_PREPROCESS_SET_ECHO_STATE = 24


def _load_speexdsp() -> ctypes.CDLL:
    candidates = [ctypes.util.find_library("speexdsp"), "libspeexdsp.so.1", "libspeexdsp.so"]
    last_err: Optional[Exception] = None
    for cand in candidates:
        if not cand:
            continue
        try:
            lib = ctypes.CDLL(cand)
            break
        except OSError as exc:
            last_err = exc
    else:
        raise OSError(f"libspeexdsp not found ({last_err}); apt install libspeexdsp1")

    vp, ci, i16p = ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int16)
    lib.speex_echo_state_init.restype = vp
    lib.speex_echo_state_init.argtypes = [ci, ci]
    lib.speex_echo_state_destroy.argtypes = [vp]
    lib.speex_echo_cancellation.argtypes = [vp, i16p, i16p, i16p]
    lib.speex_echo_ctl.argtypes = [vp, ci, vp]
    lib.speex_preprocess_state_init.restype = vp
    lib.speex_preprocess_state_init.argtypes = [ci, ci]
    lib.speex_preprocess_state_destroy.argtypes = [vp]
    lib.speex_preprocess_ctl.argtypes = [vp, ci, vp]
    lib.speex_preprocess_run.argtypes = [vp, i16p]
    return lib


class SpeexEchoCanceller(EchoCanceller):
    name = "speex"

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 10,
        filter_ms: int = 200,
        noise_suppression: bool = True,
        echo_suppress_db: int = -45,
        echo_suppress_active_db: int = -20,
        **_: object,
    ) -> None:
        super().__init__(sample_rate, frame_ms)
        self._lib = _load_speexdsp()
        tail = int(self.sample_rate * filter_ms // 1000)
        self._echo = self._lib.speex_echo_state_init(self.frame_size, tail)
        if not self._echo:
            raise RuntimeError("speex_echo_state_init failed")
        rate = ctypes.c_int(self.sample_rate)
        self._lib.speex_echo_ctl(self._echo, _SPEEX_ECHO_SET_SAMPLING_RATE, ctypes.byref(rate))
        self._pre = self._lib.speex_preprocess_state_init(self.frame_size, self.sample_rate)
        if not self._pre:
            raise RuntimeError("speex_preprocess_state_init failed")
        self._ctl_pre(_SPEEX_PREPROCESS_SET_ECHO_STATE, ctypes.c_void_p(self._echo), by_value=True)
        self._ctl_pre(_SPEEX_PREPROCESS_SET_DENOISE, ctypes.c_int(1 if noise_suppression else 0))
        self._ctl_pre(_SPEEX_PREPROCESS_SET_NOISE_SUPPRESS, ctypes.c_int(-20))
        self._ctl_pre(_SPEEX_PREPROCESS_SET_ECHO_SUPPRESS, ctypes.c_int(int(echo_suppress_db)))
        self._ctl_pre(_SPEEX_PREPROCESS_SET_ECHO_SUPPRESS_ACTIVE, ctypes.c_int(int(echo_suppress_active_db)))
        self._out = np.zeros(self.frame_size, dtype=np.int16)

    def _ctl_pre(self, req: int, value: object, by_value: bool = False) -> None:
        ptr = value if by_value else ctypes.byref(value)  # type: ignore[arg-type]
        self._lib.speex_preprocess_ctl(self._pre, req, ptr)

    @staticmethod
    def _ptr(x: np.ndarray) -> "ctypes._Pointer":
        return x.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        mic = np.ascontiguousarray(mic, dtype=np.int16)
        ref = np.ascontiguousarray(ref, dtype=np.int16)
        out = np.empty(self.frame_size, dtype=np.int16)
        self._lib.speex_echo_cancellation(self._echo, self._ptr(mic), self._ptr(ref), self._ptr(out))
        self._lib.speex_preprocess_run(self._pre, self._ptr(out))
        return out

    def close(self) -> None:
        if getattr(self, "_pre", None):
            self._lib.speex_preprocess_state_destroy(self._pre)
            self._pre = None
        if getattr(self, "_echo", None):
            self._lib.speex_echo_state_destroy(self._echo)
            self._echo = None

    def __del__(self) -> None:  # pragma: no cover - GC timing
        try:
            self.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# numpy NLMS fallback
# --------------------------------------------------------------------------


class NlmsEchoCanceller(EchoCanceller):
    """Partitioned-block frequency-domain NLMS (overlap-save) + RES.

    Two filters run in parallel (two-path double-talk handling, Ochiai 1977):
    the background filter always adapts; its coefficients are copied to the
    foreground filter (whose output is used) only when it clearly produces
    less error. During double talk the background diverges, its error grows
    and nothing is copied, so near-end speech does not destroy the model of
    the echo path. A spectral residual echo suppressor removes what the
    linear filter cannot (loudspeaker nonlinearities, estimation error).
    """

    name = "nlms"

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 10,
        filter_ms: int = 200,
        mu: float = 0.35,
        res_overestimate: float = 3.0,
        res_floor_db: float = -30.0,
        **_: object,
    ) -> None:
        super().__init__(sample_rate, frame_ms)
        B = self.frame_size
        self._B = B
        self._N = 2 * B
        self._P = max(1, int(math.ceil(self.sample_rate * filter_ms / 1000.0 / B)))
        bins = B + 1
        self._X = np.zeros((self._P, bins), dtype=np.complex128)
        self._Wf = np.zeros((self._P, bins), dtype=np.complex128)
        self._Wb = np.zeros((self._P, bins), dtype=np.complex128)
        self._prev_ref = np.zeros(B)
        self._pxx = np.full(bins, 1e-9)
        self._mu = float(mu)
        self._ef = 1e-9
        self._eb = 1e-9
        self._em = 1e-9
        self._copy_count = 0
        # residual echo suppressor (STFT, sqrt-hann, 50 % overlap)
        self._win = np.sqrt(np.hanning(self._N + 1)[:-1])
        self._prev_e = np.zeros(B)
        self._prev_y = np.zeros(B)
        self._ola = np.zeros(B)
        self._res_over = float(res_overestimate)
        self._res_floor = 10.0 ** (res_floor_db / 20.0)
        self._gain = np.ones(bins)

    def _filter(self, W: np.ndarray) -> np.ndarray:
        Y = np.sum(W * self._X, axis=0)
        return np.fft.irfft(Y, self._N)[self._B:]

    def _adapt(self, W: np.ndarray, e: np.ndarray, mu: float) -> None:
        E = np.fft.rfft(np.concatenate([np.zeros(self._B), e]))
        G = np.conj(self._X) * (E / (self._pxx * self._P + 1e-6))[None, :]
        g = np.fft.irfft(G, self._N, axis=1)
        g[:, self._B:] = 0.0  # gradient constraint -> linear, not circular, convolution
        W += mu * np.fft.rfft(g, axis=1)

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        m = mic.astype(np.float64) / 32768.0
        r = ref.astype(np.float64) / 32768.0

        self._X = np.roll(self._X, 1, axis=0)
        self._X[0] = np.fft.rfft(np.concatenate([self._prev_ref, r]))
        self._prev_ref = r
        self._pxx = 0.9 * self._pxx + 0.1 * np.abs(self._X[0]) ** 2

        yf = self._filter(self._Wf)
        yb = self._filter(self._Wb)
        ef = m - yf
        eb = m - yb

        ref_active = float(np.mean(r * r)) > 1e-7  # about -70 dBFS
        if ref_active:
            a = 0.8
            self._ef = a * self._ef + (1 - a) * float(np.mean(ef * ef))
            self._eb = a * self._eb + (1 - a) * float(np.mean(eb * eb))
            self._em = a * self._em + (1 - a) * float(np.mean(m * m))
            self._adapt(self._Wb, eb, self._mu)
            if self._eb < 0.7 * self._ef and self._eb < self._em:
                self._copy_count += 1
                if self._copy_count >= 3:
                    self._Wf[:] = self._Wb
                    self._ef = self._eb
                    ef, yf = eb, yb
            else:
                self._copy_count = 0
            if self._eb > 8.0 * self._ef and self._ef < self._em:
                # background diverged (double talk): restart it from foreground
                self._Wb[:] = self._Wf
                self._eb = self._ef

        return self._suppress(ef, yf)

    def _suppress(self, e: np.ndarray, y: np.ndarray) -> np.ndarray:
        frame_e = np.concatenate([self._prev_e, e]) * self._win
        frame_y = np.concatenate([self._prev_y, y]) * self._win
        self._prev_e, self._prev_y = e, y
        E = np.fft.rfft(frame_e)
        Y = np.fft.rfft(frame_y)
        pe = np.abs(E) ** 2
        # residual echo is assumed to follow the shape of the echo estimate
        pr = self._res_over * np.abs(Y) ** 2
        g = np.clip(1.0 - pr / (pe + 1e-12), self._res_floor, 1.0)
        # fast attack, slower release to avoid musical noise
        self._gain = np.where(g < self._gain, g, 0.6 * self._gain + 0.4 * g)
        out = np.fft.irfft(E * self._gain, self._N) * self._win
        result = self._ola + out[: self._B]
        self._ola = out[self._B:]
        return np.clip(np.rint(result * 32768.0), -32768, 32767).astype(np.int16)


# --------------------------------------------------------------------------
# factory and helpers
# --------------------------------------------------------------------------

_CLASSES = {
    "webrtc": WebRtcEchoCanceller,
    "speex": SpeexEchoCanceller,
    "nlms": NlmsEchoCanceller,
    "none": EchoCanceller,
}


def available_backends() -> Dict[str, str]:
    """Map every backend to 'ok' or the reason it cannot load."""
    status: Dict[str, str] = {}
    for name in BACKENDS:
        try:
            make_echo_canceller(name).close()
            status[name] = "ok"
        except Exception as exc:  # pragma: no cover - environment dependent
            status[name] = f"unavailable: {exc}"
    return status


def make_echo_canceller(backend: str = "auto", sample_rate: int = 16000, **kwargs: object) -> EchoCanceller:
    """Create an echo canceller; ``auto`` tries webrtc, speex, then nlms."""
    backend = (backend or "auto").lower()
    if backend == "auto":
        errors: List[str] = []
        for name in ("webrtc", "speex", "nlms"):
            try:
                aec = _CLASSES[name](sample_rate=sample_rate, **kwargs)
                _LOG.info("echo canceller backend: %s", name)
                return aec
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        raise RuntimeError("no echo canceller available: " + "; ".join(errors))
    if backend not in _CLASSES:
        raise ValueError(f"unknown AEC backend {backend!r}, choose from auto, {', '.join(BACKENDS)}")
    return _CLASSES[backend](sample_rate=sample_rate, **kwargs)


def estimate_delay(
    ref: np.ndarray,
    mic: np.ndarray,
    sample_rate: int,
    max_delay_s: float = 0.5,
    band_hz: Tuple[float, float] = (200.0, 4000.0),
) -> Tuple[int, float]:
    """Estimate how many samples ``mic`` lags ``ref`` (two-sided GCC-PHAT).

    The result is negative when the echo shows up in ``mic`` *before* the
    matching samples in ``ref``. Only ``band_hz`` is used, which keeps mains
    hum and hiss (common to nothing in ``ref``) from creating false peaks.
    Returns ``(delay_samples, confidence)``; confidence is the correlation
    peak divided by the mean absolute correlation (> ~8 is a clear peak).
    """
    n = int(ref.size + mic.size)
    nfft = 1 << (n - 1).bit_length()
    R = np.fft.rfft(ref.astype(np.float64), nfft)
    M = np.fft.rfft(mic.astype(np.float64), nfft)
    cross = M * np.conj(R)
    cross /= np.abs(cross) + 1e-12
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    cross[(freqs < band_hz[0]) | (freqs > band_hz[1])] = 0.0
    cc = np.fft.irfft(cross, nfft)
    max_lag = min(int(max_delay_s * sample_rate), nfft // 2 - 1)
    lags = np.concatenate([np.arange(-max_lag, 0), np.arange(0, max_lag + 1)])
    window = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
    k = int(np.argmax(window))
    conf = float(window[k] / (np.mean(np.abs(window)) + 1e-12))
    return int(lags[k]), conf


def erle_db(mic: np.ndarray, out: np.ndarray) -> float:
    """Echo return loss enhancement: mic power / output power, in dB."""
    pm = float(np.mean(np.square(mic.astype(np.float64)))) + 1e-9
    po = float(np.mean(np.square(out.astype(np.float64)))) + 1e-9
    return 10.0 * math.log10(pm / po)


def run_offline(aec: EchoCanceller, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Process whole aligned signals frame by frame (tests, calibration)."""
    n = aec.frame_size
    total = (min(mic.size, ref.size) // n) * n
    out = np.zeros(total, dtype=np.int16)
    for i in range(0, total, n):
        out[i:i + n] = aec.process(mic[i:i + n], ref[i:i + n])
    return out
