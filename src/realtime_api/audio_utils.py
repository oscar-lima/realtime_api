"""Small PCM helpers shared by the echo canceller, the audio engine and tests.

Everything here works on mono int16 numpy arrays and runs on Python 3.8
(robot container) as well as newer host Pythons.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional

import numpy as np

with warnings.catch_warnings():
    # audioop is deprecated since 3.11 and gone in 3.13; it is still the best
    # zero-dependency streaming resampler on the 3.8 robot container.
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import audioop  # type: ignore
    except ImportError:  # pragma: no cover - Python >= 3.13
        audioop = None


def to_int16(x: np.ndarray) -> np.ndarray:
    """Convert float [-1, 1] or any integer array to contiguous int16."""
    if x.dtype == np.int16:
        return np.ascontiguousarray(x)
    if np.issubdtype(x.dtype, np.floating):
        return np.clip(np.rint(x * 32767.0), -32768, 32767).astype(np.int16)
    return np.clip(x, -32768, 32767).astype(np.int16)


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))


def dbfs(x: np.ndarray) -> float:
    """RMS level in dB relative to int16 full scale."""
    return 20.0 * math.log10(max(rms(x), 1e-3) / 32768.0)


class StreamResampler:
    """Stateful mono int16 resampler for audio streams.

    Uses ``audioop.ratecv`` when available (keeps filter state between
    blocks, no clicks at block borders), otherwise a stateful linear
    interpolator that carries its fractional phase across calls.
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self.src_rate = int(src_rate)
        self.dst_rate = int(dst_rate)
        self._state = None
        # linear fallback state
        self._last = 0.0
        self._phase = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = to_int16(x.reshape(-1))
        if self.src_rate == self.dst_rate or x.size == 0:
            return x
        if audioop is not None:
            out, self._state = audioop.ratecv(
                x.tobytes(), 2, 1, self.src_rate, self.dst_rate, self._state
            )
            return np.frombuffer(out, dtype=np.int16).copy()
        return self._linear(x)

    def _linear(self, x: np.ndarray) -> np.ndarray:
        step = self.src_rate / float(self.dst_rate)
        src = np.concatenate([[self._last], x.astype(np.float64)])
        # positions are relative to src[0] (= last sample of previous block)
        pos = np.arange(self._phase, x.size, step)
        out = np.interp(pos, np.arange(src.size), src)
        self._phase = pos[-1] + step - x.size if pos.size else self._phase - x.size
        self._last = float(x[-1])
        return np.clip(np.rint(out), -32768, 32767).astype(np.int16)


class FrameChunker:
    """Accumulate arbitrary-length blocks and emit fixed-size frames."""

    def __init__(self, frame_size: int) -> None:
        self.frame_size = int(frame_size)
        self._buf = np.zeros(0, dtype=np.int16)

    def push(self, x: np.ndarray) -> List[np.ndarray]:
        if x.size:
            self._buf = np.concatenate([self._buf, to_int16(x.reshape(-1))])
        n = (self._buf.size // self.frame_size) * self.frame_size
        if n == 0:
            return []
        frames = [self._buf[i:i + self.frame_size] for i in range(0, n, self.frame_size)]
        self._buf = self._buf[n:].copy()
        return frames

    def clear(self) -> None:
        self._buf = np.zeros(0, dtype=np.int16)


def read_wav_mono(path: str, target_rate: Optional[int] = None) -> "tuple[np.ndarray, int]":
    """Read a PCM16 WAV file as mono int16, optionally resampled."""
    import wave

    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        data = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
    x = np.frombuffer(data, dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)
    if target_rate and target_rate != rate:
        from scipy.signal import resample_poly

        g = math.gcd(rate, target_rate)
        x = to_int16(resample_poly(x.astype(np.float64) / 32768.0, target_rate // g, rate // g))
        rate = target_rate
    return x, rate


def write_wav_mono(path: str, x: np.ndarray, rate: int) -> None:
    import wave

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(to_int16(x).tobytes())
