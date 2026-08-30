"""PCM conversions for backends that move audio through HTTP speech APIs."""

import io
import math
import wave
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, lfilter, resample_poly


_SAMPLE_WIDTH_BYTES = 2
_INT16_PEAK = 32767
# Keep the anti-alias cutoff below the new Nyquist rate, with room for the skirt.
_ANTI_ALIAS_ORDER = 4
_ANTI_ALIAS_MARGIN = 0.45
_ANTI_ALIAS_LIMIT = 0.95


def encode_wav(samples: NDArray[np.int16], sample_rate: int) -> bytes:
    """Wrap mono int16 PCM in a WAV container for multipart audio uploads."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.astype(np.int16, copy=False).tobytes())
    return buffer.getvalue()


def decode_pcm(raw: bytes) -> NDArray[np.int16]:
    """Read little-endian mono int16 PCM, dropping a trailing partial sample."""
    usable_length = len(raw) - (len(raw) % _SAMPLE_WIDTH_BYTES)
    return np.frombuffer(raw[:usable_length], dtype="<i2").astype(np.int16)


def resample(samples: NDArray[np.int16], source_rate: int, target_rate: int) -> NDArray[np.int16]:
    """Resample mono int16 PCM through SciPy's polyphase filter.

    SciPy is a Reachy Mini SDK dependency, so this needs no extra install on the robot.
    """
    if source_rate == target_rate or samples.size == 0:
        return samples

    divisor = math.gcd(source_rate, target_rate)
    resampled: NDArray[np.floating[Any]] = resample_poly(
        samples.astype(np.float32),
        target_rate // divisor,
        source_rate // divisor,
    )
    clipped: NDArray[np.floating[Any]] = np.clip(resampled, -_INT16_PEAK - 1, _INT16_PEAK)
    return clipped.astype(np.int16)


class StreamingResampler:
    """Resample a mono int16 stream chunk by chunk as it arrives.

    The anti-alias filter state and the interpolation phase carry across chunks,
    so the blocks join without the clicks that resampling each one alone leaves
    at its edges. At most one sample is dropped when the stream ends.
    """

    def __init__(self, source_rate: int, target_rate: int) -> None:
        """Design the anti-alias filter for the requested rate change."""
        self.passthrough = source_rate == target_rate
        self._step = source_rate / target_rate
        nyquist = 0.5 * source_rate
        cutoff = min(_ANTI_ALIAS_MARGIN * target_rate, _ANTI_ALIAS_LIMIT * nyquist) / nyquist
        self._numerator, self._denominator = butter(_ANTI_ALIAS_ORDER, cutoff)
        self._state = np.zeros(max(len(self._numerator), len(self._denominator)) - 1)
        self._carry = np.zeros(1)
        self._phase = 0.0

    def process(self, chunk: NDArray[np.int16]) -> NDArray[np.int16]:
        """Return the resampled audio for ``chunk``, continuing the previous chunk."""
        if self.passthrough:
            return chunk
        if chunk.size == 0:
            return np.zeros(0, dtype=np.int16)

        filtered, self._state = lfilter(
            self._numerator,
            self._denominator,
            chunk.astype(np.float64),
            zi=self._state,
        )
        samples = np.concatenate((self._carry, filtered))
        last_index = samples.size - 1
        self._carry = samples[-1:]

        count = max(0, int((last_index - self._phase) // self._step) + 1)
        positions = self._phase + np.arange(count) * self._step
        positions = positions[positions <= last_index]
        if positions.size == 0:
            self._phase -= last_index
            return np.zeros(0, dtype=np.int16)

        left = positions.astype(np.int64)
        fraction = positions - left
        right = np.minimum(left + 1, last_index)
        resampled = samples[left] * (1.0 - fraction) + samples[right] * fraction
        self._phase = positions[-1] + self._step - last_index
        clipped: NDArray[np.floating[Any]] = np.clip(resampled, -_INT16_PEAK - 1, _INT16_PEAK)
        return clipped.astype(np.int16)
