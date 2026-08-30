"""PCM conversions for backends that move audio through HTTP speech APIs."""

import io
import math
import wave
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly


_SAMPLE_WIDTH_BYTES = 2
_INT16_PEAK = 32767


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
