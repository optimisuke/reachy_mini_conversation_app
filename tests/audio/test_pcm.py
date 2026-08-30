"""Tests for the PCM helpers used by the direct backend."""

import io
import wave

import numpy as np

from reachy_mini_conversation_ja.audio.pcm import resample, decode_pcm, encode_wav


def test_encode_wav_round_trips_mono_samples() -> None:
    """The WAV container should keep the samples, rate and mono 16-bit layout."""
    samples = np.array([0, 1000, -1000, 32767], dtype=np.int16)

    with wave.open(io.BytesIO(encode_wav(samples, 16000)), "rb") as wav_file:
        assert (wav_file.getnchannels(), wav_file.getsampwidth(), wav_file.getframerate()) == (1, 2, 16000)
        decoded = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")

    np.testing.assert_array_equal(decoded, samples)


def test_decode_pcm_ignores_a_trailing_partial_sample() -> None:
    """A stream cut mid-sample should decode the whole samples it does contain."""
    raw = np.array([1, -1], dtype="<i2").tobytes() + b"\x7f"

    np.testing.assert_array_equal(decode_pcm(raw), np.array([1, -1], dtype=np.int16))


def test_resample_keeps_the_tone_while_changing_the_rate() -> None:
    """Downsampling 24 kHz speech PCM to 16 kHz should preserve length ratio and pitch."""
    source_rate, target_rate, frequency = 24000, 16000, 440.0
    time_points = np.arange(source_rate, dtype=np.float32) / source_rate
    tone = (np.sin(2 * np.pi * frequency * time_points) * 16000).astype(np.int16)

    resampled = resample(tone, source_rate, target_rate)

    assert resampled.dtype == np.int16
    assert abs(resampled.size - target_rate) <= 1
    spectrum = np.abs(np.fft.rfft(resampled.astype(np.float64)))
    peak_frequency = np.fft.rfftfreq(resampled.size, 1 / target_rate)[int(np.argmax(spectrum))]
    assert abs(peak_frequency - frequency) < 5.0


def test_resample_is_a_no_op_at_the_same_rate() -> None:
    """Matching rates should return the samples untouched."""
    samples = np.array([1, 2, 3], dtype=np.int16)

    assert resample(samples, 16000, 16000) is samples
