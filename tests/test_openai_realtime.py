"""Tests for the realtime backend."""

import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import reachy_mini_conversation_ja.openai_realtime as realtime_mod
from reachy_mini_conversation_ja.openai_realtime import OpenAIRealtimeHandler
from reachy_mini_conversation_ja.tools.core_tools import ToolDependencies


SAMPLE_RATE = 16000


def _tone(level: float, duration_s: float) -> np.ndarray:
    """Return audio whose RMS is ``level`` on the 0..1 scale."""
    count = int(SAMPLE_RATE * duration_s)
    return ((np.arange(count) % 2) * 2 - 1).astype(np.int16) * np.int16(int(level * 32768))


def _silence(duration_s: float) -> np.ndarray:
    """Return digital silence."""
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.int16)


def _handler(monkeypatch: Any) -> OpenAIRealtimeHandler:
    """Build a handler with a fake open connection."""
    monkeypatch.setattr(realtime_mod, "get_session_instructions", lambda _instance_path=None: "instructions")
    handler = OpenAIRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    connection = MagicMock()
    connection.input_audio_buffer.append = AsyncMock()
    connection.input_audio_buffer.commit = AsyncMock()
    connection.response.cancel = AsyncMock()
    handler.connection = connection
    return handler


@pytest.mark.asyncio
async def test_silence_is_never_sent(monkeypatch: Any) -> None:
    """Audio input is billed by the token, so a quiet room must cost nothing."""
    handler = _handler(monkeypatch)

    await handler.receive((SAMPLE_RATE, _silence(2.0)))

    handler.connection.input_audio_buffer.append.assert_not_awaited()
    handler.connection.input_audio_buffer.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_speech_streams_and_the_turn_is_committed(monkeypatch: Any) -> None:
    """Audio flows while someone talks, and the turn closes once they stop."""
    handler = _handler(monkeypatch)
    handler._safe_response_create = AsyncMock()  # type: ignore[method-assign]

    await handler.receive((SAMPLE_RATE, _silence(1.0)))
    await handler.receive((SAMPLE_RATE, _tone(0.05, 0.6)))
    streamed_during_speech = handler.connection.input_audio_buffer.append.await_count
    await handler.receive((SAMPLE_RATE, _silence(1.0)))

    assert streamed_during_speech > 0, "audio should reach the server while the user speaks"
    handler.connection.input_audio_buffer.commit.assert_awaited_once()
    handler._safe_response_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_audio_before_the_onset_is_not_lost(monkeypatch: Any) -> None:
    """The first word arrives before the onset is certain, so it is buffered and sent."""
    handler = _handler(monkeypatch)
    handler._safe_response_create = AsyncMock()  # type: ignore[method-assign]

    await handler.receive((SAMPLE_RATE, _silence(1.0)))
    await handler.receive((SAMPLE_RATE, _tone(0.05, 0.6)))

    sent = b"".join(
        base64.b64decode(call.kwargs["audio"]) for call in handler.connection.input_audio_buffer.append.await_args_list
    )
    # Sent at 24 kHz, so 0.6 s of speech plus pre-roll is more than 0.6 s worth of samples.
    assert len(sent) / 2 > 0.6 * handler._realtime_rate


def test_the_session_turns_the_server_detector_off(monkeypatch: Any) -> None:
    """Server-side detection would require streaming silence, which costs money."""
    handler = _handler(monkeypatch)

    session = handler._get_session_config([])

    assert session["audio"]["input"]["turn_detection"] is None
    assert session["audio"]["input"]["format"]["rate"] == handler._realtime_rate
    assert session["audio"]["output"]["format"]["rate"] == handler._realtime_rate
    assert session["audio"]["input"]["transcription"]["model"] == "gpt-4o-transcribe"


def test_the_catalog_voice_maps_onto_a_realtime_voice(monkeypatch: Any) -> None:
    """Profiles name a Hugging Face speaker, which has to become a realtime voice."""
    handler = _handler(monkeypatch)
    handler._realtime_voice_override = None
    monkeypatch.setattr(handler, "get_current_voice", lambda: "Ono_Anna")

    assert handler._realtime_voice() == "marin"

    handler._realtime_voice_override = "cedar"

    assert handler._realtime_voice() == "cedar"


def test_the_reply_comes_down_to_the_speaker_rate(monkeypatch: Any) -> None:
    """The session speaks at 24 kHz and the speaker runs at 16 kHz."""
    handler = _handler(monkeypatch)
    block = np.zeros(handler._realtime_rate // 10, dtype=np.int16)

    decoded = handler._decode_output_audio(base64.b64encode(block.tobytes()).decode())

    assert abs(decoded.size - SAMPLE_RATE // 10) <= 2
