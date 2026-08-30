"""Tests for the direct backend's speech and language services."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from reachy_mini_conversation_app.config import DirectBackendSettings
from reachy_mini_conversation_app.speech_services import (
    TextDelta,
    SentenceBuffer,
    ToolCallRequest,
    OpenAICompatibleChatModel,
    OpenAICompatibleSpeechToText,
    OpenAICompatibleTextToSpeech,
    to_chat_tools_config,
    build_speech_services,
)


def _settings(**overrides: Any) -> DirectBackendSettings:
    """Build direct backend settings with every credential present."""
    values: dict[str, Any] = {
        "stt_model": "stt",
        "stt_base_url": None,
        "stt_api_key": "key",
        "stt_language": "ja",
        "llm_model": "llm",
        "llm_base_url": "https://router.example/v1",
        "llm_api_key": "key",
        "tts_model": "tts",
        "tts_base_url": None,
        "tts_api_key": "key",
        "tts_sample_rate": 24000,
        "tts_voice": None,
    }
    values.update(overrides)
    return DirectBackendSettings(**values)


def test_sentence_buffer_releases_japanese_sentences() -> None:
    """Japanese terminators should release one speakable sentence each."""
    buffer = SentenceBuffer()

    assert buffer.push("こんにちは。元気") == ["こんにちは。"]
    assert buffer.push("ですか？はい") == ["元気ですか？"]
    assert buffer.flush() == "はい"


def test_sentence_buffer_keeps_decimals_together() -> None:
    """A period between digits should not split a number."""
    buffer = SentenceBuffer()

    assert buffer.push("円周率は3.14です") == []
    assert buffer.push("。") == ["円周率は3.14です。"]


def test_sentence_buffer_splits_long_text_without_terminators() -> None:
    """Text that never ends a sentence should still reach synthesis in pieces."""
    buffer = SentenceBuffer()

    pieces = buffer.push("あ" * 100 + "、" + "い" * 100)

    assert pieces == ["あ" * 100 + "、"]


def test_to_chat_tools_config_wraps_specs_in_functions() -> None:
    """App tool specs should become chat-completions function tools."""
    spec = {"type": "function", "name": "dance", "description": "Dance", "parameters": {"type": "object"}}

    assert to_chat_tools_config([spec]) == [
        {
            "type": "function",
            "function": {"name": "dance", "description": "Dance", "parameters": {"type": "object"}},
        }
    ]


def _chunk(content: str | None = None, tool_calls: list[Any] | None = None) -> SimpleNamespace:
    """Build one streamed chat completion chunk."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _tool_call_chunk(index: int, call_id: str | None, name: str | None, arguments: str | None) -> SimpleNamespace:
    """Build one streamed tool call fragment."""
    return SimpleNamespace(index=index, id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class _FakeStream:
    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> "_FakeStream":
        return self

    async def __anext__(self) -> SimpleNamespace:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration


def _fake_chat_client(chunks: list[SimpleNamespace], captured: dict[str, Any]) -> Any:
    """Return a client whose chat completions stream ``chunks``."""

    async def create(**kwargs: Any) -> _FakeStream:
        captured.update(kwargs)
        return _FakeStream(chunks)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.mark.asyncio
async def test_chat_model_streams_text_then_assembled_tool_calls() -> None:
    """Text should arrive as it streams and tool call fragments should be joined."""
    captured: dict[str, Any] = {}
    chunks = [
        _chunk(content="踊り"),
        _chunk(content="ます"),
        _chunk(tool_calls=[_tool_call_chunk(0, "call-1", "dance", '{"name":')]),
        _chunk(tool_calls=[_tool_call_chunk(0, None, None, '"macarena"}')]),
    ]
    model = OpenAICompatibleChatModel(_fake_chat_client(chunks, captured), "llm")

    events = [event async for event in model.stream([{"role": "user", "content": "踊って"}], [])]

    assert events == [
        TextDelta("踊り"),
        TextDelta("ます"),
        ToolCallRequest(call_id="call-1", name="dance", arguments='{"name":"macarena"}'),
    ]
    assert captured["model"] == "llm"
    assert captured["stream"] is True


@pytest.mark.asyncio
async def test_speech_to_text_uploads_a_wav_and_returns_the_transcript() -> None:
    """The utterance should be uploaded as WAV with the configured model and language."""
    captured: dict[str, Any] = {}

    async def create(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(text="おはよう")

    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    speech_to_text = OpenAICompatibleSpeechToText(client, "stt", "ja")

    transcript = await speech_to_text.transcribe(np.zeros(160, dtype=np.int16), 16000)

    assert transcript == "おはよう"
    assert (captured["model"], captured["language"]) == ("stt", "ja")
    assert captured["file"][0] == "utterance.wav"
    assert captured["file"][1].startswith(b"RIFF")


@pytest.mark.asyncio
async def test_text_to_speech_maps_the_catalog_voice_and_decodes_pcm() -> None:
    """A catalog voice should reach the provider as one of its own voices."""
    captured: dict[str, Any] = {}
    pcm = np.array([1, -1, 2], dtype=np.int16)

    async def create(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)

        async def aread() -> bytes:
            return pcm.tobytes()

        return SimpleNamespace(aread=aread)

    client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(create=create)))
    text_to_speech = OpenAICompatibleTextToSpeech(client, "tts", 24000, voice_override=None)

    np.testing.assert_array_equal(await text_to_speech.synthesize("やあ", "Ono_Anna"), pcm)
    assert captured["voice"] == "nova"
    assert captured["response_format"] == "pcm"


@pytest.mark.asyncio
async def test_text_to_speech_override_wins_over_the_catalog() -> None:
    """A configured provider voice should bypass the catalog mapping."""
    captured: dict[str, Any] = {}

    async def create(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)

        async def aread() -> bytes:
            return b""

        return SimpleNamespace(aread=aread)

    client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(create=create)))
    text_to_speech = OpenAICompatibleTextToSpeech(client, "tts", 24000, voice_override="jf_alpha")

    await text_to_speech.synthesize("やあ", "Ono_Anna")

    assert captured["voice"] == "jf_alpha"


def test_build_speech_services_reports_the_missing_credential() -> None:
    """A missing key should name the variable that fixes it."""
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_speech_services(_settings(stt_api_key=""))


def test_build_speech_services_reports_a_missing_language_model_token(monkeypatch) -> None:
    """The language model key should fall back to the Hugging Face token, then fail clearly."""
    monkeypatch.setattr("reachy_mini_conversation_app.speech_services.get_token", lambda: None)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        build_speech_services(_settings(llm_api_key=""))


def test_build_speech_services_wires_every_stage() -> None:
    """With credentials present, every stage and its client should be built."""
    services = build_speech_services(_settings())

    assert services.text_to_speech.sample_rate == 24000
    assert len(services.clients) == 3
