"""Speech and language services behind the direct conversation backend.

Each stage is a small protocol with one OpenAI-compatible implementation, so a
stage can be pointed at another provider through ``DIRECT_*_BASE_URL`` (a local
Kokoro server for speech, for instance) or replaced by a new class without
touching the handler.
"""

import uuid
import logging
from typing import Final, Protocol
from dataclasses import dataclass
from collections.abc import Sequence, AsyncIterator

import numpy as np
from openai import AsyncOpenAI
from numpy.typing import NDArray
from huggingface_hub import get_token
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionFunctionToolParam,
)

from reachy_mini_conversation_ja.config import DirectBackendSettings
from reachy_mini_conversation_ja.audio.pcm import decode_pcm, encode_wav
from reachy_mini_conversation_ja.tools.core_tools import ToolSpec


logger = logging.getLogger(__name__)

_STT_TIMEOUT_S: Final[float] = 30.0
_LLM_TIMEOUT_S: Final[float] = 60.0
_TTS_TIMEOUT_S: Final[float] = 30.0
# Roughly 170 ms of 24 kHz audio: small enough to start playback early, large
# enough that the resampler is not called per handful of samples.
_TTS_CHUNK_BYTES: Final[int] = 8192

# The voice catalog the UI and profiles use comes from the deployed Hugging Face
# backend's Qwen3-TTS speakers, so map it onto the provider's own voices. Set
# DIRECT_TTS_VOICE to bypass the map when pointing at a provider of your own.
_OPENAI_VOICE_BY_CATALOG_NAME: Final[dict[str, str]] = {
    "Aiden": "ash",
    "Ryan": "onyx",
    "Dylan": "echo",
    "Eric": "alloy",
    "Ono_Anna": "nova",
    "Serena": "shimmer",
    "Sohee": "coral",
    "Uncle_Fu": "fable",
    "Vivian": "sage",
}
_DEFAULT_PROVIDER_VOICE: Final[str] = "alloy"

_SENTENCE_END_CHARS: Final[str] = "。．！？!?…\n"
_SOFT_BREAK_CHARS: Final[str] = "、，,;:）)"
_MAX_CHUNK_CHARS: Final[int] = 160
# The first piece gates when Reachy starts talking, so let a comma end it once
# there is enough to say. Later pieces wait for a sentence, which reads better.
_FIRST_PIECE_MIN_CHARS: Final[int] = 10


@dataclass(frozen=True)
class TextDelta:
    """A streamed piece of assistant text."""

    text: str


@dataclass(frozen=True)
class ToolCallRequest:
    """A completed function call the model wants executed."""

    call_id: str
    name: str
    arguments: str


ChatEvent = TextDelta | ToolCallRequest


class SpeechToText(Protocol):
    """Transcribes one complete user utterance."""

    async def transcribe(self, samples: NDArray[np.int16], sample_rate: int) -> str:
        """Return the transcript of mono int16 PCM sampled at ``sample_rate``."""
        ...


class TextToSpeech(Protocol):
    """Synthesizes speech as mono int16 PCM at ``sample_rate``."""

    sample_rate: int

    def stream(self, text: str, voice: str) -> AsyncIterator[NDArray[np.int16]]:
        """Yield PCM for ``text`` as it is synthesized, in the catalog voice ``voice``."""
        ...


class ChatModel(Protocol):
    """Streams assistant text and tool calls for a message history."""

    def stream(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tool_specs: Sequence[ToolSpec],
    ) -> AsyncIterator[ChatEvent]:
        """Yield text deltas as they arrive, then any requested tool calls."""
        ...


def to_chat_tools_config(tool_specs: Sequence[ToolSpec]) -> list[ChatCompletionFunctionToolParam]:
    """Convert app tool specs to the chat-completions function shape."""
    return [
        ChatCompletionFunctionToolParam(
            type="function",
            function={
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        )
        for spec in tool_specs
    ]


class SentenceBuffer:
    """Split streamed assistant text into sentence-sized pieces for synthesis."""

    def __init__(self) -> None:
        """Start with an empty buffer."""
        self._buffer = ""
        self._released_any = False

    def push(self, text: str) -> list[str]:
        """Add streamed text and return the pieces that are ready to speak."""
        self._buffer += text
        pieces: list[str] = []
        while True:
            piece = self._take_piece()
            if piece is None:
                return pieces
            if piece.strip():
                pieces.append(piece.strip())
                self._released_any = True

    def flush(self) -> str:
        """Return whatever is left, emptying the buffer."""
        remainder, self._buffer = self._buffer.strip(), ""
        if remainder:
            self._released_any = True
        return remainder

    def _take_piece(self) -> str | None:
        """Cut off the next speakable piece, or None while the buffer is still short."""
        end = self._sentence_end()
        if end is None and not self._released_any:
            end = self._first_piece_break()
        if end is None:
            if len(self._buffer) < _MAX_CHUNK_CHARS:
                return None
            soft_break = self._soft_break()
            end = soft_break if soft_break is not None else _MAX_CHUNK_CHARS - 1
        piece, self._buffer = self._buffer[: end + 1], self._buffer[end + 1 :]
        return piece

    def _sentence_end(self) -> int | None:
        """Return the index of the first sentence terminator, including a run of them."""
        for index, char in enumerate(self._buffer):
            if char == ".":
                if not self._ends_sentence(index):
                    continue
            elif char not in _SENTENCE_END_CHARS:
                continue
            end = index
            while end + 1 < len(self._buffer) and self._buffer[end + 1] in _SENTENCE_END_CHARS:
                end += 1
            return end
        return None

    def _ends_sentence(self, index: int) -> bool:
        """Return whether the period at ``index`` ends a sentence rather than a decimal."""
        previous = self._buffer[index - 1] if index else ""
        if not previous.isdigit():
            return True
        following = self._buffer[index + 1 :]
        return bool(following) and not following[0].isdigit()

    def _first_piece_break(self) -> int | None:
        """Return the first comma-like break past the minimum length for an opening piece."""
        for index in range(_FIRST_PIECE_MIN_CHARS - 1, len(self._buffer)):
            if self._buffer[index] in _SOFT_BREAK_CHARS:
                return index
        return None

    def _soft_break(self) -> int | None:
        """Return the last comma-like break inside the maximum chunk length."""
        window = self._buffer[:_MAX_CHUNK_CHARS]
        for index in range(len(window) - 1, -1, -1):
            if window[index] in _SOFT_BREAK_CHARS:
                return index
        return None


class OpenAICompatibleSpeechToText:
    """Transcribes utterances through an OpenAI-compatible /v1/audio/transcriptions endpoint."""

    def __init__(self, client: AsyncOpenAI, model: str, language: str) -> None:
        """Store the client, transcription model and spoken language."""
        self._client = client
        self._model = model
        self._language = language

    async def transcribe(self, samples: NDArray[np.int16], sample_rate: int) -> str:
        """Upload the utterance as WAV and return its transcript."""
        transcription = await self._client.audio.transcriptions.create(
            model=self._model,
            file=("utterance.wav", encode_wav(samples, sample_rate), "audio/wav"),
            language=self._language,
        )
        return transcription.text


class OpenAICompatibleTextToSpeech:
    """Synthesizes speech through an OpenAI-compatible /v1/audio/speech endpoint."""

    def __init__(self, client: AsyncOpenAI, model: str, sample_rate: int, voice_override: str | None) -> None:
        """Store the client, model, PCM rate and optional provider voice override."""
        self._client = client
        self._model = model
        self.sample_rate = sample_rate
        self._voice_override = voice_override

    async def stream(self, text: str, voice: str) -> AsyncIterator[NDArray[np.int16]]:
        """Yield PCM as the provider produces it, so playback need not wait for the whole sentence."""
        partial_sample = b""
        async with self._client.audio.speech.with_streaming_response.create(
            model=self._model,
            voice=self._voice_override or _OPENAI_VOICE_BY_CATALOG_NAME.get(voice, _DEFAULT_PROVIDER_VOICE),
            input=text,
            response_format="pcm",
        ) as response:
            async for block in response.iter_bytes(chunk_size=_TTS_CHUNK_BYTES):
                if not block:
                    continue
                # A block can split a sample in half; carry the odd byte over.
                partial_sample += block
                whole_samples = len(partial_sample) - len(partial_sample) % 2
                if whole_samples:
                    yield decode_pcm(partial_sample[:whole_samples])
                    partial_sample = partial_sample[whole_samples:]


class OpenAICompatibleChatModel:
    """Streams responses through an OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        """Store the client and chat model name."""
        self._client = client
        self._model = model

    async def stream(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tool_specs: Sequence[ToolSpec],
    ) -> AsyncIterator[ChatEvent]:
        """Yield text deltas while the response streams, then the requested tool calls."""
        tools = to_chat_tools_config(tool_specs)
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=list(messages),
            tools=tools or [],
            stream=True,
        )
        calls_by_index: dict[int, dict[str, str]] = {}
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if delta.content:
                yield TextDelta(delta.content)
            for tool_call in delta.tool_calls or ():
                call = calls_by_index.setdefault(tool_call.index, {"id": "", "name": "", "arguments": ""})
                if tool_call.id:
                    call["id"] = tool_call.id
                function = getattr(tool_call, "function", None)
                if function is None:
                    continue
                if function.name:
                    call["name"] = function.name
                if function.arguments:
                    call["arguments"] += function.arguments

        for call in calls_by_index.values():
            if not call["name"]:
                logger.warning("Ignoring streamed tool call without a name: %s", call)
                continue
            yield ToolCallRequest(
                call_id=call["id"] or uuid.uuid4().hex,
                name=call["name"],
                arguments=call["arguments"] or "{}",
            )


@dataclass(frozen=True)
class SpeechServices:
    """The direct backend's transcription, chat and synthesis stages."""

    speech_to_text: SpeechToText
    chat_model: ChatModel
    text_to_speech: TextToSpeech
    clients: tuple[AsyncOpenAI, ...]

    async def aclose(self) -> None:
        """Close the underlying HTTP clients."""
        for client in self.clients:
            try:
                await client.close()
            except Exception as e:
                logger.debug("Ignoring speech service client close error: %s", e)


def build_speech_services(settings: DirectBackendSettings) -> SpeechServices:
    """Build every direct-backend stage, raising when a credential is missing."""
    if not settings.stt_api_key:
        raise RuntimeError("Set OPENAI_API_KEY (or DIRECT_STT_API_KEY) to transcribe with the direct backend.")
    if not settings.tts_api_key:
        raise RuntimeError("Set OPENAI_API_KEY (or DIRECT_TTS_API_KEY) to synthesize with the direct backend.")
    llm_api_key = settings.llm_api_key or (get_token() or "").strip()
    if not llm_api_key:
        raise RuntimeError(
            "Set HF_TOKEN, OPENAI_API_KEY or DIRECT_LLM_API_KEY to reach the direct backend's language model."
        )

    stt_client = AsyncOpenAI(api_key=settings.stt_api_key, base_url=settings.stt_base_url, timeout=_STT_TIMEOUT_S)
    llm_client = AsyncOpenAI(api_key=llm_api_key, base_url=settings.llm_base_url, timeout=_LLM_TIMEOUT_S)
    tts_client = AsyncOpenAI(api_key=settings.tts_api_key, base_url=settings.tts_base_url, timeout=_TTS_TIMEOUT_S)
    return SpeechServices(
        speech_to_text=OpenAICompatibleSpeechToText(stt_client, settings.stt_model, settings.stt_language),
        chat_model=OpenAICompatibleChatModel(llm_client, settings.llm_model),
        text_to_speech=OpenAICompatibleTextToSpeech(
            tts_client,
            settings.tts_model,
            settings.tts_sample_rate,
            settings.tts_voice,
        ),
        clients=(stt_client, llm_client, tts_client),
    )
