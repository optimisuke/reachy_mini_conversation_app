"""Tests for the direct conversation backend."""

import time
import asyncio
from typing import Any
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from collections.abc import Sequence, AsyncIterator

import numpy as np
import pytest

import reachy_mini_conversation_ja.direct_cascade as direct_mod
from reachy_mini_conversation_ja.tools import background_tool_manager
from reachy_mini_conversation_ja.streaming import AdditionalOutputs
from reachy_mini_conversation_ja.direct_cascade import SpeechRequest, DirectCascadeHandler
from reachy_mini_conversation_ja.voice_activity import UtteranceEvent
from reachy_mini_conversation_ja.speech_services import (
    ChatEvent,
    TextDelta,
    SpeechServices,
    ToolCallRequest,
)
from reachy_mini_conversation_ja.tools.core_tools import ToolDependencies


SAMPLE_RATE = 16000
PROFILE_VOICE = "Ono_Anna"


class _FakeSpeechToText:
    """Returns queued transcripts, then empty ones."""

    def __init__(self, transcripts: Sequence[str]) -> None:
        self.transcripts = list(transcripts)
        self.utterance_durations: list[float] = []

    async def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        self.utterance_durations.append(samples.size / sample_rate)
        return self.transcripts.pop(0) if self.transcripts else ""


class _FakeTextToSpeech:
    """Records what it was asked to say and streams 100 ms of audio in two blocks."""

    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.spoken: list[tuple[str, str]] = []

    async def stream(self, text: str, voice: str) -> AsyncIterator[np.ndarray]:
        self.spoken.append((text, voice))
        for _ in range(2):
            yield np.zeros(SAMPLE_RATE // 20, dtype=np.int16)


class _FakeChatModel:
    """Replays one scripted list of events per response."""

    def __init__(self, rounds: Sequence[Sequence[ChatEvent]]) -> None:
        self.rounds = [list(events) for events in rounds]
        self.seen_messages: list[list[dict[str, Any]]] = []

    async def stream(self, messages: Sequence[Any], tool_specs: Sequence[Any]) -> AsyncIterator[ChatEvent]:
        self.seen_messages.append([dict(message) for message in messages])
        events = self.rounds.pop(0) if self.rounds else []
        for event in events:
            if callable(event):
                await event()
                continue
            yield event


def _tone(level: float, duration_s: float) -> np.ndarray:
    """Return audio whose RMS is ``level`` on the 0..1 scale."""
    sample_count = int(SAMPLE_RATE * duration_s)
    return ((np.arange(sample_count) % 2) * 2 - 1).astype(np.int16) * np.int16(int(level * 32768))


def _silence(duration_s: float) -> np.ndarray:
    """Return digital silence."""
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.int16)


def _make_handler(
    monkeypatch: Any,
    *,
    transcripts: Sequence[str] = (),
    rounds: Sequence[Sequence[ChatEvent]] = (),
    greeting: str = "",
) -> tuple[DirectCascadeHandler, _FakeSpeechToText, _FakeChatModel, _FakeTextToSpeech]:
    """Build a handler wired to scripted speech services."""
    monkeypatch.setattr(direct_mod, "get_session_instructions", lambda _instance_path=None: "instructions")
    monkeypatch.setattr(direct_mod, "get_session_voice", lambda default=None: PROFILE_VOICE)
    monkeypatch.setattr(direct_mod, "get_session_greeting_prompt", lambda: greeting)
    monkeypatch.setattr(direct_mod, "get_tool_specs", lambda: [])

    speech_to_text = _FakeSpeechToText(transcripts)
    chat_model = _FakeChatModel(rounds)
    text_to_speech = _FakeTextToSpeech()
    monkeypatch.setattr(
        direct_mod,
        "build_speech_services",
        lambda _settings: SpeechServices(speech_to_text, chat_model, text_to_speech, clients=()),
    )

    movement_manager = MagicMock()
    movement_manager.is_idle.return_value = True
    handler = DirectCascadeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager))
    return handler, speech_to_text, chat_model, text_to_speech


async def _wait_for(predicate: Any, timeout: float = 3.0) -> None:
    """Wait until ``predicate`` holds, giving the handler's tasks time to run."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met in time")


@asynccontextmanager
async def _running(handler: DirectCascadeHandler) -> AsyncIterator[None]:
    """Serve a session for the duration of the block."""
    session = asyncio.create_task(handler.start_up())
    try:
        await _wait_for(handler._is_connected)
        yield
    finally:
        await handler.shutdown()
        await asyncio.wait_for(session, timeout=3.0)


def _transcript_messages(handler: DirectCascadeHandler) -> list[tuple[str, str]]:
    """Drain the output queue and return its (role, content) pairs."""
    pairs: list[tuple[str, str]] = []
    while not handler.output_queue.empty():
        item = handler.output_queue.get_nowait()
        if isinstance(item, AdditionalOutputs):
            pairs.extend((str(message["role"]), str(message["content"])) for message in item.args)
    return pairs


def _drain_audio_frames(handler: DirectCascadeHandler, collected: list[Any]) -> list[Any]:
    """Move any queued audio frames into ``collected`` and return it."""
    while not handler.output_queue.empty():
        item = handler.output_queue.get_nowait()
        if isinstance(item, tuple):
            collected.append(item)
    return collected


async def _say_something(handler: DirectCascadeHandler) -> None:
    """Feed one spoken utterance followed by enough silence to close it."""
    await handler.receive((SAMPLE_RATE, _tone(0.08, 0.9)))
    await handler.receive((SAMPLE_RATE, _silence(1.0)))


@pytest.mark.asyncio
async def test_utterance_is_transcribed_answered_and_spoken(monkeypatch: Any) -> None:
    """A finished utterance should produce a transcript, an answer and audio."""
    handler, speech_to_text, chat_model, text_to_speech = _make_handler(
        monkeypatch,
        transcripts=["こんにちは"],
        rounds=[[TextDelta("やあ。"), TextDelta("元気？")]],
    )

    async with _running(handler):
        await _say_something(handler)
        await _wait_for(lambda: len(text_to_speech.spoken) == 2)
        pairs = _transcript_messages(handler)

    assert ("user", "こんにちは") in pairs
    assert ("assistant", "やあ。元気？") in pairs
    assert text_to_speech.spoken == [("やあ。", PROFILE_VOICE), ("元気？", PROFILE_VOICE)]
    assert speech_to_text.utterance_durations[0] > 0.6
    assert chat_model.seen_messages[0][0] == {"role": "system", "content": "instructions"}


@pytest.mark.asyncio
async def test_startup_greeting_opens_the_conversation(monkeypatch: Any) -> None:
    """The profile greeting should be prompted for and spoken without user audio."""
    handler, _stt, chat_model, text_to_speech = _make_handler(
        monkeypatch,
        rounds=[[TextDelta("やあ！")]],
        greeting="短くあいさつして",
    )

    async with _running(handler):
        await _wait_for(lambda: bool(text_to_speech.spoken))

    assert text_to_speech.spoken == [("やあ！", PROFILE_VOICE)]
    assert chat_model.seen_messages[0][-1] == {"role": "user", "content": "短くあいさつして"}


@pytest.mark.asyncio
async def test_audio_reaches_the_player_in_chunks(monkeypatch: Any) -> None:
    """Synthesized speech should be queued as playable frames at the app sample rate."""
    handler, _stt, _chat, text_to_speech = _make_handler(monkeypatch, rounds=[[TextDelta("やあ。")]], greeting="hi")

    frames: list[Any] = []
    async with _running(handler):
        await _wait_for(lambda: bool(_drain_audio_frames(handler, frames)))

    assert frames
    assert {rate for rate, _ in frames} == {SAMPLE_RATE}
    assert sum(samples.size for _, samples in frames) == SAMPLE_RATE // 10


@pytest.mark.asyncio
async def test_tool_result_feeds_a_second_response(monkeypatch: Any) -> None:
    """A tool call should run, report back to the model and let it answer."""
    dispatch = AsyncMock(return_value={"b64_im": "ignored", "image_width": 4})
    monkeypatch.setattr(background_tool_manager, "dispatch_tool_call", dispatch)
    handler, _stt, chat_model, text_to_speech = _make_handler(
        monkeypatch,
        transcripts=["写真とって"],
        rounds=[
            [ToolCallRequest(call_id="call-1", name="camera", arguments="{}")],
            [TextDelta("撮ったよ。")],
        ],
    )

    async with _running(handler):
        await _say_something(handler)
        await _wait_for(lambda: bool(text_to_speech.spoken))

    assert dispatch.await_args.kwargs["tool_name"] == "camera"
    tool_messages = [message for message in chat_model.seen_messages[-1] if message["role"] == "tool"]
    assert tool_messages and tool_messages[0]["tool_call_id"] == "call-1"
    # The bulky image payload must not be echoed back into the history.
    assert "b64_im" not in tool_messages[0]["content"]
    assert "image_attached" in tool_messages[0]["content"]
    # Without a vision model the picture cannot be shown, so say so instead of guessing.
    assert "cannot see images" in tool_messages[0]["content"]
    assert not any(isinstance(message.get("content"), list) for message in chat_model.seen_messages[-1])
    assert text_to_speech.spoken == [("撮ったよ。", PROFILE_VOICE)]


@pytest.mark.asyncio
async def test_barge_in_drops_the_response_in_flight(monkeypatch: Any) -> None:
    """New speech while Reachy talks should cancel the turn and flush the player."""
    never_finishes = asyncio.Event()

    async def block() -> None:
        await never_finishes.wait()

    handler, _stt, _chat, text_to_speech = _make_handler(
        monkeypatch,
        transcripts=["ねえ", "やっぱり別の話"],
        rounds=[[TextDelta("長い話をします。"), block], [TextDelta("了解。")]],
    )
    flushes: list[bool] = []
    handler._clear_queue = lambda: flushes.append(True)

    async with _running(handler):
        await _say_something(handler)
        await _wait_for(lambda: bool(text_to_speech.spoken))
        await _say_something(handler)
        await _wait_for(lambda: bool(flushes))

    assert flushes
    never_finishes.set()


@pytest.mark.asyncio
async def test_say_speaks_the_text_verbatim(monkeypatch: Any) -> None:
    """Unlike the speech-to-speech backend, say() synthesizes exactly what it is given."""
    handler, _stt, _chat, text_to_speech = _make_handler(monkeypatch)

    async with _running(handler):
        await handler.say("充電しておくね")
        await _wait_for(lambda: bool(text_to_speech.spoken))
        pairs = _transcript_messages(handler)

    assert text_to_speech.spoken == [("充電しておくね", PROFILE_VOICE)]
    assert ("assistant", "充電しておくね") in pairs


@pytest.mark.asyncio
async def test_say_without_a_session_fails(monkeypatch: Any) -> None:
    """Speaking before the session is up should raise instead of dropping the text."""
    handler, _stt, _chat, _tts = _make_handler(monkeypatch)

    with pytest.raises(RuntimeError, match="no active session"):
        await handler.say("hello")


@pytest.mark.asyncio
async def test_change_voice_falls_back_to_a_catalog_voice(monkeypatch: Any) -> None:
    """An unsupported voice should fall back instead of reaching the provider."""
    handler, _stt, _chat, _tts = _make_handler(monkeypatch)

    assert await handler.change_voice("ono_anna") == f"Voice changed to {PROFILE_VOICE}."
    assert handler.get_current_voice() == PROFILE_VOICE

    await handler.change_voice("not-a-voice")

    assert handler.get_current_voice() in await handler.get_available_voices()


@pytest.mark.asyncio
async def test_applying_a_personality_restarts_the_history(monkeypatch: Any) -> None:
    """A new personality should replace the system prompt and drop the old turns."""
    handler, _stt, _chat, _tts = _make_handler(monkeypatch)
    monkeypatch.setattr(direct_mod.core_tools, "initialize_tools", MagicMock())
    async with _running(handler):
        await handler.say("古い話")

        status = await handler.apply_personality("mars_rover")

        assert status == "Applied personality."
        assert handler._messages == [{"role": "system", "content": "instructions"}]


@pytest.mark.asyncio
async def test_a_cancelled_tool_turn_leaves_no_unanswered_request(monkeypatch: Any) -> None:
    """A tool request must never outlive its replies: the model rejects that history."""
    tool_never_finishes = asyncio.Event()

    async def block(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        await tool_never_finishes.wait()
        return {}

    monkeypatch.setattr(background_tool_manager, "dispatch_tool_call", block)
    handler, _stt, chat_model, _tts = _make_handler(
        monkeypatch,
        transcripts=["写真とって"],
        rounds=[[ToolCallRequest(call_id="call-1", name="camera", arguments="{}")]],
    )

    async with _running(handler):
        await _say_something(handler)
        await _wait_for(lambda: bool(chat_model.seen_messages))
        await _wait_for(lambda: handler._pending_tool_results != {})
        await handler._cancel_active_turn()

        assert not any(message.get("tool_calls") for message in handler._messages)

    tool_never_finishes.set()


@pytest.mark.asyncio
async def test_a_rejected_response_rolls_the_turn_back(monkeypatch: Any) -> None:
    """A failed response must not poison the history, or every later turn fails too."""

    class _FailingChatModel:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, messages: Sequence[Any], tool_specs: Sequence[Any]) -> AsyncIterator[ChatEvent]:
            self.calls += 1
            raise RuntimeError("400 invalid_request_error")
            yield TextDelta("")  # pragma: no cover - keeps this an async generator

    handler, _stt, _chat, _tts = _make_handler(monkeypatch, transcripts=["こんにちは"])
    failing = _FailingChatModel()

    async with _running(handler):
        assert handler._services is not None
        object.__setattr__(handler._services, "chat_model", failing)
        history_before = list(handler._messages)

        await _say_something(handler)
        await _wait_for(lambda: failing.calls == 1)
        await _wait_for(lambda: handler._turn_task is None)
        pairs = _transcript_messages(handler)

    # The user turn stays, the failed assistant turn leaves nothing behind.
    assert handler._messages == [*history_before, {"role": "user", "content": "こんにちは"}]
    assert any("[error]" in content for _role, content in pairs)


@pytest.mark.asyncio
async def test_noise_while_thinking_does_not_drop_the_answer(monkeypatch: Any) -> None:
    """Only speech over Reachy's own voice interrupts; a turn still thinking is left alone."""
    handler, _stt, _chat, _tts = _make_handler(monkeypatch, rounds=[[TextDelta("はい。")]])
    flushes: list[bool] = []
    handler._clear_queue = lambda: flushes.append(True)

    async with _running(handler):
        handler._turn_task = asyncio.create_task(asyncio.sleep(5), name="pretend-turn")
        turn = handler._turn_task

        await handler._on_speech_started(UtteranceEvent(speech_started=True))

        assert handler._turn_task is turn
        assert not turn.cancelled()
        assert flushes == []
        turn.cancel()
        handler._turn_task = None


@pytest.mark.asyncio
async def test_a_vision_model_is_shown_the_camera_image(monkeypatch: Any) -> None:
    """With vision enabled the picture itself reaches the model, not a claim about it."""
    monkeypatch.setattr(background_tool_manager, "dispatch_tool_call", AsyncMock(return_value={"b64_im": "SkZJRg=="}))
    handler, _stt, chat_model, text_to_speech = _make_handler(
        monkeypatch,
        transcripts=["何が見える？"],
        rounds=[
            [ToolCallRequest(call_id="call-1", name="camera", arguments="{}")],
            [TextDelta("机が見えるよ。")],
        ],
    )

    async with _running(handler):
        handler._llm_vision = True
        await _say_something(handler)
        await _wait_for(lambda: bool(text_to_speech.spoken))

    image_parts = [
        part
        for message in chat_model.seen_messages[-1]
        if isinstance(message.get("content"), list)
        for part in message["content"]
    ]
    assert image_parts == [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,SkZJRg=="}}]


@pytest.mark.asyncio
async def test_a_turn_reports_where_the_wait_went(monkeypatch: Any, caplog: Any) -> None:
    """Each turn logs one line attributing the wait to a stage, so it can be tuned."""
    handler, _stt, _chat, text_to_speech = _make_handler(
        monkeypatch,
        transcripts=["こんにちは"],
        rounds=[[TextDelta("やあ。")]],
    )

    with caplog.at_level("INFO", logger="reachy_mini_conversation_ja.direct_cascade"):
        async with _running(handler):
            await _say_something(handler)
            await _wait_for(lambda: bool(text_to_speech.spoken))
            await _wait_for(lambda: any("Turn timing" in record.message for record in caplog.records))

    timing = next(record.getMessage() for record in caplog.records if "Turn timing" in record.message)
    for stage in ("silence", "stt", "answer", "speech"):
        assert stage in timing
    assert "to first audio" in timing


@pytest.mark.asyncio
async def test_a_new_question_drops_the_answer_to_the_last_one(monkeypatch: Any) -> None:
    """Queued speech from a past turn must not delay the answer the user is waiting for."""
    handler, _stt, _chat, text_to_speech = _make_handler(
        monkeypatch,
        transcripts=["ひとつめ", "ふたつめ"],
        rounds=[[TextDelta("古い答え。")], [TextDelta("新しい答え。")]],
    )
    flushes: list[bool] = []
    handler._clear_queue = lambda: flushes.append(True)

    async with _running(handler):
        await _say_something(handler)
        await _wait_for(lambda: text_to_speech.spoken == [("古い答え。", PROFILE_VOICE)])
        # Ask again while the first answer is still on its way to the speaker.
        handler._speech_queue.put_nowait(SpeechRequest("積み残し。", PROFILE_VOICE, None))
        await _say_something(handler)
        await _wait_for(lambda: ("新しい答え。", PROFILE_VOICE) in text_to_speech.spoken)

    assert flushes, "the player should have been flushed for the new turn"
    assert ("積み残し。", PROFILE_VOICE) not in text_to_speech.spoken
