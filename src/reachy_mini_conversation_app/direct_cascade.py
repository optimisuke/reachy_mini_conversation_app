"""Direct conversation backend: voice activity detection here, then STT, LLM and TTS over HTTP.

The deployed Hugging Face realtime backend transcribes only the languages its
speech-to-speech server covers, which leaves Japanese unusable. This backend keeps
the same handler contract but owns the turn loop itself, so every stage can point
at a provider that does support the language.
"""

import json
import time
import asyncio
import logging
from typing import Any, Final
from dataclasses import dataclass
from collections.abc import Coroutine

import numpy as np
from numpy.typing import NDArray
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
)

from reachy_mini_conversation_app.tools import core_tools
from reachy_mini_conversation_app.config import (
    config,
    get_default_voice,
    set_custom_profile,
    get_available_voices,
    resolve_available_voice,
    get_direct_backend_settings,
    get_speech_detection_settings,
)
from reachy_mini_conversation_app.prompts import (
    get_session_voice,
    get_session_instructions,
    get_session_greeting_prompt,
)
from reachy_mini_conversation_app.audio.pcm import StreamingResampler, resample
from reachy_mini_conversation_app.streaming import AdditionalOutputs, to_mono, audio_to_int16
from reachy_mini_conversation_app.voice_activity import UtteranceEvent, SpeechSegmenter
from reachy_mini_conversation_app.speech_services import (
    TextDelta,
    SentenceBuffer,
    SpeechServices,
    ToolCallRequest,
    build_speech_services,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, get_tool_specs
from reachy_mini_conversation_app.conversation_handler import QueueItem, AudioFrame, ConversationHandler
from reachy_mini_conversation_app.tools.background_tool_manager import (
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


logger = logging.getLogger(__name__)

_CAMERA_IMAGE_UNAVAILABLE: Final[str] = (
    "The camera took a picture, but this model cannot see images. Say you cannot see it; never guess."
)
_MAX_TOOL_ROUNDS: Final[int] = 4
_TOOL_RESULT_TIMEOUT_S: Final[float] = 30.0
_MAX_HISTORY_MESSAGES: Final[int] = 40
_PLAYBACK_CHUNK_SAMPLES: Final[int] = 640  # 40 ms at 16 kHz
# Stay this far ahead of the speaker so playback never starves while the pacing
# still keeps unplayed audio short enough for a barge-in to drop it.
_PLAYBACK_LEAD_S: Final[float] = 0.2
# The speaker drains and the room rings after the last frame leaves the queue.
_SPEAKING_TAIL_S: Final[float] = 0.4


@dataclass
class TurnTiming:
    """When each stage of one turn finished, so the wait can be attributed to a stage."""

    speech_ended_at: float
    utterance_at: float
    transcribed_at: float | None = None
    first_sentence_at: float | None = None
    first_audio_at: float | None = None


@dataclass(frozen=True)
class ToolOutcome:
    """What one finished tool gives the model: its output, plus any captured image."""

    payload: dict[str, Any]
    image_b64: str | None


class DirectCascadeHandler(ConversationHandler):
    """Conversation handler that segments speech locally and calls each speech stage directly."""

    SAMPLE_RATE = 16000

    def __init__(
        self,
        deps: ToolDependencies,
        instance_path: str | None = None,
        startup_voice: str | None = None,
    ) -> None:
        """Initialize the handler without contacting any service yet."""
        super().__init__()

        self.deps = deps
        self.instance_path = instance_path
        self.output_queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        self.tool_manager = BackgroundToolManager()

        self._voice_override = resolve_available_voice(startup_voice, source="persisted startup voice")
        self._services: SpeechServices | None = None
        self._segmenter: SpeechSegmenter | None = None
        self._messages: list[ChatCompletionMessageParam] = []
        self._session_open = asyncio.Event()
        self._stopped = asyncio.Event()
        self._turn_task: asyncio.Task[None] | None = None
        self._speech_task: asyncio.Task[None] | None = None
        self._speech_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._pending_tool_results: dict[str, asyncio.Future[ToolOutcome]] = {}
        self._llm_vision = False
        self._assistant_speaking = False
        self._playback_ends_at = 0.0
        self._turn_timing: TurnTiming | None = None

    def _is_connected(self) -> bool:
        """Return whether a session is serving audio."""
        return self._session_open.is_set()

    def _idle_behavior_ready(self) -> bool:
        """Hold idle behavior while a turn is being answered or spoken."""
        return self._turn_task is None and not self._assistant_speaking and self._speech_queue.empty()

    async def start_up(self) -> None:
        """Build the speech stages and serve the session until shutdown."""
        settings = get_direct_backend_settings()
        self._llm_vision = settings.llm_vision
        self._services = build_speech_services(settings)
        self._segmenter = SpeechSegmenter(self.SAMPLE_RATE, get_speech_detection_settings())
        self._reset_conversation()
        self._stopped.clear()
        self._session_open.set()
        self.tool_manager.start_up(tool_callbacks=[self._handle_tool_result])
        self._speech_task = asyncio.create_task(self._speech_loop(), name="direct-speech")
        logger.info(
            "Direct backend ready: stt=%s llm=%s tts=%s language=%r voice=%r vision=%s",
            settings.stt_model,
            settings.llm_model,
            settings.tts_model,
            settings.stt_language,
            self.get_current_voice(),
            settings.llm_vision,
        )
        try:
            await self._send_startup_greeting()
            await self._stopped.wait()
        finally:
            await self._close_session()

    async def shutdown(self) -> None:
        """Stop serving audio and release every task and client."""
        self._stopped.set()
        await self._close_session()

    async def _close_session(self) -> None:
        """Tear the session down; safe to call from both shutdown paths."""
        self._session_open.clear()
        await self._cancel_active_turn()
        await self._stop_speaking()
        await self.tool_manager.shutdown()

        for future in self._pending_tool_results.values():
            if not future.done():
                future.cancel()
        self._pending_tool_results.clear()

        services, self._services = self._services, None
        if services is not None:
            await services.aclose()

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ---- microphone side ----

    async def receive(self, frame: AudioFrame) -> None:
        """Segment microphone audio and start a turn once the user stops talking."""
        if not self._session_open.is_set() or self._segmenter is None:
            return

        sample_rate, audio = frame
        if audio.size == 0:
            return

        samples = audio_to_int16(to_mono(audio))
        if sample_rate != self.SAMPLE_RATE:
            samples = resample(samples, sample_rate, self.SAMPLE_RATE)

        for event in self._segmenter.push(samples):
            if event.speech_started:
                await self._on_speech_started(event)
            if event.utterance is not None:
                await self._on_utterance(event.utterance)

    async def _on_speech_started(self, event: UtteranceEvent) -> None:
        """Mark the user as talking, interrupting Reachy when it is mid-answer."""
        self._mark_activity("user_speech_started")
        self.deps.movement_manager.set_listening(True)
        if self._assistant_speaking:
            # The levels tell a real interruption from Reachy's own voice leaking in.
            logger.info(
                "User barge-in: dropping the response being spoken (level=%.4f threshold=%.4f)",
                event.level,
                event.threshold,
            )
            await self._cancel_active_turn()
            await self._stop_speaking()

    async def _on_utterance(self, utterance: NDArray[np.int16]) -> None:
        """Answer a completed utterance."""
        self._mark_activity("user_speech_stopped")
        self.deps.movement_manager.set_listening(False)
        logger.debug("Utterance captured: %.2fs", utterance.size / self.SAMPLE_RATE)
        now = time.monotonic()
        self._turn_timing = TurnTiming(
            speech_ended_at=now - get_speech_detection_settings().silence_end_s,
            utterance_at=now,
        )
        await self._cancel_active_turn()
        self._start_turn(self._run_turn(utterance), name="direct-turn")

    def _start_turn(self, turn: Coroutine[Any, Any, None], name: str) -> None:
        """Run ``turn`` as the one cancellable turn, freeing the slot when it ends."""
        task = asyncio.create_task(turn, name=name)
        task.add_done_callback(self._turn_finished)
        self._turn_task = task

    def _turn_finished(self, task: asyncio.Task[None]) -> None:
        """Free the turn slot once its task ends."""
        if self._turn_task is task:
            self._turn_task = None

    # ---- turn pipeline ----

    async def _run_turn(self, utterance: NDArray[np.int16]) -> None:
        """Transcribe an utterance, then answer it."""
        if self._services is None:
            return
        try:
            transcript = await self._services.speech_to_text.transcribe(utterance, self.SAMPLE_RATE)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            await self._emit_error(f"transcription failed: {e}")
            return

        transcript = transcript.strip()
        self._mark_activity("user_transcription_completed")
        if not transcript:
            logger.debug("Ignoring empty user transcript")
            return

        if self._turn_timing is not None:
            self._turn_timing.transcribed_at = time.monotonic()
        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
        self._emit_transcript("user", transcript, True)

        self._trim_history()
        self._messages.append(ChatCompletionUserMessageParam(role="user", content=transcript))
        await self._generate_response()

    async def _generate_response(self) -> None:
        """Stream the model's answer, running any tools it asks for before answering again."""
        services = self._services
        if services is None:
            return
        history_mark = len(self._messages)
        try:
            for _ in range(_MAX_TOOL_ROUNDS):
                text, tool_calls = await self._stream_one_response(services)
                if not tool_calls:
                    return
                await self._run_tool_calls(text, tool_calls)
            logger.warning("Giving up after %d tool rounds without a spoken answer", _MAX_TOOL_ROUNDS)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Roll this turn back: a history the model rejects would fail every
            # later turn too, leaving Reachy silent until the app restarts.
            del self._messages[history_mark:]
            logger.error("Response generation failed: %s", e)
            await self._emit_error(f"response failed: {e}")

    async def _stream_one_response(self, services: SpeechServices) -> tuple[str, list[ToolCallRequest]]:
        """Speak one streamed model response and return its text and requested tool calls."""
        self._mark_activity("response_created")
        voice = self.get_current_voice()
        sentences = SentenceBuffer()
        spoken_text: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        async for event in services.chat_model.stream(self._messages, get_tool_specs()):
            if isinstance(event, TextDelta):
                spoken_text.append(event.text)
                for sentence in sentences.push(event.text):
                    self._mark_first_sentence()
                    await self._speech_queue.put((sentence, voice))
                continue
            tool_calls.append(event)

        remainder = sentences.flush()
        if remainder:
            self._mark_first_sentence()
            await self._speech_queue.put((remainder, voice))

        text = "".join(spoken_text).strip()
        if not tool_calls:
            self._messages.append(self._assistant_message(text, tool_calls))
        if text:
            self._mark_activity("assistant_transcript_done")
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": text}))
            self._emit_transcript("assistant", text, True)
        return text, tool_calls

    @staticmethod
    def _assistant_message(text: str, tool_calls: list[ToolCallRequest]) -> ChatCompletionAssistantMessageParam:
        """Build the assistant history entry for one streamed response."""
        message = ChatCompletionAssistantMessageParam(role="assistant", content=text)
        if tool_calls:
            message["tool_calls"] = [
                ChatCompletionMessageFunctionToolCallParam(
                    id=call.call_id,
                    type="function",
                    function={"name": call.name, "arguments": call.arguments},
                )
                for call in tool_calls
            ]
        return message

    async def _run_tool_calls(self, text: str, tool_calls: list[ToolCallRequest]) -> None:
        """Run every requested tool, then record the request and its replies together."""
        loop = asyncio.get_running_loop()
        results: dict[str, asyncio.Future[ToolOutcome]] = {}
        try:
            for call in tool_calls:
                self._mark_activity("tool_call_received")
                logger.info(
                    "Tool call received — tool_name=%r, call_id=%s, args=%s", call.name, call.call_id, call.arguments
                )
                future: asyncio.Future[ToolOutcome] = loop.create_future()
                self._pending_tool_results[call.call_id] = future
                results[call.call_id] = future
                background_tool = await self.tool_manager.start_tool(
                    call_id=call.call_id,
                    tool_call_routine=ToolCallRoutine(
                        tool_name=call.name,
                        args_json_str=call.arguments,
                        deps=self.deps,
                    ),
                    is_idle_tool_call=False,
                )
                await self.output_queue.put(
                    AdditionalOutputs(
                        {
                            "role": "assistant",
                            "content": (
                                f"🛠️ Used tool {call.name} with args {call.arguments}. "
                                f"The tool is now running. Tool ID: {background_tool.tool_id}"
                            ),
                        },
                    ),
                )

            try:
                await asyncio.wait_for(asyncio.gather(*results.values()), timeout=_TOOL_RESULT_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning("Some tools did not finish within %.0fs", _TOOL_RESULT_TIMEOUT_S)

            # No await between these appends: the model rejects a tool request
            # whose replies are missing, so the pair has to be uninterruptible.
            self._messages.append(self._assistant_message(text, tool_calls))
            images: list[str] = []
            for call_id, future in results.items():
                outcome = (
                    future.result() if future.done() else ToolOutcome({"error": "tool did not finish in time"}, None)
                )
                payload = dict(outcome.payload)
                if outcome.image_b64 is not None:
                    if self._llm_vision:
                        images.append(outcome.image_b64)
                    else:
                        payload["note"] = _CAMERA_IMAGE_UNAVAILABLE
                self._messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        tool_call_id=call_id,
                        content=json.dumps(payload, ensure_ascii=False),
                    ),
                )
            for image_b64 in images:
                self._messages.append(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=[
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                            },
                        ],
                    ),
                )
        finally:
            for call_id in results:
                self._pending_tool_results.pop(call_id, None)

    async def _handle_tool_result(self, completed_tool: ToolNotification) -> None:
        """Surface a finished tool and hand its output to the waiting turn."""
        if completed_tool.error is not None:
            logger.error(
                "Tool '%s' (id=%s) failed: %s", completed_tool.tool_name, completed_tool.id, completed_tool.error
            )
            output: dict[str, Any] = {"error": completed_tool.error}
        elif isinstance(completed_tool.result, dict):
            logger.info("Tool '%s' (id=%s) executed successfully.", completed_tool.tool_name, completed_tool.id)
            output = self._sanitize_tool_result_for_model(completed_tool.tool_name, completed_tool.result)
        elif completed_tool.result is not None:
            output = {"result": completed_tool.result}
        else:
            logger.warning(
                "Tool '%s' (id=%s) returned no result and no error", completed_tool.tool_name, completed_tool.id
            )
            output = {"error": "No result returned from tool execution"}

        if not completed_tool.is_idle_tool_call:
            self._mark_activity("tool_result_ready")
        await self.output_queue.put(
            AdditionalOutputs({"role": "assistant", "content": json.dumps(output, ensure_ascii=False)}),
        )

        future = self._pending_tool_results.get(completed_tool.id)
        if future is not None and not future.done():
            image_b64 = completed_tool.result.get("b64_im") if isinstance(completed_tool.result, dict) else None
            future.set_result(ToolOutcome(output, image_b64 if isinstance(image_b64, str) else None))

    def _mark_first_sentence(self) -> None:
        """Record when the first speakable piece of this turn's answer was ready."""
        timing = self._turn_timing
        if timing is not None and timing.first_sentence_at is None:
            timing.first_sentence_at = time.monotonic()

    def _report_turn_timing(self) -> None:
        """Log where the wait between the user finishing and Reachy speaking went."""
        timing = self._turn_timing
        if timing is None or timing.first_audio_at is None:
            return
        self._turn_timing = None

        stages = [("silence", timing.speech_ended_at, timing.utterance_at)]
        if timing.transcribed_at is not None:
            stages.append(("stt", timing.utterance_at, timing.transcribed_at))
            if timing.first_sentence_at is not None:
                stages.append(("answer", timing.transcribed_at, timing.first_sentence_at))
                stages.append(("speech", timing.first_sentence_at, timing.first_audio_at))
        logger.info(
            "Turn timing: %s = %.0f ms to first audio",
            " + ".join(f"{name} {(end - start) * 1000:.0f}" for name, start, end in stages),
            (timing.first_audio_at - timing.speech_ended_at) * 1000,
        )

    def _trim_history(self) -> None:
        """Bound the history, keeping the system message and whole recent turns."""
        if len(self._messages) <= _MAX_HISTORY_MESSAGES:
            return
        system_message, recent = self._messages[0], self._messages[-(_MAX_HISTORY_MESSAGES - 1) :]
        while recent and recent[0]["role"] == "tool":
            recent = recent[1:]
        self._messages = [system_message, *recent]

    async def _emit_error(self, message: str) -> None:
        """Show a backend failure in the transcript, as the realtime backend does."""
        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[error] {message}"}))

    async def _cancel_active_turn(self) -> None:
        """Cancel the turn in flight, if any."""
        task, self._turn_task = self._turn_task, None
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # ---- speaker side ----

    async def _speech_loop(self) -> None:
        """Speak queued sentences one at a time, starting each as it is synthesized."""
        while True:
            sentence, voice = await self._speech_queue.get()
            services = self._services
            if services is None:
                continue
            try:
                await self._speak(services, sentence, voice)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Speech synthesis failed: %s", e)

    async def _speak(self, services: SpeechServices, sentence: str, voice: str) -> None:
        """Stream one sentence to the player, then wait out the audio already queued."""
        resampler = StreamingResampler(services.text_to_speech.sample_rate, self.SAMPLE_RATE)
        self._set_speaking(True)
        spoken = False
        async for block in services.text_to_speech.stream(sentence, voice):
            spoken = await self._queue_audio(resampler.process(block)) or spoken
        if not spoken:
            return

        remaining = self._playback_ends_at - time.monotonic()
        if not self._speech_queue.empty():
            # Stay a little ahead of the speaker while more sentences are waiting.
            await asyncio.sleep(max(0.0, remaining - _PLAYBACK_LEAD_S))
            return
        await asyncio.sleep(max(0.0, remaining) + _SPEAKING_TAIL_S)
        if self._speech_queue.empty():
            self._set_speaking(False)

    async def _queue_audio(self, pcm: NDArray[np.int16]) -> bool:
        """Hand one block of audio to the player and extend the expected playback end."""
        if pcm.size == 0:
            return False
        timing = self._turn_timing
        if timing is not None and timing.first_audio_at is None:
            timing.first_audio_at = time.monotonic()
            self._report_turn_timing()
        for start in range(0, pcm.size, _PLAYBACK_CHUNK_SAMPLES):
            self._mark_activity("assistant_audio_delta")
            await self.output_queue.put(
                (self.SAMPLE_RATE, pcm[start : start + _PLAYBACK_CHUNK_SAMPLES].reshape(1, -1))
            )
        self._playback_ends_at = max(self._playback_ends_at, time.monotonic()) + pcm.size / self.SAMPLE_RATE
        return True

    def _set_speaking(self, speaking: bool) -> None:
        """Mirror playback state into head motion and the barge-in threshold."""
        self._assistant_speaking = speaking
        self.deps.movement_manager.set_speaking(speaking)
        if self._segmenter is not None:
            self._segmenter.set_assistant_speaking(speaking)

    async def _stop_speaking(self) -> None:
        """Drop queued and playing speech after a barge-in or a personality change."""
        await self._cancel_speech_loop()
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._playback_ends_at = 0.0
        if self._clear_queue is not None:
            self._clear_queue()
        self._set_speaking(False)
        if self._session_open.is_set():
            self._speech_task = asyncio.create_task(self._speech_loop(), name="direct-speech")

    async def _cancel_speech_loop(self) -> None:
        """Stop the synthesis worker, if it is running."""
        task, self._speech_task = self._speech_task, None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def say(self, text: str) -> None:
        """Speak ``text`` verbatim; this backend synthesizes it instead of prompting a model."""
        text = (text or "").strip()
        if not text:
            raise ValueError("say: empty text")
        if not self._is_connected():
            raise RuntimeError("say: no active session")
        self._mark_activity("say")
        self._messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=text))
        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": text}))
        self._emit_transcript("assistant", text, True)
        await self._speech_queue.put((text, self.get_current_voice()))

    async def _send_startup_greeting(self) -> None:
        """Let the model open the conversation, as the realtime backend does."""
        greeting_prompt = get_session_greeting_prompt().strip()
        if not greeting_prompt:
            return
        self._messages.append(ChatCompletionUserMessageParam(role="user", content=greeting_prompt))
        self._mark_activity("startup_greeting_prompt")
        self._start_turn(self._generate_response(), name="direct-greeting")

    # ---- personality and voices ----

    def _reset_conversation(self) -> None:
        """Start a fresh history from the active profile's instructions."""
        self._messages = [
            ChatCompletionSystemMessageParam(role="system", content=get_session_instructions(self.instance_path)),
        ]

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a personality profile and restart the conversation from its instructions."""
        previous_profile = config.REACHY_MINI_CUSTOM_PROFILE
        set_custom_profile(profile)
        try:
            get_session_instructions(self.instance_path)
            core_tools.initialize_tools(force=True)
        except Exception as exc:
            set_custom_profile(previous_profile)
            logger.error("Failed to resolve personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"

        await self._cancel_active_turn()
        await self._stop_speaking()
        self._reset_conversation()
        logger.info("Applied personality: %s", profile or "default")
        return "Applied personality."

    async def get_available_voices(self) -> list[str]:
        """Return the voice catalog shared with the Hugging Face backend."""
        return get_available_voices()

    def get_current_voice(self) -> str:
        """Return the voice currently selected for this handler."""
        default_voice = get_default_voice()
        voice = self._voice_override or get_session_voice(default=default_voice)
        return resolve_available_voice(voice, source="session voice", fallback=default_voice) or default_voice

    async def change_voice(self, voice: str) -> str:
        """Switch the voice used for the next synthesized sentence."""
        default_voice = get_default_voice()
        resolved_voice = (
            resolve_available_voice(voice, source="requested voice", fallback=default_voice) or default_voice
        )
        self._voice_override = resolved_voice
        return f"Voice changed to {resolved_voice}."
