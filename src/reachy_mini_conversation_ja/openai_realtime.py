"""Realtime backend: one round trip instead of transcribe, answer, then synthesize.

The direct backend calls three services in sequence, so the user waits for all
three. OpenAI's realtime endpoint takes audio and returns audio over one socket
and starts answering while the person is still talking.

Turn detection stays here rather than on the server. The server's own detector
requires streaming audio continuously, and audio input is billed by the token,
so an idle robot would pay to be listened to. Segmenting locally means nothing is
sent until someone speaks, and it keeps the thresholds tuned on this microphone.
"""

import time
import base64
import asyncio
import logging
from typing import Any, Final
from collections import deque

import numpy as np
from openai import AsyncOpenAI
from numpy.typing import NDArray
from openai.types.realtime import (
    AudioTranscriptionParam,
    RealtimeAudioConfigParam,
    RealtimeAudioConfigInputParam,
    RealtimeAudioConfigOutputParam,
    RealtimeSessionCreateRequestParam,
)

from reachy_mini_conversation_ja.config import (
    config,
    get_direct_backend_settings,
    get_speech_detection_settings,
)
from reachy_mini_conversation_ja.prompts import get_session_instructions
from reachy_mini_conversation_ja.audio.pcm import StreamingResampler, resample
from reachy_mini_conversation_ja.streaming import to_mono, audio_to_int16
from reachy_mini_conversation_ja.voice_activity import SpeechSegmenter
from reachy_mini_conversation_ja.tools.core_tools import ToolSpec, ToolDependencies
from reachy_mini_conversation_ja.huggingface_realtime import (
    HuggingFaceRealtimeHandler,
    to_realtime_tools_config,
)


logger = logging.getLogger(__name__)

# The speaker drains and the room rings after the last block is queued.
_SPEAKING_TAIL_S: Final[float] = 0.4

# The voice catalog the UI and profiles use comes from the Hugging Face backend's
# speakers, so map it onto the realtime voices. REALTIME_VOICE overrides the map.
_REALTIME_VOICE_BY_CATALOG_NAME: Final[dict[str, str]] = {
    "Aiden": "ash",
    "Ryan": "verse",
    "Dylan": "echo",
    "Eric": "alloy",
    "Ono_Anna": "marin",
    "Serena": "shimmer",
    "Sohee": "coral",
    "Uncle_Fu": "cedar",
    "Vivian": "sage",
}


class OpenAIRealtimeHandler(HuggingFaceRealtimeHandler):
    """Realtime handler pointed at OpenAI, with turn detection kept on this side."""

    SAMPLE_RATE = 16000

    def __init__(
        self,
        deps: ToolDependencies,
        instance_path: str | None = None,
        startup_voice: str | None = None,
    ) -> None:
        """Initialize the handler and the local segmenter."""
        super().__init__(deps, instance_path=instance_path, startup_voice=startup_voice)
        settings = get_direct_backend_settings()
        self._realtime_rate = settings.realtime_rate
        self._realtime_model = settings.realtime_model
        self._realtime_voice_override = settings.realtime_voice
        self._segmenter = SpeechSegmenter(self.SAMPLE_RATE, get_speech_detection_settings())
        self._output_resampler = StreamingResampler(self._realtime_rate, self.SAMPLE_RATE)
        self._preroll: deque[NDArray[np.int16]] = deque(maxlen=_preroll_frames())
        self._streaming_speech = False
        self._playback_ends_at = 0.0
        self._speaking_tail: asyncio.Task[None] | None = None

    def _set_speaking(self, speaking: bool) -> None:
        """Raise the microphone's bar while Reachy talks, so it does not answer itself.

        Echo cancellation leaves a distorted residual of Reachy's own voice, which
        transcribes as noise; answering that starts a loop that never ends. The server
        finishing its send is not the speaker finishing its playback, so the bar comes
        back down only once the audio already queued has actually been heard.
        """
        super()._set_speaking(speaking)
        if speaking:
            self._cancel_speaking_tail()
            self._segmenter.set_assistant_speaking(True)
            return
        self._speaking_tail = asyncio.create_task(self._lower_the_bar_after_playback())

    async def _lower_the_bar_after_playback(self) -> None:
        """Keep the raised trigger until the queued audio has played, and a moment after."""
        try:
            while True:
                remaining = self._playback_ends_at - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(remaining)
            await asyncio.sleep(_SPEAKING_TAIL_S)
        except asyncio.CancelledError:
            return
        self._segmenter.set_assistant_speaking(False)

    def _cancel_speaking_tail(self) -> None:
        """Stop waiting for playback to finish, because Reachy started talking again."""
        tail, self._speaking_tail = self._speaking_tail, None
        if tail is not None and not tail.done():
            tail.cancel()

    def _connect_kwargs(self) -> dict[str, Any]:
        """Open the socket for the configured realtime model."""
        return {"model": self._realtime_model}

    def _decode_output_audio(self, delta: str) -> NDArray[np.int16]:
        """Bring the reply down to the rate the speaker runs at, tracking playback."""
        block = np.frombuffer(base64.b64decode(delta), dtype=np.int16)
        resampled = self._output_resampler.process(block)
        now = time.monotonic()
        self._playback_ends_at = max(self._playback_ends_at, now) + resampled.size / self.SAMPLE_RATE
        return resampled

    async def _build_realtime_client(self) -> AsyncOpenAI:
        """Build a client for OpenAI itself, with no session allocator in between."""
        settings = get_direct_backend_settings()
        api_key = settings.llm_api_key or settings.stt_api_key
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY to use the realtime backend.")
        logger.info("Using the OpenAI realtime endpoint with model %s", self._realtime_model)
        return AsyncOpenAI(api_key=api_key)

    def _realtime_voice(self) -> str:
        """Return the provider voice for the catalog voice in force."""
        if self._realtime_voice_override:
            return self._realtime_voice_override
        return _REALTIME_VOICE_BY_CATALOG_NAME.get(self.get_current_voice(), "marin")

    def _get_session_config(self, tool_specs: list[ToolSpec]) -> RealtimeSessionCreateRequestParam:
        """Return the session config, with the server's turn detection switched off."""
        return RealtimeSessionCreateRequestParam(
            type="realtime",
            instructions=get_session_instructions(self.instance_path),
            audio=RealtimeAudioConfigParam(
                input=RealtimeAudioConfigInputParam(
                    format={"type": "audio/pcm", "rate": self._realtime_rate},  # type: ignore[typeddict-item]
                    transcription=AudioTranscriptionParam(
                        model="gpt-4o-transcribe",
                        language=config.REALTIME_TRANSCRIPTION_LANGUAGE,
                    ),
                    # Committing the turn here is what keeps a silent room free.
                    turn_detection=None,
                ),
                output=RealtimeAudioConfigOutputParam(
                    format={"type": "audio/pcm", "rate": self._realtime_rate},  # type: ignore[typeddict-item]
                    voice=self._realtime_voice(),
                ),
            ),
            tools=to_realtime_tools_config(tool_specs),
            tool_choice="auto",
        )

    async def receive(self, frame: tuple[int, NDArray[np.int16]]) -> None:
        """Stream microphone audio while someone is speaking, and close the turn after."""
        if not self.connection:
            return
        sample_rate, audio = frame
        if audio.size == 0:
            return

        samples = audio_to_int16(to_mono(audio))
        if sample_rate != self.SAMPLE_RATE:
            samples = resample(samples, sample_rate, self.SAMPLE_RATE)

        events = self._segmenter.push(samples)
        if self._streaming_speech:
            await self._send_audio(samples)
        else:
            self._preroll.append(samples)

        for event in events:
            if event.speech_started:
                await self._open_turn(event.level, event.threshold)
            if event.utterance is not None and not event.provisional:
                await self._close_turn()

    async def _open_turn(self, level: float, threshold: float) -> None:
        """Start streaming, sending the audio buffered before the onset was certain."""
        self._mark_activity("user_speech_started")
        self.deps.movement_manager.set_listening(True)
        if not self._response_done_event.is_set():
            logger.info(
                "User barge-in: cancelling the response being spoken (level=%.4f threshold=%.4f)",
                level,
                threshold,
            )
            await self._cancel_response()
        self._streaming_speech = True
        for buffered in list(self._preroll):
            await self._send_audio(buffered)
        self._preroll.clear()

    async def _close_turn(self) -> None:
        """Tell the server the utterance is over and ask for the answer."""
        self._streaming_speech = False
        self._preroll.clear()
        self.deps.movement_manager.set_listening(False)
        self._mark_activity("user_speech_stopped")
        if not self.connection:
            return
        try:
            await self.connection.input_audio_buffer.commit()
        except Exception as e:
            logger.debug("Dropping turn: input buffer could not be committed (%s)", e)
            return
        # The transcript can land after the audio here, so time the wait from the commit.
        self._turn_user_done_at = time.perf_counter()
        self._turn_response_created_at = None
        self._turn_first_audio_at = None
        await self._safe_response_create()

    async def _send_audio(self, samples: NDArray[np.int16]) -> None:
        """Forward one block of microphone audio at the rate the session expects."""
        if not self.connection:
            return
        block = resample(samples, self.SAMPLE_RATE, self._realtime_rate)
        try:
            await self.connection.input_audio_buffer.append(
                audio=base64.b64encode(block.tobytes()).decode("utf-8"),
            )
        except Exception as e:
            logger.debug("Dropping audio frame: connection not ready (%s)", e)

    async def _cancel_response(self) -> None:
        """Stop the reply in progress after a barge-in."""
        if self._clear_queue is not None:
            self._clear_queue()
        if not self.connection:
            return
        try:
            await self.connection.response.cancel()
        except Exception as e:
            logger.debug("Response cancel ignored: %s", e)


def _preroll_frames() -> int:
    """Return how many microphone blocks to keep before an onset is confirmed."""
    settings = get_speech_detection_settings()
    # A microphone block is at most one analysis window long, so counting windows
    # over the pre-roll and the onset keeps at least that much audio.
    return max(1, round((settings.preroll_s + settings.speech_start_s) / settings.window_s))
