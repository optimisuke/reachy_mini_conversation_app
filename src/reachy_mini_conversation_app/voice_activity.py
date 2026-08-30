"""Energy-based speech segmentation for backends that run their own VAD.

The microphone array already applies echo cancellation and noise suppression, so
tracking a noise floor and triggering on a multiple of it is enough to cut the
stream into utterances without an extra model. ``SpeechSegmenter`` owns that
decision alone, so a learned detector can replace it behind the same API.
"""

import logging
from typing import Final
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.config import SpeechDetectionSettings


logger = logging.getLogger(__name__)

_INT16_SCALE: Final[float] = 32768.0
# Keeps a silent room from driving the threshold to zero, and a loud one from
# raising it so far that speech can no longer trigger.
_NOISE_FLOOR_MIN: Final[float] = 1e-4
_NOISE_FLOOR_MAX: Final[float] = 0.05
_NOISE_FLOOR_FALL: Final[float] = 0.3
_NOISE_FLOOR_RISE: Final[float] = 0.01
# Enough tail for the last word not to sound clipped, without paying to upload
# the silence the end-of-turn decision needed.
_TRAILING_SILENCE_KEEP_S: Final[float] = 0.15


@dataclass(frozen=True)
class UtteranceEvent:
    """One segmentation transition: a speech onset, or a completed utterance."""

    speech_started: bool = False
    utterance: NDArray[np.int16] | None = None


def _rms(samples: NDArray[np.int16]) -> float:
    """Return the level of ``samples`` on the 0..1 scale."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float32) / _INT16_SCALE))))


class SpeechSegmenter:
    """Cut a mono int16 stream into utterances using an adaptive noise floor."""

    def __init__(self, sample_rate: int, settings: SpeechDetectionSettings) -> None:
        """Size the analysis windows and counters for ``sample_rate``."""
        self._sample_rate = sample_rate
        self._settings = settings
        self._window_size = max(1, round(sample_rate * settings.window_s))
        self._preroll: deque[NDArray[np.int16]] = deque(maxlen=self._windows_for(settings.preroll_s))
        self._pending = np.zeros(0, dtype=np.int16)
        self._utterance: list[NDArray[np.int16]] = []
        self._noise_floor = settings.min_level
        self._voiced_windows = 0
        self._voiced_in_speech = 0
        self._silent_windows = 0
        self._in_speech = False
        self._assistant_speaking = False

    def _windows_for(self, duration_s: float) -> int:
        """Return how many analysis windows cover ``duration_s``."""
        return max(1, round(duration_s / self._settings.window_s))

    def set_assistant_speaking(self, speaking: bool) -> None:
        """Raise the trigger while Reachy talks so only a deliberate barge-in opens a turn."""
        self._assistant_speaking = speaking

    def reset(self) -> None:
        """Drop the current utterance and counters, keeping the learned noise floor."""
        self._preroll.clear()
        self._utterance = []
        self._voiced_windows = 0
        self._voiced_in_speech = 0
        self._silent_windows = 0
        self._in_speech = False

    def push(self, samples: NDArray[np.int16]) -> list[UtteranceEvent]:
        """Feed mono int16 samples and return the transitions they completed."""
        buffered = np.concatenate((self._pending, samples)) if self._pending.size else samples
        window_count = buffered.size // self._window_size
        # Claim the remainder up front: closing an utterance resets the state
        # mid-loop, and the windows below must keep coming from this buffer.
        self._pending = buffered[window_count * self._window_size :].copy()

        events: list[UtteranceEvent] = []
        for index in range(window_count):
            window = buffered[index * self._window_size : (index + 1) * self._window_size]
            events.extend(self._consume_window(window))
        return events

    def _trigger(self) -> tuple[float, float, float]:
        """Return the (minimum level, noise floor ratio, onset duration) in force right now."""
        if self._assistant_speaking:
            return (
                self._settings.barge_in_min_level,
                self._settings.barge_in_ratio,
                self._settings.barge_in_start_s,
            )
        return self._settings.min_level, self._settings.speech_start_ratio, self._settings.speech_start_s

    def _consume_window(self, window: NDArray[np.int16]) -> list[UtteranceEvent]:
        """Classify one analysis window and advance the speech state machine."""
        min_level, ratio, onset_s = self._trigger()
        level = _rms(window)
        voiced = level > max(min_level, self._noise_floor * ratio)

        if not self._in_speech:
            self._preroll.append(window)
            if not voiced:
                # Only background windows teach the floor; a candidate onset must
                # not raise the very threshold it is being measured against.
                self._track_noise_floor(level)
                self._voiced_windows = 0
                return []
            self._voiced_windows += 1
            if self._voiced_windows < self._windows_for(onset_s):
                return []
            self._in_speech = True
            self._voiced_windows = 0
            self._voiced_in_speech = 0
            self._silent_windows = 0
            self._utterance = list(self._preroll)
            self._preroll.clear()
            return [UtteranceEvent(speech_started=True)]

        self._utterance.append(window)
        if voiced:
            self._voiced_in_speech += 1
            self._silent_windows = 0
        else:
            self._silent_windows += 1
            if self._silent_windows >= self._windows_for(self._settings.silence_end_s):
                return self._close_utterance()

        if len(self._utterance) * self._settings.window_s >= self._settings.max_utterance_s:
            logger.debug("Cutting utterance at the %.1fs limit", self._settings.max_utterance_s)
            return self._close_utterance(relearn_noise_floor=True)
        return []

    def _close_utterance(self, *, relearn_noise_floor: bool = False) -> list[UtteranceEvent]:
        """End the current utterance, dropping it when it holds too little speech."""
        windows = self._utterance
        surplus_silence = self._silent_windows - self._windows_for(_TRAILING_SILENCE_KEEP_S)
        if surplus_silence > 0:
            windows = windows[:-surplus_silence]
        utterance = np.concatenate(windows) if windows else np.zeros(0, dtype=np.int16)
        voiced_s = self._voiced_in_speech * self._settings.window_s
        self.reset()

        if relearn_noise_floor:
            # Audio that never stops is the room, not a person: take its level as
            # the new floor. Real speech that hits the limit decays back in silence.
            self._noise_floor = min(max(self._noise_floor, _rms(utterance)), _NOISE_FLOOR_MAX)

        if voiced_s < self._settings.min_utterance_s:
            logger.debug("Dropping utterance holding only %.2fs of speech", voiced_s)
            return []
        return [UtteranceEvent(utterance=utterance)]

    def _track_noise_floor(self, level: float) -> None:
        """Follow the room level, dropping fast and rising slowly."""
        weight = _NOISE_FLOOR_FALL if level < self._noise_floor else _NOISE_FLOOR_RISE
        self._noise_floor += weight * (level - self._noise_floor)
        self._noise_floor = min(max(self._noise_floor, _NOISE_FLOOR_MIN), _NOISE_FLOOR_MAX)
