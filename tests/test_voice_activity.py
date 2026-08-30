"""Tests for the direct backend's speech segmentation."""

import numpy as np

from reachy_mini_conversation_app.config import SpeechDetectionSettings
from reachy_mini_conversation_app.voice_activity import SpeechSegmenter


SAMPLE_RATE = 16000


def _tone(level: float, duration_s: float) -> np.ndarray:
    """Return audio whose RMS is ``level`` on the 0..1 scale."""
    sample_count = int(SAMPLE_RATE * duration_s)
    amplitude = int(level * 32768)
    return (np.arange(sample_count) % 2 * 2 - 1).astype(np.int16) * np.int16(amplitude)


def _silence(duration_s: float) -> np.ndarray:
    """Return digital silence."""
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.int16)


def _segmenter(**overrides: float) -> SpeechSegmenter:
    """Build a segmenter with the shipped thresholds."""
    return SpeechSegmenter(SAMPLE_RATE, SpeechDetectionSettings(**overrides))


def test_silence_never_opens_a_turn() -> None:
    """A quiet room should produce no events."""
    segmenter = _segmenter()

    assert segmenter.push(_silence(3.0)) == []


def test_speech_then_silence_yields_one_utterance() -> None:
    """Talking and stopping should report an onset and then the captured audio."""
    segmenter = _segmenter()
    segmenter.push(_silence(1.0))

    onset_events = segmenter.push(_tone(0.05, 0.5))
    end_events = segmenter.push(_silence(1.0))

    assert [event.speech_started for event in onset_events] == [True]
    # A short pause offers a guess first, then the full silence confirms the turn.
    assert [event.provisional for event in end_events] == [True, False]
    utterance = end_events[-1].utterance
    assert utterance is not None
    # The captured audio covers the speech, the pre-roll ahead of it and the closing silence.
    assert utterance.size / SAMPLE_RATE > 0.5


def test_utterances_shorter_than_the_minimum_are_dropped() -> None:
    """A cough should reach neither the early nor the final transcription."""
    segmenter = _segmenter(min_utterance_s=0.5)
    segmenter.push(_silence(1.0))
    segmenter.push(_tone(0.05, 0.2))

    assert segmenter.push(_silence(1.0)) == []


def test_a_pause_offers_the_speech_so_far_for_transcription() -> None:
    """The guess carries the audio heard so far, so its transcription can be reused."""
    segmenter = _segmenter()
    segmenter.push(_silence(1.0))
    segmenter.push(_tone(0.05, 0.6))

    events = segmenter.push(_silence(0.25))

    assert len(events) == 1
    guess = events[0]
    assert guess.provisional and guess.utterance is not None
    # The windows that proved the onset are not counted again.
    settings = SpeechDetectionSettings()
    assert guess.voiced_windows == round((0.6 - settings.speech_start_s) / settings.window_s)
    # Long enough to hold the speech plus its pre-roll.
    assert guess.utterance.size / SAMPLE_RATE > 0.6


def test_the_guess_is_skipped_when_it_is_turned_off() -> None:
    """Setting the threshold to zero transcribes once, at the end of the turn."""
    segmenter = _segmenter(speculative_silence_s=0.0)
    segmenter.push(_silence(1.0))
    segmenter.push(_tone(0.05, 0.6))

    events = segmenter.push(_silence(1.0))

    assert [event.provisional for event in events] == [False]


def test_long_speech_is_cut_at_the_maximum_length() -> None:
    """A monologue should be closed at the limit instead of buffering forever."""
    segmenter = _segmenter(max_utterance_s=1.0)
    segmenter.push(_silence(0.5))

    events = segmenter.push(_tone(0.05, 3.0))

    utterances = [event.utterance for event in events if event.utterance is not None]
    assert utterances and utterances[0].size / SAMPLE_RATE <= 1.1


def test_reachy_talking_raises_the_trigger() -> None:
    """While Reachy speaks, only a clearly louder voice should interrupt it."""
    segmenter = _segmenter()
    segmenter.push(_silence(1.0))
    segmenter.set_assistant_speaking(True)

    assert segmenter.push(_tone(0.008, 0.5)) == []
    assert [event.speech_started for event in segmenter.push(_tone(0.05, 0.5))] == [True]


def test_the_noise_floor_follows_a_noisy_room() -> None:
    """Background hiss should raise the bar for what counts as speech."""
    quiet_room, noisy_room = _segmenter(), _segmenter()
    quiet_room.push(_silence(3.0))
    # Hiss below the absolute minimum never opens a turn, but it is learned.
    noisy_room.push(_tone(0.0025, 10.0))

    assert [event.speech_started for event in quiet_room.push(_tone(0.007, 0.5))] == [True]
    assert noisy_room.push(_tone(0.007, 0.5)) == []


def test_a_room_that_never_goes_quiet_stops_triggering() -> None:
    """Noise loud enough to open a turn should be relearned when the turn never ends."""
    segmenter = _segmenter(max_utterance_s=1.0)

    first_events = segmenter.push(_tone(0.02, 3.0))
    settled_events = segmenter.push(_tone(0.02, 3.0))

    assert any(event.speech_started for event in first_events)
    assert settled_events == []
