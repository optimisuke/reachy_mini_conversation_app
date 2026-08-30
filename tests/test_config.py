"""Tests for configuration helpers."""

import pytest

from reachy_mini_conversation_app import config


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("45", 45.0),
        ("", config.DEFAULT_APP_TIMEOUT_MINUTES),  # unset/blank falls back to the default
        ("soon", config.DEFAULT_APP_TIMEOUT_MINUTES),  # unparseable falls back to the default
        ("0", None),  # non-positive disables the watchdog
        ("-1", None),
    ],
)
def test_resolve_app_timeout_minutes(monkeypatch, raw_value, expected) -> None:
    """The env timeout parses to minutes, falls back to the default, or disables on non-positive."""
    monkeypatch.setenv(config.APP_TIMEOUT_MINUTES_ENV, raw_value)

    assert config.resolve_app_timeout_minutes() == expected


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("direct", config.DIRECT_BACKEND),
        ("DIRECT", config.DIRECT_BACKEND),
        ("huggingface", config.HF_BACKEND),
        ("", config.HF_BACKEND),  # unset keeps the realtime backend
        ("whisper", config.HF_BACKEND),  # unknown values must not change the backend
    ],
)
def test_get_conversation_backend(monkeypatch, raw_value, expected) -> None:
    """The backend selector normalizes case and refuses unknown values."""
    monkeypatch.setenv(config.CONVERSATION_BACKEND_ENV, raw_value)

    assert config.get_conversation_backend() == expected


def test_direct_backend_settings_default_to_the_verified_stack(monkeypatch) -> None:
    """Unset direct-backend variables fall back to the models this app was tested with."""
    for name in (
        "DIRECT_STT_MODEL",
        "DIRECT_LLM_MODEL",
        "DIRECT_LLM_BASE_URL",
        "DIRECT_TTS_MODEL",
        "DIRECT_TTS_VOICE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("REALTIME_TRANSCRIPTION_LANGUAGE", "ja")
    config.refresh_runtime_config_from_env()

    settings = config.get_direct_backend_settings()

    assert settings.stt_model == config.DIRECT_DEFAULTS.stt_model
    assert settings.llm_base_url == config.DIRECT_DEFAULTS.llm_base_url
    assert settings.stt_language == "ja"
    assert settings.tts_voice is None


def test_direct_backend_settings_read_overrides(monkeypatch) -> None:
    """Every stage can be pointed at another provider through the environment."""
    monkeypatch.setenv("DIRECT_TTS_BASE_URL", "http://localhost:8880/v1")
    monkeypatch.setenv("DIRECT_TTS_MODEL", "kokoro")
    monkeypatch.setenv("DIRECT_TTS_VOICE", "jf_alpha")
    monkeypatch.setenv("DIRECT_TTS_SAMPLE_RATE", "24000")
    monkeypatch.setenv("DIRECT_TTS_API_KEY", "local")

    settings = config.get_direct_backend_settings()

    assert settings.tts_base_url == "http://localhost:8880/v1"
    assert (settings.tts_model, settings.tts_voice, settings.tts_api_key) == ("kokoro", "jf_alpha", "local")


def test_speech_detection_settings_read_overrides(monkeypatch) -> None:
    """Voice activity thresholds are tunable without a code change."""
    monkeypatch.setenv("DIRECT_VAD_SILENCE_END_S", "1.2")
    monkeypatch.setenv("DIRECT_VAD_MIN_LEVEL", "loud")  # invalid values keep the default

    settings = config.get_speech_detection_settings()

    assert settings.silence_end_s == 1.2
    assert settings.min_level == config.SpeechDetectionSettings().min_level


def test_resolve_available_voice_normalizes_and_falls_back() -> None:
    """Voice names resolve case-insensitively and fall back when unsupported."""
    assert config.resolve_available_voice("ono_anna", source="test") == "Ono_Anna"
    assert config.resolve_available_voice("nope", source="test", fallback="Aiden") == "Aiden"
    assert config.resolve_available_voice("", source="test") is None


def test_language_model_key_falls_back_across_providers(monkeypatch) -> None:
    """A language model pointed away from Hugging Face should still find a configured key."""
    monkeypatch.delenv("DIRECT_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config.refresh_runtime_config_from_env()

    assert config.get_direct_backend_settings().llm_api_key == "openai-key"

    monkeypatch.setenv("HF_TOKEN", "hf-token")
    config.refresh_runtime_config_from_env()

    assert config.get_direct_backend_settings().llm_api_key == "hf-token"


def test_vision_is_off_unless_asked_for(monkeypatch) -> None:
    """The default language model reads text only, so images are opt-in."""
    monkeypatch.delenv("DIRECT_LLM_VISION", raising=False)

    assert config.get_direct_backend_settings().llm_vision is False

    monkeypatch.setenv("DIRECT_LLM_VISION", "1")

    assert config.get_direct_backend_settings().llm_vision is True
