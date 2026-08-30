"""Tests for the location note added to the session instructions."""

import pytest

from reachy_mini_conversation_ja import config
from reachy_mini_conversation_ja.prompts import format_location_for_prompt


def test_the_shipped_defaults_are_described(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing configured the robot still knows it is in Japan."""
    monkeypatch.delenv(config.LOCATION_TIMEZONE_ENV, raising=False)
    monkeypatch.delenv(config.LOCATION_PLACE_ENV, raising=False)

    note = format_location_for_prompt()

    assert config.LOCATION_DEFAULT_PLACE in note
    assert config.LOCATION_DEFAULT_TIMEZONE in note


def test_a_configured_place_replaces_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Someone elsewhere must be able to move the robot without editing the code."""
    monkeypatch.setenv(config.LOCATION_TIMEZONE_ENV, "Europe/Paris")
    monkeypatch.setenv(config.LOCATION_PLACE_ENV, "Paris, France")

    note = format_location_for_prompt()

    assert "Paris, France" in note
    assert "Europe/Paris" in note
    assert config.LOCATION_DEFAULT_PLACE not in note


def test_timezone_alone_covers_the_time_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timezone is passed on with the reason the default is wrong."""
    monkeypatch.setenv(config.LOCATION_TIMEZONE_ENV, "Asia/Tokyo")
    monkeypatch.delenv(config.LOCATION_PLACE_ENV, raising=False)

    note = format_location_for_prompt()

    assert "Asia/Tokyo" in note
    assert "UTC" in note
