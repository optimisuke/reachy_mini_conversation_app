"""Tests for the location note added to the session instructions."""

import pytest

from reachy_mini_conversation_ja import config
from reachy_mini_conversation_ja.prompts import format_location_for_prompt


def test_no_note_without_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unconfigured robot says nothing about where it is."""
    monkeypatch.delenv(config.LOCATION_TIMEZONE_ENV, raising=False)
    monkeypatch.delenv(config.LOCATION_PLACE_ENV, raising=False)

    assert format_location_for_prompt() == ""


def test_place_alone_covers_the_weather(monkeypatch: pytest.MonkeyPatch) -> None:
    """A place is enough to stop the weather tool asking which city."""
    monkeypatch.delenv(config.LOCATION_TIMEZONE_ENV, raising=False)
    monkeypatch.setenv(config.LOCATION_PLACE_ENV, "Kobe, Japan")

    note = format_location_for_prompt()

    assert "Kobe, Japan" in note
    assert "timezone" not in note


def test_timezone_alone_covers_the_time_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timezone is passed on with the reason the default is wrong."""
    monkeypatch.setenv(config.LOCATION_TIMEZONE_ENV, "Asia/Tokyo")
    monkeypatch.delenv(config.LOCATION_PLACE_ENV, raising=False)

    note = format_location_for_prompt()

    assert "Asia/Tokyo" in note
    assert "UTC" in note
