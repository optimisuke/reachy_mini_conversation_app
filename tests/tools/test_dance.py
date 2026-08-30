"""Tests for the dance tool."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_ja.tools.dance import RANDOM_MOVE, AVAILABLE_MOVES, Dance
from reachy_mini_conversation_ja.tools.core_tools import ToolDependencies


def _deps() -> ToolDependencies:
    """Build tool dependencies with a movement manager that records queued moves."""
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


@pytest.mark.asyncio
async def test_random_is_a_move_the_model_may_ask_for() -> None:
    """The schema offers "random", so asking for it must play something, not fail."""
    assert RANDOM_MOVE in Dance.parameters_schema["properties"]["move"]["enum"]

    result = await Dance()(_deps(), move=RANDOM_MOVE)

    assert result["move"] in AVAILABLE_MOVES
    assert result["status"] == "queued"


@pytest.mark.asyncio
async def test_omitting_the_move_still_picks_one() -> None:
    """A dance with no preference expressed plays a move of Reachy's choosing."""
    result = await Dance()(_deps(), repeat=2)

    assert result["move"] in AVAILABLE_MOVES
    assert result["repeat"] == 2


@pytest.mark.asyncio
async def test_an_unknown_move_is_reported() -> None:
    """A move that does not exist is still an error, naming what is available."""
    result = await Dance()(_deps(), move="moonwalk")

    assert "Unknown dance move 'moonwalk'" in result["error"]


@pytest.mark.asyncio
async def test_a_named_move_is_queued_once_per_repeat() -> None:
    """Each repeat queues the move again."""
    deps = _deps()
    move_name = next(iter(AVAILABLE_MOVES))

    result = await Dance()(deps, move=move_name, repeat=3)

    assert result == {"status": "queued", "move": move_name, "repeat": 3}
    assert deps.movement_manager.queue_move.call_count == 3


def test_the_move_list_reaches_the_model(monkeypatch: Any) -> None:
    """Every available move is described in the parameter the model fills in."""
    description = Dance.parameters_schema["properties"]["move"]["description"]

    assert all(move_name in description for move_name in AVAILABLE_MOVES)
