"""Tests for keeping the app's state out of the installed package."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_ja.main import ReachyMiniConversationApp


def _app(packaged: Path) -> ReachyMiniConversationApp:
    """Build the app object without running the SDK's constructor."""
    app = ReachyMiniConversationApp.__new__(ReachyMiniConversationApp)
    # The SDK returns the module file; the app uses its parent as the state directory.
    app._get_instance_path = MagicMock(return_value=packaged / "main.py")  # type: ignore[method-assign]
    return app


def test_state_lives_under_the_data_directory_not_in_site_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reinstall deletes the package, so the key and memory must not live inside it."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    packaged = tmp_path / "site-packages" / "reachy_mini_conversation_ja"
    packaged.mkdir(parents=True)

    resolved = _app(packaged).durable_instance_path()

    assert resolved == tmp_path / "data" / "reachy_mini_conversation_ja"
    assert resolved.is_dir()


def test_existing_state_is_carried_out_of_the_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Upgrading must not lose the key or the memory an earlier version wrote."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    packaged = tmp_path / "site-packages" / "reachy_mini_conversation_ja"
    packaged.mkdir(parents=True)
    (packaged / ".env").write_text("OPENAI_API_KEY=sk-old\n", encoding="utf-8")
    (packaged / "memory.v1.json").write_text('{"facts": []}', encoding="utf-8")

    resolved = _app(packaged).durable_instance_path()

    assert (resolved / ".env").read_text(encoding="utf-8") == "OPENAI_API_KEY=sk-old\n"
    assert (resolved / "memory.v1.json").exists()


def test_a_file_already_moved_is_not_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The durable copy is the live one; a stale package copy must not clobber it."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    packaged = tmp_path / "site-packages" / "reachy_mini_conversation_ja"
    packaged.mkdir(parents=True)
    (packaged / ".env").write_text("OPENAI_API_KEY=sk-stale\n", encoding="utf-8")
    durable = tmp_path / "data" / "reachy_mini_conversation_ja"
    durable.mkdir(parents=True)
    (durable / ".env").write_text("OPENAI_API_KEY=sk-current\n", encoding="utf-8")

    resolved = _app(packaged).durable_instance_path()

    assert (resolved / ".env").read_text(encoding="utf-8") == "OPENAI_API_KEY=sk-current\n"


def test_an_unwritable_data_directory_falls_back_to_the_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A robot that cannot write there should still start, just without durability."""
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("XDG_DATA_HOME", str(blocker))
    packaged = tmp_path / "site-packages" / "reachy_mini_conversation_ja"
    packaged.mkdir(parents=True)

    assert _app(packaged).durable_instance_path() == packaged
