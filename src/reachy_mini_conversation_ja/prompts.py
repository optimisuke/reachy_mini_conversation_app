"""Resolve active profile prompts and voice settings."""

import logging
from pathlib import Path

from reachy_mini_conversation_ja.config import config, get_default_voice, get_location_settings
from reachy_mini_conversation_ja.memory import format_memory_for_prompt
from reachy_mini_conversation_ja.profile_store import (
    DEFAULT_PROFILE_NAME,
    ProfileDefinition,
    ProfileFormatError,
    read_profile,
    read_packaged_default_profile,
)


logger = logging.getLogger(__name__)

DEFAULT_GREETING_PROMPT = (
    "Start the conversation now with a brief, spontaneous greeting in character. "
    "Keep it to one sentence, invite the user in naturally, and vary the wording each time."
)


def _active_profile() -> ProfileDefinition:
    return read_profile(config.REACHY_MINI_CUSTOM_PROFILE)


def get_session_instructions(instance_path: str | Path | None = None) -> str:
    """Return instructions for the active profile with memory context."""
    selected_profile = config.REACHY_MINI_CUSTOM_PROFILE
    profile_name = selected_profile or DEFAULT_PROFILE_NAME
    try:
        profile = _active_profile()
        instructions = profile.instructions.strip()
    except (FileNotFoundError, ProfileFormatError) as exc:
        logger.warning("Failed to load profile %r: %s", profile_name, exc)
        instructions = ""

    if not instructions and selected_profile and selected_profile != DEFAULT_PROFILE_NAME:
        logger.warning("Using bundled default instructions because profile %r is incomplete", selected_profile)
        try:
            instructions = read_packaged_default_profile().instructions.strip()
        except (FileNotFoundError, ProfileFormatError) as exc:
            raise RuntimeError("Default profile has no usable instructions") from exc
    if not instructions:
        raise RuntimeError("Default profile has no usable instructions")

    location_prompt = format_location_for_prompt()
    if location_prompt:
        instructions = f"{instructions}\n\n{location_prompt}"

    memory_prompt = format_memory_for_prompt(instance_path)
    if memory_prompt:
        return f"{memory_prompt}\n\n{instructions}"
    return instructions


def format_location_for_prompt() -> str:
    """Return a note telling the model where the robot is, or an empty string."""
    settings = get_location_settings()
    lines: list[str] = []
    if settings.place:
        lines.append(
            f"You and the person you are talking to are in {settings.place}. "
            f"When they ask about the weather without naming a place, look up {settings.place} "
            "instead of asking which city they mean. If the lookup finds nothing, try a broader "
            "name — a prefecture or a nearby large city — rather than asking them."
        )
    if settings.timezone:
        lines.append(
            f"The local timezone is {settings.timezone}. Always pass it to the time tool: "
            "leaving the timezone empty answers in UTC, which is the wrong time here."
        )
    if not lines:
        return ""
    return "## WHERE YOU ARE\n" + "\n".join(lines)


def get_session_voice(default: str | None = None) -> str:
    """Return the active profile voice or the backend default."""
    fallback = get_default_voice() if default is None else default
    try:
        return _active_profile().voice or fallback
    except (FileNotFoundError, ProfileFormatError) as exc:
        logger.warning("Failed to load the active profile voice: %s", exc)
        return fallback


def get_session_greeting_prompt() -> str:
    """Return the active profile greeting prompt or the app default."""
    try:
        return _active_profile().greeting or DEFAULT_GREETING_PROMPT
    except (FileNotFoundError, ProfileFormatError) as exc:
        logger.warning("Failed to load the active profile greeting: %s", exc)
        return DEFAULT_GREETING_PROMPT
