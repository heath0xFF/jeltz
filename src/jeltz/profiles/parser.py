"""TOML profile parser — reads device profiles into validated Pydantic models."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import ValidationError

from jeltz.devices.model import DeviceModel


class ProfileError(Exception):
    """Raised when a profile cannot be parsed or validated."""

    def __init__(self, message: str, path: str | None = None) -> None:
        self.path = path
        super().__init__(message)


def parse_profile(path: Path) -> DeviceModel:
    """Parse a TOML device profile from a file path."""
    if not path.exists():
        raise ProfileError(f"profile not found: {path}", path=str(path))

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ProfileError(f"cannot read profile: {e}", path=str(path)) from e

    return parse_profile_string(raw, source=str(path))


def parse_profile_string(content: str, source: str | None = None) -> DeviceModel:
    """Parse a TOML device profile from a string."""
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError as e:
        raise ProfileError(f"invalid TOML: {e}", path=source) from e

    try:
        return DeviceModel.model_validate(data)
    except ValidationError as e:
        raise ProfileError(f"invalid profile: {e}", path=source) from e
