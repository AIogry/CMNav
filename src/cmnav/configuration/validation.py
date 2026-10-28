"""Shared, side-effect-free checks for explicit configuration fields."""
from __future__ import annotations

import math
from typing import Any


class ConfigError(ValueError):
    """An experiment setting is absent, unsupported, or invalid."""


def validate_keys(value: Any, required: set[str], where: str, optional: set[str] | None = None) -> None:
    """Require a mapping with exactly the declared required and optional keys."""
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping")
    missing, extra = required - value.keys(), value.keys() - required - (optional or set())
    if missing:
        raise ConfigError(f"{where}: missing fields {sorted(missing)}")
    if extra:
        raise ConfigError(f"{where}: unknown fields {sorted(extra)}")


def validate_number(value: Any, name: str, minimum: float, maximum: float | None = None,
                    integer: bool = False) -> None:
    """Check finite numeric type and inclusive bounds, excluding booleans."""
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value):
        raise ConfigError(f"{name} must be a finite {'integer' if integer else 'number'}")
    if value < minimum or (maximum is not None and value > maximum):
        raise ConfigError(f"{name} is outside the allowed range [{minimum}, {maximum}]")
