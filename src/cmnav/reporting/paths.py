"""Checkout-scoped report destinations."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def bounded_output(path: Path) -> Path:
    result = path.expanduser().resolve()
    if not result.is_relative_to(PROJECT_ROOT / "outputs"):
        raise ValueError("Run and output directories must stay inside this project's outputs/.")
    return result
