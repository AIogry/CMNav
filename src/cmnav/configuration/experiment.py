"""Declarative invocation of registered capabilities; configuration never executes code."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import re
from typing import Any


def _argument(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Arguments must be strings or finite numbers")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Numeric arguments must be finite")
    if "\x00" in str(value):
        raise ValueError("Arguments cannot contain NUL bytes")
    return str(value)


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Resolve the working directory relative to the config, preserving option order."""
    source = Path(path).expanduser().resolve()
    content = source.read_bytes()
    if source.suffix.lower() == ".json":
        data = json.loads(content.decode("utf-8"))
    else:
        import yaml
        data = yaml.safe_load(content.decode("utf-8"))
    allowed = {"schema_version", "operation", "profile", "working_directory", "options", "positional"}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError(f"Experiment config must contain only: {', '.join(sorted(allowed))}")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("Experiment config requires schema_version: 1")
    for key in ("operation", "profile"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"Experiment config requires a nonempty {key}")
    working = data.get("working_directory", ".")
    if not isinstance(working, str):
        raise ValueError("working_directory must be a path string")
    working_path = Path(working).expanduser()
    working_path = (source.parent / working_path).resolve()
    if not working_path.is_dir():
        raise ValueError(f"Working directory does not exist: {working_path}")
    from cmnav.runtime.project import require_project_directory
    require_project_directory(working_path)
    options = data.get("options", {})
    positional = data.get("positional", [])
    if not isinstance(options, dict) or not isinstance(positional, list):
        raise ValueError("options must be a mapping and positional must be a list")
    argv = []
    for key, value in options.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", key):
            raise ValueError(f"Invalid option name: {key!r}")
        if value is None or value is False:
            continue
        argv.append("--" + key)
        if value is True:
            continue
        if isinstance(value, list):
            if not value:
                raise ValueError(f"List option {key!r} must not be empty")
            argv.extend(_argument(item) for item in value)
        else:
            argv.append(_argument(value))
    argv.extend(_argument(value) for value in positional)
    return {"operation": data["operation"], "profile": data["profile"],
            "working_directory": str(working_path), "arguments": argv,
            "config_path": str(source), "config_sha256": hashlib.sha256(content).hexdigest(),
            "config": data}
