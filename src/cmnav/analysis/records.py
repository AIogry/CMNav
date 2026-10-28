"""Strict saved-record I/O, integrity tracking, and analysis provenance.

Readers do not infer absent files to mean empty records unless the caller
explicitly requests optional=True. Historical schemas are decoded by their
analysis adapters, not rewritten by these functions.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from cmnav.runtime.artifacts import sha256_file
from cmnav.runtime.provenance import source_hashes


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_rows(path: Path, *, optional: bool = False) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if optional and not path.exists():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_rows(path: Path, *, optional: bool = False) -> list[dict[str, Any]]:
    return list(iter_rows(path, optional=optional))


def digest(path: Path) -> str:
    return sha256_file(Path(path))


def read_tracked(path: Path, inputs: dict[str, str], *, lines: bool = False) -> Any:
    path = Path(path).resolve()
    inputs[str(path)] = digest(path)
    return read_rows(path) if lines else read_json(path)


def save_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    """Serialize strict JSON without coercing nonfinite values or creating parents."""
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_new_json(path: Path, value: Any) -> None:
    save_json(path, value, exclusive=True)


def json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def implementation_provenance(path: Path) -> dict[str, Any]:
    """Retain the legacy implementation hash and identify all shared source."""
    path = Path(path).resolve()
    return {"implementation_path": str(path), "implementation_sha256": digest(path),
            "package_source_hashes": source_hashes(),
            "historical_source_note": "Current analysis implementation; does not replace source saved by the input run."}


def capture_analysis_sources(output_directory: Path) -> dict[str, Any]:
    """Archive current analysis code separately, retaining existing snapshots."""
    from datetime import datetime, timezone
    from uuid import uuid4
    from cmnav.runtime.source_snapshot import archive_package_sources

    identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    return archive_package_sources(Path(output_directory) / "analysis_sources" / identity)


def project_outputs_root(project_root: Path) -> Path:
    """Resolve an explicitly selected CMNav checkout before applying its boundary."""
    root = Path(project_root).expanduser().resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "src/cmnav/__init__.py").is_file():
        raise ValueError(f"Not a CMNav project root: {root}")
    return root / "outputs"
