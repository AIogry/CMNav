"""Recoverable package snapshots for frozen GCML workflow runs."""
from __future__ import annotations
from pathlib import Path
import shutil
import sys
from typing import Any
from cmnav.runtime.artifacts import sha256_file, write_json
from cmnav.runtime.provenance import source_hashes
from cmnav.runtime.source_snapshot import archive_package_sources
from cmnav.runtime.project import checkout_root

def snapshot_gcml_source(directory: Path, before: dict[str, str], *, runner: Path) -> dict[str, Any]:
    launcher = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    entrypoints = (runner,) + ((launcher,) if launcher is not None and launcher.is_file() and launcher != runner else ())
    archive_package_sources(directory, entrypoints=entrypoints)
    # Keep the stable source_snapshot files used by existing report readers.
    archive = directory / "source.tar.gz"
    destination = directory / "source_snapshot.tar.gz"
    shutil.copyfile(archive, destination)
    if before != source_hashes():
        raise RuntimeError("Package source changed while snapshotting")
    digest = sha256_file(runner)
    metadata = {"package_hashes": before, "script": str(runner), "script_sha256": digest,
                "script_hashes": {str(path): sha256_file(path) for path in entrypoints}, "archive": destination.name,
                "archive_sha256": sha256_file(destination)}
    write_json(directory / "source_snapshot.json", metadata)
    return metadata


def load_historical_contracts(path: Path | None = None):
    """Load explicit legacy comparison constraints, with their source identity."""
    from cmnav.runtime.gcml_artifacts import load_json
    resolved = (path or checkout_root() / "configs/experiments/gcml_historical_contracts.json").resolve()
    value = load_json(resolved)
    if value.get("schema") != "cmnav.gcml_historical_contracts.v1":
        raise ValueError("Unsupported GCML comparison contract")
    return value, resolved, sha256_file(resolved)
