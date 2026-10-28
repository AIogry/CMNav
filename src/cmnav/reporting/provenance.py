"""Recoverable reporting sources and package hashes for newly generated analyses."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from uuid import uuid4
from cmnav.analysis.records import implementation_provenance, save_json
from cmnav.analysis.report_contracts import COHORT_CONTRACT_PATH
from cmnav.runtime.source_snapshot import archive_package_sources


def report_provenance(implementation: Path, destination: Path) -> dict[str, Any]:
    """Archive actual package sources without replacing prior report snapshots.

    Snapshot references are relative to the report destination so a staged
    report can be published by an atomic directory rename.
    """
    identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    directory = destination / "report_sources" / identity
    snapshot = archive_package_sources(directory, entrypoints=(COHORT_CONTRACT_PATH,))
    snapshot["archive"] = "source.tar.gz"
    snapshot["archive_path_base"] = "manifest_directory"
    save_json(directory / "source_manifest.json", snapshot)
    return {
        **implementation_provenance(implementation),
        "source_snapshot": {
            "archive": str((directory / "source.tar.gz").relative_to(destination)),
            "manifest": str((directory / "source_manifest.json").relative_to(destination)),
            "archive_sha256": snapshot["archive_sha256"],
            "path_base": "report_output_directory",
        },
    }
