"""Recoverable package snapshots with declarative contracts and entrypoints."""
from __future__ import annotations

from pathlib import Path
import tarfile
from typing import Any

from cmnav.runtime.artifacts import sha256_file, write_json


def archive_package_sources(destination: Path, *, entrypoints: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Save actual package sources; a wrapper checksum alone cannot restore a run.

    The destination belongs to a newly created run. Existing archives are refused
    so this helper cannot replace the implementation of a historical experiment.
    Optional entrypoints are recorded under unique numbered names with original
    absolute paths in the manifest. In a source checkout, include packaging
    metadata and public experiment/contract definitions: these are part of the
    implementation now that historical defaults no longer live in Python.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "source.tar.gz"
    manifest_path = destination / "source_manifest.json"
    if archive_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Source snapshot already exists: {destination}")
    package = Path(__file__).resolve().parents[1]
    files = [(p, f"cmnav/{p.relative_to(package)}") for p in sorted(package.rglob("*.py"))]
    from cmnav.runtime.project import checkout_root
    try:
        root = checkout_root()
    except RuntimeError:
        root = None
    if root is not None:
        files.append((root / "pyproject.toml", "pyproject.toml"))
        for group in ("contracts", "experiments"):
            for path in sorted((root / "configs" / group).rglob("*")):
                if (path.is_file() and path.suffix in (".json", ".yaml", ".yml")
                        and ".local." not in path.name):
                    files.append((path, str(path.relative_to(root))))
    files.extend((Path(p).resolve(), f"entrypoints/{i:02d}_{Path(p).name}")
                 for i, p in enumerate(entrypoints))
    records = {name: {"path": str(p), "sha256": sha256_file(p)} for p, name in files}
    with tarfile.open(archive_path, "x:gz") as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
    if any(sha256_file(path) != records[name]["sha256"] for path, name in files):
        raise RuntimeError("Source files changed while creating the snapshot")
    result = {"schema": "cmnav.package_source_snapshot.v1", "archive": str(archive_path.resolve()),
              "archive_sha256": sha256_file(archive_path), "files": records}
    write_json(manifest_path, result)
    return result
