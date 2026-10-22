#!/usr/bin/env python3
"""Prepare one official MP3D scene and its PointNav v1 training episodes.

All archives, extracted assets, and provenance stay inside CMNav/data. This is
the 17DRP5sb8fy subset of Habitat's mp3d_example_v1.1 and PointNav MP3D v1,
not the complete Matterport3D release. Use with authorized MP3D access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
CACHE = DATA_ROOT / "downloads/matterport"
CHUNK_SIZE = 1024 * 1024
SCENE_ID = "17DRP5sb8fy"
ARCHIVES = (
    {
        "name": "mp3d_example_v1.1.zip",
        "url": "https://dl.fbaipublicfiles.com/habitat/mp3d/mp3d_example_v1.1.zip",
        "bytes": 67765004,
        "destination": "scene_datasets/mp3d",
        "members": (
            f"{SCENE_ID}/{SCENE_ID}.glb",
            f"{SCENE_ID}/{SCENE_ID}.navmesh",
            f"{SCENE_ID}/{SCENE_ID}.house",
            f"{SCENE_ID}/{SCENE_ID}_semantic.ply",
            "mp3d.scene_dataset_config.json",
            "MP_TOS.pdf",
        ),
    },
    {
        "name": "pointnav_mp3d_v1.zip",
        "url": "https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip",
        "bytes": 418680926,
        "destination": "datasets/pointnav/mp3d/v1",
        "members": (
            "train/train.json.gz",
            f"train/content/{SCENE_ID}.json.gz",
        ),
    },
)


def project_path(path: Path) -> Path:
    """Prevent pre-existing symlinks from redirecting data outside CMNav."""
    path.resolve().relative_to(ROOT)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(spec: dict) -> tuple[Path, str]:
    destination = project_path(CACHE / spec["name"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = project_path(destination.with_suffix(destination.suffix + ".part"))
        # Exclusive creation also avoids clobbering another active download.
        with temporary.open("xb") as stream:
            try:
                print(f"Downloading {spec['name']} ({spec['bytes']:,} bytes)", flush=True)
                request = urllib.request.Request(spec["url"], headers={"User-Agent": "CMNav/0.1"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 200:
                        raise ValueError(f"Expected HTTP 200 for {spec['name']}, got {response.status}")
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) != spec["bytes"]:
                        raise ValueError(f"Remote archive size changed: {spec['name']}")
                    received, last_progress = 0, time.monotonic()
                    for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                        received += len(chunk)
                        if received > spec["bytes"]:
                            raise ValueError(f"Archive exceeds expected size: {spec['name']}")
                        stream.write(chunk)
                        if time.monotonic() - last_progress >= 5:
                            print(f"  {received:,}/{spec['bytes']:,} bytes", flush=True)
                            last_progress = time.monotonic()
                    if received != spec["bytes"]:
                        raise ValueError(f"Incomplete download: {received:,}/{spec['bytes']:,} bytes")
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(destination)
    if destination.stat().st_size != spec["bytes"]:
        raise ValueError(f"Cached archive has an unexpected size: {destination}; move it aside and retry")
    print(f"Checking SHA-256: {destination.name}", flush=True)
    return destination, sha256(destination)


def extract_member(archive: zipfile.ZipFile, member: str, destination: Path) -> dict:
    if archive.namelist().count(member) != 1:
        raise ValueError(f"Archive must contain exactly one {member}")
    info = archive.getinfo(member)
    if info.is_dir():
        raise ValueError(f"Expected a file: {member}")
    target = project_path(destination / member)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = project_path(target.with_suffix(target.suffix + ".part"))
    digest, count = hashlib.sha256(), 0
    # Read every selected member to EOF so ZipFile checks its CRC, including
    # when an identical extracted target already exists.
    with temporary.open("xb") as output:
        try:
            with archive.open(info) as source:
                for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
            if count != info.file_size:
                raise ValueError(f"Extracted size mismatch: {member}")
            checksum = digest.hexdigest()
            if target.exists() and sha256(target) != checksum:
                raise ValueError(f"Refusing to replace a different existing asset: {target}")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    if target.exists():
        temporary.unlink()
    else:
        temporary.replace(target)
    print(f"Ready: {target.relative_to(DATA_ROOT)}", flush=True)
    return {
        "archive_member": member,
        "path": str(target.relative_to(DATA_ROOT)),
        "bytes": count,
        "sha256": checksum,
        "zip_crc32": f"{info.CRC:08x}",
    }


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    project_path(DATA_ROOT)
    print(f"Preparing MP3D single-scene subset under {DATA_ROOT}", flush=True)
    archives = []
    for spec in ARCHIVES:
        cached, checksum = download(spec)
        destination = project_path(DATA_ROOT / spec["destination"])
        with zipfile.ZipFile(cached) as archive:
            members = [extract_member(archive, member, destination) for member in spec["members"]]
        archives.append({
            "url": spec["url"],
            "path": str(cached.relative_to(DATA_ROOT)),
            "bytes": spec["bytes"],
            "sha256": checksum,
            "members": members,
        })
    manifest = {
        "dataset": "mp3d",
        "scene_ids": [SCENE_ID],
        "scope": "Single scene from official mp3d_example_v1.1 and matching PointNav MP3D v1 train shard; not the full release",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "integrity_note": "Archive sizes were observed from official HTTP headers. SHA-256 values are computed locally for provenance, not independently published official checksums. Selected ZIP members were checked against their CRC values.",
        "archives": archives,
    }
    target = project_path(DATA_ROOT / "manifests/mp3d_smoke.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = project_path(target.with_suffix(".json.part"))
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(target)
    print(f"MP3D single-scene data ready. Provenance: {target}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
        print(f"MP3D preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
