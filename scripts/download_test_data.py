#!/usr/bin/env python3
"""Fetch one official public Habitat test scene; never fetch MP3D assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVISION = "910c783fb954da8497ea5f811b843a76590ddddc"
BASE = f"https://huggingface.co/datasets/ai-habitat/habitat_test_scenes/resolve/{REVISION}"
SCENE_SHA256 = "b14e29e17f5e31d86a1002eefd77b7d345b265006481739ae480a847e6623f56"
EPISODES_URL = "https://dl.fbaipublicfiles.com/habitat/habitat-test-pointnav-dataset_v1.0.zip"
EPISODES_SHA256 = "4ddf3403507c6ea44add9e375134d3564219f4376d11b16ed3efe16245a167d7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_hash: str | None = None) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"Downloading {destination.name}", flush=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "CMNav/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if expected_hash and sha256(temporary) != expected_hash:
                raise ValueError(f"Checksum mismatch: {destination.name}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    digest = sha256(destination)
    if expected_hash and digest != expected_hash:
        raise ValueError(f"Existing cache has wrong checksum: {destination}; move it aside and retry")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    args = parser.parse_args()
    data_root, cache = args.data_root.expanduser().resolve(), args.cache_dir.expanduser().resolve()
    scene_dir = data_root / "scene_datasets/habitat-test-scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name, expected in (("skokloster-castle.glb", SCENE_SHA256),
                           ("skokloster-castle.navmesh", None), ("README.md", None)):
        url = f"{BASE}/{name}"
        cached = cache / ("habitat_test_README.md" if name == "README.md" else name)
        digest = download(url, cached, expected)
        target = scene_dir / name
        if target.exists() and sha256(target) != digest:
            raise ValueError(f"Refusing to replace a different existing asset: {target}")
        if not target.exists():
            shutil.copy2(cached, target)
        files.append({"path": str(target.relative_to(data_root)), "source": url, "sha256": digest})
    archive = cache / "habitat-test-pointnav-dataset_v1.0.zip"
    download(EPISODES_URL, archive, EPISODES_SHA256)
    with zipfile.ZipFile(archive) as compressed:
        for split in ("train", "val", "test"):
            member = f"v1/{split}/{split}.json.gz"
            target = data_root / "datasets/pointnav/habitat-test-scenes" / member
            payload = compressed.read(member)  # whitelist, not arbitrary archive extraction
            digest = hashlib.sha256(payload).hexdigest()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha256(target) != digest:
                raise ValueError(f"Refusing to replace a different episode dataset: {target}")
            if not target.exists():
                target.write_bytes(payload)
            files.append({"path": str(target.relative_to(data_root)), "source": EPISODES_URL,
                          "archive_member": member, "sha256": digest})
    manifest = {"dataset": "habitat_test", "purpose": "public environment validation, not MP3D",
                "scene_revision": REVISION, "episodes_archive_sha256": EPISODES_SHA256, "files": files}
    (data_root / "test_data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Public test data ready: {data_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

