"""Static environment and PointNav data checks, without importing Habitat.

Passing these checks establishes that the required modules and data are present.
GPU/EGL initialization, native library compatibility, and scene loading are checked
by the actual environment run, not by this inexpensive preflight.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from cmnav.configuration import EXPECTED_HABITAT_VERSION
from cmnav.data import pointnav
from cmnav.data.pointnav import (
    DEFAULT_CONTENT_SCENES_PATH,
    SCENES_PATH_PREFIX,
    EpisodeDatasetInvalid,
    EpisodeDatasetTooLarge,
    read_episode_dataset,
)


MAX_SCENE_FILES = 32
MAX_WILDCARD_SCENE_FILES = 8


def _add(checks: List[dict], name: str, status: str, message: str) -> None:
    checks.append({"name": name, "status": status, "message": message})


def _module_info(module: str, distributions: tuple) -> Dict[str, Any]:
    """Use top-level module specs and package metadata; never import native code."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, AttributeError, OSError, ValueError) as exc:
        return {"available": False, "version": None, "detail": str(exc)}
    version = None
    for distribution in distributions:
        try:
            version = importlib.metadata.version(distribution)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
        except (OSError, ValueError):
            continue
    return {
        "available": spec is not None,
        "version": version,
        "location": str(spec.origin) if spec and spec.origin else None,
    }


def _check_packages(checks: List[dict]) -> Dict[str, dict]:
    packages = {}
    for module, distributions in (
        ("numpy", ("numpy",)),
        ("yaml", ("PyYAML",)),
        ("PIL", ("Pillow",)),
        ("habitat", ("habitat-lab", "habitat")),
        ("habitat_sim", ("habitat-sim",)),
    ):
        info = _module_info(module, distributions)
        packages[module] = info
        if not info["available"]:
            _add(
                checks,
                "package:" + module,
                "error",
                f"Missing importable module {module!r} in {sys.executable}. "
                "Activate the CMNav environment and follow docs/before10-22/setup.md.",
            )
        elif module in ("habitat", "habitat_sim"):
            version = info["version"]
            expected = version == EXPECTED_HABITAT_VERSION
            _add(
                checks,
                "package:" + module,
                "ok" if expected else "warning",
                f"{module}: version {version or 'unknown (no distribution metadata)'}. "
                f"CMNav targets {EXPECTED_HABITAT_VERSION}."
                + (" Other versions have not been validated." if not expected else ""),
            )
        else:
            _add(
                checks,
                "package:" + module,
                "ok",
                f"{module}: {info['version'] or 'available; version metadata unavailable'}.",
            )
    return packages


def _read_dataset(path: Path, checks: List[dict]) -> Optional[dict]:
    """Limit decompressed input size, including unexpectedly large gzip files."""
    try:
        try:
            data = read_episode_dataset(path, max_bytes=pointnav.MAX_JSON_BYTES)
        except EpisodeDatasetTooLarge as exc:
            raise ValueError(
                f"decompressed JSON exceeds the {pointnav.MAX_JSON_BYTES // (1024 * 1024)} MiB "
                "doctor limit; use a smaller diagnostic split"
            ) from exc
        except EpisodeDatasetInvalid as exc:
            raise ValueError("expected a JSON object containing an 'episodes' list") from exc
    except (OSError, EOFError, ValueError, UnicodeError, RecursionError, zlib.error) as exc:
        _add(checks, "episodes_file", "error", f"Cannot read PointNav episodes {path}: {exc}")
        return None
    _add(checks, "episodes_file", "ok", f"Read {len(data['episodes'])} episodes from {path}.")
    return data


def _config_path(dataset: dict, key: str, checks: List[dict]) -> Optional[Path]:
    value = dataset.get(key)
    if not isinstance(value, str) or not value or "\x00" in value:
        _add(checks, key, "error", f"dataset.{key} must be a nonempty absolute path string.")
        return None
    path = Path(value)
    if not path.is_absolute():
        _add(checks, key, "error", f"dataset.{key} must be resolved to an absolute path: {value}")
        return None
    return path


def _check_dataset(dataset: dict, checks: List[dict]) -> None:
    episodes_path = _config_path(dataset, "episodes_path", checks)
    scenes_dir = _config_path(dataset, "scenes_dir", checks)
    if "scene_dataset_config" in dataset:
        scene_config = _config_path(dataset, "scene_dataset_config", checks)
        if scene_config is not None:
            try:
                if not scene_config.is_file():
                    raise ValueError(f"Scene dataset config does not exist: {scene_config}")
                if scene_config.stat().st_size > pointnav.MAX_JSON_BYTES:
                    raise ValueError(f"Scene dataset config exceeds doctor size limit: {scene_config}")
                value = json.loads(scene_config.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("Scene dataset config must be a JSON object")
                _add(checks, "scene_dataset_config", "ok", f"Read scene dataset config: {scene_config}")
            except (OSError, ValueError, UnicodeError, RecursionError) as exc:
                _add(checks, "scene_dataset_config", "error", str(exc))
    scene_ids = dataset.get("scene_ids")
    if (
        not isinstance(scene_ids, list)
        or not scene_ids
        or any(
            not isinstance(scene, str)
            or not scene
            or scene in (".", "..")
            or any(character in scene for character in ("/", "\\", "\x00"))
            for scene in scene_ids
        )
    ):
        _add(checks, "scene_ids", "error", "dataset.scene_ids must contain bare scene names or '*'.")
        return
    if episodes_path is None or scenes_dir is None:
        return
    try:
        if not scenes_dir.is_dir():
            _add(checks, "scenes_dir", "error", f"Scene asset directory does not exist: {scenes_dir}")
        else:
            _add(checks, "scenes_dir", "ok", f"Scene asset directory: {scenes_dir}")
    except OSError as exc:
        _add(checks, "scenes_dir", "error", f"Cannot inspect {scenes_dir}: {exc}")
    root = _read_dataset(episodes_path, checks)
    if root is None:
        return

    template = root.get("content_scenes_path", DEFAULT_CONTENT_SCENES_PATH)
    try:
        if not isinstance(template, str) or template.count("{scene}") != 1:
            raise ValueError("content_scenes_path must contain exactly one '{scene}' placeholder")
        prefix, suffix = template.split("{scene}")
        content_dir = Path(prefix.format(data_path=str(episodes_path.parent)))
        # Format once to detect unsupported fields before processing any shards.
        template.format(data_path=str(episodes_path.parent), scene="example")
        has_shards = content_dir.exists()
    except (KeyError, IndexError, ValueError, OSError) as exc:
        _add(checks, "content_scenes_path", "error", f"Invalid content_scenes_path in {episodes_path}: {exc}")
        return

    wildcard = "*" in scene_ids
    selected = set(scene_ids)
    assets = set()
    count = 0
    malformed = 0

    def collect(episodes: list, filter_scenes: bool = False) -> None:
        nonlocal count, malformed
        for episode in episodes:
            scene = episode.get("scene_id") if isinstance(episode, dict) else None
            if not isinstance(scene, str) or not scene or "\x00" in scene:
                malformed += 1
                continue
            if filter_scenes and not wildcard and Path(scene).stem not in selected:
                continue
            count += 1
            # Match Habitat PointNavDatasetV1.from_json: strip only this prefix,
            # then join scenes_dir. Absolute episode paths remain absolute.
            relative = scene[len(SCENES_PATH_PREFIX):] if scene.startswith(SCENES_PATH_PREFIX) else scene
            assets.add(scenes_dir / relative)

    # Habitat retains root episodes unfiltered if a content directory exists.
    collect(root["episodes"], filter_scenes=not has_shards)
    if has_shards:
        try:
            if not content_dir.is_dir():
                raise ValueError(f"content path is not a directory: {content_dir}")
            if wildcard:
                if not suffix:
                    raise ValueError("content_scenes_path must include a filename suffix")
                available = sorted(
                    entry.name[:-len(suffix)]
                    for entry in content_dir.iterdir()
                    if entry.is_file() and entry.name.endswith(suffix)
                )
                scenes = available[:MAX_WILDCARD_SCENE_FILES]
                _add(
                    checks,
                    "scene_selection",
                    "warning",
                    f"Wildcard selected {len(available)} content shards; doctor inspects the first "
                    f"{len(scenes)}. Set explicit dataset.scene_ids for a complete preflight of the run.",
                )
            else:
                scenes = list(dict.fromkeys(scene_ids))
                if len(scenes) > MAX_SCENE_FILES:
                    _add(
                        checks,
                        "scene_selection",
                        "error",
                        f"Selected {len(scenes)} scenes; doctor supports at most {MAX_SCENE_FILES} explicit "
                        "scenes per check. Use a smaller scene list for the first run.",
                    )
                    scenes = scenes[:MAX_SCENE_FILES]
            for scene in scenes:
                shard = Path(template.format(data_path=str(episodes_path.parent), scene=scene))
                data = _read_dataset(shard, checks)
                if data is not None:
                    if not data["episodes"]:
                        _add(checks, "scene_episodes", "error", f"Selected scene shard contains no episodes: {shard}")
                    collect(data["episodes"])
        except (OSError, ValueError, KeyError, IndexError) as exc:
            _add(checks, "content_shards", "error", f"Cannot inspect episode shards under {content_dir}: {exc}")
    if malformed:
        _add(checks, "episode_schema", "error", f"{malformed} episode entries lack a valid scene_id.")
    if not count:
        _add(
            checks,
            "selected_episodes",
            "error",
            f"No episodes available for scene_ids={scene_ids!r}. Check {episodes_path} and "
            f"the corresponding content shards under {content_dir}.",
        )
    else:
        _add(checks, "selected_episodes", "ok", f"Inspected {count} loadable scene references across {len(assets)} assets.")
    for asset in sorted(assets):
        try:
            if not asset.is_file():
                _add(checks, "scene_asset", "error", f"Episode references a missing scene asset: {asset}")
            elif asset.suffix.lower() != ".glb":
                _add(checks, "scene_asset", "warning", f"Expected an MP3D .glb asset; found {asset}")
            else:
                _add(checks, "scene_asset", "ok", f"Scene asset exists: {asset}")
        except OSError as exc:
            _add(checks, "scene_asset", "error", f"Cannot inspect scene asset {asset}: {exc}")


def inspect_environment(config: dict) -> dict:
    """Return JSON-serializable static preflight diagnostics for a resolved config.

    The input paths must already be resolved by the configuration loader. No
    Habitat or NumPy imports, GPU probing, data downloads, or writes occur here.
    Wildcard selection is deliberately sampled, as reported in its warning.
    """
    checks: List[dict] = []
    runtime = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if sys.version_info[:2] < (3, 9):
        _add(checks, "python", "error", "CMNav requires Python 3.9 or newer; the Habitat environment targets Python 3.9.")
    elif sys.version_info[:2] != (3, 9):
        _add(checks, "python", "warning", f"Python {runtime['python']}; the Habitat 0.3.3 setup targets Python 3.9.")
    else:
        _add(checks, "python", "ok", f"Python {runtime['python']} matches the target environment.")
    runtime["packages"] = _check_packages(checks)
    if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict):
        _add(checks, "configuration", "error", "Expected a resolved configuration with a 'dataset' mapping.")
    else:
        _check_dataset(config["dataset"], checks)
    _add(
        checks,
        "native_runtime",
        "warning",
        "Static checks do not initialize Habitat, GPU/EGL, or NavMesh. A real episode run is required to validate them.",
    )
    return {
        "ready_for_run": not any(check["status"] == "error" for check in checks),
        "checks": checks,
        "runtime": runtime,
    }
