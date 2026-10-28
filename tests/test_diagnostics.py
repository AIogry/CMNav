"""Preflight must follow Habitat's data selection without loading the simulator."""

import builtins
import gzip
import importlib.metadata
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import urllib.request

import pytest

from cmnav.envs import diagnostics
from cmnav.data import pointnav


@pytest.fixture
def available_modules(monkeypatch):
    versions = {
        "numpy": "1.26.4",
        "PyYAML": "6.0.2",
        "Pillow": "10.4.0",
        "habitat-lab": "0.3.3",
        "habitat-sim": "0.3.3",
    }

    def version(distribution):
        if distribution not in versions:
            raise importlib.metadata.PackageNotFoundError(distribution)
        return versions[distribution]

    monkeypatch.setattr(
        diagnostics.importlib.util,
        "find_spec",
        lambda module: SimpleNamespace(origin=f"/environment/{module}/__init__.py"),
    )
    monkeypatch.setattr(diagnostics.importlib.metadata, "version", version)
    return versions


@pytest.fixture
def dataset(tmp_path, available_modules):
    root = tmp_path / "episodes/train/train.json.gz"
    root.parent.mkdir(parents=True)
    assets = tmp_path / "scene_datasets"
    assets.mkdir()
    return {
        "dataset": {
            "name": "mp3d",
            "split": "train",
            "episodes_path": str(root),
            "scenes_dir": str(assets),
            "scene_ids": ["scene_a"],
        }
    }


def write_gzip(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(value, stream)


def episode(scene):
    return {
        "episode_id": "0",
        "scene_id": f"data/scene_datasets/mp3d/{scene}/{scene}.glb",
        "start_position": [0, 0, 0],
        "start_rotation": [0, 0, 0, 1],
        "goals": [{"position": [1, 0, 0]}],
    }


def touch_asset(config, scene):
    asset = Path(config["dataset"]["scenes_dir"]) / "mp3d" / scene / f"{scene}.glb"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(b"fixture: existence is the static check")
    return asset


def errors(report):
    return [check for check in report["checks"] if check["status"] == "error"]


def test_empty_root_loads_selected_content_shard_and_strips_asset_prefix(dataset):
    root = Path(dataset["dataset"]["episodes_path"])
    write_gzip(root, {"episodes": []})
    write_gzip(root.parent / "content/scene_a.json.gz", {"episodes": [episode("scene_a")]})
    # This shard and asset are absent: an explicit single-scene check must not need them.
    write_gzip(root.parent / "content/scene_b.json.gz", {"episodes": [episode("scene_b")]})
    asset = touch_asset(dataset, "scene_a")

    report = diagnostics.inspect_environment(dataset)

    assert report["ready_for_run"], errors(report)
    assert any(str(asset) in check["message"] for check in report["checks"])
    json.dumps(report)  # The CLI must be able to persist the entire report.


def test_inline_dataset_filters_unselected_scene_assets(dataset):
    write_gzip(
        Path(dataset["dataset"]["episodes_path"]),
        {"episodes": [episode("scene_a"), episode("missing_scene")]},
    )
    touch_asset(dataset, "scene_a")

    report = diagnostics.inspect_environment(dataset)

    assert report["ready_for_run"], errors(report)


@pytest.mark.parametrize("contents", [b"not gzip", gzip.compress(b"{malformed json")])
def test_malformed_episode_file_becomes_actionable_error(dataset, contents):
    root = Path(dataset["dataset"]["episodes_path"])
    root.write_bytes(contents)

    report = diagnostics.inspect_environment(dataset)

    assert not report["ready_for_run"]
    assert any(str(root) in check["message"] for check in errors(report))


def test_missing_asset_is_reported_without_network_or_download(dataset, monkeypatch, tmp_path):
    write_gzip(Path(dataset["dataset"]["episodes_path"]), {"episodes": [episode("scene_a")]})
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    def forbid_network(*args, **kwargs):
        pytest.fail("doctor must not download licensed scene assets or access the network")

    monkeypatch.setattr(socket, "create_connection", forbid_network)
    monkeypatch.setattr(urllib.request, "urlopen", forbid_network)
    report = diagnostics.inspect_environment(dataset)

    assert not report["ready_for_run"]
    assert any(check["name"] == "scene_asset" and "scene_a.glb" in check["message"] for check in errors(report))
    assert before == sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))


def test_static_package_check_does_not_import_native_modules(dataset, monkeypatch):
    write_gzip(Path(dataset["dataset"]["episodes_path"]), {"episodes": [episode("scene_a")]})
    touch_asset(dataset, "scene_a")
    original_import = builtins.__import__

    def guard_import(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "habitat", "habitat_sim"}:
            pytest.fail(f"doctor imported native module {name}")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as guarded:
        guarded.setattr(builtins, "__import__", guard_import)
        report = diagnostics.inspect_environment(dataset)
    assert report["ready_for_run"], errors(report)


def test_missing_habitat_module_blocks_run_even_when_distribution_exists(dataset, monkeypatch):
    write_gzip(Path(dataset["dataset"]["episodes_path"]), {"episodes": [episode("scene_a")]})
    touch_asset(dataset, "scene_a")
    monkeypatch.setattr(
        diagnostics.importlib.util,
        "find_spec",
        lambda module: None if module == "habitat_sim" else SimpleNamespace(origin="/environment/module.py"),
    )

    report = diagnostics.inspect_environment(dataset)

    assert not report["ready_for_run"]
    assert any(check["name"] == "package:habitat_sim" for check in errors(report))


def test_unknown_habitat_distribution_version_warns_but_does_not_crash(dataset, available_modules):
    write_gzip(Path(dataset["dataset"]["episodes_path"]), {"episodes": [episode("scene_a")]})
    touch_asset(dataset, "scene_a")
    available_modules.pop("habitat-sim")

    report = diagnostics.inspect_environment(dataset)

    assert report["ready_for_run"], errors(report)
    assert any(check["name"] == "package:habitat_sim" and check["status"] == "warning" for check in report["checks"])


def test_gzip_decompression_is_bounded(dataset, monkeypatch):
    write_gzip(Path(dataset["dataset"]["episodes_path"]), {"episodes": [episode("scene_a")]})
    monkeypatch.setattr(pointnav, "MAX_JSON_BYTES", 8)

    report = diagnostics.inspect_environment(dataset)

    assert not report["ready_for_run"]
    assert any("limit" in check["message"] for check in errors(report))
