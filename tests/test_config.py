"""Configuration boundaries that affect portability and saved sensor meaning."""

from pathlib import Path

import pytest
import yaml

from cmnav.configuration.environment import load_config
from cmnav.configuration.validation import ConfigError


@pytest.fixture
def experiment(tmp_path):
    example = Path(__file__).resolve().parents[1] / "configs/mp3d_smoke.yaml"
    config = yaml.safe_load(example.read_text(encoding="utf-8"))
    directory = tmp_path / "project/configs"
    directory.mkdir(parents=True)
    return config, directory / "experiment.yaml"


def save(experiment):
    config, path = experiment
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_paths_resolve_from_config_location_even_when_cwd_changes(experiment, tmp_path, monkeypatch):
    config, path = experiment
    config["project_root"] = ".."
    config["dataset"]["episodes_path"] = "datasets/{split}/{split}.json.gz"
    config["dataset"]["scenes_dir"] = "assets"
    config["run"]["output_dir"] = "runs"
    save(experiment)
    unrelated = tmp_path / "different-working-directory"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    loaded = load_config(path)

    project = path.parent.parent
    assert loaded["project_root"] == str(project)
    assert loaded["dataset"]["episodes_path"] == str(project / "datasets/train/train.json.gz")
    assert loaded["dataset"]["scenes_dir"] == str(project / "assets")
    assert loaded["run"]["output_dir"] == str(project / "runs")


def test_absolute_asset_path_is_preserved_and_split_override_applies_before_formatting(experiment, tmp_path):
    config, _ = experiment
    config["dataset"]["scenes_dir"] = str(tmp_path / "shared-assets")
    config["dataset"]["episodes_path"] = "episodes/{split}/{split}.json.gz"

    loaded = load_config(save(experiment), overrides={"dataset.split": "val"})

    assert loaded["dataset"]["scenes_dir"] == str(tmp_path / "shared-assets")
    assert loaded["dataset"]["episodes_path"].endswith("/episodes/val/val.json.gz")


@pytest.mark.parametrize("section", [(), ("dataset",), ("environment", "rgbd")])
def test_unknown_yaml_fields_fail_instead_of_silently_changing_experiment(experiment, section):
    target = experiment[0]
    for key in section:
        target = target[key]
    target["typo_in_setting"] = 123

    with pytest.raises(ConfigError, match="unknown fields"):
        load_config(save(experiment))


def test_unknown_cli_override_is_rejected(experiment):
    with pytest.raises(ConfigError, match="Unknown override"):
        load_config(save(experiment), overrides={"dataset.splti": "val"})


@pytest.mark.parametrize(
    "field",
    [
        ("seed",),
        ("environment", "max_episode_steps"),
        ("environment", "gpu_device_id"),
        ("environment", "rgbd", "width"),
        ("run", "episodes"),
        ("policy", "forward_probability"),
    ],
)
def test_booleans_are_not_accepted_as_integer_or_numeric_settings(experiment, field):
    target = experiment[0]
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = True

    with pytest.raises(ConfigError, match="finite"):
        load_config(save(experiment))


@pytest.mark.parametrize(
    "template",
    [
        "data/{scene}.json.gz",
        "data/{}.json.gz",
        "data/{split.json.gz",
        "data/{split.missing}.json.gz",
        "data/{split.__class__}.json.gz",
        "data/{split!r}.json.gz",
        "data/{split:>10}.json.gz",
    ],
)
def test_only_plain_split_placeholder_is_allowed(experiment, template):
    experiment[0]["dataset"]["episodes_path"] = template

    with pytest.raises(ConfigError, match="placeholder"):
        load_config(save(experiment))


@pytest.mark.parametrize("normalize_depth", [True, 0, "false"])
def test_depth_units_cannot_silently_switch_away_from_metres(experiment, normalize_depth):
    experiment[0]["environment"]["rgbd"]["normalize_depth"] = normalize_depth

    with pytest.raises(ConfigError, match="metres"):
        load_config(save(experiment))


@pytest.mark.parametrize("minimum,maximum", [(1.0, 1.0), (2.0, 1.0), (0.0, float("nan"))])
def test_invalid_depth_interval_fails_before_simulator_creation(experiment, minimum, maximum):
    sensor = experiment[0]["environment"]["rgbd"]
    sensor["min_depth"], sensor["max_depth"] = minimum, maximum

    with pytest.raises(ConfigError):
        load_config(save(experiment))


@pytest.mark.parametrize("dataset_name", ["mp3d", "habitat_test"])
def test_real_and_public_smoke_datasets_can_use_the_same_environment_contract(experiment, dataset_name):
    experiment[0]["dataset"]["name"] = dataset_name

    loaded = load_config(save(experiment))

    assert loaded["dataset"]["name"] == dataset_name
