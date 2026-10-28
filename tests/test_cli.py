"""Current and compatibility CLI names dispatch to one diagnostic workflow."""
import pytest

from cmnav.cli import main
from cmnav.configuration import graph_navigation
from cmnav.workflows import oracle_graph_navigation


@pytest.mark.parametrize("command", ["diagnose-graph-navigation", "prototype"])
def test_graph_navigation_names_share_workflow_and_new_default_config(
    command, tmp_path, monkeypatch, capsys,
):
    config = {"marker": "resolved_configuration"}
    calls = []

    def load(path):
        calls.append(("config", path))
        return config

    def run(actual_config, plan, *, reference_run=None):
        calls.append(("workflow", actual_config, plan, reference_run))
        return tmp_path / "result"

    monkeypatch.setattr(graph_navigation, "load_graph_navigation_config", load)
    monkeypatch.setattr(oracle_graph_navigation, "run_oracle_graph_navigation", run)
    assert main([command, "--plan", "frozen-plan.json", "--reference-run", "history"]) == 0
    assert calls == [
        ("config", "configs/mp3d_oracle_graph.yaml"),
        ("workflow", config, "frozen-plan.json", "history"),
    ]
    output = capsys.readouterr()
    assert str(tmp_path / "result") in output.out
    assert ("deprecated" in output.err) == (command == "prototype")
