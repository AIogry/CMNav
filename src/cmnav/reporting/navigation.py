"""Publish navigation figures and prose from prepared analysis; no training or simulation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cmnav.runtime.artifacts import sha256_file as digest, write_json
from cmnav.analysis.records import save_json
from cmnav.analysis.navigation_audit import load_navigation_audit, load_gcml_comparison
from cmnav.analysis.subgoal_comparison import load_subgoal_comparison
from cmnav.analysis.report_contracts import CONDITIONS
from cmnav.reporting.navigation_figures import render_navigation_audit, render_gcml_task, render_subgoal_task
from cmnav.reporting.navigation_text import gcml_markdown, subgoal_markdown
from cmnav.reporting.paths import bounded_output
from cmnav.reporting.provenance import report_provenance


def navigation_report(run: Path, task_index: int, output: Path) -> dict[str, Any]:
    run, output = bounded_output(run), bounded_output(output)
    prefix = output / f"task_{task_index:04d}_navigation_audit"
    if any(prefix.with_suffix(suffix).exists() for suffix in (".png", ".svg", ".json")):
        raise FileExistsError("Navigation audit already exists; choose a new --output-dir")
    data = load_navigation_audit(run, task_index)
    artifacts = render_navigation_audit(data, output)
    config_path = data["config_path"]
    graph_path = data["graph_path"]
    protocol_path = data["protocol_path"]
    methods = data["methods"]
    episode_id = data["episode_id"]
    config = data["config"]
    protocol = data["protocol"]
    decisions = data["decisions"]
    graph_height_span = data["graph_height_span"]
    execution_height_span = data["execution_height_span"]
    height_note = data["height_note"]
    source_paths = [config_path, graph_path, protocol_path]
    for data in methods.values():
        source_paths.extend(data["files"].values())
    diagnostics = {
        "schema_version": 1, "run_directory": str(run), "task_index": task_index,
        "episode_id": episode_id, "cml_dimension": config["cml_dimension"],
        "primitive_action_budget": config["max_steps"], "protocol": protocol,
        "methods": {method: data["stats"] for method, data in methods.items()},
        "cml_graph_choices": decisions,
        "projection": {"axes": ["world_x", "world_z"], "graph_height_span_metres": graph_height_span,
                       "execution_height_span_metres": execution_height_span, "height_note": height_note},
        "heading_arrows": "Rotate body forward (0,0,-1) with normalized agent-to-world rotation_xyzw; project to x/z.",
        "interpretation_limits": [
            "Observed abstract graph lines are not floor-plan walls, free-space boundaries, or executed trajectories.",
            "Graph-choice arrows show selected directed edges; physical routes come only from execution.jsonl.",
            "CML scores and W/V branches, when recorded, are in planning.jsonl; this plot does not evaluate them.",
            "Position-only X/Z projection can overlap trajectories at different heights.",
        ],
        "source_sha256": {str(path.relative_to(run)): digest(path) for path in source_paths},
        "plot_script_sha256": digest(Path(__file__)),
        "artifacts": artifacts,
        "implementation": report_provenance(Path(__file__), output),
    }
    save_json(Path(artifacts["diagnostics_json"]), diagnostics)
    return diagnostics


def gcml_report(run: Path, output: Path | None = None) -> Path:
    run = bounded_output(run)
    output = bounded_output(output or run / "analysis")
    if output.exists():
        raise FileExistsError("GCML report already exists; choose a new --output-dir")
    data = load_gcml_comparison(run)
    output.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    for i, task in enumerate(data["tasks"]):
        render_gcml_task(output, i, str(task["source_episode_id"]), data["loaded_tasks"][i],
                         data["methods"], data["positions"], data["edges"])
    implementation = report_provenance(Path(__file__), output)
    after = {path: digest(Path(path)) for path in data["input_sha256"]}
    if after != data["input_sha256"]:
        raise RuntimeError("Navigation inputs changed while reporting")
    implementation.update(input_hashes_before=data["input_sha256"], input_hashes_after=after,
                          inputs_unchanged=True)
    write_json(output / "diagnostics.json", data["rows"])
    save_json(output / "report_implementation.json", implementation)
    destination = output / "report.md"
    destination.write_text(gcml_markdown(run, data, output), encoding="utf-8")
    return destination


def subgoal_report(run: Path, output: Path | None = None) -> Path:
    run = bounded_output(run)
    output = bounded_output(output if output is not None else run / "subgoal_analysis")
    data = load_subgoal_comparison(run)
    tasks = data["tasks"]
    rows = data["rows"]
    loaded_tasks = data["loaded_tasks"]
    graph_positions = data["graph_positions"]
    edges = data["edges"]
    aggregates = data["aggregates"]
    baseline_checks = data["baseline_checks"]
    checkpoint = data["checkpoint"]
    config = data["config"]
    inputs = data["inputs"]
    plots = []
    if output.exists():
        raise ValueError("Choose a new output directory; existing reports are preserved")
    output.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    for index, task in enumerate(tasks):
        task_rows = {row["condition"]: row for row in rows if row["task_index"] == index}
        plots.append(render_subgoal_task(output, index, str(task["source_episode_id"]), loaded_tasks[index],
                                 task_rows, graph_positions, edges, run.name))
    comparison = {
        "schema": "cmnav.gcml_subgoal_comparison.v1", "run": str(run), "conditions": list(CONDITIONS),
        "scope": "privileged_teacher_subgoals_frozen_gcml_same_scene_development_diagnostic",
        "no_new_model_calls": True,
        "training_performed": False, "all_parameters_unchanged": True,
        "baseline_controls_reproduced": len(baseline_checks),
        "source_run": checkpoint["source_run"],
        "shared_settings": {key: config[key] for key in ("max_steps", "gcml_candidates", "gcml_horizon", "gcml_noise_scale",
                                                         "gcml_affordance_threshold", "gcml_association_radius",
                                                         "gcml_association_margin", "gcml_cycle_policy")},
        "model_sha256": digest(inputs["model"]), "graph_sha256": digest(inputs["graph"]),
        "teacher_plan_sha256": digest(inputs["teacher_subgoal_plan"]),
        "task_manifest_sha256": digest(inputs["supported_tasks"]),
        "stage_denominator_definitions": {
            "completed_over_attempted": "conditional on stages actually reached; later stages can be missing because an earlier one failed",
            "completed_over_total": "all frozen planned stages, including unattempted future stages",
            "final_goal_only": "not applicable; do not invent a teacher stage"},
        "aggregates": aggregates, "tasks": rows, "route_figures": list(map(str, plots)),
        "inputs": {name: {"path": str(path), "sha256": digest(path)} for name, path in inputs.items()},
        "report_script_sha256": digest(Path(__file__)),
        "implementation": report_provenance(Path(__file__), output),
    }
    save_json(output / "comparison.json", comparison)
    destination = output / "report.md"
    destination.write_text(subgoal_markdown(run, data, comparison, output, plots), encoding="utf-8")
    return destination
