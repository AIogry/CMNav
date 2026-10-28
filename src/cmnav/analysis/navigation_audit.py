"""Prepare a saved CML navigation audit without plotting or model calls."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
from cmnav.analysis.records import read_json, read_tracked, digest
from cmnav.analysis.navigation_records import load_method, graph_decisions

METHODS = ("cml", "dijkstra", "oracle_follower")

def load_navigation_audit(run: Path, task_index: int) -> dict[str, Any]:
    task = f"task_{task_index:04d}"
    config_path, graph_path, protocol_path = run / "config.json", run / "graph.json", run / "protocol.json"
    config, graph, protocol = read_json(config_path), read_json(graph_path), read_json(protocol_path)
    graph_positions = np.asarray(graph["positions"], dtype=float)
    edges = graph["edges"]
    if graph_positions.ndim != 2 or graph_positions.shape[1] != 3 or not np.isfinite(graph_positions).all():
        raise ValueError("Observed graph positions must be a finite Nx3 array.")
    for edge in edges:
        if len(edge) != 2 or any(type(node) is not int or not 0 <= node < len(graph_positions) for node in edge):
            raise ValueError("Observed graph has invalid edge endpoints.")
    methods = {method: load_method(run, task, method) for method in METHODS
               if (run / "evaluation" / method / task / "summary.json").is_file()}
    if "cml" not in methods:
        raise ValueError("The selected task must have CML execution logs.")
    cml = methods["cml"]
    episode_id = cml["stats"]["episode_id"]
    goal = np.asarray(cml["execution"][0]["goal_position"], dtype=float)
    start = cml["positions"][0]
    for method, data in methods.items():
        if data["stats"]["episode_id"] != episode_id or not np.allclose(data["positions"][0], start, atol=1e-5):
            raise ValueError(f"{method}: task identity or initial position differs from CML.")
        if not np.allclose(data["execution"][0]["goal_position"], goal, atol=1e-5):
            raise ValueError(f"{method}: final task goal differs from CML.")
    decisions = graph_decisions(cml["planning"], edges)
    pair = decisions["dominant_node_pair"]
    focus = graph_positions[pair][:, [0, 2]] if pair else cml["positions"][-1:, [0, 2]]
    span = max(float(np.ptp(focus, axis=0).max()), 0.2)
    padding = max(0.28, span * 0.8)
    bounds = (float(focus[:, 0].min() - padding), float(focus[:, 0].max() + padding),
              float(focus[:, 1].min() - padding), float(focus[:, 1].max() + padding))
    graph_height_span = float(np.ptp(graph_positions[:, 1]))
    execution_height_span = max(data["stats"]["height_max_metres"] for data in methods.values()) - min(
        data["stats"]["height_min_metres"] for data in methods.values())
    height_note = ("WARNING: vertical span exceeds 0.5 m; X/Z projection can overlap different heights."
                   if max(graph_height_span, execution_height_span) > 0.5
                   else f"Vertical spans: graph {graph_height_span:.2f} m; executed routes {execution_height_span:.2f} m.")

    return {"run": run, "task_index": task_index, "task": task, "config_path": config_path,
            "graph_path": graph_path, "protocol_path": protocol_path, "config": config,
            "graph": graph, "protocol": protocol, "graph_positions": graph_positions, "edges": edges,
            "methods": methods, "cml": cml, "episode_id": episode_id, "goal": goal, "start": start,
            "decisions": decisions, "pair": pair, "bounds": bounds, "height_note": height_note,
            "graph_height_span": graph_height_span, "execution_height_span": execution_height_span}


def load_gcml_comparison(run: Path) -> dict[str, Any]:
    inputs: dict[str, str] = {}
    config = read_tracked(run / "config.json", inputs)
    summary = read_tracked(run / "summary.json", inputs)
    if summary["status"] != "completed":
        raise ValueError("Comparison must be completed before final reporting")
    graph = read_tracked(run / "graph.json", inputs)
    tasks = read_tracked(run / "supported_tasks.json", inputs)["payload"]["evaluation"]
    methods = config["methods"]
    positions = np.asarray(graph["positions"])
    edges = graph["edges"]
    rows, loaded_tasks = [], []
    for i, task in enumerate(tasks):
        for method in methods:
            directory = run / "evaluation" / method / f"task_{i:04d}"
            for name in ("summary.json", "execution.jsonl", "steps.jsonl", "planning.jsonl"):
                path = (directory / name).resolve()
                inputs[str(path)] = digest(path)
        loaded = {method: load_method(run, f"task_{i:04d}", method) for method in methods}
        first = loaded[methods[0]]
        start = first["positions"][0]
        goal = np.asarray(first["execution"][0]["goal_position"])
        for method in methods:
            data = loaded[method]
            if (data["stats"]["episode_id"] != str(task["source_episode_id"])
                    or not np.allclose(data["positions"][0], start)
                    or not np.allclose(data["execution"][0]["goal_position"], goal)):
                raise ValueError("Comparison task identity/start/goal differs across methods")
            rows.append({"task_index": i, "method": method, **data["stats"],
                         "graph_decisions": graph_decisions(data["planning"], edges),
                         "gcml_planning": data["summary"].get("gcml_planning")})
        loaded_tasks.append(loaded)
    return {"config": config, "summary": summary, "graph": graph, "tasks": tasks,
            "methods": methods, "positions": positions, "edges": edges,
            "rows": rows, "loaded_tasks": loaded_tasks, "input_sha256": inputs}
