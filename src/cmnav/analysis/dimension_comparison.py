"""Readout and shortest-route diagnostics for an explicitly scoped dimension comparison."""
from __future__ import annotations

from collections import Counter
import gzip
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any
import numpy as np
import torch
from cmnav.memory.oracle_graph import OracleObservedGraph
from cmnav.runtime.artifacts import sha256_file
from cmnav.analysis.records import read_json as load
from cmnav.analysis.graph_diagnostics import reverse_distances as distances_to



def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "min": float(array.min()), "mean": float(array.mean()),
            "median": float(np.median(array)), "max": float(array.max())}


def classify(scores: np.ndarray, labels: np.ndarray, nodes: list[int], threshold: float) -> dict[str, Any]:
    selected, truth = scores[:, nodes], labels[:, nodes]
    prediction = selected >= threshold
    fp = prediction & ~truth
    fn = ~prediction & truth
    errors = selected.astype(np.float64) - truth.astype(np.float64)
    return {
        "states": len(nodes), "state_ids": nodes, "threshold": threshold,
        "true_positive": int(np.count_nonzero(prediction & truth)),
        "false_positive": int(np.count_nonzero(fp)), "false_negative": int(np.count_nonzero(fn)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))) if errors.size else None,
        "max_absolute_error": float(np.abs(errors).max()) if errors.size else None,
        "empty_prediction_states": int(np.count_nonzero(prediction.sum(axis=0) == 0)),
        "states_with_false_positive": int(np.count_nonzero(fp.any(axis=0))),
        "positive_min": float(selected[truth].min()) if truth.any() else None,
        "negative_max": float(selected[~truth].max()) if (~truth).any() else None,
        "false_positive_by_state": [{"state": node, "count": int(fp[:, index].sum())}
                                    for index, node in enumerate(nodes)],
    }


def summarize_choices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    branching = [row for row in rows if row["out_degree"] > 1]
    competing = [row for row in rows if row["best_correct_minus_best_other"] is not None]
    return {
        "states": len(rows), "correct": sum(row["legal_argmax_is_cost_optimal"] for row in rows),
        "accuracy": sum(row["legal_argmax_is_cost_optimal"] for row in rows) / len(rows) if rows else None,
        "branching_states": len(branching),
        "branching_correct": sum(row["legal_argmax_is_cost_optimal"] for row in branching),
        "branching_accuracy": sum(row["legal_argmax_is_cost_optimal"] for row in branching) / len(branching) if branching else None,
        "states_with_nonoptimal_alternative": len(competing),
        "normalized_gap": describe([row["best_correct_minus_best_other"] for row in competing]),
        "gap_over_noise_sigma": describe([row["gap_over_noise_sigma"] for row in competing if row["gap_over_noise_sigma"] is not None]),
        "gap_over_pairwise_noise_sd": describe([row["gap_over_pairwise_noise_sd"] for row in competing if row["gap_over_pairwise_noise_sd"] is not None]),
        "negative_gap_states": sum(row["best_correct_minus_best_other"] < 0 for row in competing),
        "positive_gap_below_sigma_states": sum(row["gap_over_noise_sigma"] is not None and 0 < row["gap_over_noise_sigma"] < 1 for row in competing),
    }


def long_goal_scores(q: torch.Tensor, w: torch.Tensor, graph: OracleObservedGraph,
                     tasks: dict[str, Any], noise: float) -> dict[str, Any]:
    edges, costs = graph.edges, graph.edge_mean_costs
    reverse: list[list[tuple[int, float]]] = [[] for _ in graph.positions]
    for (source, target), cost in zip(edges, costs):
        reverse[target].append((source, cost))
    task_results, all_rows = [], []
    for index, task in enumerate(tasks["payload"]["evaluation"]):
        coverage = task["coverage"]
        start, goal = coverage["start_node"], coverage["goal_node"]
        path = coverage["shortest_path_edges"]
        nodes, current = [], start
        for edge in path:
            if edges[edge][0] != current:
                raise ValueError("Frozen reference path is discontinuous")
            nodes.append(current)
            current = edges[edge][1]
        if current != goal:
            raise ValueError("Frozen reference path does not end at its goal")
        distances = distances_to(goal, reverse)
        if not math.isclose(sum(costs[edge] for edge in path), distances[start], rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Frozen reference path is not cost-optimal")
        utilities = w @ (q[:, goal, None] - q[:, nodes])
        norms = torch.linalg.vector_norm(utilities, dim=0)
        normalized = utilities / norms.clamp_min(1e-12)[None, :]
        rows = []
        for offset, state in enumerate(nodes):
            legal = graph.outgoing(state)
            correct = [edge for edge in legal if math.isclose(costs[edge] + distances[edges[edge][1]],
                                                             distances[state], rel_tol=1e-6, abs_tol=1e-6)]
            if not correct:
                raise ValueError("Reference node has no cost-optimal outgoing edge")
            others = [edge for edge in legal if edge not in correct]
            scores = normalized[:, offset]
            selected = legal[int(torch.argmax(scores[legal]).item())]
            best_correct = float(scores[correct].max().item())
            best_other = float(scores[others].max().item()) if others else None
            gap = best_correct - best_other if best_other is not None else None
            row = {"task_index": index, "episode_id": str(task["source_episode_id"]),
                   "reference_path_offset": offset, "state": state, "goal": goal,
                   "remaining_optimal_cost": distances[state], "out_degree": len(legal),
                   "legal_edges": legal, "cost_optimal_edges": correct, "legal_argmax_edge": selected,
                   "legal_argmax_is_cost_optimal": selected in correct,
                   "utility_all_actions_l2": float(norms[offset].item()),
                   "legal_normalized_utilities": [float(scores[edge].item()) for edge in legal],
                   "best_correct_minus_best_other": gap,
                   "gap_over_noise_sigma": gap / noise if gap is not None and noise > 0 else None,
                   "gap_over_pairwise_noise_sd": gap / (math.sqrt(2) * noise) if gap is not None and noise > 0 else None}
            rows.append(row)
        all_rows.extend(rows)
        task_results.append({"task_index": index, "episode_id": str(task["source_episode_id"]),
                             "start_node": start, "goal_node": goal, "reference_edges": len(path),
                             "summary": summarize_choices(rows), "states": rows})
    return {"scope": "offline_reference_path_labels_never_planner_feedback", "noise_sigma": noise,
            "normalization": "L2_over_all_action_ids", "allows_equal_cost_shortest_edges": True,
            "shortest_cost_equality_tolerance": {"relative": 1e-6, "absolute": 1e-6},
            "no_competing_nonoptimal_edge_gap": None,
            "summary": summarize_choices(all_rows), "tasks": task_results}


def first_candidates(task_dir: Path, method: str) -> dict[str, Any]:
    if method in {"gcml_greedy", "dijkstra"}:
        return {"applicable": False, "reason": "policy_does_not_generate_imagined_candidates"}
    path = task_dir / "gcml_candidates.jsonl.gz"
    lengths, sequences, terminations = [], set(), Counter()
    first_decision = None
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if first_decision is None:
                first_decision = row["decision_id"]
            if row["decision_id"] != first_decision:
                break
            candidate = row["candidate"]
            lengths.append(len(candidate["actions"]))
            sequences.add(tuple(candidate["actions"]))
            terminations[candidate["termination"]] += 1
    first_choice = None
    with (task_dir / "planning.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") == "graph_decision":
                first_choice = {key: row.get(key) for key in
                                ("current_node", "goal_node", "candidate_edges", "selected_edge")}
                break
    return {"applicable": True, "decision_id": first_decision, "lengths": describe(lengths),
            "distinct_action_sequences": len(sequences), "terminations": dict(terminations),
            "first_graph_decision": first_choice}


def load_dimension_comparison(run: Path, contract_path: Path | None = None) -> tuple[dict[str, Any], set[Path], dict[str, str], float]:
    run = run.expanduser().resolve()
    protocol = load(run / "protocol.json")
    dimensions, methods = protocol["dimensions"], protocol["methods"]
    if (not isinstance(dimensions, list) or not dimensions
            or any(type(dim) is not int or dim <= 0 for dim in dimensions)
            or len(set(dimensions)) != len(dimensions)):
        raise ValueError("Protocol dimensions must be nonempty unique positive integers")
    contract = load(contract_path) if contract_path is not None else None
    if contract is not None:
        if contract.get("schema") != "cmnav.dimension_report_contract.v1":
            raise ValueError("Unsupported dimension report contract")
        if dimensions != contract["dimensions"]:
            raise ValueError("Declared dimensions differ from the explicit report contract")
    task_manifest = load(run / f"dimension_{dimensions[0]}" / "supported_tasks.json")
    task_count = len(task_manifest["payload"]["evaluation"])
    if task_count <= 0:
        raise ValueError("Dimension comparison must contain a nonempty frozen task cohort")
    if any(method not in {"gcml", "gcml_graph_affordance", "gcml_greedy", "dijkstra"} for method in methods):
        raise ValueError("Unknown method in protocol")
    tracked = {run / "protocol.json"}
    if contract_path is not None:
        tracked.add(contract_path.resolve())
    for dim in dimensions:
        for folder in (f"dimension_{dim}", f"evaluation_{dim}"):
            if load(run / folder / "summary.json").get("status") != "completed":
                raise ValueError(f"{folder} must be completed before reporting")
        for name in ("config", "training", "model", "graph", "summary", "supported_tasks", "training_walks", "algorithm_reference", "reproduction"):
            tracked.add(run / f"dimension_{dim}" / f"{name}.json")
        for name in ("config", "evaluation", "summary", "checkpoint_source"):
            tracked.add(run / f"evaluation_{dim}" / f"{name}.json")
        for method in methods:
            for index in range(task_count):
                task_dir = run / f"evaluation_{dim}" / "evaluation" / method / f"task_{index:04d}"
                tracked.update((task_dir / "summary.json", task_dir / "planning.jsonl"))
                if method in {"gcml", "gcml_graph_affordance"}:
                    tracked.add(task_dir / "gcml_candidates.jsonl.gz")
    control_dir = run / "renderer_control"
    control_summary = load(control_dir / "summary.json")
    if control_summary.get("status") != "completed":
        raise ValueError("Renderer control must complete before reporting")
    control_checks = load(control_dir / "control_comparisons.json")
    tracked.update((control_dir / "summary.json", control_dir / "control_comparisons.json"))
    before = {str(path): sha256_file(path) for path in sorted(tracked)}
    torch.set_num_threads(1)
    started = perf_counter()
    result: dict[str, Any] = {
        "schema": "cmnav.gcml_dimension_report.v1", "protocol": protocol,
        "dimensions": dimensions, "methods": methods,
        "analysis_only": True, "training_performed": False, "planner_called": False,
        "renderer_control": {"summary": control_summary, "checks": control_checks,
                             "tasks": len(control_checks), "matching_tasks": sum(row["all_match"] for row in control_checks),
                             "all_match": bool(control_checks) and all(row["all_match"] for row in control_checks)},
        "results": {}, "input_hashes_before": before,
    }
    fingerprints: dict[str, str] = {}
    renderer_devices = set()
    shared_settings = None
    shared_training_config = None
    shared_evaluation_config = None
    planning_settings = None
    for dim in dimensions:
        training_dir, evaluation_dir = run / f"dimension_{dim}", run / f"evaluation_{dim}"
        config, evaluation_config = load(training_dir / "config.json"), load(evaluation_dir / "config.json")
        if config["cml_dimension"] != dim or evaluation_config["cml_dimension"] != dim:
            raise ValueError("Dimension differs from its directory/protocol")
        comparable_training = {key: value for key, value in config.items() if key not in {"cml_dimension", "output_dir"}}
        comparable_evaluation = {key: value for key, value in evaluation_config.items() if key not in {"cml_dimension", "output_dir"}}
        if shared_training_config is not None and comparable_training != shared_training_config:
            raise ValueError("Training configuration changes more than dimension/output across the comparison")
        if shared_evaluation_config is not None and comparable_evaluation != shared_evaluation_config:
            raise ValueError("Evaluation configuration changes more than dimension/output across the comparison")
        shared_training_config, shared_evaluation_config = comparable_training, comparable_evaluation
        for name in ("graph", "training_walks", "supported_tasks"):
            fingerprint = before[str(training_dir / f"{name}.json")]
            if name in fingerprints and fingerprints[name] != fingerprint:
                raise ValueError(f"Dimensions do not share identical {name}")
            fingerprints[name] = fingerprint
        checkpoint = load(evaluation_dir / "checkpoint_source.json")
        if checkpoint["model_sha256"] != before[str(training_dir / "model.json")]:
            raise ValueError("Evaluation checkpoint does not match its dimension training run")
        renderer_devices.add(evaluation_config["base"]["environment"]["gpu_device_id"])
        training, training_summary = load(training_dir / "training.json"), load(training_dir / "summary.json")
        model = load(training_dir / "model.json")
        graph = OracleObservedGraph.from_state_dict(load(training_dir / "graph.json"))
        tasks = load(training_dir / "supported_tasks.json")
        walks = load(training_dir / "training_walks.json")
        settings = {
            "node_count": len(graph.positions), "edge_count": len(graph.edges), "task_count": task_count,
            "seed": config["seed"], "epochs": training["epochs_completed"],
            "trajectory_batches_per_epoch": training["trajectories_per_epoch"],
            "transitions_per_epoch": training["transitions_per_epoch"],
        }
        planning = {key: evaluation_config[key] for key in (
            "gcml_candidates", "gcml_horizon", "gcml_noise_scale", "gcml_affordance_threshold",
            "gcml_association_radius", "gcml_association_margin", "gcml_cycle_policy", "max_steps")}
        for key in planning:
            if key in config and config[key] != planning[key]:
                raise ValueError(f"Training/evaluation planning settings differ: {key}")
        if shared_settings is not None and settings != shared_settings:
            raise ValueError("Dimensions do not share graph size, task count, seed and training budget")
        if planning_settings is not None and planning != planning_settings:
            raise ValueError("Dimensions do not share planning settings and action budget")
        shared_settings, planning_settings = settings, planning
        if contract is not None:
            for key, expected in contract.get("shared_settings", {}).items():
                if settings.get(key) != expected:
                    raise ValueError(f"Historical dimension contract differs at {key}")
            for key, expected in contract.get("planning_settings", {}).items():
                if planning.get(key) != expected:
                    raise ValueError(f"Historical planning contract differs at {key}")
        q, w, g = (torch.tensor(model[name], dtype=torch.float32) for name in ("Q", "W", "G"))
        del model
        truth = np.zeros((len(graph.edges), len(graph.positions)), dtype=bool)
        for edge, (source, _) in enumerate(graph.edges):
            truth[edge, source] = True
        supervised = sorted({transition[0] for row in walks["trajectories"] for transition in row["transitions"]})
        sinks = [state for state in range(len(graph.positions)) if not graph.outgoing(state)]
        raw = (g @ q).numpy()
        threshold = config["gcml_affordance_threshold"]
        grouped = {"supervised_sources": classify(raw, truth, supervised, threshold),
                   "graph_sinks": classify(raw, truth, sinks, threshold),
                   "all_states": classify(raw, truth, list(range(len(graph.positions))), threshold)}
        singular = np.linalg.svd(q.numpy().astype(np.float64), compute_uv=False)
        tolerance = max(q.shape) * np.finfo(np.float64).eps * singular[0]
        svd = {"dtype": "float64", "rows": q.shape[0], "columns": q.shape[1],
               "rank": int(np.count_nonzero(singular > tolerance)), "rank_tolerance": float(tolerance),
               "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
               "singular_values": singular.tolist(), "condition_definition": "largest / smallest singular value"}
        rates = {name: {"minimum": min(row["effective_readout_rate_min"][name] for row in training["epochs_detail"]),
                        "maximum": max(row["effective_readout_rate_max"][name] for row in training["epochs_detail"]),
                        "capped_batches": sum(row["capped_readout_batches"][name] for row in training["epochs_detail"])}
                 for name in ("W", "G")}
        evaluation_rows = load(evaluation_dir / "evaluation.json")
        if len(evaluation_rows) != task_count * len(methods):
            raise ValueError("Evaluation must include every method and frozen task")
        observed_pairs = set()
        task_rows = []
        for row in evaluation_rows:
            index, method = row["task_index"], row["method"]
            if (type(index) is not int or not 0 <= index < task_count
                    or method not in methods or (index, method) in observed_pairs):
                raise ValueError("Unexpected or duplicate evaluation row")
            observed_pairs.add((index, method))
            expected = tasks["payload"]["evaluation"][index]
            if (str(row["episode_id"]) != str(expected["source_episode_id"])
                    or row["start_node"] != expected["coverage"]["start_node"]
                    or row["goal_node"] != expected["coverage"]["goal_node"]):
                raise ValueError("Evaluation task differs from the frozen supported task")
            if row.get("teacher_subgoals") is not None:
                raise ValueError("Dimension full-goal comparison cannot include teacher subgoals")
            task_dir = evaluation_dir / "evaluation" / method / f"task_{index:04d}"
            task_rows.append({**row, "first_candidates": first_candidates(task_dir, method)})
        task_rows.sort(key=lambda row: (row["task_index"], methods.index(row["method"])))
        result["results"][str(dim)] = {
            "training_directory": str(training_dir), "evaluation_directory": str(evaluation_dir),
            "training_config": config, "evaluation_config": evaluation_config,
            "training": training, "training_summary": training_summary,
            "training_reproduction": load(training_dir / "reproduction.json"),
            "evaluation_summary": load(evaluation_dir / "summary.json"),
            "readout": grouped, "Q_svd": svd, "readout_rates": rates,
            "long_goal_W": long_goal_scores(q, w, graph, tasks, config["gcml_noise_scale"]),
            "tasks": task_rows,
        }
        if "algorithm_reference" not in result:
            result["algorithm_reference"] = load(training_dir / "algorithm_reference.json")
        del q, w, g
    if len(renderer_devices) != 1:
        raise ValueError("Dimensions were evaluated with different renderer devices")
    result["shared_settings"] = shared_settings
    result["planning_settings"] = planning_settings
    result["report_contract"] = {"path": str(contract_path.resolve()), "value": contract} if contract_path is not None else None
    result["renderer_device"] = next(iter(renderer_devices))
    result["shared_input_hashes"] = fingerprints
    return result, tracked, before, started
