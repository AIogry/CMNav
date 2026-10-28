"""Compare frozen CML training coverage and saved navigation decisions.

The focused diagnostic pair is an explicit contract. Shared-pair labels remain
exhaustive directed-graph diagnostics and never become learning inputs.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from cmnav.memory.oracle_graph import OracleObservedGraph
from cmnav.methods.cml import CognitiveMapLearner
from cmnav.runtime.provenance import runtime_versions, source_hashes
from cmnav.analysis.records import (digest, canonical_digest, read_json, read_tracked as read,
                                    save_new_json as write_json, json_line, capture_analysis_sources, project_outputs_root)
from cmnav.analysis.statistics import finite, distribution
from cmnav.analysis.graph_diagnostics import all_goal_distances

BINS = ("1", "2-4", "5-8", ">8")
COST_ABS_TOLERANCE = 1e-6
COST_REL_TOLERANCE = 1e-6

def hop_bin(hops):
    return "1" if hops == 1 else "2-4" if hops <= 4 else "5-8" if hops <= 8 else ">8"

def diagnostic_pairs(graph, weighted, unweighted):
    pairs = []
    edges, costs = graph.edges, graph.edge_mean_costs
    for source in range(len(graph.positions)):
        candidates = sorted(graph.outgoing(source))
        if len(candidates) < 2:
            continue
        for goal in range(len(graph.positions)):
            hops = unweighted[goal][source]
            if source == goal or hops is None:
                continue
            best = weighted[goal][source]
            candidate_costs = [costs[a] + weighted[goal][edges[a][1]] for a in candidates]
            optimal = [a for a, cost in zip(candidates, candidate_costs)
                       if math.isclose(cost, best, rel_tol=COST_REL_TOLERANCE,
                                       abs_tol=COST_ABS_TOLERANCE)]
            if not optimal:
                raise ValueError("Reachable pair has no optimal outgoing edge")
            pairs.append({"pair_id": len(pairs), "source_node": source, "goal_node": goal,
                          "shortest_hops": hops, "hop_bin": hop_bin(hops),
                          "candidate_edges": candidates, "optimal_edges": optimal,
                          "optimal_graph_cost_m": best,
                          "candidate_total_cost_m": [finite(v) for v in candidate_costs]})
    return pairs

def load_tasks(run, config, inputs):
    tasks = []
    for method in config["methods"]:
        directories = sorted((run / "evaluation" / method).glob("task_*"))
        if len(directories) != config["evaluation_episodes"]:
            raise ValueError(f"Missing completed task summaries for {run.name}/{method}")
        for directory in directories:
            summary = read(directory / "summary.json", inputs)
            planning = read(directory / "planning.jsonl", inputs, lines=True)
            tasks.append({"method": method, "task": directory.name,
                          "episode_id": str(summary["episode_id"]), "summary": summary,
                          "graph_decisions": [x for x in planning if x.get("kind") == "graph_decision"]})
    return tasks

def edge_counts(walks, graph):
    edges = graph.edges
    counts = [0] * len(edges)
    repeated_targets = repeated_actions = nonempty = transitions = 0
    for trajectory in walks["trajectories"]:
        triples = trajectory["transitions"]
        nonempty += bool(triples)
        repeated_targets += len(triples) != len({x[2] for x in triples})
        repeated_actions += len(triples) != len({x[1] for x in triples})
        previous = trajectory["start_node"]
        for source, edge, target in triples:
            if source != previous or edges[edge] != (source, target):
                raise ValueError("Saved replay has an invalid/discontinuous graph transition")
            previous = target
            counts[edge] += 1
            transitions += 1
    if counts != walks["coverage"]["edge_visit_counts"]:
        raise ValueError("Saved replay edge counts disagree with its transitions")
    return counts, {"sampled_trajectories": len(walks["trajectories"]),
                    "nonempty_trajectories": nonempty, "transitions_per_epoch": transitions,
                    "repeated_target_trajectories": repeated_targets,
                    "repeated_action_trajectories": repeated_actions,
                    "edges": len(counts), "covered_edges": sum(v > 0 for v in counts),
                    "unseen_edge_ids": [a for a, v in enumerate(counts) if not v],
                    "visits_per_edge_per_epoch": distribution(counts),
                    "visit_count_histogram": dict(sorted(Counter(counts).items()))}

def candidate_values(model, graph, state, goal, counts, weighted):
    delta = model.Q[:, goal] - model.Q[:, state]
    candidates = sorted(graph.outgoing(state))
    w = (model.W[candidates] @ delta).tolist()
    v = (model.V[:, candidates].T @ delta).tolist()
    rows = []
    for edge, ws, vs in zip(candidates, w, v):
        target = graph.edges[edge][1]
        rows.append({"edge_id": edge, "source_node": state, "target_node": target,
                     "w_dot": ws, "v_dot": vs,
                     "w_norm": float(torch.linalg.vector_norm(model.W[edge])),
                     "v_norm": float(torch.linalg.vector_norm(model.V[:, edge])),
                     "successor_q_goal_distance": float(torch.linalg.vector_norm(model.Q[:, goal] - model.Q[:, target])),
                     "predicted_q_plus_v_goal_distance": float(torch.linalg.vector_norm(delta - model.V[:, edge])),
                     "transition_residual_norm": float(torch.linalg.vector_norm(model.Q[:, target] - model.Q[:, state] - model.V[:, edge])),
                     "visits_per_epoch": counts[edge],
                     "target_can_reach_goal": math.isfinite(weighted[goal][target]),
                     "graph_total_cost_m": finite(graph.edge_mean_costs[edge] + weighted[goal][target])})
    return rows

def update_metrics(metrics, selected, pair):
    metrics["pairs"] += 1
    metrics["optimal_choices"] += selected in pair["optimal_edges"]
    total = pair["candidate_total_cost_m"][pair["candidate_edges"].index(selected)]
    if total is None:
        metrics["unreachable_choices"] += 1
    else:
        metrics["finite_excess_costs"].append(max(0.0, total - pair["optimal_graph_cost_m"]))

def choice_diagnostics(model, pairs, output):
    groups = {method: {key: {"pairs": 0, "optimal_choices": 0, "unreachable_choices": 0,
                             "finite_excess_costs": []} for key in (*BINS, "all")}
              for method in ("cml_history_empty", "w_argmax", "v_argmax")}
    branches = Counter()
    with gzip.open(output, "xt", encoding="utf-8") as stream:
        for pair in pairs:
            state, goal, candidates = pair["source_node"], pair["goal_node"], pair["candidate_edges"]
            selected, decision = model.choose_edge_with_diagnostics(state, goal, candidates, ())
            delta = model.Q[:, goal] - model.Q[:, state]
            w = (model.W[candidates] @ delta).tolist()
            v = (model.V[:, candidates].T @ delta).tolist()
            choices = {"cml_history_empty": selected,
                       "w_argmax": candidates[max(range(len(w)), key=w.__getitem__)],
                       "v_argmax": candidates[max(range(len(v)), key=v.__getitem__)]}
            branches[decision["branch"]] += 1
            for method, edge in choices.items():
                for key in (pair["hop_bin"], "all"):
                    update_metrics(groups[method][key], edge, pair)
            total = pair["candidate_total_cost_m"][candidates.index(selected)]
            stream.write(json_line({"pair_id": pair["pair_id"], "w_scores": w, "v_scores": v,
                                    "choices": choices, "cml_branch": decision["branch"],
                                    "chosen_is_optimal": selected in pair["optimal_edges"],
                                    "chosen_target_reachable": total is not None,
                                    "chosen_excess_cost_m": None if total is None else max(0.0, total - pair["optimal_graph_cost_m"])}))
    for bins in groups.values():
        for metrics in bins.values():
            count = metrics["pairs"]
            metrics["accuracy"] = metrics["optimal_choices"] / count if count else None
            metrics["unreachable_choice_rate"] = metrics["unreachable_choices"] / count if count else None
            metrics["excess_cost_m_conditional_on_reachable_choice"] = distribution(metrics.pop("finite_excess_costs"))
    return {"methods": groups, "cml_branch_counts": dict(branches)}

def analyze_run(run, reference, graph, weighted, pairs, output, current_hashes, inputs, focus):
    run_summary = read(run / "summary.json", inputs)
    if run_summary.get("status") != "completed":
        raise ValueError(f"Only completed runs can be compared: {run}")
    config = read(run / "config.json", inputs)
    protocol = read(run / "protocol.json", inputs)
    saved_hashes = read(run / "source_hashes.json", inputs)
    if protocol["map_mode"] != "frozen":
        raise ValueError("Final-checkpoint diagnostics require frozen-map runs")
    if saved_hashes.get("methods/cml.py") != current_hashes["methods/cml.py"]:
        raise ValueError(f"Learner source differs from saved run {run.name}; faithful replay unavailable")
    state = read(run / "model.json", inputs)
    model = CognitiveMapLearner.from_state_dict(state, device="cpu")
    if model.edges != tuple(graph.edges):
        raise ValueError("Saved model action IDs differ from the shared graph")
    training = read(run / "training.json", inputs)
    walks = read(run / "training_walks.json", inputs)
    counts, coverage = edge_counts(walks, graph)
    epochs = training["epochs"]
    if coverage["transitions_per_epoch"] != training["transitions_per_epoch"]:
        raise ValueError("Training log disagrees with replay transition count")
    tasks = load_tasks(run, config, inputs)
    per_edge = []
    for edge, (source, target) in enumerate(graph.edges):
        transition = model.Q[:, target] - model.Q[:, source]
        residual = float(torch.linalg.vector_norm(transition - model.V[:, edge]))
        norm = float(torch.linalg.vector_norm(transition))
        per_edge.append({"edge_id": edge, "source_node": source, "target_node": target,
                         "visits_per_epoch": counts[edge], "consumed_visits_this_fit": counts[edge] * epochs,
                         "residual_norm": residual, "residual_mse": residual ** 2 / model.dimension,
                         "q_transition_norm": norm, "relative_residual_norm": residual / norm if norm else None,
                         "w_norm": float(torch.linalg.vector_norm(model.W[edge])),
                         "v_norm": float(torch.linalg.vector_norm(model.V[:, edge]))})
    write_json(output / "edge_diagnostics.json", per_edge)
    residuals = {key: distribution([x["residual_norm"] for x in per_edge
                                   if key == "all" or (x["visits_per_epoch"] > 0) == (key == "sampled")])
                 for key in ("all", "sampled", "unseen")}
    focus_source, focus_goal = focus["source_node"], focus["goal_node"]
    critical = {"available": max(focus_source, focus_goal) < len(graph.positions),
                "source_node": focus_source, "goal_node": focus_goal}
    if critical["available"]:
        choice, decision = model.choose_edge_with_diagnostics(focus_source, focus_goal, graph.outgoing(focus_source), ())
        critical.update(history_empty_selected_edge=choice, history_empty_decision=decision,
                        candidate_values=candidate_values(model, graph, focus_source, focus_goal, counts, weighted),
                        actual_runtime_decisions=[{"episode_id": task["episode_id"], "decision": item}
                                                  for task in tasks if task["method"] == "cml"
                                                  for item in task["graph_decisions"]
                                                  if item["current_node"] == focus_source and item["goal_node"] == focus_goal])
    route_rows = []
    for route in reference["routes"]:
        route_rows.append({**route, "edges": [{"hop": hop, **per_edge[edge]}
                                             for hop, edge in enumerate(route["selected_edges"])]})
    endpoints = {}
    for method in config["methods"]:
        summaries = [task["summary"] for task in tasks if task["method"] == method]
        supported = [s for s in summaries if s["endpoints_supported"]]
        reachable = [s for s in summaries if s["directed_graph_reachable"]]
        endpoints[method] = {"tasks": len(summaries), "endpoints_supported": len(supported),
                             "directed_graph_reachable": len(reachable),
                             "successes_all": sum(s["metrics"].get("success", 0) > 0 for s in summaries),
                             "successes_reachable": sum(s["metrics"].get("success", 0) > 0 for s in reachable)}
    result = {"run": str(run), "run_id": run.name, "config": config,
              "run_summary": run_summary,
              "replay_trajectories_sha256": canonical_digest(walks["trajectories"]),
              "sampling": walks["sampling"],
              "learner_settings": {key: state.get(key) for key in
                                   ("schema", "algorithm", "dimension", "seed", "learning_rate",
                                    "state_learning_rate", "inverse_learning_rate", "normalize_actions",
                                    "dtype", "backend", "repeated_index_policy")},
              "training": {key: value for key, value in training.items()
                           if key not in ("trajectory_prediction_mse", "walk_coverage")},
              "coverage": coverage, "tasks": tasks, "endpoint_summary": endpoints,
              "residual_norms": residuals, focus["result_key"]: critical,
              "reference_dijkstra_routes": route_rows,
              "choice_diagnostics": choice_diagnostics(model, pairs, output / "choices.jsonl.gz"),
              "source_hashes": saved_hashes,
              "diagnostic_artifacts": {"edges": str(output / "edge_diagnostics.json"),
                                       "choices": str(output / "choices.jsonl.gz")}}
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--focus-contract", type=Path, required=True,
                        help="JSON object: source_node, goal_node, result_key for focused comparison")
    args = parser.parse_args(argv)
    focus = read_json(args.focus_contract.resolve())
    if (not isinstance(focus, dict) or set(focus) != {"source_node", "goal_node", "result_key"}
            or any(type(focus[key]) is not int or focus[key] < 0 for key in ("source_node", "goal_node"))
            or not isinstance(focus["result_key"], str) or not focus["result_key"].startswith("critical_")):
        parser.error("Focus contract requires nonnegative integer source_node/goal_node and a critical_* result_key")
    runs = [path.resolve() for path in args.runs]
    output = args.output.resolve()
    outputs_root = project_outputs_root(args.project_root)
    if output == outputs_root or outputs_root not in output.parents:
        parser.error("--output must be a new directory strictly inside this project's outputs")
    if any(output == run or run in output.parents for run in runs):
        parser.error("--output cannot modify an input run or its descendants")
    if len(set(runs)) != len(runs):
        parser.error("Input runs must be distinct")
    if output.exists():
        parser.error("Output already exists; choose a new directory")
    torch.set_num_threads(1)
    inputs, current_hashes = {str(args.focus_contract.resolve()): digest(args.focus_contract)}, source_hashes()
    graphs = [read(run / "graph.json", inputs) for run in runs]
    plans = [read(run / "plan.json", inputs) for run in runs]
    graph_hashes = [canonical_digest(value) for value in graphs]
    plan_hashes = [canonical_digest(value) for value in plans]
    if len(set(graph_hashes)) != 1:
        parser.error("Shared-pair comparison requires identical observed graphs and edge IDs")
    graph = OracleObservedGraph.from_state_dict(graphs[0])
    config = read(runs[0] / "config.json", inputs)
    reference_tasks = load_tasks(runs[0], config, inputs)
    routes = [{"task": item["task"], "episode_id": item["episode_id"],
               "selected_edges": [x["selected_edge"] for x in item["graph_decisions"] if x["selected_edge"] is not None]}
              for item in reference_tasks if item["method"] == "dijkstra"
              and item["summary"]["metrics"].get("success", 0) > 0]
    weighted, unweighted = all_goal_distances(graph)
    pairs = diagnostic_pairs(graph, weighted, unweighted)
    output.mkdir(parents=True, exist_ok=False)
    pair_file = output / "diagnostic_pairs.jsonl"
    with pair_file.open("x", encoding="utf-8") as stream:
        for pair in pairs:
            stream.write(json_line(pair))
    results = []
    with torch.no_grad():
        for index, run in enumerate(runs):
            print(f"Analyzing {index + 1}/{len(runs)}: {run.name}", flush=True)
            detail = output / f"run_{index:02d}"
            detail.mkdir()
            results.append(analyze_run(run, {"routes": routes}, graph, weighted, pairs,
                                       detail, current_hashes, inputs, focus))
    comparison = {"schema": "cmnav.training_comparison.v1",
                  "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "access": "offline_diagnostic_labels_not_learning_inputs",
                  "identity": {"all_graphs_identical": len(set(graph_hashes)) == 1,
                               "all_plans_identical": len(set(plan_hashes)) == 1,
                               "graph_canonical_sha256": graph_hashes,
                               "plan_canonical_sha256": plan_hashes},
                  "definitions": {"pair_scope": "all ordered reachable nonidentical source-goal pairs with source outdegree >= 2",
                                  "history": "empty for every static diagnostic; actual task history kept in saved graph_decisions",
                                  "hop_bins": "minimum directed edge count, not weighted-shortest-path edge count",
                                  "correct_action": "any candidate minimizing edge mean cost plus shortest remaining weighted cost",
                                  "tie_absolute_tolerance_m": COST_ABS_TOLERANCE,
                                  "tie_relative_tolerance": COST_REL_TOLERANCE,
                                  "unreachable_excess_cost": "infinity represented by null and separately counted; excluded from finite conditional mean",
                                  "residual": "final checkpoint L2(Q_target-Q_source-V_edge)",
                                  "visits_total": "per-epoch appearances multiplied by epochs in this fit; not independent updates or unique samples"},
                  "diagnostic_pairs": {"path": str(pair_file), "sha256": digest(pair_file),
                                       "count": len(pairs), "sampling": "exhaustive_no_cap",
                                       "bin_counts": dict(Counter(x["hop_bin"] for x in pairs))},
                  "reference_dijkstra_routes": routes, "runs": results, "focus_contract": focus,
                  "training_controls": [{"first_run": first["run_id"], "second_run": second["run_id"],
                                          "same_replay_trajectories": first["replay_trajectories_sha256"] == second["replay_trajectories_sha256"],
                                          "same_learner_settings": first["learner_settings"] == second["learner_settings"],
                                          "epochs": [first["training"]["epochs"], second["training"]["epochs"]]}
                                         for index, first in enumerate(results) for second in results[index + 1:]],
                  "provenance": {"script": str(Path(__file__).resolve()), "script_sha256": digest(__file__),
                                 "analysis_source_hashes": current_hashes, "runtime": runtime_versions(),
                                 "input_file_sha256": inputs, "argv": sys.argv}}
    comparison["provenance"]["source_snapshot"] = capture_analysis_sources(output)
    comparison["provenance"]["detail_file_sha256"] = {str(path.relative_to(output)): digest(path)
                                                        for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "comparison.json", comparison)
    from cmnav.reporting.diagnostic_reports import render_training_report
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write(render_training_report(comparison))
    print(str(output / "report.md"), flush=True)
