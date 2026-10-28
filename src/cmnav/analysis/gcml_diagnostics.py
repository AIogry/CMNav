"""Saved GCML candidate and branch diagnostics with explicit protocol adapters.

All graph distances and legality labels are retrospective evaluator outputs;
these functions do not train or call a planner or simulator.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from cmnav.analysis.records import (read_json, read_rows, iter_rows, digest, save_json,
                                    require, implementation_provenance, capture_analysis_sources)
from cmnav.analysis.graph_diagnostics import (reverse_distances, optimal_edges, audit_actions,
                                              branch_scores, minimum_hops, can_complete, score_rows)
from cmnav.analysis.statistics import classification

def analyze_mechanisms(run: Path, output: Path) -> None:
    required = [run / f"{name}.json" for name in
                ("summary", "model", "training", "training_walks", "graph", "config", "evaluation")]
    for path in required:
        if not path.is_file():
            raise ValueError(f"Completed run artifact missing: {path}")
    summary, model, training, walks, graph, config, evaluation = map(read_json, required)
    if summary.get("status") != "completed":
        raise ValueError("Analyze only a completed frozen run; do not analyze partial evaluation")
    if output.exists():
        raise ValueError("Output must be a new directory; prior analysis is retained")
    q, w, g = (np.asarray(model[name], dtype=np.float32) for name in ("Q", "W", "G"))
    edges, costs = graph["edges"], graph["edge_mean_costs"]
    n_states, n_edges = q.shape[1], len(edges)
    if model["edges"] != edges or w.shape != (n_edges, q.shape[0]) or g.shape != w.shape:
        raise ValueError("Model vocabulary does not match the observed graph")
    outgoing: list[list[int]] = [[] for _ in range(n_states)]
    reverse: list[list[tuple[int, float]]] = [[] for _ in range(n_states)]
    truth = np.zeros((n_edges, n_states), dtype=bool)
    for action, (source, target) in enumerate(edges):
        outgoing[source].append(action)
        reverse[target].append((source, float(costs[action])))
        truth[action, source] = True
    source_counts: Counter[int] = Counter()
    for trajectory in walks["trajectories"]:
        source_counts.update(transition[0] for transition in trajectory["transitions"])
    known_sources = [state for state in range(n_states) if outgoing[state]]
    sinks = [state for state in range(n_states) if not outgoing[state]]
    trained_sources = sorted(source_counts)
    raw_g = g @ q
    if not np.isfinite(raw_g).all():
        raise ValueError("The saved model has nonfinite G outputs")
    threshold = float(config["gcml_affordance_threshold"])
    gating = {
        "configured_threshold": threshold,
        "training_supervised_source_states": trained_sources,
        "source_training_transition_counts_per_epoch": dict(source_counts),
        "observed_sources_equal_training_sources": known_sources == trained_sources,
        "sink_states_have_source_supervision": any(node in source_counts for node in sinks),
        "label_scope": "known_observed_graph_outgoing_actions_not_unknown_environment_edges",
        "numeric_backend": "numpy_float32_saved_parameters",
        "groups_at_configured_threshold": {
            "all_known_states": classification(raw_g, truth, list(range(n_states)), threshold),
            "observed_sources": classification(raw_g, truth, known_sources, threshold),
            "observed_sinks": classification(raw_g, truth, sinks, threshold),
        },
    }
    if threshold != .5:
        gating["groups_at_fixed_half_threshold"] = {
            "all_known_states": classification(raw_g, truth, list(range(n_states)), .5),
            "observed_sources": classification(raw_g, truth, known_sources, .5),
            "observed_sinks": classification(raw_g, truth, sinks, .5),
        }
    task_reports, branch_reports, candidate_totals = [], [], Counter()
    selected_invalid = []
    input_files = list(required)
    distance_cache: dict[int, list[float]] = {}
    task_dirs = sorted(path.parent for path in (run / "evaluation").glob("*/task_*/summary.json"))
    if len(task_dirs) != len(evaluation):
        raise ValueError("Evaluation task directories and evaluation.json counts disagree")
    expected_ids = Counter((row["method"], str(row["episode_id"])) for row in evaluation)
    actual_ids: Counter[tuple[str, str]] = Counter()
    for directory in task_dirs:
        task = read_json(directory / "summary.json")
        method, episode = task["method"], str(task["episode_id"])
        actual_ids[method, episode] += 1
        task_key = f"{method}/{directory.name}"
        planning_path = directory / "planning.jsonl"
        decisions = [row for row in read_rows(planning_path) if row.get("kind") == "graph_decision"]
        input_files.extend((directory / "summary.json", planning_path))
        goal = task["goal_node"]
        if goal not in distance_cache:
            distance_cache[goal] = reverse_distances(goal, reverse)
        distances = distance_cache[goal]
        first = decisions[0] if decisions else None
        correct_first = optimal_edges(first["current_node"], outgoing, edges, costs, distances) if first else []
        first_frequencies: Counter[int] = Counter()
        first_terminations: Counter[str] = Counter()
        all_terminations: Counter[str] = Counter()
        first_count = first_max_prefix = max_prefix = 0
        task_counts: Counter[str] = Counter()
        selected_records = []
        for decision in decisions:
            state = decision["current_node"]
            if method.startswith("gcml") and len(outgoing[state]) >= 2:
                branch = branch_scores(state, goal, decision["selected_edge"], q, w, outgoing[state],
                                       optimal_edges(state, outgoing, edges, costs, distances),
                                       0. if method == "gcml_greedy" else float(config["gcml_noise_scale"]))
                branch.update(task=task_key, episode_id=episode, hop=decision["hop"])
                branch_reports.append(branch)
        candidate_path = directory / "gcml_candidates.jsonl.gz"
        if candidate_path.exists():
            input_files.append(candidate_path)
            for record in iter_rows(candidate_path):
                candidate = record["candidate"]
                actions = candidate["actions"]
                audit = audit_actions(actions, record["start_node"], record["goal_node"], edges)
                saved_audit = record["graph_audit"]
                if any(audit[key] != saved_audit[key] for key in
                       ("continuous_valid", "valid_prefix_actions", "graph_arrived")):
                    raise ValueError("Independent candidate audit differs from runtime audit")
                predicted = bool(candidate["predicted_arrived"])
                task_counts["candidates"] += 1
                task_counts["predicted_arrived"] += int(predicted)
                task_counts["continuous_valid"] += int(audit["continuous_valid"])
                task_counts["graph_arrived"] += int(audit["graph_arrived"])
                task_counts["false_predicted_arrival"] += int(predicted and not audit["graph_arrived"])
                max_prefix = max(max_prefix, audit["valid_prefix_actions"])
                all_terminations[candidate["termination"]] += 1
                if record["decision_id"] == 0:
                    first_count += 1
                    first_terminations[candidate["termination"]] += 1
                    first_max_prefix = max(first_max_prefix, audit["valid_prefix_actions"])
                    if actions:
                        first_frequencies[actions[0]] += 1
                if record["selected"]:
                    detail = {"decision_id": record["decision_id"], "hop": record["hop"],
                              "candidate_id": candidate["id"], "predicted_arrived": predicted,
                              "cost": candidate["cost"], "actions": actions, **audit}
                    selected_records.append(detail)
                    task_counts["selected_candidates"] += 1
                    task_counts["selected_continuous_valid"] += int(audit["continuous_valid"])
                    task_counts["selected_graph_arrived"] += int(audit["graph_arrived"])
                    if predicted and not audit["graph_arrived"]:
                        selected_invalid.append({"task": task_key, "episode_id": episode, **detail})
        candidate_totals.update(task_counts)
        first_correct_count = sum(first_frequencies[action] for action in correct_first)
        task_reports.append({
            "task": task_key, "method": method, "episode_id": episode,
            "start_node": task["start_node"], "goal_node": goal,
            "success": task["metrics"]["success"], "spl": task["metrics"]["spl"],
            "steps": task["steps"], "end_reason": task["end_reason"],
            "first_decision": {
                "present": first is not None, "current_node": first["current_node"] if first else None,
                "selected_edge": first["selected_edge"] if first else None,
                "shortest_path_first_edges": correct_first,
                "shortest_path_cost": distances[first["current_node"]] if first and math.isfinite(distances[first["current_node"]]) else None,
                "candidate_count": first_count, "candidate_first_edge_frequencies": dict(first_frequencies),
                "candidates_starting_with_correct_branch": first_correct_count,
                "contains_correct_first_branch": bool(first_correct_count) if first_count else None,
                "maximum_valid_prefix_actions": first_max_prefix,
                "terminations": dict(first_terminations),
            },
            "all_decision_candidate_counts": dict(task_counts),
            "maximum_valid_prefix_actions": max_prefix,
            "all_candidate_terminations": dict(all_terminations),
            "selected_candidate_audits": selected_records,
        })
    if actual_ids != expected_ids:
        raise ValueError("Task identities differ from evaluation.json")
    noisy_branches = [row for row in branch_reports if row["top_two_gap_over_pairwise_noise_sd"] is not None]
    ratios = [row["top_two_gap_over_pairwise_noise_sd"] for row in noisy_branches]
    analysis = {
        "schema": "cmnav.gcml_mechanism_analysis.v1", "run": str(run),
        "policy_feedback": False, "training_or_tuning_performed": False,
        "frozen_config": {key: value for key, value in config.items() if key.startswith("gcml_")},
        "epochs": training["epochs"], "gating": gating,
        "tasks": task_reports, "branch_decisions": branch_reports,
        "noise_summary": {
            "actual_branch_decisions_with_noise": len(ratios),
            "gap_ratio_median": float(np.median(ratios)) if ratios else None,
            "gap_ratio_max": max(ratios) if ratios else None,
            "gap_ratio_below_one": sum(value < 1 for value in ratios),
            "interpretation": "ratio compares the legal top-two score gap with independent Gaussian difference SD; it does not establish total candidate success probability",
        },
        "candidate_totals": dict(candidate_totals),
        "selected_false_arrival_candidates": selected_invalid,
        "shortest_path_label": "observed_mean_edge_cost_optimal_ties_abs_and_rel_tolerance_1e-6",
        "inputs": {str(path.relative_to(run)): digest(path) for path in input_files},
        "analysis_script_sha256": digest(Path(__file__).resolve()),
        "analysis_implementation": implementation_provenance(Path(__file__)),
    }
    output.mkdir(parents=True, exist_ok=False)
    analysis["analysis_source_snapshot"] = capture_analysis_sources(output)
    save_json(output / "mechanism_diagnostics.json", analysis)
    from cmnav.reporting.diagnostic_reports import write_mechanism_report
    write_mechanism_report(analysis, output)
    print(output / "report.md")

def analyze_full_goal_failures(run: Path, output: Path, *, expected_tasks: int, expected_candidates: int) -> None:
    require(expected_tasks > 0 and expected_candidates > 0, "Cohort counts must be positive")
    require(not output.exists(), "Use a new output directory to preserve earlier analyses")
    graph = read_json(run / "graph.json")
    require(read_json(run / "summary.json")["status"] == "completed", "Run must be complete")
    edges, costs = graph["edges"], graph["edge_mean_costs"]
    outgoing: list[list[int]] = [[] for _ in graph["positions"]]
    reverse: list[list[tuple[int, float]]] = [[] for _ in outgoing]
    for action, (source, target) in enumerate(edges):
        outgoing[source].append(action)
        reverse[target].append((source, float(costs[action])))
    inputs = [run / f"{name}.json" for name in ("graph", "config", "summary", "model")]
    tasks, all_gaps, association_errors, association_margins = [], [], [], []
    total: Counter[str] = Counter()
    protocol = None
    task_directories = sorted((run / "evaluation/gcml_graph_affordance").glob("task_*"))
    require(len(task_directories) == expected_tasks, "Task count differs from the explicit full-goal cohort contract")
    for directory in task_directories:
        files = [directory / name for name in
                 ("summary.json", "planning.jsonl", "execution.jsonl", "gcml_candidates.jsonl.gz")]
        inputs.extend(files)
        task = read_json(files[0])
        planning, execution, records = (read_rows(path) for path in files[1:])
        decisions = [row for row in planning if row["kind"] == "graph_decision"]
        require(len(decisions) == 1, "Expected exactly one real graph decision")
        decision = decisions[0]
        require(decision["hop"] == 0 and decision["selected_edge"] is None,
                "Expected no graph edge selected at the initial decision")
        current_protocol = decision["gcml_selection"]["planning"]
        if protocol is None:
            protocol = current_protocol
        require(current_protocol == protocol, "Planning settings differ across tasks")
        require(protocol["affordance_mode"] == "observed_graph", "Expected observed graph mask")
        require(len(records) == protocol["candidates"] == expected_candidates, "Candidate count differs from the explicit full-goal cohort contract")
        require(task["graph_hops_attempted"] == task["graph_hops_completed"] == 0,
                "Physical graph hops unexpectedly executed")
        primitive_actions = [row["action"] for row in execution if row["action"] is not None]
        local = [row for row in planning if row["kind"] == "local_execution"]
        require(len(local) == 1 and local[0]["status"] == "arrived" and not local[0]["final"],
                "Unexpected initial connector execution")
        require(primitive_actions[-1] == "STOP" and len(primitive_actions) == task["steps"]
                == local[0]["steps"] + 1, "Primitive steps are not connector plus STOP")
        goal = task["goal_node"]
        distances = reverse_distances(goal, reverse)
        hop_distances = minimum_hops(goal, reverse)
        counters: Counter[str] = Counter()
        terminal_nodes: Counter[tuple[str, int]] = Counter()
        cycle_lengths: Counter[int] = Counter()
        route_ids: dict[tuple[int, ...], list[int]] = {}
        candidates, gaps = [], []
        for record in records:
            candidate = record["candidate"]
            require(record["decision_id"] == record["hop"] == 0 and not record["selected"],
                    "Unexpected later or selected candidate")
            require(not candidate["predicted_arrived"], "Unexpected predicted goal arrival")
            nodes = [record["start_node"]]
            require(nodes[0] == task["start_node"] and record["goal_node"] == goal,
                    "Candidate task identity mismatch")
            require(len(candidate["actions"]) == len(candidate["steps"]), "Incomplete step diagnostics")
            first_trap = first_unreachable = None
            last_viable: list[int] = []
            for index, (action, step) in enumerate(zip(candidate["actions"], candidate["steps"])):
                source, target = edges[action]
                require(source == nodes[-1] and step["action"] == action and step["index"] == index,
                        "Discontinuous candidate or mismatched step")
                require(step["decoded_state_before"] == source and step["decoded_state_after"] == target
                        and step["associated_before"] and step["associated_after"],
                        "Latent association differs from graph transition")
                require(step["eligible_edges"] == outgoing[source], "Mask differs from observed outgoing edges")
                scores = score_rows(step, edges)
                top_utility = max(scores, key=lambda row: row["normalized_utility"])["edge"]
                top_logit = max(scores, key=lambda row: row["normalized_utility_plus_noise"])["edge"]
                require(top_logit == action, "Saved action differs from saved utility-plus-noise argmax")
                association_errors.append(step["nearest_distance"])
                association_margins.append(step["association_margin"])
                visited = set(nodes)
                last_viable = [edge for edge in outgoing[source]
                               if can_complete(edges[edge][1], goal, visited, outgoing, edges)]
                if first_trap is None and action not in last_viable:
                    require(bool(last_viable), "First lost simple completion should have an alternative")
                    kind = ("globally_unreachable" if not math.isfinite(distances[target]) else
                            "immediate_revisit" if target in visited else "must_revisit_prefix")
                    first_trap = {"step_index": index, "node": source, "action": action, "target": target,
                                  "kind": kind, "simple_completion_edges": last_viable,
                                  "utility_top_edge": top_utility,
                                  "utility_top_has_simple_completion": top_utility in last_viable,
                                  "saved_scores": scores}
                    counters[f"first_trap_{kind}"] += 1
                    counters["first_trap_utility_top_viable" if top_utility in last_viable
                             else "first_trap_utility_top_not_viable"] += 1
                if math.isfinite(distances[source]) and not math.isfinite(distances[target]):
                    first_unreachable = {"step_index": index, "node": source, "edge": action, "target": target}
                if len(scores) > 1:
                    counters["branch_steps"] += 1
                    counters["noise_changes_argmax"] += int(action != top_utility)
                    ranked = sorted(row["normalized_utility"] for row in scores)[::-1]
                    gaps.append(ranked[0] - ranked[1])
                    if math.isfinite(distances[source]):
                        optimal = [edge for edge in outgoing[source]
                                   if math.isclose(costs[edge] + distances[edges[edge][1]], distances[source],
                                                   rel_tol=1e-6, abs_tol=1e-6)]
                        counters["reachable_branch_steps"] += 1
                        counters["utility_top_cost_optimal"] += int(top_utility in optimal)
                        counters["selected_cost_optimal"] += int(action in optimal)
                nodes.append(target)
            require(first_trap is not None, "Expected a lost simple completion in every failed candidate")
            require(nodes == record["graph_audit"]["valid_prefix_nodes"]
                    and record["graph_audit"]["continuous_valid"], "Saved graph audit differs")
            require(goal not in nodes and len(set(nodes[:-1])) == len(nodes[:-1]),
                    "Unexpected arrival or earlier unhandled cycle")
            termination = candidate["termination"]
            loop = None
            if termination == "empty_affordance":
                require(not outgoing[nodes[-1]], "Empty mask is not an observed graph sink")
                require(first_unreachable is not None, "Sink path did not lose global reachability")
            elif termination == "retrieved_state_cycle":
                require(nodes[-1] in nodes[:-1], "Cycle flag is not a true graph revisit")
                loop = nodes[nodes.index(nodes[-1]):]
                cycle_lengths[len(loop) - 1] += 1
                counters["cycle_terminal_globally_reachable"] += int(math.isfinite(distances[nodes[-1]]))
                counters["cycle_last_has_simple_completion_alternative"] += int(bool(last_viable))
            else:
                raise ValueError(f"Unexpected termination: {termination}")
            counters[termination] += 1
            counters["candidates"] += 1
            counters["legal_transitions"] += len(candidate["actions"])
            terminal_nodes[termination, nodes[-1]] += 1
            route_ids.setdefault(tuple(nodes), []).append(candidate["id"])
            candidates.append({"id": candidate["id"], "actions": candidate["actions"], "nodes": nodes,
                               "length": len(candidate["actions"]), "termination": termination,
                               "cycle_nodes": loop, "first_lost_simple_completion": first_trap,
                               "first_globally_unreachable_transition": first_unreachable,
                               "last_step_saved_scores": score_rows(candidate["steps"][-1], edges),
                               "minimum_remaining_graph_cost": min(distances[node] for node in nodes)})
        require(counters["legal_transitions"] == task["gcml_planning"]["model_transitions"],
                "Transition totals differ from task summary")
        require({key: counters[key] for key in ("retrieved_state_cycle", "empty_affordance")}
                == task["gcml_planning"]["candidate_terminations"], "Termination totals differ")
        dijkstra_path = run / "evaluation/dijkstra" / directory.name / "summary.json"
        inputs.append(dijkstra_path)
        dijkstra = read_json(dijkstra_path)
        require(dijkstra["episode_id"] == task["episode_id"], "Dijkstra task differs")
        lengths = [row["length"] for row in candidates]
        tasks.append({"task": directory.name, "episode_id": task["episode_id"],
                      "start_node": task["start_node"], "goal_node": goal,
                      "real_planning_decisions": len(decisions), "selected_graph_edges": 0,
                      "primitive_actions": primitive_actions, "connector_steps": local[0]["steps"],
                      "primitive_steps_including_stop": task["steps"],
                      "dijkstra_success": dijkstra["metrics"]["success"],
                      "dijkstra_executed_graph_hops": dijkstra["graph_hops_completed"],
                      "minimum_graph_hops": hop_distances[task["start_node"]],
                      "initial_shortest_graph_cost": distances[task["start_node"]],
                      "counts": dict(counters), "cycle_lengths": dict(cycle_lengths),
                      "lengths": {"min": min(lengths), "mean": statistics.mean(lengths), "max": max(lengths)},
                      "terminal_node_counts": [{"termination": term, "node": node, "count": count}
                                               for (term, node), count in sorted(terminal_nodes.items())],
                      "distinct_routes": [{"nodes": list(nodes), "candidate_ids": ids, "count": len(ids)}
                                          for nodes, ids in route_ids.items()],
                      "candidates": candidates})
        total.update(counters)
        all_gaps.extend(gaps)
    result = {"schema": "cmnav.gcml_full_goal_failure_analysis.v1", "run": str(run),
              "analysis_scope": "archived_observed_graph_full_goal_candidates_only",
              "new_training_or_navigation_runs": False, "policy_feedback": False,
              "simple_completion_definition": "Goal reachable in the directed observed graph after excluding all prefix nodes; ignores horizon; evaluator only",
              "cost_optimal_definition": "Saved mean edge costs; optimal ties use absolute and relative tolerance 1e-6",
              "planning_protocol": protocol, "totals": dict(total),
              "association": {"transitions_audited": len(association_errors),
                              "decoded_graph_mismatches": 0, "mask_mismatches": 0,
                              "max_nearest_distance": max(association_errors),
                              "min_relative_margin": min(association_margins)},
              "branch_score_gaps": {"min": min(all_gaps), "median": statistics.median(all_gaps),
                                    "max": max(all_gaps), "noise_sigma": protocol["noise_scale"],
                                    "scope": "repeated candidate branches, not independent tasks or a zero-noise rerun"},
              "tasks": tasks,
              "input_sha256": {str(path.relative_to(run)): digest(path) for path in inputs},
              "analysis_script_sha256": digest(Path(__file__).resolve()),
              "analysis_implementation": implementation_provenance(Path(__file__)),
              "cohort_contract": {"tasks": expected_tasks, "candidates_per_decision": expected_candidates}}
    output.mkdir(parents=True, exist_ok=False)
    result["analysis_source_snapshot"] = capture_analysis_sources(output)
    (output / "diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    from cmnav.reporting.diagnostic_reports import write_full_goal_report
    write_full_goal_report(result, output)
    print(json.dumps({"report": str(output / "report.md"), "totals": dict(total),
                      "association": result["association"], "score_gaps": result["branch_score_gaps"]}, ensure_ascii=False))

def mechanisms_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Audit frozen GCML mechanisms from saved records.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    run = arguments.run.expanduser().resolve()
    output = (arguments.output or run / "mechanism_analysis").expanduser().resolve()
    analyze_mechanisms(run, output)

def full_goal_failures_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Audit initial no-plan failures with observed-graph full-goal masks.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-tasks", type=int, required=True, help="Frozen historical cohort task count")
    parser.add_argument("--expected-candidates", type=int, required=True, help="Frozen candidate budget per decision")
    arguments = parser.parse_args(argv)
    run = arguments.run.expanduser().resolve()
    output = (arguments.output or run / "full_goal_failure_analysis").expanduser().resolve()
    analyze_full_goal_failures(run, output, expected_tasks=arguments.expected_tasks,
                               expected_candidates=arguments.expected_candidates)
