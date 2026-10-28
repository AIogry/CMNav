"""Frozen affordance readout statistics and recorded imaginary-state gate audits.

No fitting or navigation occurs. Teacher paths are diagnostic references;
blocked reference edges do not establish that all alternative paths fail.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cmnav.analysis.records import (read_json, read_rows, iter_rows, digest, save_json,
                                    require, implementation_provenance, capture_analysis_sources, project_outputs_root)
from cmnav.analysis.statistics import classification, target_distribution
from cmnav.runtime.artifacts import require_within, sha256_file, write_json

def counts(raw: np.ndarray, truth: np.ndarray, nodes: list[int], threshold: float) -> dict[str, Any]:
    result = classification(raw, truth, nodes, threshold)
    for name in ("state_ids", "false_positive_by_state"):
        result.pop(name)
    selected = raw[:, nodes] >= threshold
    expected = truth[:, nodes]
    result["exact_set_states"] = int((selected == expected).all(axis=0).sum())
    result["states_with_false_negative"] = int((expected & ~selected).any(axis=0).sum())
    return result

def analyze_affordance_scores(run: Path, output_dir: Path) -> dict[str, Any]:
    inputs = {name: run / (name + ".json") for name in ("model", "graph", "summary", "teacher_subgoal_plan")}
    before = {name: digest(path) for name, path in inputs.items()}
    model, graph = read_json(inputs["model"]), read_json(inputs["graph"])
    if read_json(inputs["summary"])["status"] != "completed":
        raise ValueError("Only completed runs may be analyzed")
    edges = model["edges"]
    if edges != graph["edges"]:
        raise ValueError("Model and observed graph edge vocabularies disagree")
    q = torch.tensor(model["Q"], dtype=torch.float32)
    g = torch.tensor(model["G"], dtype=torch.float32)
    if not torch.isfinite(q).all() or not torch.isfinite(g).all():
        raise ValueError("Nonfinite checkpoint")
    with torch.no_grad():
        raw = (g @ q).numpy()
    m, n = raw.shape
    truth = np.zeros((m, n), dtype=bool)
    for a, (s, _) in enumerate(edges):
        truth[a, s] = True
    sources = np.flatnonzero(truth.any(axis=0)).tolist()
    sinks = np.flatnonzero(~truth.any(axis=0)).tolist()
    all_nodes = list(range(n))
    distributions = {
        "true_outgoing_edges": target_distribution(raw[truth], 1.),
        "nonedges_at_supervised_sources": target_distribution(raw[:, sources][~truth[:, sources]], 0.),
        "nonedges_at_unsupervised_sinks": target_distribution(raw[:, sinks].reshape(-1), 0.),
        "missed_true_edges_at_0_5": target_distribution(raw[truth & (raw < .5)], 1.),
    }
    bins = [-float("inf"), 0., .1, .2, .3, .4, .45, .5, .75, 1., float("inf")]
    positive_bins = [{"interval": f"[{lo}, {hi})", "count": int(((raw[truth] >= lo) & (raw[truth] < hi)).sum())}
                     for lo, hi in zip(bins[:-1], bins[1:])]
    per_node = []
    for s in all_nodes:
        outgoing = np.flatnonzero(truth[:, s]).tolist()
        absent = np.flatnonzero(~truth[:, s])
        best_absent = int(absent[np.argmax(raw[absent, s])])
        predictions = np.flatnonzero(raw[:, s] >= .5).tolist()
        per_node.append({
            "node": s, "supervised_source": bool(outgoing),
            "outgoing": [{"edge": a, "source": s, "target": edges[a][1], "score": float(raw[a, s]),
                          "rank_among_all_edges": 1 + int((raw[:, s] > raw[a, s]).sum()),
                          "passes_0_5": bool(raw[a, s] >= .5)} for a in outgoing],
            "largest_nonedge": {"edge": best_absent, "edge_source": edges[best_absent][0],
                                "edge_target": edges[best_absent][1], "score": float(raw[best_absent, s])},
            "false_positive_edges": [{"edge": a, "edge_source": edges[a][0], "edge_target": edges[a][1],
                                      "score": float(raw[a, s])} for a in predictions if not truth[a, s]],
            "positive_negative_separation_margin": float(min(raw[outgoing, s]) - raw[best_absent, s]) if outgoing else None,
        })
    thresholds = [.5, .45, .4, .35, .3, .25, .2, .1, 0.]
    sweeps = [{"threshold": t, "supervised_sources": counts(raw, truth, sources, t),
               "unsupervised_sinks": counts(raw, truth, sinks, t),
               "all_known_nodes": counts(raw, truth, all_nodes, t)} for t in thresholds]
    manifest = read_json(inputs["teacher_subgoal_plan"])["payload"]["records"]
    paths = []
    for task in manifest:
        actions, nodes = task["reference_edges"], task["reference_nodes"]
        if any(edges[a] != [nodes[i], nodes[i + 1]] for i, a in enumerate(actions)):
            raise ValueError("Discontinuous saved teacher path")
        path_scores = np.array([raw[a, nodes[i]] for i, a in enumerate(actions)])
        paths.append({
            "episode_id": task["source_episode_id"], "task_index": task["task_index"],
            "scores": target_distribution(path_scores, 1.),
            "missed_edges_at_0_5": [{"path_index": i, "edge": a, "source": nodes[i], "target": nodes[i + 1],
                                     "score": float(path_scores[i])} for i, a in enumerate(actions) if path_scores[i] < .5],
            "stage_gate_coverage_by_threshold": [{
                "threshold": t,
                "intervals": {interval: {
                    "stages": len(schedule["stage_reference_edge_ranges"]),
                    "stages_whose_reference_path_internal_edges_all_pass": sum(
                        bool(np.all(path_scores[start + 1:end] >= t))
                        for start, end in schedule["stage_reference_edge_ranges"]),
                } for interval, schedule in task["schedules"].items()},
            } for t in thresholds],
        })
    boundary_distances = np.abs(raw - .5)
    result = {
        "schema": "cmnav.gcml_affordance_score_analysis.v1", "run": str(run),
        "scope": "posthoc_frozen_checkpoint_GQ_and_observed_graph_labels_no_new_rollouts",
        "numeric_backend": "torch.float32_CPU_matrix_multiplication",
        "inputs_sha256": before, "analysis_script_sha256": digest(Path(__file__)),
        "analysis_implementation": implementation_provenance(Path(__file__)),
        "states": n, "actions": m, "supervised_source_nodes": len(sources), "unsupervised_sink_nodes": len(sinks),
        "distributions": distributions, "positive_score_bins": positive_bins,
        "threshold_sweep": sweeps,
        "smallest_absolute_distance_to_threshold_0_5": float(boundary_distances.min()),
        "supervised_nodes_with_strictly_separated_positive_and_negative_scores": sum(
            row["positive_negative_separation_margin"] > 0 for row in per_node if row["supervised_source"]),
        "per_node": per_node, "reference_paths": paths,
        "limitations": [
            "Q columns are saved state embeddings; actual imagined states can differ and are audited separately.",
            "False labels refer to absent observed edges, not all physically possible actions in the environment.",
            "Threshold counts are state classification results, not navigation success measurements.",
            "Reference-path coverage is a teacher path property, not proof of absence of another route.",
            "Stage coverage excludes the first edge because each planning call uses a known-current-outgoing mask at depth zero.",
            "Sinks were absent as training source states; their all-zero labels were not directly supervised.",
        ],
    }
    if before != {name: digest(path) for name, path in inputs.items()}:
        raise RuntimeError("Source inputs changed during analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "score_statistics.json"
    if destination.exists():
        raise FileExistsError(destination)
    result["analysis_source_snapshot"] = capture_analysis_sources(output_dir)
    save_json(destination, result)
    return result

class FrozenGateAudit:
    """Readout calculations only; no model or policy execution methods."""

    def __init__(self, state: dict[str, Any], threshold: float) -> None:
        if state["dtype"] != "float32":
            raise ValueError("This audit reproduces the float32 checkpoint protocol")
        self.q = torch.tensor(state["Q"], dtype=torch.float32)
        self.v = torch.tensor(state["V"], dtype=torch.float32)
        self.g = torch.tensor(state["G"], dtype=torch.float32)
        self.edges = [tuple(edge) for edge in state["edges"]]
        self.threshold = threshold
        self.outgoing = [[edge for edge, pair in enumerate(self.edges) if pair[0] == node]
                         for node in range(state["n_states"])]
        self.prototype_scores: dict[int, torch.Tensor] = {}

    def q_scores(self, node: int) -> torch.Tensor:
        if node not in self.prototype_scores:
            self.prototype_scores[node] = self.g @ self.q[:, node]
        return self.prototype_scores[node]

    def gate(self, z: torch.Tensor, node: int, *, depth: int,
             reference_edge: int | None = None) -> dict[str, Any]:
        scores = self.g @ z
        q_scores = self.q_scores(node)
        legal = self.outgoing[node]
        allowed = (scores >= self.threshold).nonzero(as_tuple=False).flatten().tolist()
        legal_rows = [{"edge": edge, "source": node, "target": self.edges[edge][1],
                       "Gq": float(q_scores[edge]), "Gz": float(scores[edge]),
                       "Gz_minus_Gq": float(scores[edge] - q_scores[edge]),
                       "passes_threshold": bool(scores[edge] >= self.threshold)} for edge in legal]
        top = torch.topk(scores, min(10, len(self.edges)))
        row: dict[str, Any] = {
            "depth": depth, "node": node,
            "gate_applied": depth >= 1,
            "latent_l2_from_node_Q": float(torch.norm(z - self.q[:, node])),
            "max_abs_Gz_minus_Gq": float(torch.max(torch.abs(scores - q_scores))),
            "max_Gz": float(torch.max(scores)), "threshold": self.threshold,
            "observed_outgoing": legal_rows,
            "learned_allowed_edges": allowed,
            "true_positive_count": sum(edge in legal for edge in allowed),
            "false_positive_count": sum(edge not in legal for edge in allowed),
            "all_observed_outgoing_masked": bool(legal) and not any(edge in legal for edge in allowed),
            "empty_learned_mask": not allowed,
            "top_scores": [{"edge": edge, "source": self.edges[edge][0], "target": self.edges[edge][1],
                            "Gz": score, "observed_legal": edge in legal}
                           for edge, score in zip(top.indices.tolist(), top.values.tolist())],
        }
        if reference_edge is not None:
            value = float(scores[reference_edge])
            row["reference_edge"] = {
                "edge": reference_edge, "source": self.edges[reference_edge][0],
                "target": self.edges[reference_edge][1],
                "Gq": float(q_scores[reference_edge]), "Gz": value,
                "Gz_minus_Gq": float(scores[reference_edge] - q_scores[reference_edge]),
                "threshold_deficit": self.threshold - value,
                "rank_among_all_actions": 1 + int((scores > scores[reference_edge]).sum()),
                "blocked_by_learned_gate": depth >= 1 and value < self.threshold,
                "first_layer_oracle_mask_exception": depth == 0,
            }
        return row

    def candidate(self, record: dict[str, Any], reference_by_node: dict[int, int]) -> dict[str, Any]:
        candidate = record["candidate"]
        z = self.q[:, record["start_node"]].clone()
        current = record["start_node"]
        continuous = True
        first_masked_reference = None
        first_all_legal_masked = None
        first_invalid = None
        max_logged_score_error = 0.0
        terminal_gate = None
        # Include the final latent where empty-affordance terminates, for which
        # the original log has no selected action and hence no raw G score.
        for depth in range(len(candidate["actions"]) + 1):
            if continuous:
                gate = self.gate(z, current, depth=depth, reference_edge=reference_by_node.get(current))
                ref = gate.get("reference_edge")
                if ref is not None and ref["blocked_by_learned_gate"] and first_masked_reference is None:
                    first_masked_reference = gate
                if depth >= 1 and gate["all_observed_outgoing_masked"] and first_all_legal_masked is None:
                    first_all_legal_masked = gate
                if depth == len(candidate["actions"]):
                    terminal_gate = gate
            if depth == len(candidate["actions"]):
                break
            action = candidate["actions"][depth]
            score = float((self.g @ z)[action])
            logged = candidate["steps"][depth]["affordance"]
            max_logged_score_error = max(max_logged_score_error, abs(score - logged))
            if continuous and self.edges[action][0] != current:
                first_invalid = {"depth": depth, "expected_source": current, "edge": action,
                                 "actual_source": self.edges[action][0], "target": self.edges[action][1],
                                 "Gz": score, "logged_Gz": logged,
                                 "Gq_at_expected_source": float(self.q_scores(current)[action]),
                                 "passes_threshold": score >= self.threshold,
                                 "gate_at_first_invalid": gate}
                continuous = False
            if continuous:
                current = self.edges[action][1]
            z = z + self.v[:, action]
        return {
            "actions": candidate["actions"], "termination": candidate["termination"],
            "predicted_arrived": candidate["predicted_arrived"],
            "recorded_graph_audit": record["graph_audit"],
            "terminal_gate_if_continuous": terminal_gate,
            "first_masked_reference_edge_visited": first_masked_reference,
            "first_all_observed_outgoing_masked": first_all_legal_masked,
            "first_false_positive_selected": first_invalid,
            "max_abs_reconstructed_vs_logged_selected_Gz": max_logged_score_error,
        }

def analyze_failure_gates(run: Path) -> dict[str, Any]:
    results = read_json(run / "evaluation.json")
    config = read_json(run / "config.json")
    teacher = read_json(run / "teacher_subgoal_plan.json")["payload"]["records"]
    checkpoint_hash = sha256_file(run / "model.json")
    audit = FrozenGateAudit(read_json(run / "model.json"), config["gcml_affordance_threshold"])
    rows = []
    for result in results:
        if result["method"] != "gcml" or result["metrics"]["success"]:
            continue
        directory = run / "evaluation" / result["condition"] / f"task_{result['task_index']:04d}"
        planning = read_rows(directory / "planning.jsonl")
        decisions = [row for row in planning if row.get("kind") == "graph_decision"]
        if not decisions:
            raise ValueError(f"Failure has no recorded planning decision: {directory}")
        failed = decisions[-1]
        decision_id = failed["gcml_selection"]["decision_id"]
        if failed["gcml_selection"]["branch"] != "no_plan":
            raise ValueError("This audit expects no-plan failures, not physical controller failures")
        task = teacher[result["task_index"]]
        key = "final" if result["interval"] is None else str(result["interval"])
        stage_index = failed.get("stage_index", 0)
        begin, end = task["schedules"][key]["stage_reference_edge_ranges"][stage_index]
        stage_nodes = task["reference_nodes"][begin:end + 1]
        stage_edges = task["reference_edges"][begin:end]
        reference_by_node = dict(zip(stage_nodes[:-1], stage_edges))
        current = failed["current_node"]
        goal = failed.get("planning_goal_node", failed["goal_node"])
        reference_trace = []
        if current in stage_nodes:
            offset = stage_nodes.index(current)
            remaining = stage_edges[offset:]
            z = audit.q[:, current].clone()
            for depth, edge in enumerate(remaining):
                source, _ = audit.edges[edge]
                reference_trace.append(audit.gate(z, source, depth=depth, reference_edge=edge))
                z = z + audit.v[:, edge]
        else:
            remaining = None
        first_block = next((row for row in reference_trace if row["reference_edge"]["blocked_by_learned_gate"]), None)
        groups: dict[tuple[Any, ...], dict[str, Any]] = {}
        for record in iter_rows(directory / "gcml_candidates.jsonl.gz"):
            if record["decision_id"] != decision_id:
                continue
            if record["start_node"] != current or record["goal_node"] != goal:
                raise ValueError("Candidate logs disagree with actual planning query")
            candidate = record["candidate"]
            group_key = (tuple(candidate["actions"]), candidate["termination"])
            if group_key not in groups:
                groups[group_key] = {"count": 0, "candidate_ids": [],
                                     **audit.candidate(record, reference_by_node)}
            groups[group_key]["count"] += 1
            groups[group_key]["candidate_ids"].append(candidate["id"])
        termination_counts = Counter()
        empty_nodes = Counter()
        first_invalid_nodes = Counter()
        first_block_matches = 0
        max_reconstruction_error = 0.0
        for group in groups.values():
            count = group["count"]
            termination_counts[group["termination"]] += count
            terminal = group["terminal_gate_if_continuous"]
            if group["termination"] == "empty_affordance" and terminal is not None:
                empty_nodes[terminal["node"]] += count
                if not terminal["empty_learned_mask"]:
                    raise ValueError("Reconstructed empty-affordance termination has a nonempty gate")
            invalid = group["first_false_positive_selected"]
            if invalid is not None:
                first_invalid_nodes[(invalid["expected_source"], invalid["edge"])] += count
            visited = group["first_masked_reference_edge_visited"]
            if first_block is not None and visited is not None and visited["node"] == first_block["node"]:
                first_block_matches += count
            max_reconstruction_error = max(max_reconstruction_error, group["max_abs_reconstructed_vs_logged_selected_Gz"])
        if dict(termination_counts) != failed["gcml_selection"]["termination_counts"]:
            raise ValueError("Candidate counts differ from recorded decision summary")
        execution = read_rows(directory / "execution.jsonl")
        row = {
            "condition": result["condition"], "interval": result["interval"],
            "task_index": result["task_index"], "source_episode_id": result["episode_id"],
            "actual_execution": {"stop_node": current, "final_position": execution[-1]["position"],
                                 "steps": result["steps"], "completed_graph_hops": result["graph_hops_completed"],
                                 "completed_teacher_stages": result.get("teacher_subgoals", {}).get("completed_stages", 0),
                                 "end_reason": result["end_reason"]},
            "failed_query": {"decision_id": decision_id, "stage_index": stage_index,
                             "start_node": current, "current_subgoal": goal,
                             "final_goal": result["goal_node"], "candidate_count": sum(termination_counts.values()),
                             "termination_counts": dict(termination_counts)},
            "reference_route_analysis": {
                "scope": "only_frozen_teacher_reference_route_not_all_alternative_paths",
                "stage_reference_edge_range": [begin, end], "stage_nodes": stage_nodes,
                "stage_edges": stage_edges, "remaining_edges_from_actual_current": remaining,
                "first_post_initial_masked_reference_edge": first_block,
                "trace": reference_trace,
            },
            "candidate_failure_summary": {
                "empty_affordance_terminal_nodes": [{"imagined_node": node, "count": count} for node, count in sorted(empty_nodes.items())],
                "first_invalid_action_groups": [{"imagined_valid_prefix_terminal": node, "invalid_edge": edge,
                                                  "edge_source": audit.edges[edge][0], "edge_target": audit.edges[edge][1],
                                                  "count": count}
                                                 for (node, edge), count in sorted(first_invalid_nodes.items())],
                "candidates_visiting_reference_first_block": first_block_matches,
                "max_abs_reconstructed_vs_logged_selected_Gz": max_reconstruction_error,
            },
            "candidate_groups": list(groups.values()),
            "source_files": {name: sha256_file(directory / name)
                             for name in ("summary.json", "planning.jsonl", "execution.jsonl", "gcml_candidates.jsonl.gz")},
        }
        rows.append(row)
    if sha256_file(run / "model.json") != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during read-only analysis")
    return {
        "schema": "cmnav.gcml_failure_gate_audit.v1", "source_run": str(run),
        "source_model_sha256": checkpoint_hash,
        "source_teacher_subgoal_plan_sha256": sha256_file(run / "teacher_subgoal_plan.json"),
        "source_evaluation_sha256": sha256_file(run / "evaluation.json"),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "analysis_implementation": implementation_provenance(Path(__file__)),
        "source_evaluations": len(results), "learned_G_failure_tasks_analyzed": len(rows),
        "threshold": audit.threshold,
        "formula": "z_0=Q[:,actual_current]; z_(d+1)=z_d+V[:,logged_action_d]; g_d=G@z_d",
        "depth_zero": "true_current_outgoing_mask; G does not gate the first action",
        "dtype": "float32_cpu_torch; matrix-vector reconstruction can differ slightly from original batched GEMM",
        "limitations": ["No new planning, navigation or training is performed.",
                        "Reference-route masking does not prove every alternate route is infeasible.",
                        "Threshold changes are not tested by rollouts; opening an edge also admits possible false positives.",
                        "Imagined nodes are distinct from actual robot positions; graph-invalid sequences have no valid physical terminal node."],
        "rows": rows,
    }

def affordance_scores_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Describe frozen GQ against observed-graph labels.")
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    result = analyze_affordance_scores(run, args.output_dir.resolve() if args.output_dir else run / "g_affordance_analysis")
    import json
    print(json.dumps({name: result[name] for name in ("states", "actions", "distributions", "positive_score_bins", "threshold_sweep")}, ensure_ascii=False, indent=2))

def failure_gates_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Audit frozen learned-G gates at saved failure decisions.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--display-condition", help="Optional condition to print in the compact summary")
    args = parser.parse_args(argv)
    run = require_within(args.run.expanduser().resolve(), project_outputs_root(args.project_root))
    report = analyze_failure_gates(run)
    output = run / "g_affordance_analysis" / "failure_nodes.json"
    report["analysis_source_snapshot"] = capture_analysis_sources(output.parent)
    write_json(output, report)
    print(f"Saved {output}")
    for row in report["rows"]:
        if args.display_condition is not None and row["condition"] != args.display_condition:
            continue
        gate = row["reference_route_analysis"]["first_post_initial_masked_reference_edge"]
        edge = None if gate is None else gate["reference_edge"]
        print(json.dumps({"episode": row["source_episode_id"], "actual": row["actual_execution"]["stop_node"],
                          "stage": row["failed_query"]["stage_index"], "target": row["failed_query"]["current_subgoal"],
                          "reference_block": edge, "candidate_counts": row["candidate_failure_summary"]}, ensure_ascii=False))
