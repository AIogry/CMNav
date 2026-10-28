"""Exact historical navigation replay comparisons, excluding recorded timings."""
from __future__ import annotations
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable
from cmnav.runtime.artifacts import sha256_file
from cmnav.analysis.records import read_json as load, iter_rows as json_lines, read_rows as lines
TIME_FIELDS = frozenset({"planning_seconds", "observed_planner_call_seconds", "elapsed_seconds"})

def without_times(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: without_times(item) for key, item in value.items() if key not in TIME_FIELDS}
    if isinstance(value, list):
        return [without_times(item) for item in value]
    return value




def compare_lines(observed: Path, reference: Path,
                  project: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    if not observed.exists() or not reference.exists():
        return {"match": False, "observed_exists": observed.exists(),
                "reference_exists": reference.exists(), "reason": "missing_log"}
    sentinel = object()
    count_observed = count_reference = mismatches = 0
    first_mismatches = []
    for index, (left, right) in enumerate(zip_longest(json_lines(observed), json_lines(reference), fillvalue=sentinel)):
        count_observed += int(left is not sentinel)
        count_reference += int(right is not sentinel)
        if left is sentinel or right is sentinel or project(left) != project(right):
            mismatches += 1
            if len(first_mismatches) < 20:
                first_mismatches.append(index)
    return {"match": mismatches == 0, "observed_rows": count_observed,
            "reference_rows": count_reference, "mismatched_rows": mismatches,
            "first_mismatched_row_indices": first_mismatches}


def candidate_choices(row: dict[str, Any]) -> dict[str, Any]:
    candidate = row["candidate"]
    return {"decision_id": row["decision_id"], "hop": row["hop"],
            "start_node": row["start_node"], "goal_node": row["goal_node"],
            "selected": row["selected"],
            "candidate": {key: candidate[key] for key in
                          ("id", "actions", "termination", "predicted_arrived", "cost")}}


def compare_task(result: dict[str, Any], task_dir: Path, reference: Path) -> dict[str, Any]:
    names = ("summary.json", "execution.jsonl", "planning.jsonl")
    if result["method"].startswith("gcml"):
        names += ("gcml_candidates.jsonl.gz",)
    hashes = {str(reference / name): sha256_file(reference / name)
              for name in names if (reference / name).exists()}
    if not (reference / "summary.json").exists():
        return {"reference": str(reference), "all_match": False,
                "reason": "missing_control_task", "source_hashes": hashes}
    previous = load(reference / "summary.json")
    fields = ("episode_id", "scene_id", "method", "seed", "start_node", "goal_node", "steps",
              "graph_hops_attempted", "graph_hops_completed", "end_reason", "metrics",
              "endpoints_supported", "directed_graph_reachable", "executed_path_length_metres")
    checks = {key: result.get(key) == previous.get(key) for key in fields}
    checks["gcml_planning_without_time"] = without_times(result.get("gcml_planning")) == without_times(previous.get("gcml_planning"))
    detail = {
        "primitive_actions": compare_lines(task_dir / "execution.jsonl", reference / "execution.jsonl", lambda row: row["action"]),
        "execution_records": compare_lines(task_dir / "execution.jsonl", reference / "execution.jsonl", without_times),
        "planning_records_without_time": compare_lines(task_dir / "planning.jsonl", reference / "planning.jsonl", without_times),
    }
    if result["method"].startswith("gcml"):
        detail["candidate_actions_terminations_selection"] = compare_lines(
            task_dir / "gcml_candidates.jsonl.gz", reference / "gcml_candidates.jsonl.gz", candidate_choices)
        detail["candidate_records_without_time"] = compare_lines(
            task_dir / "gcml_candidates.jsonl.gz", reference / "gcml_candidates.jsonl.gz", without_times)
    return {"reference": str(reference), "checks": checks, "log_comparisons": detail,
            "all_match": all(checks.values()) and all(row["match"] for row in detail.values()),
            "ignored_time_fields": sorted(TIME_FIELDS), "source_hashes": hashes,
            "automatic_parameter_adjustment": False}



def baseline_check(result: dict[str, Any], task_dir: Path, source_dir: Path) -> dict[str, Any]:
    original = load(source_dir / "summary.json")
    checks = {key: result[key] == original[key] for key in
              ("episode_id", "start_node", "goal_node", "steps", "end_reason",
               "graph_hops_attempted", "graph_hops_completed", "teacher_subgoals")}
    checks.update({f"metric_{key}": result["metrics"].get(key) == original["metrics"].get(key)
                   for key in ("success", "spl", "distance_to_goal")})
    checks["primitive_action_sequence"] = ([r["action"] for r in lines(task_dir / "execution.jsonl")]
                                             == [r["action"] for r in lines(source_dir / "execution.jsonl")])
    for key in ("candidates", "model_transitions", "candidate_terminations", "invalid_candidates",
                "false_predicted_arrival_candidates"):
        checks[f"planning_{key}"] = result["gcml_planning"][key] == original["gcml_planning"][key]
    current_decisions = [r for r in lines(task_dir / "planning.jsonl") if r["kind"] == "graph_decision"]
    source_decisions = [r for r in lines(source_dir / "planning.jsonl") if r["kind"] == "graph_decision"]
    checks["decision_state_goal_edge_sequence"] = (
        [(r["current_node"], r["planning_goal_node"], r["selected_edge"]) for r in current_decisions]
        == [(r["current_node"], r["planning_goal_node"], r["selected_edge"]) for r in source_decisions])
    for key in ("selected_candidate_id", "rng_state_before_sha256", "rng_state_after_sha256",
                "model_transitions", "termination_counts", "candidate_summaries"):
        checks[f"every_decision_{key}"] = ([r["gcml_selection"].get(key) for r in current_decisions]
                                            == [r["gcml_selection"].get(key) for r in source_decisions])
    return {"source": str(source_dir), "all_match": all(checks.values()), "checks": checks}

