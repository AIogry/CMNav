"""Post-hoc source audits and summaries for teacher-assisted gate comparisons."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Any
from cmnav.evaluation.oracle_graph_navigation import summarize_graph_navigation_results
from cmnav.analysis.records import iter_rows

def audit_imagined_sources(path: Path, edges: list[tuple[int, int]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for_line_counts: Counter[str] = Counter()
    for row in iter_rows(path):
        counts["candidates"] += 1
        first_invalid = row["graph_audit"]["first_invalid"]
        invalid_index = None if first_invalid is None else first_invalid["index"]
        for step in row["candidate"]["steps"]:
            counts["transitions"] += 1
            if not step["associated_before"]:
                label = "unknown_off_manifold"
            elif edges[step["action"]][0] == step["decoded_state_before"]:
                label = "decoded_source_match"
            else:
                label = "decoded_source_mismatch"
            counts[label] += 1
            scope = "after_first_graph_invalid_action" if invalid_index is not None and step["index"] > invalid_index else "up_to_first_graph_invalid_action"
            for_line_counts[f"{scope}:{label}"] += 1
    return {
        "scope": "posthoc_local_decoded_source_consistency_not_physical_path_validity",
        **{key: counts[key] for key in ("candidates", "transitions", "decoded_source_match",
                                      "decoded_source_mismatch", "unknown_off_manifold")},
        "by_graph_prefix_scope": dict(for_line_counts),
        "after_first_graph_invalid_does_not_restore_physical_continuity": True,
        "used_by_planner": False,
    }


def aggregate(results: list[dict[str, Any]], threshold: float, gate_mode: str) -> dict[str, Any]:
    group = [row for row in results if row["threshold"] == threshold and row["gate_mode"] == gate_mode]
    value = summarize_graph_navigation_results(group, ["gcml"])["gcml"]
    stage_counts = Counter(stage["status"] for row in group for stage in row["teacher_subgoals"]["stages"])
    transition_counts = {key: sum(row["imagined_transition_source_audit"][key] for row in group)
                         for key in ("transitions", "decoded_source_match", "decoded_source_mismatch", "unknown_off_manifold")}
    value.update(threshold=threshold, gate_mode=gate_mode, successes=sum(bool(row["metrics"]["success"]) for row in group),
                 stages={"completed": stage_counts["completed"], "failed": stage_counts["failed"],
                         "not_attempted": stage_counts["not_attempted"], "total": sum(stage_counts.values())},
                 imagined_transition_source_audit=transition_counts,
                 tasks=[{"episode_id": row["episode_id"], "success": row["metrics"]["success"],
                         "steps": row["steps"], "completed_stages": row["teacher_subgoals"]["completed_stages"],
                         "total_stages": row["teacher_subgoals"]["total_stages"], "end_reason": row["end_reason"]}
                        for row in group])
    return value


