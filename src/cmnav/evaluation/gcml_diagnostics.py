"""Independent graph audits and bounded-size summaries of GCML imagination.

The model receives none of these audit labels. A candidate is traced against
the frozen observed graph only after planning has finished, without repairing
an action, replacing an imagined state, or changing the selected candidate.
"""
from __future__ import annotations

from collections import Counter
import gzip
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Sequence

from cmnav.runtime.artifacts import write_json_line


GCML_METHODS = frozenset({"gcml", "gcml_graph_affordance", "gcml_greedy"})


def audit_graph_actions(
    actions: Sequence[int], start: int, goal: int,
    edges: Sequence[tuple[int, int]], edge_costs: Sequence[float],
) -> dict[str, Any]:
    """Audit continuity from the real start; never teleport after a bad edge."""
    current = start
    prefix_nodes = [start]
    first_invalid = None
    prefix_cost = 0.0
    prefix_reached_goal = start == goal
    visited = {start}
    revisits = 0
    for index, action in enumerate(actions):
        if (isinstance(action, bool) or not isinstance(action, Integral)
                or action < 0 or action >= len(edges)):
            first_invalid = {"index": index, "action": action,
                             "reason": "edge_id_out_of_range", "expected_source": current}
            break
        source, target = edges[int(action)]
        if source != current:
            first_invalid = {"index": index, "action": int(action),
                             "reason": "source_mismatch", "expected_source": current,
                             "actual_source": source, "edge_target": target}
            break
        prefix_cost += float(edge_costs[int(action)])
        current = target
        prefix_nodes.append(current)
        prefix_reached_goal = prefix_reached_goal or current == goal
        revisits += int(current in visited)
        visited.add(current)
    valid = first_invalid is None
    return {
        "scope": "independent_observed_graph_audit_not_planner_feedback",
        "continuous_valid": valid,
        "action_count": len(actions),
        "valid_prefix_actions": len(prefix_nodes) - 1,
        "valid_prefix_nodes": prefix_nodes,
        "valid_prefix_cost": prefix_cost,
        "valid_prefix_reached_goal": prefix_reached_goal,
        "graph_arrived": valid and current == goal,
        "terminal_node": current if valid else None,
        "node_revisits_in_valid_prefix": revisits,
        "first_invalid": first_invalid,
    }


def _nonfinite_values(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_nonfinite_values(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_nonfinite_values(item) for item in value)
    return int(isinstance(value, Real) and not math.isfinite(float(value)))


class GCMLPlanningRecorder:
    """Store full candidate traces in gzip and expose per-task audit totals."""

    def __init__(self, directory: Path, edges: Sequence[tuple[int, int]],
                 edge_costs: Sequence[float]) -> None:
        self.path = directory / "gcml_candidates.jsonl.gz"
        self._stream = gzip.open(self.path, "xt", encoding="utf-8")
        self._edges = tuple(edges)
        self._costs = tuple(edge_costs)
        self._counts: Counter[str] = Counter()
        self._branches: Counter[str] = Counter()
        self._terminations: Counter[str] = Counter()
        self._planning_seconds = 0.0
        self._observed_call_seconds = 0.0
        self._selected_costs: list[float] = []

    def record(self, diagnostics: dict[str, Any], *, start: int, goal: int,
               hop: int, elapsed_seconds: float) -> dict[str, Any]:
        """Return compact diagnostics after writing untouched candidate data."""
        decision_id = self._counts["decisions"]
        self._counts["decisions"] += 1
        self._branches[str(diagnostics.get("branch", "unspecified"))] += 1
        self._counts["model_transitions"] += int(diagnostics.get("model_transitions", 0))
        self._planning_seconds += float(diagnostics.get("planning_seconds", 0.0))
        self._observed_call_seconds += elapsed_seconds
        # Masked -inf scores, if intentionally exposed, are counted for audit;
        # this is not itself a floating-point failure classification.
        self._counts["nonfinite_diagnostic_values"] += _nonfinite_values(diagnostics)
        candidates = diagnostics.get("candidates", [])
        selected_id = diagnostics.get("selected_candidate_id")
        summaries = []
        selected_audit = None
        for candidate in candidates:
            actions = candidate["actions"]
            audit = audit_graph_actions(actions, start, goal, self._edges, self._costs)
            predicted = bool(candidate["predicted_arrived"])
            selected = selected_id is not None and candidate["id"] == selected_id
            self._counts["candidates"] += 1
            self._counts["predicted_arrived_candidates"] += int(predicted)
            self._counts["continuous_valid_candidates"] += int(audit["continuous_valid"])
            self._counts["invalid_candidates"] += int(not audit["continuous_valid"])
            self._counts["graph_arrived_candidates"] += int(audit["graph_arrived"])
            self._counts["false_predicted_arrival_candidates"] += int(predicted and not audit["graph_arrived"])
            self._counts["missed_graph_arrival_candidates"] += int(not predicted and audit["graph_arrived"])
            self._terminations[str(candidate["termination"])] += 1
            if selected:
                selected_audit = audit
                self._counts["selected_candidates"] += 1
                self._counts["selected_continuous_valid"] += int(audit["continuous_valid"])
                self._counts["selected_graph_arrived"] += int(audit["graph_arrived"])
                self._counts["selected_false_predicted_arrival"] += int(predicted and not audit["graph_arrived"])
                self._selected_costs.append(float(candidate["cost"]))
            write_json_line(self._stream, {
                "schema": "cmnav.gcml_candidate_audit.v1", "decision_id": decision_id,
                "hop": hop, "start_node": start, "goal_node": goal,
                "selected": selected, "candidate": candidate, "graph_audit": audit,
            })
            summaries.append({
                "id": candidate["id"], "actions": len(actions),
                "predicted_arrived": predicted, "termination": candidate["termination"],
                "cost": candidate["cost"], "selected": selected,
                "continuous_valid": audit["continuous_valid"],
                "graph_arrived": audit["graph_arrived"],
                "first_invalid": audit["first_invalid"],
            })
        compact = {key: value for key, value in diagnostics.items() if key != "candidates"}
        compact.update({
            "decision_id": decision_id,
            "observed_planner_call_seconds": elapsed_seconds,
            "candidate_log": self.path.name,
            "candidate_summaries": summaries,
            "selected_candidate_graph_audit": selected_audit,
            "graph_audit_was_used_by_planner": False,
        })
        return compact

    def invalid_selected_edge(self) -> None:
        self._counts["invalid_selected_edges"] += 1

    def close(self) -> None:
        self._stream.close()

    def summary(self) -> dict[str, Any]:
        # Explicit zeros keep unsupported tasks and greedy controls comparable.
        keys = (
            "decisions", "candidates", "model_transitions",
            "predicted_arrived_candidates", "continuous_valid_candidates",
            "invalid_candidates", "graph_arrived_candidates",
            "false_predicted_arrival_candidates", "missed_graph_arrival_candidates",
            "selected_candidates", "selected_continuous_valid", "selected_graph_arrived",
            "selected_false_predicted_arrival", "invalid_selected_edges",
            "nonfinite_diagnostic_values",
        )
        return {
            "schema": "cmnav.gcml_task_planning_audit.v1",
            **{key: self._counts[key] for key in keys},
            "branches": dict(self._branches),
            "candidate_terminations": dict(self._terminations),
            "planning_seconds": self._planning_seconds,
            "observed_planner_call_seconds": self._observed_call_seconds,
            "selected_candidate_costs": self._selected_costs,
            "candidate_log": self.path.name,
            "graph_audit_was_used_by_planner": False,
        }
