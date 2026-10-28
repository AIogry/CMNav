"""Structural audits of a frozen experience-derived landmark hierarchy.

These exact graph searches are evaluator diagnostics. They must not supply
legality masks, endpoint candidates, or successful plans to a learned policy.
All distances count abstract low-level graph edges, not robot primitives.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from typing import Any

from cmnav.analysis.statistics import distribution
from cmnav.memory.graph_search import shortest_path


def _integer(value: Any, label: str, upper: int | None = None) -> int:
    if type(value) is not int or value < 0 or (upper is not None and value >= upper):
        raise ValueError(f"{label} must be an integer in the declared node range.")
    return value


def _adjacency(n_states: int, edges: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    outgoing: list[list[int]] = [[] for _ in range(n_states)]
    incoming: list[list[int]] = [[] for _ in range(n_states)]
    for action, edge in enumerate(edges):
        if len(edge) != 2:
            raise ValueError(f"Edge {action} must contain source and target.")
        source = _integer(edge[0], f"edge {action} source", n_states)
        target = _integer(edge[1], f"edge {action} target", n_states)
        outgoing[source].append(target)
        incoming[target].append(source)
    return outgoing, incoming


def _components(outgoing: list[list[int]], incoming: list[list[int]]) -> dict[str, Any]:
    """Directed strongly connected and undirected weak components, including isolates."""
    visited: set[int] = set()
    finishing_order: list[int] = []
    for start in range(len(outgoing)):
        if start in visited:
            continue
        visited.add(start)
        pending: list[tuple[int, int]] = [(start, 0)]
        while pending:
            node, next_index = pending[-1]
            if next_index == len(outgoing[node]):
                pending.pop()
                finishing_order.append(node)
                continue
            target = outgoing[node][next_index]
            pending[-1] = (node, next_index + 1)
            if target not in visited:
                visited.add(target)
                pending.append((target, 0))

    def collect(starts: list[int], neighbors: list[list[int]]) -> list[list[int]]:
        seen: set[int] = set()
        groups: list[list[int]] = []
        for start in starts:
            if start in seen:
                continue
            seen.add(start)
            group: list[int] = []
            pending = [start]
            while pending:
                node = pending.pop()
                group.append(node)
                for target in neighbors[node]:
                    if target not in seen:
                        seen.add(target)
                        pending.append(target)
            groups.append(sorted(group))
        return sorted(groups, key=lambda group: (-len(group), group[0]))

    strong = collect(list(reversed(finishing_order)), incoming)
    weak = collect(list(range(len(outgoing))),
                   [sorted(set(outgoing[node]) | set(incoming[node])) for node in range(len(outgoing))])
    return {
        "strong_count": len(strong), "strong_components": strong,
        "strong_sizes": [len(group) for group in strong],
        "weak_count": len(weak), "weak_components": weak,
        "weak_sizes": [len(group) for group in weak],
        "isolated_nodes": [node for node in range(len(outgoing)) if not outgoing[node] and not incoming[node]],
        "sink_nodes": [node for node in range(len(outgoing)) if not outgoing[node]],
        "source_nodes": [node for node in range(len(outgoing)) if not incoming[node]],
    }


def _hop_paths(start: int, outgoing_actions: list[list[int]], edges: list[list[int]]) -> dict[int, list[int]]:
    """Shortest edge-count paths with stable edge-order tie breaking."""
    paths: dict[int, list[int]] = {start: []}
    pending = deque([start])
    while pending:
        source = pending.popleft()
        for action in outgoing_actions[source]:
            target = edges[action][1]
            if target not in paths:
                paths[target] = paths[source] + [action]
                pending.append(target)
    return paths


def _outgoing_actions(n_states: int, edges: list[list[int]]) -> list[list[int]]:
    outgoing: list[list[int]] = [[] for _ in range(n_states)]
    for action, (source, _) in enumerate(edges):
        outgoing[source].append(action)
    return outgoing


def _attachment_map(rows: list[dict[str, Any]], n_landmarks: int) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        landmark = _integer(row["landmark"], "attachment landmark", n_landmarks)
        if landmark in result:
            raise ValueError("Each node/landmark attachment must be aggregated into one relation.")
        for key in ("median_steps", "min_steps"):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Attachment {key} must be finite and nonnegative.")
        if row["min_steps"] > row["median_steps"]:
            raise ValueError("Attachment minimum cannot exceed its median step count.")
        result[landmark] = row
    return result


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def analyze_hierarchy(hierarchy: dict[str, Any], graph_edges: list[list[int]],
                      tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit a previously frozen hierarchy without changing its data or using models.

    ``tasks`` contains task_index, episode_id, start_node, goal_node. The caller
    loads those tasks only after constructing and freezing the hierarchy. Entry
    and exit candidates here exhaust saved relations; they are diagnostic upper
    coverage bounds, not bounded candidates selected by a navigation method.
    """
    if hierarchy.get("schema") != "cmnav.landmark_hierarchy.v1":
        raise ValueError("Unsupported landmark hierarchy schema.")
    n_states = _integer(hierarchy["n_states"], "n_states")
    landmark_nodes = hierarchy["landmark_nodes"]
    if len(set(landmark_nodes)) != len(landmark_nodes):
        raise ValueError("Landmark nodes must be unique.")
    for node in landmark_nodes:
        _integer(node, "landmark low-level node", n_states)
    n_landmarks = len(landmark_nodes)
    edges = hierarchy["edges"]
    low_out, low_in = _adjacency(n_states, graph_edges)
    high_out, high_in = _adjacency(n_landmarks, edges)
    macro_edges = hierarchy["macro_edges"]
    if len(macro_edges) != len(edges):
        raise ValueError("Macro edge metadata must match the high-level action count.")
    costs: list[float] = []
    for action, row in enumerate(macro_edges):
        if row["action"] != action or [row["source"], row["target"]] != list(edges[action]):
            raise ValueError("Macro edge order and action IDs must match hierarchy edges.")
        if row["source_node"] != landmark_nodes[row["source"]] or row["target_node"] != landmark_nodes[row["target"]]:
            raise ValueError("Macro edge node IDs disagree with the landmark map.")
        cost = row["median_steps"]
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
            raise ValueError("Macro edge median steps must be finite and positive.")
        if row["support_count"] != len(row["lengths"]) or not row["lengths"]:
            raise ValueError("Macro support counts and lengths disagree.")
        costs.append(float(cost))

    attachments = hierarchy["attachments"]
    if len(attachments["entry"]) != n_states or len(attachments["exit"]) != n_states:
        raise ValueError("Attachment tables must cover every low-level node.")
    entry = [_attachment_map(rows, n_landmarks) for rows in attachments["entry"]]
    exit_ = [_attachment_map(rows, n_landmarks) for rows in attachments["exit"]]
    for high_node, low_node in enumerate(landmark_nodes):
        for label, relations in (("entry", entry), ("exit", exit_)):
            if high_node not in relations[low_node] or relations[low_node][high_node]["min_steps"] != 0:
                raise ValueError(f"Landmark {low_node} must have a zero-step {label} identity relation.")
    high_actions = _outgoing_actions(n_landmarks, edges)
    high_hops = [_hop_paths(node, high_actions, edges) for node in range(n_landmarks)]
    high_costs = []
    for source in range(n_landmarks):
        paths = {}
        for target in high_hops[source]:
            path, _ = shortest_path(n_landmarks, edges, costs, source, target)
            if path is None:
                raise ValueError("Weighted and unweighted hierarchy reachability disagree.")
            paths[target] = (sum(costs[action] for action in path), path)
        high_costs.append(paths)
    task_rows: list[dict[str, Any]] = []
    for task in tasks:
        start = _integer(task["start_node"], "task start_node", n_states)
        goal = _integer(task["goal_node"], "task goal_node", n_states)
        bottom_path, _ = shortest_path(n_states, graph_edges, [1.] * len(graph_edges), start, goal)
        pairs: list[dict[str, Any]] = []
        for entry_id, entry_row in sorted(entry[start].items()):
            for exit_id, exit_row in sorted(exit_[goal].items()):
                if exit_id not in high_costs[entry_id]:
                    continue
                macro_cost, macro_path = high_costs[entry_id][exit_id]
                pairs.append({
                    "entry_landmark": entry_id, "exit_landmark": exit_id,
                    "entry_node": landmark_nodes[entry_id], "exit_node": landmark_nodes[exit_id],
                    "entry_median_steps": entry_row["median_steps"], "exit_median_steps": exit_row["median_steps"],
                    "minimum_macro_hops": len(high_hops[entry_id][exit_id]),
                    "minimum_macro_hop_actions": high_hops[entry_id][exit_id],
                    "minimum_median_cost_macro_actions": macro_path,
                    "macro_median_cost": macro_cost,
                    "estimated_total_steps": entry_row["median_steps"] + macro_cost + exit_row["median_steps"],
                })
        best_cost = min(pairs, key=lambda row: (row["estimated_total_steps"], row["entry_landmark"], row["exit_landmark"]), default=None)
        best_hops = min(pairs, key=lambda row: (row["minimum_macro_hops"], row["estimated_total_steps"], row["entry_landmark"], row["exit_landmark"]), default=None)
        missing = ([] if entry[start] else ["entry"]) + ([] if exit_[goal] else ["exit"])
        status = ("no_entry" if not entry[start] else "no_exit" if not exit_[goal]
                  else "high_disconnected" if not pairs else "structurally_supported")
        if pairs and bottom_path is None:
            raise ValueError("A supported hierarchy route must also be reachable in its source low-level graph.")
        task_rows.append({
            "task_index": task["task_index"], "episode_id": task["episode_id"],
            "start_node": start, "goal_node": goal, "already_at_goal": start == goal,
            "bottom_reachable": bottom_path is not None,
            "bottom_minimum_hops": len(bottom_path) if bottom_path is not None else None,
            "bottom_shortest_actions_audit_only": bottom_path,
            "direct_short_path_within_attachment_horizon_audit_only": (
                bottom_path is not None and len(bottom_path) <= hierarchy["config"]["attach_max_steps"]),
            "entry_candidates": sorted(entry[start]), "exit_candidates": sorted(exit_[goal]),
            "entry_candidate_count": len(entry[start]), "exit_candidate_count": len(exit_[goal]),
            "total_endpoint_pairs": len(entry[start]) * len(exit_[goal]),
            "reachable_endpoint_pairs": len(pairs),
            "estimated_total_steps": distribution([row["estimated_total_steps"] for row in pairs]),
            "minimum_macro_hops": distribution([row["minimum_macro_hops"] for row in pairs]),
            "best_median_cost_pair_audit_only": best_cost,
            "shortest_macro_route_pair_audit_only": best_hops,
            "status": status, "missing_attachments": missing,
        })
    all_entry = [bool(rows) for rows in entry]
    all_exit = [bool(rows) for rows in exit_]
    both = sum(has_entry and has_exit for has_entry, has_exit in zip(all_entry, all_exit))
    macro_lengths = [length for row in macro_edges for length in row["lengths"]]
    high_trajectories = hierarchy["high_trajectories"]
    return {
        "schema": "cmnav.landmark_hierarchy_audit.v1",
        "permission": "privileged_abstract_graph_structural_audit_only",
        "distance_unit": "directed_low_level_graph_edges_not_robot_primitives",
        "task_use": "post_freeze_development_diagnostic_not_landmark_selection_or_model_input",
        "size": {
            "low_nodes": n_states, "low_edges": len(graph_edges),
            "landmarks": n_landmarks, "macro_edges": len(edges),
            "node_fraction_retained": _ratio(n_landmarks, n_states),
            "edge_fraction_retained": _ratio(len(edges), len(graph_edges)),
            "nodes_per_landmark": _ratio(n_states, n_landmarks),
            "low_edges_per_macro_edge": _ratio(len(graph_edges), len(edges)),
        },
        "macro_support": {
            "observations": len(macro_lengths),
            "steps_per_observation": distribution(macro_lengths),
            "support_count_per_edge": distribution([row["support_count"] for row in macro_edges]),
            "median_steps_per_edge": distribution(costs),
            "single_support_actions": [row["action"] for row in macro_edges if row["support_count"] == 1],
        },
        "high_replay": {
            "sequences": len(high_trajectories),
            "transitions": sum(len(row["transitions"]) for row in high_trajectories),
            "transitions_per_sequence": distribution([len(row["transitions"]) for row in high_trajectories]),
        },
        "topology": {
            "low": _components(low_out, low_in), "high": _components(high_out, high_in),
            "high_in_degree": [len(neighbors) for neighbors in high_in],
            "high_out_degree": [len(neighbors) for neighbors in high_out],
            "reachable_ordered_landmark_pairs_including_identity": sum(len(paths) for paths in high_hops),
            "all_ordered_landmark_pairs_including_identity": n_landmarks ** 2,
        },
        "attachments": {
            "entry_nodes": sum(all_entry), "exit_nodes": sum(all_exit), "both_nodes": both,
            "neither_nodes": sum(not has_entry and not has_exit for has_entry, has_exit in zip(all_entry, all_exit)),
            "entry_fraction": _ratio(sum(all_entry), n_states),
            "exit_fraction": _ratio(sum(all_exit), n_states), "both_fraction": _ratio(both, n_states),
            "nodes_without_entry": [node for node, covered in enumerate(all_entry) if not covered],
            "nodes_without_exit": [node for node, covered in enumerate(all_exit) if not covered],
            "entry_candidates_per_node": distribution([len(rows) for rows in entry]),
            "exit_candidates_per_node": distribution([len(rows) for rows in exit_]),
            "entry_relation_median_steps": distribution([row["median_steps"] for rows in entry for row in rows.values()]),
            "exit_relation_median_steps": distribution([row["median_steps"] for rows in exit_ for row in rows.values()]),
        },
        "task_summary": {
            "count": len(task_rows), "bottom_reachable": sum(row["bottom_reachable"] for row in task_rows),
            "structurally_supported": sum(row["status"] == "structurally_supported" for row in task_rows),
            "status_counts": dict(sorted(Counter(row["status"] for row in task_rows).items())),
            "learned_planning_or_execution_evaluated": False,
        },
        "tasks": task_rows,
    }
