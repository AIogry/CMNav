"""Freeze privileged teacher subgoals on existing official development tasks.

The teacher supplies the high-level solution: Dijkstra's weighted shortest path
on the fixed observed graph. GCML receives only the current subgoal identity.
The complete path and its next edge belong exclusively to this manifest's setup
and audit layer. These schedules diagnose execution with an oracle subgoal
teacher; success does not establish autonomous hierarchical planning.

Schedules are computed once from each task's original start and final goal.
They never change in response to the learner's position, candidate plans,
success, failure or other evaluation outcomes. No environment or model is
loaded, queried, trained or modified by this module.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any

from cmnav.memory.oracle_graph import OracleObservedGraph
from cmnav.runtime.artifacts import canonical_json_bytes


SCHEMA = "cmnav.teacher_subgoals.v1"
PROTOCOL = "fixed_reference_path_teacher_subgoals_v1"
ACCESS = "environment_service_and_audit_only"


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _source_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal digest")
    return value


def _intervals(value: Any) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("intervals must be an ordered sequence of positive integers")
    result = list(value)
    if not result or any(type(item) is not int or item < 1 for item in result):
        raise ValueError("intervals must contain positive integers, not booleans")
    if len(set(result)) != len(result):
        raise ValueError("intervals must not contain duplicates")
    return result


def _schedule(nodes: list[int], interval: int | None) -> dict[str, Any]:
    length = len(nodes) - 1
    boundaries = [] if interval is None else list(range(interval, length, interval))
    # Appending exactly once also handles length zero and exact multiples.
    boundaries.append(length)
    previous = [0, *boundaries[:-1]]
    return {
        "subgoal_nodes": [nodes[end] for end in boundaries],
        "segment_edge_counts": [end - begin for begin, end in zip(previous, boundaries)],
        "stage_reference_edge_ranges": [[begin, end] for begin, end in zip(previous, boundaries)],
    }


def build_teacher_subgoal_manifest(
    graph: OracleObservedGraph,
    supported_tasks: dict[str, Any],
    *,
    intervals: Sequence[int] = (4, 8, 16),
    source_model_sha256: str,
    source_graph_sha256: str,
    source_tasks_sha256: str,
) -> dict[str, Any]:
    """Build a deterministic manifest from already validated source artifacts.

    ``source_*_sha256`` identifies exact source file bytes; the caller checks
    those files. This pure function additionally fingerprints the supplied graph
    state and verifies the supported-task content hash and graph binding.
    Schedule edge ranges are zero-based, half-open indices into reference_edges.
    Thus range ``[0, 4]`` ends at reference_nodes[4] after exactly four edges.
    The ``final`` schedule keeps only the original final goal as a control.
    """
    if not isinstance(graph, OracleObservedGraph):
        raise ValueError("graph must be an OracleObservedGraph")
    parsed_intervals = _intervals(intervals)
    model_sha = _source_hash(source_model_sha256, "source_model_sha256")
    graph_sha = _source_hash(source_graph_sha256, "source_graph_sha256")
    tasks_sha = _source_hash(source_tasks_sha256, "source_tasks_sha256")
    if (not isinstance(supported_tasks, dict)
            or type(supported_tasks.get("schema_version")) is not int
            or supported_tasks.get("schema_version") != 1
            or supported_tasks.get("access") != "environment_service_only"
            or not isinstance(supported_tasks.get("payload"), dict)
            or supported_tasks.get("manifest_id") != _hash(supported_tasks["payload"])):
        raise ValueError("Invalid supported-task manifest schema or content hash")
    tasks_payload = supported_tasks["payload"]
    if (tasks_payload.get("protocol") != "graph_supported_official_pointnav_development_v1"
            or tasks_payload.get("graph_sha256") != graph_sha):
        raise ValueError("Supported tasks belong to another graph or protocol")
    tasks = tasks_payload.get("evaluation")
    if (not isinstance(tasks, list) or not tasks
            or len(tasks) != tasks_payload.get("evaluation_episodes")):
        raise ValueError("Supported-task evaluation records are missing or inconsistent")
    edges = graph.edges
    costs = graph.edge_mean_costs
    node_count = len(graph.positions)
    records = []
    identities = set()
    for task_index, task in enumerate(tasks):
        if not isinstance(task, dict) or not isinstance(task.get("coverage"), dict):
            raise ValueError(f"Task {task_index} has no endpoint coverage audit")
        identity = task.get("source_episode_id")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("Source episode IDs must be nonempty unique strings")
        identities.add(identity)
        coverage = task["coverage"]
        start, goal = coverage.get("start_node"), coverage.get("goal_node")
        if any(type(node) is not int or not 0 <= node < node_count for node in (start, goal)):
            raise ValueError(f"Task {identity} has invalid graph endpoint IDs")
        route = graph.shortest_path(start, goal)
        if route is None:
            raise ValueError(f"Task {identity} has no directed graph path")
        reference_cost = sum(costs[edge] for edge in route)
        recorded_cost = coverage.get("shortest_path_cost_m")
        if (coverage.get("status") != "eligible"
                or coverage.get("directed_graph_reachable") is not True
                or coverage.get("shortest_path_edges") != route
                or coverage.get("shortest_path_hops") != len(route)
                or type(recorded_cost) not in (int, float)
                or not math.isfinite(recorded_cost)
                or not math.isclose(recorded_cost, reference_cost, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError(f"Task {identity} differs from its frozen weighted-path audit")
        nodes = [start]
        for edge in route:
            source, target = edges[edge]
            if source != nodes[-1]:
                raise ValueError(f"Task {identity} contains a discontinuous reference path")
            nodes.append(target)
        if nodes[-1] != goal:
            raise ValueError(f"Task {identity} reference path misses the final goal")
        schedules = {"final": _schedule(nodes, None)}
        schedules.update({str(interval): _schedule(nodes, interval) for interval in parsed_intervals})
        records.append({
            "task_index": task_index,
            "source_episode_id": identity,
            "start_node": start,
            "goal_node": goal,
            "reference_edges": route,
            "reference_nodes": nodes,
            "reference_cost_m": reference_cost,
            "schedules": schedules,
        })
    payload = {
        "protocol": PROTOCOL,
        "purpose": "oracle_subgoal_teacher_diagnostic_on_frozen_supported_tasks",
        "source_model_sha256": model_sha,
        "source_graph_sha256": graph_sha,
        "source_tasks_sha256": tasks_sha,
        "source_tasks_manifest_id": supported_tasks["manifest_id"],
        "graph_state_sha256": _hash(graph.state_dict()),
        "graph_nodes": node_count,
        "graph_edges": len(edges),
        "intervals": parsed_intervals,
        "evaluation_episodes": len(records),
        "reference_path": "original_start_to_final_goal_weighted_Dijkstra_on_fixed_observed_graph",
        "reference_edge_cost": "mean_observed_transition_distance_m",
        "stage_reference_edge_ranges_convention": "zero_based_half_open_indices_into_reference_edges",
        "schedule_rule": "every_interval_edges_on_fixed_reference_path_then_final_goal_once",
        "schedule_changes_during_execution": False,
        "selection_depends_on_model_outputs": False,
        "teacher_privileges": ["fixed_observed_graph", "original_start_node", "final_goal_node",
                               "complete_weighted_shortest_path_solution"],
        "learner_subgoal_inputs": ["current_subgoal_node_only"],
        "withheld_teacher_information": ["complete_reference_path", "next_reference_edge",
                                          "remaining_subgoal_sequence"],
        "interpretation": "teacher_supplies_high_level_route; results_do_not_establish_autonomous_hierarchical_planning",
        "records": records,
    }
    return {"schema": SCHEMA, "schema_version": 1, "access": ACCESS,
            "manifest_id": _hash(payload), "payload": payload}


def validate_teacher_subgoal_manifest(
    manifest: dict[str, Any],
    graph: OracleObservedGraph,
    supported_tasks: dict[str, Any],
    *,
    source_model_sha256: str,
    source_graph_sha256: str,
    source_tasks_sha256: str,
) -> None:
    """Require exact source bindings and exact recomputation of every schedule.

    This validates supplied objects without I/O or environment/model access.
    The caller still checks file digests before supplying the source objects.
    A changed path, interval, stage boundary, cost or permission declaration
    changes the manifest identity; inconsistent edits are rejected.
    """
    if (not isinstance(manifest, dict)
            or set(manifest) != {"schema", "schema_version", "access", "manifest_id", "payload"}
            or manifest.get("schema") != SCHEMA
            or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
            or manifest.get("access") != ACCESS
            or not isinstance(manifest.get("payload"), dict)
            or manifest.get("manifest_id") != _hash(manifest["payload"])):
        raise ValueError("Invalid teacher-subgoal manifest schema or content hash")
    expected = build_teacher_subgoal_manifest(
        graph, supported_tasks, intervals=manifest["payload"].get("intervals"),
        source_model_sha256=source_model_sha256,
        source_graph_sha256=source_graph_sha256,
        source_tasks_sha256=source_tasks_sha256,
    )
    if canonical_json_bytes(manifest) != canonical_json_bytes(expected):
        raise ValueError("Teacher subgoals differ from their source graph, tasks, model or schedule rule")
