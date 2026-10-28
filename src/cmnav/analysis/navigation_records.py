"""Validated navigation logs and physical/graph execution statistics shared by reports."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from cmnav.analysis.records import read_json, read_rows as read_lines

ACTIONS = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")

def load_method(run: Path, task: str, method: str) -> dict[str, Any]:
    directory = run / "evaluation" / method / task
    files = {name: directory / name for name in
             ("summary.json", "execution.jsonl", "steps.jsonl", "planning.jsonl")}
    summary = read_json(files["summary.json"])
    execution = read_lines(files["execution.jsonl"])
    observations = read_lines(files["steps.jsonl"])
    planning = read_lines(files["planning.jsonl"])
    if not execution or len(execution) != len(observations):
        raise ValueError(f"{method}: execution and per-step records are missing or misaligned.")
    for index, (pose, observation) in enumerate(zip(execution, observations)):
        if pose["step"] != index or observation["observation_step"] != index:
            raise ValueError(f"{method}: global step indices must be consecutive from zero.")
        if pose["action"] != observation["action_from_previous_observation"]:
            raise ValueError(f"{method}: action records disagree at step {index}.")
    if summary["steps"] != len(execution) - 1:
        raise ValueError(f"{method}: final step count disagrees with its summary.")
    positions = np.asarray([row["position"] for row in execution], dtype=float)
    rotations = np.asarray([row["rotation_xyzw"] for row in execution], dtype=float)
    if positions.shape != (len(execution), 3) or not np.isfinite(positions).all():
        raise ValueError(f"{method}: invalid world positions.")
    if rotations.shape != (len(execution), 4) or not np.isfinite(rotations).all():
        raise ValueError(f"{method}: invalid xyzw rotations.")
    norm = np.linalg.norm(rotations, axis=1)
    if np.any(norm < 1e-12):
        raise ValueError(f"{method}: zero quaternion.")
    rotations = rotations / norm[:, None]
    action_counts = Counter(row["action"] for row in execution[1:])
    if set(action_counts) - set(ACTIONS):
        raise ValueError(f"{method}: unknown primitive action in saved logs.")
    displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    metrics = [row["evaluation_metrics"] for row in observations]
    distances = np.asarray([
        np.nan if row.get("distance_to_goal") is None else row["distance_to_goal"]
        for row in metrics
    ], dtype=float)
    forward_flags = [row["action"] == "MOVE_FORWARD" for row in execution[1:]]
    stats = {
        "episode_id": str(summary["episode_id"]), "steps": len(execution) - 1,
        "success": float(summary["metrics"]["success"]),
        "spl": float(summary["metrics"]["spl"]), "end_reason": summary["end_reason"],
        "actions": {action: action_counts[action] for action in ACTIONS},
        "path_length_metres": float(displacements.sum()),
        "collisions": int(summary["metrics"].get("collisions", {}).get("count", 0)),
        "forward_no_motion": sum(
            bool(forward and distance < 1e-4)
            for forward, distance in zip(forward_flags, displacements)
        ),
        "graph_hops_attempted": summary.get("graph_hops_attempted", 0),
        "graph_hops_completed": summary.get("graph_hops_completed", 0),
        "height_min_metres": float(positions[:, 1].min()),
        "height_max_metres": float(positions[:, 1].max()),
        "final_goal_distance_metres": float(distances[-1]) if np.isfinite(distances[-1]) else None,
        "local_execution_statuses": dict(Counter(
            row["status"] for row in planning if row.get("kind") == "local_execution"
        )),
    }
    return {"summary": summary, "execution": execution, "planning": planning,
            "positions": positions, "rotations": rotations, "distances": distances,
            "stats": stats, "files": files}


def graph_decisions(planning: list[dict[str, Any]], edges: list[list[int]]) -> dict[str, Any]:
    attempts: Counter[int] = Counter()
    completed: Counter[int] = Counter()
    local_steps: defaultdict[int, list[int]] = defaultdict(list)
    sequence = []
    pending = None
    for row in planning:
        if row.get("kind") == "graph_decision":
            pending = row.get("selected_edge")
            if pending is not None:
                if type(pending) is not int or not 0 <= pending < len(edges):
                    raise ValueError("Planning log contains an invalid selected edge.")
                if edges[pending][0] != row["current_node"]:
                    raise ValueError("Selected edge source disagrees with the current node.")
                attempts[pending] += 1
                sequence.append(pending)
        elif row.get("kind") == "local_execution" and pending is not None:
            completed[pending] += int(row["status"] == "arrived")
            local_steps[pending].append(int(row["steps"]))
            pending = None
    pair_counts: Counter[tuple[int, int]] = Counter()
    for edge, count in attempts.items():
        pair_counts[tuple(sorted(edges[edge]))] += count
    dominant = pair_counts.most_common(1)[0][0] if pair_counts else None
    ranked = [
        {"edge_index": edge, "source": edges[edge][0], "target": edges[edge][1],
         "attempts": count, "completed": completed[edge],
         "local_primitive_steps": local_steps[edge]}
        for edge, count in attempts.most_common()
    ]
    return {"ranked_edges": ranked, "selected_edge_sequence": sequence,
            "dominant_node_pair": list(dominant) if dominant is not None else None,
            "dominant_pair_attempts": pair_counts[dominant] if dominant is not None else 0,
            "completion_count_method": "Next local_execution record after each graph_decision; arrived only."}
