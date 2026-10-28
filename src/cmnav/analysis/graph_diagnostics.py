"""Offline graph labels for diagnostics, never supplied to navigation policies."""
from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Any

import numpy as np

def reverse_distances(goal: int, reverse: list[list[tuple[int, float]]]) -> list[float]:
    distances = [math.inf] * len(reverse)
    distances[goal] = 0.
    queue = [(0., goal)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for source, cost in reverse[node]:
            candidate = distance + cost
            if candidate < distances[source]:
                distances[source] = candidate
                heapq.heappush(queue, (candidate, source))
    return distances

def optimal_edges(state: int, outgoing: list[list[int]], edges: list[list[int]],
                  costs: list[float], distances: list[float]) -> list[int]:
    if not math.isfinite(distances[state]):
        return []
    return [edge for edge in outgoing[state]
            if math.isclose(costs[edge] + distances[edges[edge][1]], distances[state],
                            rel_tol=1e-6, abs_tol=1e-6)]

def audit_actions(actions: list[int], state: int, goal: int, edges: list[list[int]]) -> dict[str, Any]:
    current = state
    prefix = 0
    first_invalid = None
    reached = state == goal
    for index, action in enumerate(actions):
        if type(action) is not int or action < 0 or action >= len(edges):
            first_invalid = {"index": index, "action": action, "reason": "edge_id_out_of_range"}
            break
        source, target = edges[action]
        if source != current:
            first_invalid = {"index": index, "action": action, "reason": "source_mismatch",
                             "expected_source": current, "actual_source": source, "edge_target": target}
            break
        current = target
        prefix += 1
        reached = reached or current == goal
    return {"continuous_valid": first_invalid is None,
            "valid_prefix_actions": prefix, "first_invalid": first_invalid,
            "valid_prefix_reached_goal": reached,
            "graph_arrived": first_invalid is None and current == goal}

def branch_scores(state: int, goal: int, selected: int | None, q: np.ndarray, w: np.ndarray,
                  legal: list[int], correct: list[int], noise: float) -> dict[str, Any]:
    utilities = w @ (q[:, goal] - q[:, state])
    norm = float(np.linalg.norm(utilities))
    normalized = utilities / max(norm, 1e-12)
    ranked = sorted(legal, key=lambda edge: (-float(normalized[edge]), edge))
    gap = float(normalized[ranked[0]] - normalized[ranked[1]]) if len(ranked) > 1 else None
    pairwise_sd = math.sqrt(2.) * noise
    ratio = gap / pairwise_sd if gap is not None and pairwise_sd > 0 else None
    wrong = [edge for edge in legal if edge not in correct]
    correct_margin = (float(max(normalized[edge] for edge in correct) - max(normalized[edge] for edge in wrong))
                      if correct and wrong else None)
    return {
        "current_node": state, "goal_node": goal, "selected_edge": selected,
        "legal_edges": legal, "shortest_path_first_edges": correct,
        "selected_edge_is_shortest_path_first_edge": selected in correct if selected is not None else None,
        "legal_raw_utilities": [float(utilities[edge]) for edge in legal],
        "legal_normalized_utilities": [float(normalized[edge]) for edge in legal],
        "all_actions_utility_l2_norm": norm, "normalization": "all_actions_l2",
        "noise_sigma": noise, "pairwise_noise_difference_sd": pairwise_sd,
        "top_two_legal_edges": ranked[:2], "top_two_normalized_gap": gap,
        "top_two_gap_over_pairwise_noise_sd": ratio,
        "top1_beats_top2_pairwise_probability": (.5 * (1. + math.erf(ratio / math.sqrt(2.)))) if ratio is not None else None,
        "best_correct_minus_best_incorrect_normalized": correct_margin,
        "correct_margin_over_pairwise_noise_sd": correct_margin / pairwise_sd if correct_margin is not None and pairwise_sd > 0 else None,
        "probability_scope": "one_unconditioned_two_action_Gaussian_comparison_not_full_candidate_or_navigation_success",
    }

def minimum_hops(goal: int, reverse: list[list[tuple[int, float]]]) -> list[int | None]:
    distances: list[int | None] = [None] * len(reverse)
    distances[goal] = 0
    queue = deque([goal])
    while queue:
        node = queue.popleft()
        for source, _ in reverse[node]:
            if distances[source] is None:
                distances[source] = distances[node] + 1
                queue.append(source)
    return distances

def can_complete(start: int, goal: int, blocked: set[int],
                 outgoing: list[list[int]], edges: list[list[int]]) -> bool:
    """Whether a simple completion can avoid every previously visited node."""
    if start in blocked:
        return False
    seen = blocked | {start}
    pending = [start]
    while pending:
        node = pending.pop()
        if node == goal:
            return True
        for action in outgoing[node]:
            target = edges[action][1]
            if target not in seen:
                seen.add(target)
                pending.append(target)
    return False

def score_rows(step: dict[str, Any], edges: list[list[int]]) -> list[dict[str, Any]]:
    return [{"edge": action, "target": edges[action][1],
             "raw_utility": step["eligible_raw_utilities"][index],
             "normalized_utility": step["eligible_normalized_utilities"][index],
             "noise": step["eligible_noise"][index],
             "normalized_utility_plus_noise": (step["eligible_normalized_utilities"][index]
                                               + step["eligible_noise"][index]),
             "selected": action == step["action"]}
            for index, action in enumerate(step["eligible_edges"])]

def all_goal_distances(graph):
    """Reverse Dijkstra plus BFS over saved mean costs and directed edges."""
    reverse = [[] for _ in graph.positions]
    for edge, (source, target) in enumerate(graph.edges):
        reverse[target].append((source, graph.edge_mean_costs[edge]))
    return ([reverse_distances(goal, reverse) for goal in range(len(reverse))],
            [minimum_hops(goal, reverse) for goal in range(len(reverse))])
