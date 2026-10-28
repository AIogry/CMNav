"""Shortest directed graph routes with an explicit search-work counter."""
from __future__ import annotations

import heapq
import math
from numbers import Integral, Real
from typing import Sequence


def shortest_path(
    n_states: int, edges: Sequence[Sequence[int]], costs: Sequence[float],
    start: int, goal: int,
) -> tuple[list[int] | None, int]:
    """Return edge IDs and settled-node count using deterministic Dijkstra.

    Ties preserve the original observed-graph implementation: the heap orders
    equal distances by node ID, outgoing edges retain their supplied order,
    and a predecessor changes only for a strictly smaller distance. Search
    work includes the popped goal; an initial goal requires zero expansions.
    """
    if isinstance(n_states, bool) or not isinstance(n_states, Integral) or n_states < 1:
        raise ValueError("n_states must be a positive integer")
    for value in (start, goal):
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < n_states:
            raise ValueError("Search endpoint is outside the graph vocabulary")
    if len(costs) != len(edges):
        raise ValueError("Graph costs must cover every edge")
    outgoing: list[list[int]] = [[] for _ in range(n_states)]
    for action, pair in enumerate(edges):
        if len(pair) != 2 or any(isinstance(node, bool) or not isinstance(node, Integral)
                                 or not 0 <= node < n_states for node in pair):
            raise ValueError("Graph edge is outside the node vocabulary")
        cost = costs[action]
        if (isinstance(cost, bool) or not isinstance(cost, Real)
                or not math.isfinite(float(cost)) or cost < 0):
            raise ValueError("Graph costs must be finite and nonnegative")
        outgoing[int(pair[0])].append(action)
    if start == goal:
        return [], 0
    distances = {int(start): 0.0}
    predecessor: dict[int, tuple[int, int]] = {}
    queue = [(0.0, int(start))]
    expansions = 0
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        expansions += 1
        if node == goal:
            path = []
            while node != start:
                node, action = predecessor[node]
                path.append(action)
            path.reverse()
            return path, expansions
        for action in outgoing[node]:
            target = int(edges[action][1])
            candidate = distance + float(costs[action])
            if not math.isfinite(candidate):
                raise OverflowError("Observed path cost exceeds finite floating-point range.")
            if candidate < distances.get(target, math.inf):
                distances[target] = candidate
                predecessor[target] = (node, action)
                heapq.heappush(queue, (candidate, target))
    return None, expansions
