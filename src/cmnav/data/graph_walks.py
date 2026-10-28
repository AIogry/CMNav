"""Seeded trajectory replay of an explicitly supplied observed directed graph.

These are graph-domain training sequences, not newly collected environment
experience: no Habitat actions, RGB-D observations, or extra graph edges are
generated. Every sampled transition refers to an edge supplied by the caller.
The caller retains the original graph fingerprint and reference-data provenance
alongside this derived artifact; this module neither loads them nor reads goals.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any, Sequence


SCHEMA = "cmnav.graph_walks.v1"


def _integer(value: Any, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, not a boolean")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {interval}")
    return result


def _edges(value: Any, n_states: int) -> list[tuple[int, int]]:
    if isinstance(value, (str, bytes, set, frozenset, Mapping)):
        raise ValueError("edges must be an ordered collection of (source, target) pairs")
    try:
        supplied = list(value)
    except TypeError as exc:
        raise ValueError("edges must be an ordered collection of (source, target) pairs") from exc
    result: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for edge_id, edge in enumerate(supplied):
        if isinstance(edge, (str, bytes, set, frozenset, Mapping)):
            raise ValueError(f"edges[{edge_id}] must contain exactly source and target IDs")
        try:
            pair = list(edge)
        except TypeError as exc:
            raise ValueError(f"edges[{edge_id}] must contain exactly source and target IDs") from exc
        if len(pair) != 2:
            raise ValueError(f"edges[{edge_id}] must contain exactly source and target IDs")
        source, target = (
            _integer(node, f"edges[{edge_id}].{name}", minimum=0, maximum=n_states - 1)
            for node, name in zip(pair, ("source", "target"))
        )
        parsed = (source, target)
        if parsed in seen:
            raise ValueError("each directed (source, target) edge must have one action ID")
        seen.add(parsed)
        result.append(parsed)
    return result


def random_walks(
    n_states: int,
    edges: Sequence[tuple[int, int]],
    *,
    num_trajectories: int = 200,
    trajectory_length: int = 32,
    seed: int = 7,
) -> dict[str, Any]:
    """Sample graph walks with uniform starts and uniform valid neighbours.

    ``edges[e] == (s, t)`` defines the existing directed graph action ``e``;
    the input order fixes these action IDs. Outgoing neighbours are ordered by
    target node ID before uniform sampling. Duplicate directed edges, boolean
    IDs, and nodes outside ``[0, n_states)`` are rejected. Explicit self edges
    are allowed; missing reverse edges are never added.

    As in the reference notebook, ``trajectory_length`` counts visited states,
    including the initial state: length 32 allows at most 31 transitions.
    All starting nodes are drawn together before sampling any walk transitions,
    matching the reference notebook's random draw order. Each walk starts
    uniformly among *all* nodes, including sinks and isolated nodes.
    A dead end terminates that walk without padding or a hidden restart.
    Empty walks remain in the result, and callers must skip them for training.
    If the length budget is exhausted exactly on a sink, ``length_limit`` takes
    precedence because no further transition is attempted.

    An isolated, explicitly CPU PyTorch generator makes sampling reproducible
    for the same inputs and PyTorch implementation without changing the global
    RNG. Coverage is measured, not guaranteed. This function does not execute
    the graph walk in a physical environment or provide visual observations.
    """
    n_states = _integer(n_states, "n_states", minimum=1, maximum=2**63 - 1)
    num_trajectories = _integer(num_trajectories, "num_trajectories", minimum=1)
    trajectory_length = _integer(trajectory_length, "trajectory_length", minimum=1)
    seed = _integer(seed, "seed", minimum=0, maximum=2**64 - 1)
    directed_edges = _edges(edges, n_states)

    # Import only when sampling is requested; importing data contracts stays light.
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    outgoing: list[list[int]] = [[] for _ in range(n_states)]
    for edge_id, (source, _) in enumerate(directed_edges):
        outgoing[source].append(edge_id)
    for candidates in outgoing:
        candidates.sort(key=lambda edge_id: directed_edges[edge_id][1])

    trajectories = []
    state_visit_counts = [0] * n_states
    start_node_counts = [0] * n_states
    edge_visit_counts = [0] * len(directed_edges)
    emitted_transitions = 0
    empty_trajectories = 0
    dead_end_trajectories = 0
    starts = torch.randint(0, n_states, (num_trajectories,), generator=generator,
                           device="cpu").tolist()
    for start in starts:
        node = start
        start_node_counts[start] += 1
        state_visit_counts[start] += 1
        transitions = []
        end_reason = "length_limit"
        for _ in range(trajectory_length - 1):
            candidates = outgoing[node]
            if not candidates:
                end_reason = "dead_end"
                dead_end_trajectories += 1
                break
            choice = int(torch.randint(len(candidates), (1,), generator=generator,
                                       device="cpu").item())
            edge_id = candidates[choice]
            target = directed_edges[edge_id][1]
            transitions.append([node, edge_id, target])
            edge_visit_counts[edge_id] += 1
            state_visit_counts[target] += 1
            emitted_transitions += 1
            node = target
        empty_trajectories += int(not transitions)
        trajectories.append({"start_node": start, "transitions": transitions,
                             "end_reason": end_reason})

    used_edges = sum(count > 0 for count in edge_visit_counts)
    return {
        "schema": SCHEMA,
        "sampling": {
            "experience_kind": "graph_replay_of_previously_observed_directed_edges",
            "environment_interactions": 0,
            "visual_observations_generated": False,
            "n_states": n_states,
            "n_edges": len(directed_edges),
            "num_trajectories": num_trajectories,
            "trajectory_length": trajectory_length,
            "trajectory_length_unit": "visited_states_including_start",
            "max_transitions_per_trajectory": trajectory_length - 1,
            "seed": seed,
            "backend": "torch",
            "backend_version": str(torch.__version__),
            "generator_device": "cpu",
            "start_distribution": "uniform_all_nodes",
            "start_draw_order": "all_starts_in_one_draw_before_any_walk_transitions",
            "transition_distribution": "uniform_observed_outgoing_neighbors",
            "neighbor_order": "ascending_target_node_id",
            "edge_id_order": "input_edge_sequence",
            "dead_end_behavior": "terminate_without_padding_or_restart",
            "coverage_guaranteed": False,
        },
        "trajectories": trajectories,
        "emitted_transitions": emitted_transitions,
        "coverage": {
            "state_visit_counts": state_visit_counts,
            "start_node_counts": start_node_counts,
            "edge_visit_counts": edge_visit_counts,
            "state_visits": num_trajectories + emitted_transitions,
            "unique_states": sum(count > 0 for count in state_visit_counts),
            "unique_edges": used_edges,
            "unique_state_action_pairs": used_edges,
            "unused_edge_ids": [edge_id for edge_id, count in enumerate(edge_visit_counts)
                                if count == 0],
            "dead_end_trajectories": dead_end_trajectories,
            "empty_trajectories": empty_trajectories,
        },
    }


def balanced_edge_walks(
    n_states: int,
    edges: Sequence[tuple[int, int]],
    *,
    visits_per_edge: int = 20,
    trajectory_length: int = 32,
    seed: int = 7,
) -> dict[str, Any]:
    """Replay every observed directed edge an exact, goal-independent quota.

    Input order fixes action IDs; no reverse edge or evaluation goal is added.
    Each new trajectory begins with a uniformly tie-broken least-visited edge
    whose quota is unfinished. Continuations use the same preference among
    outgoing edges with quota remaining and a target not visited in this walk.
    This is balanced replay, not a uniform random walk or new environment data.

    ``trajectory_length`` counts states including the start and must be at
    least two. Walks have unique nodes except that an explicitly supplied self
    edge can start a singleton walk. In particular, no trajectory repeats a
    target index or an action index, avoiding ambiguous repeated-index Q/V
    writes when the learner consumes one complete trajectory per batch.

    A new walk always consumes an unfinished edge. Thus all quotas are met
    after exactly ``len(edges) * visits_per_edge`` transitions, regardless of
    tie choices, sinks, disconnected components, cycles, or self edges. State
    and edge counts include independent restarts, which are graph replay and
    are not physical traversals between trajectories.

    Termination precedence is: all quotas met, initial self edge, length limit,
    actual graph dead end, exhausted outgoing quotas, then revisit prevention.
    A private CPU PyTorch generator is used only to break equally used ties.
    """
    n_states = _integer(n_states, "n_states", minimum=1, maximum=2**63 - 1)
    visits_per_edge = _integer(visits_per_edge, "visits_per_edge", minimum=1)
    trajectory_length = _integer(trajectory_length, "trajectory_length", minimum=2)
    seed = _integer(seed, "seed", minimum=0, maximum=2**64 - 1)
    directed_edges = _edges(edges, n_states)
    if not directed_edges:
        raise ValueError("balanced_edge_walks requires at least one observed directed edge")

    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    outgoing: list[list[int]] = [[] for _ in range(n_states)]
    for edge_id, (source, _) in enumerate(directed_edges):
        outgoing[source].append(edge_id)
    for candidates in outgoing:
        candidates.sort(key=lambda edge_id: directed_edges[edge_id][1])

    edge_visit_counts = [0] * len(directed_edges)
    state_visit_counts = [0] * n_states
    start_node_counts = [0] * n_states
    termination_reason_counts = {
        "all_edge_quotas_met": 0,
        "self_edge": 0,
        "length_limit": 0,
        "dead_end": 0,
        "outgoing_edge_quotas_exhausted": 0,
        "revisit_prevented": 0,
    }
    trajectories = []
    emitted_transitions = 0
    requested_transitions = len(directed_edges) * visits_per_edge

    def choose_least_used(candidates: list[int]) -> int:
        minimum_visits = min(edge_visit_counts[edge_id] for edge_id in candidates)
        tied = [edge_id for edge_id in candidates
                if edge_visit_counts[edge_id] == minimum_visits]
        choice = int(torch.randint(len(tied), (1,), generator=generator,
                                   device="cpu").item())
        return tied[choice]

    while emitted_transitions < requested_transitions:
        unfinished = [edge_id for edge_id, count in enumerate(edge_visit_counts)
                      if count < visits_per_edge]
        edge_id = choose_least_used(unfinished)
        start, _ = directed_edges[edge_id]
        start_node_counts[start] += 1
        state_visit_counts[start] += 1
        visited_nodes = {start}
        used_actions: set[int] = set()
        transitions = []

        while True:
            source, target = directed_edges[edge_id]
            transitions.append([source, edge_id, target])
            used_actions.add(edge_id)
            visited_nodes.add(target)
            edge_visit_counts[edge_id] += 1
            state_visit_counts[target] += 1
            emitted_transitions += 1

            if emitted_transitions == requested_transitions:
                end_reason = "all_edge_quotas_met"
                break
            if source == target:
                end_reason = "self_edge"
                break
            if len(transitions) == trajectory_length - 1:
                end_reason = "length_limit"
                break
            if not outgoing[target]:
                end_reason = "dead_end"
                break
            available = [candidate for candidate in outgoing[target]
                         if edge_visit_counts[candidate] < visits_per_edge]
            if not available:
                end_reason = "outgoing_edge_quotas_exhausted"
                break
            candidates = [candidate for candidate in available
                          if directed_edges[candidate][1] not in visited_nodes
                          and candidate not in used_actions]
            if not candidates:
                end_reason = "revisit_prevented"
                break
            edge_id = choose_least_used(candidates)

        termination_reason_counts[end_reason] += 1
        trajectories.append({"start_node": start, "transitions": transitions,
                             "end_reason": end_reason})

    repeated_target_counts = []
    repeated_action_counts = []
    for trajectory in trajectories:
        targets = [transition[2] for transition in trajectory["transitions"]]
        actions = [transition[1] for transition in trajectory["transitions"]]
        repeated_target_counts.append(len(targets) - len(set(targets)))
        repeated_action_counts.append(len(actions) - len(set(actions)))
    used_edges = sum(count > 0 for count in edge_visit_counts)
    num_trajectories = len(trajectories)

    return {
        "schema": SCHEMA,
        "sampling": {
            "policy": "balanced_observed_edge_quota_without_node_revisits",
            "experience_kind": "graph_replay_of_previously_observed_directed_edges",
            "environment_interactions": 0,
            "visual_observations_generated": False,
            "evaluation_goals_used": False,
            "n_states": n_states,
            "n_edges": len(directed_edges),
            "num_trajectories": num_trajectories,
            "num_trajectories_policy": "determined_by_edge_quotas_and_walk_termination",
            "visits_per_edge": visits_per_edge,
            "requested_transitions": requested_transitions,
            "trajectory_length": trajectory_length,
            "trajectory_length_unit": "visited_states_including_start",
            "max_transitions_per_trajectory": trajectory_length - 1,
            "seed": seed,
            "backend": "torch",
            "backend_version": str(torch.__version__),
            "generator_device": "cpu",
            "start_distribution": "least_used_unfinished_edge_with_uniform_tie_breaking",
            "start_draw_order": "one_start_edge_at_each_independent_restart",
            "transition_distribution": "least_used_quota_remaining_outgoing_edge_with_uniform_tie_breaking",
            "neighbor_order": "ascending_target_node_id",
            "start_edge_order": "input_edge_sequence",
            "edge_id_order": "input_edge_sequence",
            "revisit_policy": "no_repeated_nodes_except_initial_singleton_self_edge",
            "unique_target_and_action_indices_per_trajectory": True,
            "dead_end_behavior": "terminate_current_walk_then_restart_at_unfinished_edge",
            "termination_precedence": list(termination_reason_counts),
            "coverage_guaranteed": True,
            "coverage_scope": "all_supplied_directed_edges_not_isolated_states",
        },
        "trajectories": trajectories,
        "emitted_transitions": emitted_transitions,
        "coverage": {
            "state_visit_counts": state_visit_counts,
            "start_node_counts": start_node_counts,
            "edge_visit_counts": edge_visit_counts,
            "state_visits": num_trajectories + emitted_transitions,
            "unique_states": sum(count > 0 for count in state_visit_counts),
            "unique_edges": used_edges,
            "unique_state_action_pairs": used_edges,
            "unused_edge_ids": [edge_id for edge_id, count in enumerate(edge_visit_counts)
                                if count == 0],
            "dead_end_trajectories": termination_reason_counts["dead_end"],
            "empty_trajectories": 0,
            "termination_reason_counts": termination_reason_counts,
            "min_edge_visits": min(edge_visit_counts),
            "max_edge_visits": max(edge_visit_counts),
            "edge_visit_count_summary": {
                "minimum": min(edge_visit_counts),
                "maximum": max(edge_visit_counts),
                "mean": emitted_transitions / len(directed_edges),
                "total": emitted_transitions,
                "edges_at_quota": sum(count == visits_per_edge for count in edge_visit_counts),
                "edges_below_quota": sum(count < visits_per_edge for count in edge_visit_counts),
                "edges_above_quota": sum(count > visits_per_edge for count in edge_visit_counts),
            },
            "trajectories_with_repeated_targets": sum(count > 0 for count in repeated_target_counts),
            "trajectories_with_repeated_actions": sum(count > 0 for count in repeated_action_counts),
            "repeated_target_occurrences_beyond_first": sum(repeated_target_counts),
            "repeated_action_occurrences_beyond_first": sum(repeated_action_counts),
        },
    }
