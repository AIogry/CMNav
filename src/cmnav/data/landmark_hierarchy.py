"""Derive a two-level graph exclusively from ordered, goal-independent replay.

The unit of distance is one observed directed graph edge, not a simulator
action. This module does not sample new walks, search paths, read navigation
tasks, or perform learning. Every nonzero relation has a continuous supporting
slice in one supplied replay trajectory. A new schema is required if these
experience or boundary semantics change.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from numbers import Integral
import random
import re
from statistics import median
from typing import Any


SCHEMA = "cmnav.landmark_hierarchy.v1"
_WALK_SCHEMA = "cmnav.graph_walks.v1"
_EXPERIENCE_KIND = "graph_replay_of_previously_observed_directed_edges"


def _integer(value: Any, name: str, *, minimum: int = 0,
             maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, not a boolean")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _sequence(value: Any, name: str) -> list:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an ordered list or tuple")
    return list(value)


def _validated_replay(walks: dict, n_states: int, edges: list
                      ) -> tuple[list[dict], list[tuple[int, int]], int]:
    """Validate identities and boundaries without trusting summary counters."""
    if not isinstance(walks, Mapping) or walks.get("schema") != _WALK_SCHEMA:
        raise ValueError(f"walks must have schema {_WALK_SCHEMA!r}")
    sampling = walks.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError("walks.sampling must be a mapping")
    if sampling.get("experience_kind") != _EXPERIENCE_KIND:
        raise ValueError("only replay of previously observed directed edges is supported")
    if _integer(sampling.get("environment_interactions"),
                "sampling.environment_interactions") != 0:
        raise ValueError("graph replay must declare zero environment interactions")
    if sampling.get("visual_observations_generated") is not False:
        raise ValueError("graph replay must declare visual_observations_generated=false")
    if sampling.get("evaluation_goals_used", False) is not False:
        raise ValueError("goal-conditioned replay cannot form this hierarchy")
    if _integer(sampling.get("n_states"), "sampling.n_states", minimum=1) != n_states:
        raise ValueError("sampling.n_states does not match the supplied graph")

    directed_edges = []
    for action, raw_edge in enumerate(_sequence(edges, "edges")):
        pair = _sequence(raw_edge, f"edges[{action}]")
        if len(pair) != 2:
            raise ValueError(f"edges[{action}] must have two node IDs")
        directed_edges.append(tuple(
            _integer(node, f"edges[{action}][{offset}]", maximum=n_states - 1)
            for offset, node in enumerate(pair)
        ))
    if len(set(directed_edges)) != len(directed_edges):
        raise ValueError("directed endpoint pairs must have unique action IDs")
    if _integer(sampling.get("n_edges"), "sampling.n_edges") != len(directed_edges):
        raise ValueError("sampling.n_edges does not match the supplied graph")

    trajectories = []
    total = 0
    edge_visits = [0] * len(directed_edges)
    state_visits = [0] * n_states
    starts = [0] * n_states
    raw_trajectories = _sequence(walks.get("trajectories"), "walks.trajectories")
    if not raw_trajectories:
        raise ValueError("at least one replay trajectory is required")
    if _integer(sampling.get("num_trajectories"), "sampling.num_trajectories") != len(raw_trajectories):
        raise ValueError("sampling.num_trajectories disagrees with the replay")
    for trajectory_index, row in enumerate(raw_trajectories):
        name = f"trajectories[{trajectory_index}]"
        if not isinstance(row, Mapping):
            raise ValueError(f"{name} must be a mapping")
        start = _integer(row.get("start_node"), f"{name}.start_node", maximum=n_states - 1)
        if not isinstance(row.get("end_reason"), str) or not row["end_reason"]:
            raise ValueError(f"{name}.end_reason must be a nonempty string")
        previous = start
        transitions = []
        nodes = [start]
        starts[start] += 1
        state_visits[start] += 1
        for step, transition in enumerate(_sequence(row.get("transitions"), f"{name}.transitions")):
            triple = _sequence(transition, f"{name}.transitions[{step}]")
            if len(triple) != 3:
                raise ValueError(f"{name}.transitions[{step}] must contain source, action, target")
            source = _integer(triple[0], f"{name}[{step}].source", maximum=n_states - 1)
            action = _integer(triple[1], f"{name}[{step}].action", maximum=len(directed_edges) - 1)
            target = _integer(triple[2], f"{name}[{step}].target", maximum=n_states - 1)
            if source != previous:
                raise ValueError(f"discontinuous replay at {name}.transitions[{step}]")
            if directed_edges[action] != (source, target):
                raise ValueError(f"action identity mismatch at {name}.transitions[{step}]")
            transitions.append((source, action, target))
            nodes.append(target)
            previous = target
            edge_visits[action] += 1
            state_visits[target] += 1
        total += len(transitions)
        trajectories.append({"start_node": start, "nodes": nodes,
                             "transitions": transitions, "end_reason": row["end_reason"]})

    if _integer(walks.get("emitted_transitions"), "walks.emitted_transitions") != total:
        raise ValueError("walks.emitted_transitions disagrees with actual transition count")
    coverage = walks.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("walks.coverage must be a mapping")
    for key, actual in (("state_visit_counts", state_visits),
                        ("edge_visit_counts", edge_visits),
                        ("start_node_counts", starts)):
        counts = _sequence(coverage.get(key), f"walks.coverage.{key}")
        parsed = [_integer(value, f"walks.coverage.{key}[{index}]")
                  for index, value in enumerate(counts)]
        if parsed != actual:
            raise ValueError(f"walks.coverage.{key} disagrees with actual replay")
    return trajectories, directed_edges, total


def build_landmark_hierarchy(
    walks: dict,
    *,
    n_states: int,
    edges: list,
    source_sha256: str,
    h_select: int,
    formation_transitions: int,
    macro_max_steps: int,
    attach_max_steps: int,
) -> dict:
    """Select landmarks once, then extract supported macro and attachment data.

    ``edges[action]`` defines the existing low-level action identity. High-level
    node IDs follow landmark discovery order; macro action IDs follow sorted
    high-level endpoint pairs. No edge is added without a replay witness.

    Selection consumes at most ``formation_transitions`` low-level transitions
    in the supplied order. Only the very first corpus start is seeded. Every
    independent reset sets the elapsed counter to zero; an existing landmark
    encounter also resets it. After ``h_select`` transitions without one, the
    current node becomes a landmark. No later reset start is seeded. The set
    freezes before all supplied trajectories are rescanned for relationships.

    Consecutive landmark occurrences define candidate macro segments. Self
    returns and segments longer than ``macro_max_steps`` split high-level
    training sequences; they are retained under ``skipped_segments``. Prefixes,
    tails, and trajectories with no landmark are recorded there as well.

    Ordinary node occurrences connect only to the nearest subsequent landmark
    (entry) and nearest preceding landmark (exit), independently limited by
    ``attach_max_steps``. Observed landmark occurrences have zero-step self
    attachments. Every support range is [start_step, end_step) in transition
    indices, with both endpoint states and all low-level edge IDs retained.

    The returned object contains no model weights or task-derived information.
    The caller is responsible for verifying the supplied source file SHA and
    preserving that file; this pure function cannot hash a file it does not read.
    """
    n_states = _integer(n_states, "n_states", minimum=1)
    h_select = _integer(h_select, "h_select", minimum=1)
    formation_transitions = _integer(formation_transitions, "formation_transitions")
    macro_max_steps = _integer(macro_max_steps, "macro_max_steps", minimum=1)
    attach_max_steps = _integer(attach_max_steps, "attach_max_steps")
    if not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("source_sha256 must be a lowercase SHA-256 hex digest")
    trajectories, directed_edges, total = _validated_replay(walks, n_states, edges)

    landmark_nodes = [trajectories[0]["start_node"]]
    landmark_ids = {landmark_nodes[0]: 0}
    selection_events = [{"landmark": 0, "node": landmark_nodes[0], "trajectory_index": 0,
                         "state_step": 0, "formation_transition_count": 0,
                         "reason": "first_corpus_start"}]
    used = 0
    formation_last = {"trajectory_index": 0, "state_step": 0}
    for trajectory_index, trajectory in enumerate(trajectories):
        if used >= formation_transitions:
            break
        elapsed = 0
        for step, (_, _, target) in enumerate(trajectory["transitions"], start=1):
            if used >= formation_transitions:
                break
            used += 1
            elapsed += 1
            formation_last = {"trajectory_index": trajectory_index, "state_step": step}
            if target in landmark_ids:
                elapsed = 0
            elif elapsed >= h_select:
                landmark_id = len(landmark_nodes)
                landmark_ids[target] = landmark_id
                landmark_nodes.append(target)
                selection_events.append({
                    "landmark": landmark_id, "node": target,
                    "trajectory_index": trajectory_index, "state_step": step,
                    "formation_transition_count": used, "reason": "elapsed_transition_limit",
                })
                elapsed = 0

    def support(trajectory_index: int, start: int, end: int) -> dict:
        trajectory = trajectories[trajectory_index]
        return {
            "source_sha256": source_sha256, "trajectory_index": trajectory_index,
            "start_step": start, "end_step": end, "length": end - start,
            "source_node": trajectory["nodes"][start], "target_node": trajectory["nodes"][end],
            "edge_ids": [row[1] for row in trajectory["transitions"][start:end]],
        }

    macro_supports: dict[tuple[int, int], list[dict]] = defaultdict(list)
    entry_supports: dict[tuple[int, int], list[dict]] = defaultdict(list)
    exit_supports: dict[tuple[int, int], list[dict]] = defaultdict(list)
    chunks = []
    skipped = []
    candidate_segments = 0
    low_transition_categories: Counter = Counter()
    landmark_occurrences = 0

    def skip(trajectory_index: int, start: int, end: int, reason: str) -> None:
        row = support(trajectory_index, start, end)
        row["reason"] = reason
        skipped.append(row)
        low_transition_categories[reason] += end - start

    for trajectory_index, trajectory in enumerate(trajectories):
        nodes = trajectory["nodes"]
        events = [(step, landmark_ids[node]) for step, node in enumerate(nodes)
                  if node in landmark_ids]
        landmark_occurrences += len(events)
        if not events:
            skip(trajectory_index, 0, len(nodes) - 1, "no_landmark")
        else:
            if events[0][0] > 0:
                skip(trajectory_index, 0, events[0][0], "prefix_before_first_landmark")
            if events[-1][0] < len(nodes) - 1:
                skip(trajectory_index, events[-1][0], len(nodes) - 1, "tail_after_last_landmark")

        chunk = []
        segment_index = 0
        for (start, source), (end, target) in zip(events, events[1:]):
            candidate_segments += 1
            reason = "self_landmark_return" if source == target else (
                "macro_length_limit" if end - start > macro_max_steps else None)
            if reason:
                skip(trajectory_index, start, end, reason)
                if chunk:
                    chunks.append((trajectory_index, segment_index, chunk))
                    segment_index += 1
                    chunk = []
                continue
            row = support(trajectory_index, start, end)
            pair = (source, target)
            macro_supports[pair].append(row)
            chunk.append((pair, row))
            low_transition_categories["retained_macro"] += end - start
        if chunk:
            chunks.append((trajectory_index, segment_index, chunk))

        # nearest past/future positions are computed in separate linear scans.
        preceding = None
        for step, node in enumerate(nodes):
            if node in landmark_ids:
                preceding = step
            if preceding is not None and step - preceding <= attach_max_steps:
                key = (node, landmark_ids[nodes[preceding]])
                exit_supports[key].append(support(trajectory_index, preceding, step))
        subsequent = None
        for step in range(len(nodes) - 1, -1, -1):
            node = nodes[step]
            if node in landmark_ids:
                subsequent = step
            if subsequent is not None and subsequent - step <= attach_max_steps:
                key = (node, landmark_ids[nodes[subsequent]])
                entry_supports[key].append(support(trajectory_index, step, subsequent))

    pairs = sorted(macro_supports)
    actions = {pair: action for action, pair in enumerate(pairs)}
    macro_edges = []
    for pair in pairs:
        supports = macro_supports[pair]
        lengths = [row["length"] for row in supports]
        macro_edges.append({
            "action": actions[pair], "source": pair[0], "target": pair[1],
            "source_node": landmark_nodes[pair[0]], "target_node": landmark_nodes[pair[1]],
            "support_count": len(supports), "lengths": lengths,
            "min_steps": min(lengths), "median_steps": float(median(lengths)),
            "max_steps": max(lengths), "supports": supports,
        })

    high_trajectories = []
    for trajectory_index, segment_index, chunk in chunks:
        high_trajectories.append({
            "trajectory_index": trajectory_index, "segment_index": segment_index,
            "start_node": chunk[0][0][0],
            "start_step": chunk[0][1]["start_step"], "end_step": chunk[-1][1]["end_step"],
            "transitions": [[pair[0], actions[pair], pair[1]] for pair, _ in chunk],
            "support_ranges": [row for _, row in chunk],
        })

    def attachments(records: dict[tuple[int, int], list[dict]]) -> list[list[dict]]:
        result: list[list[dict]] = [[] for _ in range(n_states)]
        for (node, landmark), supports in sorted(records.items()):
            supports.sort(key=lambda row: (row["trajectory_index"], row["start_step"], row["end_step"]))
            lengths = [row["length"] for row in supports]
            result[node].append({
                "landmark": landmark, "landmark_node": landmark_nodes[landmark],
                "support_count": len(supports), "min_steps": min(lengths),
                "median_steps": float(median(lengths)), "max_steps": max(lengths),
                "supports": supports,
            })
        return result

    entry = attachments(entry_supports)
    exit_relations = attachments(exit_supports)
    if sum(low_transition_categories.values()) != total:
        raise RuntimeError("internal error: extracted range accounting lost replay transitions")
    return {
        "schema": SCHEMA, "source_sha256": source_sha256,
        "experience_kind": _EXPERIENCE_KIND, "n_states": n_states,
        "landmark_nodes": landmark_nodes, "selection_events": selection_events,
        "edges": [list(pair) for pair in pairs], "macro_edges": macro_edges,
        "high_trajectories": high_trajectories,
        "attachments": {"entry": entry, "exit": exit_relations},
        "skipped_segments": sorted(skipped, key=lambda row: (row["trajectory_index"], row["start_step"], row["end_step"])),
        "config": {
            "h_select": h_select, "formation_transitions": formation_transitions,
            "macro_max_steps": macro_max_steps, "attach_max_steps": attach_max_steps,
            "step_unit": "low_level_directed_graph_edge",
            "selection_order": "supplied_trajectory_then_transition_order",
            "initial_landmark_policy": "first_corpus_start_only",
            "reset_policy": "zero_elapsed_counter_without_registering_start",
            "landmark_id_order": "discovery_order", "macro_action_order": "sorted_high_level_endpoint_pairs",
            "macro_extraction": "consecutive_frozen_landmark_occurrences_in_same_trajectory",
            "attachment_extraction": "nearest_landmark_occurrence_in_same_trajectory_per_direction",
            "self_macro_policy": "record_and_split_without_training",
            "overlong_macro_policy": "record_and_split_without_training",
            "support_range_unit": "half_open_low_level_transition_indices",
            "evaluation_goals_used": False, "reverse_relations_inferred": False,
        },
        "counters": {
            "low_states": n_states, "low_edges": len(directed_edges),
            "source_trajectories": len(trajectories), "source_transitions": total,
            "empty_source_trajectories": sum(not row["transitions"] for row in trajectories),
            "formation_budget_requested": formation_transitions,
            "formation_transitions_used": used,
            "formation_budget_exhausted": used == formation_transitions,
            "formation_last_position": formation_last,
            "landmarks": len(landmark_nodes), "landmark_occurrences": landmark_occurrences,
            "candidate_macro_segments": candidate_segments, "macro_edges": len(macro_edges),
            "retained_macro_segments": sum(len(value) for value in macro_supports.values()),
            "high_trajectories": len(high_trajectories),
            "skipped_segment_counts": dict(sorted(Counter(row["reason"] for row in skipped).items())),
            "low_transition_categories": dict(sorted(low_transition_categories.items())),
            "entry_pairs": len(entry_supports), "exit_pairs": len(exit_supports),
            "entry_supported_nodes": sum(bool(row) for row in entry),
            "exit_supported_nodes": sum(bool(row) for row in exit_relations),
        },
    }


def balanced_macro_replay(hierarchy: dict, *, visits_per_edge: int, seed: int) -> dict:
    """Replay witnessed, contiguous macro subsequences to an exact edge quota.

    A private ``random.Random(seed)`` shuffles high-trajectory record indices
    once per pass. Each record is scanned in its original order. An edge whose
    quota is already exhausted is skipped and splits the current output batch.
    A repeated action or target also splits the batch immediately before that
    edge, so the edge can begin a new batch if its quota remains unfinished.
    Output batches are therefore continuous slices of individual witness
    records, never paths assembled from different trajectory supports.

    The returned batches are suitable for replaying the same fixed epoch
    schedule. Repetition across passes is declared replay, not fresh experience.
    No original supports or hierarchy fields are modified. The caller records
    the hierarchy artifact fingerprint alongside the result.
    """
    visits_per_edge = _integer(visits_per_edge, "visits_per_edge", minimum=1)
    seed = _integer(seed, "seed")
    if not isinstance(hierarchy, Mapping) or hierarchy.get("schema") != SCHEMA:
        raise ValueError(f"hierarchy must have schema {SCHEMA!r}")
    source_sha256 = hierarchy.get("source_sha256")
    if not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("hierarchy.source_sha256 must be a lowercase SHA-256 digest")
    n_states = _integer(hierarchy.get("n_states"), "hierarchy.n_states", minimum=1)
    landmark_nodes = [
        _integer(node, f"landmark_nodes[{index}]", maximum=n_states - 1)
        for index, node in enumerate(_sequence(hierarchy.get("landmark_nodes"), "landmark_nodes"))
    ]
    if not landmark_nodes or len(set(landmark_nodes)) != len(landmark_nodes):
        raise ValueError("landmark_nodes must be nonempty and contain unique low-level IDs")
    edges = []
    for action, raw_edge in enumerate(_sequence(hierarchy.get("edges"), "hierarchy.edges")):
        pair = _sequence(raw_edge, f"hierarchy.edges[{action}]")
        if len(pair) != 2:
            raise ValueError(f"hierarchy.edges[{action}] must contain two landmark IDs")
        source, target = [
            _integer(value, f"hierarchy.edges[{action}][{index}]", maximum=len(landmark_nodes) - 1)
            for index, value in enumerate(pair)
        ]
        if source == target:
            raise ValueError("self macro edges are excluded by the hierarchy contract")
        edges.append((source, target))
    if not edges or edges != sorted(set(edges)):
        raise ValueError("at least one macro edge is required, ordered by unique endpoints")

    records = []
    covered_actions = set()
    for record_index, raw_record in enumerate(_sequence(hierarchy.get("high_trajectories"), "high_trajectories")):
        name = f"high_trajectories[{record_index}]"
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"{name} must be a mapping")
        trajectory_index = _integer(raw_record.get("trajectory_index"), f"{name}.trajectory_index")
        start_node = _integer(raw_record.get("start_node"), f"{name}.start_node", maximum=len(landmark_nodes) - 1)
        transitions = _sequence(raw_record.get("transitions"), f"{name}.transitions")
        supports = _sequence(raw_record.get("support_ranges"), f"{name}.support_ranges")
        if not transitions or len(supports) != len(transitions):
            raise ValueError(f"{name} requires nonempty transitions and one support range per transition")
        parsed = []
        parsed_supports = []
        previous = start_node
        previous_end = None
        for step, (raw_transition, raw_support) in enumerate(zip(transitions, supports)):
            triple = _sequence(raw_transition, f"{name}.transitions[{step}]")
            if len(triple) != 3:
                raise ValueError(f"{name}.transitions[{step}] must have source, action, target")
            source = _integer(triple[0], f"{name}[{step}].source", maximum=len(landmark_nodes) - 1)
            action = _integer(triple[1], f"{name}[{step}].action", maximum=len(edges) - 1)
            target = _integer(triple[2], f"{name}[{step}].target", maximum=len(landmark_nodes) - 1)
            if previous != source or edges[action] != (source, target):
                raise ValueError(f"noncontinuous or misidentified macro at {name}.transitions[{step}]")
            if not isinstance(raw_support, Mapping):
                raise ValueError(f"{name}.support_ranges[{step}] must be a mapping")
            support_name = f"{name}.support_ranges[{step}]"
            if raw_support.get("source_sha256") != source_sha256:
                raise ValueError(f"{support_name} does not refer to the declared replay source")
            if _integer(raw_support.get("trajectory_index"), f"{support_name}.trajectory_index") != trajectory_index:
                raise ValueError(f"{support_name} crosses a low-level trajectory boundary")
            start = _integer(raw_support.get("start_step"), f"{support_name}.start_step")
            end = _integer(raw_support.get("end_step"), f"{support_name}.end_step", minimum=start + 1)
            length = _integer(raw_support.get("length"), f"{support_name}.length", minimum=1)
            if end - start != length or (previous_end is not None and start != previous_end):
                raise ValueError(f"{support_name} has an invalid or discontinuous low-level range")
            low_source = _integer(raw_support.get("source_node"), f"{support_name}.source_node", maximum=n_states - 1)
            low_target = _integer(raw_support.get("target_node"), f"{support_name}.target_node", maximum=n_states - 1)
            if (low_source, low_target) != (landmark_nodes[source], landmark_nodes[target]):
                raise ValueError(f"{support_name} endpoint identities disagree with the macro edge")
            edge_ids = [
                _integer(value, f"{support_name}.edge_ids[{index}]")
                for index, value in enumerate(_sequence(raw_support.get("edge_ids"), f"{support_name}.edge_ids"))
            ]
            if len(edge_ids) != length:
                raise ValueError(f"{support_name} must retain every low-level edge ID")
            parsed.append([source, action, target])
            parsed_supports.append({**raw_support, "edge_ids": edge_ids})
            previous = target
            previous_end = end
            covered_actions.add(action)
        records.append({"trajectory_index": trajectory_index, "transitions": parsed,
                        "support_ranges": parsed_supports})
    if covered_actions != set(range(len(edges))):
        missing = sorted(set(range(len(edges))) - covered_actions)
        raise ValueError(f"high-level witness trajectories do not cover macro actions {missing}")

    rng = random.Random(seed)
    counts = [0] * len(edges)
    requested = len(edges) * visits_per_edge
    emitted = 0
    passes = 0
    result = []

    def append_range(record_index: int, start: int, end: int, pass_index: int) -> None:
        record = records[record_index]
        result.append({
            "start_node": record["transitions"][start][0],
            "transitions": [list(row) for row in record["transitions"][start:end]],
            "source_high_trajectory_index": record_index,
            "source_trajectory_index": record["trajectory_index"],
            "start_high_step": start, "end_high_step": end,
            "replay_pass": pass_index,
            "source_sha256": source_sha256,
            "support_ranges": [{**row, "edge_ids": list(row["edge_ids"])}
                               for row in record["support_ranges"][start:end]],
        })

    while emitted < requested:
        before = emitted
        record_order = list(range(len(records)))
        rng.shuffle(record_order)
        for record_index in record_order:
            start = None
            actions_in_batch: set[int] = set()
            targets_in_batch: set[int] = set()
            transitions = records[record_index]["transitions"]
            for step, (_, action, target) in enumerate(transitions):
                exhausted = counts[action] >= visits_per_edge
                repeated = action in actions_in_batch or target in targets_in_batch
                if exhausted or repeated:
                    if start is not None:
                        append_range(record_index, start, step, passes)
                    start = None
                    actions_in_batch.clear()
                    targets_in_batch.clear()
                    if exhausted:
                        continue
                if start is None:
                    start = step
                actions_in_batch.add(action)
                targets_in_batch.add(target)
                counts[action] += 1
                emitted += 1
            if start is not None:
                append_range(record_index, start, len(transitions), passes)
        passes += 1
        if emitted == before:
            raise RuntimeError("no progress while balancing witnessed macro replay")

    if emitted != requested or any(count != visits_per_edge for count in counts):
        raise RuntimeError("internal error: macro replay did not satisfy exact edge quotas")
    return {
        "schema": "cmnav.landmark_macro_replay.v1",
        "hierarchy_schema": SCHEMA, "source_sha256": source_sha256,
        "sampling_kind": "witnessed_macro_subsequence_replay",
        "seed": seed, "visits_per_edge": visits_per_edge,
        "edge_visit_counts": counts, "emitted_transitions": emitted,
        "trajectories": result,
        "sampling": {
            "n_states": len(landmark_nodes), "n_edges": len(edges),
            "num_trajectories": len(result), "source_high_trajectories": len(records),
            "passes": passes, "generator": "python.random.Random.private",
            "record_order": "fresh_shuffle_each_pass",
            "within_record_order": "unchanged",
            "split_on": ["exhausted_edge_quota", "repeated_action", "repeated_target"],
            "unique_target_and_action_indices_per_trajectory": True,
            "high_support_range_unit": "half_open_source_high_trajectory_transition_indices",
            "new_paths_assembled": False, "evaluation_goals_used": False,
            "coverage_scope": "all_supplied_witnessed_macro_edges",
        },
    }
