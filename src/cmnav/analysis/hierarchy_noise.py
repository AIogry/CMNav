"""Pure paired summaries of saved hierarchy noise-condition results.

No model execution, graph search, checkpoint loading, or file reads occur here.
The caller establishes shared model/task provenance before supplying results.
"""

from __future__ import annotations

import math
from collections import Counter
from copy import deepcopy
from typing import Any


_METHODS = ("dijkstra", "flat_gcml", "hierarchical_dijkstra", "hierarchical_gcml")
_PLANNING_COUNTS = (
    "low_queries", "high_queries", "low_model_transitions", "high_model_transitions",
    "candidates", "invalid_candidates", "false_predicted_arrivals",
    "selected_invalid_candidates", "selected_false_predicted_arrivals",
)


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def _identity(row: dict[str, Any]) -> tuple[str, int, int]:
    if row["method"] not in _METHODS:
        raise ValueError(f"Unknown hierarchy method: {row['method']}")
    return (row["method"], _count(row["seed"], "seed"),
            _count(row["task_index"], "task_index"))


def _completed(row: dict[str, Any], *, macros_only: bool) -> int:
    key = "completed_macros" if macros_only else "completed_stages"
    if key in row:
        return _count(row[key], key)
    stages = row["stages"]
    return sum(stage.get("target") is not None and stage.get("end") == stage["target"]
               and (not macros_only or stage["phase"] == "macro") for stage in stages)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in rows if not row["success"]]
    successful = [row for row in rows if row["success"]]
    planning = {key: sum(row["planning_summary"][key] for row in rows)
                for key in _PLANNING_COUNTS}
    terminations: Counter[str] = Counter()
    for row in rows:
        terminations.update(row["planning_summary"]["candidate_terminations"])
    planning["candidate_terminations"] = dict(sorted(terminations.items()))
    diversity = [row["planning_summary"]["distinct_sequences_sum"] for row in rows
                 if "distinct_sequences_sum" in row["planning_summary"]]
    planning["distinct_sequences_sum"] = sum(diversity) if diversity else None
    planning["distinct_sequences_available_rows"] = len(diversity)
    transitions = sum(row["model_transitions"] for row in rows)
    return {
        "count": len(rows), "successes": len(successful), "failures": len(failures),
        "success_fraction": len(successful) / len(rows),
        "successful_task_indices": [row["task_index"] for row in successful],
        "graph_steps_total": sum(row["graph_steps"] for row in rows),
        "successful_graph_steps_mean": (sum(row["graph_steps"] for row in successful) / len(successful)
                                        if successful else None),
        "model_transitions_total": transitions,
        "model_transitions_mean": transitions / len(rows),
        "model_transitions_max": max(row["model_transitions"] for row in rows),
        "graph_expansions_total": sum(row.get("graph_expansions", 0) for row in rows),
        "budget_blocked_queries": sum(row.get("budget_blocked_queries", 0) for row in rows),
        "completed_stages": sum(_completed(row, macros_only=False) for row in rows),
        "completed_macros": sum(_completed(row, macros_only=True) for row in rows),
        "failure_phases": dict(sorted(Counter(row.get("failure_phase") or "unclassified"
                                               for row in failures).items())),
        "failure_reasons": dict(sorted(Counter(row["reason"] for row in failures).items())),
        "planning": planning,
    }


def _paired(before: dict, after: dict, identities: list[tuple]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    changes = []
    for identity in identities:
        old, new = before[identity], after[identity]
        state = ("retained_success" if old["success"] else "gained_success") if new["success"] else (
            "lost_success" if old["success"] else "retained_failure")
        counts[state] += 1
        if state not in ("gained_success", "lost_success"):
            continue
        changes.append({
            "method": identity[0], "seed": identity[1], "task_index": identity[2],
            "change": state,
            "before": {key: old.get(key) for key in ("reason", "failure_phase", "graph_steps",
                                                      "model_transitions", "trace_path")},
            "after": {key: new.get(key) for key in ("reason", "failure_phase", "graph_steps",
                                                      "model_transitions", "trace_path")},
        })
    return {
        "pairs": len(identities),
        **{key: counts[key] for key in ("gained_success", "lost_success", "retained_success", "retained_failure")},
        "net_success_change": counts["gained_success"] - counts["lost_success"],
        "model_transitions_change": sum(after[key]["model_transitions"] - before[key]["model_transitions"]
                                        for key in identities),
        "changed_tasks": changes,
    }


def summarize_noise_conditions(conditions: list[dict]) -> dict:
    """Summarize four matched local/long noise settings without changing rows.

    Each condition supplies name, label, local_noise, long_noise, rows and a
    method -> source-run string mapping. The first condition is the declared
    baseline. Each noise dimension must have two distinct levels and all four
    combinations must be present.
    Results must contain identical method/seed/task identities across conditions.
    A source run reused for the same method/seed/task is counted once in the
    provenance inventory, even though it appears in multiple condition tables.
    """
    if len(conditions) != 4:
        raise ValueError("This comparison requires the four frozen 2 x 2 noise conditions.")
    names, pairs, lookups = set(), set(), {}
    for condition in conditions:
        name = condition["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Noise condition names must be unique nonempty strings.")
        names.add(name)
        pair = (condition["local_noise"], condition["long_noise"])
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in pair):
            raise ValueError("Noise scales must be finite nonnegative numbers.")
        if pair in pairs:
            raise ValueError("Duplicate noise condition.")
        pairs.add(pair)
        rows = condition["rows"]
        if not rows:
            raise ValueError("Each noise condition requires saved task rows.")
        lookup = {}
        for row in rows:
            identity = _identity(row)
            if identity in lookup:
                raise ValueError(f"Duplicate result identity in {name}: {identity}")
            lookup[identity] = row
            if type(row["success"]) is not bool:
                raise ValueError("Saved success must be boolean.")
            for key in ("graph_steps", "model_transitions"):
                _count(row[key], key)
            for key in ("graph_expansions", "budget_blocked_queries"):
                if key in row:
                    _count(row[key], key)
            for key in _PLANNING_COUNTS:
                _count(row["planning_summary"][key], f"planning_summary.{key}")
            if "distinct_sequences_sum" in row["planning_summary"]:
                _count(row["planning_summary"]["distinct_sequences_sum"], "planning_summary.distinct_sequences_sum")
            for key, value in row["planning_summary"]["candidate_terminations"].items():
                _count(value, f"candidate_terminations.{key}")
            _completed(row, macros_only=True)
            _completed(row, macros_only=False)
        for method in _METHODS:
            if not any(key[0] == method for key in lookup):
                raise ValueError(f"Noise comparison is missing method {method} in {name}.")
            source = condition["sources"].get(method)
            if not isinstance(source, str) or not source:
                raise ValueError(f"Missing explicit source run for {name}/{method}.")
        lookups[name] = lookup
    local_levels = {pair[0] for pair in pairs}
    long_levels = {pair[1] for pair in pairs}
    if (len(local_levels) != 2 or len(long_levels) != 2
            or pairs != {(local, long) for local in local_levels for long in long_levels}):
        raise ValueError("Noise conditions must form the complete Cartesian product of two levels per dimension.")
    baseline = conditions[0]
    base_pair = (baseline["local_noise"], baseline["long_noise"])
    other_local = next(value for value in local_levels if value != base_pair[0])
    other_long = next(value for value in long_levels if value != base_pair[1])
    reference = lookups[baseline["name"]]
    identities = sorted(reference)
    for condition in conditions:
        if set(lookups[condition["name"]]) != set(reference):
            raise ValueError("Conditions must contain identical method/seed/task identities.")
        for identity, row in lookups[condition["name"]].items():
            for key in ("start", "goal", "episode_id"):
                if key in reference[identity] and row.get(key) != reference[identity][key]:
                    raise ValueError(f"Task identity field {key} changed for {identity}.")
    groups = [(method, seed) for method in _METHODS
              for seed in sorted({key[1] for key in identities if key[0] == method})]
    summarized = []
    source_inventory: dict[tuple, dict] = {}
    for condition in conditions:
        name, lookup = condition["name"], lookups[condition["name"]]
        by_seed, by_method = [], []
        for method, seed in groups:
            keys = [key for key in identities if key[:2] == (method, seed)]
            by_seed.append({"method": method, "seed": seed,
                            **_aggregate([lookup[key] for key in keys]),
                            "paired_to_baseline": _paired(reference, lookup, keys)})
        for method in _METHODS:
            keys = [key for key in identities if key[0] == method]
            by_method.append({"method": method,
                              **_aggregate([lookup[key] for key in keys]),
                              "paired_to_baseline": _paired(reference, lookup, keys),
                              "source": condition["sources"][method],
                              "same_source_as_baseline": condition["sources"][method] == baseline["sources"][method]})
            for key in keys:
                provenance_key = (condition["sources"][method], *key)
                source_inventory.setdefault(provenance_key, {
                    "source": provenance_key[0], "method": method, "seed": key[1],
                    "task_index": key[2], "conditions": []})["conditions"].append(name)
        summarized.append({
            "name": name, "label": condition["label"],
            "local_noise": condition["local_noise"], "long_noise": condition["long_noise"],
            "sources": deepcopy(condition["sources"]), "row_count": len(lookup),
            "method_seed": by_seed, "methods": by_method,
            "rows": deepcopy(condition["rows"]),
        })
    lookup_pairs = {(condition["local_noise"], condition["long_noise"]): condition
                    for condition in summarized}
    interactions = []
    for method, seed in groups:
        values = {}
        for pair, condition in lookup_pairs.items():
            values[pair] = next(row["successes"] for row in condition["method_seed"]
                                if (row["method"], row["seed"]) == (method, seed))
        interactions.append({"method": method, "seed": seed,
                             "success_count_difference_of_differences":
                             values[(other_local, other_long)] - values[(other_local, base_pair[1])]
                             - values[(base_pair[0], other_long)] + values[base_pair]})
    return {
        "schema": "cmnav.hierarchy_noise_comparison.v1", "baseline": baseline["name"],
        "conditions": summarized, "methods": list(_METHODS),
        "task_indices": sorted({key[2] for key in identities}),
        "planning_seeds": sorted({key[1] for key in identities}),
        "condition_row_count": len(identities), "displayed_row_count": len(identities) * len(conditions),
        "unique_source_method_seed_task_count": len(source_inventory),
        "source_records": list(source_inventory.values()), "descriptive_interactions": interactions,
        "scope": {
            "execution": "abstract_graph_only", "model_updates": False,
            "local_noise": "low_entry_candidate_exit_candidate_macro_execution_queries",
            "long_noise": "high_gcml_queries_and_low_full_goal_direct_candidate_queries",
            "distance_definition": "query_role_proxy_not_measured_graph_distance",
            "paired_unit": "same_method_planning_seed_development_task",
            "provenance_validation": "caller_responsibility_input_hashes_and_frozen_parameters",
        },
    }
