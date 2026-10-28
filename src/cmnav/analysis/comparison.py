"""Reusable paired comparisons and additive counters; repetitions are not independent tasks."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


def sum_counters(values: Iterable[Mapping[str, int]]) -> dict[str, int]:
    """Keep Counter addition semantics used by the historical analysis."""
    return dict(sum((Counter(value) for value in values), Counter()))


def paired_success_counts(before: Mapping[Any, Mapping[str, Any]],
                          after: Mapping[Any, Mapping[str, Any]]) -> dict[str, int]:
    matched = before.keys() & after.keys()
    return {
        "improved_repetitions": sum(not before[key]["success"] and after[key]["success"] for key in matched),
        "worsened_repetitions": sum(before[key]["success"] and not after[key]["success"] for key in matched),
        "unchanged_repetitions": sum(before[key]["success"] == after[key]["success"] for key in matched),
        "unmatched_baseline_repetitions": len(before.keys() - after.keys()),
        "unmatched_treatment_repetitions": len(after.keys() - before.keys()),
    }
