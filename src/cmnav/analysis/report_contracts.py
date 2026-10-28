"""Historical report contracts. These adapters retain their original comparison units."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from cmnav.analysis.records import read_json

@dataclass(frozen=True)
class FrozenTaskCohort:
    task_count: int
    training_seed_count: int


COHORT_CONTRACT_PATH = Path(__file__).resolve().parents[3] / "configs/contracts/reporting_frozen_cohort.json"
_cohort = read_json(COHORT_CONTRACT_PATH)
if _cohort.get("schema") != "cmnav.historical_report_cohort.v1":
    raise ValueError("Unsupported historical report cohort contract")
if any(type(_cohort.get(key)) is not int or _cohort[key] <= 0 for key in ("task_count", "training_seed_count")):
    raise ValueError("Historical report cohort sizes must be positive integers")
FIVE_TASK_COHORT = FrozenTaskCohort(_cohort["task_count"], _cohort["training_seed_count"])
MODES = tuple(_cohort["subgoal_modes"])


INTERVALS = tuple(_cohort["subgoal_intervals"])


CONDITIONS = tuple(f"{mode}_{interval if interval is not None else 'final'}"
                   for mode in MODES for interval in INTERVALS)


COUNT_KEYS = ("decisions", "candidates", "model_transitions", "predicted_arrived_candidates",
              "graph_arrived_candidates", "invalid_candidates", "selected_candidates",
              "false_predicted_arrival_candidates")


PENDING_STATUSES = frozenset({"pending", "not_started", "unattempted", "not_attempted"})
PLANNING_COUNTS = (
    "decisions", "candidates", "model_transitions", "predicted_arrived_candidates",
    "continuous_valid_candidates", "invalid_candidates", "graph_arrived_candidates",
    "false_predicted_arrival_candidates", "missed_graph_arrival_candidates",
    "selected_candidates", "selected_continuous_valid", "selected_graph_arrived",
    "selected_false_predicted_arrival", "invalid_selected_edges", "nonfinite_diagnostic_values",
)


CACHE_COUNTS = (
    "fresh_planning_calls", "plans_stored", "replacements", "invalidations",
    "fallback_attempts", "fallback_cache_misses", "fallback_actions_selected",
    "fallback_actions_completed", "cached_plan_actions_completed",
    "invalid_fallback_actions", "invalid_fresh_actions",
)


NOISE_CONDITIONS = tuple(tuple(item) for item in _cohort["noise_conditions"])


CONTRASTS = tuple(tuple(item) for item in _cohort["paired_contrasts"])
