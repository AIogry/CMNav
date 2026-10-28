"""Bounded two-level GCML navigation on an explicitly observed abstract graph.

The evaluator confirms arrivals using actual node IDs. Learned plans are
selected before any graph audit, then executed without recovery or retry.
Both flat and local GCML execute a cached complete prediction open loop.
This module neither trains models nor writes experiment artifacts.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Any

from cmnav.evaluation.gcml_diagnostics import audit_graph_actions
from cmnav.memory.graph_search import shortest_path


DEFAULT_CONFIG = {
    "candidates": 64,
    "flat_candidates": 1024,
    "local_horizon": 8,
    "high_horizon": 32,
    "flat_horizon": 64,
    "noise_scale": 0.02,
    "local_noise_scale": None,
    "high_noise_scale": None,
    "flat_noise_scale": None,
    "affordance_threshold": 0.5,
    "association_radius": 0.1,
    "association_margin": 0.1,
    "latent_projection": "none",
    "noise_schedule": "indexed_full",
    "cycle_policy": "terminate_on_retrieved_state",
    "endpoint_candidates": 3,
    "max_graph_steps": 500,
    "max_model_transitions": 65536,
}
METHODS = frozenset({"flat_gcml", "hierarchical_dijkstra", "hierarchical_gcml", "dijkstra"})


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _configuration(config: dict[str, Any]) -> dict[str, Any]:
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown hierarchical evaluation settings: {sorted(unknown)}")
    result = {**DEFAULT_CONFIG, **config}
    for key in ("candidates", "flat_candidates", "local_horizon", "high_horizon", "flat_horizon",
                "endpoint_candidates", "max_graph_steps", "max_model_transitions"):
        result[key] = _integer(result[key], key, minimum=1)
    for key in ("noise_scale", "affordance_threshold", "association_radius", "association_margin"):
        value = result[key]
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(float(value)) or value < 0):
            raise ValueError(f"{key} must be finite and nonnegative")
        result[key] = float(value)
    for key in ("local_noise_scale", "high_noise_scale", "flat_noise_scale"):
        value = result[key]
        if value is None:
            continue
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(float(value)) or value < 0):
            raise ValueError(f"{key} must be null or finite and nonnegative")
        result[key] = float(value)
    if result["association_radius"] <= 0:
        raise ValueError("association_radius must be positive")
    if result["affordance_threshold"] > 1 or result["association_margin"] > 1:
        raise ValueError("Affordance threshold and association margin cannot exceed one")
    if result["latent_projection"] not in {"none", "confident_nearest"}:
        raise ValueError("Unknown latent projection")
    if result["noise_schedule"] not in {"legacy_active", "indexed_full"}:
        raise ValueError("Unknown planning noise schedule")
    if result["cycle_policy"] not in {"none", "terminate_on_retrieved_state"}:
        raise ValueError("Unknown planning cycle policy")
    return result


def resolved_noise_scales(config: dict[str, Any]) -> dict[str, float]:
    """Resolve optional role overrides; omitted overrides preserve legacy noise."""
    settings = _configuration(config)
    return {role: settings[f"{role}_noise_scale"] if settings[f"{role}_noise_scale"] is not None
            else settings["noise_scale"] for role in ("local", "high", "flat")}


def _edges(raw: list, n_states: int) -> list[tuple[int, int]]:
    result = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each edge must contain source and target")
        source, target = (_integer(value, "edge endpoint") for value in pair)
        if source >= n_states or target >= n_states:
            raise ValueError("Edge endpoint is outside node vocabulary")
        result.append((source, target))
    if len(result) != len(set(result)):
        raise ValueError("Action vocabulary contains duplicate directed edges")
    return result


def _outgoing(n_states: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    result: list[list[int]] = [[] for _ in range(n_states)]
    for action, (source, _) in enumerate(edges):
        result[source].append(action)
    return result


@dataclass
class _Task:
    config: dict[str, Any]
    low_edges: list[tuple[int, int]]
    low_outgoing: list[list[int]]
    high_edges: list[tuple[int, int]]
    high_outgoing: list[list[int]]
    high_costs: list[float]
    goal: int
    current: int
    queries: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(default_factory=list)
    model_transitions: int = 0
    graph_expansions: int = 0
    budget_blocked_queries: int = 0

    def query(self, model, start: int, goal: int, *, level: str,
              phase: str, horizon: int, candidate_limit: int | None = None) -> dict[str, Any]:
        """Generate once; actual graph audits are diagnostic-only fields."""
        edges = self.low_edges if level == "low" else self.high_edges
        outgoing = self.low_outgoing if level == "low" else self.high_outgoing
        costs = [1.0] * len(edges) if level == "low" else self.high_costs
        query_id = len(self.queries)
        noise_role = ("high" if level == "high" else
                      "flat" if phase in {"full_goal", "direct_candidate"} else "local")
        noise_scale = resolved_noise_scales(self.config)[noise_role]
        record: dict[str, Any] = {
            "id": query_id, "kind": "gcml", "level": level, "phase": phase,
            "start": start, "goal": goal, "horizon": horizon,
            "model_transitions_before": self.model_transitions,
            "ground_truth_audit_used_for_selection": False,
            "noise_role": noise_role, "noise_scale": noise_scale,
        }
        self.queries.append(record)
        if start == goal:
            record.update(status="initial_goal", actions=[], predicted_arrived=True,
                          model_transitions=0, theoretical_reservation=0)
            return record
        remaining = self.config["max_model_transitions"] - self.model_transitions
        requested_candidates = self.config["candidates"] if candidate_limit is None else candidate_limit
        candidates = min(requested_candidates, remaining // horizon)
        if candidates == 0:
            self.budget_blocked_queries += 1
            record.update(status="model_budget_exhausted", actions=None, predicted_arrived=False,
                          model_transitions=0, theoretical_reservation=0,
                          remaining_transitions=remaining)
            return record
        _, diagnostics = model.plan_with_diagnostics(
            start, goal, outgoing[start], costs, candidates=candidates, horizon=horizon,
            noise_scale=noise_scale,
            affordance_threshold=self.config["affordance_threshold"],
            association_radius=self.config["association_radius"],
            association_margin=self.config["association_margin"],
            affordance_mode="learned", first_step_affordance_mode="observed_graph",
            cycle_policy=self.config["cycle_policy"], diagnostic_detail="full",
            latent_projection=self.config["latent_projection"],
            noise_schedule=self.config["noise_schedule"],
        )
        consumed = _integer(diagnostics["model_transitions"], "planner model transitions")
        reservation = candidates * horizon
        if consumed > reservation:
            raise RuntimeError("Planner exceeded its reserved model-transition budget")
        self.model_transitions += consumed
        selected = diagnostics["selected_candidate_id"]
        candidate = next((item for item in diagnostics["candidates"] if item["id"] == selected), None)
        actions = list(candidate["actions"]) if candidate is not None else None
        record.update(status=diagnostics["branch"], actions=actions,
                      predicted_arrived=bool(candidate and candidate["predicted_arrived"]),
                      model_transitions=consumed, theoretical_reservation=reservation,
                      model_transitions_after=self.model_transitions,
                      requested_candidates=requested_candidates, actual_candidates=candidates,
                      diagnostics=diagnostics)
        # Computed only after the model has irrevocably selected its candidate.
        # No caller's plan selection or execution reads these audit labels.
        record["candidate_audits"] = [
            {"id": item["id"], "selected": item["id"] == selected,
             "predicted_arrived": bool(item["predicted_arrived"]),
             **audit_graph_actions(item["actions"], start, goal, edges, costs)}
            for item in diagnostics["candidates"]
        ]
        return record

    def search(self, start: int, goal: int, *, level: str,
               phase: str, horizon: int | None = None) -> dict[str, Any]:
        edges = self.low_edges if level == "low" else self.high_edges
        n_states = len(self.low_outgoing if level == "low" else self.high_outgoing)
        costs = [1.0] * len(edges) if level == "low" else self.high_costs
        actions, expansions = shortest_path(n_states, edges, costs, start, goal)
        self.graph_expansions += expansions
        too_long = actions is not None and horizon is not None and len(actions) > horizon
        record = {
            "id": len(self.queries), "kind": "dijkstra", "level": level, "phase": phase,
            "start": start, "goal": goal, "horizon": horizon,
            "status": "horizon_exhausted" if too_long else "complete_plan" if actions is not None else "no_plan",
            "actions": None if too_long else actions,
            "predicted_arrived": actions is not None and not too_long,
            "model_transitions": 0, "graph_expansions": expansions,
            "route_cost": sum(costs[action] for action in actions) if actions is not None else None,
            "unbounded_route_length": len(actions) if actions is not None else None,
            "information": "observed_high_graph" if level == "high" else "observed_low_graph",
        }
        self.queries.append(record)
        return record

    def execute(self, actions: list[int], target: int, *, phase: str,
                query_id: int, macro_action: int | None = None) -> str:
        """Execute only the selected suffix; an invalid edge ends the task."""
        stage: dict[str, Any] = {
            "index": len(self.stages), "phase": phase, "start": self.current,
            "target": target, "query_id": query_id, "macro_action": macro_action,
            "trace_start": len(self.trace), "planned_actions": len(actions),
        }
        self.stages.append(stage)
        status = "arrived" if self.current == target else "false_predicted_arrival"
        for index, action in enumerate(actions):
            if self.current == self.goal:
                status = "goal_reached"
                break
            if self.current == target:
                status = "arrived"
                break
            if len(self.trace) >= self.config["max_graph_steps"]:
                status = "graph_budget_exhausted"
                break
            if (isinstance(action, bool) or not isinstance(action, Integral)
                    or not 0 <= action < len(self.low_edges)):
                stage["invalid_action"] = {"action": action, "reason": "edge_id_out_of_range"}
                status = "illegal_action"
                break
            source, successor = self.low_edges[action]
            if source != self.current:
                stage["invalid_action"] = {"action": int(action), "reason": "source_mismatch",
                                           "expected_source": self.current, "actual_source": source}
                status = "illegal_action"
                break
            self.trace.append({
                "step": len(self.trace), "source": self.current, "action": int(action),
                "target": successor, "phase": phase, "stage": stage["index"],
                "query_id": query_id, "planned_step": index, "macro_action": macro_action,
            })
            self.current = successor
            if self.current == self.goal:
                status = "goal_reached"
                break
            if self.current == target:
                status = "arrived"
                break
        if self.current == self.goal:
            status = "goal_reached"
        elif self.current == target:
            status = "arrived"
        stage.update(status=status, end=self.current, trace_end=len(self.trace),
                     executed_steps=len(self.trace) - stage["trace_start"])
        return status


def _endpoint_candidates(hierarchy: dict[str, Any], node: int, direction: str,
                         count: int) -> list[dict[str, Any]]:
    landmarks = hierarchy["landmark_nodes"]
    collection = hierarchy["attachments"][direction]
    if len(collection) != hierarchy["n_states"]:
        raise ValueError("Attachment vocabulary must cover every fine node")
    records = [dict(item) for item in collection[node]]
    seen = set()
    for record in records:
        landmark = _integer(record["landmark"], "attachment landmark")
        if landmark >= len(landmarks) or record["landmark_node"] != landmarks[landmark]:
            raise ValueError("Attachment landmark IDs disagree")
        if landmark in seen:
            raise ValueError("Duplicate endpoint attachment")
        seen.add(landmark)
        value = record["median_steps"]
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(float(value)) or value < 0):
            raise ValueError("Attachment median steps must be finite and nonnegative")
    if node in landmarks and landmarks.index(node) not in seen:
        records.append({"landmark": landmarks.index(node), "landmark_node": node,
                        "median_steps": 0.0, "min_steps": 0, "support_count": 0,
                        "identity_attachment": True})
    return sorted(records, key=lambda item: (item["median_steps"], item["landmark"]))[:count]


def _planning_summary(queries: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    terminations: Counter[str] = Counter()
    for query in queries:
        counts[f"{query['level']}_queries"] += 1
        counts[f"{query['level']}_model_transitions"] += query["model_transitions"]
        diagnostics = query.get("diagnostics", {})
        counts["distinct_sequences_sum"] += diagnostics.get("distinct_action_sequences", 0)
        terminations.update(diagnostics.get("termination_counts", {}))
        for audit in query.get("candidate_audits", []):
            counts["candidates"] += 1
            counts["invalid_candidates"] += int(not audit["continuous_valid"])
            counts["false_predicted_arrivals"] += int(audit["predicted_arrived"] and not audit["graph_arrived"])
            if audit["selected"]:
                counts["selected_invalid_candidates"] += int(not audit["continuous_valid"])
                counts["selected_false_predicted_arrivals"] += int(audit["predicted_arrived"] and not audit["graph_arrived"])
    keys = ("low_queries", "high_queries", "low_model_transitions", "high_model_transitions",
            "candidates", "invalid_candidates", "false_predicted_arrivals",
            "selected_invalid_candidates", "selected_false_predicted_arrivals")
    return {**{key: counts[key] for key in keys},
            "distinct_sequences_sum": counts["distinct_sequences_sum"],
            "candidate_terminations": dict(terminations)}


def run_graph_task(
    *, method: str, low_model, high_model=None, hierarchy: dict[str, Any],
    start: int, goal: int, low_edges: list, seed: int, config: dict[str, Any],
) -> dict[str, Any]:
    """Run one frozen-model diagnostic task and return full query/step records.

    Attachment tables only supply endpoint candidates; their stored paths are
    never executed. Every learned query uses a known outgoing set at its
    declared query start, including prospective landmark endpoints. Future
    rollout states use learned G without any true-graph correction. All low
    and high calls share one actual model-transition cap, including failed
    endpoint proposals and macro execution calls. Dijkstra work is separately
    counted and is not presented as matching neural FLOPs.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown abstract graph method: {method}")
    settings = _configuration(config)
    n_states = _integer(hierarchy["n_states"], "n_states", minimum=1)
    start, goal = _integer(start, "start"), _integer(goal, "goal")
    if start >= n_states or goal >= n_states:
        raise ValueError("Task endpoint is outside the fine graph")
    seed = _integer(seed, "planning seed")
    if seed > 2**64 - 1:
        raise ValueError("Planning seed must fit an unsigned 64-bit integer")
    edges = _edges(low_edges, n_states)
    landmarks = list(hierarchy["landmark_nodes"])
    if (len(landmarks) != len(set(landmarks))
            or any(isinstance(node, bool) or not isinstance(node, Integral)
                   or not 0 <= node < n_states for node in landmarks)):
        raise ValueError("Invalid landmark vocabulary")
    high_edges = _edges(hierarchy["edges"], len(landmarks))
    macros = hierarchy["macro_edges"]
    if len(macros) != len(high_edges):
        raise ValueError("Macro metadata must cover the high action vocabulary")
    high_costs = []
    for action, (pair, record) in enumerate(zip(high_edges, macros)):
        source, target = pair
        if (record["action"] != action or record["source"] != source or record["target"] != target
                or record["source_node"] != landmarks[source] or record["target_node"] != landmarks[target]):
            raise ValueError("Macro action endpoints disagree with the high graph")
        cost = record["median_steps"]
        if isinstance(cost, bool) or not isinstance(cost, Real) or not math.isfinite(float(cost)) or cost <= 0:
            raise ValueError("Macro median length must be finite and positive")
        high_costs.append(float(cost))
    if method != "dijkstra":
        if low_model is None or low_model.n_states != n_states or list(low_model.edges) != edges:
            raise ValueError("Frozen low model vocabulary does not match the graph")
        low_model.reset_planning_rng(seed)
    if method == "hierarchical_gcml" and high_edges:
        if (high_model is None or high_model.n_states != len(landmarks)
                or list(high_model.edges) != high_edges):
            raise ValueError("High model vocabulary does not match the landmark graph")
        high_model.reset_planning_rng((seed + 0x9E3779B97F4A7C15) % 2**64)
    task = _Task(settings, edges, _outgoing(n_states, edges), high_edges,
                 _outgoing(len(landmarks), high_edges), high_costs, goal, start)
    options: list[dict[str, Any]] = []
    selected_option = None
    failure_phase = None
    reason = "goal_reached" if start == goal else "no_plan"
    endpoint_records: dict[str, Any] = {}

    if start != goal and method in {"dijkstra", "flat_gcml"}:
        query = (task.search(start, goal, level="low", phase="full_goal") if method == "dijkstra" else
                 task.query(low_model, start, goal, level="low", phase="full_goal",
                            horizon=settings["flat_horizon"], candidate_limit=settings["flat_candidates"]))
        if query["actions"] is None:
            reason, failure_phase = query["status"], "full_goal_planning"
        else:
            reason = task.execute(query["actions"], goal, phase="full_goal", query_id=query["id"])
            failure_phase = None if task.current == goal else "full_goal_execution"
    elif start != goal:
        direct = task.query(low_model, start, goal, level="low", phase="direct_candidate", horizon=settings["local_horizon"])
        if direct["actions"] is not None:
            options.append({"kind": "direct", "cost": float(len(direct["actions"])),
                            "entry_query": direct["id"], "high_actions": [],
                            "entry": None, "exit": None})
        entries = _endpoint_candidates(hierarchy, start, "entry", settings["endpoint_candidates"])
        exits = _endpoint_candidates(hierarchy, goal, "exit", settings["endpoint_candidates"])
        endpoint_records = {"entries": entries, "exits": exits}
        entry_plans = [(record, task.query(low_model, start, record["landmark_node"], level="low",
                                          phase="entry_candidate", horizon=settings["local_horizon"]))
                       for record in entries]
        exit_plans = [(record, task.query(low_model, record["landmark_node"], goal, level="low",
                                         phase="exit_candidate", horizon=settings["local_horizon"]))
                      for record in exits]
        for entry, entry_plan in entry_plans:
            if entry_plan["actions"] is None:
                continue
            for exit_record, exit_plan in exit_plans:
                if exit_plan["actions"] is None:
                    continue
                source, target = entry["landmark"], exit_record["landmark"]
                if method == "hierarchical_dijkstra":
                    high_plan = task.search(source, target, level="high", phase="high_candidate", horizon=settings["high_horizon"])
                elif not high_edges and source != target:
                    # An empty high vocabulary cannot instantiate a GCML model.
                    continue
                else:
                    high_plan = task.query(high_model, source, target, level="high", phase="high_candidate", horizon=settings["high_horizon"])
                if high_plan["actions"] is None:
                    continue
                options.append({
                    "kind": "hierarchical", "entry": source, "exit": target,
                    "entry_query": entry_plan["id"], "exit_query": exit_plan["id"],
                    "high_query": high_plan["id"], "high_actions": list(high_plan["actions"]),
                    "cost": float(len(entry_plan["actions"]) + len(exit_plan["actions"]))
                            + sum(high_costs[action] for action in high_plan["actions"]),
                })
        if not options:
            reason = ("model_budget_exhausted" if task.budget_blocked_queries else
                      "no_entry_attachment" if not entries else "no_exit_attachment" if not exits else
                      "no_entry_plan" if not any(plan["actions"] is not None for _, plan in entry_plans) else
                      "no_exit_plan" if not any(plan["actions"] is not None for _, plan in exit_plans) else "no_high_plan")
            failure_phase = "joint_route_planning"
        else:
            # Only predicted action counts and experience macro lengths enter
            # this ranking. Diagnostics cannot veto a predicted complete route.
            selected_option = min(range(len(options)), key=lambda index: (options[index]["cost"], index))
            option = options[selected_option]
            entry_plan = task.queries[option["entry_query"]]
            target = goal if option["kind"] == "direct" else landmarks[option["entry"]]
            phase = "direct" if option["kind"] == "direct" else "entry"
            reason = task.execute(entry_plan["actions"], target, phase=phase, query_id=entry_plan["id"])
            failure_phase = None if task.current == goal else f"{phase}_execution"
            if option["kind"] == "hierarchical" and reason == "arrived":
                for action in option["high_actions"]:
                    source, high_target = high_edges[action]
                    if task.current != landmarks[source]:
                        reason, failure_phase = "illegal_macro_action", "high_execution"
                        task.stages.append({"index": len(task.stages), "phase": "macro",
                                            "macro_action": action, "start": task.current,
                                            "expected_source": landmarks[source], "status": reason})
                        break
                    if len(task.trace) >= settings["max_graph_steps"]:
                        reason, failure_phase = "graph_budget_exhausted", "macro_execution"
                        break
                    macro_plan = task.query(low_model, task.current, landmarks[high_target], level="low",
                                            phase="macro_execution", horizon=settings["local_horizon"])
                    if macro_plan["actions"] is None:
                        reason, failure_phase = macro_plan["status"], "macro_planning"
                        task.stages.append({"index": len(task.stages), "phase": "macro",
                                            "macro_action": action, "start": task.current,
                                            "target": landmarks[high_target], "end": task.current,
                                            "query_id": macro_plan["id"], "executed_steps": 0,
                                            "status": reason})
                        break
                    reason = task.execute(macro_plan["actions"], landmarks[high_target], phase="macro",
                                          query_id=macro_plan["id"], macro_action=action)
                    failure_phase = None if task.current == goal else "macro_execution"
                    if reason != "arrived":
                        break
                if reason == "arrived":
                    if task.current != landmarks[option["exit"]]:
                        reason, failure_phase = "false_predicted_high_arrival", "high_execution"
                    else:
                        exit_plan = task.queries[option["exit_query"]]
                        reason = task.execute(exit_plan["actions"], goal, phase="exit", query_id=exit_plan["id"])
                        failure_phase = None if task.current == goal else "exit_execution"
    success = task.current == goal
    nodes = [start] + [step["target"] for step in task.trace]
    return {
        "schema": "cmnav.hierarchical_graph_task.v1", "method": method,
        "start": start, "goal": goal, "seed": seed, "config": settings,
        "success": success, "reason": "goal_reached" if success else reason,
        "failure_phase": None if success else failure_phase, "final_node": task.current,
        "graph_steps": len(task.trace), "nodes": nodes,
        "node_revisits": len(nodes) - len(set(nodes)),
        "model_transitions": task.model_transitions,
        "graph_expansions": task.graph_expansions,
        "budget_blocked_queries": task.budget_blocked_queries,
        "completed_stages": sum(stage.get("target") is not None and stage.get("end") == stage["target"]
                                for stage in task.stages),
        "completed_macros": sum(stage["phase"] == "macro" and stage.get("target") is not None
                                and stage.get("end") == stage["target"] for stage in task.stages),
        "endpoint_candidates": endpoint_records, "route_options": options,
        "selected_option": selected_option, "queries": task.queries,
        "stages": task.stages, "trace": task.trace,
        "planning_summary": _planning_summary(task.queries),
        "protocol": {
            "state": "known_abstract_node_id", "fine_action": "one_observed_graph_edge",
            "execution": "cached_complete_local_plan_open_loop_until_actual_target",
            "arrival": "actual_node_id", "pass_other_landmark": "retain_selected_target",
            "failure_recovery": "none", "model_parameter_updates": False,
            "query_initial_mask": "observed_outgoing_at_known_query_start_including_prospective_endpoints",
            "learned_future_mask": "learned_G_only", "graph_audit_used_for_selection": False,
            "endpoint_ranking": "experience_median_steps_then_landmark_id",
            "route_ranking": "local_predicted_action_counts_plus_macro_experience_median_steps",
            "direct_route": "competes_with_joint_endpoint_routes_under_same_query_budget",
            "model_budget": "shared_actual_transitions_all_queries_pre_reserve_candidates_times_horizon",
            "flat_candidate_allocation": "one_full_goal_query_with_independent_flat_candidates_limit",
            "comparison_budget": "equal_total_transition_upper_bound_not_equal_actual_transitions_or_FLOPs",
            "noise_allocation": "local_for_entry_exit_macro; flat_for_full_goal_and_direct_candidate; high_for_high_queries; no_true_distance_input",
            "dijkstra_budget": "settled_nodes_reported_separately_not_neural_compute_equivalence",
        },
    }
