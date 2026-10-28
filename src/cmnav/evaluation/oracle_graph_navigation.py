"""Execute and summarize frozen-map oracle graph navigation tasks.

Graph methods share the same task, graph snapshot, primitive budget, and local
NavMesh controller. The full oracle follower is a separate diagnostic. This
module does not train, update shared memory, or infer visual state identities.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from numbers import Integral
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from cmnav.envs.oracle_navigation import OracleNavigationEnvironment, OracleFollowerError
from cmnav.evaluation.gcml_diagnostics import GCML_METHODS, GCMLPlanningRecorder, audit_graph_actions
from cmnav.memory.oracle_graph import OracleObservedGraph
from cmnav.methods.cml import CognitiveMapLearner
from cmnav.runtime.artifacts import write_json, write_json_line
from cmnav.runtime.recording import EpisodeRecorder


def _gcml_parameter_version(model: Any) -> str:
    """Identify actual parameters, excluding the changing planning RNG state."""
    digest = hashlib.sha256()
    digest.update(repr((model.n_states, model.dimension, tuple(model.edges))).encode())
    for name in ("Q", "V", "W", "G"):
        tensor = getattr(model, name).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _selected_suffix(diagnostics: dict[str, Any], *, current: int, goal: int,
                     model_version: str, decision_id: int, hop: int) -> dict[str, Any]:
    """Copy a model-selected route without querying graph validity or targets."""
    candidate_id = diagnostics.get("selected_candidate_id")
    matches = [candidate for candidate in diagnostics["candidates"]
               if candidate["id"] == candidate_id]
    if len(matches) != 1:
        raise RuntimeError("A selected complete plan must identify exactly one candidate")
    candidate = matches[0]
    actions, steps = candidate["actions"], candidate["steps"]
    if (not candidate["predicted_arrived"] or not actions or len(actions) != len(steps)
            or diagnostics["selected_edge"] != actions[0]
            or any(step["action"] != action for step, action in zip(steps, actions))
            or not steps[-1]["associated_after"] or steps[-1]["decoded_state_after"] != goal):
        raise RuntimeError("Selected candidate is not a complete, aligned model plan")
    return {
        "actions": list(actions), "cursor": 0,
        "expected_nodes": [current] + [step["decoded_state_after"] for step in steps],
        "expected_associated": [True] + [step["associated_after"] for step in steps],
        "goal_node": goal, "model_version": model_version,
        "source_decision_id": decision_id, "source_hop": hop,
        "source_planning_call_index": diagnostics.get("planning_call_index"),
        "source_candidate_id": candidate_id,
    }


def evaluate_graph_navigation_task(config: dict[str, Any], env: OracleNavigationEnvironment,
                   graph: OracleObservedGraph, model: CognitiveMapLearner | None,
                   method: str, directory: Path, seed: int, *,
                   gcml_model: Any = None,
                   teacher_subgoals: list[int] | None = None,
                   teacher_reference_path: list[int] | None = None) -> dict[str, Any]:
    """Execute one task, optionally querying fixed externally supplied subgoals.

    Teacher subgoals are a privileged diagnostic available to GCML policies.
    The environment's final target and global
    budgets remain unchanged. ``teacher_reference_path`` contains directed
    edge IDs from the original start to the original goal; it is metadata,
    never a source of actions or dynamically generated replacement subgoals.
    """
    if teacher_subgoals is not None:
        if method not in GCML_METHODS:
            raise ValueError("Teacher subgoals require a GCML policy")
        if not isinstance(teacher_subgoals, list) or not teacher_subgoals:
            raise ValueError("Teacher subgoals must be a nonempty list of node IDs")
        if any(isinstance(node, bool) or not isinstance(node, Integral)
               or not 0 <= node < len(graph.positions) for node in teacher_subgoals):
            raise ValueError("Teacher subgoal is outside the observed node vocabulary")
        teacher_subgoals = [int(node) for node in teacher_subgoals]
    elif teacher_reference_path is not None:
        raise ValueError("A teacher reference path requires teacher subgoals")
    if config.get("gcml_gate_mode", "hard") not in {"hard", "soft"}:
        raise ValueError("gcml_gate_mode must be hard or soft")
    current_affordance = config.get("gcml_first_step_affordance_mode", "observed_graph")
    projection = config.get("gcml_latent_projection", "none")
    noise_schedule = config.get("gcml_noise_schedule", "legacy_active")
    cache_mode = config.get("gcml_cache_mode", "none")
    if current_affordance not in {"observed_graph", "learned"}:
        raise ValueError("gcml_first_step_affordance_mode must be observed_graph or learned")
    if projection not in {"none", "confident_nearest"}:
        raise ValueError("gcml_latent_projection must be none or confident_nearest")
    if noise_schedule not in {"legacy_active", "indexed_full"}:
        raise ValueError("gcml_noise_schedule must be legacy_active or indexed_full")
    if cache_mode not in {"none", "suffix_on_no_plan"}:
        raise ValueError("gcml_cache_mode must be none or suffix_on_no_plan")
    cache_enabled = cache_mode == "suffix_on_no_plan"
    if cache_enabled and method not in {"gcml", "gcml_graph_affordance"}:
        raise ValueError("Plan suffix caching requires a GCML imagination policy")
    observation = env.reset()
    info = env.episode_info()
    pose = env.privileged_state()["position"]
    goal = env.reference_target
    start_node = graph.nearest(pose, env.geodesic_distance, config["attachment_radius"])
    goal_node = graph.nearest(goal, env.geodesic_distance, config["attachment_radius"])
    reachable = (start_node is not None and goal_node is not None
                 and graph.shortest_path(start_node, goal_node) is not None)
    graph_edges = graph.edges
    if teacher_subgoals is not None:
        if start_node is None or goal_node is None:
            raise ValueError("Teacher subgoals require supported start and goal nodes")
        if teacher_subgoals[-1] != goal_node:
            raise ValueError("The final teacher subgoal must be the original task goal node")
        if teacher_reference_path is not None:
            if not isinstance(teacher_reference_path, list):
                raise ValueError("Teacher reference path must be a list of edge IDs")
            path_nodes = [start_node]
            for edge in teacher_reference_path:
                if (isinstance(edge, bool) or not isinstance(edge, Integral)
                        or not 0 <= edge < len(graph_edges)
                        or graph_edges[int(edge)][0] != path_nodes[-1]):
                    raise ValueError("Teacher reference path is not a continuous path from the task start")
                path_nodes.append(graph_edges[int(edge)][1])
            if path_nodes[-1] != goal_node:
                raise ValueError("Teacher reference path does not end at the task goal")
            next_index = 0
            for node in teacher_subgoals:
                try:
                    next_index = path_nodes.index(node, next_index) + 1
                except ValueError as exc:
                    raise ValueError("Teacher subgoals must occur in reference-path order") from exc
            teacher_reference_path = [int(edge) for edge in teacher_reference_path]
    recorder = EpisodeRecorder(directory, save_every=5, fps=5,
                               max_depth=config["base"]["environment"]["rgbd"]["max_depth"])
    steps, hops, completed_hops = 0, 0, 0
    path_length = 0.0
    last_position = np.asarray(pose, dtype=float)
    used_edges: list[int] = []
    reason = "unknown"
    stage_index = 0
    stages = ([{"index": index, "goal_node": node, "status": "not_attempted",
                "start_node": None, "end_node": None,
                "start_step": None, "start_hop": None,
                "end_step": None, "end_hop": None, "end_reason": None}
               for index, node in enumerate(teacher_subgoals)]
              if teacher_subgoals is not None else [])
    current = start_node
    rng = np.random.default_rng(seed)
    recorder.record(step=0, action=None, observation=observation,
                    metrics=env.metrics(), episode_over=False)
    queries = (directory / "planning.jsonl").open("x", encoding="utf-8")
    execution = (directory / "execution.jsonl").open("x", encoding="utf-8")
    edge_costs = graph.edge_mean_costs
    imagination = (GCMLPlanningRecorder(directory, graph_edges, edge_costs)
                   if method in GCML_METHODS else None)
    write_json_line(execution, {"step": 0, "action": None, **env.privileged_state()})
    plan_cache: dict[str, Any] | None = None
    cache_counts: Counter[str] = Counter()
    cache_invalidation_reasons: Counter[str] = Counter()
    cache_decision_id = 0

    def cache_reference() -> dict[str, Any] | None:
        if plan_cache is None:
            return None
        cursor = plan_cache["cursor"]
        return {key: plan_cache[key] for key in (
            "source_decision_id", "source_hop", "source_planning_call_index",
            "source_candidate_id", "goal_node", "model_version")} | {
                "cursor": cursor, "remaining_actions": len(plan_cache["actions"]) - cursor,
                "expected_current_node": plan_cache["expected_nodes"][cursor],
                "expected_current_associated": plan_cache["expected_associated"][cursor],
            }

    def invalidate_cache(cause: str) -> None:
        nonlocal plan_cache
        if plan_cache is None:
            return
        cache_counts["invalidations"] += 1
        cache_invalidation_reasons[cause] += 1
        write_json_line(queries, {"kind": "gcml_cache_event", "event": "invalidate",
                                 "cause": cause, "hop": hops, "step": steps,
                                 "current_node": current, "cache": cache_reference()})
        plan_cache = None

    def finish_stage(status: str, end_reason: str) -> None:
        if stage_index >= len(stages) or stages[stage_index]["status"] != "active":
            return
        stage = stages[stage_index]
        stage.update(status=status, end_node=current, end_step=steps,
                     end_hop=hops, end_reason=end_reason)
        write_json_line(queries, {"kind": "teacher_stage_end", **stage})

    def advance_stages() -> None:
        """Advance only at the actually completed node, without acting/resetting."""
        nonlocal stage_index
        while stage_index < len(stages):
            stage = stages[stage_index]
            if stage["status"] == "not_attempted":
                stage.update(status="active", start_node=current,
                             start_step=steps, start_hop=hops)
                write_json_line(queries, {"kind": "teacher_stage_start", **stage})
            if current != stage["goal_node"]:
                break
            finish_stage("completed", "subgoal_node_reached")
            stage_index += 1

    def execute(action: str) -> None:
        nonlocal observation, steps, path_length, last_position
        observation = env.step(action)
        steps += 1
        state = env.privileged_state()
        position = np.asarray(state["position"], dtype=float)
        path_length += float(np.linalg.norm(position - last_position))
        last_position = position
        write_json_line(execution, {"step": steps, "action": action, **state})
        recorder.record(step=steps, action=action, observation=observation,
                        metrics=env.metrics(), episode_over=env.episode_over)

    def follow(target: list[float], final: bool = False) -> str:
        local_steps = 0
        start_position = env.privileged_state()["position"]

        def finish(status: str) -> str:
            arrival = env.privileged_state()["position"]
            distance = env.geodesic_distance(arrival, target)
            if status == "arrived" and distance > config["local_goal_radius"] + 1e-4:
                status = "local_arrival_mismatch"
            write_json_line(queries, {"kind": "local_execution", "start_position": start_position,
                            "target_position": target, "arrival_position": arrival,
                            "remaining_geodesic_distance": distance, "steps": local_steps,
                            "final": final, "status": status})
            return status

        while not env.episode_over:
            action = env.get_reference_action(target)
            if action == "STOP":
                if final:
                    execute(action)
                    local_steps += 1
                return finish("arrived")
            if method != "oracle_follower" and local_steps >= config["max_local_steps"]:
                return finish("local_step_budget")
            execute(action)
            local_steps += 1
        return finish("episode_budget")

    try:
        if method in GCML_METHODS:
            if gcml_model is None:
                raise RuntimeError("GCML evaluation requires a saved GCML model")
            # A task has an independent random stream; a repeated visit does
            # not rewind it to the same candidate samples.
            gcml_model.reset_planning_rng(seed)
        if method == "oracle_follower":
            outcome = follow(goal, final=True)
            reason = "reference_stop" if outcome == "arrived" else outcome
        elif start_node is None or goal_node is None:
            reason = "unsupported_start" if start_node is None else "unsupported_goal"
        else:
            # A bounded connector is allowed only for endpoints associated by
            # <= attachment_radius geodesic distance, identically for all graph policies.
            reason = follow(graph.positions[start_node])
            current = start_node
            while reason == "arrived" and not env.episode_over:
                if teacher_subgoals is not None:
                    if (current == goal_node and stage_index < len(stages)
                            and stages[stage_index]["goal_node"] != goal_node):
                        # Success at the real final target takes precedence;
                        # do not force a detour to a now-unnecessary teacher node.
                        finish_stage("failed", "final_goal_node_reached_early")
                    else:
                        advance_stages()
                if current == goal_node:
                    actual = env.privileged_state()["position"]
                    if env.geodesic_distance(actual, goal) > config["attachment_radius"] + config["local_goal_radius"]:
                        reason = "final_connector_out_of_bounds"
                    else:
                        outcome = follow(goal, final=True)
                        reason = "goal_stop" if outcome == "arrived" else outcome
                    break
                if hops >= config["max_graph_hops"]:
                    reason = "graph_hop_budget"
                    break
                planning_goal = (stages[stage_index]["goal_node"]
                                 if teacher_subgoals is not None else goal_node)
                candidates = graph.outgoing(current)
                selection = None
                gcml_selection = None
                cache_selection = None
                if method == "dijkstra":
                    path = graph.shortest_path(current, goal_node)
                    edge = path[0] if path else None
                elif method == "cml":
                    if model is None:
                        raise RuntimeError("CML evaluation requires a saved model")
                    edge, selection = model.choose_edge_with_diagnostics(
                        current, goal_node, candidates, used_edges=used_edges)
                elif method == "random_graph":
                    edge = int(rng.choice(candidates)) if candidates else None
                elif method in GCML_METHODS:
                    parameter_version = None
                    if cache_enabled:
                        parameter_version = _gcml_parameter_version(gcml_model)
                        if plan_cache is not None:
                            cursor = plan_cache["cursor"]
                            if plan_cache["model_version"] != parameter_version:
                                invalidate_cache("model_version_changed")
                            elif plan_cache["goal_node"] != planning_goal:
                                invalidate_cache("planning_goal_changed")
                            elif cursor >= len(plan_cache["actions"]):
                                invalidate_cache("suffix_exhausted")
                            elif not plan_cache["expected_associated"][cursor]:
                                invalidate_cache("expected_current_state_unassociated")
                            elif plan_cache["expected_nodes"][cursor] != current:
                                invalidate_cache("current_node_mismatch")
                        cache_counts["fresh_planning_calls"] += 1
                    planning_started = monotonic()
                    if method == "gcml_greedy":
                        edge, diagnostics = gcml_model.greedy_edge_with_diagnostics(
                            current, planning_goal, candidates,
                            affordance_mode=current_affordance,
                            affordance_threshold=config["gcml_affordance_threshold"])
                    else:
                        edge, diagnostics = gcml_model.plan_with_diagnostics(
                            current, planning_goal, candidates, edge_costs,
                            candidates=config["gcml_candidates"],
                            horizon=config["gcml_horizon"],
                            noise_scale=config["gcml_noise_scale"],
                            affordance_threshold=config["gcml_affordance_threshold"],
                            association_radius=config["gcml_association_radius"],
                            association_margin=config["gcml_association_margin"],
                            cycle_policy=config["gcml_cycle_policy"],
                            affordance_mode=("observed_graph" if method == "gcml_graph_affordance"
                                             else "learned_soft" if config.get("gcml_gate_mode", "hard") == "soft"
                                             else "learned"),
                            soft_temperature=config.get("gcml_soft_temperature", .1),
                            soft_weight=config.get("gcml_soft_weight", .5),
                            diagnostic_detail=("compact" if method == "gcml" and config.get("gcml_gate_mode", "hard") == "soft" else "full"),
                            first_step_affordance_mode=current_affordance,
                            latent_projection=projection,
                            noise_schedule=noise_schedule,
                        )
                    planning_elapsed = monotonic() - planning_started
                    if cache_enabled:
                        # Decide cache retention/reuse from model data before
                        # the independent graph-continuity audit is computed.
                        fresh_edge = edge
                        previous_cache = cache_reference()
                        fallback_attempted = (edge is None and diagnostics.get("branch")
                                              in {"no_plan", "no_candidates"})
                        fallback_used = False
                        if edge is not None:
                            if plan_cache is not None:
                                cache_counts["replacements"] += 1
                                write_json_line(queries, {
                                    "kind": "gcml_cache_event", "event": "replace",
                                    "hop": hops, "step": steps, "cache": previous_cache})
                            plan_cache = _selected_suffix(
                                diagnostics, current=current, goal=planning_goal,
                                model_version=parameter_version, decision_id=cache_decision_id, hop=hops)
                            cache_counts["plans_stored"] += 1
                            write_json_line(queries, {
                                "kind": "gcml_cache_event", "event": "store",
                                "hop": hops, "step": steps, "cache": cache_reference(),
                                "actions": plan_cache["actions"],
                                "expected_nodes": plan_cache["expected_nodes"],
                                "expected_associated": plan_cache["expected_associated"],
                                "future_graph_audit_used": False})
                        elif fallback_attempted:
                            cache_counts["fallback_attempts"] += 1
                            if plan_cache is not None:
                                edge = plan_cache["actions"][plan_cache["cursor"]]
                                fallback_used = True
                                cache_counts["fallback_actions_selected"] += 1
                            else:
                                cache_counts["fallback_cache_misses"] += 1
                        cache_selection = {
                            "schema": "cmnav.gcml_cached_suffix_selection.v1",
                            "mode": cache_mode, "decision_id": cache_decision_id,
                            "fresh_planner_edge": fresh_edge,
                            "fresh_planner_branch": diagnostics.get("branch"),
                            "full_fresh_planning_performed": True,
                            "fallback_attempted": fallback_attempted, "fallback_used": fallback_used,
                            "execution_selected_edge": edge,
                            "execution_selection_source": "cached_suffix" if fallback_used else "fresh_planner",
                            "cache_before_planning": previous_cache,
                            "selected_cache": cache_reference(),
                            "future_graph_audit_used": False,
                        }
                        cache_decision_id += 1
                    # Independent labels are computed after the selected edge
                    # is fixed and are never passed back to the planner.
                    gcml_selection = imagination.record(
                        diagnostics, start=current, goal=planning_goal, hop=hops,
                        elapsed_seconds=planning_elapsed)
                    if cache_enabled and cache_selection["fallback_used"]:
                        # This audit is intentionally later than the reuse
                        # decision. Its labels never invalidate/repair a suffix.
                        suffix_audit = audit_graph_actions(
                            plan_cache["actions"][plan_cache["cursor"]:], current,
                            planning_goal, graph_edges, edge_costs)
                        cache_selection["fallback_suffix_graph_audit"] = suffix_audit
                        cache_counts["fallback_continuous_suffixes"] += int(suffix_audit["continuous_valid"])
                        cache_counts["fallback_graph_arrived_suffixes"] += int(suffix_audit["graph_arrived"])
                        cache_counts["fallback_false_predicted_arrival_suffixes"] += int(not suffix_audit["graph_arrived"])
                else:
                    raise ValueError(f"Unsupported graph method: {method}")
                valid_edge = (edge is not None and not isinstance(edge, bool)
                              and isinstance(edge, Integral) and edge in candidates
                              and 0 <= edge < len(graph_edges)
                              and graph_edges[int(edge)][0] == current)
                decision = {"kind": "graph_decision", "hop": hops, "current_node": current, "goal_node": goal_node,
                                "candidate_edges": candidates, "selected_edge": edge,
                                "cml_selection": selection,
                                "selected_edge_is_legal": bool(valid_edge) if edge is not None else None,
                                "position": env.privileged_state()["position"]}
                if gcml_selection is not None:
                    decision["gcml_selection"] = gcml_selection
                if cache_selection is not None:
                    # gcml_selection continues to describe the fresh search;
                    # this explicit record identifies a different executed edge.
                    decision["gcml_cached_suffix"] = cache_selection
                if teacher_subgoals is not None:
                    decision.update(planning_goal_node=planning_goal, stage_index=stage_index,
                                    teacher_subgoal=planning_goal)
                write_json_line(queries, decision)
                if edge is None:
                    reason = ("no_route" if method == "dijkstra" else
                              "no_plan" if method in GCML_METHODS and method != "gcml_greedy"
                              else "no_action")
                    break
                if not valid_edge:
                    # The privileged follower may reach arbitrary positions;
                    # it must never silently repair a graph-invalid decision.
                    reason = "invalid_action"
                    if imagination is not None:
                        imagination.invalid_selected_edge()
                    if cache_enabled:
                        cache_counts["invalid_fallback_actions" if cache_selection["fallback_used"]
                                     else "invalid_fresh_actions"] += 1
                        invalidate_cache("selected_current_action_invalid")
                    break
                edge = int(edge)
                used_edges.append(edge)
                target_node = graph.edge_target(edge)
                reason = follow(graph.positions[target_node])
                hops += 1
                if reason == "arrived":
                    current = target_node
                    completed_hops += 1
                    if cache_enabled and plan_cache is not None:
                        plan_cache["cursor"] += 1
                        cache_counts["cached_plan_actions_completed"] += 1
                        cache_counts["fallback_actions_completed"] += int(cache_selection["fallback_used"])
                        write_json_line(queries, {
                            "kind": "gcml_cache_event", "event": "advance",
                            "hop": hops, "step": steps, "current_node": current,
                            "executed_edge": edge, "fallback_used": cache_selection["fallback_used"],
                            "cache": cache_reference()})
                elif cache_enabled:
                    invalidate_cache("local_execution_failed")
        if not env.episode_over:
            # Giving up is an actual STOP, never a fabricated success label.
            execute("STOP")
    except OracleFollowerError as exc:
        reason = "follower_failure"
        write_json_line(queries, {"error": str(exc)})
        if not env.episode_over:
            execute("STOP")
    finally:
        if cache_enabled:
            invalidate_cache("task_end")
        if teacher_subgoals is not None:
            finish_stage("failed", reason)
        queries.close()
        execution.close()
        if imagination is not None:
            imagination.close()
        recorder.close()
        recorder.save_preview()
    result = {**info, "method": method, "seed": seed,
              "start_node": start_node, "goal_node": goal_node,
              "endpoints_supported": start_node is not None and goal_node is not None,
              "directed_graph_reachable": bool(reachable), "steps": steps,
              "graph_hops_attempted": hops, "graph_hops_completed": completed_hops,
              "executed_path_length_metres": path_length,
              "end_reason": reason, "metrics": env.metrics()}
    if imagination is not None:
        result["gcml_planning"] = imagination.summary()
    if cache_enabled:
        result["gcml_plan_cache"] = {
            "schema": "cmnav.gcml_cached_suffix_task.v1", "mode": cache_mode,
            "full_fresh_planning_before_fallback": True,
            "fresh_planning_counters_include_all_replans": True,
            "gcml_planning_selected_candidates_count_fresh_search_only": True,
            "version_scope": "sha256_of_Q_V_W_G_and_state_action_vocabulary",
            "reuse_requires_current_goal_and_parameter_version_match": True,
            "unassociated_expected_current_state_invalidates_cache": True,
            "current_action_legality_checked_after_selection": True,
            "future_graph_audit_used": False,
            "counts": {key: cache_counts[key] for key in (
                "fresh_planning_calls", "plans_stored", "replacements", "invalidations",
                "fallback_attempts", "fallback_cache_misses", "fallback_actions_selected",
                "fallback_actions_completed", "cached_plan_actions_completed",
                "invalid_fallback_actions", "invalid_fresh_actions",
                "fallback_continuous_suffixes", "fallback_graph_arrived_suffixes",
                "fallback_false_predicted_arrival_suffixes")},
            "invalidation_reasons": dict(cache_invalidation_reasons),
        }
    if teacher_subgoals is not None:
        result["teacher_subgoals"] = {
            "enabled": True, "provenance": "externally_fixed_teacher_subgoals",
            "nodes": teacher_subgoals, "reference_path": teacher_reference_path,
            "reference_path_unit": "directed_edge_ids",
            "stages": stages, "total_stages": len(stages),
            "attempted_stages": sum(stage["start_step"] is not None for stage in stages),
            "completed_stages": sum(stage["status"] == "completed" for stage in stages),
            "failed_stages": sum(stage["status"] == "failed" for stage in stages),
            "not_attempted_stages": sum(stage["status"] == "not_attempted" for stage in stages),
            "final_goal_node": goal_node,
            "final_goal_success": bool(result["metrics"].get("success", False)),
            "intermediate_success_is_task_success": False,
            "stage_clock": "global_primitive_step_and_graph_hop_counts",
            "first_stage_starts_after_start_connector": True,
            "environment_target_changed": False, "reset_between_stages": False,
            "rng_reset_between_stages": False, "stop_at_intermediate_subgoals": False,
            "global_primitive_budget": config["base"]["environment"]["max_episode_steps"],
            "global_graph_hop_budget": config["max_graph_hops"],
        }
    write_json(directory / "summary.json", result)
    return result


def summarize_graph_navigation_results(
    results: list[dict[str, Any]], methods: list[str],
) -> dict[str, Any]:
    """Aggregate the unchanged metrics over every requested method's tasks."""
    aggregates = {}
    for method in methods:
        group = [result for result in results if result["method"] == method]
        aggregates[method] = {
            "episodes": len(group),
            "success": float(np.mean([row["metrics"]["success"] for row in group])),
            "spl": float(np.mean([row["metrics"]["spl"] for row in group])),
            "mean_steps": float(np.mean([row["steps"] for row in group])),
            "endpoints_supported": sum(row["endpoints_supported"] for row in group),
            "directed_graph_reachable": sum(row["directed_graph_reachable"] for row in group),
            "end_reasons": dict(Counter(row["end_reason"] for row in group)),
        }
        if method in GCML_METHODS:
            planning = [row["gcml_planning"] for row in group]
            counters = (
                "decisions", "candidates", "model_transitions",
                "predicted_arrived_candidates", "continuous_valid_candidates",
                "invalid_candidates", "graph_arrived_candidates",
                "false_predicted_arrival_candidates", "missed_graph_arrival_candidates",
                "selected_candidates", "selected_continuous_valid", "selected_graph_arrived",
                "selected_false_predicted_arrival", "invalid_selected_edges",
                "nonfinite_diagnostic_values", "planning_seconds", "observed_planner_call_seconds",
            )
            totals = {key: sum(row[key] for row in planning) for key in counters}
            branches: Counter[str] = Counter()
            terminations: Counter[str] = Counter()
            for row in planning:
                branches.update(row["branches"])
                terminations.update(row["candidate_terminations"])
            aggregates[method]["gcml_planning"] = {
                **totals, "branches": dict(branches), "candidate_terminations": dict(terminations),
                "graph_audit_was_used_by_planner": False,
            }
            cached = [row["gcml_plan_cache"] for row in group if "gcml_plan_cache" in row]
            if cached:
                cache_totals: Counter[str] = Counter()
                invalidations: Counter[str] = Counter()
                for row in cached:
                    cache_totals.update(row["counts"])
                    invalidations.update(row["invalidation_reasons"])
                aggregates[method]["gcml_plan_cache"] = {
                    "episodes_with_cache": len(cached), "counts": dict(cache_totals),
                    "invalidation_reasons": dict(invalidations),
                    "fresh_search_statistics_exclude_fallback_route_selection": True,
                    "future_graph_audit_used": False,
                }
    return aggregates
