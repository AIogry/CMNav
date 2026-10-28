"""Compare fixed final goals with teacher-provided intermediate GCML goals.

This workflow does not fit a model. The teacher route is frozen once from the
initial graph endpoints, and only the current waypoint is given to GCML.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

from cmnav.data.navigation_plans import load_navigation_plan
from cmnav.data.supported_navigation_tasks import load_supported_navigation_tasks, supported_episode_list
from cmnav.data.teacher_subgoals import build_teacher_subgoal_manifest, validate_teacher_subgoal_manifest
from cmnav.envs.diagnostics import inspect_environment
from cmnav.envs.oracle_navigation import OracleNavigationEnvironment
from cmnav.evaluation.oracle_graph_navigation import evaluate_graph_navigation_task, summarize_graph_navigation_results
from cmnav.memory.oracle_graph import OracleObservedGraph
from cmnav.memory.snapshots import SceneMemoryStore
from cmnav.methods.gcml import GenerativeCognitiveMapLearner
from cmnav.runtime.artifacts import sha256_file, write_json
from cmnav.runtime.provenance import runtime_versions, source_hashes
from cmnav.workflows.oracle_graph_navigation import validate_reference_source


def _parameter_hash(model: GenerativeCognitiveMapLearner) -> str:
    digest = hashlib.sha256()
    for name in ("Q", "V", "W", "G"):
        digest.update(name.encode())
        digest.update(getattr(model, name).detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _baseline_comparison(result: dict[str, Any], task_dir: Path, source_dir: Path) -> dict[str, Any]:
    previous = json.loads((source_dir / "summary.json").read_text())
    fields = ("episode_id", "start_node", "goal_node", "steps", "end_reason", "graph_hops_attempted", "graph_hops_completed")
    checks = {key: result[key] == previous[key] for key in fields}
    for key in ("success", "spl", "distance_to_goal"):
        checks[key] = result["metrics"].get(key) == previous["metrics"].get(key)
    def actions(path: Path) -> list[Any]:
        return [json.loads(line)["action"] for line in path.read_text().splitlines()]
    checks["primitive_action_sequence"] = actions(task_dir / "execution.jsonl") == actions(source_dir / "execution.jsonl")
    for key in ("candidates", "model_transitions", "candidate_terminations"):
        checks[key] = result["gcml_planning"][key] == previous["gcml_planning"][key]
    return {"source": str(source_dir), "checks": checks, "all_match": all(checks.values())}


def _condition_summary(results: list[dict[str, Any]], condition: str, method: str,
                       interval: int | None) -> dict[str, Any]:
    group = [row for row in results if row["condition"] == condition]
    summary = summarize_graph_navigation_results(group, [method])[method]
    summary.update(condition=condition, method=method, interval=interval)
    if interval is not None:
        stages = [stage for row in group for stage in row["teacher_subgoals"]["stages"]]
        complete = sum(stage["status"] == "completed" for stage in stages)
        attempted = sum(stage["start_step"] is not None for stage in stages)
        total = sum(row["teacher_subgoals"]["total_stages"] for row in group)
        summary["stages"] = {"completed": complete, "attempted": attempted, "total": total,
                             "completion_among_attempted": complete / attempted if attempted else None,
                             "completion_among_prescribed": complete / total if total else None,
                             "unattempted_are_not_successes": True}
    return summary


def run_gcml_subgoal_comparison(config: dict[str, Any]) -> Path:
    """Replay final-goal controls and evaluate all declared teacher intervals."""
    source = Path(config["source_run"])
    base = config["base"]
    source_summary = json.loads((source / "summary.json").read_text())
    if source_summary["status"] != "completed":
        raise ValueError("Source GCML comparison is not complete")
    plan = load_navigation_plan(source / "plan.json", base)
    reference = json.loads((source / "reference_source.json").read_text())
    validate_reference_source(reference["path"], config, plan)
    original_graph = Path(reference["graph_run"]) / "graph.json"
    supported = load_supported_navigation_tasks(source / "supported_tasks.json", base, plan, original_graph)
    fingerprints = {name: sha256_file(source / f"{name}.json") for name in
                    ("graph", "model", "config", "supported_tasks", "training", "source_hashes", "summary")}
    if fingerprints["graph"] != supported["payload"]["graph_sha256"]:
        raise ValueError("Source copied graph differs from supported task graph")
    if supported["payload"]["attachment_radius"] != config["attachment_radius"]:
        raise ValueError("Attachment radius differs from frozen task coverage")
    before_sources = source_hashes()
    original_sources = json.loads((source / "source_hashes.json").read_text())
    if before_sources["methods/gcml.py"] != original_sources["methods/gcml.py"]:
        raise ValueError("This comparison requires the unchanged original GCML algorithm")
    graph = OracleObservedGraph.from_state_dict(json.loads((source / "graph.json").read_text()))
    model_state = json.loads((source / "model.json").read_text())
    model = GenerativeCognitiveMapLearner.from_state_dict(model_state, device=config["cml_device"])
    if model.n_states != len(graph.positions) or list(model.edges) != graph.edges:
        raise ValueError("Model state/action vocabulary differs from the graph")
    frozen_parameter_hash = _parameter_hash(model)
    manifests = dict(source_model_sha256=fingerprints["model"], source_graph_sha256=fingerprints["graph"],
                     source_tasks_sha256=fingerprints["supported_tasks"])
    teacher = build_teacher_subgoal_manifest(graph, supported, intervals=tuple(config["subgoal_intervals"]), **manifests)
    validate_teacher_subgoal_manifest(teacher, graph, supported, **manifests)
    records = teacher["payload"]["records"]
    if len(records) != config["evaluation_episodes"]:
        raise ValueError("Evaluation task count changed")
    preflight = inspect_environment(base)
    if not preflight["ready_for_run"]:
        raise RuntimeError("Environment dependencies/assets are incomplete")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = Path(config["output_dir"]) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    print(f"Teacher-subgoal comparison artifacts: {directory}", flush=True)
    for name in ("graph", "model", "plan", "supported_tasks", "training", "algorithm_reference"):
        shutil.copyfile(source / f"{name}.json", directory / f"{name}.json")
    for name, value in (("config", config), ("runtime", runtime_versions()), ("source_hashes", before_sources),
                        ("teacher_subgoal_plan", teacher), ("preflight", preflight)):
        write_json(directory / f"{name}.json", value)
    write_json(directory / "checkpoint_source.json", {
        "source_run": str(source), "fingerprints": fingerprints,
        "parameter_tensor_sha256": frozen_parameter_hash, "training_performed": False,
        "epochs_inherited": model.epochs_completed, "updates_inherited": model.updates,
    })
    write_json(directory / "protocol.json", {
        "name": config["subgoal_protocol"], "teacher_assisted": True,
        "formal_visual_result": False, "autonomous_subgoal_selection": False,
        "teacher": "fixed_initial_Dijkstra_route_on_observed_graph",
        "teacher_interval_unit": "number_of_directed_edges_on_the_fixed_reference_route",
        "teacher_goal_information_to_gcml": "only_current_subgoal_node_embedding",
        "reference_actions_or_remaining_distance_given_to_gcml": False,
        "reference_route_recomputed_during_task": False,
        "training_performed": False, "model_parameters": "frozen_same_checkpoint_each_task",
        "map_mode": "frozen", "retention": "reset",
        "environment_resets_within_task": 0, "rng_resets_within_task": 0,
        "intermediate_STOP_actions": 0, "subgoal_switch": "actual_arrival_then_advance",
        "task_success": "original_Habitat_final_goal_with_STOP",
        "failed_stage": "stop_whole_task_no_skip_no_extra_retry",
        "physical_action_budget": config["max_steps"],
        "planner_candidates_per_decision": config["gcml_candidates"],
        "planner_horizon_per_decision": config["gcml_horizon"],
        "noise_scale": config["gcml_noise_scale"],
        "same_total_imagination_budget_across_conditions": False,
        "compute_comparison": "fixed_per_decision_cap_report_total_transitions_and_time",
        "evaluation_subset": "same_five_graph_supported_development_tasks",
    })
    started = monotonic()
    results, baseline_checks, aggregates = [], [], {}
    conditions = [(f"{method}_final", method, None) for method in config["methods"]]
    conditions += [(f"{method}_{interval}", method, interval)
                   for interval in config["subgoal_intervals"] for method in config["methods"]]
    write_json(directory / "conditions.json", [{"condition": name, "method": method, "interval": interval}
                                               for name, method, interval in conditions])
    try:
        store = SceneMemoryStore(directory / "memory")
        context = {"scene_id": base["dataset"]["scene_ids"][0], "method_id": "gcml_teacher_subgoal_comparison",
                   "goal_protocol": "C", "data_version": teacher["manifest_id"],
                   "model_version": model_state["algorithm"], "protocol_version": config["subgoal_protocol"],
                   "privileged_world_pose": True, "formal_visual_result": False}
        snapshot_id = store.create_snapshot({"graph": graph.state_dict(), "model": model_state}, context)
        write_json(directory / "snapshot.json", {"snapshot_id": snapshot_id, "context": context})
        episodes = supported_episode_list(supported, base)
        for condition, method, interval in conditions:
            env = OracleNavigationEnvironment(base, episodes=episodes, oracle_state_diagnostic=True,
                                              goal_radius=config["local_goal_radius"])
            try:
                write_json(directory / "habitat_config.json", env.resolved_config())
                for index, record in enumerate(records):
                    task_dir = directory / "evaluation" / condition / f"task_{index:04d}"
                    session = store.begin_task(snapshot_id, task_id=f"{condition}_{index:04d}",
                                               expected_metadata=context, map_mode="frozen", retention="reset")
                    try:
                        state = session.read()
                        task_graph = OracleObservedGraph.from_state_dict(state["graph"])
                        task_model = GenerativeCognitiveMapLearner.from_state_dict(state["model"], device=config["cml_device"])
                        schedule = record["schedules"][str(interval)] if interval is not None else None
                        result = evaluate_graph_navigation_task(
                            config, env, task_graph, None, method, task_dir, config["seed"] + index,
                            gcml_model=task_model,
                            teacher_subgoals=schedule["subgoal_nodes"] if schedule else None,
                            teacher_reference_path=record["reference_edges"] if schedule else None)
                        if (str(result["episode_id"]) != record["source_episode_id"]
                                or result["start_node"] != record["start_node"]
                                or result["goal_node"] != record["goal_node"]):
                            raise RuntimeError("Runtime task or graph endpoints differ from the frozen plan")
                        if not result["endpoints_supported"] or not result["directed_graph_reachable"]:
                            raise RuntimeError("Supported task lost coverage")
                        if (_parameter_hash(task_model) != frozen_parameter_hash
                                or task_model.updates != model.updates or task_model.epochs_completed != model.epochs_completed):
                            raise RuntimeError("Frozen GCML parameters were modified during evaluation")
                        result.update(condition=condition, interval=interval, task_index=index,
                                      parameters_unchanged=True, teacher_manifest_id=teacher["manifest_id"])
                        if interval is None:
                            check = _baseline_comparison(result, task_dir, source / "evaluation" / method / f"task_{index:04d}")
                            baseline_checks.append({"condition": condition, "task_index": index, **check})
                            write_json(directory / "baseline_replay_checks.json", baseline_checks)
                            if not check["all_match"]:
                                raise RuntimeError("Final-goal control did not reproduce the source run")
                        result["memory_session"] = session.finish(commit=False)
                    except BaseException as exc:
                        session.abort(str(exc))
                        raise
                    write_json(task_dir / "summary.json", result)
                    results.append(result)
                    write_json(directory / "evaluation.json", results)
                    stage_info = result.get("teacher_subgoals", {})
                    print(f"{condition} {index+1}/{len(records)}: success={result['metrics']['success']}, "
                          f"steps={result['steps']}, stages={stage_info.get('completed_stages','-')}/"
                          f"{stage_info.get('total_stages','-')}, {result['end_reason']}", flush=True)
            finally:
                env.close()
            aggregates[condition] = _condition_summary(results, condition, method, interval)
            write_json(directory / "partial_summary.json", aggregates)
        if source_hashes() != before_sources:
            raise RuntimeError("Package source changed during the comparison")
        if any(sha256_file(source / f"{name}.json") != expected for name, expected in fingerprints.items()):
            raise RuntimeError("Source run artifacts changed during the comparison")
        write_json(directory / "summary.json", {
            "status": "completed", "protocol": config["subgoal_protocol"], "teacher_assisted": True,
            "formal_visual_result": False, "training_performed": False,
            "source_run": str(source), "teacher_manifest_id": teacher["manifest_id"],
            "snapshot_id": snapshot_id, "parameter_tensor_sha256": frozen_parameter_hash,
            "all_parameters_unchanged": True, "baseline_replay_all_match": all(row["all_match"] for row in baseline_checks),
            "tasks": len(records), "conditions": len(conditions), "evaluations": len(results),
            "methods": aggregates, "elapsed_seconds": monotonic() - started})
        return directory
    except BaseException as exc:
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(directory / "summary.json", {"status": "failed", "error": str(exc),
                                                "elapsed_seconds": monotonic() - started})
        raise
