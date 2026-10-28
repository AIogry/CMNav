"""Train GCML from frozen experience, then compare on separately frozen tasks.

Task coverage selection happens before this workflow. No evaluation goal enters
training; imagined transitions never become new environment experience.
"""
from __future__ import annotations

import json
import shutil
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

from cmnav.data.graph_walks import balanced_edge_walks
from cmnav.data.navigation_plans import load_navigation_plan
from cmnav.data.supported_navigation_tasks import load_supported_navigation_tasks, supported_episode_list
from cmnav.envs.diagnostics import inspect_environment
from cmnav.envs.oracle_navigation import OracleNavigationEnvironment
from cmnav.evaluation.oracle_graph_navigation import evaluate_graph_navigation_task, summarize_graph_navigation_results
from cmnav.memory.oracle_graph import OracleObservedGraph
from cmnav.memory.snapshots import SceneMemoryStore
from cmnav.methods.gcml import GenerativeCognitiveMapLearner
from cmnav.runtime.artifacts import sha256_file, write_json
from cmnav.runtime.provenance import runtime_versions, source_hashes
from cmnav.workflows.oracle_graph_navigation import validate_reference_source


def run_gcml_navigation(config: dict[str, Any], plan_path: str | Path, *,
                        supported_tasks: str | Path, graph_run: str | Path,
                        reference_run: str | Path) -> Path:
    """Run all declared methods with identical graph, episodes and controller."""
    base = config["base"]
    root = Path(config["project_root"])
    plan = load_navigation_plan(plan_path, base)
    if plan["payload"]["seed"] != config["seed"]:
        raise ValueError("Learning manifest seed differs from the configuration")
    if len(plan["payload"]["learning"]) != config["learning_episodes"]:
        raise ValueError("Learning manifest count differs from the configuration")
    source, references = validate_reference_source(reference_run, config, plan)
    graph_source = Path(graph_run).resolve()
    if not graph_source.is_relative_to(root / "outputs"):
        raise ValueError("Graph source must stay inside this checkout's outputs")
    graph_plan = load_navigation_plan(graph_source / "plan.json", base)
    if graph_plan["plan_id"] != plan["plan_id"]:
        raise ValueError("Graph source uses different learning experience")
    graph_config = json.loads((graph_source / "config.json").read_text())
    for key in ("merge_radius", "attachment_radius", "local_goal_radius"):
        if graph_config[key] != config[key]:
            raise ValueError(f"Graph source differs in {key}")
    origin = json.loads((graph_source / "reference_source.json").read_text())
    if (Path(origin["path"]).resolve() != source
            or origin["graph_associations_sha256"] != sha256_file(source / "graph_associations.json")
            or origin["reference_summary_sha256"] != sha256_file(source / "reference_summary.json")):
        raise ValueError("Graph source reference provenance changed")
    graph_file = graph_source / "graph.json"
    graph = OracleObservedGraph.from_state_dict(json.loads(graph_file.read_text()))
    tasks = load_supported_navigation_tasks(supported_tasks, base, plan, graph_file)
    if tasks["payload"]["attachment_radius"] != config["attachment_radius"]:
        raise ValueError("Supported manifest attachment radius differs from execution")
    records = tasks["payload"]["evaluation"]
    if len(records) != config["evaluation_episodes"] or len(records) < 5:
        raise ValueError("GCML comparison requires the configured count of at least five supported tasks")
    reference_metadata = json.loads((root / "configs" / "gcml_reference.json").read_text())
    preflight = inspect_environment(base)
    if not preflight["ready_for_run"]:
        raise RuntimeError("Habitat assets or dependencies are incomplete")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = Path(config["output_dir"]) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    print(f"GCML comparison artifacts: {directory}", flush=True)
    for name, value in (("config", config), ("plan", plan), ("supported_tasks", tasks),
                        ("preflight", preflight), ("runtime", runtime_versions()),
                        ("source_hashes", source_hashes()), ("algorithm_reference", reference_metadata),
                        ("reference_summary", references)):
        write_json(directory / f"{name}.json", value)
    shutil.copyfile(graph_file, directory / "graph.json")
    write_json(directory / "reference_source.json", {
        **origin, "graph_run": str(graph_source), "graph_sha256": sha256_file(graph_file),
        "supported_tasks_path": str(Path(supported_tasks).resolve()),
        "supported_tasks_sha256": sha256_file(Path(supported_tasks)),
    })
    write_json(directory / "protocol.json", {
        "name": config["gcml_protocol"], "access": "privileged_diagnostic_only",
        "formal_visual_result": False, "split": "same_scene_graph_coverage_conditioned_development",
        "state": "observed_world_position_with_navmesh_association",
        "action": "observed_directed_graph_edge", "map_mode": "frozen", "retention": "reset",
        "controller": "official_navmesh_follower_with_actual_discrete_actions",
        "rgbd_used_by_gcml": False, "training_uses_evaluation_goals": False,
        "training_new_environment_actions": 0,
        "training": "balanced_observed_edge_replay_pytorch_local_updates",
        "W_update": "notebook_residual_delta_rule_with_batch_curvature_step_cap",
        "G_update": "current_state_aligned_residual_rule_with_batch_curvature_step_cap",
        "imagined_state": "continuous_z_plus_V_action_without_snapping",
        "imagined_affordance": {"gcml": "learned_G_threshold", "gcml_graph_affordance": "observed_graph_after_latent_retrieval"},
        "noise": "independent_gaussian_after_full_utility_L2_normalization",
        "goal_detection": "post_update_latent_nearest_Q_with_distance_and_margin_rejection",
        "candidate_selection": "minimum_observed_edge_cost_among_predicted_complete_candidates",
        "no_complete_candidate": "no_plan_then_actual_STOP",
        "execution": "first_edge_then_reobserve_and_replan",
        "candidate_graph_audit_used_by_learned_planner": False,
        "permanent_real_visited_edge_ban": False,
        "max_imagined_transitions_per_decision": config["gcml_candidates"] * config["gcml_horizon"],
        "primitive_action_budget": config["max_steps"],
    })
    started = monotonic()
    try:
        walks = balanced_edge_walks(len(graph.positions), graph.edges,
                                    visits_per_edge=config["cml_edge_visits"],
                                    trajectory_length=config["cml_walk_length"], seed=config["seed"])
        walks["source"] = {"graph_sha256": sha256_file(graph_file), "plan_id": plan["plan_id"],
                            "reference_run": str(source), "uses_evaluation_goals": False,
                            "new_environment_actions": 0}
        write_json(directory / "training_walks.json", walks)
        trajectories = [item["transitions"] for item in walks["trajectories"] if item["transitions"]]
        model = GenerativeCognitiveMapLearner(
            len(graph.positions), graph.edges, dimension=config["cml_dimension"], seed=config["seed"],
            learning_rate=config["learning_rate"], state_learning_rate=config["state_learning_rate"],
            inverse_learning_rate=config["gcml_inverse_learning_rate"],
            affordance_learning_rate=config["gcml_affordance_learning_rate"],
            readout_step_safety=config["gcml_readout_step_safety"], device=config["cml_device"],
            reference_metadata=reference_metadata)
        print(f"GCML training: {len(graph.positions)} nodes, {len(graph.edges)} edges, "
              f"{walks['emitted_transitions']} replay transitions/epoch, {config['cml_epochs']} epochs", flush=True)
        def progress(row: dict[str, Any]) -> None:
            if row["epoch"] == 1 or row["epoch"] % 10 == 0:
                print(f"Epoch {row['epoch']}/{config['cml_epochs']}: "
                      f"transition MSE={row['prediction_mse']:.3g}, "
                      f"inverse MSE={row['inverse_mse']:.3g}, "
                      f"affordance MSE={row['affordance_mse']:.3g}", flush=True)
        training = model.fit(trajectories, epochs=config["cml_epochs"], progress_callback=progress)
        training.update(training_protocol=config["gcml_protocol"], walk_coverage=walks["coverage"],
                        walks_sha256=sha256_file(directory / "training_walks.json"),
                        replayed_transitions_per_epoch=walks["emitted_transitions"],
                        sampled_trajectories=len(trajectories))
        write_json(directory / "training.json", training)
        model_state = model.state_dict()
        write_json(directory / "model.json", model_state)
        print("GCML training complete; evaluating frozen model on supported tasks", flush=True)
        store = SceneMemoryStore(directory / "memory")
        context = {"scene_id": base["dataset"]["scene_ids"][0], "method_id": "shared_gcml_graph_bundle",
                   "goal_protocol": "C", "data_version": plan["plan_id"],
                   "model_version": model_state["algorithm"], "protocol_version": config["gcml_protocol"],
                   "privileged_world_pose": True, "formal_visual_result": False}
        snapshot_id = store.create_snapshot({"graph": graph.state_dict(), "model": model_state}, context)
        write_json(directory / "snapshot.json", {"snapshot_id": snapshot_id, "context": context})
        episodes = supported_episode_list(tasks, base)
        results = []
        for method in config["methods"]:
            env = OracleNavigationEnvironment(base, episodes=episodes, oracle_state_diagnostic=True,
                                              goal_radius=config["local_goal_radius"])
            try:
                write_json(directory / "habitat_config.json", env.resolved_config())
                for index, record in enumerate(records):
                    session = store.begin_task(snapshot_id, task_id=f"{method}_{index:04d}",
                                               expected_metadata=context, map_mode="frozen", retention="reset")
                    try:
                        state = session.read()
                        task_graph = OracleObservedGraph.from_state_dict(state["graph"])
                        task_model = (GenerativeCognitiveMapLearner.from_state_dict(state["model"], device=config["cml_device"])
                                      if method.startswith("gcml") else None)
                        result = evaluate_graph_navigation_task(
                            config, env, task_graph, None, method,
                            directory / "evaluation" / method / f"task_{index:04d}", config["seed"] + index,
                            gcml_model=task_model)
                        if str(result["episode_id"]) != str(record["source_episode_id"]):
                            raise RuntimeError("Task ordering differs from frozen supported manifest")
                        if not result["endpoints_supported"] or not result["directed_graph_reachable"]:
                            raise RuntimeError("A frozen supported task lost its coverage at runtime")
                        if (result["start_node"] != record["coverage"]["start_node"]
                                or result["goal_node"] != record["coverage"]["goal_node"]):
                            raise RuntimeError("Runtime endpoint associations differ from the frozen manifest")
                        result["memory_session"] = session.finish(commit=False)
                    except BaseException as exc:
                        session.abort(str(exc))
                        raise
                    results.append(result)
                    write_json(directory / "evaluation.json", results)
                    print(f"{method} {index+1}/{len(records)}: success={result['metrics'].get('success')}, "
                          f"{result['steps']} steps, {result['end_reason']}", flush=True)
            finally:
                env.close()
        write_json(directory / "summary.json", {
            "status": "completed", "protocol": config["gcml_protocol"], "formal_visual_result": False,
            "plan_id": plan["plan_id"], "snapshot_id": snapshot_id,
            "graph_nodes": len(graph.positions), "graph_edges": len(graph.edges),
            "edge_observations": len(graph.observed_edge_indices), "learning_episodes": len(references),
            "training_replay_transitions": walks["emitted_transitions"],
            "methods": summarize_graph_navigation_results(results, config["methods"]),
            "elapsed_seconds": monotonic() - started})
        return directory
    except BaseException as exc:
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(directory / "summary.json", {"status": "failed", "error": str(exc),
                                                "elapsed_seconds": monotonic() - started})
        raise
