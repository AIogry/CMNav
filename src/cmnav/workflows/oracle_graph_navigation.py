"""Coordinate privileged reference collection, graph learning, and navigation.

This frozen-map diagnostic is separate from RGB-D policy training and formal
A/B/C evaluation. State association and local motion consume simulator truth;
no evaluation transition updates the shared graph or CML parameters.
"""
from __future__ import annotations

import json
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from cmnav.data.graph_walks import balanced_edge_walks, random_walks
from cmnav.data.navigation_plans import episode_lists, load_navigation_plan
from cmnav.envs.diagnostics import inspect_environment
from cmnav.envs.oracle_navigation import OracleNavigationEnvironment
from cmnav.evaluation.oracle_graph_navigation import (
    evaluate_graph_navigation_task,
    summarize_graph_navigation_results,
)
from cmnav.memory.oracle_graph import OracleObservedGraph, build_observed_graph
from cmnav.memory.snapshots import SceneMemoryStore
from cmnav.methods.cml import CognitiveMapLearner
from cmnav.runtime.artifacts import sha256_file, write_json
from cmnav.runtime.provenance import runtime_versions, source_hashes
from cmnav.workflows.reference_collection import collect_reference


def validate_reference_source(path: str | Path, config: dict[str, Any], plan: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    """Validate original reference records against the frozen plan and hashes."""
    source = Path(path).expanduser().resolve()
    if not source.is_relative_to(Path(config["project_root"]) / "outputs"):
        raise ValueError("Reference source must be a run inside this project's outputs")
    if not (source / "reference").is_dir() or not (source / "audit").is_dir():
        raise ValueError("Choose an original collection run with reference/ and audit/; "
                         "a derived run's reference_source.json identifies that source")
    saved_plan = load_navigation_plan(source / "plan.json", config["base"])
    if saved_plan["plan_id"] != plan["plan_id"]:
        raise ValueError("Reference source belongs to a different frozen task plan")
    saved_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if (saved_config["protocol"] != config["protocol"]
            or saved_config["local_goal_radius"] != config["local_goal_radius"]):
        raise ValueError("Reference source used different collection semantics")
    records = json.loads((source / "reference_summary.json").read_text(encoding="utf-8"))
    if (len(records) != len(plan["payload"]["learning"])
            or [str(row["episode_id"]) for row in records]
            != [row["source_episode_id"] for row in plan["payload"]["learning"]]):
        raise ValueError("Reference source episode identities differ from the frozen plan")
    for row in records:
        if Path(row["scene_id"]).stem != config["base"]["dataset"]["scene_ids"][0]:
            raise ValueError("Reference source scene differs from the frozen task scene")
    associations = json.loads((source / "graph_associations.json").read_text(encoding="utf-8"))
    complete = [row for row in records if row["status"] == "complete"]
    if [row["trajectory_id"] for row in associations] != [row["trajectory_id"] for row in complete]:
        raise ValueError("Reference source graph provenance differs from completed trajectories")
    for row in associations:
        name = row["trajectory_id"]
        if Path(name).name != name:
            raise ValueError("Invalid reference trajectory identifier")
        if (row["audit_sha256"] != sha256_file(source / "audit" / f"{name}.jsonl")
                or row["trajectory_manifest_sha256"]
                != sha256_file(source / "reference" / name / "manifest.json")):
            raise ValueError("Reference source checksum changed")
    return source, records


def run_oracle_graph_navigation(config: dict[str, Any], plan_path: str | Path, *,
                  reference_run: str | Path | None = None) -> Path:
    """Collect, reread, fit, snapshot, then evaluate a frozen task manifest."""
    base = config["base"]
    plan = load_navigation_plan(plan_path, base)
    payload = plan["payload"]
    if payload["seed"] != config["seed"]:
        raise ValueError("Plan selection seed differs from the graph navigation configuration")
    if (len(payload["learning"]) != config["learning_episodes"]
            or len(payload["evaluation"]) != config["evaluation_episodes"]):
        raise ValueError("Plan counts differ from the graph navigation configuration")
    source, references = validate_reference_source(reference_run, config, plan) if reference_run else (None, None)
    preflight = inspect_environment(base)
    if not preflight["ready_for_run"]:
        raise RuntimeError("Oracle graph navigation assets/dependencies are incomplete; run cmnav doctor")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = Path(config["output_dir"]) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    print(f"Oracle graph navigation artifacts: {directory}", flush=True)
    write_json(directory / "config.json", config)
    write_json(directory / "plan.json", plan)
    write_json(directory / "preflight.json", preflight)
    write_json(directory / "runtime.json", runtime_versions())
    write_json(directory / "source_hashes.json", source_hashes())
    write_json(directory / "protocol.json", {
        "name": config["protocol"], "access": "privileged_diagnostic_only",
        "formal_visual_result": False, "split": "same_scene_development",
        "state": "observed_world_position_with_navmesh_association",
        "action": "observed_directed_graph_edge",
        "controller": "official_navmesh_follower_with_actual_discrete_actions",
        "map_mode": "frozen", "retention": "reset", "rgbd_used_by_cml": False,
        "graph_methods": "same_snapshot_same_local_controller_same_tasks",
        "cml_training_protocol": config["cml_training_protocol"],
        "cml_training_experience": "target_independent_replay_on_observed_directed_graph",
        "cml_sampling": config.get("cml_sampling", "random_walks"),
        "cml_training_replay_is_new_environment_experience": False,
        "cml_training_backend": "pytorch_float32",
        "cml_training_device": config["cml_device"],
        "cml_training_update": "official_trajectory_batch_indexed_writes",
        "cml_training_batch_indices": (
            "unique_target_and_action_indices" if config.get("cml_sampling") == "balanced_edge_walks"
            else "repeated_indices_possible_reference_write_semantics"
        ),
        "cml_training_order": "shuffle_trajectories_each_epoch",
        "edge_cost": "mean_observed_displacement_accumulated_between_node_changes",
        "edge_cost_is_actual_execution_cost": False,
        "oracle_follower": "separate_full_navmesh_diagnostic",
        "endpoint_association_radius_metres": config["attachment_radius"],
        "local_arrival_tolerance_metres": config["local_goal_radius"],
        "final_connector_max_metres": config["attachment_radius"] + config["local_goal_radius"],
        "local_motion_action_budget": config["max_local_steps"],
        "local_final_stop_extra_actions": 1,
    })
    if source is not None:
        write_json(directory / "reference_source.json", {
            "path": str(source), "plan_id": plan["plan_id"],
            "graph_associations_sha256": sha256_file(source / "graph_associations.json"),
            "reference_summary_sha256": sha256_file(source / "reference_summary.json"),
        })
    started = monotonic()
    try:
        learning_episodes, evaluation_episodes = episode_lists(plan, base)
        env = OracleNavigationEnvironment(base, episodes=learning_episodes,
                                   oracle_state_diagnostic=True, goal_radius=config["local_goal_radius"])
        try:
            if source is None:
                references = collect_reference(config, env, directory, payload["learning"], plan["plan_id"])
            else:
                # Load only the training scene for geometry-assisted association.
                # No evaluation observations enter the rebuilt graph.
                env.reset()
                write_json(directory / "reference_summary.json", references)
            write_json(directory / "habitat_config.json", env.resolved_config())
            graph = build_observed_graph(config, directory, references, env.geodesic_distance,
                                 plan_id=plan["plan_id"], reference_directory=source)
        finally:
            env.close()
        if config.get("cml_sampling", "random_walks") == "balanced_edge_walks":
            walks = balanced_edge_walks(
                len(graph.positions), graph.edges,
                visits_per_edge=config["cml_edge_visits"],
                trajectory_length=config["cml_walk_length"], seed=config["seed"],
            )
        else:
            walks = random_walks(
                len(graph.positions), graph.edges,
                num_trajectories=config["cml_random_walks"],
                trajectory_length=config["cml_walk_length"], seed=config["seed"],
            )
        walks["source"] = {
            "kind": "replay_of_observed_directed_graph",
            "graph_sha256": sha256_file(directory / "graph.json"),
            "plan_id": plan["plan_id"],
            "reference_run": str(source or directory),
            "uses_evaluation_goals": False,
            "new_environment_actions": 0,
        }
        write_json(directory / "training_walks.json", walks)
        trajectories = [walk["transitions"] for walk in walks["trajectories"]
                        if walk["transitions"]]
        if not trajectories:
            raise RuntimeError("Training replay contains no transitions; inspect the saved graph coverage")
        model = CognitiveMapLearner(len(graph.positions), graph.edges,
                                    dimension=config["cml_dimension"], seed=config["seed"],
                                    learning_rate=config["learning_rate"],
                                    state_learning_rate=config["state_learning_rate"],
                                    inverse_learning_rate=config["inverse_learning_rate"],
                                    device=config["cml_device"])
        training = model.fit(trajectories, epochs=config["cml_epochs"])
        training["training_protocol"] = config["cml_training_protocol"]
        training["walks_sha256"] = sha256_file(directory / "training_walks.json")
        training["walk_coverage"] = walks["coverage"]
        training["sampled_trajectories"] = len(walks["trajectories"])
        training["nonempty_training_trajectories"] = len(trajectories)
        training["replayed_transitions_per_epoch"] = walks["emitted_transitions"]
        write_json(directory / "training.json", training)
        model_state = model.state_dict()
        write_json(directory / "model.json", model_state)
        print(f"Learned graph: {len(graph.positions)} nodes, {len(graph.edges)} directed edges, "
              f"{len(graph.observed_edge_indices)} observed transitions; "
              f"{walks['emitted_transitions']} replay transitions per epoch", flush=True)
        store = SceneMemoryStore(directory / "memory")
        context = {"scene_id": base["dataset"]["scene_ids"][0], "method_id": "shared_graph_bundle",
                   "goal_protocol": "C", "data_version": plan["plan_id"],
                   "model_version": model_state["algorithm"], "protocol_version": config["protocol"],
                   "privileged_world_pose": True, "formal_visual_result": False}
        snapshot_id = store.create_snapshot({"graph": graph.state_dict(), "model": model_state}, context)
        write_json(directory / "snapshot.json", {"snapshot_id": snapshot_id, "context": context})
        results = []
        for method in config["methods"]:
            env = OracleNavigationEnvironment(base, episodes=evaluation_episodes,
                                       oracle_state_diagnostic=True, goal_radius=config["local_goal_radius"])
            try:
                for index, record in enumerate(payload["evaluation"]):
                    session = store.begin_task(snapshot_id, task_id=f"{method}_{index:04d}",
                                               expected_metadata=context, map_mode="frozen", retention="reset")
                    try:
                        state = session.read()
                        task_graph = OracleObservedGraph.from_state_dict(state["graph"])
                        task_model = (CognitiveMapLearner.from_state_dict(
                            state["model"], device=config["cml_device"])
                            if method == "cml" else None)
                        result = evaluate_graph_navigation_task(config, env, task_graph, task_model, method,
                                                directory / "evaluation" / method / f"task_{index:04d}",
                                                config["seed"] + index)
                        if str(result["episode_id"]) != str(record["source_episode_id"]):
                            raise RuntimeError("Evaluation task order differs from frozen manifest")
                        result["memory_session"] = session.finish(commit=False)
                    except BaseException as exc:
                        session.abort(str(exc))
                        raise
                    results.append(result)
                    write_json(directory / "evaluation.json", results)
                    print(f"{method} {index + 1}/{len(evaluation_episodes)}: "
                          f"success={result['metrics'].get('success')}, {result['end_reason']}", flush=True)
            finally:
                env.close()
        aggregates = summarize_graph_navigation_results(results, config["methods"])
        write_json(directory / "summary.json", {
            "status": "completed", "protocol": config["protocol"], "formal_visual_result": False,
            "plan_id": plan["plan_id"], "snapshot_id": snapshot_id,
            "learning_episodes": len(references), "reference_steps": sum(row["steps"] for row in references),
            "reference_success": float(np.mean([row["metrics"]["success"] for row in references])),
            "graph_nodes": len(graph.positions), "graph_edges": len(graph.edges),
            "edge_observations": len(graph.observed_edge_indices),
            "cml_training_protocol": config["cml_training_protocol"],
            "training_walks": len(walks["trajectories"]),
            "training_replay_transitions": walks["emitted_transitions"],
            "training_replay_unique_edges": walks["coverage"]["unique_edges"],
            "methods": aggregates, "elapsed_seconds": monotonic() - started,
        })
        return directory
    except BaseException as exc:
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(directory / "summary.json", {"status": "failed", "error": str(exc),
                                                "elapsed_seconds": monotonic() - started})
        raise
