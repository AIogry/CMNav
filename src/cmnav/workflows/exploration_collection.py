"""Collect goal-free RGB-D under an exact, frozen exploration start plan."""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cmnav.configuration.exploration import data_path
from cmnav.data.exploration_plans import load_collection_plan, trajectory_metadata
from cmnav.data.trajectories import TrajectoryWriter
from cmnav.envs.diagnostics import inspect_environment
from cmnav.policies.depth_walk import DepthWalkPolicy
from cmnav.runtime.artifacts import atomic_write_json, canonical_json_bytes, sha256_file
from cmnav.runtime.provenance import runtime_versions, source_hashes


def collect_trajectories(config: dict[str, Any], plan_path: str | Path, *,
                         env_factory: Callable | None = None) -> Path:
    """Execute an exact plan once; never resume, replace starts, or retry silently."""
    plan, plan_hash = load_collection_plan(config, plan_path)
    report = inspect_environment(config["environment_config"])
    if not report["ready_for_run"]:
        raise RuntimeError("Collection preflight failed:\n" + canonical_json_bytes(report).decode())
    if env_factory is None:
        from cmnav.envs.exploration_env import ExplorationEnvironment
        env_factory = ExplorationEnvironment
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = data_path(Path(config["project_root"]), config["collection"]["output_dir"]) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    audit = directory / "audit"
    atomic_write_json(audit / "plan.json", plan)
    atomic_write_json(audit / "config.json", config)
    atomic_write_json(audit / "preflight.json", report)
    atomic_write_json(audit / "runtime.json", runtime_versions())
    atomic_write_json(audit / "source_fingerprints.json", source_hashes())
    manifest = {"schema_version": 1, "run_id": run_id, "status": "collecting",
                "started_at": datetime.now(timezone.utc).isoformat(), "protocol": config["protocol"],
                "purpose": config["purpose"], "plan_sha256": plan_hash,
                "expected_actions": plan["payload"]["budget_actions"], "trajectories": []}
    atomic_write_json(directory / "dataset.json", manifest)
    env, writer = None, None
    try:
        env = env_factory(config["environment_config"], plan["payload"]["starts"])
        atomic_write_json(audit / "habitat_config.json", env.resolved_config())
        for start in plan["payload"]["starts"]:
            observation = env.reset()
            info = env.episode_info()
            if (info["episode_id"] != start["trajectory_id"]
                    or Path(info["scene_id"]).resolve() != Path(start["scene_id"])):
                raise RuntimeError("Environment reset did not follow the collection manifest")
            destination = directory / "trajectories" / start["trajectory_id"]
            writer = TrajectoryWriter(destination, trajectory_metadata(config, start, plan_hash, run_id))
            policy = DepthWalkPolicy(start["policy_seed"], config["policy"])
            budget = config["collection"]["steps_per_trajectory"]
            with (audit / f"{start['trajectory_id']}.jsonl").open("x", encoding="utf-8") as log:
                def record_audit(index: int) -> None:
                    log.write(json.dumps({"observation_index": index, **env.audit()},
                                         allow_nan=False) + "\n")
                    log.flush()

                writer.start(observation)
                record_audit(0)
                for step in range(1, budget + 1):
                    action = policy.act(observation)
                    observation = env.step(action)
                    if env.episode_over:
                        raise RuntimeError("Goal-free environment ended before external truncation")
                    writer.append(action, observation, truncated=step == budget,
                                  end_reason="collector_budget" if step == budget else None)
                    record_audit(step)
                writer.finish()
            writer = None
            manifest["trajectories"].append({
                "trajectory_id": start["trajectory_id"], "scene_id": start["scene_key"],
                "path": str(destination.relative_to(directory)), "actions": budget,
                "manifest_sha256": sha256_file(destination / "manifest.json"),
            })
            atomic_write_json(directory / "dataset.json", manifest)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        if writer is not None:
            try:
                writer.abort(manifest["error"])
            except Exception as abort_error:
                manifest["abort_error"] = str(abort_error)
        print(f"Collection artifacts: {directory}", file=sys.stderr)
        raise
    finally:
        pending_error = sys.exc_info()[0] is not None
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        try:
            if env is not None:
                env.close()
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            if not pending_error:
                raise
        finally:
            atomic_write_json(directory / "dataset.json", manifest)
    return directory
