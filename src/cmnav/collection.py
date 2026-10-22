"""Plan fixed starts, then collect goal-free RGB-D with separate audit logs."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cmnav.collection_config import data_path
from cmnav.diagnostics import MAX_JSON_BYTES, SCENES_PATH_PREFIX, inspect_environment
from cmnav.exploration_policy import DepthWalkPolicy
from cmnav.runner import _versions
from cmnav.trajectories import TrajectoryWriter


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_episodes(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        contents = stream.read(MAX_JSON_BYTES + 1)
    if len(contents) > MAX_JSON_BYTES:
        raise ValueError(f"Episode JSON exceeds {MAX_JSON_BYTES} bytes: {path}")
    value = json.loads(contents)
    if not isinstance(value, dict) or not isinstance(value.get("episodes"), list):
        raise ValueError(f"Invalid episode dataset: {path}")
    return value


def _inside(path: Path, parent: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError(f"Asset path escapes {parent}: {path}")
    return path


def _pose(value: Any, length: int, name: str) -> list[float]:
    if (not isinstance(value, list) or len(value) != length
            or any(type(item) not in (int, float) or not math.isfinite(item)
                   for item in value)):
        raise ValueError(f"Invalid {name}: expected {length} finite numbers")
    result = [float(item) for item in value]
    if length == 4 and abs(sum(item * item for item in result) - 1) > 1e-3:
        raise ValueError("Start quaternion must be normalized, in xyzw order")
    return result


def _plan_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Deterministically rebuild selection and fingerprints before collection."""
    dataset = config["environment_config"]["dataset"]
    source = Path(dataset["episodes_path"])
    scene_root = Path(dataset["scenes_dir"])
    root = _read_episodes(source)
    files = {source}
    template = root.get("content_scenes_path", "{data_path}/content/{scene}.json.gz")
    selected = sorted(set(dataset["scene_ids"]))
    rng = random.Random(config["seed"])
    starts = []
    for scene in selected:
        candidates = [episode for episode in root["episodes"]
                      if isinstance(episode, dict)
                      and Path(str(episode.get("scene_id", ""))).stem == scene]
        try:
            shard = _inside(Path(template.format(data_path=str(source.parent), scene=scene)),
                            source.parent)
        except (AttributeError, KeyError, IndexError) as exc:
            raise ValueError("Invalid content_scenes_path") from exc
        if shard.exists():
            files.add(shard)
            for episode in _read_episodes(shard)["episodes"]:
                if (not isinstance(episode, dict)
                        or Path(str(episode.get("scene_id", ""))).stem != scene):
                    raise ValueError(f"Scene mismatch in {shard}")
                candidates.append(episode)
        by_id = {}
        for episode in candidates:
            identity = str(episode.get("episode_id", ""))
            if not identity or identity in by_id:
                raise ValueError(f"Missing/duplicate episode ID in scene {scene}")
            by_id[identity] = episode
        count = config["collection"]["trajectories_per_scene"]
        if len(by_id) < count:
            raise ValueError(f"Scene {scene} has {len(by_id)} starts; need {count}")
        for identity in rng.sample(sorted(by_id), count):
            episode = by_id[identity]
            raw_scene = episode["scene_id"]
            relative = raw_scene.removeprefix(SCENES_PATH_PREFIX)
            asset = _inside(scene_root / relative, scene_root)
            if not asset.is_file():
                raise ValueError(f"Missing scene: {asset}")
            # Pin the mesh and available navigation/semantic resources as well.
            files.add(asset)
            for sibling in (asset.with_suffix(".navmesh"), asset.with_suffix(".house"),
                            asset.with_name(asset.stem + "_semantic.ply")):
                if sibling.is_file():
                    files.add(_inside(sibling, scene_root))
            index = len(starts)
            starts.append({
                "trajectory_id": f"traj_{index:06d}", "scene_key": scene,
                "scene_id": str(asset), "source_episode_id": identity,
                "start_position": _pose(episode.get("start_position"), 3, "start_position"),
                "start_rotation": _pose(episode.get("start_rotation"), 4, "start_rotation"),
                "policy_seed": (config["seed"] + index) % 2**32,
            })
    if dataset.get("scene_dataset_config"):
        files.add(Path(dataset["scene_dataset_config"]))
    return {
        "protocol": config["protocol"], "purpose": config["purpose"],
        "config_sha256": hashlib.sha256(_json(config)).hexdigest(),
        "selection": "seeded_sample_without_replacement_from_official_train_episodes",
        "selection_version": 1,
        "selection_scope": "episode IDs; distinct IDs may share physical start poses",
        "budget_actions": len(starts) * config["collection"]["steps_per_trajectory"],
        "fingerprints": {str(path): _sha(path) for path in sorted(files)},
        "starts": starts,
    }


def create_collection_plan(config: dict[str, Any]) -> Path:
    """Write an audit-only manifest; this command never initializes Habitat."""
    plan = {"schema_version": 1, "plan_id": uuid.uuid4().hex, "created_at": _now(),
            "access": "simulator_setup_and_audit_only", "payload": _plan_payload(config)}
    path = data_path(Path(config["project_root"]), config["collection"]["plan_dir"])
    path = path / f"{plan['plan_id']}.json"
    _save(path, plan)
    return path


def _metadata(config: dict[str, Any], start: dict[str, Any], plan_hash: str,
              collection_id: str) -> dict[str, Any]:
    environment = config["environment_config"]["environment"]
    return {
        "trajectory_id": start["trajectory_id"], "scene_id": start["scene_key"],
        "collection_id": collection_id,
        "source_episode_id": start["source_episode_id"],
        "start_selection": "seeded official train episode IDs without replacement",
        "collection_protocol": config["protocol"], "purpose": config["purpose"],
        "policy": config["policy"], "policy_seed": start["policy_seed"],
        "plan_sha256": plan_hash,
        "requested_actions": config["collection"]["steps_per_trajectory"],
        "sensor_spec": environment["rgbd"],
        "sensor_interpretation": {
            "projection": "pinhole", "hfov_unit": "horizontal degrees",
            "depth_storage": "Habitat metric depth clipped to configured min/max; unnormalized",
            "depth_validity": "nonpositive is unknown; max_depth may be saturated",
            "pixel_intrinsics": "derive under an explicitly documented pixel convention",
        },
        "coordinate_convention": {"body_axes": "x right, y up, -z forward",
                                  "camera_translation_body_m":
                                  [0.0, environment["rgbd"]["sensor_height"], 0.0],
                                  "camera_rotation_body_euler_rad": [0.0, 0.0, 0.0]},
        "action_spec": {"forward_step_m": environment["forward_step_size"],
                        "turn_angle_degrees": environment["turn_angle"],
                        "actions": ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"],
                        "displacement": "commanded; actual displacement may differ"},
    }


def collect_trajectories(config: dict[str, Any], plan_path: str | Path, *,
                         env_factory: Callable | None = None) -> Path:
    """Execute an exact plan once; never resume, replace starts, or retry silently."""
    plan_path = data_path(Path(config["project_root"]), plan_path)
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if (not isinstance(plan, dict) or plan.get("schema_version") != 1
            or plan.get("access") != "simulator_setup_and_audit_only"
            or plan.get("payload") != _plan_payload(config)):
        raise ValueError("Plan/config/assets mismatch; explicitly generate a new plan")
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    report = inspect_environment(config["environment_config"])
    if not report["ready_for_run"]:
        raise RuntimeError("Collection preflight failed:\n" + _json(report).decode())
    if env_factory is None:
        from cmnav.envs.exploration_env import ExplorationEnvironment
        env_factory = ExplorationEnvironment
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = data_path(Path(config["project_root"]), config["collection"]["output_dir"]) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    audit = directory / "audit"
    _save(audit / "plan.json", plan)
    _save(audit / "config.json", config)
    _save(audit / "preflight.json", report)
    _save(audit / "runtime.json", _versions())
    code_root = Path(__file__).parent
    _save(audit / "source_fingerprints.json", {
        str(path.relative_to(code_root)): _sha(path) for path in sorted(code_root.rglob("*.py"))})
    manifest = {"schema_version": 1, "run_id": run_id, "status": "collecting",
                "started_at": _now(), "protocol": config["protocol"],
                "purpose": config["purpose"], "plan_sha256": plan_hash,
                "expected_actions": plan["payload"]["budget_actions"], "trajectories": []}
    _save(directory / "dataset.json", manifest)
    env, writer = None, None
    try:
        env = env_factory(config["environment_config"], plan["payload"]["starts"])
        _save(audit / "habitat_config.json", env.resolved_config())
        for start in plan["payload"]["starts"]:
            observation = env.reset()
            info = env.episode_info()
            if (info["episode_id"] != start["trajectory_id"]
                    or Path(info["scene_id"]).resolve() != Path(start["scene_id"])):
                raise RuntimeError("Environment reset did not follow the collection manifest")
            destination = directory / "trajectories" / start["trajectory_id"]
            writer = TrajectoryWriter(destination, _metadata(config, start, plan_hash, run_id))
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
                "manifest_sha256": _sha(destination / "manifest.json"),
            })
            _save(directory / "dataset.json", manifest)
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
        manifest["finished_at"] = _now()
        try:
            if env is not None:
                env.close()
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            if not pending_error:
                raise
        finally:
            _save(directory / "dataset.json", manifest)
    return directory
