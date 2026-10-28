"""Frozen, graph-supported official PointNav development task overlays.

The learning plan and observed graph remain immutable. Selection conditions on
their endpoint coverage before fitting a model; it is therefore a same-scene,
coverage-conditioned development diagnostic, not an unbiased benchmark result.
NavMesh queries are supplied explicitly by the environment service. This module
never starts an environment or collects goal-conditioned training experience.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from cmnav.data.pointnav import MAX_JSON_BYTES, SCENES_PATH_PREFIX, read_episode_dataset, validate_pose
from cmnav.memory.oracle_graph import OracleObservedGraph
from cmnav.runtime.artifacts import canonical_json_bytes, require_within, sha256_file


PROTOCOL = "graph_supported_official_pointnav_development_v1"
ACCESS = "environment_service_only"


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError(f"Nonfinite JSON value: {value}")

    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"Manifest exceeds size limit: {path}")
    result = json.loads(raw, object_pairs_hook=unique, parse_constant=reject)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def _validate_base(base: dict[str, Any], plan: dict[str, Any]) -> None:
    if plan.get("plan_id") != _hash(plan.get("payload")):
        raise ValueError("Original learning plan content hash is invalid")
    payload = plan["payload"]
    contract = payload["environment_contract"]
    environment = copy.deepcopy(base["environment"])
    environment.pop("gpu_device_id")
    dataset = copy.deepcopy(base["dataset"])
    dataset["scene_ids"] = sorted(dataset["scene_ids"])
    if (contract["dataset"] != dataset or contract["environment"] != environment
            or contract["environment_seed"] != base["seed"]):
        raise ValueError("Original learning plan and current environment contract differ")
    if (dataset["name"] != "mp3d" or dataset["split"] != "train"
            or len(dataset["scene_ids"]) != 1):
        raise ValueError("Supported tasks require one explicit MP3D train scene")
    for path, expected in payload["fingerprints"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"Original source/assets changed: {path}")


def _official_records(base: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Read only source files already fingerprinted by the original plan."""
    dataset = base["dataset"]
    scene = dataset["scene_ids"][0]
    scenes_root = Path(dataset["scenes_dir"])
    episodes_root = Path(dataset["episodes_path"]).parent
    sources = sorted(Path(path) for path in plan["payload"]["fingerprints"]
                     if path.endswith(".json.gz") and Path(path).is_relative_to(episodes_root))
    records = []
    identities = set()
    for source in sources:
        for episode in read_episode_dataset(source)["episodes"]:
            if Path(episode.get("scene_id", "")).stem != scene:
                continue
            identity = str(episode["episode_id"])
            if (scene, identity) in identities:
                raise ValueError(f"Duplicate official task: {scene}/{identity}")
            identities.add((scene, identity))
            asset = require_within(scenes_root / episode["scene_id"].removeprefix(SCENES_PATH_PREFIX), scenes_root)
            start = validate_pose(episode["start_position"], 3, "start_position")
            validate_pose(episode["start_rotation"], 4, "start_rotation")
            if len(episode["goals"]) != 1:
                raise ValueError("Supported PointNav task requires exactly one goal")
            goal = validate_pose(episode["goals"][0]["position"], 3, "goal.position")
            geometry = {"scene_id": str(asset),
                        "start_position": [x if x else 0.0 for x in start],
                        "goal_position": [x if x else 0.0 for x in goal]}
            canonical_json_bytes(episode)
            records.append({"source_episode_id": identity, "scene_key": scene,
                            "source_file": str(source), "task_geometry_sha256": _hash(geometry),
                            "episode": copy.deepcopy(episode)})
    return sorted(records, key=lambda row: (row["scene_key"], row["source_episode_id"]))


def _coverage(record: dict[str, Any], graph: OracleObservedGraph,
              geodesic: Callable[[Any, Any], float], radius: float,
              min_hops: int) -> dict[str, Any]:
    episode = record["episode"]
    start = episode["start_position"]
    goal = episode["goals"][0]["position"]
    start_node = graph.nearest(start, geodesic, radius)
    goal_node = graph.nearest(goal, geodesic, radius)
    path = None if start_node is None or goal_node is None else graph.shortest_path(start_node, goal_node)
    if start_node is None and goal_node is None:
        reason = "both_endpoints_unsupported"
    elif start_node is None:
        reason = "start_unsupported"
    elif goal_node is None:
        reason = "goal_unsupported"
    elif path is None:
        reason = "directed_graph_unreachable"
    elif len(path) < min_hops:
        reason = "shorter_than_min_path_hops"
    else:
        reason = "eligible"
    positions = graph.positions
    costs = graph.edge_mean_costs
    return {"status": reason, "start_node": start_node, "goal_node": goal_node,
            "start_attachment_geodesic_m": None if start_node is None else float(geodesic(start, positions[start_node])),
            "goal_attachment_geodesic_m": None if goal_node is None else float(geodesic(goal, positions[goal_node])),
            "directed_graph_reachable": path is not None,
            "shortest_path_edges": path,
            "shortest_path_hops": None if path is None else len(path),
            "shortest_path_cost_m": None if path is None else sum(costs[edge] for edge in path)}


def create_supported_navigation_tasks(
    base_config: dict[str, Any], base_plan: dict[str, Any], graph_path: str | Path,
    output_dir: str | Path, geodesic: Callable[[Any, Any], float], *,
    count: int = 5, seed: int = 23, attachment_radius: float = 0.5,
    min_path_hops: int = 8,
) -> Path:
    """Audit all official tasks, then freeze seeded eligible task selection.

    Exact learning task geometries and IDs are excluded before coverage checks.
    Selected tasks also have distinct directed graph endpoint pairs. Graph path
    costs are eligibility/audit metadata only, never GCML planner inputs.
    """
    if type(count) is not int or count < 1 or type(min_path_hops) is not int or min_path_hops < 1:
        raise ValueError("count and min_path_hops must be positive integers")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if (type(attachment_radius) not in (float, int) or not math.isfinite(attachment_radius)
            or not 0 < attachment_radius <= 1):
        raise ValueError("attachment_radius must be finite and in (0, 1]")
    _validate_base(base_config, base_plan)
    root = Path(base_config["project_root"])
    graph_path = require_within(Path(graph_path).expanduser(), root / "outputs")
    graph = OracleObservedGraph.from_state_dict(_read(graph_path))
    origin_plan = _read(graph_path.parent / "plan.json")
    if origin_plan != base_plan:
        raise ValueError("Observed graph run does not belong to the original learning plan")
    records = _official_records(base_config, base_plan)
    learning = base_plan["payload"]["learning"]
    learning_ids = {(row["scene_key"], row["source_episode_id"]) for row in learning}
    learning_geometry = {row["task_geometry_sha256"] for row in learning}
    reasons: Counter[str] = Counter()
    eligible = []
    geometries = set()
    for index, row in enumerate(records):
        key = (row["scene_key"], row["source_episode_id"])
        geometry = row["task_geometry_sha256"]
        if key in learning_ids or geometry in learning_geometry:
            reasons["learning_id_or_geometry"] += 1
            continue
        if geometry in geometries:
            reasons["duplicate_task_geometry"] += 1
            continue
        geometries.add(geometry)
        coverage = _coverage(row, graph, geodesic, float(attachment_radius), min_path_hops)
        reasons[coverage["status"]] += 1
        if coverage["status"] == "eligible":
            eligible.append({**row, "coverage": coverage})
        if (index + 1) % 10000 == 0:
            print(f"Supported-task coverage: {index + 1}/{len(records)}, eligible={len(eligible)}", flush=True)
    eligible.sort(key=lambda row: row["task_geometry_sha256"])
    eligible_fingerprint = _hash([{"scene": row["scene_key"], "id": row["source_episode_id"],
                                  "geometry": row["task_geometry_sha256"], "coverage": row["coverage"]}
                                 for row in eligible])
    indices = list(range(len(eligible)))
    random.Random(seed).shuffle(indices)
    selected = []
    endpoint_pairs = set()
    for index in indices:
        row = eligible[index]
        pair = (row["coverage"]["start_node"], row["coverage"]["goal_node"])
        if pair not in endpoint_pairs:
            endpoint_pairs.add(pair)
            selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Need {count} supported distinct routes; found {len(selected)}; counts={dict(reasons)}")
    original_coverage = [{"source_episode_id": row["source_episode_id"],
                          "coverage": _coverage(row, graph, geodesic, float(attachment_radius), min_path_hops)}
                         for row in base_plan["payload"]["evaluation"]]
    payload = {
        "protocol": PROTOCOL, "purpose": "coverage_conditioned_same_scene_development_diagnostic",
        "source_split": "train", "selection_depends_on_model_outputs": False,
        "collection_uses_new_evaluation_goals": False,
        "generalization_claim": "none; selection explicitly conditions on fixed observed graph coverage",
        "base_plan_id": base_plan["plan_id"], "base_plan_sha256": _hash(base_plan),
        "environment_contract": copy.deepcopy(base_plan["payload"]["environment_contract"]),
        "source_fingerprints": copy.deepcopy(base_plan["payload"]["fingerprints"]),
        "graph_path": str(graph_path), "graph_sha256": sha256_file(graph_path),
        "graph_origin_plan_path": str(graph_path.parent / "plan.json"),
        "graph_nodes": len(graph.positions), "graph_edges": len(graph.edges),
        "seed": seed, "attachment_radius": float(attachment_radius), "min_path_hops": min_path_hops,
        "evaluation_episodes": count,
        "selection": "sort_geometry_hash_then_seeded_shuffle_then_first_distinct_directed_endpoint_pairs",
        "exclusion": "learning source IDs and exact scene/start/goal geometry; duplicate evaluation geometry",
        "association": "nearest Euclidean candidate with NavMesh geodesic distance <= attachment_radius",
        "candidate_count": len(records), "candidate_status_counts": dict(sorted(reasons.items())),
        "eligible_pool_sha256": eligible_fingerprint,
        "eligible_directed_endpoint_pairs": len({(r["coverage"]["start_node"], r["coverage"]["goal_node"]) for r in eligible}),
        "original_evaluation_coverage": original_coverage, "evaluation": selected,
    }
    manifest = {"schema_version": 1, "access": ACCESS, "manifest_id": _hash(payload), "payload": payload}
    directory = require_within(Path(output_dir).expanduser(), root / "data")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{manifest['manifest_id']}.json"
    temporary = directory / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_supported_navigation_tasks(path: str | Path, base_config: dict[str, Any],
                                    base_plan: dict[str, Any], graph_path: str | Path) -> dict[str, Any]:
    """Validate the frozen overlay, exact official episodes, source and graph.

    This does not repeat NavMesh selection. The frozen geodesic audit and source
    asset hashes establish its origin; actual evaluation reattaches endpoints.
    """
    manifest = _read(Path(path).expanduser().resolve())
    if (set(manifest) != {"schema_version", "access", "manifest_id", "payload"}
            or manifest["schema_version"] != 1 or manifest["access"] != ACCESS
            or manifest["manifest_id"] != _hash(manifest["payload"])):
        raise ValueError("Invalid supported-task manifest schema or content hash")
    _validate_base(base_config, base_plan)
    payload = manifest["payload"]
    graph_path = Path(graph_path).expanduser().resolve()
    if (payload["protocol"] != PROTOCOL or payload["base_plan_id"] != base_plan["plan_id"]
            or payload["base_plan_sha256"] != _hash(base_plan)
            or payload["environment_contract"] != base_plan["payload"]["environment_contract"]
            or payload["source_fingerprints"] != base_plan["payload"]["fingerprints"]
            or payload["graph_path"] != str(graph_path)
            or payload["graph_sha256"] != sha256_file(graph_path)):
        raise ValueError("Supported-task source, original plan or graph changed")
    graph = OracleObservedGraph.from_state_dict(_read(graph_path))
    official = {(r["scene_key"], r["source_episode_id"]): r for r in _official_records(base_config, base_plan)}
    learning_ids = {(r["scene_key"], r["source_episode_id"]) for r in base_plan["payload"]["learning"]}
    learning_geometries = {r["task_geometry_sha256"] for r in base_plan["payload"]["learning"]}
    seen_ids, seen_geometries, seen_pairs = set(), set(), set()
    if len(payload["evaluation"]) != payload["evaluation_episodes"]:
        raise ValueError("Supported task count changed")
    for row in payload["evaluation"]:
        key = (row["scene_key"], row["source_episode_id"])
        geometry = row["task_geometry_sha256"]
        if (key in learning_ids or geometry in learning_geometries or key in seen_ids
                or geometry in seen_geometries or key not in official):
            raise ValueError("Selected task overlaps learning/another evaluation task or is not official")
        if {k: v for k, v in row.items() if k != "coverage"} != official[key]:
            raise ValueError("Selected episode differs from official source")
        audit = row["coverage"]
        pair = (audit["start_node"], audit["goal_node"])
        route = graph.shortest_path(*pair)
        if (pair in seen_pairs or audit["status"] != "eligible" or route is None
                or route != audit["shortest_path_edges"] or len(route) < payload["min_path_hops"]
                or len(route) != audit["shortest_path_hops"]):
            raise ValueError("Selected task's graph path audit is invalid")
        if not math.isclose(sum(graph.edge_mean_costs[e] for e in route), audit["shortest_path_cost_m"], rel_tol=1e-12):
            raise ValueError("Selected graph path cost changed")
        for end in ("start", "goal"):
            distance = audit[f"{end}_attachment_geodesic_m"]
            if type(distance) not in (float, int) or not 0 <= distance <= payload["attachment_radius"]:
                raise ValueError("Selected endpoint has no bounded geodesic attachment")
        seen_ids.add(key)
        seen_geometries.add(geometry)
        seen_pairs.add(pair)
    return manifest


def supported_episode_list(manifest: dict[str, Any], base_config: dict[str, Any]) -> list[Any]:
    """Instantiate ordered privileged episodes from a validated overlay."""
    from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

    dataset = PointNavDatasetV1(config=None)
    raw = {"episodes": [row["episode"] for row in manifest["payload"]["evaluation"]]}
    dataset.from_json(canonical_json_bytes(raw).decode("utf-8"), scenes_dir=base_config["dataset"]["scenes_dir"])
    scene_config = base_config["dataset"].get("scene_dataset_config")
    if scene_config:
        for episode in dataset.episodes:
            episode.scene_dataset_config = scene_config
    return list(dataset.episodes)
