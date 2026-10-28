"""Collect reference RGB-D experience with an explicitly privileged controller.

The supplied oracle environment performs actual discrete actions. World poses
and measurements stay in separate audit files; trajectory observations remain
RGB-D only. The caller owns the environment and its finite episode list.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cmnav.data.trajectories import TrajectoryWriter
from cmnav.envs.oracle_navigation import OracleNavigationEnvironment, OracleFollowerError
from cmnav.runtime.artifacts import write_json, write_json_line


def _rgbd(observation: dict[str, Any]) -> dict[str, Any]:
    return {key: observation[key] for key in ("rgb", "depth")}


def _preview(directory: Path) -> None:
    paths = sorted((directory / "frames").glob("*.rgb.png"))[::5][:200]
    frames = []
    for path in paths:
        with Image.open(path) as picture:
            frames.append(picture.convert("RGB").resize((256, 256)))
    if frames:
        frames[0].save(directory / "preview.gif", save_all=True,
                       append_images=frames[1:], duration=200, loop=0)


def collect_reference(config: dict[str, Any], env: OracleNavigationEnvironment,
             directory: Path, records: list[dict[str, Any]], plan_id: str) -> list[dict[str, Any]]:
    """Collect the ordered learning tasks without changing their action budget."""
    summaries = []
    for index, record in enumerate(records):
        observation = env.reset()
        info = env.episode_info()
        if str(info["episode_id"]) != str(record["source_episode_id"]):
            raise RuntimeError("Learning episode order differs from the frozen manifest")
        trajectory_id = f"reference_{index:04d}"
        target = directory / "reference" / trajectory_id
        audit = directory / "audit" / f"{trajectory_id}.jsonl"
        audit.parent.mkdir(parents=True, exist_ok=True)
        writer = TrajectoryWriter(target, {
            "trajectory_id": trajectory_id, "scene_id": info["scene_id"],
            "source_episode_id": info["episode_id"], "plan_id": plan_id,
            "collection_protocol": "goal_directed_official_follower_v1",
            "policy": "privileged_habitat_shortest_path_follower",
            "sensor_spec": config["base"]["environment"]["rgbd"],
            "action_spec": {"forward_step_size": config["base"]["environment"]["forward_step_size"],
                            "turn_angle": config["base"]["environment"]["turn_angle"]},
        })
        steps = 0
        actions: Counter[str] = Counter()
        forward_collisions = 0
        no_motion = 0
        failure = None
        try:
            writer.start(_rgbd(observation))
            with audit.open("x", encoding="utf-8") as stream:
                state = env.privileged_state()
                write_json_line(stream, {"index": 0, "action": None, **state})
                while not env.episode_over:
                    try:
                        action = env.get_reference_action()
                    except OracleFollowerError as exc:
                        failure = str(exc)
                        writer.abort(failure)
                        break
                    previous = np.asarray(state["position"])
                    observation = env.step(action)
                    steps += 1
                    actions[action] += 1
                    state = env.privileged_state()
                    metrics = env.metrics()
                    collision = bool(metrics.get("collisions", {}).get("is_collision", False))
                    if action == "MOVE_FORWARD":
                        forward_collisions += int(collision)
                        no_motion += int(np.linalg.norm(previous - state["position"]) < 1e-4)
                    terminated = action == "STOP"
                    truncated = env.episode_over and not terminated
                    reason = "reference_stop" if terminated else "step_budget" if truncated else None
                    writer.append(action, _rgbd(observation), terminated=terminated,
                                  truncated=truncated, end_reason=reason)
                    write_json_line(stream, {"index": steps, "action": action, **state,
                                   "metrics": metrics})
            if failure is None:
                writer.finish()
                if index == 0:
                    _preview(target)
        except BaseException as exc:
            if writer.summary["status"] == "collecting":
                writer.abort(str(exc) or type(exc).__name__)
            raise
        result = {"trajectory_id": trajectory_id, **info, "steps": steps,
                  "status": "aborted" if failure else "complete", "error": failure,
                  "actions": dict(actions), "forward_collisions": forward_collisions,
                  "forward_no_motion": no_motion, "metrics": env.metrics()}
        summaries.append(result)
        write_json(directory / "reference_summary.json", summaries)
        print(f"Reference {index + 1}/{len(records)}: {steps} actions, "
              f"success={result['metrics'].get('success')}, status={result['status']}", flush=True)
    return summaries
