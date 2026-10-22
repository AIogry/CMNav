"""The experiment loop shared by policies; metrics never enter policy input."""
from __future__ import annotations

import importlib.metadata
import platform
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from cmnav import __version__
from cmnav.policies import RandomPolicy
from cmnav.recording import EpisodeRecorder, write_json


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _versions() -> dict[str, Any]:
    packages = {}
    for name in ("cmnav", "numpy", "Pillow", "PyYAML", "habitat-lab", "habitat-sim", "hydra-core"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"cmnav_version": __version__, "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": packages}


def run_experiment(config: dict[str, Any], *, env_factory: Callable | None = None) -> Path:
    if env_factory is None:
        from cmnav.envs.habitat_env import HabitatEnvironment
        env_factory = HabitatEnvironment
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = Path(config["run"]["output_dir"]) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(directory / "runtime.json", _versions())
    manifest: dict[str, Any] = {"run_id": run_id, "started_at_utc": _utc(),
                                "execution_status": "running", "protocol": config["protocol"],
                                "dataset": config["dataset"]["name"], "policy": config["policy"]["name"],
                                "episodes": [], "result_scope": "environment validation only"}
    write_json(directory / "summary.json", manifest)
    env = None
    started = time.monotonic()
    try:
        env = env_factory(config)
        write_json(directory / "habitat_config.json", env.resolved_config())
        for index in range(config["run"]["episodes"]):
            observation = env.reset()
            info = env.episode_info()
            policy = RandomPolicy(seed=config["seed"] + index,
                                  forward_probability=config["policy"]["forward_probability"],
                                  max_steps=config["environment"]["max_episode_steps"])
            episode_dir = directory / f"episode_{index:04d}"
            recorder = EpisodeRecorder(episode_dir, save_every=config["run"]["save_every"],
                                       fps=config["run"]["preview_fps"],
                                       max_depth=config["environment"]["rgbd"]["max_depth"])
            steps, action = 0, None
            try:
                write_json(episode_dir / "episode.json", {**info, "index": index,
                                                          "policy_seed": config["seed"] + index})
                recorder.record(step=0, action=None, observation=observation,
                                metrics=env.metrics(), episode_over=env.episode_over)
                while not env.episode_over and steps < config["environment"]["max_episode_steps"]:
                    action = policy.act(observation)
                    observation = env.step(action)
                    steps += 1
                    recorder.record(step=steps, action=action, observation=observation,
                                    metrics=env.metrics(), episode_over=env.episode_over)
                final_metrics = env.metrics()
            finally:
                recorder.close()
            recorder.save_preview()
            result = {**info, "index": index, "actions": steps,
                      "termination_reason": "stop" if action == "STOP" and env.episode_over else
                      "environment_done" if env.episode_over else "runner_limit",
                      "metrics": final_metrics,
                      "artifacts": str(episode_dir.relative_to(directory))}
            write_json(episode_dir / "summary.json", result)
            manifest["episodes"].append(result)
            write_json(directory / "summary.json", manifest)
        aggregates = {}
        for key in ("success", "spl", "soft_spl", "distance_to_goal"):
            values = [item["metrics"].get(key) for item in manifest["episodes"]]
            if values and all(isinstance(v, (int, float, np.number)) and np.isfinite(v) for v in values):
                aggregates[f"mean_{key}"] = float(np.mean(values))
        manifest["aggregate"] = aggregates
        manifest["execution_status"] = "completed"
    except BaseException as exc:
        manifest["execution_status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Run artifacts: {directory}", file=sys.stderr)
        raise
    finally:
        pending_error = sys.exc_info()[0] is not None
        manifest["finished_at_utc"] = _utc()
        manifest["wall_time_seconds"] = time.monotonic() - started
        try:
            if env is not None:
                env.close()
        except Exception as exc:
            manifest["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            manifest["execution_status"] = "failed"
            if not pending_error:
                raise
        finally:
            write_json(directory / "summary.json", manifest)
    return directory
