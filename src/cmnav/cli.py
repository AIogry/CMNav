"""Environment checks, fixed-plan exploration, and offline trajectory access."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cmnav import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cmnav", description="CMNav visual navigation experiments")
    parser.add_argument("--version", action="version", version=f"CMNav {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check dependencies and local scene/episode assets")
    doctor.add_argument("--config", default="configs/mp3d_smoke.yaml")
    doctor.add_argument("--output", help="Optional JSON report path (relative to project root)")
    run = commands.add_parser("run", help="Run the configured real Habitat environment")
    run.add_argument("--config", default="configs/mp3d_smoke.yaml")
    run.add_argument("--output-dir")
    run.add_argument("--episodes", type=int)
    run.add_argument("--max-steps", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--gpu", type=int)
    plan = commands.add_parser("plan-collection", help="Save deterministic development starts")
    plan.add_argument("--config", default="configs/mp3d_collect.yaml")
    collect = commands.add_parser("collect", help="Collect goal-free RGB-D from a saved plan")
    collect.add_argument("--config", default="configs/mp3d_collect.yaml")
    collect.add_argument("--plan", required=True)
    info = commands.add_parser("trajectory-info", help="Read a complete trajectory and verify hashes")
    info.add_argument("--path", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "trajectory-info":
            from cmnav.trajectories import TrajectoryReader
            trajectory = TrajectoryReader(Path(args.path).expanduser())
            print(json.dumps({"metadata": trajectory.metadata, "summary": trajectory.summary},
                             ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if args.command in ("plan-collection", "collect"):
            from cmnav.collection_config import load_collection_config
            from cmnav.collection import create_collection_plan, collect_trajectories
            collection_config = load_collection_config(args.config)
            if args.command == "plan-collection":
                destination = create_collection_plan(collection_config)
                print(f"Collection plan (simulator setup/audit only): {destination}")
            else:
                destination = collect_trajectories(collection_config, args.plan)
                print(f"Completed development collection: {destination}")
            return 0
        from cmnav.config import load_config
        from cmnav.diagnostics import inspect_environment
        overrides = None
        if args.command == "run":
            overrides = {"run.output_dir": args.output_dir, "run.episodes": args.episodes,
                         "environment.max_episode_steps": args.max_steps, "seed": args.seed,
                         "environment.gpu_device_id": args.gpu}
        config = load_config(args.config, overrides=overrides)
        report = inspect_environment(config)
        if args.command == "doctor":
            if args.output:
                report_path = Path(config["project_root"]) / Path(args.output).expanduser()
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                  allow_nan=False) + "\n", encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            return 0 if report["ready_for_run"] else 2
        if not report["ready_for_run"]:
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), file=sys.stderr)
            print("Preflight failed. Resolve the reported errors before running Habitat.", file=sys.stderr)
            return 2
        from cmnav.runner import run_experiment
        from cmnav.recording import write_json
        directory = run_experiment(config)
        write_json(directory / "preflight.json", report)
        print(f"Completed environment check. Artifacts: {directory}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"CMNav: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
