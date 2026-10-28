"""Environment checks, fixed-plan exploration, and offline trajectory access."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cmnav import __version__


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["experiment"]:
        from cmnav.experiments import main as experiment_main
        return experiment_main(argv[1:])
    parser = argparse.ArgumentParser(prog="cmnav", description="CMNav visual navigation experiments")
    parser.add_argument("--version", action="version", version=f"CMNav {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("experiment", help="Run registered training/evaluation/analysis capabilities from config")
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
    navigation_plan = commands.add_parser("plan-navigation", help="Freeze independent official development tasks")
    navigation_plan.add_argument("--config", default="configs/mp3d_oracle_graph.yaml")
    graph_navigation = commands.add_parser(
        "diagnose-graph-navigation", aliases=["prototype"],
        help="Run the privileged frozen-map graph diagnostic (prototype is a deprecated alias)")
    graph_navigation.add_argument("--config", default="configs/mp3d_oracle_graph.yaml")
    graph_navigation.add_argument("--plan", required=True)
    graph_navigation.add_argument("--reference-run", help="Reuse immutable reference RGB-D/audits from a previous collection run")
    gcml = commands.add_parser("diagnose-gcml-navigation", help="Compare GCML and graph search on a frozen supported task manifest")
    gcml.add_argument("--config", default="configs/mp3d_gcml.yaml")
    gcml.add_argument("--plan", required=True, help="Original learning/evaluation manifest")
    gcml.add_argument("--supported-tasks", required=True, help="Independent frozen graph-covered evaluation manifest")
    gcml.add_argument("--graph-run", required=True, help="Existing immutable observed graph run")
    gcml.add_argument("--reference-run", required=True, help="Original reference RGB-D and audits")
    subgoals = commands.add_parser("diagnose-gcml-subgoals", help="Evaluate a frozen GCML checkpoint with teacher subgoals")
    subgoals.add_argument("--config", default="configs/mp3d_gcml_subgoals.yaml")
    args = parser.parse_args(argv)
    try:
        if args.command == "diagnose-gcml-subgoals":
            from cmnav.configuration.gcml_subgoals import load_gcml_subgoal_config
            from cmnav.workflows.gcml_subgoals import run_gcml_subgoal_comparison
            destination = run_gcml_subgoal_comparison(load_gcml_subgoal_config(args.config))
            print(f"Completed teacher-subgoal diagnostic: {destination}")
            return 0
        if args.command == "diagnose-gcml-navigation":
            from cmnav.configuration.gcml_navigation import load_gcml_navigation_config
            from cmnav.workflows.gcml_navigation import run_gcml_navigation
            config = load_gcml_navigation_config(args.config)
            destination = run_gcml_navigation(config, args.plan, supported_tasks=args.supported_tasks,
                                              graph_run=args.graph_run, reference_run=args.reference_run)
            print(f"Completed privileged GCML comparison: {destination}")
            return 0
        if args.command in ("plan-navigation", "diagnose-graph-navigation", "prototype"):
            if args.command == "prototype":
                print("'prototype' is deprecated; use 'diagnose-graph-navigation'.", file=sys.stderr)
            from cmnav.configuration.graph_navigation import load_graph_navigation_config
            config = load_graph_navigation_config(args.config)
            if args.command == "plan-navigation":
                from cmnav.data.navigation_plans import create_navigation_plan
                destination = create_navigation_plan(
                    config["base"], Path(config["project_root"]) / "data" / "navigation_plans",
                    learning_episodes=config["learning_episodes"],
                    evaluation_episodes=config["evaluation_episodes"], seed=config["seed"])
                print(f"Privileged development task plan: {destination}")
            else:
                from cmnav.workflows.oracle_graph_navigation import run_oracle_graph_navigation
                destination = run_oracle_graph_navigation(config, args.plan, reference_run=args.reference_run)
                print(f"Completed privileged diagnostic (not visual navigation): {destination}")
            return 0
        if args.command == "trajectory-info":
            from cmnav.data.trajectories import TrajectoryReader
            trajectory = TrajectoryReader(Path(args.path).expanduser())
            print(json.dumps({"metadata": trajectory.metadata, "summary": trajectory.summary},
                             ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if args.command in ("plan-collection", "collect"):
            from cmnav.configuration.exploration import load_collection_config
            collection_config = load_collection_config(args.config)
            if args.command == "plan-collection":
                from cmnav.data.exploration_plans import create_collection_plan
                destination = create_collection_plan(collection_config)
                print(f"Collection plan (simulator setup/audit only): {destination}")
            else:
                from cmnav.workflows.exploration_collection import collect_trajectories
                destination = collect_trajectories(collection_config, args.plan)
                print(f"Completed development collection: {destination}")
            return 0
        from cmnav.configuration.environment import load_config
        from cmnav.envs.diagnostics import inspect_environment
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
        from cmnav.workflows.environment_check import run_environment_check
        from cmnav.runtime.artifacts import write_json
        directory = run_environment_check(config)
        write_json(directory / "preflight.json", report)
        print(f"Completed environment check. Artifacts: {directory}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"CMNav: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
