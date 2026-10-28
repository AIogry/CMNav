"""A fixed registry of reusable experiment capabilities with lazy dispatch."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
import uuid


# Values are maintained in code, never imported from a config-supplied module.
# Profiles identify protocol/record formats, not a date, dimension or run ID.
OPERATIONS = {
    ("hierarchy", "build"): ("cmnav.workflows.landmark_hierarchy", "build_main"),
    ("hierarchy", "train"): ("cmnav.workflows.landmark_hierarchy", "train_main"),
    ("hierarchy", "evaluate"): ("cmnav.workflows.landmark_hierarchy", "evaluate_main"),
    ("hierarchy", "report"): ("cmnav.workflows.landmark_hierarchy", "report_main"),
    ("hierarchy", "compare-noise"): ("cmnav.workflows.landmark_hierarchy", "compare_noise_main"),
    ("train-gcml", "replay"): ("cmnav.workflows.gcml_training", "replay_main"),
    ("train-gcml", "checkpoints"): ("cmnav.workflows.gcml_training", "checkpoints_main"),
    ("evaluate-gcml", "full-goal"): ("cmnav.workflows.gcml_evaluation", "full_goal_main"),
    ("evaluate-gcml", "condition"): ("cmnav.workflows.gcml_evaluation", "condition_main"),
    ("evaluate-gcml", "teacher-gates"): ("cmnav.workflows.gcml_gate_comparison", "main"),
    ("fit-gcml-readout", "frozen"): ("cmnav.workflows.gcml_readout", "main"),
    ("fit-gcml-readout", "reference"): ("cmnav.workflows.gcml_readout", "reference_main"),
    ("schedule", "conditions"): ("cmnav.workflows.condition_scheduling", "main"),
    ("analyze", "mechanisms"): ("cmnav.analysis.gcml_diagnostics", "mechanisms_main"),
    ("analyze", "affordances"): ("cmnav.analysis.affordance", "affordance_scores_main"),
    ("analyze", "full-goal"): ("cmnav.analysis.gcml_diagnostics", "full_goal_failures_main"),
    ("analyze", "failure-gates"): ("cmnav.analysis.affordance", "failure_gates_main"),
    ("analyze", "training"): ("cmnav.analysis.training_comparison", "main"),
    ("analyze", "collection"): ("cmnav.analysis.collection", "quality_main"),
    ("audit", "collection"): ("cmnav.analysis.collection", "audit_main"),
    ("report", "navigation"): ("cmnav.reporting.commands", "navigation_main"),
    ("report", "gcml"): ("cmnav.reporting.commands", "gcml_main"),
    ("report", "subgoals"): ("cmnav.reporting.commands", "subgoals_main"),
    ("report", "dimensions"): ("cmnav.reporting.commands", "dimensions_main"),
    ("report", "mechanisms"): ("cmnav.reporting.commands", "mechanisms_main"),
    ("report", "readout"): ("cmnav.reporting.commands", "readout_main"),
}


def _invoke(request: dict, *, record: bool = True) -> int:
    from cmnav.runtime.artifacts import atomic_write_json
    from cmnav.runtime.source_snapshot import archive_package_sources
    from cmnav.runtime.project import checkout_root, require_project_directory

    key = (request["operation"], request["profile"])
    if key not in OPERATIONS:
        raise ValueError(f"Unknown capability/profile {key}; use --list")
    module, name = OPERATIONS[key]
    working_directory = require_project_directory(Path(request["working_directory"]))
    previous_directory = Path.cwd()
    directory = None
    state = {"schema": "cmnav.experiment_invocation.v1", **request,
             "implementation": f"{module}:{name}", "python": sys.executable,
             "started_at": datetime.now(timezone.utc).isoformat(), "status": "running"}
    try:
        os.chdir(working_directory)
        if record:
            identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
            directory = checkout_root() / "outputs/experiment_invocations" / identity
            directory.mkdir(parents=True, exist_ok=False)
            if request.get("config_path"):
                config_source = Path(request["config_path"])
                content = config_source.read_bytes()
                state["config_sha256"] = hashlib.sha256(content).hexdigest()
                if state["config_sha256"] != request["config_sha256"]:
                    raise RuntimeError("Experiment configuration changed after parsing")
                (directory / ("input_config" + config_source.suffix)).write_bytes(content)
            atomic_write_json(directory / "invocation.json", state)
            archive_package_sources(directory / "source")
        result = getattr(importlib.import_module(module), name)(request["arguments"])
        status = result if type(result) is int else 0
        state.update(status="completed" if status == 0 else "failed", exit_code=status)
        return status
    except BaseException as exc:
        # argparse --help exits normally; preserve real parser/runner exit codes.
        successful_help = isinstance(exc, SystemExit) and exc.code in (None, 0)
        state.update(status="completed" if successful_help else "failed",
                     error=None if successful_help else f"{type(exc).__name__}: {exc}")
        if isinstance(exc, SystemExit):
            state["exit_code"] = (0 if exc.code is None else exc.code) if exc.code is None or type(exc.code) is int else 1
        elif isinstance(exc, KeyboardInterrupt):
            state["exit_code"] = 130
        else:
            state["exit_code"] = 1
        raise
    finally:
        try:
            if directory is not None:
                state["finished_at"] = datetime.now(timezone.utc).isoformat()
                atomic_write_json(directory / "invocation.json", state)
        finally:
            os.chdir(previous_directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cmnav experiment", description=__doc__)
    parser.add_argument("--list", action="store_true", help="List capabilities without importing models")
    parser.add_argument("--config", type=Path, help="JSON/YAML invocation configuration")
    parser.add_argument("--operation")
    parser.add_argument("--profile")
    parser.add_argument("--describe", action="store_true", help="Print resolved arguments; do not execute or write artifacts")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Direct options after --")
    args = parser.parse_args(argv)
    if args.list:
        for (operation, profile), (module, name) in OPERATIONS.items():
            print(f"{operation:20s} {profile:16s} {module}:{name}")
        return 0
    try:
        if args.config:
            if args.operation or args.profile or args.arguments:
                parser.error("--config cannot be mixed with direct operation/profile/arguments")
            from cmnav.configuration.experiment import load_experiment_config
            request = load_experiment_config(args.config)
        else:
            if not args.operation or not args.profile:
                parser.error("Provide --config or both --operation and --profile")
            arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
            request = {"operation": args.operation, "profile": args.profile,
                       "working_directory": str(Path.cwd()), "arguments": arguments}
        if (request["operation"], request["profile"]) not in OPERATIONS:
            parser.error("Unknown operation/profile; use --list")
        if args.describe:
            print(json.dumps(request, ensure_ascii=False, indent=2))
            return 0
        return _invoke(request, record=not any(x in ("--help", "-h") for x in request["arguments"]))
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"CMNav experiment: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
