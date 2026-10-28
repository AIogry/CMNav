"""Stable report operations. These parse arguments and delegate to package capabilities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from cmnav.reporting.paths import bounded_output


def navigation_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Render validated saved physical trajectories and graph choices.')
    parser.add_argument("--run", type=Path, required=True, help="Saved graph-navigation run under this project's outputs/.")
    parser.add_argument("--task-index", type=int, required=True, help="Zero-based frozen task index.")
    parser.add_argument("--output-dir", type=Path, help="Destination under outputs/; default RUN/visualizations/.")
    args = parser.parse_args(argv)
    from cmnav.reporting.navigation import navigation_report
    if args.task_index < 0:
        parser.error("--task-index must be nonnegative")
    run = bounded_output(args.run)
    output = bounded_output(args.output_dir or run / "visualizations")
    diagnostics = navigation_report(run, args.task_index, output)
    print(json.dumps({"artifacts": diagnostics["artifacts"], "methods": diagnostics["methods"],
                      "dominant_node_pair": diagnostics["cml_graph_choices"]["dominant_node_pair"],
                      "dominant_pair_attempts": diagnostics["cml_graph_choices"]["dominant_pair_attempts"]},
                     indent=2, allow_nan=False))

def gcml_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Compare saved GCML navigation task records and routes.')
    parser.add_argument("--run", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    from cmnav.reporting.navigation import gcml_report
    run = Path(args.run).resolve()
    root = Path(__file__).resolve().parents[3]
    if not run.is_relative_to(root / "outputs"):
        raise ValueError("Run must be under this checkout's outputs")
    print(gcml_report(run, args.output_dir))

def subgoals_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Report the explicitly contracted frozen teacher-subgoal cohort.')
    parser.add_argument("run", nargs="?", type=Path)
    parser.add_argument("--run", dest="run_option", type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args(argv)
    from cmnav.reporting.navigation import subgoal_report
    run_path = arguments.run_option or arguments.run
    if run_path is None or (arguments.run_option is not None and arguments.run is not None):
        parser.error("provide exactly one RUN_DIRECTORY or --run RUN_DIRECTORY")
    print(subgoal_report(run_path, arguments.output_dir))

def dimensions_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Report protocol-declared GCML dimensions with an optional exact historical contract.')
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--contract", type=Path, help="Optional explicit historical expectation JSON; otherwise use the run protocol and manifests.")
    parser.add_argument("--output-dir", type=Path, help="New report directory inside the source run; existing reports are preserved.")
    arguments = parser.parse_args(argv)
    from cmnav.reporting.comparisons import dimension_report
    print(dimension_report(arguments.run, arguments.contract, arguments.output_dir))

def mechanisms_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Report the contracted CUDA mechanism matrix and separately labeled exploratory jobs.')
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    from cmnav.reporting.comparisons import mechanism_report
    print(mechanism_report(arguments.run, arguments.output))

def readout_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Plot saved fixed-Q readout curves with recorded label counts and thresholds.')
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args(argv)
    from cmnav.reporting.comparisons import readout_report
    print(readout_report(arguments.directory.expanduser().resolve(), arguments.output_dir))
