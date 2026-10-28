#!/usr/bin/env python3
"""Freeze graph-supported official tasks before training or method evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cmnav.data.navigation_plans import episode_lists, load_navigation_plan
from cmnav.data.supported_navigation_tasks import create_supported_navigation_tasks
from cmnav.envs.oracle_navigation import OracleNavigationEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-run", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/supported_navigation_tasks"))
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--min-path-hops", type=int, default=8)
    args = parser.parse_args()
    source = args.graph_run.expanduser().resolve()
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    base = config["base"]
    plan = load_navigation_plan(args.plan, base)
    learning, _ = episode_lists(plan, base)
    environment = OracleNavigationEnvironment(base, episodes=learning[:1], oracle_state_diagnostic=True,
                                               goal_radius=config["local_goal_radius"])
    try:
        environment.reset()
        output = args.output_dir
        if not output.is_absolute():
            output = Path(base["project_root"]) / output
        destination = create_supported_navigation_tasks(
            base, plan, source / "graph.json", output, environment.geodesic_distance,
            count=args.count, seed=args.seed, attachment_radius=config["attachment_radius"],
            min_path_hops=args.min_path_hops,
        )
        print(f"Frozen supported task manifest: {destination}", flush=True)
        payload = json.loads(destination.read_text(encoding="utf-8"))["payload"]
        print(json.dumps({"candidate_status_counts": payload["candidate_status_counts"],
                          "selected": [{"episode": row["source_episode_id"], **row["coverage"]}
                                       for row in payload["evaluation"]]}, indent=2))
    finally:
        environment.close()


if __name__ == "__main__":
    main()
