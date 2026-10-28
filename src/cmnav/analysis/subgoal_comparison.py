"""Frozen teacher-subgoal cohort validation and stage-denominator statistics."""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any
import numpy as np
from cmnav.analysis.records import read_json
from cmnav.runtime.artifacts import sha256_file as digest
from cmnav.analysis.navigation_records import load_method
from cmnav.analysis.report_contracts import MODES, INTERVALS, CONDITIONS, COUNT_KEYS, PENDING_STATUSES, FIVE_TASK_COHORT, COHORT_CONTRACT_PATH
from cmnav.analysis.comparison import sum_counters


def nonnegative_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def stage_diagnostics(summary: dict[str, Any], interval: int | None, node_count: int) -> dict[str, Any] | None:
    teacher = summary.get("teacher_subgoals")
    if interval is None:
        if teacher is not None:
            raise ValueError("Final-goal control unexpectedly contains teacher subgoals")
        return None
    if not isinstance(teacher, dict):
        raise ValueError("A staged condition is missing teacher_subgoals diagnostics")
    nodes = teacher["nodes"]
    if not isinstance(nodes, list) or not nodes or any(type(n) is not int or not 0 <= n < node_count for n in nodes):
        raise ValueError("Teacher subgoals must be nonempty observed node IDs")
    completed = nonnegative_integer(teacher["completed_stages"], "completed_stages")
    total = nonnegative_integer(teacher["total_stages"], "total_stages")
    if total != len(nodes) or completed > total:
        raise ValueError("Teacher stage counts disagree with their frozen node list")
    if nodes[-1] != teacher["final_goal_node"] or teacher["final_goal_node"] != summary["goal_node"]:
        raise ValueError("The final teacher subgoal is not the final task goal node")
    if bool(teacher["final_goal_success"]) != bool(summary["metrics"]["success"]):
        raise ValueError("Teacher final success differs from environment task success")
    stages = teacher["stages"]
    if not isinstance(stages, list):
        raise ValueError("teacher_subgoals.stages must be a list")
    attempted = [stage for stage in stages if stage.get("start_step") is not None
                 and stage.get("status") not in PENDING_STATUSES]
    if not completed <= len(attempted) <= total:
        raise ValueError("Teacher stages must satisfy completed <= attempted <= total")
    if (len(stages) != total or sum(stage.get("status") == "completed" for stage in stages) != completed
            or teacher.get("attempted_stages", len(attempted)) != len(attempted)):
        raise ValueError("Teacher stage status records disagree with their summary counts")
    if (any(stage.get("status") not in {"completed", "failed", "not_attempted"} for stage in stages)
            or any(stage.get("status") != "completed" for stage in attempted[:completed])
            or any(stage.get("status") == "completed" for stage in attempted[completed:])):
        raise ValueError("Teacher stages do not follow the declared sequential, no-skipping protocol")
    indices = [nonnegative_integer(stage["index"], "stage index") for stage in stages]
    if len(set(indices)) != len(indices) or indices != sorted(indices):
        raise ValueError("Teacher stages must be unique and in temporal order")
    for stage in attempted:
        if stage["goal_node"] not in nodes:
            raise ValueError("Attempted stage goal is absent from the frozen teacher node list")
        start = nonnegative_integer(stage["start_step"], "stage start_step")
        end = stage.get("end_step")
        if end is not None and (type(end) is not int or not start <= end <= summary["steps"]):
            raise ValueError("Teacher stage step boundary lies outside the saved task")
    first_failure = None
    if len(attempted) > completed:
        stage = attempted[completed]
        first_failure = {"stage_ordinal": completed + 1, "recorded_index": stage["index"],
                         "goal_node": stage["goal_node"], "status": stage.get("status"),
                         "end_reason": stage.get("end_reason", summary["end_reason"]),
                         "start_step": stage["start_step"], "end_step": stage.get("end_step"),
                         "start_hop": stage.get("start_hop"), "end_hop": stage.get("end_hop")}
    elif not summary["metrics"]["success"]:
        first_failure = {"stage_ordinal": None, "recorded_index": None, "goal_node": summary["goal_node"],
                         "status": "outside_attempted_stage", "end_reason": summary["end_reason"],
                         "scope": "before_first_stage_or_after_last_stage_connector"}
    return {
        "nodes": nodes, "completed": completed, "attempted": len(attempted), "total": total,
        "unattempted": total - len(attempted),
        "completion_given_attempted": completed / len(attempted) if attempted else None,
        "completion_of_all_planned": completed / total if total else None,
        "first_failure": first_failure, "stages": stages,
        "counting": "attempted_has_start_step_and_nonpending_status; completed_from_saved_stage_summary",
    }


def aggregate(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]
    if not selected:
        raise ValueError(f"No results for {condition}")
    result = {
        "condition": condition, "method": selected[0]["method"], "interval": selected[0]["interval"],
        "tasks": len(selected), "successes": sum(bool(row["success"]) for row in selected),
        "success_rate": sum(row["success"] for row in selected) / len(selected),
        "mean_spl": float(np.mean([row["spl"] for row in selected])),
        "mean_steps": float(np.mean([row["steps"] for row in selected])),
        "sum_steps": sum(row["steps"] for row in selected),
        "no_plan_tasks": sum(row["end_reason"] == "no_plan" for row in selected),
        "end_reasons": dict(Counter(row["end_reason"] for row in selected)),
        "planning": {key: sum(row["planning"][key] for row in selected) for key in COUNT_KEYS},
        "planner_seconds": sum(row["planner_seconds"] for row in selected),
        "observed_planner_call_seconds": sum(row["observed_planner_call_seconds"] for row in selected),
        "candidate_terminations": sum_counters(row["candidate_terminations"] for row in selected),
    }
    staged = [row["stages"] for row in selected if row["stages"] is not None]
    if staged:
        completed, attempted, total = (sum(stage[key] for stage in staged) for key in ("completed", "attempted", "total"))
        result["stages"] = {
            "completed": completed, "attempted": attempted, "total": total, "unattempted": total - attempted,
            "completion_given_attempted": completed / attempted if attempted else None,
            "completion_of_all_planned": completed / total if total else None,
        }
    else:
        result["stages"] = None
    return result


def load_subgoal_comparison(run: Path) -> dict[str, Any]:
    inputs = {name: run / f"{name}.json" for name in
              ("summary", "config", "graph", "model", "teacher_subgoal_plan", "supported_tasks",
               "checkpoint_source", "protocol", "baseline_replay_checks")}
    if any(not path.is_file() for path in inputs.values()):
        raise ValueError("A completed subgoal run must contain summary/config/graph/model/teacher plan/supported tasks")
    inputs["report_cohort_contract"] = COHORT_CONTRACT_PATH
    summary, config, graph = (read_json(inputs[name]) for name in ("summary", "config", "graph"))
    if summary.get("status") != "completed":
        raise ValueError("Report only a completed comparison")
    if set(summary["methods"]) != set(CONDITIONS):
        raise ValueError("Subgoal comparison must contain all eight declared conditions")
    checkpoint = read_json(inputs["checkpoint_source"])
    if summary.get("training_performed") is not False or summary.get("all_parameters_unchanged") is not True:
        raise ValueError("This report requires the same frozen model without additional training")
    for name in ("model", "graph", "supported_tasks"):
        if digest(inputs[name]) != checkpoint["fingerprints"][name]:
            raise ValueError(f"The copied {name} differs from its frozen source fingerprint")
    baseline_checks = read_json(inputs["baseline_replay_checks"])
    if len(baseline_checks) != FIVE_TASK_COHORT.task_count * len(MODES) or not all(row["all_match"] for row in baseline_checks):
        raise ValueError("All ten final-goal controls must reproduce the original run")
    teacher_records = read_json(inputs["teacher_subgoal_plan"])["payload"]["records"]
    tasks = read_json(inputs["supported_tasks"])["payload"]["evaluation"]
    if len(tasks) != FIVE_TASK_COHORT.task_count:
        raise ValueError("This comparison requires the five frozen official tasks")
    if len(teacher_records) != len(tasks):
        raise ValueError("Teacher schedule and official task counts differ")
    graph_positions = np.asarray(graph["positions"], dtype=float)
    if graph_positions.ndim != 2 or graph_positions.shape[1] != 3 or not np.isfinite(graph_positions).all():
        raise ValueError("Observed graph positions must be a finite Nx3 array")
    edges = graph["edges"]
    if any(len(edge) != 2 or any(type(n) is not int or not 0 <= n < len(graph_positions) for n in edge) for edge in edges):
        raise ValueError("Invalid observed graph edge endpoints")
    existing = set(path.parent.relative_to(run / "evaluation").as_posix()
                   for path in (run / "evaluation").glob("*/task_*/summary.json"))
    expected = {f"{condition}/task_{index:04d}" for condition in CONDITIONS for index in range(len(tasks))}
    if existing != expected:
        raise ValueError("Exactly 40 task records (eight conditions × five tasks) are required")
    max_steps = nonnegative_integer(config["max_steps"], "max_steps")
    rows = []
    loaded_tasks: list[dict[str, dict[str, Any]]] = []
    for index, task in enumerate(tasks):
        task_name = f"task_{index:04d}"
        loaded = {condition: load_method(run, task_name, condition) for condition in CONDITIONS}
        first = loaded[CONDITIONS[0]]
        for mode in MODES:
            for interval in INTERVALS:
                condition = f"{mode}_{interval if interval is not None else 'final'}"
                data = loaded[condition]
                task_summary, stats = data["summary"], data["stats"]
                if task_summary["method"] != mode or task_summary.get("condition", condition) != condition:
                    raise ValueError("Task method/condition metadata differs from its directory")
                if task_summary.get("interval") != interval or task_summary.get("parameters_unchanged") is not True:
                    raise ValueError("Task interval or frozen-parameter verification is missing or inconsistent")
                if (stats["episode_id"] != str(task["source_episode_id"])
                        or not np.allclose(data["positions"][0], first["positions"][0], rtol=0, atol=1e-5)
                        or not np.allclose(data["execution"][0]["goal_position"], first["execution"][0]["goal_position"], rtol=0, atol=1e-5)):
                    raise ValueError("Task identity, physical start or final goal differs across conditions")
                if stats["steps"] > max_steps:
                    raise ValueError("A condition exceeded the shared whole-task environment budget")
                if not task_summary["endpoints_supported"] or not task_summary["directed_graph_reachable"]:
                    raise ValueError("A frozen comparison task lost its graph coverage")
                stages = stage_diagnostics(task_summary, interval, len(graph_positions))
                frozen_record = teacher_records[index]
                if (str(frozen_record["source_episode_id"]) != stats["episode_id"]
                        or frozen_record["start_node"] != task_summary["start_node"]
                        or frozen_record["goal_node"] != task_summary["goal_node"]):
                    raise ValueError("Task endpoints disagree with the frozen teacher schedule")
                if stages is not None and stages["nodes"] != frozen_record["schedules"][str(interval)]["subgoal_nodes"]:
                    raise ValueError("Actually supplied teacher goals differ from the frozen schedule")
                planning = task_summary["gcml_planning"]
                row = {"task_index": index, "task": task_name, "condition": condition, "method": mode,
                       "interval": interval, **stats, "start_node": task_summary["start_node"],
                       "goal_node": task_summary["goal_node"], "stages": stages,
                       "planning": {key: int(planning.get(key, 0)) for key in COUNT_KEYS},
                       "planner_seconds": float(planning.get("planning_seconds", 0)),
                       "observed_planner_call_seconds": float(planning.get("observed_planner_call_seconds", 0)),
                       "candidate_terminations": planning.get("candidate_terminations", {}),
                       "summary_path": str(data["files"]["summary.json"])}
                if any(not math.isfinite(row[key]) for key in ("success", "spl", "planner_seconds", "observed_planner_call_seconds")):
                    raise ValueError("A reported metric is nonfinite")
                rows.append(row)
                inputs[f"{condition}/{task_name}/summary"] = data["files"]["summary.json"]
                inputs[f"{condition}/{task_name}/execution"] = data["files"]["execution.jsonl"]
                inputs[f"{condition}/{task_name}/planning"] = data["files"]["planning.jsonl"]
                inputs[f"{condition}/{task_name}/steps"] = data["files"]["steps.jsonl"]
        loaded_tasks.append(loaded)
    aggregates = [aggregate(rows, condition) for condition in CONDITIONS]
    for value in aggregates:
        saved = summary["methods"][value["condition"]]
        for ours, saved_key in (("success_rate", "success"), ("mean_spl", "spl"), ("mean_steps", "mean_steps")):
            if saved_key in saved and not math.isclose(value[ours], float(saved[saved_key]), rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError(f"Recomputed aggregate differs from summary: {value['condition']} {ours}")
    return {"run": run, "inputs": inputs, "summary": summary, "config": config,
            "graph_positions": graph_positions, "edges": edges, "tasks": tasks,
            "rows": rows, "loaded_tasks": loaded_tasks, "aggregates": aggregates,
            "baseline_checks": baseline_checks, "checkpoint": checkpoint, "max_steps": max_steps}
