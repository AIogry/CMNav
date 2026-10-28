"""CUDA/cohort provenance validation and paired mechanism statistics from saved artifacts."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
from cmnav.analysis.records import read_json
from cmnav.runtime.artifacts import sha256_file as digest
from cmnav.analysis.report_contracts import PLANNING_COUNTS, CACHE_COUNTS, CONTRASTS, FIVE_TASK_COHORT, COHORT_CONTRACT_PATH
from cmnav.analysis.comparison import sum_counters, paired_success_counts


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}: {value!r}")
    return value


def number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data: {value!r}")
    return float(value)


def successful(value: Any) -> bool:
    if value not in (False, True, 0, 1, 0., 1.):
        raise ValueError(f"Unexpected success indicator: {value!r}")
    return bool(value)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def resolve_input(value: str, run: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else run / path).resolve()


def unique_existing(paths: list[Path], description: str) -> Path:
    matches = [path for path in paths if path.exists()]
    if len(matches) != 1:
        raise ValueError(f"Expected one {description}; found {[str(path) for path in matches]}")
    return matches[0]


def count_map(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return {str(key): integer(count, f"{label}.{key}") for key, count in value.items()}


def normalize_condition(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a condition mapping")
    if "name" not in value and isinstance(value.get("condition"), dict):
        value = value["condition"]
    for key in ("name", "method", "checkpoint_variant"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"{label}.{key} is missing")
    integer(value.get("epoch"), f"{label}.epoch", 1)
    integer(value.get("planning_seed"), f"{label}.planning_seed")
    if not isinstance(value.get("overrides", {}), dict):
        raise ValueError(f"{label}.overrides must be a mapping")
    return {**value, "overrides": value.get("overrides", {}), "cohort": value.get("cohort", "unspecified"),
            "analysis_phase": value.get("analysis_phase", "prespecified")}


def condition_identity(condition: dict[str, Any]) -> tuple[str, int, int]:
    return condition["name"], condition["epoch"], condition["planning_seed"]


def condition_settings(condition: dict[str, Any]) -> dict[str, Any]:
    return {key: condition[key] for key in ("name", "method", "epoch", "checkpoint_variant", "overrides", "analysis_phase")}


def task_row(raw: dict[str, Any], condition: dict[str, Any], output: Path) -> dict[str, Any]:
    index = integer(raw.get("task_index"), "task_index")
    if raw.get("method") != condition["method"]:
        raise ValueError(f"Method mismatch in {output}, task {index}")
    for key in ("epoch", "planning_seed"):
        if raw.get(key) != condition[key]:
            raise ValueError(f"{key} mismatch in {output}, task {index}")
    row_condition = raw.get("condition")
    row_name = row_condition.get("name") if isinstance(row_condition, dict) else row_condition
    if row_name != condition["name"]:
        raise ValueError(f"Condition mismatch in {output}, task {index}")
    if raw.get("teacher_subgoals") is not None:
        raise ValueError(f"Unexpected teacher subgoals in full-goal study: {output}")
    metrics = raw.get("metrics", {})
    episode = raw.get("episode_id")
    if episode is None:
        raise ValueError(f"Missing episode_id in {output}, task {index}")
    planning = raw.get("gcml_planning")
    planning_applicable = condition["method"].startswith("gcml")
    if not planning_applicable and planning is None:
        # A declared graph-search control does not run the GCML planner.
        # These zeros are structural, not substitutes for missing GCML data.
        planning = {**dict.fromkeys(PLANNING_COUNTS, 0), "candidate_terminations": {},
                    "branches": {}, "planning_seconds": 0., "observed_planner_call_seconds": 0.,
                    "graph_audit_was_used_by_planner": False}
    if not isinstance(planning, dict):
        raise ValueError(f"Missing gcml_planning in {output}, task {index}; missing is not zero")
    counts = {key: integer(planning.get(key), f"gcml_planning.{key}") for key in PLANNING_COUNTS}
    terminations = count_map(planning.get("candidate_terminations"), "candidate_terminations")
    branches = count_map(planning.get("branches"), "branches")
    if sum(terminations.values()) != counts["candidates"]:
        raise ValueError(f"Candidate termination totals disagree in {output}, task {index}")
    if planning.get("graph_audit_was_used_by_planner") is not False:
        raise ValueError(f"Missing/unsafe independent-audit declaration in {output}, task {index}")
    cache = raw.get("gcml_plan_cache")
    cache_expected = condition["overrides"].get("gcml_cache_mode", "none") != "none"
    if cache_expected != (cache is not None):
        raise ValueError(f"Cache configuration/counters disagree in {output}, task {index}")
    if cache is not None:
        if cache.get("future_graph_audit_used") is not False:
            raise ValueError(f"Future graph audit affected cache in {output}, task {index}")
        if cache.get("full_fresh_planning_before_fallback") is not True:
            raise ValueError(f"Cache does not preserve fresh planning budget in {output}, task {index}")
        cache = {
            "mode": cache.get("mode"),
            "counts": {key: integer(cache.get("counts", {}).get(key), f"cache.{key}") for key in CACHE_COUNTS},
            "invalidation_reasons": count_map(cache.get("invalidation_reasons"), "cache.invalidation_reasons"),
        }
    return {
        "task_index": index, "episode_id": str(episode), "scene_id": raw.get("scene_id"),
        "start_node": raw.get("start_node"), "goal_node": raw.get("goal_node"),
        "condition": condition["name"], "epoch": condition["epoch"],
        "analysis_phase": condition["analysis_phase"],
        "planning_seed": condition["planning_seed"], "method": condition["method"],
        "task_planning_seed": raw.get("task_planning_seed", condition["planning_seed"] + index),
        "success": successful(metrics.get("success")), "spl": number(metrics.get("spl"), "SPL"),
        "steps": integer(raw.get("steps"), "steps"),
        "graph_hops_attempted": integer(raw.get("graph_hops_attempted"), "graph_hops_attempted"),
        "graph_hops_completed": integer(raw.get("graph_hops_completed"), "graph_hops_completed"),
        "end_reason": str(raw.get("end_reason", "missing")),
        "planning_counts": counts, "candidate_terminations": terminations, "planner_branches": branches,
        "gcml_planning_metrics_applicable": planning_applicable,
        "planning_seconds": number(planning.get("planning_seconds"), "planning_seconds"),
        "observed_planner_call_seconds": number(planning.get("observed_planner_call_seconds"), "observed_planner_call_seconds"),
        "cache": cache, "source_job": str(output),
    }


def load_study(run: Path) -> dict[str, Any]:
    inputs: dict[str, str] = {str(COHORT_CONTRACT_PATH): digest(COHORT_CONTRACT_PATH)}

    def read(path: Path) -> Any:
        result = read_json(path)
        inputs[str(path)] = digest(path)
        return result

    protocol = read(run / "protocol.json")
    jobs = read(run / "jobs.json")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs.json must be the nonempty complete scheduled job list")
    primary_job_count = len(jobs)
    job_phases = ["prespecified"] * primary_job_count
    exploratory_protocol_path = run / "exploratory_cache_002_protocol.json"
    exploratory_jobs_path = run / "exploratory_jobs.json"
    exploratory_protocol = None
    exploratory_job_count = 0
    if exploratory_protocol_path.exists() or exploratory_jobs_path.exists():
        if not exploratory_protocol_path.is_file() or not exploratory_jobs_path.is_file():
            raise ValueError("Exploratory protocol and jobs manifest must both be present")
        exploratory_protocol = read(exploratory_protocol_path)
        supplemental_jobs = read(exploratory_jobs_path)
        if not isinstance(supplemental_jobs, list) or not supplemental_jobs:
            raise ValueError("exploratory_jobs.json must contain its complete scheduled job list")
        if (exploratory_protocol.get("analysis_phase") != "posthoc_exploratory"
                or exploratory_protocol.get("selection_used_existing_development_results") is not True
                or exploratory_protocol.get("independent_test_claim") is not False):
            raise ValueError("Supplemental protocol must explicitly identify selection after observing development results")
        if exploratory_protocol.get("parent_planned_jobs") != primary_job_count:
            raise ValueError("Exploratory protocol's primary job count differs from the original jobs manifest")
        if resolve_input(exploratory_protocol["parent_protocol"], run) != run / "protocol.json":
            raise ValueError("Exploratory protocol does not reference this run's original protocol")
        if exploratory_protocol.get("dimension") != protocol.get("dimension"):
            raise ValueError("Exploratory protocol changes the trained representation dimension")
        exploratory_job_count = len(supplemental_jobs)
        jobs = jobs + supplemental_jobs
        job_phases += ["posthoc_exploratory"] * exploratory_job_count
    manifest_path = run / "training" / "manifest.json"
    manifest = read(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError(f"Training is incomplete: {manifest_path}")
    training = read(run / "training" / "training.json")
    if not str(training.get("device", "")).startswith("cuda"):
        raise ValueError("Saved training diagnostics do not confirm CUDA training")
    if training["device"] != protocol.get("training_device"):
        raise ValueError("Saved training device differs from the frozen protocol")
    expected_epochs = integer(protocol.get("epochs"), "protocol.epochs", 1)
    epoch_rows = training.get("epochs_detail")
    if not isinstance(epoch_rows, list) or [row.get("epoch") for row in epoch_rows] != list(range(1, expected_epochs + 1)):
        raise ValueError("Training history must contain every epoch through the planned final epoch")
    if any(row.get("finite") is not True for row in epoch_rows):
        raise ValueError("Training history includes nonfinite or unverified epochs")
    checkpoints = []
    for epoch in protocol["checkpoints"]:
        directory = unique_existing([
            run / "checkpoints" / f"epoch_{epoch:03d}",
            run / "training" / "checkpoints" / f"epoch_{epoch:03d}",
        ], f"epoch {epoch} checkpoint")
        checkpoint_manifest = read(directory / "manifest.json")
        if checkpoint_manifest.get("status") != "completed" or checkpoint_manifest.get("epoch") != epoch:
            raise ValueError(f"Checkpoint publication is incomplete: {directory}")
        diagnostics = read(directory / "diagnostics.json")
        readout = read(directory / "readout.json")
        if not isinstance(diagnostics.get("raw"), dict):
            raise ValueError(f"Missing raw checkpoint diagnostics: {directory}")
        for variant in ("raw", "minimum_change", "minimum_norm"):
            for scope in ("sources", "sinks", "all_states"):
                group = readout.get("variants", {}).get(variant, {}).get(scope)
                if not isinstance(group, dict):
                    raise ValueError(f"Missing readout {variant}/{scope}: {directory}")
                for key in ("fp", "fn"):
                    integer(group.get(key), f"readout.{variant}.{scope}.{key}")
                number(group.get("rmse"), f"readout.{variant}.{scope}.rmse")
        if not str(readout.get("solve", {}).get("device", "")).startswith("cuda"):
            raise ValueError(f"Readout fit does not confirm CUDA device: {directory}")
        checkpoints.append({"epoch": epoch, "directory": str(directory), "manifest": checkpoint_manifest,
                            "diagnostics": diagnostics, "readout": readout})

    numerical_path = run / "training" / "historical_cpu_checkpoint_comparison.json"
    numerical_comparison = read(numerical_path) if numerical_path.exists() else None
    control_audits = {}
    for name in ("historical_gcml_cuda_comparison", "greedy_checkpoint_log_comparison",
                 "remaining_historical_controls", "noise_suffix_opportunities"):
        audit_path = run / "control_audit" / f"{name}.json"
        audit_markdown = audit_path.with_suffix(".md")
        if not audit_path.exists() and not audit_markdown.exists():
            continue
        if not audit_path.is_file() or not audit_markdown.is_file():
            raise ValueError(f"Control audit publication is incomplete: {audit_path}")
        audit = read(audit_path)
        if name in ("historical_gcml_cuda_comparison", "remaining_historical_controls") and audit.get("source_inputs_unchanged") is not True:
            raise ValueError(f"Control audit does not confirm unchanged source artifacts: {audit_path}")
        if name == "greedy_checkpoint_log_comparison":
            # This audit records source hashes rather than a before/after flag.
            # Confirm those saved logs and diagnostics still match, and retain
            # the hashes in the report's own input manifest.
            sources = dict(audit["input_log_sha256"])
            sources[audit["graph"]["path"]] = audit["graph"]["sha256"]
            for item in audit["task2_epoch80_vs_epoch100"]["saved_full_checkpoint_inverse_mse"].values():
                sources[item["path"]] = item["sha256"]
            for source, expected in sources.items():
                actual = digest(Path(source))
                if actual != expected:
                    raise ValueError(f"A saved greedy audit input changed: {source}")
                inputs[source] = actual
            graph = read(Path(audit["graph"]["path"]))

            def verify_audit_edges(value: Any) -> None:
                if isinstance(value, list):
                    for child in value:
                        verify_audit_edges(child)
                elif isinstance(value, dict):
                    if all(key in value for key in ("edge_id", "source_node", "target_node")):
                        if graph["edges"][value["edge_id"]] != [value["source_node"], value["target_node"]]:
                            raise ValueError(f"Greedy audit edge endpoints disagree with graph: {value['edge_id']}")
                    if "selected_edge_id" in value and "selected_edge_nodes" in value:
                        if graph["edges"][value["selected_edge_id"]] != value["selected_edge_nodes"]:
                            raise ValueError(f"Greedy audit selected edge disagrees with graph: {value['selected_edge_id']}")
                    for child in value.values():
                        verify_audit_edges(child)

            verify_audit_edges(audit["task2_epoch80_vs_epoch100"])
        inputs[str(audit_markdown)] = digest(audit_markdown)
        audit_script = audit_path.with_suffix(".py")
        if audit_script.exists():
            inputs[str(audit_script)] = digest(audit_script)
        control_audits[name] = {"json_path": str(audit_path), "markdown_path": str(audit_markdown), "data": audit}

    loaded_jobs = []
    pending = []
    for job, phase in zip(jobs, job_phases):
        if not isinstance(job, dict) or not isinstance(job.get("output"), str):
            raise ValueError("Every job must name its output directory")
        output = resolve_input(job["output"], run)
        if not output.is_relative_to(run):
            raise ValueError(f"Evaluation output is outside this run: {output}")
        summary_path = output / "summary.json"
        if not summary_path.exists():
            pending.append(f"{output}: no summary")
            continue
        summary = read(summary_path)
        if summary.get("status") != "completed":
            pending.append(f"{output}: {summary.get('status')}")
            continue
        loaded_jobs.append((job, output, summary, phase))
    if pending:
        raise ValueError("Report not published: scheduled jobs are incomplete:\n" + "\n".join(pending))

    rows_by_key: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    conditions: dict[tuple[str, int], dict[str, Any]] = {}
    job_records = []
    identities: dict[int, tuple[Any, ...]] = {}
    devices = set()
    for job, output, summary, phase in loaded_jobs:
        condition = normalize_condition(summary.get("condition"), f"{output}/summary.condition")
        if condition["analysis_phase"] != phase:
            raise ValueError(f"Condition analysis phase disagrees with its jobs manifest: {output}")
        external_condition = job.get("condition_file", job.get("condition_path", job.get("condition")))
        if isinstance(external_condition, str):
            declared = normalize_condition(read(resolve_input(external_condition, run)), "scheduled condition")
        elif isinstance(external_condition, dict):
            declared = normalize_condition(external_condition, "scheduled condition")
        else:
            raise ValueError(f"Scheduled condition file is missing: {output}")
        if condition_identity(declared) != condition_identity(condition) or condition_settings(declared) != condition_settings(condition):
            raise ValueError(f"Scheduled and completed condition differ: {output}")
        if summary.get("parameters_unchanged") is not True:
            raise ValueError(f"Evaluation did not confirm frozen parameters: {output}")
        device = str(summary.get("device", ""))
        if not device.startswith("cuda") or device not in protocol.get("evaluation_devices", []):
            raise ValueError(f"Evaluation does not confirm CUDA inference: {output}")
        devices.add(device)
        evaluation = read(output / "evaluation.json")
        if isinstance(evaluation, dict) and isinstance(evaluation.get("results"), list):
            evaluation = evaluation["results"]
        if not isinstance(evaluation, list) or len(evaluation) != FIVE_TASK_COHORT.task_count:
            raise ValueError(f"Expected all five task results: {output}")
        normalized = [task_row(row, condition, output) for row in evaluation]
        if sorted(row["task_index"] for row in normalized) != list(range(FIVE_TASK_COHORT.task_count)):
            raise ValueError(f"Task indices must be exactly 0..4: {output}")
        saved_metrics = summary.get("metrics", {})
        if condition["method"] in saved_metrics:
            saved_metrics = saved_metrics[condition["method"]]
        for key, expected in (("success", sum(row["success"] for row in normalized) / FIVE_TASK_COHORT.task_count),
                              ("spl", sum(row["spl"] for row in normalized) / FIVE_TASK_COHORT.task_count),
                              ("mean_steps", sum(row["steps"] for row in normalized) / FIVE_TASK_COHORT.task_count)):
            if not math.isclose(number(saved_metrics.get(key), f"summary.metrics.{key}"), expected, rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError(f"Saved metrics disagree with task rows ({key}): {output}")
        if saved_metrics.get("episodes") != FIVE_TASK_COHORT.task_count:
            raise ValueError(f"Saved summary episode count is not five: {output}")
        group_key = condition["name"], condition["epoch"]
        if group_key in conditions and condition_settings(conditions[group_key]) != condition_settings(condition):
            raise ValueError(f"Condition settings changed across planning seeds: {group_key}")
        conditions.setdefault(group_key, condition)
        for row in normalized:
            signature = tuple(row[key] for key in ("episode_id", "scene_id", "start_node", "goal_node"))
            previous = identities.setdefault(row["task_index"], signature)
            if previous != signature:
                raise ValueError(f"Fixed task identity changed: task {row['task_index']}, {output}")
            key = (*condition_identity(condition), row["task_index"])
            if key in rows_by_key:
                # A checkpoint evaluation may be explicitly reused by the final
                # cohort. Never count it as a second seed/task repetition.
                prior = {k: v for k, v in rows_by_key[key].items() if k != "source_job"}
                current = {k: v for k, v in row.items() if k != "source_job"}
                if canonical(prior) != canonical(current):
                    raise ValueError(f"Conflicting duplicate seed/task evaluation: {key}")
            else:
                rows_by_key[key] = row
        job_records.append({"output": str(output), "condition": condition, "device": device,
                            "status": "completed", "parameters_unchanged": True, "analysis_phase": phase,
                            "checkpoint": str(resolve_input(job["checkpoint"], run)) if job.get("checkpoint") else None})
    rows = sorted(rows_by_key.values(), key=lambda row: (row["epoch"], row["condition"], row["planning_seed"], row["task_index"]))
    checkpoint_seed = protocol["planning_seeds"][0]
    row_keys = set(rows_by_key)
    for epoch in protocol["checkpoints"]:
        for name in protocol["checkpoint_conditions"]:
            for task in range(FIVE_TASK_COHORT.task_count):
                if (name, epoch, checkpoint_seed, task) not in row_keys:
                    raise ValueError(f"Checkpoint curve is incomplete: {name}, epoch {epoch}, seed {checkpoint_seed}")
    for name in protocol["final_conditions"]:
        if (name, expected_epochs) not in conditions:
            raise ValueError(f"A final protocol condition has no completed job: {name}")
    allowed_seeds = set(protocol["planning_seeds"])
    if any(row["planning_seed"] not in allowed_seeds for row in rows):
        raise ValueError("A completed job uses an undeclared planning seed")
    if exploratory_protocol is not None:
        baseline_name = exploratory_protocol["baseline_condition"]
        treatment_name = exploratory_protocol["condition"]
        baseline_jobs = {item["condition"]["planning_seed"]: item for item in job_records
                         if item["condition"]["name"] == baseline_name and item["analysis_phase"] == "prespecified"}
        supplemental_records = [item for item in job_records if item["analysis_phase"] == "posthoc_exploratory"]
        if len(supplemental_records) != len(exploratory_protocol["planning_seeds"]):
            raise ValueError("Exploratory job count differs from the explicitly planned seed repeats")
        if sorted(item["condition"]["planning_seed"] for item in supplemental_records) != sorted(exploratory_protocol["planning_seeds"]):
            raise ValueError("Exploratory seed set differs from its protocol")
        for treatment_job in supplemental_records:
            treatment = treatment_job["condition"]
            baseline_job = baseline_jobs.get(treatment["planning_seed"])
            if baseline_job is None or treatment["name"] != treatment_name:
                raise ValueError("An exploratory cache job has no matching primary baseline")
            baseline = baseline_job["condition"]
            for key in ("epoch", "method", "checkpoint_variant"):
                if treatment[key] != baseline[key]:
                    raise ValueError(f"Exploratory cache comparison changed {key}")
            if treatment["epoch"] != exploratory_protocol["epoch"]:
                raise ValueError("Exploratory checkpoint epoch differs from its protocol")
            if (treatment["checkpoint_variant"] != exploratory_protocol["variant"]
                    or treatment["overrides"].get("gcml_noise_scale") != exploratory_protocol["noise"]):
                raise ValueError("Exploratory checkpoint variant or noise differs from its protocol")
            if treatment_job["checkpoint"] is None or treatment_job["checkpoint"] != baseline_job["checkpoint"]:
                raise ValueError("Exploratory cache comparison did not schedule the same checkpoint file")
            expected_overrides = {**baseline["overrides"], "gcml_cache_mode": "suffix_on_no_plan"}
            if treatment["overrides"] != expected_overrides or baseline["overrides"].get("gcml_cache_mode", "none") != "none":
                raise ValueError("Exploratory cache overrides change more than cache_mode none -> suffix_on_no_plan")
    return {"run": str(run), "protocol": protocol, "manifest": manifest, "training": training,
            "checkpoints": checkpoints, "rows": rows, "conditions": list(conditions.values()),
            "jobs": job_records, "evaluation_devices": sorted(devices), "input_sha256": inputs,
            "planned_jobs": len(jobs), "completed_jobs": len(loaded_jobs),
            "primary_planned_jobs": primary_job_count, "primary_completed_jobs": primary_job_count,
            "exploratory_planned_jobs": exploratory_job_count, "exploratory_completed_jobs": exploratory_job_count,
            "exploratory_protocol": exploratory_protocol,
            "historical_cpu_checkpoint_comparison": numerical_comparison,
            "control_audits": control_audits}


def aggregate(rows: list[dict[str, Any]], condition: dict[str, Any]) -> dict[str, Any]:
    selected = [row for row in rows if (row["condition"], row["epoch"]) == (condition["name"], condition["epoch"])]
    seeds = sorted({row["planning_seed"] for row in selected})
    per_seed = []
    for seed in seeds:
        seed_rows = [row for row in selected if row["planning_seed"] == seed]
        if len(seed_rows) != FIVE_TASK_COHORT.task_count:
            raise ValueError(f"Incomplete seed block: {condition['name']}, {seed}")
        per_seed.append({"planning_seed": seed, "successes": sum(row["success"] for row in seed_rows),
                         "tasks": FIVE_TASK_COHORT.task_count, "mean_spl": float(np.mean([row["spl"] for row in seed_rows])),
                         "no_plan_tasks": sum(row["end_reason"] == "no_plan" for row in seed_rows)})
    per_task = []
    for task in range(FIVE_TASK_COHORT.task_count):
        task_rows = [row for row in selected if row["task_index"] == task]
        per_task.append({"task_index": task, "episode_id": task_rows[0]["episode_id"],
                         "successes": sum(row["success"] for row in task_rows), "planning_seed_repeats": len(task_rows),
                         "success_frequency": sum(row["success"] for row in task_rows) / len(task_rows),
                         "end_reasons": dict(Counter(row["end_reason"] for row in task_rows))})
    counts = {key: sum(row["planning_counts"][key] for row in selected) for key in PLANNING_COUNTS}
    terminations = sum_counters(row["candidate_terminations"] for row in selected)
    branches = sum_counters(row["planner_branches"] for row in selected)
    cache_rows = [row for row in selected if row["cache"] is not None]
    cache = None
    if cache_rows:
        if len(cache_rows) != len(selected):
            raise ValueError(f"Cache mode changed within {condition['name']}")
        cache_counts = {key: sum(row["cache"]["counts"][key] for row in selected) for key in CACHE_COUNTS}
        cache = {"counts": cache_counts,
                 "invalidation_reasons": sum_counters(row["cache"]["invalidation_reasons"] for row in selected),
                 "task_repetitions_with_plan_stored": sum(row["cache"]["counts"]["plans_stored"] > 0 for row in selected),
                 "task_repetitions_with_fallback_used": sum(row["cache"]["counts"]["fallback_actions_selected"] > 0 for row in selected),
                 "no_cache_opportunity": cache_counts["plans_stored"] == 0,
                 "still_replans_with_full_budget_before_each_fallback": True}
    return {
        "condition": condition, "unique_tasks": FIVE_TASK_COHORT.task_count, "planning_seeds": seeds,
        "seed_task_repetitions": len(selected), "training_seed_count": FIVE_TASK_COHORT.training_seed_count,
        "per_seed": per_seed, "per_task": per_task,
        "mean_successes_per_seed": float(np.mean([block["successes"] for block in per_seed])),
        "repeat_success_fraction": sum(row["success"] for row in selected) / len(selected),
        "mean_spl": float(np.mean([row["spl"] for row in selected])),
        "mean_steps": float(np.mean([row["steps"] for row in selected])),
        "sum_graph_hops_completed": sum(row["graph_hops_completed"] for row in selected),
        "no_plan_task_repetitions": sum(row["end_reason"] == "no_plan" for row in selected),
        "end_reasons": dict(Counter(row["end_reason"] for row in selected)),
        "planning_counts": counts, "candidate_terminations": terminations, "planner_branches": branches,
        "fresh_no_plan_or_no_candidates_calls": branches.get("no_plan", 0) + branches.get("no_candidates", 0),
        "planning_seconds": sum(row["planning_seconds"] for row in selected),
        "observed_planner_call_seconds": sum(row["observed_planner_call_seconds"] for row in selected),
        "cache": cache,
    }


def contrasts(rows: list[dict[str, Any]], final_epoch: int,
              definitions: tuple[tuple[str, str, str], ...] = CONTRASTS) -> list[dict[str, Any]]:
    result = []
    for label, baseline, treatment in definitions:
        before = {(row["planning_seed"], row["task_index"]): row for row in rows
                  if row["epoch"] == final_epoch and row["condition"] == baseline}
        after = {(row["planning_seed"], row["task_index"]): row for row in rows
                 if row["epoch"] == final_epoch and row["condition"] == treatment}
        matched = sorted(before.keys() & after.keys())
        if not matched:
            continue
        result.append({"label": label, "baseline": baseline, "treatment": treatment,
                       "matched_seed_task_repetitions": len(matched),
                       "matched_unique_tasks": len({task for _, task in matched}),
                       **paired_success_counts(before, after),
                       "statistical_independence_claim": False})
    return result


def exploratory_result(study: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any] | None:
    protocol = study["exploratory_protocol"]
    if protocol is None:
        return None
    baseline_name, treatment_name = protocol["baseline_condition"], protocol["condition"]
    epoch = protocol["epoch"]
    lookup = {(group["condition"]["name"], group["condition"]["epoch"]): group for group in groups}
    paired = contrasts(study["rows"], epoch, (("观察后追加：噪声 0.02 的 cache", baseline_name, treatment_name),))
    before = {(row["planning_seed"], row["task_index"]): row for row in study["rows"]
              if row["epoch"] == epoch and row["condition"] == baseline_name}
    after = {(row["planning_seed"], row["task_index"]): row for row in study["rows"]
             if row["epoch"] == epoch and row["condition"] == treatment_name}
    if before.keys() != after.keys():
        raise ValueError("Exploratory cache comparison must have the same complete seed/task blocks as its baseline")
    task_pairs = []
    for key in sorted(after):
        baseline, treatment = before[key], after[key]
        if treatment["cache"] is None:
            raise ValueError("Exploratory cache row has no cache counters")
        task_pairs.append({
            "planning_seed": key[0], "task_index": key[1], "episode_id": baseline["episode_id"],
            "baseline": baseline, "treatment": treatment,
            "baseline_failed_after_completed_graph_hops": (baseline["end_reason"] == "no_plan"
                                                            and baseline["graph_hops_completed"] > 0),
            "success_change": int(treatment["success"]) - int(baseline["success"]),
        })
    return {"analysis_phase": "posthoc_exploratory", "selection_used_existing_development_results": True,
            "independent_test_claim": False, "protocol": protocol,
            "planned_jobs": study["exploratory_planned_jobs"], "completed_jobs": study["exploratory_completed_jobs"],
            "baseline": lookup[baseline_name, epoch], "treatment": lookup[treatment_name, epoch],
            "paired_contrast": paired[0], "task_pairs": task_pairs}
