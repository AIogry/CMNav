"""Teacher-assisted gate comparison adapter with strict historical replay controls.

No future graph audit feeds the planner. The paired-soft profile must reproduce
its declared old-code hard control before starting soft navigation.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import shutil
from time import monotonic
import traceback
from typing import Any
import uuid
import torch
from cmnav.data.navigation_plans import load_navigation_plan
from cmnav.data.supported_navigation_tasks import load_supported_navigation_tasks, supported_episode_list
from cmnav.data.teacher_subgoals import validate_teacher_subgoal_manifest
from cmnav.envs.diagnostics import inspect_environment
from cmnav.envs.oracle_navigation import OracleNavigationEnvironment
from cmnav.evaluation.oracle_graph_navigation import evaluate_graph_navigation_task
from cmnav.evaluation.frozen_gcml import restore_model
from cmnav.memory.oracle_graph import OracleObservedGraph
from cmnav.memory.snapshots import SceneMemoryStore
from cmnav.methods.gcml import GenerativeCognitiveMapLearner
from cmnav.runtime.artifacts import require_within, sha256_file, write_json
from cmnav.runtime.provenance import runtime_versions, source_hashes
from cmnav.runtime.gcml_artifacts import load_json as read, parameter_hash as tensor_hash, load_checkpoint
from cmnav.runtime.gcml_source import checkout_root, snapshot_gcml_source, load_historical_contracts
from cmnav.analysis.navigation_comparison import baseline_check
from cmnav.analysis.gate_comparison import audit_imagined_sources, aggregate
from cmnav.reporting.gate_comparison import report_markdown

PROTOCOL = "frozen_gcml_four_edge_teacher_gate_comparison_v2"

def run(args: argparse.Namespace) -> Path:
    ROOT = checkout_root()
    contracts, contract_path, contract_hash = load_historical_contracts(getattr(args, "contract", None))
    expected = contracts["teacher_gate"]
    source_path = args.source or ROOT / expected["source_run"]
    source = require_within(source_path.expanduser().resolve(), ROOT / "outputs")
    if read(source / "summary.json").get("status") != "completed":
        raise ValueError("Source subgoal experiment must be complete")
    source_config = read(source / "config.json")
    config = copy.deepcopy(source_config)
    base = config["base"]
    if any(config[key] != expected[key] for key in ("max_steps", "gcml_candidates", "gcml_horizon", "gcml_noise_scale")):
        raise ValueError("Expected the existing 500-step, K64/H96/noise0.2 protocol")
    thresholds = args.thresholds
    if (not thresholds or len(set(thresholds)) != len(thresholds)
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in thresholds)):
        raise ValueError("thresholds must be distinct finite values in [0,1]")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        raise ValueError("Checkpoint label must contain only letters, digits, underscore or hyphen")
    modes = args.gate_modes
    if modes not in (["hard"], ["hard", "soft"]):
        raise ValueError("gate-modes must be hard, or hard soft in that order")
    paired = modes == ["hard", "soft"]
    if paired and (thresholds != [0.5] or args.control_run is None):
        raise ValueError("Paired soft requires thresholds 0.50 and a completed same-checkpoint old-code --control-run")
    if args.control_run is not None and not paired:
        raise ValueError("--control-run is reserved for the hard/soft pair")
    checkpoint_path = require_within((args.checkpoint or source / "model.json").expanduser().resolve(), ROOT / "outputs")
    model_state = load_checkpoint(checkpoint_path)
    original_state = read(source / "model.json")
    model = GenerativeCognitiveMapLearner.from_state_dict(model_state, device=config["cml_device"])
    for name in ("Q", "V", "W"):
        if not torch.equal(getattr(model, name).cpu(), torch.tensor(original_state[name], dtype=torch.float32)):
            raise ValueError("An alternate checkpoint may change G only; Q/V/W must remain fixed")
    graph = OracleObservedGraph.from_state_dict(read(source / "graph.json"))
    if model.n_states != len(graph.positions) or list(model.edges) != graph.edges:
        raise ValueError("Model and graph vocabularies differ")
    fingerprints = {name: sha256_file(source / f"{name}.json") for name in
                    ("model", "graph", "config", "supported_tasks", "teacher_subgoal_plan", "training", "source_hashes", "summary")}
    checkpoint_sha = sha256_file(checkpoint_path)
    original_checkpoint = checkpoint_sha == fingerprints["model"]
    adaptation = model_state.get("readout_adaptation")
    adaptation_source = None
    adaptation_sha = None
    if adaptation is not None:
        if adaptation.get("metadata") != "readout_training.json":
            raise ValueError("Unsupported G-only adaptation metadata reference")
        adaptation_source = checkpoint_path.parent / "readout_training.json"
        adaptation_sha = sha256_file(adaptation_source)
        if read(adaptation_source).get("model_sha256") != checkpoint_sha:
            raise ValueError("Readout training metadata model hash mismatch")
    plan = load_navigation_plan(source / "plan.json", base)
    tasks_raw = read(source / "supported_tasks.json")
    tasks = load_supported_navigation_tasks(source / "supported_tasks.json", base, plan, tasks_raw["payload"]["graph_path"])
    if fingerprints["graph"] != tasks["payload"]["graph_sha256"]:
        raise ValueError("Source graph differs from frozen supported tasks")
    teacher = read(source / "teacher_subgoal_plan.json")
    validate_teacher_subgoal_manifest(teacher, graph, tasks, source_model_sha256=fingerprints["model"],
                                     source_graph_sha256=fingerprints["graph"], source_tasks_sha256=fingerprints["supported_tasks"])
    records = teacher["payload"]["records"]
    if len(records) != 5 or any("4" not in row["schedules"] for row in records):
        raise ValueError("Expected the original five four-edge teacher schedules")
    before_sources = source_hashes()
    original_sources = read(source / "source_hashes.json")
    source_contract_path = getattr(args, "source_contract", None)
    source_contract_hash = sha256_file(source_contract_path) if source_contract_path is not None else None
    expected_current_sources = read(source_contract_path) if source_contract_path is not None else None
    if expected_current_sources is not None and before_sources != expected_current_sources:
        raise ValueError("Current package does not match the explicitly supplied source contract")
    changed_sources = {}
    allowed_changes = {"methods/gcml.py", "evaluation/oracle_graph_navigation.py"} if paired else set()
    for name in sorted(set(before_sources) | set(original_sources)):
        if before_sources.get(name) != original_sources.get(name):
            if expected_current_sources is None and name not in allowed_changes:
                raise ValueError(f"Package source changed outside the declared paired patch: {name}")
            changed_sources[name] = {"original_sha256": original_sources.get(name), "current_sha256": before_sources.get(name)}
    control = None
    control_task_dirs = {}
    control_fingerprints = {}
    if paired:
        control = require_within(args.control_run.expanduser().resolve(), ROOT / "outputs")
        if read(control / "summary.json").get("status") != "completed":
            raise ValueError("Paired hard control must be completed")
        for name in ("model", "graph", "teacher_subgoal_plan", "supported_tasks"):
            expected = checkpoint_sha if name == "model" else fingerprints[name]
            if sha256_file(control / f"{name}.json") != expected:
                raise ValueError(f"Paired control frozen artifact mismatch: {name}")
        control_sources = read(control / "source_hashes.json")
        if control_sources != original_sources:
            raise ValueError("Control must have been run under the original hard implementation")
        if sha256_file(control / "source_config.json") != fingerprints["config"]:
            raise ValueError("Paired control original source configuration bytes differ")
        if read(control / "checkpoint_source.json").get("source_run") != str(source):
            raise ValueError("Paired control provenance points to a different source run")
        control_config = read(control / "config.json")
        for name, value in source_config.items():
            if name not in {"methods", "output_dir", "source_run"} and control_config.get(name) != value:
                raise ValueError(f"Paired control source configuration mismatch: {name}")
        control_rows = [r for r in read(control / "evaluation.json")
                        if r["threshold"] == 0.5 and r.get("gate_mode", "hard") == "hard"]
        if len(control_rows) != 5 or {r["task_index"] for r in control_rows} != set(range(5)):
            raise ValueError("Paired control must contain exactly the five hard 0.50 tasks")
        control_task_dirs = {r["task_index"]: control / "evaluation" / r["condition"] / f"task_{r['task_index']:04d}"
                             for r in control_rows}
        control_fingerprints = {name: sha256_file(control / f"{name}.json") for name in
                                ("model", "graph", "teacher_subgoal_plan", "supported_tasks", "source_hashes", "config", "source_config", "checkpoint_source", "evaluation", "summary")}
    frozen_parameters = tensor_hash(model)
    preflight = inspect_environment(base)
    if not preflight["ready_for_run"]:
        raise RuntimeError("Habitat assets/dependencies are incomplete")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = ROOT / "outputs" / "gcml_gates" / run_id
    output.mkdir(parents=True, exist_ok=False)
    print(f"GCML gate comparison artifacts: {output}", flush=True)
    write_json(output / "comparison_contract.json", {"path": str(contract_path), "sha256": contract_hash, "expected": expected})
    for name in ("graph", "plan", "supported_tasks", "teacher_subgoal_plan", "training", "algorithm_reference"):
        shutil.copyfile(source / f"{name}.json", output / f"{name}.json")
    shutil.copyfile(checkpoint_path, output / "model.json")
    if adaptation_source is not None:
        shutil.copyfile(adaptation_source, output / "readout_training.json")
        write_json(output / "readout_training_source.json", {
            "source_path": str(adaptation_source), "source_sha256": adaptation_sha,
            "copied_path": "readout_training.json", "checkpoint_sha256": checkpoint_sha,
            "readout_adaptation": adaptation,
            "training_json_scope": "inherited_original_joint_training_not_G_adaptation",
            "readout_training_json_scope": "actual_G_only_adaptation_of_active_checkpoint"})
    shutil.copyfile(source / "config.json", output / "source_config.json")
    shutil.copyfile(Path(__file__), output / "experiment_script.py")
    snapshot_gcml_source(output, before_sources, runner=Path(__file__).resolve())
    for relative, digest in before_sources.items():
        package_file = ROOT / "src/cmnav" / relative
        archived = output / "source_package" / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package_file, archived)
        if sha256_file(archived) != digest:
            raise RuntimeError(f"Package source archive hash mismatch: {relative}")
    config.update(gate_protocol=PROTOCOL, checkpoint_label=args.label, source_run=str(source),
                  methods=["gcml"], output_dir=str(output), gate_thresholds=thresholds, gate_modes=modes, teacher_interval=4)
    for name, value in (("config", config), ("source_hashes", before_sources), ("runtime", runtime_versions()),
                        ("preflight", preflight), ("checkpoint_source", {
                            "source_run": str(source), "fingerprints": fingerprints,
                            "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
                            "checkpoint_label": args.label, "original_checkpoint": original_checkpoint,
                            "teacher_plan_reused_unchanged": True, "training_performed": False,
                            "parameter_tensor_sha256": frozen_parameters,
                            "declared_source_changes": changed_sources,
                            "source_contract_path": str(source_contract_path) if source_contract_path else None,
                            "source_contract_sha256": source_contract_hash,
                            "paired_control_run": str(control) if control else None,
                            "paired_control_fingerprints": control_fingerprints})):
        write_json(output / f"{name}.json", value)
    write_json(output / "protocol.json", {
        "name": PROTOCOL, "teacher_assisted": True, "formal_visual_result": False,
        "modified_setting_only": "hard_vs_soft_affordance_gate" if paired else "gcml_affordance_threshold", "model_and_graph": "frozen_reset_each_task",
        "teacher_interval": 4, "teacher_plan_recomputed": False, "teacher_input": "only_current_subgoal",
        "first_imagined_action_mask": "observed_current_outgoing", "subsequent_mask": {"hard": "Gz>=threshold", "soft": "all_action_ids"} if paired else "Gz>=threshold",
        "future_oracle_filter": False, "source_audit_feedback": False,
        "environment_or_rng_reset_within_task": False, "physical_budget": 500,
        "candidates": 64, "horizon": 96, "noise_scale": 0.2,
        "gate_modes": modes, "soft_center": .5, "soft_temperature": .1, "soft_weight": .5,
        "soft_formula": "u_l2_normalized + .5*logsigmoid((Gz-.5)/.1) + Gaussian_noise",
        "soft_depth_zero_bonus": 0., "soft_future_candidates": "all_495_actions",
        "soft_no_abstain_including_true_graph_sinks": True,
        "soft_logging": {"schema": "cmnav.gcml_soft_step_compact.v1", "top_by_total_logit": 8,
                         "all_action_scores_stored": False, "hard_logs_unchanged": True},
        "equal_per_decision_budget_not_equal_total_compute": True})
    started = monotonic()
    results, baseline, paired_checks = [], [], []
    comparison = {"status": "running", "source_run": str(source), "checkpoint_label": args.label,
                  "checkpoint_sha256": checkpoint_sha, "training_performed": False,
                  "all_parameters_unchanged": True, "baseline_replay_status": "pending" if original_checkpoint and 0.5 in thresholds else "not_applicable",
                  "paired_hard_replay_status": "pending" if paired else "not_applicable",
                  "conditions": []}
    try:
        store = SceneMemoryStore(output / "memory")
        context = {"scene_id": base["dataset"]["scene_ids"][0], "method_id": "gcml_gate_comparison",
                   "goal_protocol": "C", "data_version": teacher["manifest_id"], "model_version": checkpoint_sha,
                   "protocol_version": PROTOCOL, "privileged_world_pose": True, "formal_visual_result": False}
        snapshot = store.create_snapshot({"graph": graph.state_dict(), "model": model_state}, context)
        write_json(output / "snapshot.json", {"snapshot_id": snapshot, "context": context})
        episodes = supported_episode_list(tasks, base)
        for mode, threshold in [(mode, threshold) for mode in modes for threshold in thresholds]:
            if mode == "soft" and not (len(paired_checks) == 5 and all(row["all_match"] for row in paired_checks)):
                raise RuntimeError("Soft cannot start before all five new-code hard tasks reproduce the old-code artifact")
            condition = f"gate_{threshold:.6g}".replace(".", "p")
            if paired:
                condition = f"{mode}_{condition}"
            condition_config = copy.deepcopy(config)
            condition_config["gcml_affordance_threshold"] = threshold
            condition_config.update(gcml_gate_mode=mode, gcml_soft_temperature=.1, gcml_soft_weight=.5)
            write_json(output / "conditions" / f"{condition}.json", condition_config)
            env = OracleNavigationEnvironment(base, episodes=episodes, oracle_state_diagnostic=True,
                                              goal_radius=config["local_goal_radius"])
            try:
                write_json(output / "habitat_config.json", env.resolved_config())
                for index, record in enumerate(records):
                    task_dir = output / "evaluation" / condition / f"task_{index:04d}"
                    session = store.begin_task(snapshot, task_id=f"{condition}_{index:04d}", expected_metadata=context,
                                               map_mode="frozen", retention="reset")
                    try:
                        state = session.read()
                        task_graph = OracleObservedGraph.from_state_dict(state["graph"])
                        task_model = restore_model(state["model"], task_graph, config["cml_device"], expected_hash=frozen_parameters)
                        result = evaluate_graph_navigation_task(condition_config, env, task_graph, None, "gcml", task_dir,
                            config["seed"] + index, gcml_model=task_model,
                            teacher_subgoals=record["schedules"]["4"]["subgoal_nodes"],
                            teacher_reference_path=record["reference_edges"])
                        if (result["episode_id"] != record["source_episode_id"] or result["start_node"] != record["start_node"]
                                or result["goal_node"] != record["goal_node"] or not result["directed_graph_reachable"]):
                            raise RuntimeError("Runtime task differs from frozen task identity/coverage")
                        if (tensor_hash(task_model) != frozen_parameters or task_model.updates != model.updates
                                or task_model.epochs_completed != model.epochs_completed):
                            raise RuntimeError("Model parameters or training counters changed during evaluation")
                        result.update(threshold=threshold, gate_mode=mode, condition=condition, interval=4, task_index=index,
                                      checkpoint_label=args.label, parameters_unchanged=True,
                                      parameter_tensor_sha256=frozen_parameters, teacher_manifest_id=teacher["manifest_id"])
                        result["imagined_transition_source_audit"] = audit_imagined_sources(task_dir / "gcml_candidates.jsonl.gz", graph.edges)
                        if mode == "hard" and threshold == 0.5 and original_checkpoint:
                            check = baseline_check(result, task_dir, source / "evaluation" / "gcml_4" / f"task_{index:04d}")
                            baseline.append({"task_index": index, **check})
                            write_json(output / "baseline_replay_checks.json", baseline)
                            if not check["all_match"]:
                                raise RuntimeError("0.50 control did not reproduce source gcml_4")
                        if paired and mode == "hard":
                            check = baseline_check(result, task_dir, control_task_dirs[index])
                            paired_checks.append({"task_index": index, **check})
                            write_json(output / "paired_hard_replay_checks.json", paired_checks)
                            if not check["all_match"]:
                                raise RuntimeError("New-code hard did not reproduce the old-code same-checkpoint control")
                        result["memory_session"] = session.finish(commit=False)
                    except BaseException as exc:
                        session.abort(str(exc))
                        raise
                    write_json(task_dir / "summary.json", result)
                    results.append(result)
                    write_json(output / "evaluation.json", results)
                    print(f"{condition} {index+1}/5 episode={result['episode_id']}: success={result['metrics']['success']}, "
                          f"steps={result['steps']}, stages={result['teacher_subgoals']['completed_stages']}/"
                          f"{result['teacher_subgoals']['total_stages']}, {result['end_reason']}", flush=True)
            finally:
                env.close()
            comparison["conditions"].append(aggregate(results, threshold, mode))
            if baseline:
                comparison["baseline_replay_status"] = "all_five_match" if len(baseline) == 5 and all(row["all_match"] for row in baseline) else "partial"
            if paired_checks:
                comparison["paired_hard_replay_status"] = "all_five_match" if len(paired_checks) == 5 and all(row["all_match"] for row in paired_checks) else "partial"
            write_json(output / "comparison.json", comparison)
            print(f"Completed threshold {threshold}: {comparison['conditions'][-1]['successes']}/5 successes", flush=True)
        if sha256_file(contract_path) != contract_hash:
            raise RuntimeError("Teacher protocol contract changed during experiment")
        if source_contract_path is not None and sha256_file(source_contract_path) != source_contract_hash:
            raise RuntimeError("Explicit source contract changed during experiment")
        if source_hashes() != before_sources or sha256_file(checkpoint_path) != checkpoint_sha:
            raise RuntimeError("Package code or checkpoint changed during experiment")
        if control is not None and any(sha256_file(control / f"{name}.json") != digest for name, digest in control_fingerprints.items()):
            raise RuntimeError("Paired control artifact changed during experiment")
        if adaptation_source is not None and sha256_file(adaptation_source) != adaptation_sha:
            raise RuntimeError("Readout adaptation metadata changed during experiment")
        if any(sha256_file(source / f"{name}.json") != digest for name, digest in fingerprints.items()):
            raise RuntimeError("Source experiment artifact changed")
        comparison.update(status="completed", evaluations=len(results), elapsed_seconds=monotonic() - started,
                          all_source_fingerprints_unchanged=True, all_package_source_fingerprints_unchanged=True)
        write_json(output / "comparison.json", comparison)
        write_json(output / "summary.json", comparison)
        (output / "report.md").write_text(report_markdown(comparison), encoding="utf-8")
        return output
    except BaseException as exc:
        comparison.update(status="failed", error=str(exc), elapsed_seconds=monotonic() - started)
        write_json(output / "comparison.json", comparison)
        write_json(output / "summary.json", comparison)
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main(argv=None) -> None:
    ROOT = checkout_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Historical source defaults to the explicitly named comparison contract")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--label", default="frozen_original")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.50, 0.40, 0.45])
    parser.add_argument("--gate-modes", nargs="+", choices=["hard", "soft"], default=["hard"])
    parser.add_argument("--control-run", type=Path)
    parser.add_argument("--source-contract", type=Path, help="Exact current package hashes explicitly accepted for a source migration")
    parser.add_argument("--contract", type=Path, help="Historical teacher protocol contract")
    args = parser.parse_args(argv)
    print(f"Finished GCML gate comparison: {run(args)}", flush=True)

