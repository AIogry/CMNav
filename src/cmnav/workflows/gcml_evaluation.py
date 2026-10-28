"""Frozen full-goal GCML evaluation with explicit historical and CUDA profiles.

Both protocols restore each task independently, consume the same task manifest,
and call the existing oracle graph evaluator. They differ only in declared
comparison constraints, checkpoint format and device policy.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from time import perf_counter
import traceback
from typing import Any
import torch
from cmnav.data.supported_navigation_tasks import supported_episode_list
from cmnav.envs.diagnostics import inspect_environment
from cmnav.envs.oracle_navigation import OracleNavigationEnvironment
from cmnav.evaluation.oracle_graph_navigation import evaluate_graph_navigation_task, summarize_graph_navigation_results
from cmnav.evaluation.frozen_gcml import restore_model, require_cuda_model, training_counters, assert_frozen, load_frozen_tasks
from cmnav.memory.oracle_graph import OracleObservedGraph
from cmnav.methods.gcml import GenerativeCognitiveMapLearner
from cmnav.runtime.artifacts import sha256_file, write_json, write_json_line
from cmnav.runtime.provenance import runtime_versions, source_hashes
from cmnav.runtime.gcml_artifacts import load_json as load, parameter_hash, evaluation_input_hashes as input_hashes, load_checkpoint
from cmnav.runtime.gcml_source import snapshot_gcml_source, load_historical_contracts
from cmnav.analysis.navigation_comparison import compare_task

METHODS = ("gcml", "gcml_graph_affordance", "gcml_greedy", "dijkstra")
VARIANTS = {"raw", "minimum_change", "minimum_norm", "historical"}
COHORTS = {"checkpoint", "final", "control"}

def validate_condition(condition: dict[str, Any]) -> None:
    required = {"name", "method", "epoch", "checkpoint_variant", "planning_seed", "overrides", "cohort"}
    if not isinstance(condition, dict) or not required.issubset(condition):
        raise ValueError(f"Condition must contain {sorted(required)}")
    if not isinstance(condition["name"], str) or not condition["name"].strip():
        raise ValueError("Condition name must be nonempty")
    if condition["method"] not in METHODS or condition["checkpoint_variant"] not in VARIANTS or condition["cohort"] not in COHORTS:
        raise ValueError("Unknown condition method, checkpoint variant, or cohort")
    if type(condition["epoch"]) is not int or condition["epoch"] < 1:
        raise ValueError("Condition epoch must be a positive integer")
    if type(condition["planning_seed"]) is not int or not 0 <= condition["planning_seed"] < 2**64:
        raise ValueError("Planning seed must be an unsigned 64-bit integer")
    overrides = condition["overrides"]
    if not isinstance(overrides, dict) or any(not isinstance(key, str) or not key.startswith("gcml_") for key in overrides):
        raise ValueError("Condition overrides must be explicit gcml_* configuration fields")


def run_full_goal(source: Path, directory: Path, renderer: int, methods: list[str], control: Path | None, *, contract: Path | None = None) -> Path:
    contracts, contract_path, contract_hash = load_historical_contracts(contract)
    expected_source = contracts["full_goal_source"]
    source, directory = source.expanduser().resolve(), directory.expanduser().resolve()
    control = control.expanduser().resolve() if control is not None else None
    if renderer < -1:
        raise ValueError("renderer-device must be -1 or a nonnegative device ID")
    if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
        raise ValueError(f"methods must be distinct entries from {METHODS}")
    if directory == source or directory.is_relative_to(source):
        raise ValueError("Output must be a new directory outside the training run")
    if control is not None and (directory == control or directory.is_relative_to(control)):
        raise ValueError("Output must be outside the control run")
    directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    before: dict[str, str] = {}
    package_before: dict[str, str] = {}
    control_before: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    write_json(directory / "summary.json", {"status": "initializing", "training_performed": False,
                                            "source_run": str(source), "started_at": started_at})
    with (directory / "evaluation.log").open("x", encoding="utf-8") as log:
        def announce(message: str) -> None:
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        try:
            torch.set_num_threads(1)
            before = input_hashes(source)
            write_json(directory / "input_hashes_before.json", before)
            original_config = load(source / "config.json")
            if load(source / "summary.json").get("status") != "completed":
                raise ValueError("Training source is not complete")
            training = load(source / "training.json")
            if training.get("epochs_completed") != expected_source["cml_epochs"]:
                raise ValueError("Training checkpoint violates the declared epoch contract")
            for key, expected in expected_source.items():
                actual = original_config.get(key, "hard" if key == "gcml_gate_mode" else None)
                if actual != expected:
                    raise ValueError(f"Source violates declared evaluation contract: {key}")
            write_json(directory / "comparison_contract.json", {"path": str(contract_path), "sha256": contract_hash,
                                                                  "expected_source": expected_source})

            # Validate the original task contract first. Its schema permits only
            # nonnegative GPU IDs; the explicit -1 runtime override below is not
            # a change to the frozen sensor/task contract (which excludes device).
            original_base = original_config["base"]
            plan, tasks, original_graph_path = load_frozen_tasks(source, original_config, before["graph.json"])
            records = tasks["payload"]["evaluation"]
            if len(records) != expected_source["evaluation_episodes"] or original_config["evaluation_episodes"] != expected_source["evaluation_episodes"]:
                raise ValueError("Dimension comparison requires exactly the original five tasks")

            config = deepcopy(original_config)
            old_renderer = original_base["environment"]["gpu_device_id"]
            config.update(output_dir=str(directory), methods=methods)
            config["base"]["environment"]["gpu_device_id"] = renderer
            base = config["base"]
            write_json(directory / "config.json", config)
            write_json(directory / "source_config.json", original_config)
            write_json(directory / "configuration_changes.json", {
                "output_dir": {"source": original_config["output_dir"], "evaluation": str(directory)},
                "methods": {"source": original_config["methods"], "evaluation": methods},
                "base.environment.gpu_device_id": {"source": old_renderer, "evaluation": renderer},
                "other_configuration_changes": [],
            })
            for name in ("graph", "plan", "supported_tasks", "training", "algorithm_reference", "reference_source"):
                shutil.copyfile(source / f"{name}.json", directory / f"{name}.json")
            package_before = source_hashes()
            original_hashes = load(source / "source_hashes.json")
            differences = {name: {"source_run": original_hashes.get(name), "current": package_before.get(name)}
                           for name in sorted(set(original_hashes) | set(package_before))
                           if original_hashes.get(name) != package_before.get(name)}
            write_json(directory / "source_hashes.json", package_before)
            script_hash = sha256_file(Path(__file__).resolve())
            shutil.copyfile(Path(__file__).resolve(), directory / "evaluation_runner.py")
            snapshot_gcml_source(directory, package_before, runner=Path(__file__).resolve())
            write_json(directory / "source_comparison.json", {
                "source_run": str(source), "differences": differences,
                "historical_source_equality_required": False,
                "hard_gate_default_requested": True, "runner_sha256": script_hash,
                "note": "Current package includes explicit optional soft-gate support; this run requests the original hard path. Differences are recorded, never silently repaired.",
            })
            runtime = runtime_versions()
            runtime.update(torch_num_threads=torch.get_num_threads(),
                           environment_threads={key: os.environ.get(key) for key in
                                                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
            write_json(directory / "runtime.json", runtime)
            preflight = inspect_environment(base)
            write_json(directory / "preflight.json", preflight)
            if not preflight["ready_for_run"]:
                raise RuntimeError("Environment assets or dependencies are incomplete")

            graph_state = load(source / "graph.json")
            graph = OracleObservedGraph.from_state_dict(graph_state)
            model_state = load(source / "model.json")
            if model_state.get("readout_adaptation"):
                raise ValueError("G-refitted checkpoints are excluded from the dimension comparison")
            model = restore_model(model_state, graph, config["cml_device"])
            if model.dimension != config["cml_dimension"] or model.epochs_completed != expected_source["cml_epochs"]:
                raise ValueError("Model/configuration dimension or training budget differs")
            frozen_hash = parameter_hash(model)
            frozen_counters = training_counters(model)
            del model
            write_json(directory / "checkpoint_source.json", {
                "source_run": str(source), "model_path": str(source / "model.json"),
                "model_sha256": before["model.json"], "parameter_tensor_sha256": frozen_hash,
                "dimension": model_state["dimension"], "training_performed": False,
                "copied_model": False, "input_hashes": before,
            })
            write_json(directory / "protocol.json", {
                "name": "gcml_dimension_frozen_full_goal_comparison_v1", "training_performed": False,
                "formal_visual_result": False, "subgoals": False, "goal": "original_final_goal",
                "gate": "hard", "affordance_threshold": config["gcml_affordance_threshold"],
                "renderer_override": {"source": old_renderer, "evaluation": renderer,
                                      "applied_after_original_task_contract_validation": True},
                "model_frozen": True, "map_frozen": True, "task_rng": "seed_plus_task_index_reset_once_before_task_planning",
                "current_package_vs_training_source_differences": list(differences),
                "control_run": str(control) if control is not None else None,
                "control_mismatch_policy": "report_without_parameter_adjustment",
            })
            episodes = supported_episode_list(tasks, original_base)
            announce(f"Evaluating dimension {model_state['dimension']}, methods={methods}, renderer={renderer}")
            with (directory / "parameter_checks.jsonl").open("x", encoding="utf-8") as checks:
                for method in methods:
                    env = OracleNavigationEnvironment(base, episodes=episodes, oracle_state_diagnostic=True,
                                                      goal_radius=config["local_goal_radius"])
                    try:
                        write_json(directory / f"habitat_config_{method}.json", env.resolved_config())
                        for index, record in enumerate(records):
                            task_dir = directory / "evaluation" / method / f"task_{index:04d}"
                            task_graph = OracleObservedGraph.from_state_dict(graph_state)
                            task_model = (restore_model(model_state, task_graph, config["cml_device"], expected_hash=frozen_hash)
                                          if method.startswith("gcml") else None)
                            if task_model is not None and parameter_hash(task_model) != frozen_hash:
                                raise RuntimeError("Task model did not restore the frozen parameters")
                            # The original evaluator calls reset_planning_rng(seed)
                            # once at the beginning of each GCML task.
                            result = evaluate_graph_navigation_task(
                                config, env, task_graph, None, method, task_dir, config["seed"] + index,
                                gcml_model=task_model)
                            if (str(result["episode_id"]) != str(record["source_episode_id"])
                                    or result["start_node"] != record["coverage"]["start_node"]
                                    or result["goal_node"] != record["coverage"]["goal_node"]):
                                raise RuntimeError("Runtime task endpoints or ordering differ from the frozen manifest")
                            if not result["endpoints_supported"] or not result["directed_graph_reachable"]:
                                raise RuntimeError("A frozen supported task lost graph coverage")
                            if task_model is not None:
                                after_hash, counters = assert_frozen(task_model, frozen_hash, frozen_counters)
                                check = {"method": method, "task_index": index, "seed": config["seed"] + index,
                                         "expected_parameter_sha256": frozen_hash, "after_parameter_sha256": after_hash,
                                         "parameters_unchanged": after_hash == frozen_hash,
                                         "training_counters_unchanged": counters == frozen_counters,
                                         "planning_calls": task_model.planning_calls}
                                write_json_line(checks, check)
                                if not check["parameters_unchanged"] or not check["training_counters_unchanged"]:
                                    raise RuntimeError("Evaluation modified the frozen model")
                            result.update(condition=f"{method}_final", task_index=index,
                                          dimension=model_state["dimension"], parameters_unchanged=True)
                            write_json(task_dir / "summary.json", result)
                            results.append(result)
                            write_json(directory / "evaluation.json", results)
                            if control is not None:
                                comparison = compare_task(result, task_dir, control / "evaluation" / method / f"task_{index:04d}")
                                comparison.update(method=method, task_index=index)
                                comparisons.append(comparison)
                                control_before.update(comparison["source_hashes"])
                                write_json(directory / "control_comparisons.json", comparisons)
                            announce(f"{method} {index + 1}/{len(records)}: success={result['metrics'].get('success')}, "
                                     f"steps={result['steps']}, hops={result['graph_hops_completed']}, {result['end_reason']}")
                            del task_model
                    finally:
                        env.close()
            after = input_hashes(source)
            package_after = source_hashes()
            write_json(directory / "input_hashes_after.json", after)
            write_json(directory / "source_hashes_after.json", package_after)
            control_after = {path: sha256_file(Path(path)) for path in control_before}
            write_json(directory / "control_hashes.json", {"before": control_before, "after": control_after,
                                                           "unchanged": control_before == control_after})
            if (before != after or package_before != package_after
                    or control_before != control_after or sha256_file(Path(__file__).resolve()) != script_hash
                    or sha256_file(contract_path) != contract_hash):
                raise RuntimeError("Inputs or executing source changed during evaluation")
            write_json(directory / "summary.json", {
                "status": "completed", "training_performed": False, "formal_visual_result": False,
                "dimension": model_state["dimension"], "source_run": str(source),
                "checkpoint_source": {"model_path": str(source / "model.json"),
                                      "model_sha256": before["model.json"],
                                      "parameter_tensor_sha256": frozen_hash,
                                      "graph_sha256": before["graph.json"]},
                "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
                "graph_nodes": len(graph.positions), "graph_edges": len(graph.edges),
                "methods": summarize_graph_navigation_results(results, methods),
                "renderer_device": renderer, "source_inputs_unchanged": True,
                "package_source_unchanged": True, "parameters_unchanged": True,
                "control_comparison": {"performed": control is not None, "tasks": len(comparisons),
                                       "all_match": all(row["all_match"] for row in comparisons) if comparisons else None,
                                       "matching_tasks": sum(row["all_match"] for row in comparisons)},
                "elapsed_seconds": perf_counter() - started,
            })
            announce(f"Completed frozen dimension evaluation: {directory}")
            return directory
        except BaseException as exc:
            error = traceback.format_exc()
            (directory / "error.txt").write_text(error, encoding="utf-8")
            log.write(error)
            log.flush()
            try:
                write_json(directory / "input_hashes_after.json", input_hashes(source))
                write_json(directory / "source_hashes_after.json", source_hashes())
            except OSError:
                pass
            write_json(directory / "summary.json", {
                "status": "failed", "training_performed": False, "source_run": str(source),
                "started_at": started_at, "error": str(exc), "error_type": type(exc).__name__,
                "completed_task_count": len(results), "elapsed_seconds": perf_counter() - started,
            })
            raise


def run_job(run: Path, checkpoint: Path, condition_path: Path, device_name: str, output: Path) -> Path:
    run, checkpoint, condition_path, output = (path.expanduser().resolve() for path in
                                             (run, checkpoint, condition_path, output))
    if device_name not in {"cuda:0", "cuda:1"}:
        raise ValueError("device must explicitly be cuda:0 or cuda:1")
    if output == run or output == checkpoint.parent or checkpoint.is_relative_to(output):
        raise ValueError("Output must not contain or replace the run/checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    started, started_at = perf_counter(), datetime.now(timezone.utc).isoformat()
    before: dict[str, str] = {}
    package_before: dict[str, str] = {}
    condition: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    checkpoint_hash = None
    write_json(output / "summary.json", {"status": "initializing", "device": device_name,
                                        "training_performed": False, "checkpoint": str(checkpoint)})
    with (output / "evaluation.log").open("x", encoding="utf-8") as log:
        def announce(message: str) -> None:
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        try:
            torch.set_num_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            device = torch.device(device_name)
            if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
                raise RuntimeError(f"Required GPU {device_name} is unavailable; CPU fallback is forbidden")
            torch.cuda.set_device(device)
            torch.empty(1, device=device)
            torch.cuda.synchronize(device)

            protocol_path = run / "protocol.json"
            protocol = load(protocol_path)
            source = Path(protocol["source_run"]).expanduser().resolve()
            if output == source or output.is_relative_to(source):
                raise ValueError("Evaluation output must stay outside the historical source run")
            condition = load(condition_path)
            validate_condition(condition)
            method = condition["method"]
            checkpoint_hash = sha256_file(checkpoint)
            source_hashes_before = input_hashes(source)
            before = {str(source / name): value for name, value in source_hashes_before.items()}
            before.update({str(protocol_path): sha256_file(protocol_path), str(condition_path): sha256_file(condition_path),
                           str(checkpoint): checkpoint_hash})
            original_config = load(source / "config.json")
            if load(source / "summary.json").get("status") != "completed":
                raise ValueError("Historical task source is not a completed run")
            original_base = original_config["base"]
            plan, tasks, original_graph_path = load_frozen_tasks(source, original_config, source_hashes_before["graph.json"])
            before[str(original_graph_path)] = sha256_file(original_graph_path)
            for path, expected in plan["payload"]["fingerprints"].items():
                observed = sha256_file(Path(path))
                if observed != expected:
                    raise ValueError(f"Frozen task asset changed: {path}")
                before[str(Path(path))] = observed
            write_json(output / "input_hashes_before.json", before)
            if source_hashes_before["graph.json"] != tasks["payload"]["graph_sha256"]:
                raise ValueError("Historical source graph and supported-task graph differ")
            if tasks["payload"]["attachment_radius"] != original_config["attachment_radius"]:
                raise ValueError("Frozen task and controller attachment radii differ")
            records = tasks["payload"]["evaluation"]
            expected_tasks = protocol.get("evaluation_episodes", original_config["evaluation_episodes"])
            if len(records) != expected_tasks or original_config["evaluation_episodes"] != expected_tasks:
                raise ValueError("Condition task count differs from the frozen protocol")
            if condition["planning_seed"] + len(records) - 1 >= 2**64:
                raise ValueError("Planning seed would overflow the declared task seed sequence")

            model_state = load_checkpoint(checkpoint)
            if model_state["epochs_completed"] != condition["epoch"]:
                raise ValueError("Checkpoint epoch differs from the declared condition")
            if model_state["dimension"] != protocol["dimension"]:
                raise ValueError("Checkpoint dimension differs from the run protocol")
            if model_state["dtype"] != "float32":
                raise ValueError("The declared inference checkpoint must use float32")
            config = deepcopy(original_config)
            config.update(output_dir=str(output), methods=[method], cml_device=device_name,
                          cml_dimension=model_state["dimension"], cml_epochs=model_state["epochs_completed"])
            config["base"]["environment"]["gpu_device_id"] = device.index
            config.update(condition["overrides"])
            base = config["base"]
            write_json(output / "config.json", config)
            write_json(output / "source_config.json", original_config)
            write_json(output / "condition.json", condition)
            for name in ("graph", "plan", "supported_tasks", "algorithm_reference", "reference_source"):
                shutil.copyfile(source / f"{name}.json", output / f"{name}.json")
            package_before = source_hashes()
            write_json(output / "source_hashes.json", package_before)
            snapshot = snapshot_gcml_source(output, package_before, runner=Path(__file__).resolve())
            historical_sources = load(source / "source_hashes.json")
            changes = {name: {"historical": historical_sources.get(name), "current": package_before.get(name)}
                       for name in sorted(set(historical_sources) | set(package_before))
                       if historical_sources.get(name) != package_before.get(name)}
            runtime = runtime_versions()
            runtime.update(device=device_name, cuda_version=torch.version.cuda,
                           cuda_device_index=device.index, cuda_device_name=torch.cuda.get_device_name(device),
                           cuda_device_capability=list(torch.cuda.get_device_capability(device)),
                           torch_num_threads=torch.get_num_threads(),
                           tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                           tf32_cudnn=torch.backends.cudnn.allow_tf32,
                           float32_matmul_precision=torch.get_float32_matmul_precision(),
                           cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                           environment_threads={key: os.environ.get(key) for key in
                                                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
            write_json(output / "runtime.json", runtime)
            preflight = inspect_environment(base)
            write_json(output / "preflight.json", preflight)
            if not preflight["ready_for_run"]:
                raise RuntimeError("Habitat assets or dependencies are incomplete")

            graph_state = load(source / "graph.json")
            graph = OracleObservedGraph.from_state_dict(graph_state)
            reference_model = restore_model(model_state, graph, device, require_cuda=True)
            frozen_hash = parameter_hash(reference_model)
            frozen_counters = training_counters(reference_model)
            del reference_model
            checkpoint_metadata = {
                "path": str(checkpoint), "sha256": checkpoint_hash,
                "parameter_tensor_sha256": frozen_hash, "dimension": model_state["dimension"],
                "epoch": model_state["epochs_completed"], "variant": condition["checkpoint_variant"],
                "training_counters": frozen_counters,
                "readout_adaptation": model_state.get("readout_adaptation"),
                "copied_model": False, "training_performed": False,
            }
            write_json(output / "checkpoint_source.json", checkpoint_metadata)
            write_json(output / "provenance.json", {
                "run": str(run), "run_protocol_path": str(protocol_path), "source_run": str(source),
                "historical_5000": protocol.get("historical_5000"),
                "condition_path": str(condition_path), "condition": condition,
                "checkpoint": checkpoint_metadata, "device": device_name,
                "renderer_device_index": device.index, "cpu_inference_fallback": False,
                "subgoals": False, "formal_visual_result": False,
                "physical_environment_seed": base["seed"],
                "planning_task_seeds": [condition["planning_seed"] + index for index in range(len(records))],
                "parameter_memory_hash_cpu_copy_is_audit_only": True,
                "changes_from_historical_config": {
                    "output_dir": str(output), "methods": [method], "cml_device": device_name,
                    "cml_dimension": model_state["dimension"], "cml_epochs": model_state["epochs_completed"],
                    "base.environment.gpu_device_id": device.index, **condition["overrides"],
                },
                "historical_package_source_differences": changes,
                "source_snapshot": snapshot["archive"],
                "core_evaluator": "cmnav.evaluation.oracle_graph_navigation.evaluate_graph_navigation_task",
                "illegal_current_G_action": "independent_evaluator_failure_without_oracle_repair",
            })
            episodes = supported_episode_list(tasks, original_base)
            write_json(output / "summary.json", {
                "status": "running", "condition": condition, "checkpoint": checkpoint_metadata,
                "device": device_name, "runtime": runtime, "started_at": started_at,
                "training_performed": False,
            })
            announce(f"Condition {condition['name']}, epoch={condition['epoch']}, "
                     f"variant={condition['checkpoint_variant']}, method={method}, device={device_name}")
            env = OracleNavigationEnvironment(base, episodes=episodes, oracle_state_diagnostic=True,
                                              goal_radius=config["local_goal_radius"])
            try:
                write_json(output / "habitat_config.json", env.resolved_config())
                with (output / "parameter_checks.jsonl").open("x", encoding="utf-8") as checks:
                    for index, record in enumerate(records):
                        task_model = (restore_model(model_state, graph, device, expected_hash=frozen_hash, require_cuda=True)
                                      if method.startswith("gcml") else None)
                        if task_model is not None:
                            require_cuda_model(task_model, device)
                            if parameter_hash(task_model) != frozen_hash:
                                raise RuntimeError("Task failed to restore the frozen model")
                        task_graph = OracleObservedGraph.from_state_dict(graph_state)
                        task_dir = output / "evaluation" / method / f"task_{index:04d}"
                        task_seed = condition["planning_seed"] + index
                        result = evaluate_graph_navigation_task(config, env, task_graph, None, method,
                                                                task_dir, task_seed, gcml_model=task_model)
                        if (str(result["episode_id"]) != str(record["source_episode_id"])
                                or result["start_node"] != record["coverage"]["start_node"]
                                or result["goal_node"] != record["coverage"]["goal_node"]
                                or not result["endpoints_supported"] or not result["directed_graph_reachable"]):
                            raise RuntimeError("Actual task ordering/coverage differs from the frozen manifest")
                        if task_model is not None:
                            require_cuda_model(task_model, device)
                            after_hash, counters = assert_frozen(task_model, frozen_hash, frozen_counters, cuda_device=device)
                            unchanged = after_hash == frozen_hash and counters == frozen_counters
                            check = {"task_index": index, "method": method, "device": str(task_model.device),
                                     "task_planning_seed": task_seed, "before_parameter_sha256": frozen_hash,
                                     "after_parameter_sha256": after_hash, "parameters_unchanged": after_hash == frozen_hash,
                                     "training_counters_unchanged": counters == frozen_counters,
                                     "planning_calls": task_model.planning_calls}
                            write_json_line(checks, check)
                            if not unchanged:
                                raise RuntimeError("Inference changed frozen parameters or training counters")
                        result.update(task_index=index, condition=condition["name"], epoch=condition["epoch"],
                                      checkpoint_variant=condition["checkpoint_variant"], cohort=condition["cohort"],
                                      planning_seed=condition["planning_seed"], task_planning_seed=task_seed,
                                      device=device_name, parameters_unchanged=True)
                        write_json(task_dir / "summary.json", result)
                        results.append(result)
                        write_json(output / "evaluation.json", results)
                        announce(f"{condition['name']} {index + 1}/{len(records)}: success={result['metrics'].get('success')}, "
                                 f"steps={result['steps']}, hops={result['graph_hops_completed']}, {result['end_reason']}")
                        del task_model
            finally:
                env.close()
            torch.cuda.synchronize(device)
            after = {path: sha256_file(Path(path)) for path in before}
            package_after = source_hashes()
            scripts_unchanged = all(sha256_file(Path(path)) == value for path, value in snapshot["script_hashes"].items())
            write_json(output / "input_hashes_after.json", after)
            write_json(output / "source_hashes_after.json", package_after)
            if before != after or package_before != package_after or not scripts_unchanged:
                raise RuntimeError("Input artifacts or running source changed during the job")
            aggregate = summarize_graph_navigation_results(results, [method])
            write_json(output / "summary.json", {
                "status": "completed", "condition": condition, "checkpoint": checkpoint_metadata,
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                "device": device_name, "runtime": runtime,
                "epoch": condition["epoch"], "checkpoint_variant": condition["checkpoint_variant"],
                "cohort": condition["cohort"], "planning_seed": condition["planning_seed"],
                "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
                "methods": aggregate, "metrics": aggregate[method],
                "graph_nodes": len(graph.positions), "graph_edges": len(graph.edges),
                "tasks": len(results), "parameters_unchanged": True, "training_counters_unchanged": True,
                "source_inputs_unchanged": True, "package_source_unchanged": True, "runner_source_unchanged": True,
                "training_performed": False, "formal_visual_result": False, "cpu_inference_fallback": False,
                "elapsed_seconds": perf_counter() - started,
            })
            announce(f"Completed {condition['name']}: {output}")
            return output
        except BaseException as exc:
            error = traceback.format_exc()
            (output / "error.txt").write_text(error, encoding="utf-8")
            log.write(error)
            log.flush()
            integrity = None
            try:
                after = {path: sha256_file(Path(path)) for path in before}
                write_json(output / "input_hashes_after.json", after)
                write_json(output / "source_hashes_after.json", source_hashes())
                integrity = before == after if before else None
            except OSError:
                pass
            write_json(output / "summary.json", {
                "status": "failed", "condition": condition, "device": device_name,
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                "started_at": started_at, "error": str(exc), "error_type": type(exc).__name__,
                "completed_tasks": len(results), "source_inputs_unchanged": integrity,
                "training_performed": False, "cpu_inference_fallback": False,
                "elapsed_seconds": perf_counter() - started,
            })
            raise


def full_goal_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--renderer-device", type=int, default=-1)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS[:3]))
    parser.add_argument("--control-run", type=Path)
    parser.add_argument("--contract", type=Path, help="Explicit frozen full-goal source comparison contract")
    args = parser.parse_args(argv)
    run_full_goal(args.training_run, args.output, args.renderer_device, args.methods, args.control_run, contract=args.contract)


def condition_main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate one declared GCML condition with required CUDA inference")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local .pt state dictionary or historical model.json")
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_job(args.run, args.checkpoint, args.condition, args.device, args.output)
