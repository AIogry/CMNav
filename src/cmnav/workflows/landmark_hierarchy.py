"""Build, learn and evaluate experience-derived landmark maps in separate runs.

This workflow consumes saved abstract-graph experience. It never collects RGB-D
or invokes Habitat. Task endpoints are read only after the hierarchy is frozen.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter
import traceback
from typing import Any
from uuid import uuid4

from cmnav.runtime.artifacts import canonical_json_bytes, require_within, sha256_file, write_json
from cmnav.runtime.project import checkout_root
from cmnav.runtime.provenance import runtime_versions, source_hashes
from cmnav.runtime.source_snapshot import archive_package_sources


def _read(path: Path, inputs: dict[str, str]) -> Any:
    path = path.expanduser().resolve()
    content = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def _settings(path: Path, inputs: dict[str, str]) -> dict:
    value = _read(path, inputs)
    if value.get("schema") != "cmnav.landmark_hierarchy_protocol.v1":
        raise ValueError("Unknown landmark experiment protocol")
    if value.get("scope") != "frozen_abstract_graph_development":
        raise ValueError("This workflow only supports frozen abstract graph development")
    required = {"schema", "scope", "structure", "training", "planning", "evaluation", "permissions", "execution"}
    if not required <= set(value) or set(value) - required - {"interpretation", "budget_interpretation"}:
        raise ValueError("Missing or unknown hierarchy protocol sections")
    permissions = {
        "node_identity": "given_abstract_graph_ids",
        "first_step_mask": "known_outgoing_at_query_start_including_hypothetical_endpoint_queries",
        "future_imagination": "learned_G_no_true_graph_filtering",
        "readout_labels": "all_observed_nodes_including_sinks_at_both_levels",
        "high_dijkstra": "observed_macro_graph_baseline_only",
        "task_endpoints_used_for_structure": False,
        "future_graph_audit_used_for_selection": False,
    }
    execution = {
        "unit": "directed_low_graph_edge", "rule": "execute_selected_local_plan_until_target_or_failure",
        "intermediate_other_landmarks": "continue_to_selected_target",
        "macro_completion": "actual_node_equals_selected_target", "recovery_retries": 0,
        "model_parameters_frozen": True, "habitat_execution": False,
    }
    if value["permissions"] != permissions or value["execution"] != execution:
        raise ValueError("Declared permissions/execution differ from the implemented protocol")
    evaluation = value["evaluation"]
    seeds, methods = evaluation["planning_seeds"], evaluation["methods"]
    if (not isinstance(seeds, list) or not seeds or
            any(type(seed) is not int or not 0 <= seed < 2**64 for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("Planning seeds must be unique nonnegative 64-bit integers")
    allowed = {"dijkstra", "flat_gcml", "hierarchical_dijkstra", "hierarchical_gcml"}
    if not isinstance(methods, list) or not methods or any(method not in allowed for method in methods) or len(set(methods)) != len(methods):
        raise ValueError("Methods must be a unique list of supported methods")
    return value


@contextmanager
def _run(output_root: Path, phase: str, inputs: dict[str, str], config: dict):
    output_root = require_within(output_root.expanduser(), checkout_root() / "outputs")
    identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    directory = output_root / identity
    directory.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    state = {"schema": "cmnav.landmark_hierarchy_run.v1", "phase": phase,
             "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
             "inputs": inputs, "configuration": config, "package_source_hashes": before,
             "environment_interactions": 0, "visual_navigation": False}
    started = perf_counter()
    print(f"Landmark {phase}: {directory}", flush=True)
    write_json(directory / "manifest.json", state)
    try:
        write_json(directory / "protocol.json", config)
        write_json(directory / "runtime.json", runtime_versions())
        archive_package_sources(directory / "source")
        yield directory, state
        if before != source_hashes():
            raise RuntimeError("Package implementation changed during this run")
        if any(sha256_file(Path(path)) != digest for path, digest in inputs.items()):
            raise RuntimeError("A frozen input changed during this run")
        state.update(status="completed", inputs_unchanged=True, sources_unchanged=True)
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error=f"{type(exc).__name__}: {exc}")
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        state.update(finished_at=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=perf_counter() - started)
        write_json(directory / "manifest.json", state)


def _tasks(document: dict, graph_sha: str, n_states: int) -> list[dict]:
    payload = document["payload"]
    if document.get("manifest_id") != hashlib.sha256(canonical_json_bytes(payload)).hexdigest():
        raise ValueError("Task manifest content identity changed")
    if payload.get("graph_sha256") != graph_sha:
        raise ValueError("Tasks belong to a different graph")
    rows = []
    for index, task in enumerate(payload["evaluation"]):
        coverage = task["coverage"]
        start, goal = coverage["start_node"], coverage["goal_node"]
        if any(type(node) is not int or not 0 <= node < n_states for node in (start, goal)):
            raise ValueError("Task endpoints outside graph")
        rows.append({"task_index": index, "episode_id": task["source_episode_id"],
                     "start_node": start, "goal_node": goal})
    if not rows or len(rows) != payload["evaluation_episodes"]:
        raise ValueError("Task count mismatch")
    if len({row["episode_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate task identities")
    return rows


def build_structure(source: Path, settings_path: Path, output_root: Path) -> Path:
    from cmnav.data.landmark_hierarchy import build_landmark_hierarchy
    from cmnav.analysis.landmark_hierarchy import analyze_hierarchy
    from cmnav.reporting.landmark_hierarchy import write_structure_report

    source = source.expanduser().resolve()
    inputs: dict[str, str] = {}
    settings = _settings(settings_path, inputs)
    graph = _read(source / "graph.json", inputs)
    walks = _read(source / "training_walks.json", inputs)
    training = _read(source / "training.json", inputs)
    graph_sha, walks_sha = inputs[str(source / "graph.json")], inputs[str(source / "training_walks.json")]
    if walks["source"]["graph_sha256"] != graph_sha or training["walks_sha256"] != walks_sha:
        raise ValueError("Training experience does not match its source graph/provenance")
    with _run(output_root, "structure", inputs, settings) as (directory, manifest):
        hierarchy = build_landmark_hierarchy(
            walks, n_states=len(graph["positions"]), edges=graph["edges"],
            source_sha256=walks_sha, **settings["structure"])
        write_json(directory / "hierarchy.json", hierarchy)
        frozen_sha = sha256_file(directory / "hierarchy.json")
        # No evaluation file has been read before the hierarchy is persisted.
        tasks = _tasks(_read(source / "supported_tasks.json", inputs), graph_sha, hierarchy["n_states"])
        audit = analyze_hierarchy(hierarchy, graph["edges"], tasks)
        write_json(directory / "graph.json", graph)
        write_json(directory / "tasks.json", tasks)
        write_json(directory / "structure_audit.json", audit)
        write_structure_report(directory, hierarchy, audit, graph_positions=graph["positions"])
        if sha256_file(directory / "hierarchy.json") != frozen_sha:
            raise RuntimeError("The structure changed after reading task endpoints")
        manifest.update(source_run=str(source), hierarchy_sha256=frozen_sha,
                        graph_sha256=graph_sha, source_walks_sha256=walks_sha,
                        task_goals_used_for_structure=False,
                        artifacts={name: sha256_file(directory / name) for name in
                                   ("hierarchy.json", "graph.json", "tasks.json", "structure_audit.json")})
        print(json.dumps(hierarchy["counters"], ensure_ascii=False), flush=True)
    return directory


def _structure(directory: Path, inputs: dict[str, str]) -> tuple[dict, dict, list, dict]:
    directory = directory.expanduser().resolve()
    manifest = _read(directory / "manifest.json", inputs)
    if manifest.get("phase") != "structure" or manifest.get("status") != "completed":
        raise ValueError("A completed frozen hierarchy structure is required")
    values = {}
    for name in ("hierarchy.json", "graph.json", "tasks.json"):
        values[name] = _read(directory / name, inputs)
        if inputs[str(directory / name)] != manifest["artifacts"][name]:
            raise ValueError(f"Hierarchy artifact changed: {name}")
    return values["hierarchy.json"], values["graph.json"], values["tasks.json"], manifest


def _cuda(device_name: str) -> dict:
    import torch
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available() or device.index is None:
        raise RuntimeError("Explicit CUDA device is required; CPU fallback is forbidden")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {"device": str(device), "name": torch.cuda.get_device_name(device),
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "dtype": "float32", "tf32": False, "cpu_fallback": False}


def train_high(structure: Path, settings_path: Path, reference_path: Path, output_root: Path) -> Path:
    import torch
    from cmnav.data.landmark_hierarchy import balanced_macro_replay
    from cmnav.methods.gcml import GenerativeCognitiveMapLearner
    from cmnav.methods.linear_readout import fit_frozen_availability
    from cmnav.runtime.gcml_artifacts import tensor_state, parameter_hash

    inputs: dict[str, str] = {}
    settings = _settings(settings_path, inputs)
    hierarchy, _, _, structure_manifest = _structure(structure, inputs)
    reference = _read(reference_path, inputs)
    cfg = settings["training"]
    if len(hierarchy["landmark_nodes"]) < 2 or not hierarchy["edges"]:
        raise ValueError("The frozen structure has no trainable high-level transitions")
    if settings["structure"] != {k: hierarchy["config"][k] for k in settings["structure"]}:
        raise ValueError("Training protocol differs from the frozen hierarchy")
    with _run(output_root, "training", inputs, settings) as (directory, manifest):
        write_json(directory / "device.json", _cuda(cfg["device"]))
        replay = balanced_macro_replay(hierarchy, visits_per_edge=cfg["visits_per_edge"], seed=cfg["seed"])
        write_json(directory / "training_samples.json", replay)
        model = GenerativeCognitiveMapLearner(
            len(hierarchy["landmark_nodes"]), hierarchy["edges"], dimension=cfg["dimension"],
            seed=cfg["seed"], device=cfg["device"], reference_metadata=reference,
            **cfg["learning"])
        with (directory / "epochs.jsonl").open("x", encoding="utf-8") as log:
            def progress(row):
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                if row["epoch"] % 20 == 0:
                    print(json.dumps({"epoch": row["epoch"], "prediction_mse": row["prediction_mse"],
                                      "inverse_mse": row["inverse_mse"], "elapsed": row["elapsed_seconds"]}), flush=True)
            training = model.fit([row["transitions"] for row in replay["trajectories"]],
                                 epochs=cfg["epochs"], progress_callback=progress)
        write_json(directory / "training.json", training)
        torch.save(tensor_state(model), directory / "model_raw.pt")
        labels = torch.zeros((model.n_edges, model.n_states), device=model.device)
        for action, (source, _) in enumerate(model.edges):
            labels[action, source] = 1.
        variants, readout = fit_frozen_availability(model.Q, model.G, labels, rcond=cfg["readout_rcond"])
        write_json(directory / "readout.json", readout)
        model.G.copy_(variants["minimum_change"])
        torch.save(tensor_state(model), directory / "model.pt")
        write_json(directory / "diagnostics.json", model.training_diagnostics())
        manifest.update(structure_run=str(structure.resolve()),
                        hierarchy_sha256=structure_manifest["hierarchy_sha256"],
                        source_walks_sha256=structure_manifest["source_walks_sha256"],
                        model_sha256=sha256_file(directory / "model.pt"), parameter_hash=parameter_hash(model),
                        training_samples_sha256=sha256_file(directory / "training_samples.json"),
                        low_model_trained=False, model_selection="fixed_final_epoch_plus_minimum_change_G",
                        readout_supervision="all_observed_macro_nodes_including_zero_outdegree",
                        joint_training=training["final_diagnostics"],
                        fitted_readout=readout["variants"]["minimum_change"])
    return directory


def evaluate_hierarchy(structure: Path, high_training: Path, low_checkpoint: Path,
                       low_training: Path, settings_path: Path, output_root: Path) -> Path:
    from cmnav.methods.gcml import GenerativeCognitiveMapLearner
    from cmnav.runtime.gcml_artifacts import load_checkpoint, parameter_hash
    from cmnav.evaluation.hierarchical_gcml import run_graph_task
    from cmnav.reporting.landmark_hierarchy import write_experiment_report

    inputs: dict[str, str] = {}
    settings = _settings(settings_path, inputs)
    hierarchy, graph, tasks, structure_manifest = _structure(structure, inputs)
    high_training, low_checkpoint, low_training = (p.expanduser().resolve() for p in
                                                  (high_training, low_checkpoint, low_training))
    high_manifest = _read(high_training / "manifest.json", inputs)
    if high_manifest.get("status") != "completed" or high_manifest.get("phase") != "training":
        raise ValueError("High-level training has not completed")
    if high_manifest["hierarchy_sha256"] != structure_manifest["hierarchy_sha256"]:
        raise ValueError("High model belongs to another hierarchy")
    if settings["structure"] != {k: hierarchy["config"][k] for k in settings["structure"]}:
        raise ValueError("Evaluation protocol differs from the frozen hierarchy")
    if settings["training"] != high_manifest["configuration"]["training"]:
        raise ValueError("Evaluation protocol misstates the high checkpoint training conditions")
    low_sources = _read(low_training / "input_hashes_before.json", inputs)
    if (low_sources["graph.json"] != structure_manifest["graph_sha256"] or
            low_sources["training_walks.json"] != structure_manifest["source_walks_sha256"]):
        raise ValueError("Low model did not train on this graph replay corpus")
    low_manifest = _read(low_checkpoint.parent / "manifest.json", inputs)
    if low_manifest.get("status") != "completed":
        raise ValueError("Low checkpoint has not completed")
    if not low_checkpoint.is_relative_to(low_training / "checkpoints"):
        raise ValueError("Low checkpoint must belong to the declared training run")
    low_sha = sha256_file(low_checkpoint)
    if low_sha != low_manifest["checkpoint_hashes"]["minimum_change"]:
        raise ValueError("This protocol requires the declared minimum-change fitted low G checkpoint")
    inputs[str(low_checkpoint)] = low_sha
    high_path = high_training / "model.pt"
    inputs[str(high_path)] = sha256_file(high_path)
    if inputs[str(high_path)] != high_manifest["model_sha256"]:
        raise ValueError("High checkpoint changed")
    with _run(output_root, "evaluation", inputs, settings) as (directory, manifest):
        device = settings["evaluation"]["device"]
        write_json(directory / "device.json", _cuda(device))
        low = GenerativeCognitiveMapLearner.from_state_dict(load_checkpoint(low_checkpoint), device=device)
        high = GenerativeCognitiveMapLearner.from_state_dict(load_checkpoint(high_path), device=device)
        if (low.n_states != hierarchy["n_states"] or list(map(list, low.edges)) != graph["edges"] or
                high.n_states != len(hierarchy["landmark_nodes"]) or list(map(list, high.edges)) != hierarchy["edges"]):
            raise ValueError("Checkpoint vocabularies do not match the frozen maps")
        frozen = {"low": parameter_hash(low), "high": parameter_hash(high)}
        write_json(directory / "frozen_parameters.json", frozen)
        rows = []
        for method in settings["evaluation"]["methods"]:
            seeds = settings["evaluation"]["planning_seeds"] if method != "dijkstra" else settings["evaluation"]["planning_seeds"][:1]
            for seed in seeds:
                for task in tasks:
                    result = run_graph_task(method=method, low_model=low, high_model=high,
                                            hierarchy=hierarchy, start=task["start_node"], goal=task["goal_node"],
                                            low_edges=graph["edges"], seed=seed,
                                            config=settings["planning"])
                    result.update(task_index=task["task_index"], episode_id=task["episode_id"], planning_seed=seed)
                    task_dir = directory / "evaluation" / method / f"seed_{seed}" / f"task_{task['task_index']:04d}"
                    task_dir.mkdir(parents=True, exist_ok=False)
                    with gzip.open(task_dir / "trace.json.gz", "wt", encoding="utf-8") as stream:
                        json.dump(result, stream, ensure_ascii=False, allow_nan=False)
                    row = {k: v for k, v in result.items() if k not in ("queries", "query_records")}
                    row["trace_path"] = str(task_dir / "trace.json.gz")
                    write_json(task_dir / "summary.json", row)
                    rows.append(row)
                    print(json.dumps({k: v for k, v in row.items() if k in
                                      ("method", "task_index", "planning_seed", "success", "reason", "graph_steps", "failure_phase")}), flush=True)
                    write_json(directory / "results.json", rows)
        if frozen != {"low": parameter_hash(low), "high": parameter_hash(high)}:
            raise RuntimeError("An evaluation modified learned parameters")
        summary = {"protocol": settings, "rows": rows, "training": high_manifest,
                   "source_structure": str(structure.resolve()), "low_checkpoint": str(low_checkpoint),
                   "high_checkpoint": str(high_path), "parameters_frozen": True,
                   "task_count": len(tasks), "task_claim": "same_five_reused_development_tasks",
                   "success_unit": "exact_goal_node_in_abstract_graph_not_Habitat_success"}
        write_json(directory / "summary.json", summary)
        write_experiment_report(directory, summary, graph_positions=graph["positions"],
                                graph_edges=graph["edges"], hierarchy=hierarchy)
        manifest.update(structure_run=str(structure.resolve()), high_training_run=str(high_training),
                        low_checkpoint=str(low_checkpoint), frozen_parameters=frozen,
                        tasks=len(tasks), executions=len(rows), no_automatic_recovery=True,
                        simulator_used=False, hierarchy_sha256=structure_manifest["hierarchy_sha256"])
    return directory


def report_saved_evaluation(source: Path, output_root: Path) -> Path:
    """Render a complete saved cohort without rerunning or repairing decisions."""
    from cmnav.reporting.landmark_hierarchy import write_experiment_report

    source = source.expanduser().resolve()
    inputs: dict[str, str] = {}
    original = _read(source / "manifest.json", inputs)
    if original.get("phase") != "evaluation" or original.get("status") not in {"completed", "failed"}:
        raise ValueError("Reporting requires a stopped evaluation with saved complete results")
    summary = _read(source / "summary.json", inputs)
    if not summary.get("parameters_frozen"):
        raise ValueError("Evaluation did not confirm frozen parameters")
    for path, digest in original["inputs"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"Original evaluation input changed: {path}")
        inputs[path] = digest
    hierarchy, graph, tasks, structure_manifest = _structure(Path(summary["source_structure"]), inputs)
    if (summary["protocol"] != original["configuration"] or
            summary["training"]["hierarchy_sha256"] != structure_manifest["hierarchy_sha256"]):
        raise ValueError("Saved results disagree with their source protocol or hierarchy")
    evaluation = summary["protocol"]["evaluation"]
    expected = {(method, seed, task["task_index"]) for method in evaluation["methods"]
                for seed in (evaluation["planning_seeds"][:1] if method == "dijkstra" else evaluation["planning_seeds"])
                for task in tasks}
    rows = summary["rows"]
    actual = {(row["method"], row["seed"], row["task_index"]) for row in rows}
    if len(rows) != len(expected) or actual != expected or summary["task_count"] != len(tasks):
        raise ValueError("Saved evaluation is not the complete configured cohort")
    normalized = []
    for row in rows:
        path = source / "evaluation" / row["method"] / f"seed_{row['seed']}" / f"task_{row['task_index']:04d}" / "trace.json.gz"
        content = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(content).hexdigest()
        full = json.loads(gzip.decompress(content))
        # The first saved schema used trace for the path in its thin summary;
        # the complete compressed record always retained the actual step list.
        for key, value in row.items():
            if key in {"trace_path"} or (key == "trace" and isinstance(value, str)):
                if Path(value).resolve() != path.resolve():
                    raise ValueError("Saved trace reference disagrees with task identity")
                continue
            if full.get(key) != value:
                raise ValueError(f"Summary disagrees with complete trace: {path}, {key}")
        normalized.append({**{key: value for key, value in row.items() if key != "trace"},
                           "trace": full["trace"], "trace_path": str(path)})
    summary = {**summary, "rows": normalized, "source_evaluation": str(source),
               "source_evaluation_status": original["status"], "report_only": True,
               "source_error": original.get("error"), "no_policy_rerun": True}
    with _run(output_root, "report", inputs, summary["protocol"]) as (directory, manifest):
        write_json(directory / "summary.json", summary)
        write_experiment_report(directory, summary, graph_positions=graph["positions"],
                                graph_edges=graph["edges"], hierarchy=hierarchy)
        manifest.update(source_evaluation=str(source), source_status=original["status"],
                        source_error=original.get("error"), report_only=True, no_policy_rerun=True,
                        complete_cohort=True, executions=len(rows), parameters_frozen=True)
    return directory


def compare_noise_conditions(study_path: Path, output_root: Path) -> Path:
    """Compare frozen-model noise conditions with explicit per-method reuse."""
    from cmnav.analysis.hierarchy_noise import summarize_noise_conditions
    from cmnav.reporting.hierarchy_noise import write_noise_comparison_report
    from cmnav.evaluation.hierarchical_gcml import resolved_noise_scales

    inputs: dict[str, str] = {}
    study = _read(study_path, inputs)
    if study.get("schema") != "cmnav.hierarchy_noise_comparison.v1":
        raise ValueError("Unknown hierarchy noise comparison schema")
    conditions, cache = [], {}
    signature = None
    for condition in study["conditions"]:
        rows, source_paths = [], {}
        for method, source_value in condition["sources"].items():
            source = Path(source_value).expanduser().resolve()
            if str(source) not in cache:
                manifest = _read(source / "manifest.json", inputs)
                summary = _read(source / "summary.json", inputs)
                if (manifest.get("phase") != "evaluation" or manifest.get("status") != "completed"
                        or not manifest.get("inputs_unchanged") or not manifest.get("sources_unchanged")
                        or not summary.get("parameters_frozen")):
                    raise ValueError("Noise comparison requires complete frozen-model evaluations")
                protocol = json.loads(json.dumps(summary["protocol"]))
                for key in ("noise_scale", "local_noise_scale", "high_noise_scale", "flat_noise_scale"):
                    protocol["planning"].pop(key, None)
                protocol["evaluation"].pop("methods")
                protocol["evaluation"].pop("device")
                identity = {"protocol": protocol, "structure": summary["source_structure"],
                            "parameters": manifest["frozen_parameters"],
                            "package_sources": manifest["package_source_hashes"],
                            "tasks": sorted({(r["task_index"], r["episode_id"], r["start"], r["goal"])
                                             for r in summary["rows"]})}
                if signature is None:
                    signature = identity
                elif identity != signature:
                    raise ValueError("Conditions differ in non-noise protocol, models, source, or tasks")
                cache[str(source)] = (summary, manifest)
            summary, _ = cache[str(source)]
            actual_noise = resolved_noise_scales(summary["protocol"]["planning"])
            expected_noise = {"local": condition["local_noise"],
                              "high": condition["long_noise"], "flat": condition["long_noise"]}
            relevant = {"dijkstra": (), "flat_gcml": ("flat",),
                        "hierarchical_dijkstra": ("local", "flat"),
                        "hierarchical_gcml": ("local", "flat", "high")}
            if method not in relevant or any(actual_noise[key] != expected_noise[key] for key in relevant[method]):
                raise ValueError(f"Incorrect noise condition or reused method: {condition['name']} / {method}")
            selected = [row for row in summary["rows"] if row["method"] == method]
            if not selected:
                raise ValueError(f"Missing method in comparison source: {method}")
            rows.extend(selected)
            source_paths[method] = str(source)
        conditions.append({**condition, "rows": rows, "sources": source_paths})
    with _run(output_root, "noise_comparison", inputs, study) as (directory, manifest):
        analysis = summarize_noise_conditions(conditions)
        write_json(directory / "summary.json", analysis)
        write_noise_comparison_report(directory, analysis)
        manifest.update(source_runs=sorted(cache), checkpoint_parameters=signature["parameters"],
                        no_policy_rerun=True, per_method_reuse_explicit=True)
    return directory


def build_main(argv=None):
    parser = argparse.ArgumentParser(description="Derive and audit a goal-independent landmark hierarchy")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(build_structure(args.source_run, args.settings, args.output_root))


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Train a high GCML on witnessed macro replay using CUDA")
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(train_high(args.structure, args.settings, args.reference, args.output_root))


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate frozen flat/hierarchical models on abstract graph endpoints")
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--high-training", type=Path, required=True)
    parser.add_argument("--low-checkpoint", type=Path, required=True)
    parser.add_argument("--low-training", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(evaluate_hierarchy(args.structure, args.high_training, args.low_checkpoint,
                             args.low_training, args.settings, args.output_root))


def report_main(argv=None):
    parser = argparse.ArgumentParser(description="Render a complete saved hierarchy cohort without rerunning policies")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(report_saved_evaluation(args.source_run, args.output_root))


def compare_noise_main(argv=None):
    parser = argparse.ArgumentParser(description="Compare declared noise conditions from frozen hierarchy evaluations")
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(compare_noise_conditions(args.study, args.output_root))
