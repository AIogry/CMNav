"""Fit G only with a frozen saved GCML representation; no navigation.

Capacity references use float64 SVD. Iterative experiments retain the
original trajectory batches, float32 delta rule, and effective-step cap.
Paired sink-supervised variants add one explicitly counted zero-label batch
per epoch. No package source or original checkpoint is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from cmnav.methods.gcml import GenerativeCognitiveMapLearner
from cmnav.runtime.artifacts import sha256_file, write_json, write_json_line
from cmnav.runtime.provenance import runtime_versions, source_hashes


from cmnav.analysis.readout import grouped, readout_norms
from cmnav.methods.linear_readout import weighted_numpy_reference as svd_reference
from cmnav.runtime.source_snapshot import archive_package_sources


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def save_checkpoint(directory: Path, source: dict[str, Any], g: torch.Tensor,
                    metadata: dict[str, Any], frozen: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    state = dict(source)
    state["G"] = g.cpu().tolist()
    # The original joint-training counters/hyperparameters remain historical.
    # G-only exposure/update counts are recorded separately and never added
    # to the counters that imply Q/V/W were trained again.
    state["readout_adaptation"] = {
        "scope": "G_only_frozen_Q_V_W", "metadata": "readout_training.json",
        "variant": metadata["variant"], "joint_training_counters_are_inherited": True,
    }
    loaded = GenerativeCognitiveMapLearner.from_state_dict(state, device="cpu")
    observed = {name: tensor_hash(getattr(loaded, name)) for name in ("Q", "V", "W")}
    if observed != frozen or not torch.equal(loaded.G, g):
        raise RuntimeError("Reloaded checkpoint did not preserve frozen tensors or G")
    metadata = {**metadata, "frozen_tensor_hashes": observed,
                "G_tensor_sha256": tensor_hash(g), "existing_loader_roundtrip_verified": True,
                "joint_training_counters_unchanged": all(state[key] == source[key] for key in
                    ("updates", "trajectory_updates", "epochs_completed", "planning_calls"))}
    write_json(directory / "model.json", state)
    metadata["model_sha256"] = sha256_file(directory / "model.json")
    write_json(directory / "readout_training.json", metadata)


def _fit_readout(args: argparse.Namespace) -> Path:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    source_dir = args.source_run.expanduser().resolve()
    checkpoint_source = load(source_dir / "checkpoint_source.json")
    original_dir = Path(checkpoint_source["source_run"])
    input_paths = {
        "model": source_dir / "model.json", "graph": source_dir / "graph.json",
        "checkpoint_source": source_dir / "checkpoint_source.json",
        "original_training": original_dir / "training.json",
        "walks": original_dir / "training_walks.json",
    }
    fingerprints = {name: sha256_file(path) for name, path in input_paths.items()}
    original_training = load(input_paths["original_training"])
    if fingerprints["walks"] != original_training["walks_sha256"]:
        raise ValueError("Original trajectory checksum does not match training provenance")
    if fingerprints["model"] != checkpoint_source["fingerprints"]["model"]:
        raise ValueError("Source checkpoint fingerprint changed")
    model, graph, walks = (load(input_paths[name]) for name in ("model", "graph", "walks"))
    if model["edges"] != graph["edges"]:
        raise ValueError("Checkpoint and observed graph vocabularies differ")
    q, v, w, g0 = (torch.tensor(model[name], dtype=torch.float32) for name in ("Q", "V", "W", "G"))
    frozen = {name: tensor_hash(value) for name, value in (("Q", q), ("V", v), ("W", w))}
    n_actions, n_states = len(model["edges"]), model["n_states"]
    truth = np.zeros((n_actions, n_states), dtype=bool)
    for action, (source, _) in enumerate(model["edges"]):
        truth[action, source] = True
    availability = torch.from_numpy(truth.astype(np.float32))
    source_counts: Counter[int] = Counter()
    edge_counts: Counter[int] = Counter()
    state_weights = np.zeros(n_states, dtype=np.float64)
    source_batches = []
    for trajectory in walks["trajectories"]:
        triples = trajectory["transitions"]
        if not triples:
            raise ValueError("Unexpected empty original trajectory")
        previous = None
        sources = []
        for source, edge, target in triples:
            if model["edges"][edge] != [source, target] or (previous is not None and source != previous):
                raise ValueError("Original replay has inconsistent transition provenance")
            source_counts[source] += 1
            edge_counts[edge] += 1
            state_weights[source] += 1 / len(triples)
            sources.append(source)
            previous = target
        source_batches.append(torch.tensor(sources, dtype=torch.long))
    sources = sorted(source_counts)
    sinks = [state for state in range(n_states) if not truth[:, state].any()]
    if not sinks:
        raise ValueError("The frozen readout profile compares source-only and sink-supervised fits; it requires sink states")
    if sources != np.flatnonzero(truth.any(axis=0)).tolist():
        raise ValueError("Observed graph sources differ from directly supervised replay sources")
    contract = load(args.experience_contract.expanduser().resolve())
    if set(contract) != {"source_batches_per_epoch", "source_exposures_per_epoch", "exposures_per_edge"}:
        raise ValueError("Experience contract must declare batch/sample/edge exposure counts")
    if any(type(value) is not int or value < 1 for value in contract.values()):
        raise ValueError("Experience contract counts must be positive integers")
    if (len(source_batches), sum(source_counts.values())) != (contract["source_batches_per_epoch"], contract["source_exposures_per_epoch"]):
        raise ValueError("Replay does not match the explicit experience contract")
    if set(edge_counts.values()) != {contract["exposures_per_edge"]} or len(edge_counts) != n_actions:
        raise ValueError("Original edge sampling quota changed")
    if any(state in source_counts for state in sinks):
        raise ValueError("Sink state unexpectedly occurred in original source samples")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = args.output_root.expanduser().resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    args._active_output = directory
    print(f"G-only readout artifacts: {directory}", flush=True)
    started = perf_counter()
    package_before = source_hashes()
    archive_package_sources(directory / "source")
    write_json(directory / "experience_contract.json", contract)
    write_json(directory / "runtime.json", runtime_versions())
    write_json(directory / "source_hashes.json", package_before)
    manifest = {
        "schema": "cmnav.gcml_frozen_G_experiment.v1", "status": "running",
        "source_run": str(source_dir), "original_training_run": str(original_dir),
        "inputs": {name: {"path": str(path), "sha256": fingerprints[name]} for name, path in input_paths.items()},
        "frozen_tensor_hashes": frozen, "original_G_tensor_sha256": tensor_hash(g0),
        "frozen_modules": ["Q", "V", "W"], "navigation_or_task_execution": False,
        "new_environment_experience": False, "static_availability_threshold": .5,
        "epochs_per_iterative_condition": args.epochs, "learning_rates": args.rates,
        "readout_step_safety": model["readout_step_safety"],
        "source_batches_per_epoch": len(source_batches), "source_exposures_per_epoch": sum(source_counts.values()),
        "source_state_ids_from_actual_samples": sources, "sink_state_ids": sinks,
        "source_exposures_by_state_per_epoch": dict(source_counts),
        "sink_pair_extra_batches_per_epoch": 1, "sink_pair_extra_exposures_per_epoch": len(sinks),
        "sink_pair_has_equal_source_budget_but_extra_total_budget": True,
        "svd_rcond": args.rcond, "script_sha256": sha256_file(Path(__file__)),
        "experience_contract_sha256": sha256_file(args.experience_contract.expanduser().resolve()),
        "environment_threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
    write_json(directory / "manifest.json", manifest)
    original_metrics = grouped((g0 @ q).numpy(), truth, sources, sinks)
    original_norms = readout_norms(g0.numpy())
    write_json(directory / "initial_metrics.json", {"metrics": original_metrics, "G_norms": original_norms})
    q64, g64 = q.numpy().astype(np.float64), g0.numpy().astype(np.float64)
    all_states = list(range(n_states))
    all_weights = state_weights.copy()
    all_weights[sinks] = 1 / len(sinks)
    references = [
        ("source_uniform_minimum_norm", sources, np.ones(len(sources)), False),
        ("source_trajectory_weighted_minimum_norm", sources, state_weights[sources], False),
        ("source_trajectory_weighted_minimum_change", sources, state_weights[sources], True),
        ("all_states_uniform_minimum_norm", all_states, np.ones(n_states), False),
        ("all_states_trajectory_plus_sink_batch_weighted_minimum_norm", all_states, all_weights, False),
    ]
    ls_summaries = []
    for name, nodes, weights, minimum_change in references:
        fitted64, solver = svd_reference(q64, g64, truth, nodes, weights,
                                         rcond=args.rcond, minimum_change=minimum_change)
        fitted32 = torch.from_numpy(fitted64.astype(np.float32))
        metrics64 = grouped(fitted64 @ q64, truth, sources, sinks)
        metrics32 = grouped((fitted32 @ q).numpy(), truth, sources, sinks)
        detail = {
            "variant": name, "kind": "least_squares_capacity_reference",
            "directly_supervised_state_ids": nodes, "solver": solver,
            "float64_metrics": metrics64, "float32_metrics": metrics32,
            "G_float64_norms": readout_norms(fitted64),
            "G_float32_norms": readout_norms(fitted32.numpy()),
            "source_checkpoint": str(input_paths["model"]),
            "G_only_iterative_updates": 0, "state_labels_used": len(nodes),
            "nullspace_interpretation": (
                "Retains the original G component orthogonal to the supervised representation span; this is the fixed-feature unregularized warm-start GD limit if optimized to interpolation."
                if minimum_change else
                "Chooses zero G in unconstrained directions; sink behavior can improve by changing that nullspace even without sink labels."
            ),
            "drift_bound": "||G delta_z||_2 <= ||G||_2 ||delta_z||_2; this is a bound, not an observed-rollout measurement",
        }
        target = directory / "least_squares" / name
        save_checkpoint(target, model, fitted32, detail, frozen)
        np.save(target / "G_float64.npy", fitted64)
        ls_summaries.append({key: value for key, value in detail.items() if key != "directly_supervised_state_ids"})
        write_json(directory / "least_squares" / "summary.json", ls_summaries)
        print(json.dumps({"least_squares": name, "rank": solver["rank"],
                          "condition": solver["condition_largest_over_smallest"],
                          "float32_source_rmse": metrics32["sources"]["rmse"],
                          "float32_source_fp": metrics32["sources"]["fp"],
                          "float32_source_fn": metrics32["sources"]["fn"],
                          "float32_sink_fp": metrics32["sinks"]["fp"]}), flush=True)
    generator = torch.Generator(device="cpu")
    generator.set_state(torch.tensor(model["shuffle_generator_state"], dtype=torch.uint8))
    initial_order_hash = hashlib.sha256(bytes(generator.get_state().tolist())).hexdigest()
    loader = DataLoader(list(range(len(source_batches))), batch_size=1, shuffle=True,
                        num_workers=0, generator=generator)
    orders = [[int(batch.item()) for batch in loader] for _ in range(args.epochs)]
    order_path = directory / "batch_orders.json"
    write_json(order_path, {"policy": "continue_original_checkpoint_DataLoader_generator_one_trajectory_per_batch",
                           "initial_generator_sha256": initial_order_hash,
                           "final_generator_sha256": hashlib.sha256(bytes(generator.get_state().tolist())).hexdigest(),
                           "epochs": orders})
    conditions = [{"name": f"{mode}_lr_{rate:g}", "rate": rate, "sink_supervision": mode == "sources_plus_sinks"}
                  for mode in ("sources_only", "sources_plus_sinks") for rate in args.rates]
    states = torch.stack([g0.clone() for _ in conditions])
    rates = torch.tensor([row["rate"] for row in conditions], dtype=torch.float32)
    requested_rates = [row["rate"] for row in conditions]
    safety = float(model["readout_step_safety"])
    cached = [(q[:, ids], availability[:, ids]) for ids in source_batches]
    caps = [safety * features.shape[1] / max(float(features.square().sum().item()), 1e-30)
            for features, _ in cached]
    # Python min then cast reproduces the implementation's scalar effective
    # step and update expression; both paired variants use identical source steps.
    rate_vectors = [torch.tensor([min(rate, cap) / len(ids) for rate in requested_rates], dtype=torch.float32)
                    for cap, ids in zip(caps, source_batches)]
    sink_features = q[:, sinks]
    sink_cap = safety * len(sinks) / max(float(sink_features.square().sum().item()), 1e-30)
    sink_factors = torch.tensor([min(rate, sink_cap) / len(sinks) for rate in args.rates], dtype=torch.float32)
    epoch_records: list[list[dict[str, Any]]] = [[] for _ in conditions]
    with (directory / "readout_epochs.jsonl").open("x", encoding="utf-8") as progress, torch.no_grad():
        for epoch, order in enumerate(orders, 1):
            loss_sums = np.zeros(len(conditions), dtype=np.float64)
            update_norm_sums = np.zeros(len(conditions), dtype=np.float64)
            for index in order:
                features, desired = cached[index]
                residual = desired.unsqueeze(0) - torch.matmul(states, features)
                if not bool(torch.isfinite(residual).all().item()):
                    raise FloatingPointError("Nonfinite fixed-Q G residual")
                update = torch.matmul(residual, features.T) * rate_vectors[index][:, None, None]
                states.add_(update)
                loss_sums += residual.square().mean(dim=(1, 2)).numpy()
                update_norm_sums += torch.linalg.vector_norm(update, dim=(1, 2)).numpy()
            sink_residual = -torch.matmul(states[len(args.rates):], sink_features)
            sink_updates = torch.matmul(sink_residual, sink_features.T) * sink_factors[:, None, None]
            states[len(args.rates):].add_(sink_updates)
            if not bool(torch.isfinite(states).all().item()):
                raise FloatingPointError("Nonfinite fixed-Q G parameters")
            scores = torch.matmul(states, q).numpy()
            brief = []
            for index, condition in enumerate(conditions):
                rate = condition["rate"]
                effective = [min(rate, cap) for cap in caps]
                record = {
                    "variant": condition["name"], "epoch": epoch,
                    "source_preupdate_batch_mean_mse": float(loss_sums[index] / len(cached)),
                    "source_mean_batch_update_norm": float(update_norm_sums[index] / len(cached)),
                    "metrics": grouped(scores[index], truth, sources, sinks),
                    "G_frobenius_norm": float(torch.linalg.vector_norm(states[index]).item()),
                    "source_effective_rate_min": min(effective), "source_effective_rate_max": max(effective),
                    "source_capped_batches": sum(value < rate for value in effective),
                    "source_exposures_cumulative": epoch * sum(source_counts.values()),
                    "source_updates_cumulative": epoch * len(cached),
                    "sink_exposures_cumulative": epoch * len(sinks) if condition["sink_supervision"] else 0,
                    "sink_updates_cumulative": epoch if condition["sink_supervision"] else 0,
                    "sink_effective_rate": min(rate, sink_cap) if condition["sink_supervision"] else None,
                    "sink_capped_updates_this_epoch": int(rate > sink_cap) if condition["sink_supervision"] else 0,
                    "finite": True, "elapsed_seconds": perf_counter() - started,
                }
                epoch_records[index].append(record)
                write_json_line(progress, record)
                brief.append({"variant": condition["name"], "source_rmse": record["metrics"]["sources"]["rmse"],
                              "source_fp": record["metrics"]["sources"]["fp"], "source_fn": record["metrics"]["sources"]["fn"],
                              "sink_fp": record["metrics"]["sinks"]["fp"]})
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(json.dumps({"epoch": epoch, "conditions": brief}), flush=True)
    condition_summaries = []
    for index, condition in enumerate(conditions):
        detail = {
            "variant": condition["name"], "kind": "fixed_Q_G_only_delta_rule",
            "source_checkpoint": str(input_paths["model"]),
            "initial_G_tensor_sha256": tensor_hash(g0), "nominal_learning_rate": condition["rate"],
            "readout_step_safety": safety,
            "readout_step_formula": "min(nominal_rate,safety*B/||S||_F^2)",
            "update_formula": "G += effective_rate/B * (A-GS)S^T",
            "update_dtype": "torch.float32", "vectorized_conditions": True,
            "epochs": args.epochs, "batch_orders_sha256": sha256_file(order_path),
            "training_walks_sha256": fingerprints["walks"],
            "source_exposures": args.epochs * sum(source_counts.values()),
            "source_batches": args.epochs * len(source_batches),
            "sink_supervision": condition["sink_supervision"],
            "sink_exposures": args.epochs * len(sinks) if condition["sink_supervision"] else 0,
            "sink_batches": args.epochs if condition["sink_supervision"] else 0,
            "sink_batch_position": "after_all_source_batches_each_epoch" if condition["sink_supervision"] else None,
            "total_update_count": args.epochs * (len(source_batches) + int(condition["sink_supervision"])),
            "final_metrics": epoch_records[index][-1]["metrics"],
            "final_G_norms": readout_norms(states[index].numpy()),
            "epochs_detail": epoch_records[index],
            "parameter_updates_exclude_Q_V_W": True,
        }
        save_checkpoint(directory / "iterative" / condition["name"], model, states[index], detail, frozen)
        condition_summaries.append({key: value for key, value in detail.items() if key != "epochs_detail"})
    comparisons = []
    for index, rate in enumerate(args.rates):
        only = condition_summaries[index]["final_metrics"]
        added = condition_summaries[index + len(args.rates)]["final_metrics"]
        comparisons.append({
            "learning_rate": rate, "source_only": only, "sources_plus_sinks": added,
            "source_rmse_delta_with_sink_supervision": added["sources"]["rmse"] - only["sources"]["rmse"],
            "source_fp_delta_with_sink_supervision": added["sources"]["fp"] - only["sources"]["fp"],
            "source_fn_delta_with_sink_supervision": added["sources"]["fn"] - only["sources"]["fn"],
            "sink_fp_delta_with_sink_supervision": added["sinks"]["fp"] - only["sinks"]["fp"],
            "extra_sink_exposures": args.epochs * len(sinks), "extra_sink_updates": args.epochs,
        })
    if fingerprints != {name: sha256_file(path) for name, path in input_paths.items()}:
        raise RuntimeError("An original input changed during G-only fitting")
    if package_before != source_hashes():
        raise RuntimeError("Package source changed during G-only fitting")
    if frozen != {name: tensor_hash(value) for name, value in (("Q", q), ("V", v), ("W", w))}:
        raise RuntimeError("A frozen tensor changed")
    manifest.update(status="completed", elapsed_seconds=perf_counter() - started,
                    frozen_tensors_verified=True, original_inputs_verified=True,
                    batch_orders_sha256=sha256_file(order_path))
    write_json(directory / "summary.json", {
        "status": "completed", "initial_metrics": original_metrics, "initial_G_norms": original_norms,
        "least_squares": ls_summaries, "iterative": condition_summaries, "sink_pairs": comparisons,
        "no_navigation_performed": True, "frozen_tensor_hashes": frozen,
    })
    from cmnav.reporting.readout_fit import write_readout_report
    write_readout_report(directory, source_dir, ls_summaries, condition_summaries,
                         epochs=args.epochs, source_batch_count=len(source_batches),
                         source_exposure_count=sum(source_counts.values()), sink_count=len(sinks))
    write_json(directory / "manifest.json", manifest)
    print(f"Completed G-only readout experiment: {directory}", flush=True)
    return directory


def fit_readout(args: argparse.Namespace) -> Path:
    """Fit with an accurate terminal status even through a legacy entrypoint."""
    args._active_output = None
    try:
        return _fit_readout(args)
    except BaseException as exc:
        if args._active_output is not None:
            import traceback
            directory = args._active_output
            for filename in ("manifest.json", "summary.json"):
                path = directory / filename
                state = load(path) if path.exists() else {}
                state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                write_json(path, state)
            (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def add_reference(directory: Path) -> Path:
    torch.set_num_threads(1)
    manifest = load(directory / "manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError("Supplemental references require a completed readout fit")
    if (directory / "minimum_change_supplement.json").exists():
        raise FileExistsError("The supplemental reference already exists")
    source = Path(manifest["inputs"]["model"]["path"])
    if sha256_file(source) != manifest["inputs"]["model"]["sha256"]:
        raise ValueError("Original model changed")
    state = load(source)
    q = torch.tensor(state["Q"], dtype=torch.float32)
    g0 = torch.tensor(state["G"], dtype=torch.float32)
    frozen = {name: tensor_hash(torch.tensor(state[name], dtype=torch.float32))
              for name in ("Q", "V", "W")}
    if frozen != manifest["frozen_tensor_hashes"]:
        raise ValueError("Frozen Q/V/W do not match original experiment")
    truth = np.zeros((len(state["edges"]), state["n_states"]), dtype=bool)
    for edge, (node, _) in enumerate(state["edges"]):
        truth[edge, node] = True
    sources, sinks = manifest["source_state_ids_from_actual_samples"], manifest["sink_state_ids"]
    nodes = list(range(state["n_states"]))
    q64 = q.numpy().astype(np.float64)
    fitted, solver = svd_reference(q64, g0.numpy().astype(np.float64), truth, nodes,
                                    np.ones(len(nodes)), rcond=manifest["svd_rcond"],
                                    minimum_change=True)
    g = torch.from_numpy(fitted.astype(np.float32))
    variant = "all_states_uniform_minimum_change"
    metadata = {
        "variant": variant, "kind": "supplemental_least_squares_capacity_reference",
        "source_checkpoint": str(source), "source_checkpoint_sha256": sha256_file(source),
        "directly_supervised_state_ids": nodes, "solver": solver,
        "float64_metrics": grouped(fitted @ q64, truth, sources, sinks),
        "float32_metrics": grouped((g @ q).numpy(), truth, sources, sinks),
        "G_float64_norms": readout_norms(fitted), "G_float32_norms": readout_norms(g.numpy()),
        "G_only_iterative_updates": 0, "state_labels_used": len(nodes),
        "purpose": "Separate perfect known-state readout from minimum-norm nullspace removal",
        "nullspace_interpretation": f"Retains original G outside the span of all {state['n_states']} Q columns",
        "script_sha256": sha256_file(Path(__file__)),
    }
    target = directory / "least_squares" / variant
    save_checkpoint(target, state, g, metadata, frozen)
    np.save(target / "G_float64.npy", fitted)
    archive_package_sources(target / "source")
    write_json(directory / "minimum_change_supplement.json", {
        "checkpoint": str(target / "model.json"), "model_sha256": sha256_file(target / "model.json"),
        "readout_training": str(target / "readout_training.json"),
        "float32_metrics": metadata["float32_metrics"], "rank": solver["rank"],
        "G_norms": metadata["G_float32_norms"],
    })
    print(json.dumps({"checkpoint": str(target / "model.json"),
                      "metrics": metadata["float32_metrics"], "G_norms": metadata["G_float32_norms"]}))
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/gcml_g_readout"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--rates", type=float, nargs="+", default=[.0005, .001, .002])
    parser.add_argument("--rcond", type=float, default=1e-12)
    parser.add_argument("--experience-contract", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.epochs < 1 or not args.rates or any(not math.isfinite(rate) or rate <= 0 for rate in args.rates):
        parser.error("epochs and rates must be positive")
    if len({f"{rate:g}" for rate in args.rates}) != len(args.rates):
        parser.error("rates must produce unique condition names when formatted with :g")
    if not math.isfinite(args.rcond) or args.rcond <= 0:
        parser.error("rcond must be finite and positive")
    fit_readout(args)


def reference_main(argv: list[str] | None = None) -> None:
    """Add the declared all-state minimum-change reference to a saved fit study."""
    parser = argparse.ArgumentParser(description=reference_main.__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    add_reference(args.directory.expanduser().resolve())


def legacy_main(defaults_path: Path, argv: list[str] | None = None) -> None:
    """Compatibility defaults are data, while fitting stays in one implementation."""
    import sys
    defaults_path = defaults_path.resolve()
    defaults = load(defaults_path)
    main(["--source-run", str((defaults_path.parent / defaults["source_run"]).resolve()),
          "--experience-contract", str((defaults_path.parent / defaults["experience_contract"]).resolve()),
          *(sys.argv[1:] if argv is None else argv)])
