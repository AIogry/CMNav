"""Frozen-experience GCML training with replay and checkpoint publication profiles.

The replay profile preserves historical defaults and records explicit training overrides.
The checkpoint profile requires CUDA and publishes independently fitted readouts;
both consume identical saved batches without resampling or goal supervision.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from time import perf_counter
import traceback
from typing import Any
import torch
from cmnav.memory.oracle_graph import OracleObservedGraph
from cmnav.methods.gcml import GenerativeCognitiveMapLearner
from cmnav.methods.linear_readout import fit_frozen_availability
from cmnav.runtime.artifacts import sha256_file, write_json, write_json_line
from cmnav.runtime.provenance import runtime_versions, source_hashes
from cmnav.runtime.gcml_artifacts import (
    TRAINING_INPUT_NAMES as INPUT_NAMES, SHARED_TRAINING_OUTPUT_NAMES as SHARED_OUTPUT_NAMES,
    load_json as load, training_input_hashes as fingerprint_inputs,
    tensor_hash, parameter_hash, tensor_state, publish_json,
)
from cmnav.runtime.gcml_source import snapshot_gcml_source, load_historical_contracts


def source_snapshot(directory, before):
    return snapshot_gcml_source(directory, before, runner=Path(__file__).resolve())


def load_replay_batches(frozen: Path, fingerprints, previous_training):
    """Load saved ordered batches and enforce their original graph/count identity."""
    graph = OracleObservedGraph.from_state_dict(load(frozen / "graph.json"))
    walks = load(frozen / "training_walks.json")
    trajectories = [row["transitions"] for row in walks["trajectories"] if row["transitions"]]
    transitions = sum(map(len, trajectories))
    if (transitions != walks["emitted_transitions"] or
            transitions != previous_training["transitions_per_epoch"] or
            len(trajectories) != previous_training["trajectories_per_epoch"] or
            walks["source"]["graph_sha256"] != fingerprints["graph.json"]):
        raise ValueError("Frozen training graph/batch/count identity mismatch")
    return graph, walks, trajectories, transitions


def create_replay_model(graph, config, reference):
    return GenerativeCognitiveMapLearner(
        len(graph.positions), graph.edges, dimension=config["cml_dimension"], seed=config["seed"],
        learning_rate=config["learning_rate"], state_learning_rate=config["state_learning_rate"],
        inverse_learning_rate=config["gcml_inverse_learning_rate"],
        affordance_learning_rate=config["gcml_affordance_learning_rate"],
        readout_step_safety=config["gcml_readout_step_safety"],
        device=config["cml_device"], reference_metadata=reference,
    )

def reproduction(model: GenerativeCognitiveMapLearner, source: Path,
                 original_dimension: int) -> dict[str, Any]:
    if model.dimension != 1000 or original_dimension != 1000:
        return {"performed": False, "reason": "Exact checkpoint comparison is reserved for the 1000-dimensional control"}
    original = load(source / "model.json")
    tensors = {}
    for name in ("Q", "V", "W", "G"):
        expected = torch.tensor(original[name], dtype=torch.float32, device="cpu")
        observed = getattr(model, name).detach().cpu()
        same_shape = tuple(expected.shape) == tuple(observed.shape)
        tensors[name] = {
            "shape_matches": same_shape,
            "exact": same_shape and torch.equal(observed, expected),
            "max_absolute_delta": float((observed - expected).abs().max().item()) if same_shape else None,
            "source_tensor_sha256": tensor_hash(expected),
            "trained_tensor_sha256": tensor_hash(observed),
        }
    counters = {name: {"source": original[name], "trained": getattr(model, name),
                       "matches": original[name] == getattr(model, name)}
                for name in ("updates", "trajectory_updates", "epochs_completed", "planning_calls")}
    return {"performed": True, "source_model": str(source / "model.json"),
            "all_tensors_exact": all(row["exact"] for row in tensors.values()),
            "tensors": tensors, "counters": counters,
            "automatic_parameter_adjustment": False}


def run_replay(source: Path, dimension: int, directory: Path, *, epochs: int | None = None,
               seed: int | None = None, device: str | None = None, contract: Path | None = None) -> Path:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be positive")
    if seed is not None and not 0 <= seed < 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    contracts, contract_path, contract_hash = load_historical_contracts(contract)
    expected_source = contracts["replay_source"]
    source, directory = source.expanduser().resolve(), directory.expanduser().resolve()
    if directory == source or directory.is_relative_to(source):
        raise ValueError("Output must be a new directory outside the historical source run")
    directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    fingerprints_before: dict[str, str] = {}
    package_before: dict[str, str] = {}
    config: dict[str, Any] = {}
    write_json(directory / "summary.json", {
        "status": "initializing", "training_only": True, "navigation_evaluated": False,
        "dimension": dimension, "source_run": str(source), "started_at": started_at,
    })
    with (directory / "training.log").open("x", encoding="utf-8") as log:
        def announce(message: str) -> None:
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        try:
            torch.set_num_threads(1)
            announce(f"GCML dimension training: source={source}, dimension={dimension}, output={directory}")
            fingerprints_before = fingerprint_inputs(source)
            write_json(directory / "input_hashes_before.json", fingerprints_before)
            frozen = directory / "frozen_inputs"
            frozen.mkdir()
            for name in INPUT_NAMES:
                shutil.copyfile(source / f"{name}.json", frozen / f"{name}.json")
            if fingerprint_inputs(frozen) != fingerprints_before:
                raise RuntimeError("Frozen input copies differ from source fingerprints")

            original_config = load(frozen / "config.json")
            original_summary = load(frozen / "summary.json")
            original_training = load(frozen / "training.json")
            if original_summary.get("status") != "completed":
                raise ValueError("Historical source run is not completed")
            for key, expected in expected_source.items():
                if original_config.get(key) != expected:
                    raise ValueError(f"Source violates declared replay contract: {key}")
            if original_training.get("epochs") != original_config["cml_epochs"]:
                raise ValueError("Historical training/configuration epoch counts disagree")
            if original_training.get("walks_sha256") != fingerprints_before["training_walks.json"]:
                raise ValueError("Historical training does not identify these exact trajectory batches")

            config = dict(original_config)
            config.update(cml_dimension=dimension, output_dir=str(directory))
            for key, value in (("cml_epochs", epochs), ("seed", seed), ("cml_device", device)):
                if value is not None:
                    config[key] = value
            requested_device = torch.device(config["cml_device"])
            if requested_device.type == "cuda":
                if (requested_device.index is None or not torch.cuda.is_available()
                        or requested_device.index >= torch.cuda.device_count()):
                    raise RuntimeError("Declared CUDA device unavailable; CPU fallback is forbidden")
                torch.cuda.set_device(requested_device)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.set_float32_matmul_precision("highest")
            elif requested_device.type != "cpu":
                raise ValueError("Replay supports explicit CPU or CUDA devices")
            write_json(directory / "comparison_contract.json", {
                "path": str(contract_path), "sha256": contract_hash, "expected_source": expected_source,
                "explicit_training_overrides": {"epochs": epochs, "seed": seed, "device": device},
            })
            changes = {key: {"source": original_config[key], "trained": value}
                       for key, value in config.items() if value != original_config[key]}
            if set(changes) - {"cml_dimension", "output_dir", "cml_epochs", "seed", "cml_device"}:
                raise RuntimeError("Unexpected configuration change")
            write_json(directory / "config.json", config)
            write_json(directory / "configuration_changes.json", changes)
            for name in SHARED_OUTPUT_NAMES:
                shutil.copyfile(frozen / f"{name}.json", directory / f"{name}.json")
            package_before = source_hashes()
            write_json(directory / "source_hashes.json", package_before)
            snapshot = source_snapshot(directory, package_before)
            runtime = runtime_versions()
            runtime.update(device=str(requested_device), torch_cuda_version=torch.version.cuda,
                           cpu_fallback=False,
                           device_name=torch.cuda.get_device_name(requested_device) if requested_device.type == "cuda" else "cpu")
            runtime.update(torch_num_threads=torch.get_num_threads(),
                           torch_num_interop_threads=torch.get_num_interop_threads(),
                           environment_threads={key: os.environ.get(key) for key in
                                                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
            write_json(directory / "runtime.json", runtime)

            graph, walks, trajectories, transitions = load_replay_batches(frozen, fingerprints_before, original_training)
            reference = load(frozen / "algorithm_reference.json")
            model = create_replay_model(graph, config, reference)
            write_json(directory / "initial_parameters.json", {
                "tensor_sha256": {name: tensor_hash(getattr(model, name)) for name in ("Q", "V", "W", "G")},
                "parameter_tensor_sha256": parameter_hash(model),
                "dimension": dimension, "seed": model.seed,
                "initialization": "unchanged_GenerativeCognitiveMapLearner_constructor",
                "parameter_count": sum(getattr(model, name).numel() for name in ("Q", "V", "W", "G")),
            })
            write_json(directory / "summary.json", {
                "status": "training", "training_only": True, "navigation_evaluated": False,
                "dimension": dimension, "source_run": str(source), "started_at": started_at,
                "graph_nodes": len(graph.positions), "graph_edges": len(graph.edges),
                "trajectories_per_epoch": len(trajectories), "transitions_per_epoch": transitions,
                "epochs": config["cml_epochs"],
            })
            announce(f"Training {config['cml_epochs']} epochs with {len(trajectories)} original batches and {transitions} transitions per epoch")
            with (directory / "epoch_progress.jsonl").open("x", encoding="utf-8") as progress:
                def record_epoch(row: dict[str, Any]) -> None:
                    write_json_line(progress, row)
                    announce(f"Epoch {row['epoch']}/{config['cml_epochs']}: "
                             f"transition_MSE={row['prediction_mse']:.7g} "
                             f"inverse_MSE={row['inverse_mse']:.7g} "
                             f"affordance_MSE={row['affordance_mse']:.7g} "
                             f"rate_min={row['effective_readout_rate_min']} "
                             f"capped_batches={row['capped_readout_batches']}")
                training = model.fit(trajectories, epochs=config["cml_epochs"], progress_callback=record_epoch)
            training.update(
                training_protocol=config["gcml_protocol"], walk_coverage=walks["coverage"],
                walks_sha256=fingerprints_before["training_walks.json"],
                replayed_transitions_per_epoch=transitions, sampled_trajectories=len(trajectories),
                source_run=str(source), trajectories_loaded_without_resampling=True,
                dimension=dimension, configuration_changes=changes,
            )
            write_json(directory / "training.json", training)
            write_json(directory / "model.json", model.state_dict())
            legacy_control = all(config[key] == original_config[key] for key in ("cml_epochs", "seed", "cml_device"))
            reproduced = (reproduction(model, frozen, original_config["cml_dimension"]) if legacy_control else
                          {"performed": False, "reason": "Explicit training overrides differ from historical control"})
            write_json(directory / "reproduction.json", reproduced)

            fingerprints_after = fingerprint_inputs(source)
            frozen_fingerprints_after = fingerprint_inputs(frozen)
            package_after = source_hashes()
            write_json(directory / "input_hashes_after.json", fingerprints_after)
            write_json(directory / "frozen_input_hashes_after.json", frozen_fingerprints_after)
            write_json(directory / "source_hashes_after.json", package_after)
            unchanged = fingerprints_before == fingerprints_after
            frozen_unchanged = fingerprints_before == frozen_fingerprints_after
            package_unchanged = package_before == package_after
            script_unchanged = sha256_file(Path(__file__).resolve()) == snapshot["script_sha256"]
            if (not unchanged or not frozen_unchanged or not package_unchanged or not script_unchanged
                    or sha256_file(contract_path) != contract_hash):
                raise RuntimeError("Historical inputs or running source changed during training")
            summary = {
                "status": "completed", "training_only": True, "navigation_evaluated": False,
                "dimension": dimension, "source_run": str(source), "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "graph_nodes": len(graph.positions), "graph_edges": len(graph.edges),
                "epochs": model.epochs_completed, "updates": model.updates,
                "trajectory_updates": model.trajectory_updates,
                "original_inputs_unchanged": unchanged, "package_source_unchanged": package_unchanged,
                "frozen_inputs_unchanged": frozen_unchanged,
                "runner_source_unchanged": script_unchanged,
                "model_sha256": sha256_file(directory / "model.json"),
                "parameter_tensor_sha256": parameter_hash(model),
                "tensor_sha256": {name: tensor_hash(getattr(model, name)) for name in ("Q", "V", "W", "G")},
                "final_diagnostics": training["final_diagnostics"], "reproduction": reproduced,
                "elapsed_seconds": perf_counter() - started,
            }
            write_json(directory / "summary.json", summary)
            announce(f"Completed dimension {dimension}: {directory}")
            if reproduced["performed"]:
                announce(f"1000-dimensional tensor reproduction exact: {reproduced['all_tensors_exact']}")
            return directory
        except BaseException as exc:
            error = traceback.format_exc()
            (directory / "error.txt").write_text(error, encoding="utf-8")
            log.write(error)
            log.flush()
            try:
                fingerprints_after = fingerprint_inputs(source)
                write_json(directory / "input_hashes_after.json", fingerprints_after)
                inputs_unchanged = fingerprints_before == fingerprints_after if fingerprints_before else None
            except (OSError, ValueError):
                inputs_unchanged = None
            try:
                package_after = source_hashes()
                write_json(directory / "source_hashes_after.json", package_after)
                package_unchanged = package_before == package_after if package_before else None
            except OSError:
                package_unchanged = None
            write_json(directory / "summary.json", {
                "status": "failed", "training_only": True, "navigation_evaluated": False,
                "dimension": dimension, "source_run": str(source), "started_at": started_at,
                "error": str(exc), "error_type": type(exc).__name__,
                "original_inputs_unchanged": inputs_unchanged,
                "package_source_unchanged": package_unchanged,
                "elapsed_seconds": perf_counter() - started,
            })
            raise


def run_checkpoints(run_directory: Path):
    run_directory = run_directory.resolve()
    protocol_path = run_directory / "protocol.json"
    protocol = load(protocol_path)
    directory = run_directory / "training"
    directory.mkdir(exist_ok=False)
    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    publish_json(directory / "manifest.json", {"status": "initializing"})
    try:
        device = torch.device(protocol["training_device"])
        if device.type != "cuda" or not torch.cuda.is_available() or (device.index is None or device.index >= torch.cuda.device_count()):
            raise RuntimeError("The declared CUDA device is required; no CPU fallback")
        torch.cuda.set_device(device)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        source = Path(protocol["source_run"])
        fingerprints = fingerprint_inputs(source)
        protocol_hash = sha256_file(protocol_path)
        package_hashes = source_hashes()
        snapshot = source_snapshot(directory, package_hashes)
        script_hashes = snapshot["script_hashes"]
        write_json(directory / "input_hashes_before.json", fingerprints)
        frozen = directory / "frozen_inputs"
        frozen.mkdir()
        for name in INPUT_NAMES:
            shutil.copyfile(source / f"{name}.json", frozen / f"{name}.json")
        if fingerprint_inputs(frozen) != fingerprints:
            raise RuntimeError("Frozen inputs differ")
        original = load(frozen / "config.json")
        old_training = load(frozen / "training.json")
        if load(frozen / "summary.json")["status"] != "completed":
            raise ValueError("Incomplete source experiment")
        if old_training["walks_sha256"] != fingerprints["training_walks.json"]:
            raise ValueError("Source training walk identity mismatch")
        config = dict(original)
        config.update(cml_dimension=protocol["dimension"], cml_epochs=protocol["epochs"],
                      cml_device=str(device), seed=protocol["training_seed"], output_dir=str(directory))
        write_json(directory / "config.json", config)
        runtime = runtime_versions()
        runtime.update(device=str(device), device_name=torch.cuda.get_device_name(device),
                       torch_cuda_version=torch.version.cuda, torch_version=torch.__version__,
                       tf32=False, torch_num_threads=torch.get_num_threads(),
                       total_memory=torch.cuda.get_device_properties(device).total_memory)
        write_json(directory / "runtime.json", runtime)
        graph, walks, trajectories, transitions = load_replay_batches(frozen, fingerprints, old_training)
        model = create_replay_model(graph, config, load(frozen / "algorithm_reference.json"))
        write_json(directory / "initial_parameters.json", {
            "device": str(model.device), "tensor_sha256": {
                name: tensor_hash(getattr(model, name)) for name in ("Q", "V", "W", "G")}})
        labels = torch.zeros((model.n_edges, model.n_states), dtype=torch.float32, device=device)
        for action, (state, _) in enumerate(graph.edges):
            labels[action, state] = 1
        checkpoint_root = directory / "checkpoints"
        checkpoint_root.mkdir()
        checkpoints = []

        def assert_unchanged():
            if source_hashes() != package_hashes or any(sha256_file(Path(p)) != h for p, h in script_hashes.items()):
                raise RuntimeError("Running package/runner source changed")
            if sha256_file(protocol_path) != protocol_hash:
                raise RuntimeError("Frozen protocol changed")

        with (directory / "epoch_progress.jsonl").open("x", encoding="utf-8") as progress:
            def record(row):
                write_json_line(progress, row)
                print(f"Epoch {row['epoch']}: QV={row['prediction_mse']:.7g} W={row['inverse_mse']:.7g} G={row['affordance_mse']:.7g}", flush=True)
                if row["epoch"] not in protocol["checkpoints"]:
                    return
                assert_unchanged()
                checkpoint_started = perf_counter()
                target = checkpoint_root / f"epoch_{row['epoch']:03d}"
                target.mkdir()
                raw_hash = parameter_hash(model)
                variants, readout = fit_frozen_availability(model.Q, model.G, labels,
                                                          rcond=protocol["fit_G"]["rcond"])
                if readout["solve"]["rank"] != model.n_states:
                    raise RuntimeError("Frozen Q is not full column rank")
                diagnostics = {"raw": model.training_diagnostics()}
                state = tensor_state(model)
                torch.save(state, target / "model.pt")
                hashes = {"raw": sha256_file(target / "model.pt")}
                for name, fitted in variants.items():
                    variant_state = dict(state)
                    variant_state["G"] = fitted.cpu()
                    variant_state["readout_adaptation"] = {
                        "variant": name, "all_observed_states": True, "rcond": protocol["fit_G"]["rcond"]}
                    filename = target / f"model_{name}.pt"
                    torch.save(variant_state, filename)
                    hashes[name] = sha256_file(filename)
                if parameter_hash(model) != raw_hash:
                    raise RuntimeError("Readout fitting modified the training model")
                write_json(target / "diagnostics.json", diagnostics)
                write_json(target / "readout.json", readout)
                write_json(target / "config.json", config)
                assert_unchanged()
                manifest = {"status": "completed", "epoch": row["epoch"], "device": str(device),
                            "checkpoint_hashes": hashes, "raw_parameter_hash": raw_hash,
                            "training_counters": {key: getattr(model, key) for key in
                                                   ("epochs_completed", "updates", "trajectory_updates")},
                            "checkpoint_seconds": perf_counter() - checkpoint_started}
                publish_json(target / "manifest.json", manifest)
                checkpoints.append(manifest)
                publish_json(directory / "manifest.json", {"status": "training", "epoch": row["epoch"],
                                                             "device": str(device), "checkpoints": checkpoints})
                print(f"Published checkpoint {row['epoch']} on {device}; G rank={readout['solve']['rank']}", flush=True)
            training = model.fit(trajectories, epochs=config["cml_epochs"], progress_callback=record)
        training.update(walks_sha256=fingerprints["training_walks.json"],
                        trajectories_loaded_without_resampling=True, dimension=model.dimension,
                        source_run=str(source), device=str(device),
                        callback_time_in_elapsed_seconds=True)
        write_json(directory / "training.json", training)
        assert_unchanged()
        if fingerprint_inputs(source) != fingerprints or fingerprint_inputs(frozen) != fingerprints:
            raise RuntimeError("Training inputs changed")
        reference_run = protocol.get("historical_reference_run", protocol.get("historical_5000"))
        if reference_run is not None:
            # Historical tensors are transferred to CUDA for this numerical audit.
            reference_path = Path(reference_run) / "model.json"
            reference = load(reference_path)
            comparison = {}
            for name in ("Q", "V", "W", "G"):
                old = torch.tensor(reference[name], dtype=torch.float32, device=device)
                current = getattr(model, name)
                difference = current - old
                comparison[name] = {"max_absolute": float(difference.abs().max().item()),
                                    "rms": float(difference.square().mean().sqrt().item()),
                                    "relative_frobenius": float((difference.norm() / old.norm()).item())}
            write_json(directory / "historical_cpu_checkpoint_comparison.json", {
                "reference": str(reference_path), "reference_sha256": sha256_file(reference_path),
                "audit_device": str(device), "tensors": comparison,
                "note": "CPU-trained historical tensors; comparison performed on CUDA, no CPU training rerun"})
        publish_json(directory / "manifest.json", {"status": "completed", "device": str(device),
            "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
            "checkpoints": checkpoints, "epochs": model.epochs_completed,
            "elapsed_seconds": perf_counter() - started, "inputs_unchanged": True,
            "package_source_unchanged": True, "protocol_sha256": protocol_hash})
    except BaseException:
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        publish_json(directory / "manifest.json", {"status": "failed", "error": traceback.format_exc()})
        raise


def replay_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original completed 1000-dimensional joint-training run")
    parser.add_argument("--dimension", type=int, required=True, help="Latent dimension; other settings remain fixed unless explicitly overridden")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are refused")
    parser.add_argument("--epochs", type=int, help="Explicit epoch override; otherwise preserve source budget")
    parser.add_argument("--seed", type=int, help="Explicit training seed override; no batches are resampled")
    parser.add_argument("--device", help="Explicit cpu or cuda:N; CUDA never falls back to CPU")
    parser.add_argument("--contract", type=Path, help="Declared replay-source requirements; defaults to the historical comparison contract")
    args = parser.parse_args(argv)
    run_replay(args.source, args.dimension, args.output, epochs=args.epochs, seed=args.seed,
               device=args.device, contract=args.contract)


def checkpoints_main(argv=None):
    parser = argparse.ArgumentParser(description="Replay frozen GCML batches and publish declared CUDA checkpoints")
    parser.add_argument("--run", type=Path, required=True)
    run_checkpoints(parser.parse_args(argv).run)
