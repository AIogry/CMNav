"""Schedule declared checkpoint dependencies and conditions across CUDA devices.

Study values live in an explicit JSON profile or the run protocol. Workers call
the same package capability entrypoints used by direct experiments.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
import re
from pathlib import Path
import subprocess
import sys
import time
from cmnav.runtime.gcml_artifacts import load_json as load, publish_json as save
from cmnav.runtime.gcml_source import checkout_root
from cmnav.runtime.artifacts import sha256_file

def declare_jobs(run, protocol, profile):
    conditions = run / "conditions"
    conditions.mkdir(exist_ok=True)
    base = profile["base_overrides"]
    definitions = profile["definitions"]
    jobs, identities = [], set()

    def add(name, epoch, seed, cohort, method, variant, overrides, checkpoint):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("Condition names must be safe single path components")
        identity = (name, epoch, seed)
        if identity in identities:
            return
        identities.add(identity)
        identifier = f"e{epoch:03d}_{name}_s{seed}"
        condition = {"name": name, "epoch": epoch, "planning_seed": seed, "cohort": cohort,
                     "method": method, "checkpoint_variant": variant, "overrides": overrides}
        path = conditions / f"{identifier}.json"
        save(path, condition)
        jobs.append({"id": identifier, "condition_file": str(path), "checkpoint": str(checkpoint),
                     "output": str(run / "evaluations" / identifier), "cohort": cohort, "epoch": epoch})

    for epoch in protocol["checkpoints"]:
        for name in protocol["checkpoint_conditions"]:
            definition = definitions[name]
            method, variant, overrides = (definition[k] for k in ("method", "variant", "overrides"))
            filename = "model.pt" if variant == "raw" else f"model_{variant}.pt"
            add(name, epoch, protocol["planning_seeds"][0], "checkpoint", method, variant, {**base, **overrides},
                run / "training" / "checkpoints" / f"epoch_{epoch:03d}" / filename)
    for name in protocol["final_conditions"]:
        for seed in protocol["planning_seeds"]:
            definition = definitions[name]
            method, variant, overrides = (definition[k] for k in ("method", "variant", "overrides"))
            filename = "model.pt" if variant == "raw" else f"model_{variant}.pt"
            add(name, protocol["epochs"], seed, "final", method, variant, {**base, **overrides},
                run / "training" / "checkpoints" / f"epoch_{protocol['epochs']:03d}" / filename)
    controls = profile.get("historical_controls")
    if controls:
        for method in controls["methods"]:
            add(f"historical_{method}_cuda", controls["epoch"], controls["planning_seed"], "control", method, "historical", {},
                Path(protocol[controls["source_field"]]) / "model.json")
    save(run / "jobs.json", jobs)
    return jobs


def run_schedule(run: Path, profile_path: Path | None = None):
    run = run.resolve()
    protocol = load(run / "protocol.json")
    project = checkout_root()
    if profile_path is None:
        declared = protocol.get("condition_profile")
        profile_path = run / declared if declared is not None else project / "configs/experiments/gcml_gpu_conditions.json"
    profile_path = profile_path.expanduser().resolve()
    profile = load(profile_path)
    profile_hash = sha256_file(profile_path)
    protocol_hash = sha256_file(run / "protocol.json")
    jobs = load(run / "jobs.json") if (run / "jobs.json").exists() else declare_jobs(run, protocol, profile)
    jobs_hash = sha256_file(run / "jobs.json")
    profile_record = {"path": str(profile_path), "sha256": profile_hash, "profile": profile}
    saved_profile = run / "scheduler_profile.json"
    if saved_profile.exists() and load(saved_profile) != profile_record:
        raise RuntimeError("Existing scheduler profile differs; use a new run for changed conditions")
    if not saved_profile.exists():
        save(saved_profile, profile_record)
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONUNBUFFERED="1", MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet")
    # Never mask/reorder CUDA_VISIBLE_DEVICES: device IDs refer to physical GPUs 0 and 1.
    if environment.get("CUDA_VISIBLE_DEVICES") not in profile["allowed_cuda_visible_devices"]:
        raise RuntimeError("Unexpected CUDA_VISIBLE_DEVICES mapping")
    train, train_log, active = None, None, {}
    try:
        train = None
        train_log = None
        training_manifest = run / "training" / "manifest.json"
        if not training_manifest.exists():
            train_log = (run / "training_process.log").open("x", encoding="utf-8")
            train = subprocess.Popen([sys.executable, "-m", "cmnav", "experiment", "--operation", "train-gcml", "--profile", "checkpoints", "--", "--run", str(run)],
                                     cwd=project, env=environment, stdout=train_log, stderr=subprocess.STDOUT)
        elif load(training_manifest).get("status") != "completed":
            raise RuntimeError("Existing training is incomplete; inspect it rather than silently retraining")
        active = {}
        completed, failed = set(), {}
        for job in jobs:
            summary = Path(job["output"]) / "summary.json"
            if summary.exists():
                status = load(summary).get("status")
                if status == "completed":
                    completed.add(job["id"])
                else:
                    raise RuntimeError(f"Existing job {job['id']} is {status}; retain and inspect before retrying")
        (run / "evaluations").mkdir(exist_ok=True)
        (run / "process_logs").mkdir(exist_ok=True)
        while True:
            if (sha256_file(profile_path) != profile_hash or sha256_file(run / "protocol.json") != protocol_hash
                    or sha256_file(run / "jobs.json") != jobs_hash):
                raise RuntimeError("Declared scheduler profile, protocol or jobs changed during execution")
            if train is not None and train.poll() is not None:
                train_log.close()
                if train.returncode != 0:
                    failed["training"] = train.returncode
                train = None
            for device, (process, stream, job) in list(active.items()):
                if process.poll() is None:
                    continue
                stream.close()
                if process.returncode == 0:
                    completed.add(job["id"])
                    print(f"Completed {job['id']} on {device} ({len(completed)}/{len(jobs)})", flush=True)
                else:
                    failed[job["id"]] = process.returncode
                    print(f"Failed {job['id']} on {device}, exit={process.returncode}", flush=True)
                del active[device]
            training_done = training_manifest.exists() and load(training_manifest).get("status") == "completed"
            if not failed:
                for device in protocol["evaluation_devices"]:
                    if device in active or (device == protocol["training_device"] and not training_done):
                        continue
                    scheduled = {record[2]["id"] for record in active.values()}
                    for job in jobs:
                        if job["id"] in completed or job["id"] in scheduled:
                            continue
                        checkpoint = Path(job["checkpoint"])
                        ready = checkpoint.exists() and (job["cohort"] == "control" or
                                ((checkpoint.parent / "manifest.json").exists() and
                                 load(checkpoint.parent / "manifest.json")["status"] == "completed"))
                        if not ready:
                            continue
                        stream = (run / "process_logs" / f"{job['id']}.log").open("x", encoding="utf-8")
                        process = subprocess.Popen([sys.executable, "-m", "cmnav", "experiment", "--operation", "evaluate-gcml", "--profile", "condition", "--",
                            "--run", str(run), "--checkpoint", str(checkpoint), "--condition", job["condition_file"],
                            "--device", device, "--output", job["output"]], cwd=project, env=environment,
                            stdout=stream, stderr=subprocess.STDOUT)
                        active[device] = (process, stream, job)
                        print(f"Started {job['id']} on {device}", flush=True)
                        break
            save(run / "run_state.json", {"status": "failed" if failed else "completed" if len(completed) == len(jobs) and training_done else "running",
                "updated_at": datetime.now(timezone.utc).isoformat(), "training_pid": train.pid if train else None,
                "training_completed": training_done, "completed_jobs": len(completed), "total_jobs": len(jobs),
                "active": {device: {"job": record[2]["id"], "pid": record[0].pid} for device, record in active.items()},
                "failed": failed})
            if not active and train is None and (failed or len(completed) == len(jobs)):
                break
            time.sleep(2)
        if failed:
            raise RuntimeError(f"Study failed; inspect retained logs: {failed}")
    finally:
        # Stop only children created by this scheduler if it is interrupted or fails.
        children = [(p, stream) for p, stream, _ in active.values()]
        if train is not None:
            children.append((train, train_log))
        for process, stream in children:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            if stream is not None:
                stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--condition-profile", type=Path, help="Declared condition definitions and historical controls")
    args = parser.parse_args(argv)
    run_schedule(args.run, args.condition_profile)
