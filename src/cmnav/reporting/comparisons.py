"""Publish validated comparison records, figures and text with source provenance."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cmnav.analysis.dimension_comparison import load_dimension_comparison
from cmnav.analysis.mechanism_comparison import load_study, aggregate, contrasts, exploratory_result
from cmnav.analysis.readout_curves import load_readout_curves
from cmnav.analysis.report_contracts import FIVE_TASK_COHORT
from cmnav.runtime.artifacts import sha256_file, write_json
from cmnav.analysis.records import save_json
from cmnav.reporting.comparison_figures import dimension_figures, plot_training, plot_checkpoints, plot_mechanisms, readout_figure
from cmnav.reporting.comparison_text import dimension_markdown, mechanism_markdown
from cmnav.reporting.paths import PROJECT_ROOT, bounded_output
from cmnav.reporting.provenance import report_provenance

digest = sha256_file


def dimension_report(run: Path, contract: Path | None = None, output: Path | None = None) -> Path:
    run = run.expanduser().resolve()
    output = (output or run).expanduser().resolve()
    if not output.is_relative_to(run):
        raise ValueError("Dimension report output must stay within its source run")
    if any((output / name).exists() for name in ("comparison.json", "report.md", "dimension_figures")):
        raise FileExistsError("Report outputs already exist; choose a new --output-dir")
    result, tracked, before, started = load_dimension_comparison(run, contract)
    output.mkdir(parents=True, exist_ok=True)
    result["figures"] = dimension_figures(output / "dimension_figures", result)
    after = {str(path): sha256_file(path) for path in sorted(tracked)}
    if before != after:
        raise RuntimeError("Input artifacts changed during reporting")
    result.update(input_hashes_after=after, inputs_unchanged=True, elapsed_seconds=perf_counter() - started,
                  report_script_sha256=sha256_file(Path(__file__).resolve()))
    result["implementation"] = report_provenance(Path(__file__), output)
    write_json(output / "comparison.json", result)
    destination = output / "report.md"
    destination.write_text(dimension_markdown(run, result), encoding="utf-8")
    return destination


def mechanism_report(run: Path, output: Path | None = None) -> Path:
    run = run.expanduser().resolve()
    project = PROJECT_ROOT
    if not run.is_relative_to(project / "outputs"):
        raise ValueError("Run must be inside this checkout's outputs directory")
    output = (output or run / "analysis").expanduser().resolve()
    if not output.is_relative_to(run) or output == run:
        raise ValueError("Report output must be a new subdirectory of the run")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing report: {output}")
    study = load_study(run)
    groups = [aggregate(study["rows"], condition) for condition in study["conditions"]]
    paired = contrasts(study["rows"], study["protocol"]["epochs"])
    exploratory = exploratory_result(study, groups)
    payload = {
        "schema": "cmnav.gcml_gpu_mechanism_report.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed", "run": str(run), "protocol": study["protocol"],
        "planned_jobs": study["planned_jobs"], "completed_jobs": study["completed_jobs"],
        "primary_planned_jobs": study["primary_planned_jobs"], "primary_completed_jobs": study["primary_completed_jobs"],
        "exploratory_planned_jobs": study["exploratory_planned_jobs"], "exploratory_completed_jobs": study["exploratory_completed_jobs"],
        "training_seed_count": FIVE_TASK_COHORT.training_seed_count, "unique_tasks": FIVE_TASK_COHORT.task_count,
        "independent_task_sample_count_is_not_seed_task_repetitions": True,
        "teacher_subgoals": False, "new_training_or_evaluation_performed_by_report": False,
        "training_device": study["training"]["device"], "evaluation_devices": study["evaluation_devices"],
        "cpu_cuda_bitwise_equivalence_claim": False,
        "historical_cpu_checkpoint_comparison": study["historical_cpu_checkpoint_comparison"],
        "control_audits": study["control_audits"],
        "checkpoint_diagnostics": study["checkpoints"], "aggregates": groups,
        "paired_contrasts": paired, "task_results": study["rows"], "jobs": study["jobs"],
        "exploratory_supplement": exploratory,
        "input_sha256": study["input_sha256"], "report_script_sha256": digest(Path(__file__)),
    }
    stage = Path(tempfile.mkdtemp(prefix=".gcml_mechanism_report_", dir=run))
    try:
        payload["implementation"] = report_provenance(Path(__file__), stage)
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
        plot_training(study, stage / "training_curves.png")
        plot_checkpoints(study, groups, stage / "checkpoint_success.png")
        plot_mechanisms(study, groups, stage / "mechanism_success.png")
        (stage / "comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (stage / "report.md").write_text(mechanism_markdown(study, groups, paired, output, exploratory), encoding="utf-8")
        for path, expected in study["input_sha256"].items():
            if digest(Path(path)) != expected:
                raise RuntimeError(f"Input changed while reporting: {path}")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Output appeared while reporting: {output}")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output / "report.md"


def readout_report(directory: Path, output: Path | None = None) -> Path:
    directory = directory.expanduser().resolve()
    output = (output or directory).expanduser().resolve()
    if not output.is_relative_to(directory):
        raise ValueError("Readout report output must stay within its source run")
    if any((output / name).exists() for name in ("readout_learning_curves.png", "plot_implementation.json")):
        raise FileExistsError("Readout plot already exists; choose a new --output-dir")
    data = load_readout_curves(directory)
    output.mkdir(parents=True, exist_ok=True)
    target = readout_figure(output, data)
    implementation = report_provenance(Path(__file__), output)
    after = {path: digest(Path(path)) for path in data["input_sha256"]}
    if after != data["input_sha256"]:
        raise RuntimeError("Readout inputs changed while plotting")
    implementation.update(input_hashes_before=data["input_sha256"], input_hashes_after=after,
                          inputs_unchanged=True)
    save_json(output / "plot_implementation.json", implementation)
    return target
