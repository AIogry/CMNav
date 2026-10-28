# Stable tools and compatibility entrypoints

Updated: 2026-10-27. Business implementations live in `src/cmnav/`; a new
dimension, seed, checkpoint, noise level or output run does not need a new script.

## Preferred experiment entry

From the CMNav checkout, using the environment belonging to this worktree:

```bash
.env/bin/python -m cmnav experiment --list
.env/bin/python -m cmnav experiment --config configs/experiments/train_gcml_3000.yaml --describe
.env/bin/python -m cmnav experiment --config configs/experiments/train_gcml_5000.yaml --describe
```

`--describe` only resolves and displays configuration. It does not import a
model, train, evaluate, create an output directory or reserve a GPU. The two
examples use the same replay training implementation with different parameters.
Before an actual run, choose a new output path; existing training outputs are
refused. To execute a GPU configuration, use `.env-gpu/bin/python` and omit
`--describe`. CUDA availability is required when CUDA is requested.

Direct invocation and operation-specific help also use the same registry:

```bash
.env/bin/python -m cmnav experiment --operation train-gcml --profile replay -- --help
.env/bin/python -m cmnav experiment --operation report --profile mechanisms -- --help
```

The JSON/YAML schema contains `schema_version: 1`, `operation`, `profile`,
`working_directory`, `options` and optional `positional`. The working directory
is resolved relative to the configuration file; option paths are then resolved
in that working directory. Options use long flag names without `--`; lists
produce multiple arguments, `true` adds a flag and `false`/`null` omits it. Only
registered operations can be selected. Configurations do not import modules or
execute shell fragments.

Each actual dispatch records the resolved invocation, configuration and package
source snapshot under `outputs/experiment_invocations/<run_id>/`. The snapshot
includes Python sources, packaging metadata and public JSON/YAML definitions in
`configs/contracts/` and `configs/experiments/` (excluding `*.local.*`). The operation
still owns its training/evaluation/report output; the invocation record is a
source record, not a claim that navigation succeeded. Existing workflow CLI
commands such as `diagnose-gcml-navigation` remain available.

## Retained independent tools

- `prepare_mp3d_data.py`: MP3D assets and official task preparation.
- `download_test_data.py`: separate public smoke-test asset preparation.
- `build_supported_navigation_tasks.py`: thin task-manifest preparation entry.

## Compatibility map

The following filenames remain so old command lines and local callers can be
migrated gradually. They contain no training, evaluation or report business
implementation. Historical results continue to identify their archived original
source, not these new delegates.

| Old script | `experiment --operation` | `--profile` |
| --- | --- | --- |
| `train_gcml_dimension.py` | `train-gcml` | `replay` |
| `train_gcml_checkpoints.py` | `train-gcml` | `checkpoints` |
| `evaluate_gcml_dimension.py` | `evaluate-gcml` | `full-goal` |
| `evaluate_gcml_mechanisms.py` | `evaluate-gcml` | `condition` |
| `compare_gcml_gates.py` | `evaluate-gcml` | `teacher-gates` |
| `run_gcml_gpu_mechanisms.py` | `schedule` | `conditions` |
| `fit_gcml_readout.py` | `fit-gcml-readout` | `frozen` |
| `add_gcml_minimum_change_reference.py` | `fit-gcml-readout` | `reference` |
| `analyze_gcml_mechanisms.py` | `analyze` | `mechanisms` |
| `analyze_gcml_affordance_scores.py` | `analyze` | `affordances` |
| `analyze_gcml_full_goal_failures.py` | `analyze` | `full-goal` |
| `audit_gcml_failure_gates.py` | `analyze` | `failure-gates` |
| `summarize_training_comparison.py` | `analyze` | `training` |
| `summarize_collection_quality.py` | `analyze` | `collection` |
| `audit_collection.py` | `audit` | `collection` |
| `plot_navigation_run.py` | `report` | `navigation` |
| `report_gcml_run.py` | `report` | `gcml` |
| `report_gcml_subgoals.py` | `report` | `subgoals` |
| `report_gcml_dimensions.py` | `report` | `dimensions` |
| `report_gcml_gpu_mechanisms.py` | `report` | `mechanisms` |
| `plot_gcml_readout.py` | `report` | `readout` |

Profiles preserve distinct input formats and permissions. In particular,
`teacher-gates` uses teacher subgoals; `condition` and `checkpoints` require their
declared CUDA study protocol. Historical cohort contracts and compatibility
defaults are explicit JSON data under `configs/contracts/` and
`configs/experiments/`. A contract-specific report must not be used to claim an
unvalidated cohort is equivalent.

New reports refuse to replace existing report artifacts. Use a new
`--output-dir` for navigation, GCML, subgoal, dimension and readout reports;
the mechanisms profile uses `--output`. Some profiles require this directory
to remain inside the source run. Their `--help` describes the exact constraint.
Legacy analysis commands with an explicit `--overwrite` remain available;
new work must preserve the original observations and conclusions.

The frozen-readout profile compares source-only and sink-supervised fitting,
so it requires sink states and an explicit experience contract. The subgoal
and mechanism report profiles retain contracted historical layouts. Unified
dispatch does not turn these into unrestricted readers for arbitrary studies.

## Admission rule

Add a configuration for a new experiment. Add a package capability only for a
new algorithm, statistic, input format or independent responsibility, and
reuse shared loading, execution and provenance. Do not import one script from
another, or import `scripts` from `cmnav`. One-off explorations belong in an
identified temporary location or a run's `scratch/`, with retained-source status
recorded. Independent data preparation tools may remain in this directory.

Full local rules: `docs/代码结构与命名规范.md` section 11,
`docs/Agent执行规则说明.md`, and `docs/实验与结果记录规范.md`.

## Landmark hierarchy capabilities

The existing experiment dispatcher provides `hierarchy` with `build`, `train`,
`evaluate`, and `report` profiles. Use the corresponding
`configs/experiments/{build,train,evaluate,report}_landmark_hierarchy.yaml` files and
the shared `landmark_hierarchy_protocol.json`. No new script is required.

Each execution creates a unique output directory. Training and evaluation
configurations name their structure and checkpoint inputs explicitly; update
these references when choosing a new run. The current protocol operates on
abstract graph nodes, freezes the low model, and uses fitted G readouts. It
does not execute Habitat. Details and results are in
[Landmark 层次 GCML 原型](../docs/10-26/Landmark层次GCML原型.md).

The report profile reads a complete saved cohort and its compressed traces;
it does not rerun policies or change the original run status. Its source input
may be a run whose task execution finished but whose report export failed.

Hierarchy evaluation accepts optional `local_noise_scale`, `high_noise_scale`
and `flat_noise_scale` overrides; omitted/null values inherit `noise_scale`.
The local override covers entry, exit and macro-execution queries. The flat
override covers full-goal and direct-goal queries, including the short-horizon
direct candidate in hierarchical evaluation. Every query records its actual
noise role and scale. These roles do not use oracle shortest-path distance.

The `hierarchy/compare-noise` profile reads a saved complete 2×2 study via
`--study PATH --output-root PATH`; it does not execute policies. The study lists
each condition's `local_noise`, `long_noise`, and explicit per-method source
runs. It verifies completed runs, frozen parameters, identical task identities,
structure, package source and non-noise protocol before comparison. Unchanged
methods can reuse a source only when their relevant noise settings match.
The first condition is the paired baseline. See the completed
[noise comparison](../docs/10-27/层次GCML噪声分档对照.md) for configuration and
artifact references.
