# CMNav prototype workspaces

Updated: 2026-10-22 (Asia/Shanghai).

## Branches and ownership

| Branch | Checkout | Purpose |
| --- | --- | --- |
| `main` | No dedicated checkout required | Reviewed integration baseline |
| `proto/pipeline` | `/home/eai/Navigation/CMNav` | Early CML/GCML navigation pipeline, training, baselines and evaluation |
| `proto/data` | `/home/eai/Navigation/CMNav/.worktrees/proto-data` | Research data generation, collection quality and dataset delivery |

Work in your own checkout and commit on its branch. Do not switch another
agent's checkout or edit its source files. The nested worktree directory is
ignored by the parent checkout. Git branches share repository metadata, but
have separate checked-out source and staging areas.

## Data is the primary deliverable from proto/data

The data branch may develop independently; merging its entire implementation
into `main` or `proto/pipeline` is not required. Preserve its source commits
and configuration so each published dataset remains reproducible. Reuse
individual implementation commits later only when useful and reviewed.

Keep a stable, versioned delivery contract:

- Continuous RGB-D/action records, actual successor observations and reset boundaries.
- Dataset version, scene/task split, collection seed/configuration, source commit
  and dependency snapshot.
- Permission/privileged-supervision declaration, integrity checks and quality report.
- A documented reader or adapter when changing the existing trajectory schema.
- Frozen dataset directories that consumers read without modifying in place.

Independent implementations are acceptable. Undocumented changes to units,
coordinate conventions, sensor/action settings or record semantics are not
interoperable dataset deliveries.

## Data and outputs

- Pipeline data: `/home/eai/Navigation/CMNav/data`.
- Data-agent data: `/home/eai/Navigation/CMNav/.worktrees/proto-data/data`.
- Each checkout writes its own `outputs/` and unique run directories.
- Existing scene assets and official task files may be shared as read-only
  inputs through absolute paths in ignored `configs/*.local.yaml` files.
- Do not symlink the entire `data/` directory to the other checkout.
- Keep the data checkout's `project_root` pointing to its own root. The current
  collection implementation requires output and plan paths under that root's
  `data/`; this supports the independent output directory above.

For existing assets, the data checkout can use these absolute input paths:

```text
scenes_dir: /home/eai/Navigation/CMNav/data/scene_datasets
episodes_path: /home/eai/Navigation/CMNav/data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz
scene_dataset_config: /home/eai/Navigation/CMNav/data/scene_datasets/mp3d/mp3d.scene_dataset_config.json
```

Both source branches can consume a frozen dataset by its manifest and path;
Git merging is not necessary to transfer data. Worktrees do not isolate GPU
or disk throughput, so coordinate resource-intensive jobs separately.

## Independent environments

The pipeline uses `/home/eai/Navigation/CMNav/.env`. The data checkout uses
`/home/eai/Navigation/CMNav/.worktrees/proto-data/.env` and can manage its own
dependencies. Do not install or upgrade packages in the other checkout's env.

To create the data environment once, when its target does not already exist:

```bash
conda create --prefix /home/eai/Navigation/CMNav/.worktrees/proto-data/.env \
  --clone /home/eai/Navigation/CMNav/.env --copy

/home/eai/Navigation/CMNav/.worktrees/proto-data/.env/bin/python -m pip install \
  --no-deps --no-build-isolation -e /home/eai/Navigation/CMNav/.worktrees/proto-data
```

Use Conda's cloning mechanism instead of a plain directory copy. Reinstalling
the editable project is required because the original installation references
the pipeline checkout's `src/`. Run commands using the environment belonging
to the checkout, preferably its explicit `.env/bin/python` path.

Record dependency changes with each dataset release. Separate environments
allow different package versions; reproducibility still requires recording
the environment used to generate each artifact.

## Local research documents

The existing `.gitignore` excludes `docs/`; that choice is preserved. These
documents remain local and are not automatically copied by Git worktrees.
Read the handoff document from:

`/home/eai/Navigation/CMNav/docs/10-22/数据采集说明.md`

The original research discussion is at:

`/home/eai/Navigation/docs/10-21-v2.md`

The latest user instructions and this workspace arrangement supersede the
handoff document's earlier assumption that both workstreams should closely
share implementation. They still share a documented data delivery contract.
