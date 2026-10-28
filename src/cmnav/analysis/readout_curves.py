"""Load saved fixed-Q learning curves without training or evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from cmnav.analysis.records import read_json, digest


def load_readout_curves(directory: Path) -> dict[str, Any]:
    inputs = {str((directory / name).resolve()): digest(directory / name)
              for name in ("manifest.json", "initial_metrics.json", "readout_epochs.jsonl")}
    manifest = read_json(directory / "manifest.json")
    if manifest["status"] != "completed":
        raise ValueError("Wait for the readout experiment to complete")
    initial = read_json(directory / "initial_metrics.json")["metrics"]
    groups = {}
    with (directory / "readout_epochs.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            groups.setdefault(row["variant"], []).append(row)
    rates = manifest["learning_rates"]
    for mode in ("sources_only", "sources_plus_sinks"):
        for rate in rates:
            name = f"{mode}_lr_{rate:g}"
            groups[name] = sorted(groups[name], key=lambda row: row["epoch"])
            if len(groups[name]) != manifest["epochs_per_iterative_condition"]:
                raise ValueError(f"Incomplete curve: {name}")
            for row in groups[name]:
                if any(row["metrics"][scope]["threshold"] != initial[scope]["threshold"]
                       for scope in ("sources", "sinks")):
                    raise ValueError(f"Readout curve changes classification threshold: {name}")
    return {"manifest": manifest, "initial": initial, "groups": groups, "rates": rates,
            "input_sha256": inputs}
