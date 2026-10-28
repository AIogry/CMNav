#!/usr/bin/env python3
"""Compatibility entry; prefer cmnav experiment --operation analyze --profile full-goal.

Business implementation is maintained in the cmnav package.
"""
from pathlib import Path
import sys

from cmnav.analysis.gcml_diagnostics import full_goal_failures_main
from cmnav.analysis.records import read_json


def main(argv=None):
    contract = Path(__file__).resolve().parents[1] / "configs/contracts/full_goal_failure_legacy.json"
    defaults = read_json(contract)["arguments"]
    return full_goal_failures_main([*defaults, *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
