#!/usr/bin/env python3
"""Compatibility entry; prefer cmnav experiment --operation analyze --profile failure-gates.

Business implementation is maintained in the cmnav package.
"""
from pathlib import Path
import sys

from cmnav.analysis.affordance import failure_gates_main
from cmnav.analysis.records import read_json


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    defaults = read_json(root / "configs/contracts/failure_gates_legacy.json")["arguments"]
    return failure_gates_main(["--project-root", str(root), *defaults,
                               *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
