#!/usr/bin/env python3
"""Compatibility entry; prefer cmnav experiment --operation analyze --profile training.

Business implementation is maintained in the cmnav package.
"""
from pathlib import Path
import sys

from cmnav.analysis.training_comparison import main as comparison_main


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    return comparison_main(["--project-root", str(root), "--focus-contract",
                            str(root / "configs/contracts/training_comparison_focus_legacy.json"),
                            *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
