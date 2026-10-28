#!/usr/bin/env python3
"""Compatibility entry; prefer cmnav experiment --operation analyze --profile affordances.

Business implementation is maintained in the cmnav package.
"""
from cmnav.analysis.affordance import affordance_scores_main as main


if __name__ == "__main__":
    main()
