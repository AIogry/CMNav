#!/usr/bin/env python3
"""Compatibility entry; prefer cmnav experiment --operation evaluate-gcml --profile full-goal.

Business implementation is maintained in the cmnav package.
"""
from cmnav.workflows.gcml_evaluation import full_goal_main as main

if __name__ == "__main__":
    main()
