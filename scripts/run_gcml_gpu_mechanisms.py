#!/usr/bin/env python3
"""Compatibility entry; prefer cmnav experiment --operation schedule --profile conditions.

Business implementation is maintained in the cmnav package.
"""
from cmnav.workflows.condition_scheduling import main as main

if __name__ == "__main__":
    main()
