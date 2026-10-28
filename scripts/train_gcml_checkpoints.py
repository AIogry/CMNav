#!/usr/bin/env python3
"""Compatibility entry; prefer cmnav experiment --operation train-gcml --profile checkpoints.

Business implementation is maintained in the cmnav package.
"""
from cmnav.workflows.gcml_training import checkpoints_main as main

if __name__ == "__main__":
    main()
