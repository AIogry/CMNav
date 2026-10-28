#!/usr/bin/env python3
"""Compatibility entry; prefer cmnav experiment --operation evaluate-gcml --profile teacher-gates.

Business implementation is maintained in the cmnav package.
"""
from cmnav.workflows.gcml_gate_comparison import main as main

if __name__ == "__main__":
    main()
