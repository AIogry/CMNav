#!/usr/bin/env python3
"""Compatibility entry; prefer cmnav experiment --operation analyze --profile mechanisms.

Business implementation is maintained in the cmnav package.
"""
from cmnav.analysis.gcml_diagnostics import mechanisms_main as main


if __name__ == "__main__":
    main()
