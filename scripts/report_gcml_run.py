#!/usr/bin/env python3
"""Compatibility entry; prefer cmnav experiment --operation report --profile gcml.

Business implementation is maintained in the cmnav package.
"""
from cmnav.reporting.commands import gcml_main as main


if __name__ == "__main__":
    main()
