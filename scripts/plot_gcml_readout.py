#!/usr/bin/env python3
"""Compatibility entry; prefer cmnav experiment --operation report --profile readout.

Business implementation is maintained in the cmnav package.
"""
from cmnav.reporting.commands import readout_main as main


if __name__ == "__main__":
    main()
