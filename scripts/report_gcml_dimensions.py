#!/usr/bin/env python3
"""Compatibility entry; prefer cmnav experiment --operation report --profile dimensions.

Business implementation is maintained in the cmnav package.
"""
from cmnav.reporting.commands import dimensions_main as main


if __name__ == "__main__":
    main()
