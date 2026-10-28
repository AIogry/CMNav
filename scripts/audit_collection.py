#!/usr/bin/env python3
"""Compatibility entry; prefer cmnav experiment --operation audit --profile collection.

Business implementation is maintained in the cmnav package.
"""
from cmnav.analysis.collection import audit_main as main


if __name__ == "__main__":
    main()
