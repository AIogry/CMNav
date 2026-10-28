#!/usr/bin/env python3
"""Compatibility entry; prefer cmnav experiment --operation fit-gcml-readout --profile reference.

Business implementation is maintained in the cmnav package.
"""
from cmnav.workflows.gcml_readout import reference_main

if __name__ == "__main__":
    reference_main()
