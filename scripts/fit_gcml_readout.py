#!/usr/bin/env python3
"""Compatibility entry; prefer cmnav experiment --operation fit-gcml-readout --profile frozen.

Business implementation is maintained in the cmnav package.
"""
from pathlib import Path
from cmnav.workflows.gcml_readout import legacy_main

if __name__ == "__main__":
    legacy_main(Path(__file__).resolve().parents[1] / "configs/contracts/gcml_readout_legacy_defaults.json")
