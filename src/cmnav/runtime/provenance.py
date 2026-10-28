"""Runtime versions and fingerprints of the full installed CMNav source package."""
from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

from cmnav import __version__
from cmnav.runtime.artifacts import sha256_file


def runtime_versions() -> dict[str, Any]:
    """Report interpreter and dependency versions without importing Habitat."""
    packages = {}
    for name in ("cmnav", "numpy", "torch", "Pillow", "PyYAML", "habitat-lab", "habitat-sim", "hydra-core"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"cmnav_version": __version__, "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": packages}


def source_hashes() -> dict[str, str]:
    """Hash all Python source files relative to the complete CMNav package.

    The root is the package containing this module, not the caller's workflow
    directory. This preserves package-wide coverage after moving workflows into
    subpackages; paths remain relative to ``cmnav`` as in historical records.
    """
    package_root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(package_root)): sha256_file(path)
            for path in sorted(package_root.rglob("*.py"))}
