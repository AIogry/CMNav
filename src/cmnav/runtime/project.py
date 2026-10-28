"""Identify the checkout belonging to this imported package."""
from pathlib import Path


def checkout_root() -> Path:
    package = Path(__file__).resolve().parents[1]
    for candidate in package.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/cmnav").resolve() == package:
            return candidate
    raise RuntimeError("This capability requires the CMNav checkout containing the installed package")


def require_project_directory(path: Path) -> Path:
    path = Path(path).resolve()
    if not path.is_dir() or not path.is_relative_to(checkout_root()):
        raise ValueError(f"Working directory must be inside this CMNav checkout: {checkout_root()}")
    return path
