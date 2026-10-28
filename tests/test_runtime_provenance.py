"""Migration boundaries for package-wide provenance and lightweight imports."""
from __future__ import annotations

import hashlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from cmnav.runtime import provenance


def test_source_hashes_cover_entire_package_with_package_relative_keys(tmp_path, monkeypatch):
    package = tmp_path / "cmnav"
    sources = {
        "__init__.py": b'__version__ = "test"\n',
        "methods/cml.py": b"# pure learner\n",
        "configuration/environment.py": b"# configuration\n",
        "runtime/provenance.py": b"# fingerprint source\n",
        "workflows/oracle_graph_navigation.py": b"# workflow\n",
    }
    for name, contents in sources.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    (package / "model.json").write_text("{}", encoding="utf-8")
    (tmp_path / "outside.py").write_text("# unrelated", encoding="utf-8")
    monkeypatch.setattr(provenance, "__file__", str(package / "runtime/provenance.py"))

    assert provenance.source_hashes() == {
        name: hashlib.sha256(contents).hexdigest() for name, contents in sources.items()
    }


@pytest.mark.parametrize(
    "modules,blocked",
    [
        (("cmnav.configuration", "cmnav.configuration.validation",
          "cmnav.runtime.artifacts", "cmnav.runtime.provenance"),
         ("habitat", "habitat_sim", "numpy", "PIL", "yaml", "torch")),
        (("cmnav.configuration.environment", "cmnav.configuration.exploration",
          "cmnav.configuration.graph_navigation"),
         ("habitat", "habitat_sim", "numpy", "PIL", "torch")),
    ],
)
def test_configuration_and_provenance_import_without_simulator(tmp_path, modules, blocked):
    source_root = Path(__file__).resolve().parents[1] / "src"
    program = textwrap.dedent(f"""
        import importlib
        import importlib.abc
        import sys

        sys.path.insert(0, {str(source_root)!r})
        class RejectHeavyImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {blocked!r}:
                    raise AssertionError('Unexpected eager dependency: ' + fullname)
                return None

        sys.meta_path.insert(0, RejectHeavyImports())
        for module in {modules!r}:
            importlib.import_module(module)
        assert not set({blocked!r}).intersection(sys.modules)
    """)
    result = subprocess.run([sys.executable, "-c", program], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
