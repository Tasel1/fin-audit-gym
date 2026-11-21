"""Sanity test verifying environment package structure."""

import fin_audit_gym


def test_package_version():
    assert fin_audit_gym.__version__ == "0.1.0"


def test_isolated_submodule_imports():
    """Verify that all submodules can be imported cleanly without circular dependency errors."""
    import subprocess
    import sys

    for mod in [
        "fin_audit_gym.generator",
        "fin_audit_gym.evaluator",
        "fin_audit_gym.core",
        "fin_audit_gym.baselines",
        "fin_audit_gym.utils",
    ]:
        res = subprocess.run(
            [sys.executable, "-c", f"import {mod}"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, f"Failed to import {mod}: {res.stderr}"
