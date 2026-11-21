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
            timeout=30,
        )
        assert res.returncode == 0, f"Failed to import {mod}: {res.stderr}"


def test_env_initial_attributes_before_reset():
    """Verify that FinAuditEnv initializes all state attributes on construction before reset()."""
    from fin_audit_gym.core.env import FinAuditEnv

    env = FinAuditEnv()
    assert env._steps_taken == 0
    assert env._total_queries == 0
    assert env._terminated is False
    assert env._truncated is False
    assert env._world is None
    assert env._ground_truth_findings == []
    assert env.reward_engine is not None
