"""Sanity test verifying environment package structure."""

import fin_audit_gym


def test_package_version():
    assert fin_audit_gym.__version__ == "0.1.0"
