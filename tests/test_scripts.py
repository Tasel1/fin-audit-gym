"""Tests for dataset generation, trajectory recording, and evaluation CLI scripts."""

import json
import subprocess
import sys
from pathlib import Path


def test_generate_episodes_script(tmp_path: Path):
    cmd = [
        sys.executable,
        "scripts/generate_episodes.py",
        "--output-dir",
        str(tmp_path),
        "--num-train",
        "1",
        "--num-dev",
        "1",
        "--seed-offset",
        "500",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Script failed: {res.stderr}"

    train_ep = tmp_path / "train" / "ep_001"
    assert (train_ep / "manifest.json").is_file()
    assert (train_ep / "ledger.csv").is_file()
    assert (train_ep / "invoices.jsonl").is_file()
    assert (train_ep / "payments.jsonl").is_file()
    assert (train_ep / "fx.csv").is_file()
    assert (train_ep / "truth.json").is_file()

    dev_ep = tmp_path / "dev" / "ep_001"
    assert (dev_ep / "manifest.json").is_file()


def test_record_trajectories_script(tmp_path: Path):
    # First generate 1 episode to record trajectories on
    episodes_dir = tmp_path / "data"
    gen_cmd = [
        sys.executable,
        "scripts/generate_episodes.py",
        "--output-dir",
        str(episodes_dir),
        "--num-train",
        "1",
        "--num-dev",
        "0",
        "--seed-offset",
        "600",
    ]
    assert subprocess.run(gen_cmd, check=False).returncode == 0

    traj_file = tmp_path / "trajectories.jsonl"
    record_cmd = [
        sys.executable,
        "scripts/record_trajectories.py",
        "--data-dir",
        str(episodes_dir / "train"),
        "--output-file",
        str(traj_file),
        "--policies",
        "oracle,reconciler",
    ]
    res = subprocess.run(record_cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Script failed: {res.stderr}"
    assert traj_file.is_file()

    lines = traj_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2  # At least 1 trajectory per policy
    for line in lines:
        record = json.loads(line)
        assert "policy" in record
        assert "episode_id" in record
        assert "reward" in record
        assert "trajectory" in record
        assert isinstance(record["trajectory"], list)


def test_run_eval_cli(tmp_path: Path):
    # First generate 1 episode to run evaluation on
    episodes_dir = tmp_path / "eval_data"
    gen_cmd = [
        sys.executable,
        "scripts/generate_episodes.py",
        "--output-dir",
        str(episodes_dir),
        "--num-train",
        "1",
        "--num-dev",
        "0",
        "--seed-offset",
        "700",
    ]
    assert subprocess.run(gen_cmd, check=False).returncode == 0

    eval_cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--data-dir",
        str(episodes_dir / "train"),
        "--policy",
        "oracle",
    ]
    res = subprocess.run(eval_cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Script failed: {res.stderr}"
    assert "POLICY BENCHMARK SCOREBOARD" in res.stdout
    assert "oracle" in res.stdout
