"""Tests for rigid-block experiment launch guards."""

from pathlib import Path
import json
import os
import subprocess
import sys


def test_launcher_refuses_training_without_phase0_pass_stamp(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["CGFM_GATE_STAMP"] = str(tmp_path / "missing-phase0.json")
    result = subprocess.run(
        ["bash", "cgfm/scripts/launch_mpts52_blocks.sh", "task", "0"],
        cwd=repository, env=environment, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "missing Phase-0 pass stamp" in result.stderr


def test_launcher_refuses_training_without_overfit_pass_stamp(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    phase0 = tmp_path / "phase0.json"
    phase0.write_text(json.dumps({"passed": True}))
    environment = dict(os.environ)
    environment["CGFM_GATE_STAMP"] = str(phase0)
    environment["CGFM_OVERFIT_GATE_STAMP"] = str(tmp_path / "missing-phase1.json")
    result = subprocess.run(
        ["bash", "cgfm/scripts/launch_mpts52_blocks.sh", "task", "0"],
        cwd=repository, env=environment, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "missing Phase-1 overfit pass stamp" in result.stderr


def test_overfit_gate_stamps_only_a_passing_run(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    metrics = tmp_path / "metrics.csv"
    stamp = tmp_path / "phase1.json"
    metrics.write_text("epoch,match_rate\n0,0.79\n")
    command = [
        sys.executable, "-m", "cgfm.scripts.check_overfit_gate",
        "--metrics", str(metrics), "--consensus-weight", "0.1", "--stamp", str(stamp),
    ]
    failed = subprocess.run(command, cwd=repository, capture_output=True, text=True, check=False)
    assert failed.returncode != 0
    assert not stamp.exists()

    metrics.write_text("epoch,match_rate\n0,0.79\n10,0.81\n")
    passed = subprocess.run(command, cwd=repository, capture_output=True, text=True, check=False)
    assert passed.returncode == 0, passed.stderr
    payload = json.loads(stamp.read_text())
    assert payload["passed"] is True
    assert payload["best_match_rate"] == 0.81
    assert payload["consensus_weight"] == 0.1

    metrics.write_text("epoch,match_rate\n20,0.79\n")
    failed_again = subprocess.run(command, cwd=repository, capture_output=True, text=True, check=False)
    assert failed_again.returncode != 0
    assert not stamp.exists()


def test_training_completion_marker_rejects_a_shortened_run(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("epoch,loss\n0,1.0\n398,0.1\n")
    command = [
        sys.executable, "-m", "cgfm.scripts.check_training_complete",
        "--run-dir", str(tmp_path), "--expected-epochs", "400",
    ]
    failed = subprocess.run(command, cwd=repository, capture_output=True, text=True, check=False)
    assert failed.returncode != 0
    assert (tmp_path / "INCOMPLETE.json").exists()
    assert not (tmp_path / "COMPLETED.json").exists()

    metrics.write_text("epoch,loss\n0,1.0\n399,0.1\n")
    passed = subprocess.run(command, cwd=repository, capture_output=True, text=True, check=False)
    assert passed.returncode == 0, passed.stderr
    assert (tmp_path / "COMPLETED.json").exists()
    assert not (tmp_path / "INCOMPLETE.json").exists()


def test_job_resumes_latest_checkpoint_in_a_version_directory(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    checkpoint = tmp_path / "atomwise" / "seed0" / "version_2" / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    environment = dict(os.environ)
    environment["CGFM_RUN_ROOT"] = str(tmp_path)
    environment["CGFM_PYTHON_ATOMWISE"] = "/bin/echo"
    result = subprocess.run(
        ["bash", "cgfm/scripts/run_mpts52_block_job.sh", "atomwise", "0"],
        cwd=repository, env=environment, capture_output=True, text=True, check=False)
    assert result.returncode != 0  # The echo stub cannot produce completion metrics.
    assert f"Resuming atomwise seed 0 from {checkpoint}" in result.stdout
    assert f"--ckpt_path {checkpoint}" in result.stdout
