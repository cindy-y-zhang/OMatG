"""Tests for the Phase-0 G1/G2 gate command."""

import json
import sys
import pytest
from cgfm.scripts.check_readout_gates import _load, main
from cgfm.scripts.readout_ceiling import G1_FREE_MATCH_RATE, G2_TOLERANCE_POINTS


def _summary(rate: float) -> dict:
    """Build a ceiling JSON payload with the G1 field the gate command reads."""
    return {"g1": {"free_match_rate_comparison": rate, "passed": rate >= G1_FREE_MATCH_RATE}}


def test_gate_constants_match_the_plan():
    assert G1_FREE_MATCH_RATE == 0.90
    assert G2_TOLERANCE_POINTS == 3.0


def test_check_readout_gates_loads_a_ceiling_summary(tmp_path):
    path = tmp_path / "coarse.json"
    path.write_text(json.dumps(_summary(0.91)))
    assert _load(path)["g1"]["free_match_rate_comparison"] == 0.91


def test_gate_command_writes_launch_stamp_only_after_both_gates_pass(tmp_path, monkeypatch):
    coarse = tmp_path / "coarse.json"
    fine = tmp_path / "fine.json"
    stamp = tmp_path / "phase0.json"
    coarse.write_text(json.dumps(_summary(0.91)))
    fine.write_text(json.dumps(_summary(0.92)))
    monkeypatch.setattr(sys, "argv", [
        "check_readout_gates", "--coarse", str(coarse), "--fine", str(fine), "--stamp", str(stamp)])
    main()
    payload = json.loads(stamp.read_text())
    assert payload["passed"] is True
    assert len(payload["coarse_sha256"]) == len(payload["fine_sha256"]) == 64

    coarse.write_text(json.dumps(_summary(0.89)))
    with pytest.raises(SystemExit):
        main()
    assert not stamp.exists()
