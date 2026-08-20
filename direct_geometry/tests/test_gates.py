"""
Tests for the gate reader, whose one job is to be right about what authorises spending eight A100s.

The case that motivated it is the third test below: DG2 records ``passed: false`` when the angular channels fail to earn
their place, which is a completed selection and the reason the promoted descriptor is ``radial`` rather than ``both``. A
generic ``passed`` lookup reads that as a failed gate and refuses to build a bundle precisely because a factor was
correctly dropped. Both directions of that mistake are pinned here: a correctly dropped factor must not block, and a
genuinely failed DG1 must.
"""

import json
from pathlib import Path
import pytest

from direct_geometry.scripts.gates import authorisation, main


def write_audit(reports: Path, passed: bool = True) -> None:
    """
    Write a DG0 stamp.

    :param reports:
        Directory to write into.
    :type reports: Path
    :param passed:
        Whether the audit passed.
        Defaults to True.
    :type passed: bool
    """
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "DG0-AUDIT.json").write_text(json.dumps({
        "passed": passed, "failures": [] if passed else ["arm D costs 1.62x the control's step time"],
        "costs": {"A": {"time_ratio": 1.0, "memory_ratio": 1.0},
                  "D": {"time_ratio": 1.16 if passed else 1.62, "memory_ratio": 1.10}}}))


def write_probes(reports: Path, dg1: bool = True, dg2: bool = False,
                 promoted: str = "radial") -> None:
    """
    Write a DG1/DG2 stamp.

    :param reports:
        Directory to write into.
    :type reports: Path
    :param dg1:
        Whether the descriptor beat the chemistry-only control.
        Defaults to True.
    :type dg1: bool
    :param dg2:
        Whether the angular channels earned their place.
        Defaults to False, which is what the real probes found.
    :type dg2: bool
    :param promoted:
        Descriptor that was promoted.
        Defaults to "radial".
    :type promoted: str
    """
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "DG1-DG2-PROBES.json").write_text(json.dumps({
        "verdict": {"DG1": {"passed": dg1, "accuracy_points": 26.1, "bits": 0.81},
                    "DG2": {"passed": dg2, "shape_points": 2.7, "bits": 0.21},
                    "promoted": promoted if dg1 else None}}))


def write_local_gate(reports: Path, passed: bool = True) -> None:
    """
    Write a DG3 stamp.

    :param reports:
        Directory to write into.
    :type reports: Path
    :param passed:
        Whether the local screen passed.
        Defaults to True.
    :type passed: bool
    """
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "LOCAL-GATE.json").write_text(json.dumps({
        "verdict": {"passed": passed, "failures": [] if passed else ["paired D-A averages -0.40 points"]},
        "contrasts": {"paired": {"match": {"mean_points": 2.6 if passed else -0.4}}}}))


def test_all_three_stamps_present_and_sound_authorise_the_sweep(tmp_path: Path) -> None:
    """The ordinary case, and the only one that should let fifteen runs start."""
    write_audit(tmp_path)
    write_probes(tmp_path)
    write_local_gate(tmp_path)
    report = authorisation(tmp_path, ("dg0", "probes", "dg3"))

    assert report["authorised"] is True
    assert report["problems"] == []
    assert report["promoted"] == "radial"


def test_dropping_the_angular_channels_is_a_decision_and_not_a_failure(tmp_path: Path) -> None:
    """
    The bug this module exists to prevent.

    DG2 records ``passed: false`` when angular features do not earn their place. That is the finding, and it is why the
    promoted descriptor is ``radial``. Refusing to launch on it would refuse because the experiment worked.
    """
    write_audit(tmp_path)
    write_probes(tmp_path, dg1=True, dg2=False, promoted="radial")
    write_local_gate(tmp_path)
    report = authorisation(tmp_path, ("dg0", "probes", "dg3"))

    assert report["authorised"] is True
    assert report["status"]["probes"]["angular_kept"] is False
    assert "angular channels dropped" in report["status"]["probes"]["summary"]
    assert "promoted 'radial'" in report["status"]["probes"]["summary"]


def test_keeping_the_angular_channels_promotes_both_and_still_authorises(tmp_path: Path) -> None:
    """The other branch of the same selection, so the reader is not merely insensitive to DG2."""
    write_audit(tmp_path)
    write_probes(tmp_path, dg1=True, dg2=True, promoted="both")
    write_local_gate(tmp_path)
    report = authorisation(tmp_path, ("dg0", "probes", "dg3"))

    assert report["authorised"] is True
    assert report["promoted"] == "both"
    assert report["status"]["probes"]["angular_kept"] is True
    assert "angular channels kept" in report["status"]["probes"]["summary"]


def test_a_failed_dg1_stops_the_pivot(tmp_path: Path) -> None:
    """
    DG1 is a bar, unlike DG2.

    If the descriptor carries nothing a chemistry-only control does not, there is nothing to inject and no arm C or D to
    define, so this must block however good the other stamps look.
    """
    write_audit(tmp_path)
    write_probes(tmp_path, dg1=False)
    write_local_gate(tmp_path)
    report = authorisation(tmp_path, ("dg0", "probes", "dg3"))

    assert report["authorised"] is False
    assert any("DG1 failed" in problem for problem in report["problems"])
    assert any("no descriptor was promoted" in problem for problem in report["problems"])


def test_a_failed_audit_blocks_and_says_what_broke(tmp_path: Path) -> None:
    """A cost breach means the arms are not comparable at equal budget, so nothing downstream is interpretable."""
    write_audit(tmp_path, passed=False)
    write_probes(tmp_path)
    write_local_gate(tmp_path)
    report = authorisation(tmp_path, ("dg0", "probes", "dg3"))

    assert report["authorised"] is False
    assert any("1.62x" in problem for problem in report["problems"])


def test_a_failed_local_gate_blocks_and_carries_its_reason(tmp_path: Path) -> None:
    """DG3 is the bar on spending, and its own reason has to reach the person who tried to launch."""
    write_audit(tmp_path)
    write_probes(tmp_path)
    write_local_gate(tmp_path, passed=False)
    report = authorisation(tmp_path, ("dg0", "probes", "dg3"))

    assert report["authorised"] is False
    assert any("-0.40 points" in problem for problem in report["problems"])


def test_a_gate_that_is_not_required_is_reported_and_not_enforced(tmp_path: Path) -> None:
    """
    What makes one reader serve both an informational status read and an enforcing one.

    `verify` prints the status of all three before DG3 exists; `launch` requires it. Without this distinction the two
    would need separate parsers, which is how they would come to disagree.
    """
    write_audit(tmp_path)
    write_probes(tmp_path)
    report = authorisation(tmp_path, ("dg0", "probes"))

    assert report["authorised"] is True
    assert report["status"]["dg3"]["present"] is False
    assert report["status"]["dg3"]["passed"] is False


def test_a_missing_required_stamp_blocks(tmp_path: Path) -> None:
    """The same absence, now required, must stop a launch rather than be read as a pass."""
    write_audit(tmp_path)
    write_probes(tmp_path)
    report = authorisation(tmp_path, ("dg0", "probes", "dg3"))

    assert report["authorised"] is False
    assert any("LOCAL-GATE.json is missing" in problem for problem in report["problems"])


def test_a_corrupt_stamp_is_refused_rather_than_skipped(tmp_path: Path) -> None:
    """
    A truncated JSON file is the likely result of an interrupted write or a partial transfer.

    Silently treating it as absent would be survivable; silently treating it as a pass would not, so it is reported as a
    failure with the parse error attached.
    """
    write_audit(tmp_path)
    write_probes(tmp_path)
    (tmp_path / "LOCAL-GATE.json").write_text('{"verdict": {"passed": tru')
    report = authorisation(tmp_path, ("dg0", "probes", "dg3"))

    assert report["authorised"] is False
    assert any("could not be read" in problem for problem in report["problems"])


def test_the_promoted_descriptor_is_printed_for_a_launcher_to_read(tmp_path: Path, capsys) -> None:
    """The launchers read this on standard output, so it has to be the bare word and nothing else."""
    write_audit(tmp_path)
    write_probes(tmp_path)

    assert main(["--reports", str(tmp_path), "--promoted"]) == 0
    assert capsys.readouterr().out.strip() == "radial"


def test_an_undefined_descriptor_is_refused_rather_than_defaulted(tmp_path: Path, capsys) -> None:
    """
    A launcher that read an empty descriptor would train arms C and D on whatever the config defaulted to.

    That run would look successful and would be answering a question nobody asked, so the reader exits nonzero and says
    which command produces the missing stamp.
    """
    assert main(["--reports", str(tmp_path), "--promoted"]) == 1
    assert "probe_features" in capsys.readouterr().out


def test_an_unknown_gate_name_is_an_error(tmp_path: Path) -> None:
    """A typo in a launcher's --require list must not silently require nothing."""
    with pytest.raises(SystemExit):
        main(["--reports", str(tmp_path), "--require", "dg0,dg9"])


def test_the_status_read_exits_zero_when_nothing_is_required(tmp_path: Path) -> None:
    """`verify` prints the status early in a project's life, when DG3 legitimately does not exist yet."""
    write_audit(tmp_path)
    write_probes(tmp_path)

    assert main(["--reports", str(tmp_path)]) == 0
    assert main(["--reports", str(tmp_path), "--require", "dg3"]) == 1
