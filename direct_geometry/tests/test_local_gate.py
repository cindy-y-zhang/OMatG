"""
Tests for Gate DG3 and its stamp.

WHY THIS FILE EXISTS

This gate is the only thing standing between a local screen and eight A100s, and it is read once, at the end, when the
runs it grades have already cost a day. A gate that passes when it should fail is worse than no gate, because it launders
a null result into an authorisation. So every threshold gets a test that moves one number across it and nothing else.

The fixtures build the metrics files rather than training anything, because what is under test is the arithmetic and the
refusals, not the model. Whether the runs themselves are trustworthy is what the COMPLETE marker and the hashes in the
stamp are for, and those are tested here too.
"""

import csv
import json
import shutil
from pathlib import Path
from typing import Optional
import pytest

from direct_geometry.scripts.local_gate import (CELL_TOLERANCE, COST_CEILING, MEMORISATION_FLOOR, MSE_IMPROVEMENT,
                                                PAIRED_GAIN_POINTS, attribution, collect, cost_verdict, error_contrast,
                                                gate_failures, main, match_contrast, read_metrics)


TIMES = ("0.1", "0.3", "0.5", "0.7", "0.9", "0.99")
"""The time buckets the denoising error is reported in, as the evaluator labels them."""


def write_run(run_dir: Path, match_rates: list[float], pos: Optional[float] = None, cell: Optional[float] = None,
              complete: bool = True, identity_failures: Optional[list[str]] = None) -> Path:
    """
    Write one run's artefacts: a metrics CSV, optionally a target-error report, optionally a COMPLETE marker.

    :param run_dir:
        Directory to write into.
    :type run_dir: Path
    :param match_rates:
        Match rate of each validation, in order.
    :type match_rates: list[float]
    :param pos:
        Coordinate error to report in every time bucket, or None to write no error report.
    :type pos: Optional[float]
    :param cell:
        Cell error to report in every time bucket. Defaults to the coordinate error.
    :type cell: Optional[float]
    :param complete:
        Whether to write the COMPLETE marker.
        Defaults to True.
    :type complete: bool
    :param identity_failures:
        Identity failures to record in the error report, or None for none.
    :type identity_failures: Optional[list]

    :return:
        The directory written to.
    :rtype: Path
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "match_rate"])
        for epoch, rate in enumerate(match_rates):
            writer.writerow([epoch, rate])
    if pos is not None:
        errors = {label: {field: {"mean_squared_error": value, "relative_error": 0.5, "identity_holds": True}
                          for field, value in (("pos", pos), ("cell", cell if cell is not None else pos))}
                  for label in TIMES}
        (run_dir / "TARGET-MSE.json").write_text(json.dumps(
            {"errors": errors, "identity_failures": identity_failures or []}))
    if complete:
        (run_dir / "COMPLETE").write_text("2026-01-01T00:00:00Z\n")
    (run_dir / "COMMAND").write_text("python -m omg.main fit\n")
    return run_dir


def write_audit(path: Path, worst: float = 1.05, passed: bool = True) -> Path:
    """
    Write a DG0 stamp with a chosen worst cost ratio.

    :param path:
        Where to write it.
    :type path: Path
    :param worst:
        Worst cost ratio to record.
        Defaults to 1.05, comfortably inside the ceiling.
    :type worst: float
    :param passed:
        Whether the audit itself passed.
        Defaults to True.
    :type passed: bool

    :return:
        The path written.
    :rtype: Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"passed": passed, "failures": [] if passed else ["something"],
                                "costs": {"fc/none": {"time_ratio": 1.0, "memory_ratio": 1.0},
                                          "periodic_distance/radial": {"time_ratio": worst,
                                                                       "memory_ratio": 1.0}}}))
    return path


def passing_tree(root: Path, paired_gain: float = 0.05, pos_ratio: float = 1.0 - 2 * MSE_IMPROVEMENT,
                 cell_ratio: float = 1.0, memorisation: float = 0.9) -> Path:
    """
    Build a run tree that Gate DG3 should pass, so that each test can spoil exactly one thing.

    :param root:
        Run root to build under.
    :type root: Path
    :param paired_gain:
        Match-rate gain of D over A in each paired seed, as a fraction.
        Defaults to 0.05, which is five points against a required two.
    :type paired_gain: float
    :param pos_ratio:
        Ratio of D's coordinate error to A's.
        Defaults to twice the required improvement.
    :type pos_ratio: float
    :param cell_ratio:
        Ratio of D's cell error to A's.
        Defaults to 1.0, unchanged.
    :type cell_ratio: float
    :param memorisation:
        Match rate every arm reaches on the memorisation check.
        Defaults to 0.9, above the floor.
    :type memorisation: float

    :return:
        The run root.
    :rtype: Path
    """
    for arm in ("A", "B", "C", "D", "E"):
        write_run(root / "memorise" / f"{arm}_seed0", [0.1, memorisation])
    # The 100-epoch screen. C and D beat their controls on both match rate and coordinate error, which is what the
    # descriptor condition asks for.
    write_run(root / "screen" / "A_seed0", [0.05, 0.10], pos=1.0)
    write_run(root / "screen" / "B_seed0", [0.05, 0.11], pos=0.99)
    write_run(root / "screen" / "C_seed0", [0.05, 0.11], pos=0.95)
    write_run(root / "screen" / "D_seed0", [0.05, 0.12], pos=0.94)
    write_run(root / "screen" / "E_seed0", [0.05, 0.105], pos=0.98)
    for seed in ("0", "1"):
        write_run(root / "paired" / f"A_seed{seed}", [0.10, 0.20], pos=1.0, cell=1.0)
        write_run(root / "paired" / f"D_seed{seed}", [0.10, 0.20 + paired_gain],
                  pos=pos_ratio, cell=cell_ratio)
    return root


def failures_of(root: Path, audit: Path) -> list[str]:
    """
    Read the gate over a tree and return its failure list.

    :param root:
        Run root.
    :type root: Path
    :param audit:
        DG0 stamp.
    :type audit: Path

    :return:
        Failure descriptions.
    :rtype: list[str]
    """
    runs = collect(root)
    screen, paired = runs.get("screen", {}), runs.get("paired", {})
    contrasts = {
        "paired": {"match": match_contrast(paired, "D", "A"),
                   "pos": error_contrast(paired, "D", "A", "pos"),
                   "cell": error_contrast(paired, "D", "A", "cell")},
        "screen": {"descriptor": [{"match": match_contrast(screen, better, worse),
                                   "pos": error_contrast(screen, better, worse, "pos")}
                                  for better, worse in (("C", "A"), ("D", "B"))],
                   "attribution": attribution(screen)},
    }
    return gate_failures(runs, contrasts, cost_verdict(audit))


def test_the_gate_passes_a_tree_that_should_pass(tmp_path: Path) -> None:
    """
    The fixture is a pass, so every failure below is attributable to the one thing that test changed.

    Without this the suite could not distinguish a threshold that works from a gate that rejects everything.
    """
    assert failures_of(passing_tree(tmp_path / "runs"), write_audit(tmp_path / "DG0.json")) == []


def test_the_gate_fails_an_arm_that_cannot_memorise(tmp_path: Path) -> None:
    """
    An arm below the memorisation floor is disqualified, however good its screen looks.

    This is the only condition in the gate that is about the arm's ability to optimise at all rather than about a
    contrast, and it has to come first: a contrast between two models that did not converge is not a contrast.
    """
    root = passing_tree(tmp_path / "runs", memorisation=MEMORISATION_FLOOR - 0.01)
    failures = failures_of(root, write_audit(tmp_path / "DG0.json"))
    assert any("memorisation" in failure for failure in failures)


def test_the_gate_fails_a_paired_gain_below_the_threshold(tmp_path: Path) -> None:
    """
    A D-A mean under the predeclared points requirement fails, even though it is positive.

    A positive-but-small gain is the most likely outcome and the most tempting to accept, which is exactly why the
    threshold was fixed in advance and is asserted here.
    """
    root = passing_tree(tmp_path / "runs", paired_gain=(PAIRED_GAIN_POINTS - 0.5) / 100.0)
    failures = failures_of(root, write_audit(tmp_path / "DG0.json"))
    assert any("averages" in failure and "match-rate points" in failure for failure in failures)


def test_the_gate_fails_when_one_seed_is_negative_even_if_the_mean_clears(tmp_path: Path) -> None:
    """
    A mean carried by a single seed is not a reproducible effect.

    Constructed so the mean is comfortably above the threshold while one seed is behind, which is the case a mean-only
    reading would wave through.
    """
    root = passing_tree(tmp_path / "runs")
    # Seed 0 gains five points, seed 1 loses half a point, so the mean is 2.25 against a threshold of 2.
    write_run(root / "paired" / "D_seed1", [0.10, 0.195], pos=0.9, cell=1.0)
    failures = failures_of(root, write_audit(tmp_path / "DG0.json"))
    assert any("negative on seed" in failure for failure in failures)
    # The mean really is above the threshold, so the seed condition is what rejected it and not the mean condition.
    match = match_contrast(collect(root)["paired"], "D", "A")
    assert match["mean_points"] > PAIRED_GAIN_POINTS
    assert not any("averages" in failure for failure in failures)


def test_the_gate_fails_a_match_rate_gain_with_no_denoising_gain(tmp_path: Path) -> None:
    """
    A match-rate win without a coordinate-error win does not pass.

    Match rate is thresholded and sampled, so it is the noisier of the two measurements; requiring the quieter one to
    agree is what separates a real gain from a lucky draw.
    """
    root = passing_tree(tmp_path / "runs", pos_ratio=1.0 - MSE_IMPROVEMENT / 2.0)
    failures = failures_of(root, write_audit(tmp_path / "DG0.json"))
    assert any("coordinate target error improved" in failure for failure in failures)


def test_the_gate_fails_a_cell_regression_the_loss_would_not_show(tmp_path: Path) -> None:
    """
    A cell error that worsens past tolerance fails.

    The cell carries about six ten-thousandths of the loss weight, so a large cell regression is invisible in the
    training objective. That is the reason this is a separate explicit condition rather than something assumed.
    """
    root = passing_tree(tmp_path / "runs", cell_ratio=1.0 + 2 * CELL_TOLERANCE)
    failures = failures_of(root, write_audit(tmp_path / "DG0.json"))
    assert any("cell target error worsened" in failure for failure in failures)


def test_the_gate_fails_when_the_descriptor_itself_never_helps(tmp_path: Path) -> None:
    """
    A D-A win driven entirely by the graph is not a direct-feature result.

    Built so that D beats A on the paired runs while both pure descriptor contrasts, C-A and D-B, are flat or behind.
    This is the misattribution the plan singles out, and it is the one a single headline number would hide.
    """
    root = passing_tree(tmp_path / "runs")
    write_run(root / "screen" / "C_seed0", [0.05, 0.09], pos=1.02)
    write_run(root / "screen" / "D_seed0", [0.05, 0.10], pos=1.01)
    write_run(root / "screen" / "B_seed0", [0.05, 0.11], pos=0.99)
    failures = failures_of(root, write_audit(tmp_path / "DG0.json"))
    assert any("no pure descriptor contrast" in failure for failure in failures)


def test_the_gate_fails_a_cost_over_the_ceiling(tmp_path: Path) -> None:
    """
    The DG0 ceiling is re-read here, so a code change between the audit and the runs cannot slip through.
    """
    root = passing_tree(tmp_path / "runs")
    failures = failures_of(root, write_audit(tmp_path / "DG0.json", worst=COST_CEILING + 0.01))
    assert any("cost ceiling" in failure for failure in failures)


def test_the_gate_fails_a_failed_audit_even_at_an_affordable_cost(tmp_path: Path) -> None:
    """
    DG0 can fail for reasons that are not cost -- a dead channel, an isolated atom, a graph that is secretly the control.

    Reading only the ratios would let those through, so the audit's own verdict is required as well as its numbers.
    """
    root = passing_tree(tmp_path / "runs")
    failures = failures_of(root, write_audit(tmp_path / "DG0.json", worst=1.01, passed=False))
    assert any("cost ceiling" in failure for failure in failures)


def test_the_gate_fails_an_unmeasured_denoising_error(tmp_path: Path) -> None:
    """
    A missing target-error report is a failure, not a silently skipped condition.

    Two of the gate's conditions are stated on that error. If absence were tolerated, the easiest way to pass would be
    not to measure it.
    """
    root = passing_tree(tmp_path / "runs")
    for seed in ("0", "1"):
        for arm in ("A", "D"):
            (root / "paired" / f"{arm}_seed{seed}" / "TARGET-MSE.json").unlink()
    failures = failures_of(root, write_audit(tmp_path / "DG0.json"))
    assert any("was not measured" in failure for failure in failures)


def test_the_gate_fails_a_broken_target_reconstruction(tmp_path: Path) -> None:
    """
    If the evaluator's own identity check failed, its numbers are not usable and the gate must say so.

    The target velocity is recovered by differentiating OMatG's loss. That reconstruction is verified against
    ``MSE = loss + mean(target^2)``; when it does not hold, every error in the report is of an unknown quantity.
    """
    root = passing_tree(tmp_path / "runs")
    write_run(root / "paired" / "D_seed0", [0.10, 0.25], pos=0.9, cell=1.0,
              identity_failures=["0.5/pos"])
    failures = failures_of(root, write_audit(tmp_path / "DG0.json"))
    assert any("identity check" in failure for failure in failures)


def test_the_gate_fails_an_interrupted_run(tmp_path: Path) -> None:
    """
    A run with metrics but no COMPLETE marker was cut short, and grading it grades a shorter experiment.

    Easy to hit in practice: one card, sequential runs, and a launch that is stopped to free the GPU.
    """
    root = passing_tree(tmp_path / "runs")
    (root / "paired" / "D_seed1" / "COMPLETE").unlink()
    failures = failures_of(root, write_audit(tmp_path / "DG0.json"))
    assert any("COMPLETE marker" in failure for failure in failures)


def test_the_gate_fails_when_a_seed_has_only_one_arm_of_the_pair(tmp_path: Path) -> None:
    """
    An unpaired seed cannot enter a paired difference and must not be quietly dropped.

    Dropping it would compute the mean over whichever seeds happened to finish, which is a different and unstated
    experiment.
    """
    root = passing_tree(tmp_path / "runs")
    write_run(root / "paired" / "D_seed2", [0.10, 0.30], pos=0.9, cell=1.0)
    failures = failures_of(root, write_audit(tmp_path / "DG0.json"))
    assert any("only one arm of the D-A pair" in failure for failure in failures)


def test_the_paired_difference_is_taken_within_seed(tmp_path: Path) -> None:
    """
    Differences are formed inside each seed and then averaged, not by averaging the arms first.

    On this task the seed-to-seed spread is larger than the effect being looked for, so the two orders are not
    interchangeable: a difference of means would be dominated by which seeds each arm happened to get.
    """
    root = tmp_path / "runs"
    write_run(root / "paired" / "A_seed0", [0.10])
    write_run(root / "paired" / "D_seed0", [0.14])
    write_run(root / "paired" / "A_seed1", [0.30])
    write_run(root / "paired" / "D_seed1", [0.33])
    match = match_contrast(collect(root)["paired"], "D", "A")
    assert match["points"] == pytest.approx({"0": 4.0, "1": 3.0})
    assert match["mean_points"] == pytest.approx(3.5)


def test_the_reported_match_rate_is_the_best_one_and_the_final_one_is_kept(tmp_path: Path) -> None:
    """
    The gate reads the best validation match rate, which is the checkpoint downstream evaluation loads, and records the
    final one beside it.

    Reading the last value of a noisy rising curve would grade a single draw; reading the best without recording the last
    would hide that it is an upper order statistic.
    """
    entry = read_metrics(write_run(tmp_path / "run", [0.10, 0.25, 0.18]))
    assert entry["best_match_rate"] == pytest.approx(0.25)
    assert entry["best_epoch"] == pytest.approx(1.0)
    assert entry["final_match_rate"] == pytest.approx(0.18)
    assert entry["validations"] == 3


def test_arm_e_decomposes_the_graph_effect_and_never_gates_it(tmp_path: Path) -> None:
    """
    Arm E splits B-A into a length channel and a topology, and appears in no threshold.

    Arm E was added after the gate's thresholds were fixed. Letting it into the arithmetic would be moving a predeclared
    line to fit a later idea, so this asserts both halves: that the decomposition is reported, and that a tree which
    passes still passes when E is absent entirely.
    """
    root = passing_tree(tmp_path / "runs")
    report = attribution(collect(root)["screen"])
    # B is 1.0 point over A and E is 0.5, so the length channel accounts for half of the graph factor.
    assert report["graph_B_minus_A"]["mean_points"] == pytest.approx(1.0)
    assert report["length_channel_E_minus_A"]["mean_points"] == pytest.approx(0.5)
    assert report["topology_B_minus_E"]["mean_points"] == pytest.approx(0.5)
    assert report["length_share_of_graph_effect"] == pytest.approx(0.5)

    audit = write_audit(tmp_path / "DG0.json")
    assert failures_of(root, audit) == []
    shutil.rmtree(root / "screen" / "E_seed0")
    shutil.rmtree(root / "memorise" / "E_seed0")
    assert failures_of(root, audit) == []


def test_the_stamp_records_the_hashes_and_commands_a_bundle_would_be_traced_to(tmp_path: Path) -> None:
    """
    The stamp carries source, data and metrics hashes, the resolved commands, and the verdict.

    The packager refuses to run without a passing stamp, so the stamp is the only link between a tar file and the
    evidence that justified building it. If it did not pin the sources it graded, that link would not mean anything.
    """
    root = passing_tree(tmp_path / "runs")
    audit = write_audit(tmp_path / "DG0.json")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.lmdb").write_bytes(b"not really an lmdb")
    destination = tmp_path / "LOCAL-GATE.json"

    code = main(["--run-root", str(root), "--audit-report", str(audit), "--data-dir", str(data_dir),
                 "--probe-report", str(tmp_path / "absent.json"), "--out", str(destination)])
    assert code == 0
    stamp = json.loads(destination.read_text())
    assert stamp["verdict"]["passed"] is True
    assert stamp["thresholds"]["paired_gain_points"] == PAIRED_GAIN_POINTS
    assert str(data_dir / "train.lmdb") in stamp["data_sha256"]
    assert any(name.endswith("direct_geometry/scripts/local_gate.py") for name in stamp["source_sha256"])
    assert "paired/D_seed0" in stamp["commands"]
    assert stamp["runs"]["paired"]["D_seed0"]["metrics_sha256"]


def test_a_superseded_check_cannot_be_read_as_the_current_one(tmp_path: Path) -> None:
    """
    Runs from an earlier version of a check must be visible and unread.

    This is not hypothetical. The memorisation check was rebuilt once: its first version stopped while the curve was still
    climbing and validated with a sampler four times cheaper than the reported one, and every arm including the baseline
    missed the floor as a result. Those runs are worth keeping as the evidence for why the check changed, and reading them
    would gate the sweep on a measurement nobody intends to make. Passing the floor here while the stale directory holds
    failing numbers is the property that makes the two impossible to confuse.
    """
    root = passing_tree(tmp_path / "runs")
    for arm in ("A", "B", "D", "E"):
        write_run(root / "overfit" / f"{arm}_seed0", [0.03, 0.53])
    audit = write_audit(tmp_path / "DG0.json")
    destination = tmp_path / "LOCAL-GATE.json"

    code = main(["--run-root", str(root), "--audit-report", str(audit), "--data-dir", str(tmp_path),
                 "--probe-report", str(tmp_path / "absent.json"), "--out", str(destination)])
    assert code == 0, "the stale directory's failing numbers must not reach the floor"
    stamp = json.loads(destination.read_text())
    assert stamp["verdict"]["passed"] is True
    assert "overfit" in stamp["modes_declined"]
    assert "overfit" not in stamp["runs"]
    assert stamp["modes_read"] == ["memorise", "screen", "paired"]


def test_a_failing_gate_returns_nonzero_and_still_writes_its_stamp(tmp_path: Path) -> None:
    """
    A failure exits nonzero, so a launcher chaining onto it stops, and still writes the stamp so the reasons survive.

    Writing nothing on failure would mean the only record of why the pivot stopped was a terminal that has since scrolled.
    """
    root = passing_tree(tmp_path / "runs", paired_gain=-0.01)
    destination = tmp_path / "LOCAL-GATE.json"
    code = main(["--run-root", str(root), "--audit-report", str(write_audit(tmp_path / "DG0.json")),
                 "--data-dir", str(tmp_path), "--probe-report", str(tmp_path / "absent.json"),
                 "--out", str(destination)])
    assert code == 1
    stamp = json.loads(destination.read_text())
    assert stamp["verdict"]["passed"] is False
    assert stamp["verdict"]["failures"]


def test_an_empty_run_root_is_refused_rather_than_passed(tmp_path: Path) -> None:
    """
    No runs is not a pass. The most likely way to reach this gate with nothing is a launcher path typo.
    """
    assert main(["--run-root", str(tmp_path / "absent"), "--out", str(tmp_path / "stamp.json")]) == 1
