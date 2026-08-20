"""
Tests for the A100 report, which is the only thing that turns fifteen runs into a decision.

Its arithmetic is worth testing more than most code here, because a mistake in it is not a crash: it is a plausible number
that ends or continues a research programme. The tests below therefore pin the arithmetic against hand-computed values, and
pin the four possible verdicts against trees constructed to produce exactly one of them.

Two properties are checked more than once on purpose, because they are the ones a reasonable refactor would break: that a
difference is taken within seed and never between arm means, and that the seed spread and the within-checkpoint spread are
kept apart.
"""

import json
from pathlib import Path
from typing import Optional
import pytest

from direct_geometry.scripts.a100_report import (DEPLOYMENT_POINTS, SIGMAS, arm_table, main, outcome, readiness,
                                                 sampling_table, verdicts)
from direct_geometry.scripts.runs import average_contrasts, clears, collect, match_contrast, spread


ARMS = ("A", "B", "C", "D", "E")
"""The sweep's arms, in the order the report prints them."""

SEEDS = ("0", "1", "2")
"""The sweep's seeds."""


def write_run(run_dir: Path, match_rates: list[float], complete: bool = True,
              sampling: Optional[dict[str, list[float]]] = None) -> Path:
    """
    Write one synthetic finished run.

    :param run_dir:
        Directory to write.
    :type run_dir: Path
    :param match_rates:
        Validation match rates, in epoch order.
    :type match_rates: list[float]
    :param complete:
        Whether to write the COMPLETE marker.
        Defaults to True.
    :type complete: bool
    :param sampling:
        Repeated-draw match rates keyed by "<split>:<steps>", or None to write no sampling scores.
    :type sampling: Optional[dict[str, list[float]]]

    :return:
        The run directory.
    :rtype: Path
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = ["epoch,match_rate,step"]
    for index, rate in enumerate(match_rates):
        lines.append(f"{100 * (index + 1)},{rate},{index + 1}")
    (run_dir / "metrics.csv").write_text("\n".join(lines) + "\n")
    (run_dir / "COMMAND").write_text("python\n-m\nomg.main\nfit\n")
    if complete:
        (run_dir / "COMPLETE").write_text("2026-01-01T00:00:00Z\n")
    for key, draws in (sampling or {}).items():
        split, steps = key.split(":")
        mean = sum(draws) / len(draws)
        variance = sum((draw - mean) ** 2 for draw in draws) / (len(draws) - 1) if len(draws) > 1 else 0.0
        directory = run_dir / "eval" / f"{split}_nfe{steps}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "SCORE.json").write_text(json.dumps({
            "split": split, "integration_time_steps": int(steps), "match_rates": draws,
            "mean_match_rate": mean, "standard_deviation": variance ** 0.5,
            "checkpoint": "best_match_rate.ckpt"}))
    return run_dir


def write_sweep(root: Path, rates: dict[str, list[float]], sample: bool = True) -> Path:
    """
    Write a whole synthetic sweep, one best match rate per arm per seed.

    :param root:
        Run root to write under.
    :type root: Path
    :param rates:
        Best match rate per seed, keyed by arm. Each list is one value per seed, in seed order.
    :type rates: dict[str, list[float]]
    :param sample:
        Whether to write sampling scores, without which the sweep is not ready.
        Defaults to True.
    :type sample: bool

    :return:
        The launch directory.
    :rtype: Path
    """
    launch = root / "launch"
    for arm, per_seed in rates.items():
        for seed, rate in zip(SEEDS, per_seed):
            sampling = {"val:210": [rate, rate + 0.002, rate - 0.002], "test:210": [rate - 0.01],
                        "test:50": [rate - 0.05]} if sample else None
            write_run(launch / f"{arm}_seed{seed}", [rate - 0.05, rate], sampling=sampling)
    return launch


def flat(base: float = 0.25) -> dict[str, list[float]]:
    """
    Return a sweep in which every arm is identical, as the starting point for a targeted change.

    :param base:
        Match rate every arm reaches at every seed.
    :type base: float

    :return:
        Rates keyed by arm.
    :rtype: dict[str, list[float]]
    """
    return {arm: [base, base + 0.01, base - 0.01] for arm in ARMS}


def read(launch: Path) -> dict:
    """
    Read a synthetic sweep the way the report does.

    :param launch:
        The launch directory.
    :type launch: Path

    :return:
        The runs of that sweep.
    :rtype: dict
    """
    return collect(launch.parent)["launch"]


def test_a_difference_is_taken_within_seed_and_not_between_arm_means(tmp_path: Path) -> None:
    """
    A paired contrast must be the mean of within-seed differences.

    With one arm ahead by a constant on every seed, the paired mean is that constant and its standard error is zero,
    however widely the seeds themselves differ. A difference of arm means would give the same mean here but would carry
    the seed spread into its error, which is exactly the mistake the design exists to avoid.
    """
    rates = flat()
    rates["D"] = [value + 0.03 for value in rates["A"]]
    runs = read(write_sweep(tmp_path, rates))

    contrast = match_contrast(runs, "D", "A")
    assert sorted(contrast["points"]) == list(SEEDS)
    assert all(value == pytest.approx(3.0) for value in contrast["points"].values())
    assert contrast["mean_points"] == pytest.approx(3.0)
    assert contrast["standard_error"] == pytest.approx(0.0)

    spread_of_arms = spread([100.0 * value for value in rates["A"]])
    assert spread_of_arms["standard_error"] > 0.5, "the seeds must actually differ for this test to mean anything"


def test_a_main_effect_pools_the_per_seed_differences_of_both_estimates(tmp_path: Path) -> None:
    """
    The descriptor effect is estimated twice, and pooling must keep the seed as the unit.

    Three seeds seen at both levels of the graph factor is six differences, not two. Averaging the two contrast means
    instead would throw away the information that makes the standard error a statement about seeds.
    """
    rates = flat()
    rates["C"] = [value + 0.02 for value in rates["A"]]
    rates["D"] = [value + 0.02 for value in rates["B"]]
    runs = read(write_sweep(tmp_path, rates))

    effect = average_contrasts([match_contrast(runs, "C", "A"), match_contrast(runs, "D", "B")], "descriptor")
    assert effect["count"] == 6
    assert effect["mean_points"] == pytest.approx(2.0)
    assert effect["from"] == ["C-A", "D-B"]


def test_an_effect_must_clear_both_the_margin_and_the_noise() -> None:
    """
    Neither condition alone is sufficient, and the test states both failures separately.

    A large mean inside the noise is not a finding; a tiny mean outside it is not worth deploying. Both readings occur in
    practice, so both are pinned.
    """
    assert clears({"mean_points": 3.0, "standard_error": 0.5}, DEPLOYMENT_POINTS, SIGMAS) is True
    # Large but inside the noise.
    assert clears({"mean_points": 3.0, "standard_error": 2.0}, DEPLOYMENT_POINTS, SIGMAS) is False
    # Outside the noise but below the practical margin.
    assert clears({"mean_points": 0.5, "standard_error": 0.01}, DEPLOYMENT_POINTS, SIGMAS) is False
    # Too few seeds to form an error at all: unreadable, which is not the same as failing.
    assert clears({"mean_points": 3.0, "standard_error": None}, DEPLOYMENT_POINTS, SIGMAS) is None
    # An arm that changed nothing. Both bars are vacuous here when the margin is zero, as it is for the descriptor
    # effect, so this is the case strict positivity exists for.
    assert clears({"mean_points": 0.0, "standard_error": 0.0}, 0.0, SIGMAS) is False
    assert clears({"mean_points": -1.0, "standard_error": 0.0}, 0.0, SIGMAS) is False
    # Unanimous across seeds, so a zero error is real precision rather than a degenerate null.
    assert clears({"mean_points": 3.0, "standard_error": 0.0}, 0.0, SIGMAS) is True


def test_a_descriptor_that_carries_the_gain_is_reported_as_the_finding(tmp_path: Path) -> None:
    """A deployment win the descriptor accounts for is the result this pivot was designed to detect."""
    rates = flat()
    rates["C"] = [value + 0.03 for value in rates["A"]]
    rates["D"] = [value + 0.03 for value in rates["B"]]
    report = verdicts(read(write_sweep(tmp_path, rates)))

    assert report["deployment"]["passed"] is True
    assert report["descriptor_main_effect"]["passed"] is True
    name, sentence = outcome(report)
    assert name == "adapter"
    assert sentence.startswith("GO.")


def test_a_deployment_win_the_graph_explains_is_not_evidence_for_the_adapter(tmp_path: Path) -> None:
    """
    The distinction the whole factorial exists for.

    Here the graph moves the score and the descriptor does nothing. D-A clears its margin, so a design that only read D-A
    would call this a win for direct geometric features. It is not one, and the verdict has to say so in the name.
    """
    rates = flat()
    rates["B"] = [value + 0.03 for value in rates["A"]]
    rates["D"] = [value + 0.03 for value in rates["A"]]
    rates["E"] = [value + 0.025 for value in rates["A"]]
    report = verdicts(read(write_sweep(tmp_path, rates)))

    assert report["deployment"]["passed"] is True
    assert report["descriptor_main_effect"]["passed"] is False
    assert report["graph_main_effect"]["passed"] is True
    name, sentence = outcome(report)
    assert name == "graph"
    assert "node-adapter program ends here" in sentence
    # And the decomposition must say the cheap explanation accounts for most of it.
    assert report["graph_decomposition"]["length_share_of_graph_effect"] == pytest.approx(0.025 / 0.03, rel=1.0e-6)


def test_a_real_but_undeployable_descriptor_is_reported_and_not_pursued(tmp_path: Path) -> None:
    """
    A positive effect below the deployment margin is a result, and explicitly not a licence to search.

    This is the outcome most likely to invite an unregistered architecture hunt, so the sentence has to close that door.
    """
    rates = flat()
    rates["C"] = [value + 0.008 for value in rates["A"]]
    rates["D"] = [value + 0.008 for value in rates["B"]]
    report = verdicts(read(write_sweep(tmp_path, rates)))

    assert report["descriptor_main_effect"]["passed"] is True
    assert report["deployment"]["passed"] is False
    name, sentence = outcome(report)
    assert name == "descriptor-only"
    assert "do not search for an architecture" in sentence


def test_a_sweep_that_moves_nothing_stops_the_programme(tmp_path: Path) -> None:
    """An identical set of arms must produce a no-go, not an inconclusive shrug."""
    report = verdicts(read(write_sweep(tmp_path, flat())))

    assert report["deployment"]["passed"] is False
    assert report["descriptor_main_effect"]["passed"] is False
    name, sentence = outcome(report)
    assert name == "stop"
    assert sentence.startswith("NO GO.")


def test_the_two_spreads_are_reported_separately(tmp_path: Path) -> None:
    """
    The seed standard error and the within-checkpoint standard deviation must not be merged.

    Constructed so the two differ by an order of magnitude: the seeds of each arm span two points while the draws of one
    checkpoint span a fifth of one. A reader who was shown only a pooled number could not tell which of the two a
    two-point difference should be judged against.
    """
    launch = write_sweep(tmp_path, flat())
    table = sampling_table(read(launch))

    entry = table["A"]["val_nfe210"]
    assert entry["seeds"] == 3
    assert entry["draws_per_seed"] == [3, 3, 3]
    assert entry["seed_standard_error"] > 0.4
    assert entry["within_checkpoint_standard_deviation"] == pytest.approx(0.2, rel=1.0e-6)
    assert set(table["A"]) == {"val_nfe210", "test_nfe210", "test_nfe50"}


def test_the_locked_test_read_keeps_both_sampling_budgets(tmp_path: Path) -> None:
    """
    Both budgets are reported, so neither can be quoted as though it were the only one measured.

    A method that only helps at a large sampling budget is a different claim from one that helps cheaply, and the report
    has to make that visible rather than letting the better number stand alone.
    """
    table = sampling_table(read(write_sweep(tmp_path, flat(0.30))))

    assert table["A"]["test_nfe210"]["mean_match_rate"] == pytest.approx(29.0, abs=1.0e-6)
    assert table["A"]["test_nfe50"]["mean_match_rate"] == pytest.approx(25.0, abs=1.0e-6)


def test_a_partial_sweep_is_refused_rather_than_read(tmp_path: Path) -> None:
    """A verdict from a partial sweep would be a verdict on a smaller experiment that did not say so."""
    rates = flat()
    del rates["E"]
    launch = write_sweep(tmp_path, rates)
    problems = readiness(read(launch), ARMS, len(SEEDS))

    assert any("arm E has no runs" in problem for problem in problems)


def test_a_missing_seed_and_an_interrupted_run_are_both_caught(tmp_path: Path) -> None:
    """
    Two different kinds of incompleteness, both of which would otherwise pass silently.

    A missing seed shrinks the number of pairs; an interrupted run contributes a best match rate taken from fewer epochs
    than the others, which is a comparison at unequal budgets.
    """
    launch = write_sweep(tmp_path, flat())
    (launch / "E_seed2" / "COMPLETE").unlink()
    problems = readiness(read(launch), ARMS, len(SEEDS))
    assert any("E_seed2" in problem and "interrupted" in problem for problem in problems)

    import shutil
    shutil.rmtree(launch / "C_seed2")
    problems = readiness(read(launch), ARMS, len(SEEDS))
    assert any("arm C has 2 of 3 seeds" in problem for problem in problems)


def test_a_sweep_without_sampling_scores_is_not_ready(tmp_path: Path) -> None:
    """The within-checkpoint spread and the locked test numbers are part of the report, not an optional extra."""
    launch = write_sweep(tmp_path, flat(), sample=False)
    problems = readiness(read(launch), ARMS, len(SEEDS))

    assert any("no sampling scores" in problem for problem in problems)


def test_the_reported_match_rate_is_the_best_one(tmp_path: Path) -> None:
    """
    The best validation match rate is what downstream evaluation loads, so it is what the arm table shows.

    Written with a falling curve so that the best and the final differ and the wrong choice would be visible.
    """
    launch = tmp_path / "launch"
    write_run(launch / "A_seed0", [0.10, 0.30, 0.22], sampling={"val:210": [0.30]})
    table = arm_table(read(launch))

    assert table["A"]["seeds"]["0"]["best_match_rate"] == pytest.approx(0.30)
    assert table["A"]["seeds"]["0"]["final_match_rate"] == pytest.approx(0.22)
    assert table["A"]["mean"] == pytest.approx(30.0)


def test_a_finished_sweep_is_reported_and_a_partial_one_needs_asking(tmp_path: Path) -> None:
    """
    The exit code and the stamp together, because a launcher reads the first and a person reads the second.

    A partial sweep must exit nonzero without being asked, so a report step wired into a script cannot quietly produce a
    verdict from half a sweep, and must still write its stamp so the reason is recoverable.
    """
    rates = flat()
    rates["C"] = [value + 0.03 for value in rates["A"]]
    rates["D"] = [value + 0.03 for value in rates["B"]]
    launch = write_sweep(tmp_path, rates)
    out = tmp_path / "A100-REPORT.json"

    assert main(["--run-root", str(launch.parent), "--out", str(out)]) == 0
    stamp = json.loads(out.read_text())
    assert stamp["verdict"]["outcome"] == "adapter"
    assert stamp["verdict"]["provisional"] is False
    assert stamp["readiness"] == []
    assert any(name.endswith("direct_geometry/scripts/a100_report.py") for name in stamp["source_sha256"])

    import shutil
    shutil.rmtree(launch / "E_seed2")
    assert main(["--run-root", str(launch.parent), "--out", str(out)]) == 1
    assert json.loads(out.read_text())["verdict"]["provisional"] is True
    assert main(["--run-root", str(launch.parent), "--out", str(out), "--allow-partial"]) == 0
    assert json.loads(out.read_text())["verdict"]["provisional"] is True


def test_an_empty_run_root_is_refused(tmp_path: Path) -> None:
    """Nothing to read is not the same as nothing to report, and must not write a stamp saying otherwise."""
    out = tmp_path / "A100-REPORT.json"

    assert main(["--run-root", str(tmp_path / "nothing"), "--out", str(out)]) == 1
    assert not out.exists()
