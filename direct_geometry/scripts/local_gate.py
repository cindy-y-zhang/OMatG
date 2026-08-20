"""
Gate DG3: the verdict of the local screen, and the stamp the A100 packager demands.

WHAT THIS GATE IS FOR

The baseline reaches 26.82 per cent match rate at 1200 epochs. A 200-epoch number is therefore a point on a rising curve,
and two arms that are close at 200 epochs cannot be ranked from it. This gate is deliberately blunt: it can catch an arm
that fails to optimise, whose channels interfere, whose topology hurts, or whose cost breaks the ceiling, and it authorises
spending eight A100s. It supports no claim of its own.

That is also why it refuses rather than warns. Every threshold below was fixed before any of these runs existed, and the
stamp records the hashes of the sources, the data and the metrics it read, so that a later bundle cannot be traced back to
a passing verdict that was produced from different code.

WHICH MATCH RATE IS READ

The best validation match rate over the run, not the last. Two reasons: it is the checkpoint that downstream evaluation
actually loads, so gating on anything else would grade a model nobody uses; and the last validation of a noisy rising
curve is a single draw. The maximum of several draws is biased upwards, but every arm has an identical validation cadence
and an identical number of draws, so the bias is common to both sides of every contrast reported here and largely cancels
in the difference. The final value is recorded alongside for anyone who wants the unbiased-but-noisier reading.

WHY THE DECOMPOSITION IS WIDER THAN THE GATE

Arm E was added after the thresholds were set. It appears in the report and never in a threshold. Moving a predeclared
gate to accommodate a later idea is how a screen becomes a search, so E's contrasts are recorded as attribution -- E-A is
the edge length channel alone and B-E is the periodic topology net of it -- and the pass/fail arithmetic is exactly what
was written down in advance.

Usage:

    python -m direct_geometry.scripts.local_gate
    python -m direct_geometry.scripts.local_gate --run-root direct_geometry/runs --out reports/LOCAL-GATE.json
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import torch

# Re-exported rather than redefined. The A100 report reads the same runs with the same arithmetic, and two definitions of
# "which match rate is read" would eventually disagree about the same directory.
from .runs import (MSE_IMPROVEMENT, arm_of, collect, commit, digest, error_contrast, match_contrast, read_metrics,
                   read_target_mse, seed_of, tree_digest, unread_modes)


MODES = ("memorise", "screen", "paired")
"""
The run modes this gate reads, named rather than discovered.

Reading whatever directories happen to exist is how a superseded result gets read as a current one. The memorisation check
was rebuilt once already -- its first version stopped mid-ascent and sampled at a quarter of the production budget -- and its
old runs are still on disk as the evidence for why. They are answers to a different question, so they sit under a different
name and this gate declines them by construction. Anything present and unread is reported in the stamp.
"""

MEMORISATION_FLOOR = 0.80
"""
Match rate every arm must reach on the 100-structure memorisation check.

Training and validation are the same hundred structures there, so anything below this is a failure to optimise and cannot
be anything else. An arm that cannot memorise a hundred crystals will not learn fifty thousand.
"""

PAIRED_GAIN_POINTS = 2.0
"""
Match-rate points the paired D-A difference must average over seeds.

Two points on a 26.82 baseline is roughly a tenth of the score, which is the smallest gain worth eight A100s. Predeclared.
"""

CELL_TOLERANCE = 0.02
"""
Fractional worsening of cell error tolerated.

The descriptor is a function of positions and the cell, so it can in principle trade one against the other. The cell
carries almost no loss weight, so a large cell regression would not show up in training loss at all -- hence an explicit
bound rather than trust.
"""

COST_CEILING = 1.30
"""
Cost ceiling from Gate DG0, re-checked here so a code change between the audit and the runs cannot slip past.
"""

SOURCES = ("direct_geometry", "cgfm/configs/atomwise_mpts52.yaml", "omg")
"""
Paths whose contents are hashed into the stamp.

The encoder, the launcher, the configs and OMatG itself: everything that determines what these numbers mean.
"""


def cost_verdict(audit_path: Path) -> dict:
    """
    Re-read Gate DG0's cost table and restate its verdict.

    Read from the audit's own stamp rather than re-measured, so the two cannot disagree, but restated here because DG3
    names the ceiling as one of its conditions and a stamp that only referred to another file would not be self-contained.

    :param audit_path:
        Path of the DG0 stamp.
    :type audit_path: Path

    :return:
        Worst observed ratios, the ceiling, whether it holds, and the stamp's digest.
    :rtype: dict
    """
    if not audit_path.is_file():
        return {"passed": False, "reason": f"no DG0 stamp at {audit_path}"}
    report = json.loads(audit_path.read_text())
    costs = report.get("costs", {})
    ratios = {arm: {"time": entry.get("time_ratio"), "memory": entry.get("memory_ratio")}
              for arm, entry in costs.items()}
    # A NaN memory ratio is the CPU case, where peak allocation is not reported. Dropped from the maximum rather than
    # compared against the ceiling, matching the audit's own handling; passing an unmeasured budget silently is how a
    # memory regression ships.
    measured = [value for entry in ratios.values() for value in entry.values()
                if value is not None and value == value]
    if not measured:
        return {"ceiling": COST_CEILING, "ratios": ratios, "worst_ratio": None, "passed": False,
                "reason": f"the DG0 stamp at {audit_path} records no cost ratios",
                "audit_sha256": digest(audit_path)}
    return {"ceiling": COST_CEILING, "ratios": ratios, "worst_ratio": max(measured),
            "passed": max(measured) <= COST_CEILING and bool(report.get("passed")),
            "audit_passed": report.get("passed"), "audit_failures": report.get("failures", []),
            "audit_sha256": digest(audit_path)}


def gate_failures(runs: dict[str, dict[str, dict]], contrasts: dict, cost: dict) -> list[str]:
    """
    Return every reason Gate DG3 fails, or an empty list if it passes.

    Every condition here is the one written down before the runs existed. Collected as a list rather than short-circuited,
    so that one launch reports every problem instead of one per re-read.

    :param runs:
        Runs keyed by mode and run name.
    :type runs: dict[str, dict[str, dict]]
    :param contrasts:
        The computed contrasts, as assembled by ``main``.
    :type contrasts: dict
    :param cost:
        The cost verdict.
    :type cost: dict

    :return:
        Failure descriptions.
    :rtype: list[str]
    """
    failures: list[str] = []

    memorise = runs.get("memorise", {})
    if not memorise:
        failures.append("no memorisation runs: an arm that cannot memorise a hundred crystals must be disqualified "
                        "before it is screened, so this check is not optional.")
    for name, entry in sorted(memorise.items()):
        if entry["best_match_rate"] < MEMORISATION_FLOOR:
            failures.append(f"memorisation: arm {arm_of(name)} reached {entry['best_match_rate']:.1%} on the hundred "
                            f"structures it was also validated on, below the {MEMORISATION_FLOOR:.0%} floor. That is a "
                            f"failure to optimise rather than a weak result.")

    paired = contrasts.get("paired", {})
    match = paired.get("match", {})
    if not match.get("points"):
        failures.append("no paired D-A runs: the gate's headline condition is a paired difference and cannot be read "
                        "from unpaired runs.")
    else:
        mean = match["mean_points"]
        if mean < PAIRED_GAIN_POINTS:
            failures.append(f"paired D-A averages {mean:+.2f} match-rate points over seeds {match['seeds']}, below the "
                            f"required +{PAIRED_GAIN_POINTS:.0f}.")
        negative = {seed: value for seed, value in match["points"].items() if value < 0.0}
        if negative:
            failures.append(f"paired D-A is negative on seed(s) {sorted(negative)} "
                            f"({', '.join(f'{s}: {v:+.2f}' for s, v in sorted(negative.items()))}). The gate requires "
                            f"every seed nonnegative, because a mean carried by one seed is not a reproducible effect.")
        if match["unpaired_seeds"]:
            failures.append(f"seed(s) {match['unpaired_seeds']} have only one arm of the D-A pair, so they cannot enter "
                            f"a paired difference.")

    position = paired.get("pos", {})
    if not position.get("total_buckets"):
        failures.append("paired coordinate target error was not measured: run "
                        "'direct_geometry.scripts.target_mse' for the paired runs before reading this gate.")
    elif position["improved_buckets"] * 2 <= position["total_buckets"]:
        failures.append(f"coordinate target error improved by at least {MSE_IMPROVEMENT:.0%} in "
                        f"{position['improved_buckets']} of {position['total_buckets']} time buckets, which is not a "
                        f"majority. A match-rate gain without a denoising gain is most likely sampling luck.")

    cell = paired.get("cell", {})
    regressions = {label: value for label, value in cell.get("fractional_change", {}).items()
                   if value > CELL_TOLERANCE}
    if regressions:
        failures.append(f"cell target error worsened by more than {CELL_TOLERANCE:.0%} at "
                        f"{', '.join(f'{k} ({v:+.1%})' for k, v in sorted(regressions.items()))}. The cell carries "
                        f"almost no loss weight, so this would not have shown up in the training loss.")

    descriptor = contrasts.get("screen", {}).get("descriptor", [])
    if not descriptor:
        failures.append("neither pure descriptor contrast (C-A or D-B) could be formed from the 100-epoch screen.")
    else:
        supported = [entry for entry in descriptor
                     if entry["match"]["mean_points"] is not None and entry["match"]["mean_points"] >= 0.0
                     and entry["pos"]["total_buckets"] and entry["pos"]["improved_buckets"] > 0]
        if not supported:
            summary = "; ".join(
                f"{entry['match']['contrast']} {entry['match']['mean_points']:+.2f} points, "
                f"{entry['pos']['improved_buckets']}/{entry['pos']['total_buckets']} buckets improved"
                for entry in descriptor if entry["match"]["mean_points"] is not None)
            failures.append(f"no pure descriptor contrast is both nonnegative in match rate and improved in coordinate "
                            f"target error at 100 epochs ({summary}). The descriptor is the thing this pivot is about, "
                            f"so a D-A win without it is not the result being tested for.")

    if not cost["passed"]:
        failures.append(f"the DG0 cost ceiling of {COST_CEILING:.0%} no longer holds: worst ratio "
                        f"{cost.get('worst_ratio')}. {cost.get('reason', '')}".strip())

    identity = [f"{mode}/{name}" for mode, entries in runs.items() for name, entry in entries.items()
                if entry.get("identity_failures")]
    if identity:
        failures.append(f"the target reconstruction failed its own identity check in {', '.join(sorted(identity))}, so "
                        f"those denoising errors cannot be trusted and the gate cannot be read.")

    incomplete = [f"{mode}/{name}" for mode, entries in runs.items() for name, entry in entries.items()
                  if not entry["complete"]]
    if incomplete:
        failures.append(f"run(s) {', '.join(sorted(incomplete))} have metrics but no COMPLETE marker, so they were "
                        f"interrupted. A gate read from a truncated run is a gate read from a shorter experiment.")

    return failures


def attribution(screen: dict[str, dict]) -> dict:
    """
    Decompose the graph factor using arm E, for the report and never for a threshold.

    Arm B changes the topology and the edge length channel together. Arm E changes only the length channel, on the
    baseline's own topology. So E-A is the length channel and B-E is the topology net of it. If the graph factor is real
    but E-A accounts for most of it, the finding is "the trunk cannot form a distance", which is a cheaper and more
    portable result than "periodic multiedges help" -- and a different one.

    :param screen:
        Runs of the 100-epoch screen, keyed by run name.
    :type screen: dict[str, dict]

    :return:
        The three graph-factor contrasts and, where all are available, the share of B-A that E-A accounts for.
    :rtype: dict
    """
    graph = match_contrast(screen, "B", "A")
    length = match_contrast(screen, "E", "A")
    topology = match_contrast(screen, "B", "E")
    share = None
    if graph["mean_points"] not in (None, 0.0) and length["mean_points"] is not None:
        share = length["mean_points"] / graph["mean_points"]
    return {"graph_B_minus_A": graph, "length_channel_E_minus_A": length, "topology_B_minus_E": topology,
            "length_share_of_graph_effect": share}


def main(argv: Optional[list[str]] = None) -> int:
    """
    Read Gate DG3 and write the stamp.

    :param argv:
        Command line arguments, or None to read them from the process.
    :type argv: Optional[list[str]]

    :return:
        Zero if the gate passes, one if it fails.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", default="direct_geometry/runs", help="where the launcher wrote its runs")
    parser.add_argument("--probe-report", default="direct_geometry/reports/DG1-DG2-PROBES.json",
                        help="the DG1/DG2 stamp, for the promoted descriptor")
    parser.add_argument("--audit-report", default="direct_geometry/reports/DG0-AUDIT.json",
                        help="the DG0 stamp, for the cost ceiling")
    parser.add_argument("--data-dir", default="omg/data/mpts_52", help="the split the runs trained on")
    parser.add_argument("--out", default="direct_geometry/reports/LOCAL-GATE.json", help="where to write the stamp")
    settings = parser.parse_args(argv)

    run_root = Path(settings.run_root)
    runs = collect(run_root, MODES)
    declined = unread_modes(run_root, MODES)
    if not runs:
        print(f"no runs under {run_root} in {list(MODES)}. Run the memorise, screen and paired modes first.")
        if declined:
            print(f"There are finished runs in {declined}, which this gate does not read.")
        return 1

    probe_path = Path(settings.probe_report)
    promoted = None
    if probe_path.is_file():
        promoted = json.loads(probe_path.read_text()).get("verdict", {}).get("promoted")

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
    cost = cost_verdict(Path(settings.audit_report))
    failures = gate_failures(runs, contrasts, cost)

    sources = {}
    for source in SOURCES:
        sources.update(tree_digest(Path(source)))
    data = {str(path): digest(path) for path in sorted(Path(settings.data_dir).glob("*.lmdb"))}
    commands = {f"{mode}/{name}": (Path(entry["path"]) / "COMMAND").read_text().splitlines()
                for mode, entries in runs.items() for name, entry in entries.items()
                if (Path(entry["path"]) / "COMMAND").is_file()}

    print(f"read {sum(len(entries) for entries in runs.values())} runs under {run_root}, "
          f"promoted descriptor {promoted!r}\n")
    if declined:
        print(f"not read, and not part of this gate: {', '.join(declined)}\n")
    for mode in MODES:
        if mode not in runs:
            continue
        print(f"{mode}")
        for name, entry in sorted(runs[mode].items()):
            errors = "measured" if entry.get("errors") else "not measured"
            print(f"  {name:<12} best {entry['best_match_rate']:.4f} at epoch {entry['best_epoch']:.0f}, "
                  f"final {entry['final_match_rate']:.4f}, target error {errors}")
    match = contrasts["paired"]["match"]
    if match["points"]:
        print(f"\npaired D-A: {', '.join(f'seed {s} {v:+.2f}' for s, v in sorted(match['points'].items()))}, "
              f"mean {match['mean_points']:+.2f} points")
    share = contrasts["screen"]["attribution"]["length_share_of_graph_effect"]
    if share is not None:
        print(f"graph factor attribution: the length channel alone accounts for {share:.0%} of B-A")

    verdict = {"passed": not failures, "failures": failures}
    print(f"\nGate DG3: {'PASS' if not failures else 'FAIL'}")
    for failure in failures:
        print(f"  - {failure}")

    stamp = {"gate": "DG3", "verdict": verdict, "thresholds": {
                 "memorisation_floor": MEMORISATION_FLOOR, "paired_gain_points": PAIRED_GAIN_POINTS,
                 "mse_improvement": MSE_IMPROVEMENT, "cell_tolerance": CELL_TOLERANCE, "cost_ceiling": COST_CEILING},
             "promoted_descriptor": promoted, "modes_read": list(MODES), "modes_declined": declined,
             "runs": runs, "contrasts": contrasts, "cost": cost,
             "commands": commands, "source_sha256": sources, "data_sha256": data, "git": commit(),
             "environment": {"torch": torch.__version__,
                             "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
             "arguments": vars(settings)}
    destination = Path(settings.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n")
    print(f"wrote {destination}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
