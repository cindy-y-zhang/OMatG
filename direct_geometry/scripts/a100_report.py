"""
The verdict on the full sweep: paired contrasts, the two spreads kept apart, and a go/no-go this project is bound by.

WHAT THIS DECIDES

Whether a compact invariant description of the local environment, injected through a zero-initialised residual, improves
atomwise crystal generation on MPTS-52. Two earlier attempts in this project gave the denoiser a semantic variable and both
failed; this is the narrow version of the same question, and it gets one answer.

THREE READINGS, IN THIS ORDER

- Deployment, D-A. Is the configuration anyone would actually ship better than the baseline code path? Must clear both a
  practical margin and two paired standard errors.
- Descriptor main effect, pooled from C-A and D-B. Is the *node descriptor* -- the thing this pivot is about -- carrying
  its weight? A D-A win produced entirely by the message graph is reported as a graph win and is explicitly not evidence
  for the adapter. This is the reading the program's continuation depends on.
- Graph effect, pooled from B-A and D-C, decomposed by arm E. E-A is the edge length channel alone on the baseline's own
  topology; B-E is the periodic multiedge topology net of it. If E-A accounts for most of B-A, the finding is "the trunk
  cannot form a distance", which is cheaper, more portable, and a different result from "periodic multiedges help".

WHY BOTH A MARGIN AND A STANDARD ERROR

Either alone is readable the wrong way. A difference inside the noise is not a finding however large it looks with three
seeds; a difference outside the noise but worth a fraction of a point is not worth deploying. With three seeds the standard
error is itself a weak estimate, which is a property of the design and is reported rather than hidden.

THE TWO SPREADS ARE NEVER POOLED

Across seeds of one arm is uncertainty about the arm, and every verdict here is read on it. Across repeated draws of one
checkpoint is a property of the sampler, and it is reported beside the numbers so a reader knows how precisely each is
known. Pooling them would understate the first.

Usage:

    python -m direct_geometry.scripts.a100_report
    python -m direct_geometry.scripts.a100_report --run-root direct_geometry/a100_runs/launch
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import torch

from .runs import (arm_of, average_contrasts, clears, collect, commit, digest, match_contrast, seed_of, spread,
                   tree_digest)


DEPLOYMENT_POINTS = 2.0
"""
Match-rate points the paired D-A difference must average over seeds.

Two points on a 26.82 baseline is roughly a tenth of the score. Predeclared, and the same margin the local gate used, so
the two readings are of the same effect at different budgets rather than of two different questions.
"""

DESCRIPTOR_POINTS = 0.0
"""
Match-rate points the pooled descriptor main effect must average.

Zero, deliberately: the descriptor is required to be positive beyond the noise, not to clear a second practical bar of its
own. Its practical bar is the deployment contrast, and asking the same margin twice would make a component that carries
half the gain look like a failure.
"""

SIGMAS = 2.0
"""
Multiples of the paired standard error an effect must exceed.

Two, matching the conventional reading of a standard error, and stated as a multiple rather than a p-value because three
seeds do not support a distributional claim.
"""

SOURCES = ("direct_geometry", "cgfm/configs/atomwise_mpts52.yaml", "omg")
"""Paths whose contents are hashed into the report: everything that determines what these numbers mean."""


def arm_table(runs: dict[str, dict]) -> dict[str, dict]:
    """
    Summarise every arm across its seeds.

    :param runs:
        Runs keyed by run name.
    :type runs: dict[str, dict]

    :return:
        Per-arm best validation match rates by seed, and their mean and spread.
    :rtype: dict[str, dict]
    """
    table: dict[str, dict] = {}
    for name, entry in sorted(runs.items()):
        arm = table.setdefault(arm_of(name), {"seeds": {}})
        arm["seeds"][seed_of(name)] = {"best_match_rate": entry["best_match_rate"],
                                       "best_epoch": entry["best_epoch"],
                                       "final_match_rate": entry["final_match_rate"],
                                       "complete": entry["complete"]}
    for arm in table.values():
        arm.update(spread([100.0 * seed["best_match_rate"] for seed in arm["seeds"].values()]))
    return table


def sampling_table(runs: dict[str, dict]) -> dict[str, dict]:
    """
    Collect the repeated-draw and locked-test scores, per arm and sampling budget.

    Averaged over seeds within a budget, with the within-checkpoint spread carried alongside rather than combined into it.

    :param runs:
        Runs keyed by run name.
    :type runs: dict[str, dict]

    :return:
        Per-arm, per-budget mean match rate over seeds, the seed spread, and the mean within-checkpoint spread.
    :rtype: dict[str, dict]
    """
    gathered: dict[str, dict[str, list[dict]]] = {}
    for name, entry in sorted(runs.items()):
        for budget, score in (entry.get("sampling") or {}).items():
            gathered.setdefault(arm_of(name), {}).setdefault(budget, []).append(score)
    table: dict[str, dict] = {}
    for arm, budgets in gathered.items():
        for budget, scores in budgets.items():
            across_seeds = spread([100.0 * score["mean"] for score in scores])
            within = [100.0 * score["standard_deviation"] for score in scores
                      if score["standard_deviation"] is not None]
            table.setdefault(arm, {})[budget] = {
                "mean_match_rate": across_seeds["mean"], "seed_standard_error": across_seeds["standard_error"],
                "seeds": across_seeds["count"],
                "within_checkpoint_standard_deviation": sum(within) / len(within) if within else None,
                "draws_per_seed": [score["draws"] for score in scores]}
    return table


def verdicts(runs: dict[str, dict]) -> dict:
    """
    Compute the three readings and their pass/fail.

    :param runs:
        Runs keyed by run name.
    :type runs: dict[str, dict]

    :return:
        The contrasts, the pooled effects, the decomposition and the verdict on each reading.
    :rtype: dict
    """
    pairs = {f"{better}-{worse}": match_contrast(runs, better, worse)
             for better, worse in (("D", "A"), ("C", "A"), ("D", "B"), ("B", "A"), ("D", "C"), ("E", "A"), ("B", "E"))}
    deployment = pairs["D-A"]
    descriptor = average_contrasts([pairs["C-A"], pairs["D-B"]], "descriptor")
    graph = average_contrasts([pairs["B-A"], pairs["D-C"]], "graph")

    length, topology = pairs["E-A"], pairs["B-E"]
    share = None
    if graph["mean_points"] not in (None, 0.0) and length["mean_points"] is not None:
        share = length["mean_points"] / graph["mean_points"]

    return {
        "contrasts": pairs,
        "deployment": {**deployment, "required_points": DEPLOYMENT_POINTS, "required_sigmas": SIGMAS,
                       "passed": clears(deployment, DEPLOYMENT_POINTS, SIGMAS)},
        "descriptor_main_effect": {**descriptor, "required_points": DESCRIPTOR_POINTS, "required_sigmas": SIGMAS,
                                   "passed": clears(descriptor, DESCRIPTOR_POINTS, SIGMAS)},
        "graph_main_effect": {**graph, "passed": clears(graph, DESCRIPTOR_POINTS, SIGMAS)},
        "graph_decomposition": {"length_channel_E_minus_A": length, "topology_B_minus_E": topology,
                                "length_share_of_graph_effect": share},
    }


def readiness(runs: dict[str, dict], expected_arms: tuple[str, ...], expected_seeds: int) -> list[str]:
    """
    Return every reason the sweep is not yet readable, or an empty list if it is.

    Read before the verdicts rather than after, because a verdict computed from a partial sweep is a verdict on a smaller
    experiment than the one that was designed, and it would not say so.

    :param runs:
        Runs keyed by run name.
    :type runs: dict[str, dict]
    :param expected_arms:
        Arms the sweep was designed with.
    :type expected_arms: tuple[str, ...]
    :param expected_seeds:
        Seeds per arm the sweep was designed with.
    :type expected_seeds: int

    :return:
        Descriptions of what is missing or interrupted.
    :rtype: list[str]
    """
    problems: list[str] = []
    present: dict[str, set[str]] = {}
    for name in runs:
        present.setdefault(arm_of(name), set()).add(seed_of(name))
    for arm in expected_arms:
        seeds = present.get(arm, set())
        if not seeds:
            problems.append(f"arm {arm} has no runs, so every contrast it appears in is unavailable")
        elif len(seeds) < expected_seeds:
            problems.append(f"arm {arm} has {len(seeds)} of {expected_seeds} seeds ({sorted(seeds)}), so its "
                            f"contrasts rest on fewer pairs than the design specifies")
    interrupted = sorted(name for name, entry in runs.items() if not entry["complete"])
    if interrupted:
        problems.append(f"run(s) {', '.join(interrupted)} have metrics but no COMPLETE marker, so they were "
                        f"interrupted and their best match rate is from a shorter run than the others")
    unmeasured = sorted(name for name, entry in runs.items() if not entry.get("sampling"))
    if unmeasured:
        problems.append(f"run(s) {', '.join(unmeasured)} have no sampling scores, so the within-checkpoint spread and "
                        f"the locked test numbers are missing. Run 'direct_geometry/scripts/evaluate.sh' on them")
    return problems


def outcome(report: dict) -> tuple[str, str]:
    """
    Turn the three readings into the one sentence this project agreed to be bound by.

    :param report:
        The computed verdicts.
    :type report: dict

    :return:
        A short verdict name and the sentence.
    :rtype: tuple[str, str]
    """
    deployment = report["deployment"]
    descriptor = report["descriptor_main_effect"]
    graph = report["graph_main_effect"]
    decomposition = report["graph_decomposition"]
    share = decomposition["length_share_of_graph_effect"]

    if deployment["passed"] and descriptor["passed"]:
        return ("adapter", f"GO. The deployed arm beats the baseline by "
                           f"{deployment['mean_points']:+.2f} points and the node descriptor carries "
                           f"{descriptor['mean_points']:+.2f} of it beyond {SIGMAS:.0f} standard errors. The direct "
                           f"invariant adapter is the finding, and the full sweep is the evidence for it.")
    if deployment["passed"] and descriptor["passed"] is False and graph["passed"]:
        detail = "" if share is None else (f" Of that, the edge length channel alone accounts for {share:.0%}, so the "
                                          f"mechanism is most likely that the trunk cannot form an interatomic distance "
                                          f"rather than that periodic multiedges matter.")
        return ("graph", f"GRAPH ONLY. The deployed arm beats the baseline by {deployment['mean_points']:+.2f} points, "
                         f"but the node descriptor contributes {descriptor['mean_points']:+.2f} points and does not "
                         f"clear the noise; the message graph carries {graph['mean_points']:+.2f}.{detail} The "
                         f"node-adapter program ends here. What survives is a corrected graph, which is a fix to "
                         f"OMatG rather than a new representation.")
    if descriptor["passed"]:
        return ("descriptor-only", f"PARTIAL. The node descriptor is positive beyond the noise "
                                   f"({descriptor['mean_points']:+.2f} points) but the deployed arm does not clear "
                                   f"+{DEPLOYMENT_POINTS:.0f} points ({deployment['mean_points']:+.2f}). The effect is "
                                   f"real and too small to deploy. Report it and stop; do not search for an "
                                   f"architecture that enlarges it, which is what an unregistered search would be.")
    return ("stop", f"NO GO. The deployed arm is {deployment['mean_points']:+.2f} points and the node descriptor "
                    f"{descriptor['mean_points']:+.2f}, neither clearing {SIGMAS:.0f} standard errors. Direct "
                    f"invariant features injected into the node stream do not improve this model. That is the third "
                    f"negative result on adding a variable to this denoiser, and the program stops rather than "
                    f"continuing into an architecture search the evidence does not support.")


def main(argv: Optional[list[str]] = None) -> int:
    """
    Read the sweep and write the report.

    :param argv:
        Command line arguments, or None to read them from the process.
    :type argv: Optional[list[str]]

    :return:
        Zero if the sweep is readable, one if it is not yet.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", default="direct_geometry/a100_runs",
                        help="directory the A100 launcher wrote its runs into")
    parser.add_argument("--mode", default="launch", help="which subdirectory of the run root to read")
    parser.add_argument("--probe-report", default="direct_geometry/reports/DG1-DG2-PROBES.json",
                        help="the DG1/DG2 stamp, for the promoted descriptor")
    parser.add_argument("--data-dir", default="omg/data/mpts_52", help="the split the runs trained on")
    parser.add_argument("--arms", default="A,B,C,D,E", help="arms the sweep was designed with")
    parser.add_argument("--seeds", type=int, default=3, help="seeds per arm the sweep was designed with")
    parser.add_argument("--out", default="direct_geometry/reports/A100-REPORT.json", help="where to write the report")
    parser.add_argument("--allow-partial", action="store_true",
                        help="report a sweep that is not finished, marking the verdict provisional")
    settings = parser.parse_args(argv)

    modes = collect(Path(settings.run_root))
    runs = modes.get(settings.mode, {})
    if not runs:
        print(f"no runs under {Path(settings.run_root) / settings.mode}. Launch the sweep first.")
        return 1

    expected = tuple(part.strip() for part in settings.arms.split(",") if part.strip())
    problems = readiness(runs, expected, settings.seeds)
    report = verdicts(runs)
    arms = arm_table(runs)
    sampling = sampling_table(runs)
    provisional = bool(problems)
    name, sentence = outcome(report)

    promoted = None
    probe_path = Path(settings.probe_report)
    if probe_path.is_file():
        promoted = json.loads(probe_path.read_text()).get("verdict", {}).get("promoted")

    print(f"{sum(len(entries) for entries in modes.values())} run(s) under {settings.run_root}, "
          f"{len(runs)} in '{settings.mode}', promoted descriptor {promoted!r}\n")
    print(f"{'arm':<5}{'seeds':<28}{'mean':>8}{'s.e.':>8}")
    for arm, entry in sorted(arms.items()):
        by_seed = " ".join(f"{seed}:{100.0 * value['best_match_rate']:.2f}"
                           for seed, value in sorted(entry["seeds"].items()))
        error = "-" if entry["standard_error"] is None else f"{entry['standard_error']:.2f}"
        print(f"{arm:<5}{by_seed:<28}{entry['mean']:>7.2f}%{error:>8}")

    print("\npaired contrasts, within seed")
    for key, contrast in report["contrasts"].items():
        if contrast["mean_points"] is None:
            print(f"  {key:<6} unavailable: no seed has both arms")
            continue
        error = "-" if contrast["standard_error"] is None else f"{contrast['standard_error']:.2f}"
        by_seed = ", ".join(f"{seed} {value:+.2f}" for seed, value in sorted(contrast["points"].items()))
        print(f"  {key:<6} mean {contrast['mean_points']:+6.2f} +- {error:>5}   ({by_seed})")

    print("\neffects")
    for key in ("deployment", "descriptor_main_effect", "graph_main_effect"):
        effect = report[key]
        if effect["mean_points"] is None:
            print(f"  {key:<22} unavailable")
            continue
        error = "-" if effect["standard_error"] is None else f"{effect['standard_error']:.2f}"
        mark = {True: "clears", False: "does not clear", None: "unreadable"}[effect["passed"]]
        source = effect.get("from", [effect.get("contrast")])
        print(f"  {key:<22} {effect['mean_points']:+6.2f} +- {error:>5} points from {source}: {mark}")
    share = report["graph_decomposition"]["length_share_of_graph_effect"]
    if share is not None:
        print(f"  {'graph decomposition':<22} the edge length channel alone accounts for {share:.0%} of the graph "
              f"effect")

    if sampling:
        print("\nsampling, mean over seeds (within-checkpoint spread in brackets)")
        for arm, budgets in sorted(sampling.items()):
            for budget, entry in sorted(budgets.items()):
                within = ("-" if entry["within_checkpoint_standard_deviation"] is None
                          else f"{entry['within_checkpoint_standard_deviation']:.2f}")
                error = "-" if entry["seed_standard_error"] is None else f"{entry['seed_standard_error']:.2f}"
                print(f"  {arm} {budget:<12} {entry['mean_match_rate']:6.2f}%  seed s.e. {error:>5}  "
                      f"[draw sd {within:>5}]")

    if problems:
        print("\nthe sweep is not finished:")
        for problem in problems:
            print(f"  - {problem}")
        if not settings.allow_partial:
            print("\nNo verdict. Finish the sweep, or pass --allow-partial to read a provisional one.")
        else:
            print(f"\nPROVISIONAL {sentence}")
    else:
        print(f"\n{sentence}")

    sources: dict[str, str] = {}
    for source in SOURCES:
        sources.update(tree_digest(Path(source)))
    stamp = {"verdict": {"outcome": name, "sentence": sentence, "provisional": provisional},
             "readiness": problems, "thresholds": {"deployment_points": DEPLOYMENT_POINTS,
                                                   "descriptor_points": DESCRIPTOR_POINTS, "sigmas": SIGMAS},
             "promoted_descriptor": promoted, "arms": arms, "sampling": sampling, **report,
             "runs": runs, "source_sha256": sources,
             "data_sha256": {str(path): digest(path) for path in sorted(Path(settings.data_dir).glob("*.lmdb"))},
             "git": commit(),
             "environment": {"torch": torch.__version__,
                             "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
             "arguments": vars(settings)}
    destination = Path(settings.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n")
    print(f"wrote {destination}")
    return 1 if problems and not settings.allow_partial else 0


if __name__ == "__main__":
    raise SystemExit(main())
