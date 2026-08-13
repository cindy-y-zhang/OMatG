"""
Collect the annealing-factor sweeps of cgfm/scripts/sweep_annealing.sh into one table.

The question this answers is whether the eta sweep measured the probability path or the sampler. Each arm was scored
with the velocity annealing factor OMatG tuned for the atomwise path, while the bump raises the terminal slope of the
within-group schedule to 1 + eta. If a coarse-grained arm recovers its deficit once the factor is retuned, the deficit
was an artifact of the sampler; if it does not, the deficit belongs to the path.

Only match_rate and the RMSDs are reported. The validation loss written by these runs is not meaningful, because
generation never evaluates a grouping and the runs therefore do not reproduce the training-time grouping state.

Each run is reported once per scope, the scope being how much of the validation split was scored. A prefix and the whole
split give different absolute rates, so they are never mixed into one table.

Usage:

    python cgfm/scripts/collect_annealing.py --runs sweeps/mp20/eta0.5/version_0
    python cgfm/scripts/collect_annealing.py --runs sweeps
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


TUNED_FACTOR = 10.182659004291072
"""
Default reference factor, from OMatG's tuned crystal-structure-prediction configuration
omg/conf_examples/csp_linear_ode_mp_20.yaml. This is the value every run of the eta sweep was scored at.
"""

METRICS = (("match_rate", "match %", 100.0), ("mean_rmsd", "mean RMSD", 1.0), ("corr_rmsd", "corr RMSD", 1.0))
"""Reported metrics, as (field of metrics.csv, heading, factor to multiply by)."""


def read_metrics(metrics_file: Path) -> Dict[str, float]:
    """
    Read the last recorded value of every reported metric from one metrics.csv.

    :param metrics_file:
        Path of the file the CSV logger wrote.
    :type metrics_file: Path

    :return:
        Mapping from column heading to value, holding only the metrics the file recorded.
    :rtype: Dict[str, float]
    """
    values: Dict[str, float] = {}
    with metrics_file.open() as handle:
        for row in csv.DictReader(handle):
            for field, heading, scale in METRICS:
                raw = row.get(field, "")
                if raw not in ("", None):
                    values[heading] = scale * float(raw)
    return values


def collect(root: Path) -> Dict[str, Dict[float, Dict[str, float]]]:
    """
    Collect every annealing result below a directory, keyed by run and scope and then by factor.

    A run appears once per scope, because a match rate on a prefix of the validation split and one on the whole split
    are different quantities and must not be tabulated as if they were comparable.

    :param root:
        Directory to search recursively for anneal/<scope>/factor<value>/metrics.csv.
    :type root: Path

    :return:
        Mapping from "<run directory> [<scope>]" to a mapping from annealing factor to its metrics.
    :rtype: Dict[str, Dict[float, Dict[str, float]]]
    """
    collected: Dict[str, Dict[float, Dict[str, float]]] = {}
    for metrics_file in sorted(root.rglob("anneal/*/factor*/metrics.csv")):
        factor_dir = metrics_file.parent
        while factor_dir.name and not factor_dir.name.startswith("factor"):
            factor_dir = factor_dir.parent
        scope_dir = factor_dir.parent
        run_dir = scope_dir.parent.parent
        values = read_metrics(metrics_file)
        if values:
            name = f"{run_dir if run_dir != root else run_dir.name} [{scope_dir.name}]"
            collected.setdefault(name, {})[float(factor_dir.name.removeprefix("factor"))] = values
    return collected


def format_run(factors: Dict[float, Dict[str, float]], tuned: float) -> List[str]:
    """
    Render the results of one run, one line per annealing factor.

    The line of the reference factor is marked "as scored" and the best line "best", so that the two numbers the
    decision rests on can be read off without comparing columns by eye.

    :param factors:
        Mapping from annealing factor to its metrics.
    :type factors: Dict[float, Dict[str, float]]
    :param tuned:
        Reference factor that the eta sweep was scored at.
    :type tuned: float

    :return:
        The rendered lines, without a trailing newline.
    :rtype: List[str]
    """
    headings = [heading for _, heading, _ in METRICS]
    reference: Optional[float] = None
    if tuned in factors and "match %" in factors[tuned]:
        reference = factors[tuned]["match %"]
    rated = {factor: values["match %"] for factor, values in factors.items() if "match %" in values}
    best = max(rated, key=lambda factor: rated[factor]) if rated else None

    lines = ["  " + f"{'factor':>10}" + "".join(f"{heading:>12}" for heading in headings)
             + f"{'vs tuned':>11}  note"]
    for factor in sorted(factors):
        values = factors[factor]
        line = "  " + f"{factor:>10.4f}"
        for heading in headings:
            line += f"{values[heading]:>12.4f}" if heading in values else f"{'-':>12}"
        if reference is not None and "match %" in values:
            line += f"{values['match %'] - reference:>+11.3f}"
        else:
            line += f"{'-':>11}"
        notes = []
        if factor == tuned:
            notes.append("as scored")
        if factor == best and len(rated) > 1:
            notes.append("best")
        lines.append(line + "  " + ", ".join(notes))
    return lines


def main() -> None:
    """Collect and print the annealing sweeps below a directory."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="sweeps", help="directory holding the runs")
    parser.add_argument("--tuned", type=float, default=TUNED_FACTOR,
                        help="reference annealing factor that the runs being compared were originally scored at")
    parser.add_argument("--json", default=None, help="also write the collected numbers to this JSON file")
    arguments = parser.parse_args()

    collected = collect(Path(arguments.runs))
    if not collected:
        print(f"No annealing results under {arguments.runs}.")
        return

    print("Validation match rate by inference velocity annealing factor. Match rates are percentages.\n")
    for name in sorted(collected):
        print(name)
        print("\n".join(format_run(collected[name], arguments.tuned)))
        print()

    if arguments.json is not None:
        Path(arguments.json).write_text(json.dumps(collected, indent=2))
        print(f"Wrote {arguments.json}.")


if __name__ == "__main__":
    main()
