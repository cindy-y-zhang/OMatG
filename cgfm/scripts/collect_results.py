"""
Collect the results of a set of runs into one table.

Two things get collected, because two questions get asked at different stages.

The test metrics come from the score.json files that cgfm/scripts/evaluate.sh writes, aggregated over seeds. Runs that
have not finished are listed as missing rather than quietly dropped, so a partial table cannot be mistaken for a
complete one.

The validation match rate comes from the metrics.csv that the CSV logger writes during training. That is what the
Stage 1 eta sweep is judged on, and it is also the quantity checkpoints are selected by. Training loss is deliberately
not collected: the arms define different conditional velocity distributions, so their losses are not comparable, and
putting them side by side in a table would invite exactly the comparison that is not valid.

Usage:

    python -m cgfm.scripts.collect_results --runs runs
    python -m cgfm.scripts.collect_results --runs runs --source test --nfe 210
    python -m cgfm.scripts.collect_results --runs sweeps --source validation
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional


ARMS = ("atomwise", "kmedoids", "shells", "learned")
"""Arms of the experiment, in the order they are reported."""

TEST_COLUMNS = (
    ("one_shot", "match_rate", "one-shot match"),
    ("one_shot", "mean_crmse", "one-shot cRMSE"),
    ("one_shot_metre", "match_rate", "METRe"),
    ("best_of_n", "match_rate", "best-of-n match"),
    ("best_of_n", "mean_crmse", "best-of-n cRMSE"),
)
"""Columns of the test table, as (section of score.json, field, heading)."""


def read_score(path: Path) -> Dict[str, float]:
    """
    Flatten the headline numbers of one score.json.

    :param path:
        Path of the score file.
    :type path: Path

    :return:
        Mapping from column heading to value.
    :rtype: Dict[str, float]
    """
    document = json.loads(path.read_text())
    values = {}
    for section, field, heading in TEST_COLUMNS:
        if section in document and field in document[section]:
            value = document[section][field]
            values[heading] = 100.0 * value if field == "match_rate" else value
    return values


def best_validation_match_rate(run_dir: Path) -> Optional[float]:
    """
    Return the best validation match rate the CSV logger recorded for a run.

    :param run_dir:
        Directory holding the run, searched recursively for metrics.csv.
    :type run_dir: Path

    :return:
        The best match rate as a percentage, or None if the run logged none.
    :rtype: Optional[float]
    """
    best = None
    for metrics_file in run_dir.rglob("metrics.csv"):
        with metrics_file.open() as handle:
            for row in csv.DictReader(handle):
                raw = row.get("match_rate", "")
                if raw not in ("", None):
                    value = float(raw)
                    best = value if best is None else max(best, value)
    return None if best is None else 100.0 * best


def format_table(rows: Dict[str, Dict[str, List[float]]], headings: List[str], label: str) -> str:
    """
    Render collected values as a fixed-width table with a mean and a spread over seeds.

    :param rows:
        Mapping from row name to a mapping from column heading to the values collected over seeds.
    :type rows: Dict[str, Dict[str, List[float]]]
    :param headings:
        Column headings, in order.
    :type headings: List[str]
    :param label:
        Heading of the first column.
    :type label: str

    :return:
        The rendered table.
    :rtype: str
    """
    width = max([len(label)] + [len(name) for name in rows])
    header = f"{label:<{width}}  runs  " + "  ".join(f"{heading:>18}" for heading in headings)
    lines = [header, "-" * len(header)]
    for name, columns in rows.items():
        counts = [len(values) for values in columns.values()] or [0]
        line = f"{name:<{width}}  {max(counts):>4}  "
        for heading in headings:
            values = columns.get(heading, [])
            if not values:
                line += f"{'-':>18}  "
            elif len(values) == 1:
                line += f"{values[0]:>18.3f}  "
            else:
                line += f"{mean(values):>11.3f} +- {stdev(values):.3f}  "
        lines.append(line.rstrip())
    return "\n".join(lines)


def collect_test(root: Path, nfe: int) -> tuple[Dict[str, Dict[str, List[float]]], List[str]]:
    """
    Collect the test metrics of every arm at one number of Euler steps.

    :param root:
        Directory holding <arm>/seed<n>/eval/nfe<steps>/score.json.
    :type root: Path
    :param nfe:
        Number of Euler steps to report.
    :type nfe: int

    :return:
        The collected values keyed by arm, and the missing run directories.
    :rtype: tuple[Dict[str, Dict[str, List[float]]], List[str]]
    """
    collected, missing = {}, []
    for arm in ARMS:
        columns: Dict[str, List[float]] = {}
        for seed_dir in sorted(root.glob(f"{arm}/seed*")):
            score_file = seed_dir / "eval" / f"nfe{nfe}" / "score.json"
            if not score_file.exists():
                missing.append(str(score_file))
                continue
            for heading, value in read_score(score_file).items():
                columns.setdefault(heading, []).append(value)
        collected[arm] = columns
    return collected, missing


def collect_validation(root: Path) -> Dict[str, Dict[str, List[float]]]:
    """
    Collect the best validation match rate of every run below a directory.

    Any two-level layout works, which covers both <arm>/seed<n> for the full experiment and <dataset>/eta<value> for
    the Stage 1 sweep.

    :param root:
        Directory holding the runs.
    :type root: Path

    :return:
        The collected values keyed by run name.
    :rtype: Dict[str, Dict[str, List[float]]]
    """
    collected: Dict[str, Dict[str, List[float]]] = {}
    for run_dir in sorted(path for path in root.glob("*/*") if path.is_dir()):
        rate = best_validation_match_rate(run_dir)
        name = f"{run_dir.parent.name}/{run_dir.name}"
        collected[name] = {"validation match": [] if rate is None else [rate]}
    return collected


def main() -> None:
    """Collect and print the results of every run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="runs", help="directory holding the runs")
    parser.add_argument("--source", choices=("test", "validation"), default="test",
                        help="report scored test metrics or the best validation match rate seen during training")
    parser.add_argument("--nfe", type=int, default=210, help="number of Euler steps to report, for --source test")
    parser.add_argument("--json", default=None, help="also write the collected numbers to this JSON file")
    arguments = parser.parse_args()

    root = Path(arguments.runs)
    if arguments.source == "validation":
        collected = collect_validation(root)
        print(format_table(collected, ["validation match"], "run"))
    else:
        collected, missing = collect_test(root, arguments.nfe)
        headings = [heading for _, _, heading in TEST_COLUMNS]
        print(f"Test metrics at {arguments.nfe} Euler steps. Match rates are percentages.\n")
        print(format_table(collected, headings, "arm"))
        if missing:
            print(f"\nMissing scores for:\n  " + "\n  ".join(missing))

    if arguments.json is not None:
        Path(arguments.json).write_text(json.dumps(collected, indent=2))
        print(f"\nWrote {arguments.json}.")


if __name__ == "__main__":
    main()
