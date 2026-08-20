"""Require a finite, materially falling loss in the fixed-step memorization run."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def memorization_statistics(
    values: list[float],
    minimum_relative_drop: float,
) -> dict[str, float | bool]:
    """Summarise the fixed-step objective trace and apply its stop gate."""
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("The memorization run has fewer than two finite loss measurements.")
    window = min(5, max(1, len(values) // 2))
    initial = float(np.mean(values[:window]))
    final = float(np.mean(values[-window:]))
    relative_drop = (initial - final) / max(abs(initial), 1.0e-12)
    return {
        "initial_loss": initial,
        "final_loss": final,
        "relative_drop": relative_drop,
        "minimum_relative_drop": minimum_relative_drop,
        "passed": relative_drop >= minimum_relative_drop,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics")
    parser.add_argument("--minimum-relative-drop", type=float, default=0.20)
    parser.add_argument("--structures", type=int, default=10)
    parser.add_argument("--optimizer-steps", type=int, default=100)
    parser.add_argument("--out", default="joint_geometry/reports/MEMORIZATION-GATE.json")
    arguments = parser.parse_args()
    values = []
    with Path(arguments.metrics).open() as handle:
        for row in csv.DictReader(handle):
            for key in ("loss_total_epoch", "loss_total_step"):
                if row.get(key) not in (None, ""):
                    values.append(float(row[key]))
                    break
    report = {
        "metrics": arguments.metrics,
        **memorization_statistics(values, arguments.minimum_relative_drop),
        "structures": arguments.structures,
        "optimizer_steps": arguments.optimizer_steps,
    }
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
