"""Mark a training run complete only after every configured epoch was reached."""

import argparse
import csv
import json
from pathlib import Path


def maximum_epoch(path: Path) -> int:
    """Return the largest integer epoch logged in a Lightning CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"No training metrics at {path}.")
    epochs = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("epoch", "")
            if raw not in ("", None):
                epochs.append(int(float(raw)))
    if not epochs:
        raise ValueError(f"{path} contains no epoch measurements.")
    return max(epochs)


def main() -> None:
    """Write exactly one of the complete and incomplete status files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-epochs", type=int, default=400)
    arguments = parser.parse_args()
    if arguments.expected_epochs <= 0:
        raise ValueError("expected-epochs must be positive.")

    run_dir = Path(arguments.run_dir)
    metrics = run_dir / "metrics.csv"
    last_epoch = maximum_epoch(metrics)
    payload = {
        "complete": last_epoch >= arguments.expected_epochs - 1,
        "last_epoch": last_epoch,
        "expected_epochs": arguments.expected_epochs,
        "metrics": str(metrics),
    }
    complete = run_dir / "COMPLETED.json"
    incomplete = run_dir / "INCOMPLETE.json"
    if payload["complete"]:
        incomplete.unlink(missing_ok=True)
        complete.write_text(json.dumps(payload, indent=2))
        print(f"Training complete at epoch {last_epoch}; wrote {complete}.")
        return

    complete.unlink(missing_ok=True)
    incomplete.write_text(json.dumps(payload, indent=2))
    raise SystemExit(
        f"Training stopped at epoch {last_epoch}, before epoch {arguments.expected_epochs - 1}; "
        f"wrote {incomplete}. Resume this run.")


if __name__ == "__main__":
    main()
