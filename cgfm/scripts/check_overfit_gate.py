"""Check the 100-structure overfit gate before full MPTS-52 training.

The gate passes when any logged validation of the deliberately identical
100-structure train/validation subsets reaches the requested match rate.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


OVERFIT_MATCH_RATE = 0.80


def _file_sha256(path: Path) -> str:
    """Return a stable digest of the metrics file certified by the stamp."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def best_match_rate(path: Path) -> float:
    """Return the best finite match rate in a Lightning CSV metrics file."""
    if not path.exists():
        raise FileNotFoundError(f"No overfit metrics at {path}.")
    values = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get("match_rate", "")
            if value not in ("", None):
                rate = float(value)
                if 0.0 <= rate <= 1.0:
                    values.append(rate)
    if not values:
        raise ValueError(f"{path} contains no valid match_rate measurements.")
    return max(values)


def main() -> None:
    """Evaluate the overfit gate and write its launch stamp only on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="CSV metrics from the 100-structure overfit run")
    parser.add_argument("--consensus-weight", required=True, type=float,
                        help="selected consensus weight to freeze for both block arms")
    parser.add_argument("--threshold", type=float, default=OVERFIT_MATCH_RATE)
    parser.add_argument("--stamp", default="cgfm/blocks/mpts_52/phase1_passed.json")
    arguments = parser.parse_args()
    if not OVERFIT_MATCH_RATE <= arguments.threshold <= 1.0:
        raise ValueError(f"The match-rate threshold must be in [{OVERFIT_MATCH_RATE}, 1].")
    if arguments.consensus_weight < 0.0:
        raise ValueError("The consensus weight must be non-negative.")

    metrics = Path(arguments.metrics)
    stamp = Path(arguments.stamp)
    stamp.unlink(missing_ok=True)
    rate = best_match_rate(metrics)
    print(f"G3  best 100-structure training match: {100.0 * rate:.2f} per cent "
          f"(threshold {100.0 * arguments.threshold:.0f})")
    if rate < arguments.threshold:
        raise SystemExit("Phase-1 overfit gate failed; do not launch full training.")

    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps({
        "passed": True,
        "metrics": str(metrics),
        "metrics_sha256": _file_sha256(metrics),
        "best_match_rate": rate,
        "threshold": arguments.threshold,
        "consensus_weight": arguments.consensus_weight,
    }, indent=2))
    print(f"Phase-1 overfit gate passed; wrote {stamp}.")


if __name__ == "__main__":
    main()
