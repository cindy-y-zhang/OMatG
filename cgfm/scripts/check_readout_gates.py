"""
Check the Phase-0 readout gates against the JSON summaries of ``readout_ceiling``.

G1 requires the free readout of the coarse vocabulary to match at least 90 per cent of structures at 0.10 Angstrom /
5 degrees under the comparison tolerance (0.3 / 0.5 / 10). G2 requires that same coarse free readout to sit within
three percentage points of the fine readout. Either failure is a hard stop: GPU training must not be launched.

Usage:

    python -m cgfm.scripts.check_readout_gates \\
        --coarse cgfm/reports/readout_g1_coarse.json \\
        --fine cgfm/reports/readout_g1_fine.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from .readout_ceiling import G1_FREE_MATCH_RATE, G2_TOLERANCE_POINTS


def _file_sha256(path: Path) -> str:
    """Return the digest of a ceiling report certified by a pass stamp."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    """
    Read one ceiling summary.

    :param path:
        Path of the JSON file written by ``readout_ceiling --json-out``.
    :type path: Path

    :return:
        The summary.
    :rtype: dict

    :raises FileNotFoundError:
        If the file does not exist.
    :raises ValueError:
        If the file is not a ceiling summary.
    """
    if not path.exists():
        raise FileNotFoundError(f"No ceiling summary at {path}. Run cgfm.scripts.readout_ceiling with --json-out.")
    document = json.loads(path.read_text())
    if "g1" not in document or "free_match_rate_comparison" not in document["g1"]:
        raise ValueError(f"{path} is not a readout-ceiling summary.")
    return document


def main() -> None:
    """Evaluate G1 and G2 and exit non-zero if either fails."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coarse", required=True, help="JSON summary of the centre-cn (capped coarse) ceiling")
    parser.add_argument("--fine", required=True, help="JSON summary of the centre-cn-ligands (fine) ceiling")
    parser.add_argument("--stamp", default=None, help="write a machine-readable pass stamp for the launch guard")
    arguments = parser.parse_args()
    stamp = Path(arguments.stamp) if arguments.stamp else None
    if stamp is not None:
        stamp.unlink(missing_ok=True)

    coarse_path = Path(arguments.coarse)
    fine_path = Path(arguments.fine)
    coarse = _load(coarse_path)
    fine = _load(fine_path)
    coarse_rate = float(coarse["g1"]["free_match_rate_comparison"])
    fine_rate = float(fine["g1"]["free_match_rate_comparison"])
    gap = abs(100.0 * coarse_rate - 100.0 * fine_rate)
    g1 = coarse_rate >= G1_FREE_MATCH_RATE
    g2 = gap <= G2_TOLERANCE_POINTS

    print(f"G1  coarse free match at 0.10 A / 5 deg, comparison tolerance: "
          f"{100.0 * coarse_rate:.2f} per cent  ({'pass' if g1 else 'fail'}, "
          f"threshold {100.0 * G1_FREE_MATCH_RATE:.0f})")
    print(f"G2  |coarse - fine| = {gap:.2f} percentage points  "
          f"({'pass' if g2 else 'fail'}, tolerance {G2_TOLERANCE_POINTS:.1f})")
    print(f"    fine free match: {100.0 * fine_rate:.2f} per cent")

    if not g1 or not g2:
        print("Phase-0 gates failed; do not launch GPU training.", file=sys.stderr)
        sys.exit(1)
    if stamp is not None:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(json.dumps({
            "passed": True,
            "coarse": str(coarse_path),
            "fine": str(fine_path),
            "coarse_sha256": _file_sha256(coarse_path),
            "fine_sha256": _file_sha256(fine_path),
            "coarse_rate": coarse_rate,
            "fine_rate": fine_rate,
            "g1_threshold": G1_FREE_MATCH_RATE,
            "g2_tolerance_points": G2_TOLERANCE_POINTS,
        }, indent=2))
        print(f"Wrote {stamp}.")
    print("Phase-0 gates passed.")


if __name__ == "__main__":
    main()
