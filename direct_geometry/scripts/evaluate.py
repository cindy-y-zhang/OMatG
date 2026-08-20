"""
Score several independent sampling draws of one checkpoint, one draw at a time.

WHY NOT THE EXISTING SCORER

``cgfm.scripts.score`` reports a best-of-n rate: a composition counts as solved if any draw matched it. That is the right
metric for a crystal-structure-prediction budget and the wrong one here. What the A100 report needs is the *spread* of the
one-shot match rate over repeated draws of a single checkpoint, and a best-of-n number destroys exactly that by pooling the
draws before it measures anything. So each draw is scored separately, and the same ``match_rmsds`` call with the same
tolerances the training loop's own ``match_rate`` uses, so the numbers here and the logged validation curve mean the same
thing.

WHAT THE SPREAD IS FOR, AND WHAT IT IS NOT FOR

Two spreads matter in this project and they must never be pooled:

- across draws of one checkpoint, measured here: a property of the sampler. It says how precisely one checkpoint's match
  rate is known, and therefore whether a reported number is worth the digits it is printed with.
- across seeds of one arm, measured by the report: uncertainty about the arm, which is what every verdict is read on.

Pooling them would understate the second by diluting it with the first, which is the mistake that makes a two-point
difference look decisive when three seeds of the same arm differ by more than that.

Usage:

    python -m direct_geometry.scripts.evaluate --generated eval/draw0.xyz eval/draw1.xyz \\
        --reference omg/data/mpts_52/val.lmdb --split val --integration-time-steps 210 --out eval/SCORE.json
"""

import argparse
import json
from pathlib import Path
from typing import Optional
from omg.analysis import match_rmsds, ValidAtoms
from omg.datamodule import StructureDataset
from omg.utils import xyz_reader


LTOL = 0.3
"""Fractional length tolerance. OMatG's csp_metrics default, and what the training loop's match_rate uses."""

STOL = 0.5
"""Site tolerance. OMatG's csp_metrics default, and what the training loop's match_rate uses."""

ANGLE_TOL = 10.0
"""Angle tolerance in degrees. OMatG's csp_metrics default, and what the training loop's match_rate uses."""


def reference_structures(path: Path) -> list:
    """
    Read a reference split.

    :param path:
        An ``.lmdb`` split or an extended XYZ file.
    :type path: Path

    :return:
        The reference structures.
    :rtype: list[ase.Atoms]
    """
    if path.suffix == ".xyz":
        return xyz_reader(path)
    dataset = StructureDataset(file_path=str(path), lazy_storage=True, niggli_reduce=False,
                              convert_to_fractional=False)
    return [dataset[index].get_ase_atoms() for index in range(len(dataset))]


def score_draw(draw: Path, reference: list, workers: int) -> dict:
    """
    Score one draw against the reference split.

    ``skip_validation`` matches the training loop, which also skips it: a structure that fails the validity check would
    still be handed to the matcher, and excluding it here but not there would put the two numbers on different
    denominators.

    :param draw:
        Extended XYZ file holding one generated structure per reference composition, in the same order.
    :type draw: Path
    :param reference:
        Validated reference structures.
    :type reference: list[ValidAtoms]
    :param workers:
        Number of worker processes for the matcher.
    :type workers: int

    :return:
        The one-shot match rate, mean root-mean-square distance and corrected mean.
    :rtype: dict
    """
    atoms = xyz_reader(draw)
    if len(atoms) != len(reference):
        raise ValueError(
            f"{draw} holds {len(atoms)} structures but the reference split holds {len(reference)}. Every draw must "
            f"generate exactly one structure per reference composition, in the same order, or the match rate is not "
            f"the one the training loop reports.")
    generated = ValidAtoms.get_valid_atoms(atoms, desc=f"Validating {draw.name}", skip_validation=True,
                                           number_cpus=1)
    match_rate, mean_rmsd, _, _, _, _, corrected_rmsd, _ = match_rmsds(
        generated, reference, ltol=LTOL, stol=STOL, angle_tol=ANGLE_TOL, number_cpus=workers,
        enable_progress_bar=True)
    return {"draw": draw.name, "match_rate": float(match_rate), "mean_rmsd": float(mean_rmsd),
            "corrected_rmsd": float(corrected_rmsd)}


def summarise(match_rates: list[float]) -> dict[str, Optional[float]]:
    """
    Return the mean and sample spread of a list of match rates.

    :param match_rates:
        One match rate per draw.
    :type match_rates: list[float]

    :return:
        Mean, sample standard deviation and standard error. The spreads are None below two draws.
    :rtype: dict[str, Optional[float]]
    """
    count = len(match_rates)
    mean = sum(match_rates) / count
    if count < 2:
        return {"mean_match_rate": mean, "standard_deviation": None, "standard_error": None}
    variance = sum((rate - mean) ** 2 for rate in match_rates) / (count - 1)
    deviation = variance ** 0.5
    return {"mean_match_rate": mean, "standard_deviation": deviation, "standard_error": deviation / count ** 0.5}


def main(argv: Optional[list[str]] = None) -> int:
    """
    Score every draw and write the report.

    :param argv:
        Command line arguments, or None to read them from the process.
    :type argv: Optional[list[str]]

    :return:
        Zero on success.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generated", required=True, nargs="+", help="one extended XYZ file per draw")
    parser.add_argument("--reference", required=True, help="the split the draws were generated for")
    parser.add_argument("--split", required=True, choices=("val", "test"), help="which split that is")
    parser.add_argument("--integration-time-steps", type=int, required=True,
                        help="sampling budget the draws were generated at")
    parser.add_argument("--checkpoint", default=None, help="checkpoint the draws came from, recorded in the report")
    parser.add_argument("--out", required=True, help="where to write the report")
    parser.add_argument("--workers", type=int, default=8, help="worker processes for the matcher")
    settings = parser.parse_args(argv)

    reference_atoms = reference_structures(Path(settings.reference))
    reference = ValidAtoms.get_valid_atoms(reference_atoms, desc="Validating reference structures",
                                           skip_validation=True, number_cpus=1)
    draws = [score_draw(Path(path), reference, settings.workers) for path in settings.generated]
    match_rates = [draw["match_rate"] for draw in draws]

    report = {"split": settings.split, "integration_time_steps": settings.integration_time_steps,
              "checkpoint": settings.checkpoint, "reference": settings.reference,
              "num_compositions": len(reference), "tolerances": {"ltol": LTOL, "stol": STOL, "angle_tol": ANGLE_TOL},
              "draws": draws, "match_rates": match_rates, **summarise(match_rates)}
    destination = Path(settings.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"\n{settings.split} split, {settings.integration_time_steps} sampling steps, "
          f"{len(reference)} compositions")
    for draw in draws:
        print(f"  {draw['draw']:<12} match {100.0 * draw['match_rate']:6.2f}%   RMSE {draw['mean_rmsd']:.4f}")
    deviation = report["standard_deviation"]
    if deviation is None:
        print(f"  mean {100.0 * report['mean_match_rate']:.2f}% from a single draw, so no spread")
    else:
        print(f"  mean {100.0 * report['mean_match_rate']:.2f}%, standard deviation {100.0 * deviation:.2f} points "
              f"over {len(draws)} draws of this one checkpoint")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
