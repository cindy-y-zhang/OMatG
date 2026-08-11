"""
Score generated structures against the reference split.

OMatG's csp_metrics covers the one-shot setting, but it cannot score several samples per composition: both match_rmsds
and metre_rmsds refuse a generated list longer than the reference list. Its METRe rate is also the fraction of
*generated* structures that match some reference, which is a precision measure. Feeding it five samples per composition
would make it fall as the model became more diverse, the opposite of the best-of-n rate the crystal-structure
prediction literature reports. So the multi-sample metrics are computed here instead, on top of the same ValidAtoms
validation and the same pymatgen StructureMatcher call that OMatG uses, and the one-shot numbers still come from OMatG
itself so that they stay comparable with published results.

Three things are reported:

- one-shot, from the first draw only: the one-to-one match rate, its RMSE and cRMSE, and the polymorph-aware METRe rate
  with its RMSE and cRMSE. This is the setting the four arms are compared in.
- best-of-n over all draws: a composition counts as solved if any of its samples matches its own reference structure,
  and its error is the smallest RMSE among the matching samples. This is the usual match-rate-at-n-samples metric.
- both of the above split by atom count, in the bins the plan asks for. With a shared per-structure group count, small
  structures get very few groups, and at one group the coarse component is a pure translation that the centre-of-mass
  correction already removes. A gain that grows with atom count is therefore the mechanistically expected signature of
  a useful intermediate resolution, rather than of a merely different schedule.

Unmatched structures contribute stol to the corrected RMSE, matching OMatG's convention, and are left out of the plain
RMSE.

Usage:

    python -m cgfm.scripts.score --generated runs/shells/seed0/draw0.xyz runs/shells/seed0/draw1.xyz \\
        --reference omg/data/mpts_52/test.lmdb --out runs/shells/seed0/score.json
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from ase import Atoms
from tqdm import tqdm
from omg.analysis import match_rmsds, metre_rmsds, ValidAtoms
# Private only in the sense of not being re-exported. Reusing it is the whole point: the best-of-n rate has to be built
# on exactly the composition check and StructureMatcher call that OMatG's own metrics use, or the numbers would not be
# comparable with the one-shot ones reported beside them.
from omg.analysis.analysis import _get_match_and_rmsd
from omg.datamodule import StructureDataset
from omg.utils import xyz_reader


ATOM_COUNT_BINS = ((1, 10), (11, 20), (21, 36), (37, 52))
"""Atom-count bins, inclusive at both ends, from the plan's Stage 5."""

DEFAULT_LTOL = 0.3
"""Fractional length tolerance, matching OMatG's csp_metrics default."""

DEFAULT_STOL = 0.5
"""Site tolerance, matching OMatG's csp_metrics default. Also the penalty applied to unmatched structures."""

DEFAULT_ANGLE_TOL = 10.0
"""Angle tolerance in degrees, matching OMatG's csp_metrics default."""


_DRAWS: List[List[ValidAtoms]] = []
"""Per-worker handle on the generated structures, one list per draw."""

_REFERENCE: List[ValidAtoms] = []
"""Per-worker handle on the reference structures."""


def _initialise_worker(draws: List[List[ValidAtoms]], reference: List[ValidAtoms]) -> None:
    """
    Give a worker process its copy of the structures.

    :param draws:
        Validated generated structures, one list per draw.
    :type draws: List[List[ValidAtoms]]
    :param reference:
        Validated reference structures.
    :type reference: List[ValidAtoms]
    """
    global _DRAWS, _REFERENCE
    _DRAWS, _REFERENCE = draws, reference


def _best_rmsd(index: int, ltol: float, stol: float, angle_tol: float, check_reduced: bool) -> Optional[float]:
    """
    Return the smallest root-mean-square distance with which any draw reproduces one reference structure.

    :param index:
        Index of the composition within the reference split.
    :type index: int
    :param ltol:
        Fractional length tolerance for pymatgen's StructureMatcher.
    :type ltol: float
    :param stol:
        Site tolerance for pymatgen's StructureMatcher.
    :type stol: float
    :param angle_tol:
        Angle tolerance in degrees for pymatgen's StructureMatcher.
    :type angle_tol: float
    :param check_reduced:
        Whether structures whose compositions are simple multiples of each other may match.
    :type check_reduced: bool

    :return:
        The smallest root-mean-square distance, or None if no draw matched.
    :rtype: Optional[float]
    """
    reference = _REFERENCE[index]
    rmsds = [_get_match_and_rmsd(draw[index], reference, ltol, stol, angle_tol, check_reduced) for draw in _DRAWS]
    matched = [rmsd for rmsd in rmsds if rmsd is not None]
    return min(matched) if matched else None


def summarise(rmsds: Sequence[Optional[float]], stol: float) -> Dict[str, float]:
    """
    Turn per-composition root-mean-square distances into a match rate, an RMSE and a corrected RMSE.

    :param rmsds:
        Root-mean-square distance of every composition, or None where nothing matched.
    :type rmsds: Sequence[Optional[float]]
    :param stol:
        Penalty applied to unmatched compositions in the corrected RMSE.
    :type stol: float

    :return:
        Mapping holding the match rate, the mean RMSE over matches, and the corrected mean RMSE over all compositions.
    :rtype: Dict[str, float]
    """
    if len(rmsds) == 0:
        return {"count": 0, "match_rate": 0.0, "mean_rmse": float("nan"), "mean_crmse": float("nan")}
    matched = [rmsd for rmsd in rmsds if rmsd is not None]
    corrected = [rmsd if rmsd is not None else stol for rmsd in rmsds]
    return {
        "count": len(rmsds),
        "match_rate": len(matched) / len(rmsds),
        "mean_rmse": sum(matched) / len(matched) if matched else float("nan"),
        "mean_crmse": sum(corrected) / len(corrected),
    }


def bin_by_atom_count(rmsds: Sequence[Optional[float]], atom_counts: Sequence[int],
                      stol: float) -> Dict[str, Dict[str, float]]:
    """
    Summarise per-composition root-mean-square distances separately in each atom-count bin.

    :param rmsds:
        Root-mean-square distance of every composition, or None where nothing matched.
    :type rmsds: Sequence[Optional[float]]
    :param atom_counts:
        Number of atoms of every reference structure.
    :type atom_counts: Sequence[int]
    :param stol:
        Penalty applied to unmatched compositions in the corrected RMSE.
    :type stol: float

    :return:
        Mapping from bin name to its summary.
    :rtype: Dict[str, Dict[str, float]]
    """
    binned = {}
    for low, high in ATOM_COUNT_BINS:
        selected = [rmsd for rmsd, count in zip(rmsds, atom_counts) if low <= count <= high]
        binned[f"{low}-{high}"] = summarise(selected, stol)
    return binned


def load_reference(path: str) -> List[Atoms]:
    """
    Read the reference structures from an extended XYZ file or an LMDB split.

    :param path:
        Path of the reference file.
    :type path: str

    :return:
        The reference structures.
    :rtype: List[ase.Atoms]
    """
    if path.endswith(".xyz"):
        return xyz_reader(Path(path))
    dataset = StructureDataset(file_path=path, lazy_storage=True, niggli_reduce=False, convert_to_fractional=False)
    return [dataset[index].get_ase_atoms() for index in range(len(dataset))]


def main() -> None:
    """Score one run and write the metrics to a JSON file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generated", required=True, nargs="+",
                        help="extended XYZ file of each draw, one structure per reference composition")
    parser.add_argument("--reference", required=True, help="reference split, either an .lmdb or an .xyz file")
    parser.add_argument("--out", required=True, help="path of the JSON file to write")
    parser.add_argument("--ltol", type=float, default=DEFAULT_LTOL, help="fractional length tolerance")
    parser.add_argument("--stol", type=float, default=DEFAULT_STOL, help="site tolerance and unmatched penalty")
    parser.add_argument("--angle-tol", type=float, default=DEFAULT_ANGLE_TOL, help="angle tolerance in degrees")
    parser.add_argument("--skip-validation", action="store_true", help="treat every structure as valid")
    parser.add_argument("--workers", type=int, default=None, help="number of worker processes")
    parser.add_argument("--label", default=None, help="name recorded in the JSON, for example the arm and seed")
    arguments = parser.parse_args()

    reference_atoms = load_reference(arguments.reference)
    reference = ValidAtoms.get_valid_atoms(reference_atoms, desc="Validating reference structures",
                                           skip_validation=arguments.skip_validation, number_cpus=arguments.workers)
    draws = []
    for path in arguments.generated:
        atoms = xyz_reader(Path(path))
        if len(atoms) != len(reference):
            raise ValueError(
                f"{path} holds {len(atoms)} structures but the reference split holds {len(reference)}. Every draw must "
                f"generate exactly one structure per reference composition, in the same order.")
        draws.append(ValidAtoms.get_valid_atoms(atoms, desc=f"Validating {Path(path).name}",
                                                skip_validation=arguments.skip_validation,
                                                number_cpus=arguments.workers))

    atom_counts = [len(atoms) for atoms in reference_atoms]
    stol = arguments.stol
    common = {"ltol": arguments.ltol, "stol": stol, "angle_tol": arguments.angle_tol,
              "number_cpus": arguments.workers}

    # One-shot metrics come from OMatG so that they remain directly comparable with its published numbers.
    _, _, _, _, one_to_one_rmsds, _, _, _ = match_rmsds(draws[0], reference, **common)
    _, _, _, _, metre_rmsd_list, _, _, _ = metre_rmsds(draws[0], reference, **common)

    # The structures are handed to each worker once through the initialiser rather than travelling with every task,
    # because a ValidAtoms carries a pymatgen Structure and its fingerprints.
    worker = partial(_best_rmsd, ltol=arguments.ltol, stol=stol, angle_tol=arguments.angle_tol, check_reduced=True)
    with ProcessPoolExecutor(max_workers=arguments.workers if arguments.workers is not None else os.cpu_count(),
                             initializer=_initialise_worker, initargs=(draws, reference)) as executor:
        best_rmsds = list(tqdm(executor.map(worker, range(len(reference)), chunksize=8), total=len(reference),
                               desc=f"Best of {len(draws)}"))

    result = {
        "label": arguments.label,
        "reference": arguments.reference,
        "num_compositions": len(reference),
        "num_draws": len(draws),
        "tolerances": {"ltol": arguments.ltol, "stol": stol, "angle_tol": arguments.angle_tol},
        "one_shot": summarise(one_to_one_rmsds, stol),
        "one_shot_metre": summarise(metre_rmsd_list, stol),
        "best_of_n": summarise(best_rmsds, stol),
        "one_shot_bins": bin_by_atom_count(one_to_one_rmsds, atom_counts, stol),
        "best_of_n_bins": bin_by_atom_count(best_rmsds, atom_counts, stol),
    }

    out_path = Path(arguments.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\n{arguments.label or out_path.stem}: {len(draws)} draw(s) over {len(reference)} compositions")
    for name in ("one_shot", "one_shot_metre", "best_of_n"):
        summary = result[name]
        print(f"  {name:<15} match {100.0 * summary['match_rate']:6.2f}%   RMSE {summary['mean_rmse']:.4f}   "
              f"cRMSE {summary['mean_crmse']:.4f}")
    print("  best-of-n by atom count:")
    for name, summary in result["best_of_n_bins"].items():
        if summary["count"] > 0:
            print(f"    {name:>7} atoms  n={summary['count']:<6} match {100.0 * summary['match_rate']:6.2f}%   "
                  f"cRMSE {summary['mean_crmse']:.4f}")
    print(f"\nWrote {out_path}.")


if __name__ == "__main__":
    main()
