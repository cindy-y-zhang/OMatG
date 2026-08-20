"""Write per-composition CSP outcomes for paired statistical comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from direct_geometry.scripts.evaluate import ANGLE_TOL, LTOL, STOL, reference_structures
from omg.analysis import ValidAtoms, match_rmsds
from omg.utils import xyz_reader


def target_errors(generated, reference) -> tuple[float, float]:
    """Return permutation-aware fractional-position and raw-cell endpoint MSEs."""
    generated_numbers = np.asarray(generated.numbers)
    reference_numbers = np.asarray(reference.numbers)
    if sorted(generated_numbers.tolist()) != sorted(reference_numbers.tolist()):
        raise ValueError("Generated and reference structures have different compositions.")
    generated_positions = np.asarray(generated.get_scaled_positions(wrap=True))
    reference_positions = np.asarray(reference.get_scaled_positions(wrap=True))
    squared_errors = []
    for atomic_number in np.unique(reference_numbers):
        generated_indices = np.flatnonzero(generated_numbers == atomic_number)
        reference_indices = np.flatnonzero(reference_numbers == atomic_number)
        difference = (
            generated_positions[generated_indices, None, :]
            - reference_positions[None, reference_indices, :]
        )
        difference -= np.round(difference)
        cost = np.mean(difference**2, axis=-1)
        rows, columns = linear_sum_assignment(cost)
        squared_errors.extend(cost[rows, columns].tolist())
    position_mse = float(np.mean(squared_errors))
    cell_difference = np.asarray(generated.cell) - np.asarray(reference.cell)
    return position_mse, float(np.mean(cell_difference**2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--draw", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()

    generated_atoms = xyz_reader(Path(arguments.generated))
    reference_atoms = reference_structures(Path(arguments.reference))
    if len(generated_atoms) > len(reference_atoms):
        raise ValueError("Generated output is longer than its reference split.")
    reference_atoms = reference_atoms[: len(generated_atoms)]
    generated = ValidAtoms.get_valid_atoms(
        generated_atoms,
        desc="Validating generated structures",
        skip_validation=True,
        number_cpus=1,
    )
    reference = ValidAtoms.get_valid_atoms(
        reference_atoms,
        desc="Validating reference structures",
        skip_validation=True,
        number_cpus=1,
    )
    (
        match_rate,
        mean_rmsd,
        _,
        _,
        rmsds,
        _,
        corrected_rmsd,
        _,
    ) = match_rmsds(
        generated,
        reference,
        ltol=LTOL,
        stol=STOL,
        angle_tol=ANGLE_TOL,
        number_cpus=arguments.workers,
        enable_progress_bar=True,
    )
    outcomes = []
    for index, rmsd in enumerate(rmsds):
        position_mse, cell_mse = target_errors(generated_atoms[index], reference_atoms[index])
        outcomes.append(
            {
                "index": index,
                "match": rmsd is not None,
                "rmsd": None if rmsd is None else float(rmsd),
                "corrected_rmsd": STOL if rmsd is None else float(rmsd),
                "position_target_mse": position_mse,
                "cell_target_mse": cell_mse,
            }
        )
    report = {
        "arm": arguments.arm,
        "seed": arguments.seed,
        "draw": arguments.draw,
        "generated": arguments.generated,
        "reference": arguments.reference,
        "num_compositions": len(outcomes),
        "match_rate": float(match_rate),
        "mean_rmsd": float(mean_rmsd) if np.isfinite(mean_rmsd) else None,
        "corrected_rmsd": float(corrected_rmsd),
        "position_target_mse": float(
            np.mean([entry["position_target_mse"] for entry in outcomes])
        ),
        "cell_target_mse": float(
            np.mean([entry["cell_target_mse"] for entry in outcomes])
        ),
        "tolerances": {"ltol": LTOL, "stol": STOL, "angle_tol": ANGLE_TOL},
        "outcomes": outcomes,
    }
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(
        f"{arguments.arm} seed {arguments.seed}: "
        f"{100.0 * report['match_rate']:.2f}% over {len(outcomes)} compositions"
    )


if __name__ == "__main__":
    main()
