"""
Measure the ceiling of a rigid-block parameterization of inorganic crystals.

A rigid-body generative model replaces the 3N atomic coordinates of a structure by M block poses in SE(3), one per
coordination polyhedron. That is the parameterization in which the motif hypothesis is actually expressible: a polyhedron
becomes an object with a shape, a position and an orientation, and placing it is a single move rather than a re-timed
component of every atom's straight-line displacement. It is also a large build, so its ceiling is worth knowing first.

The ceiling is measured by giving away everything a generative model would have to predict. Every polyhedron is replaced
by the canonical template of its type, placed at its true centre with the rotation that best fits its true geometry, and
atoms claimed by several polyhedra are placed at the consensus of their predictions. What remains is exactly the
information the parameterization cannot hold: the distortion of an individual polyhedron away from its type, and the
disagreement between polyhedra that share an atom. If the reconstruction matches the original structure, the
parameterization is sound and the remaining problem is learning; if it does not, no amount of learning recovers it.

Templates are fitted on the training split and evaluated on the validation split, so a type seen once is not fitted to
the structure it is scored on. Blocks whose type never appears in training fall back to the coarser template that ignores
vertex composition, and the number that fall back further, to their own geometry, is reported because those are free.

The second half of the measurement is the pose precision the parameterization demands. Oracle poses are perturbed by
Gaussian translation and rotation noise of increasing width, which converts the ceiling into a specification: this is how
accurately a model would have to predict a pose to reach a given match rate.

Usage:

    python -m cgfm.scripts.block_ceiling --data omg/data/mp_20/val.lmdb \\
        --template-data omg/data/mp_20/train.lmdb --workers 8
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, Structure as PymatgenStructure
from tqdm import tqdm
from omg.datamodule import StructureDataset
from ..blocks import Decomposition, Template, decompose, fit_templates, reconstruct


DEFAULT_POSE_NOISE = ((0.0, 0.0), (0.02, 1.0), (0.05, 2.0), (0.10, 5.0), (0.20, 10.0), (0.40, 20.0))
"""Translation standard deviation in Angstrom and rotation standard deviation in degrees of the sensitivity curve."""

LTOL, STOL, ANGLE_TOL = 0.2, 0.3, 5.0
"""Structure matching tolerances, which are the ones the sweep's validation match rate is reported under."""


_DATASET: Optional[StructureDataset] = None
"""Per-worker dataset handle, since LMDB environments cannot be shared across processes."""

_TYPE_KEY_MODE: str = "centre-cn-ligands"
"""Per-worker block type mode."""

_TEMPLATES: dict[tuple, Template] = {}
"""Per-worker template table, used by the reconstruction pass."""

_POSE_NOISE: tuple[tuple[float, float], ...] = DEFAULT_POSE_NOISE
"""Per-worker pose noise levels, used by the reconstruction pass."""


def _initialise_worker(file_path: str, type_key_mode: str, templates: Optional[dict[tuple, Template]] = None,
                       pose_noise: Optional[tuple[tuple[float, float], ...]] = None) -> None:
    """
    Open the dataset inside a worker process and install the shared configuration.

    :param file_path:
        Path of the dataset split.
    :type file_path: str
    :param type_key_mode:
        How block types are defined, passed through to the decomposition.
    :type type_key_mode: str
    :param templates:
        Template table for the reconstruction pass, or None for the decomposition pass.
        Defaults to None.
    :type templates: Optional[dict[tuple, Template]]
    :param pose_noise:
        Pose noise levels for the reconstruction pass, or None for the decomposition pass.
        Defaults to None.
    :type pose_noise: Optional[tuple[tuple[float, float], ...]]
    """
    global _DATASET, _TYPE_KEY_MODE, _TEMPLATES, _POSE_NOISE
    _DATASET = StructureDataset(file_path=file_path, lazy_storage=True, niggli_reduce=False,
                                convert_to_fractional=True, floating_point_precision="64-true")
    _TYPE_KEY_MODE = type_key_mode
    if templates is not None:
        _TEMPLATES = templates
    if pose_noise is not None:
        _POSE_NOISE = pose_noise


def _decompose_structure(index: int) -> Optional[Decomposition]:
    """
    Decompose one structure of the split into overlapping coordination polyhedra.

    :param index:
        Index of the structure within the split.
    :type index: int

    :return:
        The decomposition, or None if the neighbour analysis failed.
    :rtype: Optional[Decomposition]
    """
    assert _DATASET is not None
    structure = _DATASET[index]
    identifier = str(structure.metadata.get("identifier", index))
    return decompose(structure.get_pymatgen_structure(), identifier=identifier, type_key_mode=_TYPE_KEY_MODE)


def _as_pymatgen(decomposition: Decomposition, coords: np.ndarray) -> PymatgenStructure:
    """
    Build a pymatgen structure from Cartesian coordinates on the decomposition's lattice.

    :param decomposition:
        The decomposition the coordinates belong to.
    :type decomposition: Decomposition
    :param coords:
        Cartesian coordinates of shape (N, 3).
    :type coords: numpy.ndarray

    :return:
        The structure.
    :rtype: pymatgen.core.Structure
    """
    species = [Element.from_Z(int(number)) for number in decomposition.numbers]
    return PymatgenStructure(decomposition.lattice, species, coords, coords_are_cartesian=True)


def _score_structure(task: tuple[int, Decomposition]) -> tuple[str, list[Optional[float]], list[float]]:
    """
    Reconstruct one structure at every pose noise level and match each reconstruction against the original.

    :param task:
        Index of the structure within the split, used to seed the pose noise, and its decomposition.
    :type task: tuple[int, Decomposition]

    :return:
        Material identifier, the root-mean-square distance of every noise level or None where the reconstruction did not
        match, and the mean per-atom Cartesian displacement of every noise level in Angstrom.
    :rtype: tuple[str, list[Optional[float]], list[float]]
    """
    index, decomposition = task
    matcher = StructureMatcher(ltol=LTOL, stol=STOL, angle_tol=ANGLE_TOL)
    original = _as_pymatgen(decomposition, decomposition.coords)

    distances: list[Optional[float]] = []
    displacements: list[float] = []
    for level, (translation_sigma, rotation_sigma) in enumerate(_POSE_NOISE):
        # The seed is offset by the structure index and the noise level so that the curve is reproducible and independent
        # of how the work is distributed across processes.
        rng = np.random.default_rng(1_000_003 * index + level)
        rebuilt = reconstruct(decomposition, _TEMPLATES, translation_sigma=translation_sigma,
                              rotation_sigma=rotation_sigma, rng=rng)
        displacements.append(float(np.mean(np.linalg.norm(rebuilt - decomposition.coords, axis=-1))))
        result = matcher.get_rms_dist(original, _as_pymatgen(decomposition, rebuilt))
        distances.append(None if result is None else float(result[0]))
    return decomposition.identifier, distances, displacements


def _run_decomposition(file_path: str, count: int, workers: int, type_key_mode: str,
                       description: str) -> tuple[list[Decomposition], int]:
    """
    Decompose a whole split in parallel.

    :param file_path:
        Path of the dataset split.
    :type file_path: str
    :param count:
        Number of structures to process from the start of the split.
    :type count: int
    :param workers:
        Number of worker processes.
    :type workers: int
    :param type_key_mode:
        How block types are defined.
    :type type_key_mode: str
    :param description:
        Label of the progress bar.
    :type description: str

    :return:
        The successful decompositions and the number of structures whose neighbour analysis failed.
    :rtype: tuple[list[Decomposition], int]
    """
    decompositions, failures = [], 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialise_worker,
                             initargs=(file_path, type_key_mode)) as executor:
        for decomposition in tqdm(executor.map(_decompose_structure, range(count), chunksize=16),
                                  total=count, desc=description, unit=" structures"):
            if decomposition is None:
                failures += 1
            else:
                decompositions.append(decomposition)
    return decompositions, failures


def _collect_instances(decompositions: list[Decomposition],
                       coarse: bool) -> dict[tuple, list[tuple[np.ndarray, tuple[str, ...]]]]:
    """
    Group every observed polyhedron by its type, ready for template fitting.

    :param decompositions:
        Decompositions of the split the templates are fitted on.
    :type decompositions: list[Decomposition]
    :param coarse:
        If True, types are reduced to the central element and the coordination number, dropping vertex composition. This
        is the fallback vocabulary for a type that the evaluation split shows and the training split does not.
    :type coarse: bool

    :return:
        Vertex offsets and vertex species of every block, grouped by type.
    :rtype: dict[tuple, list[tuple[numpy.ndarray, tuple[str, ...]]]]
    """
    instances = defaultdict(list)
    for decomposition in decompositions:
        for block in decomposition.blocks:
            if len(block.ligands) == 0:
                continue
            key = block.type_key[:2] if coarse else block.type_key
            instances[key].append((block.offsets, block.species))
    return instances


def _resolve_templates(evaluation: list[Decomposition], fine: dict[tuple, Template],
                       coarse: dict[tuple, Template]) -> tuple[dict[tuple, Template], Counter]:
    """
    Choose the template every block of the evaluation split is reconstructed from.

    :param evaluation:
        Decompositions of the split being scored.
    :type evaluation: list[Decomposition]
    :param fine:
        Templates keyed by the full block type.
    :type fine: dict[tuple, Template]
    :param coarse:
        Templates keyed by the central element and the coordination number only.
    :type coarse: dict[tuple, Template]

    :return:
        Template of every block type present in the evaluation split, and a count of how each was resolved.
    :rtype: tuple[dict[tuple, Template], collections.Counter]
    """
    resolved, provenance = {}, Counter()
    for decomposition in evaluation:
        for block in decomposition.blocks:
            if len(block.ligands) == 0:
                provenance["singleton block"] += 1
                continue
            if block.type_key in fine:
                resolved[block.type_key] = fine[block.type_key]
                provenance["type template"] += 1
            elif block.type_key[:2] in coarse:
                resolved[block.type_key] = coarse[block.type_key[:2]]
                provenance["composition-blind template"] += 1
            else:
                provenance["own geometry"] += 1
    return resolved, provenance


def _report_decomposition(decompositions: list[Decomposition], failures: int, count: int) -> None:
    """
    Print the statistics of the block decomposition of the evaluation split.

    :param decompositions:
        The successful decompositions.
    :type decompositions: list[Decomposition]
    :param failures:
        Number of structures whose neighbour analysis failed.
    :type failures: int
    :param count:
        Number of structures attempted.
    :type count: int
    """
    atoms = np.array([len(d.coords) for d in decompositions])
    polyhedra = np.array([sum(1 for b in d.blocks if len(b.ligands) > 0) for d in decompositions])
    singletons = np.array([d.num_singletons for d in decompositions])
    vertices = np.array([len(b.ligands) for d in decompositions for b in d.blocks if len(b.ligands) > 0])
    rules = Counter(d.centre_rule for d in decompositions)
    # How many blocks claim each atom, which is the sharing the partition in shells.py had to discard.
    memberships = np.concatenate([np.bincount(np.concatenate([np.array([b.centre for b in d.blocks]),
                                                              *(b.ligands for b in d.blocks)]),
                                              minlength=len(d.coords)) for d in decompositions])
    atomwise_dof = 3 * atoms.sum() + 9 * len(decompositions)
    block_dof = 6 * polyhedra.sum() + 3 * singletons.sum() + 9 * len(decompositions)

    print(f"\nDecomposition of {len(decompositions)} structures, {failures} of {count} failed CrystalNN.")
    for rule, number in rules.most_common():
        print(f"  centres chosen by {rule:<15} {100.0 * number / len(decompositions):.1f} per cent of structures")
    print(f"  atoms per structure               {atoms.mean():.2f}")
    print(f"  polyhedra per structure           {polyhedra.mean():.2f}")
    print(f"  vertices per polyhedron           {vertices.mean():.2f} (max {vertices.max()})")
    print(f"  singleton blocks per structure    {singletons.mean():.3f}")
    print(f"  blocks claiming each atom         {memberships.mean():.2f} "
          f"({100.0 * (memberships > 1).mean():.1f} per cent of atoms shared)")
    print(f"  degrees of freedom                {block_dof} against {atomwise_dof} atomwise, "
          f"a factor {atomwise_dof / block_dof:.2f}")


def _report_templates(templates: dict[tuple, Template], provenance: Counter) -> None:
    """
    Print the statistics of the fitted templates, whose spread is the rigidity of a block type.

    :param templates:
        Template of every block type present in the evaluation split.
    :type templates: dict[tuple, Template]
    :param provenance:
        Count of how each block's template was resolved.
    :type provenance: collections.Counter
    """
    spreads = np.array([template.spread for template in templates.values()])
    counts = np.array([template.count for template in templates.values()])
    weighted = float(np.sum(spreads * counts) / np.sum(counts))

    print(f"\nTemplates: {len(templates)} types used, fitted on the training split.")
    print(f"  rigidity, instance-weighted mean deviation from the type template   {weighted:.3f} Angstrom")
    print(f"  rigidity, unweighted median over types                             {np.median(spreads):.3f} Angstrom")
    print(f"  types whose instances sit within 0.1 Angstrom of the template      "
          f"{100.0 * (spreads < 0.1).mean():.1f} per cent")
    total = sum(provenance.values())
    for label, number in provenance.most_common():
        print(f"  blocks resolved by {label:<30} {number} ({100.0 * number / total:.1f} per cent)")


def _report_ceiling(results: list[tuple[str, list[Optional[float]], list[float]]],
                    pose_noise: tuple[tuple[float, float], ...]) -> None:
    """
    Print the match rate ceiling and the pose precision the parameterization demands.

    :param results:
        Per-structure matching results at every noise level.
    :type results: list[tuple[str, list[Optional[float]], list[float]]]
    :param pose_noise:
        The noise levels, as translation standard deviation in Angstrom and rotation standard deviation in degrees.
    :type pose_noise: tuple[tuple[float, float], ...]
    """
    print(f"\nCeiling under oracle poses, matched at ltol {LTOL}, stol {STOL}, angle_tol {ANGLE_TOL} degrees.")
    print(f"  {'translation':>12} {'rotation':>10} {'match rate':>12} {'mean rmsd':>11} {'displacement':>14}")
    print(f"  {'(Angstrom)':>12} {'(degrees)':>10} {'(per cent)':>12} {'(matched)':>11} {'(Angstrom)':>14}")
    for level, (translation_sigma, rotation_sigma) in enumerate(pose_noise):
        distances = [result[1][level] for result in results]
        matched = [distance for distance in distances if distance is not None]
        displacement = float(np.mean([result[2][level] for result in results]))
        rmsd = f"{np.mean(matched):.4f}" if matched else "n/a"
        print(f"  {translation_sigma:>12.2f} {rotation_sigma:>10.1f} {100.0 * len(matched) / len(distances):>12.2f} "
              f"{rmsd:>11} {displacement:>14.4f}")


def _parse_pose_noise(specification: str) -> tuple[tuple[float, float], ...]:
    """
    Parse the pose noise levels of the sensitivity curve.

    :param specification:
        Comma-separated levels, each a translation standard deviation in Angstrom and a rotation standard deviation in
        degrees joined by a colon, for example "0:0,0.1:5".
    :type specification: str

    :return:
        The noise levels.
    :rtype: tuple[tuple[float, float], ...]

    :raises ValueError:
        If a level is not a colon-separated pair of non-negative numbers.
    """
    levels = []
    for entry in specification.split(","):
        parts = entry.split(":")
        if len(parts) != 2:
            raise ValueError(f"Pose noise level {entry!r} is not a translation and a rotation joined by a colon.")
        translation, rotation = float(parts[0]), float(parts[1])
        if translation < 0.0 or rotation < 0.0:
            raise ValueError(f"Pose noise level {entry!r} is negative.")
        levels.append((translation, rotation))
    return tuple(levels)


def main() -> None:
    """Measure the rigid-block ceiling of one dataset split and print the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="split to score, for example omg/data/mp_20/val.lmdb")
    parser.add_argument("--template-data", required=True, help="split to fit templates on, normally the training split")
    parser.add_argument("--limit", type=int, default=2000, help="number of structures to score")
    parser.add_argument("--template-limit", type=int, default=6000,
                        help="number of structures to fit templates from; templates converge well before this")
    parser.add_argument("--workers", type=int, default=8, help="number of worker processes")
    parser.add_argument("--type-key", choices=("centre-cn", "centre-cn-ligands"), default="centre-cn-ligands",
                        help="what a rigid template is shared across")
    parser.add_argument("--pose-noise", default=",".join(f"{t}:{r}" for t, r in DEFAULT_POSE_NOISE),
                        help="pose noise levels as translation:rotation pairs in Angstrom and degrees")
    arguments = parser.parse_args()
    pose_noise = _parse_pose_noise(arguments.pose_noise)

    def split_size(path: str, limit: int) -> int:
        probe = StructureDataset(file_path=path, lazy_storage=True, niggli_reduce=False, convert_to_fractional=True,
                                 floating_point_precision="64-true")
        size = min(limit, len(probe))
        del probe
        return size

    training, _ = _run_decomposition(arguments.template_data,
                                     split_size(arguments.template_data, arguments.template_limit),
                                     arguments.workers, arguments.type_key, "Decomposing train")
    evaluation_count = split_size(arguments.data, arguments.limit)
    evaluation, failures = _run_decomposition(arguments.data, evaluation_count, arguments.workers,
                                              arguments.type_key, "Decomposing eval ")

    print("\nFitting templates.")
    fine = fit_templates(_collect_instances(training, coarse=False))
    coarse = fit_templates(_collect_instances(training, coarse=True))
    templates, provenance = _resolve_templates(evaluation, fine, coarse)

    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_initialise_worker,
                             initargs=(arguments.data, arguments.type_key, templates, pose_noise)) as executor:
        results = list(tqdm(executor.map(_score_structure, list(enumerate(evaluation)), chunksize=8),
                            total=len(evaluation), desc="Reconstructing", unit=" structures"))

    _report_decomposition(evaluation, failures, evaluation_count)
    _report_templates(templates, provenance)
    _report_ceiling(results, pose_noise)


if __name__ == "__main__":
    main()
