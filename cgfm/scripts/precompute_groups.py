"""
Precompute the fixed partitions of a dataset split.

Both fixed arms are built here, in one pass, because they have to agree on the number of groups. CrystalNN decides that
number: the coordination shells of a structure are what they are, and forcing them to a round fraction of the atom count
would defeat the arm. The k-medoids partition is then computed at exactly that number, so the two fixed arms and the
learned arm all compress each structure by the same factor and differ only in which atoms share a group.

Structures whose CrystalNN analysis fails fall back to the group count round(N / 5) from the original specification and
receive the k-medoids partition in both files. The script reports how often that happens.

Usage:

    python -m cgfm.scripts.precompute_groups --data omg/data/mpts_52/train.lmdb --out-dir cgfm/groups/mpts_52
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm
from omg.datamodule import StructureDataset
from ..groupfile import GroupTable
from ..kmedoids import min_image_distance_matrix, periodic_kmedoids
from ..shells import coordination_shell_partition


FALLBACK_ATOMS_PER_GROUP = 5
"""Target group size used to choose the group count when the CrystalNN analysis fails."""


_DATASET: Optional[StructureDataset] = None
"""Per-worker dataset handle, since LMDB environments cannot be shared across processes."""

_SEED: int = 0
"""Per-worker seed offset for k-medoids."""


def _initialise_worker(file_path: str, seed: int) -> None:
    """
    Open the dataset inside a worker process.

    :param file_path:
        Path of the dataset split.
    :type file_path: str
    :param seed:
        Base seed for k-medoids.
    :type seed: int
    """
    global _DATASET, _SEED
    _DATASET = StructureDataset(file_path=file_path, lazy_storage=True, niggli_reduce=False,
                                convert_to_fractional=True, floating_point_precision="64-true")
    _SEED = seed


def _partition_structure(index: int) -> tuple[np.ndarray, np.ndarray, bool, str]:
    """
    Compute both fixed partitions of one structure.

    :param index:
        Index of the structure within the split.
    :type index: int

    :return:
        Coordination-shell labels, k-medoids labels at the same group count, whether the CrystalNN analysis failed, and
        the material identifier of the structure.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, bool, str]
    """
    assert _DATASET is not None
    structure = _DATASET[index]
    identifier = str(structure.metadata.get("identifier", index))
    frac = structure.pos.numpy()
    cell = structure.cell.numpy()

    distances = min_image_distance_matrix(frac, cell)
    shell_labels = coordination_shell_partition(structure.get_pymatgen_structure(), distances)
    failed = shell_labels is None
    if failed:
        num_groups = max(1, round(len(frac) / FALLBACK_ATOMS_PER_GROUP))
    else:
        num_groups = len(np.unique(shell_labels))

    # The seed is offset by the structure index so that clustering is reproducible per structure and independent of how
    # the work is distributed across processes.
    medoid_labels = periodic_kmedoids(distances, num_groups, seed=_SEED + index)
    return (medoid_labels if failed else shell_labels), medoid_labels, failed, identifier


def _relabel_consecutively(labels: np.ndarray) -> np.ndarray:
    """
    Relabel a partition so that its labels are exactly 0 to K - 1.

    :param labels:
        Group label of every atom.
    :type labels: numpy.ndarray

    :return:
        Equivalent partition with consecutive labels.
    :rtype: numpy.ndarray
    """
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(np.int64)


def _report(name: str, table: GroupTable) -> None:
    """
    Print summary statistics of a partition table.

    :param name:
        Name of the partition method.
    :type name: str
    :param table:
        The table to summarise.
    :type table: GroupTable
    """
    sizes = np.concatenate([np.bincount(table[i][0]) for i in range(len(table))])
    atoms = np.diff(table.offsets)
    print(f"{name}: {len(table)} structures, {atoms.mean():.1f} atoms and {table.num_groups.mean():.2f} groups per "
          f"structure, group size mean {sizes.mean():.2f} max {sizes.max()}, "
          f"singletons {100.0 * (sizes == 1).mean():.1f} per cent, "
          f"compression {atoms.sum() / table.num_groups.sum():.2f} atoms per group")


def main() -> None:
    """Precompute both fixed partitions of one dataset split and write them to disk."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="path of the dataset split, for example a .lmdb file")
    parser.add_argument("--out-dir", required=True, help="directory in which the group files are written")
    parser.add_argument("--seed", type=int, default=0, help="base seed for k-medoids seeding")
    parser.add_argument("--workers", type=int, default=8, help="number of worker processes")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N structures, for smoke tests")
    arguments = parser.parse_args()

    probe = StructureDataset(file_path=arguments.data, lazy_storage=True, niggli_reduce=False,
                             convert_to_fractional=True, floating_point_precision="64-true")
    count = len(probe) if arguments.limit is None else min(arguments.limit, len(probe))
    del probe

    shell_labels, medoid_labels, identifiers, failures = [], [], [], 0
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_initialise_worker,
                             initargs=(arguments.data, arguments.seed)) as executor:
        for shells, medoids, failed, identifier in tqdm(
                executor.map(_partition_structure, range(count), chunksize=32),
                total=count, desc="Partitioning", unit=" structures"):
            shell_labels.append(_relabel_consecutively(shells))
            medoid_labels.append(_relabel_consecutively(medoids))
            identifiers.append(identifier)
            failures += int(failed)

    split = Path(arguments.data).stem
    out_dir = Path(arguments.out_dir)
    for method, labels in (("shells", shell_labels), ("kmedoids", medoid_labels)):
        table = GroupTable.from_labels(labels, identifiers, method=method)
        table.validate()
        table.save(out_dir / f"{split}.{method}.npz")
        _report(method, table)
    if failures > 0:
        print(f"CrystalNN failed on {failures} of {count} structures, which fell back to the k-medoids partition at "
              f"round(N / {FALLBACK_ATOMS_PER_GROUP}) groups.")
    print(f"Wrote group files to {out_dir}.")


if __name__ == "__main__":
    main()
