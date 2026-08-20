"""
Reading MPTS-52 splits and putting them on the baseline's probability path.

Every measurement in this package -- the Gate DG0 audit, the information probes, the target-MSE evaluation -- describes
the geometry the denoiser is handed at some denoising time, and none of them describes a crystal. So there is exactly one
definition of that geometry here, and it is built from the baseline configuration's own base distributions and
interpolants rather than from an approximation of them.

WHY THIS IS NOT A DETAIL

An earlier draft of the audit interpolated towards a lognormal jitter of each structure's own cell, on the reasonable
assumption that any smooth path from noise to data would do. It does not. The baseline's cell prior is
``InformedLatticeDistribution`` fitted to MPTS-52, whose lengths have log-means near 1.66, 1.84 and 2.12 -- systematically
*smaller* than the cells of the 25-atom structures it is a prior for. Real prior draws are therefore about twice the
number density of the data, and the noisy end of the path is where a radius graph is at its largest. Measured on the
approximated path the periodic graph cost 1.7 times the control; measured on the real one it cost 2.5, which is the
difference between passing Gate DG0 and failing it.
"""

from typing import Iterator, Optional
import torch
from omg.datamodule import OMGDataset, StructureDataset
from omg.sampler.cell_distributions import InformedLatticeDistribution
from omg.sampler.position_distributions import UniformPositionDistribution
from omg.si.interpolants import LinearInterpolant, PeriodicLinearInterpolant
from omg.utils import DataField, reshape_t
from torch_geometric.data import Batch


DATASET_NAME = "mpts_52"
"""
Dataset whose fitted lattice prior is used.

Named here rather than passed in, because it has to be the one the baseline configuration names. A prior fitted to a
different dataset would put the structures at a different density and every count measured on the path would shift.
"""


def load_split(path: str) -> OMGDataset:
    """
    Open an LMDB split in the form the encoder receives it.

    Wrapped in ``OMGDataset`` rather than read from ``StructureDataset`` directly, because that wrapper is what converts
    Cartesian positions to the fractional ones the encoder expects. Auditing Cartesian coordinates as though they were
    fractional would describe a crystal a few hundred Angstrom across and every conclusion would be wrong.

    :param path:
        Path of the LMDB split.
    :type path: str

    :return:
        The split.
    :rtype: omg.datamodule.OMGDataset
    """
    return OMGDataset(StructureDataset(file_path=path, lazy_storage=True, niggli_reduce=False))


def collate(dataset: OMGDataset, indices: list[int]) -> Batch:
    """
    Batch a chosen set of structures.

    :param dataset:
        The split to read from.
    :type dataset: omg.datamodule.OMGDataset
    :param indices:
        Structure indices to batch, in the order they should appear.
    :type indices: list[int]

    :raises AssertionError:
        If the split did not yield fractional coordinates, which would silently reinterpret Angstrom as cell fractions.

    :return:
        The batched structures.
    :rtype: torch_geometric.data.Batch
    """
    batch = Batch.from_data_list([dataset[int(index)] for index in indices])
    assert bool(batch.pos_is_fractional.all()), "the split did not yield fractional coordinates"
    return batch


def sample_structures(path: str, count: int, seed: int) -> Batch:
    """
    Read a random batch of structures from a split.

    :param path:
        Path of the LMDB split.
    :type path: str
    :param count:
        Number of structures.
    :type count: int
    :param seed:
        Seed of the selection.
    :type seed: int

    :return:
        The batched structures.
    :rtype: torch_geometric.data.Batch
    """
    dataset = load_split(path)
    chosen = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count]
    return collate(dataset, chosen.tolist())


def iterate_structures(dataset: OMGDataset, batch_size: int,
                       limit: Optional[int] = None) -> Iterator[tuple[int, Batch]]:
    """
    Walk a whole split in its stored order, in batches.

    In stored order, and yielding each batch's first structure index, because the precomputed label tables are indexed by
    that order through their atom offsets. Shuffling here would silently misalign every label.

    :param dataset:
        The split to walk.
    :type dataset: omg.datamodule.OMGDataset
    :param batch_size:
        Number of structures per batch.
    :type batch_size: int
    :param limit:
        Number of structures to read, or None for all of them.
        Defaults to None.
    :type limit: Optional[int]

    :return:
        Pairs of first structure index and batch.
    :rtype: Iterator[tuple[int, torch_geometric.data.Batch]]
    """
    total = len(dataset) if limit is None else min(int(limit), len(dataset))
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        yield start, collate(dataset, list(range(start, stop)))


def interpolate(batch: Batch, time_value: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Put a batch on the baseline's probability path at one denoising time.

    The random state is forked rather than set, so that calling this inside a training loop does not perturb the stream
    the loop draws its own randomness from. Determinism still comes from the seed alone.

    :param batch:
        The reference structures.
    :type batch: torch_geometric.data.Batch
    :param time_value:
        Interpolation time in ``[0, 1]``. Zero is the base draw, one is the data.
    :type time_value: float
    :param seed:
        Seed of the base draw.
    :type seed: int

    :return:
        Interpolated fractional coordinates and lattices.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base_pos, _ = UniformPositionDistribution().sample_batch(batch.pos, True)
        base_cell = InformedLatticeDistribution(dataset_name=DATASET_NAME).sample_batch(batch.cell)
    times = torch.full((len(batch.n_atoms),), time_value, dtype=batch.pos.dtype)

    position_interpolant = PeriodicLinearInterpolant()
    frac = position_interpolant.interpolate(reshape_t(times, batch.n_atoms, DataField.pos), base_pos, batch.pos)
    # The periodic interpolant unwraps the target to the nearest image of the base draw, so the interpolated point can sit
    # outside the unit cell. Applying the path's own corrector is what the sampler does, and it is what makes the
    # fractional coordinates the neighbour builder receives comparable to the ones training passes in.
    frac = position_interpolant.get_corrector().correct(frac)
    cell = LinearInterpolant().interpolate(reshape_t(times, batch.n_atoms, DataField.cell), base_cell, batch.cell)
    return frac, cell


def displacement(batch: Batch, frac: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """
    Return how far every atom has moved from its crystal position, in Angstrom.

    Measured in the *interpolated* cell and against the nearest periodic image, which is the only reading that means
    anything: the cell is being interpolated too, so a fractional coordinate that has not changed still corresponds to a
    moved atom, and an atom that has drifted across a cell boundary has not moved far.

    Reported by the probes so that a denoising time can be quoted as a corruption in Angstrom. Gate DG2 is stated at
    "near 0.2 Angstrom", which is a physical tolerance and not a time.

    :param batch:
        The reference structures.
    :type batch: torch_geometric.data.Batch
    :param frac:
        Interpolated fractional coordinates of shape ``(atoms, 3)``.
    :type frac: torch.Tensor
    :param cell:
        Interpolated lattices of shape ``(structures, 3, 3)``.
    :type cell: torch.Tensor

    :return:
        Displacement of every atom, of shape ``(atoms,)``, in Angstrom.
    :rtype: torch.Tensor
    """
    offset = frac - batch.pos
    offset = offset - torch.round(offset)
    return torch.einsum("ak,akj->aj", offset, cell[batch.batch]).norm(dim=-1)
