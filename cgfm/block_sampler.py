"""
Base-distribution sampler for rigid-block flows.

Oracle mode copies the target coordination number into the base state. Joint mode masks it. Translations, rotations and
cells are drawn the same way in both modes, and in that order, so a given seed produces identical continuous draws.
"""

import numpy as np
import torch
from omg.datamodule import OMGData
from omg.sampler import Sampler
from omg.sampler.abstracts import CellDistribution, PositionDistribution
from .so3 import sample_uniform


class BlockSampler(Sampler):
    """
    Sampler of block translations, orientations, lattices and (optionally) coordination numbers.

    :param pos_distribution:
        Base distribution of fractional translations.
    :type pos_distribution: PositionDistribution
    :param cell_distribution:
        Base distribution of lattices.
    :type cell_distribution: CellDistribution
    :param cn_mode:
        ``"oracle"`` copies the target CN; ``"joint"`` masks it.
        Defaults to "joint".
    :type cn_mode: str

    :raises ValueError:
        If ``cn_mode`` is not ``oracle`` or ``joint``.
    """

    def __init__(self, pos_distribution: PositionDistribution, cell_distribution: CellDistribution,
                 cn_mode: str = "joint") -> None:
        """Construct the sampler."""
        super().__init__()
        if cn_mode not in ("oracle", "joint"):
            raise ValueError(f"Unknown cn_mode {cn_mode!r}.")
        self._pos_distribution = pos_distribution
        self._cell_distribution = cell_distribution
        self.cn_mode = cn_mode

    def sample_p_0(self, x_1: OMGData) -> OMGData:
        """
        Sample the base distribution given a batch of block graphs from the training data.

        :param x_1:
            Batch of target block graphs.
        :type x_1: OMGData

        :return:
            The base sample, carrying the target's templates, sharing map and composition unchanged.
        :rtype: OMGData
        """
        x_0 = x_1.clone()
        if hasattr(self._pos_distribution, "sample_batch"):
            pos, _ = self._pos_distribution.sample_batch(x_1.pos, True)
        else:
            pos, _ = self._pos_distribution(x_1.pos, True)
        x_0.pos = torch.as_tensor(pos, dtype=x_1.pos.dtype, device=x_1.pos.device)
        x_0.pos_is_fractional = torch.ones_like(x_1.pos_is_fractional)
        if hasattr(self._cell_distribution, "sample_batch"):
            sampled_cells = self._cell_distribution.sample_batch(x_1.cell)
            x_0.cell = torch.as_tensor(sampled_cells, dtype=x_1.cell.dtype, device=x_1.cell.device)
        else:
            cells = []
            for index in range(len(x_1.n_atoms)):
                cells.append(torch.as_tensor(self._cell_distribution(x_1.cell[index]), dtype=x_1.cell.dtype))
            x_0.cell = torch.stack(cells, dim=0).to(x_1.cell.device)
        x_0.rot = sample_uniform(int(x_1.pos.shape[0]), device=x_1.pos.device, dtype=x_1.pos.dtype)
        x_0.species = x_1.species.clone()
        if self.cn_mode == "oracle":
            x_0.block_type = x_1.block_type.clone()
        else:
            x_0.block_type = torch.zeros_like(x_1.block_type)
        return x_0
