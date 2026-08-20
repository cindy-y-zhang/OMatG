"""Base-distribution wrapper that carries the motif census into the model's input state."""

from __future__ import annotations

from omg.datamodule import OMGData
from omg.sampler import Sampler

from .data import CENSUS_FIELD


class MotifCensusSampler(Sampler):
    """Delegate structural sampling and copy the census onto the sampled state.

    The census is conditioning, not a diffused field: it is identical on ``x_0`` and
    ``x_1`` and therefore identical at every point of the path. Attaching it here is what
    makes it visible to the model, because both the training interpolation and the
    inference integration build their states by cloning ``x_0``.
    """

    def __init__(self, structural_sampler: Sampler, census_dimension: int) -> None:
        super().__init__()
        if census_dimension <= 0:
            raise ValueError("census_dimension must be positive.")
        self.structural_sampler = structural_sampler
        self.census_dimension = int(census_dimension)

    def sample_p_0(self, x_1: OMGData) -> OMGData:
        if not hasattr(x_1, CENSUS_FIELD):
            raise ValueError("The clean batch carries no precomputed motif census.")
        census = getattr(x_1, CENSUS_FIELD)
        expected = (len(x_1.n_atoms), self.census_dimension)
        if tuple(census.shape) != expected:
            raise ValueError(
                f"Expected a motif census of shape {expected}, got {tuple(census.shape)}."
            )
        x_0 = self.structural_sampler.sample_p_0(x_1)
        setattr(x_0, CENSUS_FIELD, census.clone())
        return x_0
