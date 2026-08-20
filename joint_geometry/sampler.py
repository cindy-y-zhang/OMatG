"""Base-distribution wrappers for jointly generated geometry states."""

from __future__ import annotations

import torch

from omg.datamodule import OMGData
from omg.sampler import Sampler

from .data import GEOMETRY_FIELD


FORBIDDEN_DEPLOYABLE_FIELDS = (
    "geometry_oracle_base",
    "geometry_oracle_target",
    "geometry_target",
)
"""Clean-target fields that may never enter a deployable inference state."""


class JointGeometrySampler(Sampler):
    """Delegate structural sampling and append an independent Gaussian state."""

    def __init__(self, structural_sampler: Sampler, geometry_dimension: int) -> None:
        super().__init__()
        if geometry_dimension <= 0:
            raise ValueError("geometry_dimension must be positive.")
        self.structural_sampler = structural_sampler
        self.geometry_dimension = int(geometry_dimension)

    def sample_p_0(self, x_1: OMGData) -> OMGData:
        if not hasattr(x_1, GEOMETRY_FIELD):
            raise ValueError("The clean batch carries no precomputed geometry endpoint.")
        target = getattr(x_1, GEOMETRY_FIELD)
        expected = (int(x_1.n_atoms.sum()), self.geometry_dimension)
        if tuple(target.shape) != expected:
            raise ValueError(
                f"Expected clean geometry endpoints of shape {expected}, got {tuple(target.shape)}."
            )
        x_0 = self.structural_sampler.sample_p_0(x_1)
        # Independent noise is the product prior required by joint flow matching.
        setattr(x_0, GEOMETRY_FIELD, torch.randn_like(target))
        leaked = [name for name in FORBIDDEN_DEPLOYABLE_FIELDS if hasattr(x_0, name)]
        if leaked:
            raise ValueError(
                "The deployable joint sampler received forbidden clean-target fields: "
                f"{leaked}."
            )
        return x_0


class OracleGeometrySampler(JointGeometrySampler):
    """Diagnostic sampler that retains the forbidden clean endpoint privately."""

    def sample_p_0(self, x_1: OMGData) -> OMGData:
        x_0 = super().sample_p_0(x_1)
        # These fields are consumed only by OracleGeometryInterpolants.  Their
        # presence is deliberate target leakage and must never enter J/P/H.
        x_0.geometry_oracle_base = x_0.geometry.clone()
        x_0.geometry_oracle_target = x_1.geometry.clone()
        return x_0
