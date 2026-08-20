"""Special diagnostic interpolant collections for the joint-state study."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from omg.si.stochastic_interpolants import StochasticInterpolants


class OracleGeometryInterpolants(StochasticInterpolants):
    """
    Integrate geometry along its exact target path while predicting structure.

    Training remains ordinary joint flow matching.  Only inference is
    overridden, making this a ceiling test that is intentionally undeployable.
    """

    def integrate(self, x_0: Data, model_function, save_intermediate: bool = False):
        if not hasattr(x_0, "geometry_oracle_base") or not hasattr(
            x_0, "geometry_oracle_target"
        ):
            raise ValueError(
                "OracleGeometryInterpolants requires an OracleGeometrySampler base state."
            )
        oracle_velocity = x_0.geometry_oracle_target - x_0.geometry_oracle_base

        def oracle_model(state: Data, time: torch.Tensor) -> Data:
            result = model_function(state, time)
            # Structural heads consume the exact evolving geometry state; the
            # model's own geometry prediction is ignored for this ceiling.
            result.geometry_b = oracle_velocity
            result.geometry_eta = torch.zeros_like(oracle_velocity)
            return result

        return super().integrate(x_0, oracle_model, save_intermediate=save_intermediate)
