"""CSPNet encoder that consumes and predicts a jointly diffused geometry state."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter

from direct_geometry.encoder import (
    GEOMETRY_PROJECTION_KEY,
    DirectGeometryCSPNet,
)
from omg.model.model_utils import prop_indicator

from .data import GEOMETRY_FIELD


JOINT_PROJECTION_KEY = "joint_geometry_projection.weight"
JOINT_OUTPUT_KEY = "geometry_out.weight"


class JointGeometryCSPNet(DirectGeometryCSPNet):
    """
    Distance-readable CSPNet with a continuous per-site state in both directions.

    ``geometry_input=False`` retains the head and objective but severs the
    state-to-structure path, which is the head-only control.  All joint arms
    therefore have the same parameters.
    """

    def __init__(
        self,
        geometry_dimension: int = 4,
        geometry_input: bool = True,
        feature_mode: str = "none",
        **kwargs: Any,
    ) -> None:
        if geometry_dimension <= 0:
            raise ValueError("geometry_dimension must be positive.")
        if feature_mode != "none":
            raise ValueError(
                "Joint-state arms must disable the recomputed direct descriptor; "
                "use the separate R control to test it."
            )
        super().__init__(feature_mode=feature_mode, **kwargs)
        self.geometry_dimension = int(geometry_dimension)
        self.geometry_input = bool(geometry_input)
        self.joint_geometry_projection = nn.Linear(
            self.geometry_dimension, self.hidden_dim, bias=False
        )
        # The structural function is exactly the selected backbone at initialisation.
        with torch.no_grad():
            self.joint_geometry_projection.weight.zero_()
        self.geometry_out = nn.Linear(self.hidden_dim, self.geometry_dimension, bias=False)

    def _convert_inputs(self, x: Data, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        if not hasattr(x, GEOMETRY_FIELD):
            raise ValueError("JointGeometryCSPNet received a state without a geometry field.")
        geometry = getattr(x, GEOMETRY_FIELD)
        expected = (int(x.n_atoms.sum()), self.geometry_dimension)
        if tuple(geometry.shape) != expected:
            raise ValueError(f"Expected geometry state shape {expected}, got {tuple(geometry.shape)}.")
        return (
            x.species,
            x.pos,
            x.cell,
            x.n_atoms,
            x.batch,
            geometry,
        )

    def _forward(
        self,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
        num_atoms: torch.Tensor,
        node2graph: torch.Tensor,
        geometry: torch.Tensor,
        t: torch.Tensor,
        prop: Optional[torch.Tensor] = None,
    ) -> Data:
        neighbors = (
            self.build_neighbors(frac_coords, lattices, num_atoms) if self.uses_neighbors else None
        )
        edges, frac_diff, distance_features = self._graph(
            neighbors, num_atoms, frac_coords, lattices, node2graph
        )
        edge2graph = node2graph[edges[0]]

        node_features = self.node_embedding(atom_types - self.species_shift)
        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = self.atom_latent_emb(torch.cat([node_features, t_per_atom], dim=1))
        if self.geometry_input:
            node_features = node_features + self.joint_geometry_projection(
                geometry.to(node_features.dtype)
            )

        indicator = (
            prop_indicator(batch_size=len(num_atoms), p_uncond=0.2)
            if prop is not None
            else None
        )
        extras = {} if distance_features is None else {"distance_features": distance_features}
        for index in range(self.num_layers):
            if prop is not None:
                node_features = self.adapters[index](
                    node_features, prop, indicator, num_atoms
                )
            node_features = self._modules[f"csp_layer_{index}"](
                node_features,
                frac_coords,
                lattices,
                edges,
                edge2graph,
                frac_diff=frac_diff,
                **extras,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)
        if self.pred_scalar:
            raise ValueError("Joint geometry generation is incompatible with pred_scalar=True.")

        coord_b = self.coord_out(node_features)
        coord_eta = self.coord_out_2(node_features)
        graph_features = scatter(node_features, node2graph, dim=0, reduce="mean")
        lattice_b = self.lattice_out(graph_features).view(-1, 3, 3)
        lattice_eta = self.lattice_out_2(graph_features).view(-1, 3, 3)
        if self.ip:
            lattice_b = torch.einsum("bij,bjk->bik", lattice_b, lattices)
            lattice_eta = torch.einsum("bij,bjk->bik", lattice_eta, lattices)

        geometry_b = self.geometry_out(node_features)
        output = {
            "pos_b": coord_b,
            "pos_eta": coord_eta,
            "cell_b": lattice_b,
            "cell_eta": lattice_eta,
            "geometry_b": geometry_b,
            # ODE training/integration still asks every model result for an eta
            # field even when gamma is null.  It has no trainable parameters.
            "geometry_eta": torch.zeros_like(geometry_b),
        }
        if self.pred_type:
            output.update(
                species_b=self.type_out(node_features),
                species_eta=self.type_out_2(node_features),
            )
        return Data(**output)

    def load_baseline_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load stock encoder weights while accounting only for explicit additions."""
        prepared = dict(state_dict)
        for name, target in self.state_dict().items():
            if name not in prepared or prepared[name].shape == target.shape:
                continue
            source = prepared[name]
            widened = (
                source.dim() == 2
                and target.dim() == 2
                and source.shape[0] == target.shape[0]
                and source.shape[1] < target.shape[1]
            )
            if not widened:
                raise ValueError(
                    f"Parameter {name} has shape {tuple(source.shape)} in the checkpoint "
                    f"but {tuple(target.shape)} in the joint encoder."
                )
            padded = torch.zeros_like(target)
            padded[:, : source.shape[1]] = source.to(target.dtype)
            prepared[name] = padded

        missing, unexpected = self.load_state_dict(prepared, strict=False)
        allowed = {GEOMETRY_PROJECTION_KEY, JOINT_PROJECTION_KEY, JOINT_OUTPUT_KEY}
        unaccounted = [name for name in missing if name not in allowed]
        if unaccounted or unexpected:
            raise ValueError(
                f"The checkpoint does not describe this encoder. Missing {unaccounted}, "
                f"unexpected {list(unexpected)}."
            )
