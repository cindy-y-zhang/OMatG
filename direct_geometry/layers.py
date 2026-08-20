"""
A ``CSPNet`` message-passing layer that also sees the Cartesian length of its edges.

WHY THE BASELINE LAYER CANNOT SEE IT

``CSPLayer`` builds its edge input from the two node states, the nine numbers of the lattice Gram matrix, and a sinusoidal
embedding of the *fractional* offset. Recovering a distance from that means learning the quadratic form
``|dr|^2 = df^T (L L^T) df`` from a Fourier basis of ``df`` multiplied against nine lattice numbers, uniformly over every
cell shape in the dataset. The raw offset is not even an input, only sines and cosines of it.

On a periodic graph there is a second and sharper problem. ``SinusoidsEmbedding`` computes ``sin`` and ``cos`` of
``2 pi k df``, which for integer image shifts satisfies ``sin(2 pi k (df + n)) = sin(2 pi k df)``. The embedding is
therefore *blind to the image*: every image of the same pair of atoms produces a byte-identical edge feature. A periodic
neighbour list can hand the trunk eight distinct neighbours where the fully connected graph offered one, and without a
distance channel the trunk cannot tell them apart. That is why the graph and the edge feature are introduced together and
are not offered as separate options: the graph alone is eight copies of one number.

The layer is otherwise the one it subclasses, down to the initial weights. The appended columns start at zero, so a run
with the channels and a run without begin as the same function and any divergence is something training found.
"""

from typing import Any, Optional
import torch
import torch.nn as nn

from omg.model.encoders.diffcsp_copies import CSPLayer


class DistanceCSPLayer(CSPLayer):
    """
    A ``CSPLayer`` whose edge network additionally reads a radial expansion of the edge length.

    :param distance_dim:
        Width of the radial expansion.
    :type distance_dim: int
    """

    def __init__(self, *args: Any, distance_dim: int, **kwargs: Any) -> None:
        """
        Construct the layer, widening the edge network's first weight and zeroing the new columns.

        :raises ValueError:
            If the expansion is empty, which would make this the baseline layer under a different name.
        """
        super().__init__(*args, **kwargs)
        if distance_dim <= 0:
            raise ValueError(f"The radial expansion must have at least one channel, got {distance_dim}. A layer with "
                             f"none is the baseline layer wearing a different class name, which is not a control "
                             f"anybody asked for.")
        self.distance_dim = distance_dim
        first = self.edge_mlp[0]
        assert isinstance(first, nn.Linear)
        widened = nn.Linear(first.in_features + distance_dim, first.out_features)
        with torch.no_grad():
            widened.weight[:, :first.in_features] = first.weight
            # Zero, so the layer starts as an exact copy of the one it subclasses. Anything else would make a comparison
            # between the two a comparison of initialisations.
            widened.weight[:, first.in_features:] = 0.0
            widened.bias.copy_(first.bias)
        self.edge_mlp[0] = widened
        self.baseline_edge_features = first.in_features

    def edge_model(self, node_features: torch.Tensor, frac_coords: torch.Tensor, lattices: torch.Tensor,
                   edge_index: torch.Tensor, edge2graph: torch.Tensor, frac_diff: Optional[torch.Tensor] = None,
                   distance_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Build the edge features, with the radial expansion appended.

        :param node_features:
            Node states of shape ``(atoms, hidden)``.
        :type node_features: torch.Tensor
        :param frac_coords:
            Fractional coordinates of shape ``(atoms, 3)``.
        :type frac_coords: torch.Tensor
        :param lattices:
            Lattices of shape ``(structures, 3, 3)``.
        :type lattices: torch.Tensor
        :param edge_index:
            Edges of shape ``(2, edges)``, centres first.
        :type edge_index: torch.Tensor
        :param edge2graph:
            Structure of every edge, of shape ``(edges,)``.
        :type edge2graph: torch.Tensor
        :param frac_diff:
            Fractional offset of every edge, of shape ``(edges, 3)``.
            Defaults to None, in which case it is recomputed in the baseline's wrapped convention.
        :type frac_diff: Optional[torch.Tensor]
        :param distance_features:
            Radial expansion of every edge length, of shape ``(edges, distance_dim)``.
        :type distance_features: Optional[torch.Tensor]

        :raises ValueError:
            If the expansion is missing, which would silently reduce this to the baseline layer.

        :return:
            Edge features of shape ``(edges, hidden)``.
        :rtype: torch.Tensor
        """
        if distance_features is None:
            raise ValueError("DistanceCSPLayer was given no radial expansion of the edge lengths. Falling back to the "
                             "baseline edge input would turn a measured factor into an unmeasured one, so this raises.")
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            frac_diff = (frac_coords[edge_index[1]] - frac_coords[edge_index[0]]) % 1.0
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        products = lattices @ lattices.transpose(-1, -2) if self.ip else lattices
        return self.edge_mlp(
            torch.cat([hi, hj, products.reshape(-1, 9)[edge2graph], frac_diff, distance_features], dim=1))

    def forward(self, node_features: torch.Tensor, frac_coords: torch.Tensor, lattices: torch.Tensor,
                edge_index: torch.Tensor, edge2graph: torch.Tensor, frac_diff: Optional[torch.Tensor] = None,
                distance_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Update the nodes, threading the radial expansion through to the edge network.

        :param node_features:
            Node states of shape ``(atoms, hidden)``.
        :type node_features: torch.Tensor
        :param frac_coords:
            Fractional coordinates of shape ``(atoms, 3)``.
        :type frac_coords: torch.Tensor
        :param lattices:
            Lattices of shape ``(structures, 3, 3)``.
        :type lattices: torch.Tensor
        :param edge_index:
            Edges of shape ``(2, edges)``, centres first.
        :type edge_index: torch.Tensor
        :param edge2graph:
            Structure of every edge, of shape ``(edges,)``.
        :type edge2graph: torch.Tensor
        :param frac_diff:
            Fractional offset of every edge, of shape ``(edges, 3)``.
            Defaults to None.
        :type frac_diff: Optional[torch.Tensor]
        :param distance_features:
            Radial expansion of every edge length, of shape ``(edges, distance_dim)``.
        :type distance_features: Optional[torch.Tensor]

        :return:
            Updated node states of shape ``(atoms, hidden)``.
        :rtype: torch.Tensor
        """
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff,
                                        distance_features)
        return node_input + self.node_model(node_features, edge_features, edge_index)

    def copy_baseline_weights(self, source: CSPLayer) -> None:
        """
        Copy every weight of a baseline layer in, leaving the appended columns at zero.

        Copied entry by entry rather than through ``load_state_dict``, because the one deliberately widened weight has a
        different shape: a strict load would reject it and a lax one would silently leave the copied weights at this
        layer's own initialisation, which is the difference between a control and a confound.

        :param source:
            The baseline layer to copy from.
        :type source: omg.model.encoders.diffcsp_copies.CSPLayer
        """
        with torch.no_grad():
            reference = source.state_dict()
            for name, value in self.state_dict().items():
                if name in reference and reference[name].shape == value.shape:
                    value.copy_(reference[name])
            first, previous = self.edge_mlp[0], source.edge_mlp[0]
            first.weight[:, :previous.in_features] = previous.weight
            first.weight[:, previous.in_features:] = 0.0
            first.bias.copy_(previous.bias)
