"""
CSPNet encoder over rigid blocks rather than atoms.

Each node is a coordination polyhedron. The input embedding concatenates the centre element, the current coordination
number (including the mask token), the flattened current rotation and the time embedding. The heads predict a
translation velocity, a body-frame rotation velocity, a lattice velocity and 13 CN logits. Impossible CN logits for a
centre element are masked from a train-only validity table.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.data import Data
from omg.globals import MAX_ATOM_NUM
from omg.model.encoders.cspnet_full import CSPNetFull
from .blocks import CN_CLASSES


class BlockCSPNet(CSPNetFull):
    """
    CSPNet whose nodes are blocks and whose extra heads emit rotation velocities and CN logits.

    :param hidden_dim:
        Hidden width of the message-passing layers.
        Defaults to 512.
    :type hidden_dim: int
    :param latent_dim:
        Width of the time embedding concatenated onto each node.
        Defaults to 256.
    :type latent_dim: int
    :param num_layers:
        Number of CSP layers.
        Defaults to 6.
    :type num_layers: int
    :param cn_classes:
        Number of CN logits, excluding the mask token.
        Defaults to CN_CLASSES.
    :type cn_classes: int
    """

    def __init__(self, hidden_dim: int = 512, latent_dim: int = 256, num_layers: int = 6,
                 cn_classes: int = CN_CLASSES, **kwargs) -> None:
        """Construct the encoder."""
        kwargs.setdefault("pred_type", True)
        kwargs.setdefault("max_atoms", MAX_ATOM_NUM)
        super().__init__(hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers, **kwargs)
        self.cn_classes = cn_classes
        self.cn_embedding = nn.Embedding(cn_classes + 1, hidden_dim)
        self.rot_embedding = nn.Linear(9, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim * 3 + latent_dim, hidden_dim)
        self.rot_out = nn.Linear(hidden_dim, 3, bias=False)
        self.cn_out = nn.Linear(hidden_dim, cn_classes)
        self.register_buffer("cn_valid", torch.ones(MAX_ATOM_NUM + 1, cn_classes, dtype=torch.bool))

    def set_cn_mask(self, cn_valid: torch.Tensor) -> None:
        """
        Install the train-only CN validity mask.

        :param cn_valid:
            Boolean mask of shape (MAX_ATOM_NUM + 1, cn_classes), indexed ``[Z, CN]``.
        :type cn_valid: torch.Tensor
        """
        if cn_valid.shape != self.cn_valid.shape:
            raise ValueError(f"CN mask has shape {tuple(cn_valid.shape)}, expected {tuple(self.cn_valid.shape)}.")
        self.cn_valid = cn_valid.to(device=self.cn_valid.device, dtype=torch.bool)

    def _forward(self, atom_types, frac_coords, lattices, num_atoms, node2graph, t, prop=None,
                 block_type=None, rotations=None, edges=None):
        """Run message passing and emit translation, rotation, lattice and CN heads."""
        if edges is None:
            edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        else:
            frac_diff = (frac_coords[edges[1]] - frac_coords[edges[0]]) % 1.0
        edge2graph = node2graph[edges[0]]
        node_features = self.node_embedding(atom_types - self.species_shift)
        cn_features = self.cn_embedding(block_type)
        rot_features = self.rot_embedding(rotations.reshape(rotations.shape[0], 9))
        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([node_features, cn_features, rot_features, t_per_atom], dim=1)
        node_features = self.atom_latent_emb(node_features)

        for index in range(self.num_layers):
            node_features = self._modules[f"csp_layer_{index}"](
                node_features, frac_coords, lattices, edges, edge2graph, frac_diff=frac_diff)

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        coord_b = self.coord_out(node_features)
        coord_eta = self.coord_out_2(node_features)
        rot_b = self.rot_out(node_features)
        graph_features = scatter(node_features, node2graph, dim=0, reduce="mean")

        lattice_b = self.lattice_out(graph_features).view(-1, 3, 3)
        lattice_eta = self.lattice_out_2(graph_features).view(-1, 3, 3)
        if self.ip:
            lattice_b = torch.einsum("bij,bjk->bik", lattice_b, lattices)
            lattice_eta = torch.einsum("bij,bjk->bik", lattice_eta, lattices)

        type_b = self.cn_out(node_features)
        valid = self.cn_valid[atom_types]
        type_b = type_b.masked_fill(~valid, -1.0e9)
        return Data(species_b=type_b, species_eta=type_b, pos_b=coord_b, pos_eta=coord_eta,
                    cell_b=lattice_b, cell_eta=lattice_eta, rot_b=rot_b, rot_eta=rot_b,
                    block_type_b=type_b, block_type_eta=type_b)

    def _convert_inputs(self, x, **kwargs):
        """Unpack a block graph into the tensors the encoder forward pass consumes."""
        return (x.species, x.pos, x.cell, x.n_atoms, x.batch, x.block_type, x.rot)

    def forward(self, x, t, prop=None, **kwargs):
        """
        Encode a block graph.

        The parent ``Encoder.forward`` unpacks ``_convert_inputs`` into ``_forward(*x, t, prop)``. This override
        threads the extra block fields through as keyword arguments instead.
        """
        atom_types, frac_coords, lattices, num_atoms, node2graph, block_type, rotations = self._convert_inputs(x)
        edges = getattr(x, "block_edge_index", None)
        if self.edge_style == "fc" and edges is None:
            edges, _ = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
            x.block_edge_index = edges
        return self._forward(atom_types, frac_coords, lattices, num_atoms, node2graph, t, prop,
                             block_type=block_type, rotations=rotations, edges=edges)
