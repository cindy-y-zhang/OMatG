"""
Learned anchor-and-membership grouping network.

A small periodic message-passing network reads the clean target structure and produces, for each structure, a score for
every atom to act as a group anchor and a compatibility score between every atom and every selected anchor. The number
of anchors is fixed to the same per-structure group count the precomputed partitions use, so all coarse-to-fine arms
work at an identical compression ratio and differ only in which atoms end up together.

Three constraints keep the partition non-degenerate without any auxiliary loss term, as the experiment requires:

- the number of groups is fixed rather than learned, so the all-in-one-group and all-singleton solutions are unreachable;
- every selected anchor is assigned to its own group, so no group can be empty;
- membership is restricted to anchors within a locality radius, so groups stay at coordination scale.

Assignments are soft. Endpoint exactness of the path only needs the bump to vanish at t = 0 and t = 1, not the
group-mean operator to be a projector, so nothing requires a hard partition during training. A soft assignment is
differentiable everywhere with no straight-through estimator, and the temperature is annealed towards a hard partition
so that the coarse and fine schedules become exact and the final partition can be compared against coordination
environments.

Anchor selection itself is discrete. Gradient still reaches the anchor scorer because each anchor's score enters its
column of the membership logits, the gating trick used by top-k graph pooling.
"""

from typing import Optional
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from .graph import dense_positions, gaussian_basis, neighbour_lists, pair_distances
from .grouping import Grouping, read_num_groups


class AnchorMembershipGrouper(nn.Module, Grouping):
    """
    Grouping network that selects anchors and assigns every atom to a nearby anchor.

    :param hidden_dim:
        Width of the node and edge representations.
        Defaults to 128.
    :type hidden_dim: int
    :param num_layers:
        Number of message-passing rounds.
        Defaults to 3.
    :type num_layers: int
    :param num_basis:
        Number of radial basis functions used to embed interatomic distances.
        Defaults to 32.
    :type num_basis: int
    :param cutoff:
        Neighbour cutoff of the message-passing graph, in Angstrom.
        Defaults to 6.0.
    :type cutoff: float
    :param max_neighbours:
        Largest number of neighbours kept per atom.
        Defaults to 24.
    :type max_neighbours: int
    :param locality_radius:
        Largest distance at which an atom may join an anchor's group, in Angstrom. Atoms with no anchor inside this
        radius fall back to their nearest anchor.
        Defaults to 5.0.
    :type locality_radius: float
    :param temperature:
        Initial softmax temperature of the membership distribution. Annealed by GroupingTemperatureSchedule.
        Defaults to 1.0.
    :type temperature: float

    :raises ValueError:
        If any size or radius is not positive.
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, num_basis: int = 32, cutoff: float = 6.0,
                 max_neighbours: int = 24, locality_radius: float = 5.0, temperature: float = 1.0) -> None:
        """Construct the grouping network."""
        super().__init__()
        if min(hidden_dim, num_layers, num_basis, max_neighbours) <= 0:
            raise ValueError("The network sizes must be positive.")
        if cutoff <= 0.0 or locality_radius <= 0.0 or temperature <= 0.0:
            raise ValueError("The cutoff, locality radius and temperature must be positive.")
        self._hidden_dim = hidden_dim
        self._num_basis = num_basis
        self._cutoff = cutoff
        self._max_neighbours = max_neighbours
        self._locality_radius = locality_radius
        self.temperature = float(temperature)

        # Atomic numbers of the benchmark datasets fit comfortably below 119.
        self.species_embedding = nn.Embedding(119, hidden_dim)
        self.edge_networks = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim + num_basis, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_layers)])
        self.node_networks = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_layers)])
        self.anchor_score = nn.Linear(hidden_dim, 1)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        # The final layer carries no bias: its output is added to logits that are immediately softmaxed over the
        # anchors, and a term constant across anchors cancels there, so a bias would be unidentifiable.
        self.distance_bias = nn.Sequential(nn.Linear(num_basis, hidden_dim), nn.SiLU(),
                                           nn.Linear(hidden_dim, 1, bias=False))

    def is_hard(self) -> bool:
        """
        Whether this grouping always returns one-hot assignments.

        :return:
            Always False, since membership is a tempered softmax.
        :rtype: bool
        """
        return False

    def _encode(self, species_dense: torch.Tensor, distances: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Run message passing over the periodic neighbour graph.

        :param species_dense:
            Padded atomic numbers of shape (B, Nmax).
        :type species_dense: torch.Tensor
        :param distances:
            Pairwise distances of shape (B, Nmax, Nmax) with invalid pairs set to infinity.
        :type distances: torch.Tensor
        :param mask:
            Validity mask of shape (B, Nmax).
        :type mask: torch.Tensor

        :return:
            Node representations of shape (B, Nmax, hidden_dim).
        :rtype: torch.Tensor
        """
        neighbour_index, neighbour_mask = neighbour_lists(distances, self._cutoff, self._max_neighbours)
        neighbour_distances = torch.gather(distances, 2, neighbour_index)
        neighbour_distances = torch.where(neighbour_mask, neighbour_distances,
                                          torch.zeros_like(neighbour_distances))
        basis = gaussian_basis(neighbour_distances, self._num_basis, self._cutoff)

        h = self.species_embedding(species_dense) * mask.unsqueeze(-1)
        for edge_network, node_network in zip(self.edge_networks, self.node_networks):
            neighbour_h = torch.gather(h, 1, neighbour_index.reshape(h.shape[0], -1, 1).expand(-1, -1, h.shape[-1]))
            neighbour_h = neighbour_h.reshape(*neighbour_index.shape, h.shape[-1])
            messages = edge_network(
                torch.cat([h.unsqueeze(2).expand_as(neighbour_h), neighbour_h, basis], dim=-1))
            pooled = (messages * neighbour_mask.unsqueeze(-1)).sum(dim=2)
            h = h + node_network(torch.cat([h, pooled], dim=-1))
            h = h * mask.unsqueeze(-1)
        return h

    def _select_anchors(self, h: torch.Tensor, mask: torch.Tensor,
                        num_groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select the highest-scoring atoms of every structure as group anchors.

        :param h:
            Node representations of shape (B, Nmax, hidden_dim).
        :type h: torch.Tensor
        :param mask:
            Validity mask of shape (B, Nmax).
        :type mask: torch.Tensor
        :param num_groups:
            Number of groups of every structure, of shape (B,).
        :type num_groups: torch.Tensor

        :return:
            Anchor atom indices of shape (B, Kmax), a mask of which anchor columns a structure actually uses, and the
            anchor scores of shape (B, Nmax).
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        scores = self.anchor_score(h).squeeze(-1)
        scores = scores.masked_fill(~mask, -float('inf'))
        max_groups = int(num_groups.max())
        anchor_index = torch.topk(scores, k=max_groups, dim=1).indices
        column = torch.arange(max_groups, device=h.device).unsqueeze(0)
        return anchor_index, column < num_groups.unsqueeze(1), scores

    def assignment(self, x_1: Data) -> torch.Tensor:
        """
        Produce a row-stochastic assignment of atoms to groups for a batch of clean target structures.

        :param x_1:
            Batch of clean target structures with fractional positions.
        :type x_1: torch_geometric.data.Data

        :return:
            Assignment of shape (N, Kmax) whose rows sum to one.
        :rtype: torch.Tensor
        """
        num_groups = read_num_groups(x_1)
        num_structures = int(num_groups.shape[0])
        batch = x_1.batch

        frac_dense, mask = dense_positions(x_1.pos, batch, num_structures)
        species_dense, _ = to_dense_batch(x_1.species.long(), batch, batch_size=num_structures, fill_value=0)
        distances = pair_distances(frac_dense, x_1.cell, mask)

        h = self._encode(species_dense, distances, mask)
        anchor_index, column_valid, scores = self._select_anchors(h, mask, num_groups)
        max_groups = anchor_index.shape[1]

        # Distances from every atom to every selected anchor. The self-distance is infinite by construction, which is
        # harmless because anchors are given their own group below.
        anchor_distances = torch.gather(distances, 2, anchor_index.unsqueeze(1).expand(-1, mask.shape[1], -1))
        finite_distances = torch.where(torch.isfinite(anchor_distances), anchor_distances,
                                       torch.full_like(anchor_distances, 10.0 * self._cutoff))

        query = self.query(h)
        key = torch.gather(self.key(h), 1,
                           anchor_index.unsqueeze(-1).expand(-1, -1, self._hidden_dim))
        logits = torch.einsum('bnd,bkd->bnk', query, key) / (self._hidden_dim ** 0.5)
        logits = logits + self.distance_bias(gaussian_basis(finite_distances, self._num_basis, self._cutoff)).squeeze(-1)
        # Each anchor's own score gates its column, which is how gradient reaches the anchor scorer despite the
        # discrete top-k selection.
        anchor_scores = torch.gather(scores, 1, anchor_index)
        logits = logits + nn.functional.logsigmoid(anchor_scores).unsqueeze(1)

        allowed = column_valid.unsqueeze(1) & (finite_distances <= self._locality_radius)
        # An atom with no anchor inside the locality radius joins its nearest anchor.
        orphan = ~allowed.any(dim=2)
        nearest = finite_distances.masked_fill(~column_valid.unsqueeze(1), float('inf')).argmin(dim=2)
        allowed = allowed | (orphan.unsqueeze(-1)
                             & (torch.arange(max_groups, device=h.device) == nearest.unsqueeze(-1)))

        membership = torch.softmax(logits.masked_fill(~allowed, -float('inf')) / self.temperature, dim=2)

        anchor_one_hot = self._anchor_one_hot(anchor_index, column_valid, mask.shape[1])
        is_anchor = anchor_one_hot.sum(dim=2, keepdim=True) > 0
        dense_assignment = torch.where(is_anchor, anchor_one_hot, membership) * mask.unsqueeze(-1)
        return dense_assignment[mask]

    @staticmethod
    def _anchor_one_hot(anchor_index: torch.Tensor, column_valid: torch.Tensor, num_atoms: int) -> torch.Tensor:
        """
        Build the one-hot rows that assign every selected anchor to its own group.

        :param anchor_index:
            Anchor atom indices of shape (B, Kmax).
        :type anchor_index: torch.Tensor
        :param column_valid:
            Mask of which anchor columns a structure actually uses, of shape (B, Kmax).
        :type column_valid: torch.Tensor
        :param num_atoms:
            Padded number of atoms Nmax.
        :type num_atoms: int

        :return:
            Dense one-hot rows of shape (B, Nmax, Kmax), zero for atoms that are not anchors.
        :rtype: torch.Tensor
        """
        num_structures, max_groups = anchor_index.shape
        flat_index = anchor_index * max_groups + torch.arange(max_groups, device=anchor_index.device).unsqueeze(0)
        values = column_valid.to(torch.get_default_dtype())
        flat = torch.zeros((num_structures, num_atoms * max_groups), dtype=values.dtype,
                           device=anchor_index.device).scatter(1, flat_index, values)
        return flat.reshape(num_structures, num_atoms, max_groups)


def build_grouper(kind: str, **kwargs: object) -> Optional[Grouping]:
    """
    Build a grouping from a short name, so that configuration files can select an arm with a single string.

    :param kind:
        One of "none" for the atomwise baseline, "precomputed" for a partition attached to the batch, or "learned" for
        the anchor-and-membership network.
    :type kind: str
    :param kwargs:
        Keyword arguments forwarded to the grouping constructor.
    :type kwargs: object

    :return:
        The grouping, or None for the atomwise baseline.
    :rtype: Optional[Grouping]

    :raises ValueError:
        If the name is not recognised.
    """
    from .grouping import PrecomputedGrouping
    if kind == "none":
        return None
    if kind == "precomputed":
        return PrecomputedGrouping()
    if kind == "learned":
        return AnchorMembershipGrouper(**kwargs)
    raise ValueError(f"Unknown grouping kind '{kind}'. Expected one of 'none', 'precomputed', 'learned'.")
