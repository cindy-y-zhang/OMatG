"""
The only model class in this package: ``CSPNetFull`` with two independently switchable geometric additions.

THE TWO FACTORS, AND WHY THEY ARE SEPARATE

``feature_mode`` adds a fixed invariant summary of each atom's neighbourhood to the node stream. ``message_graph`` replaces
the fully connected one-edge-per-pair topology with a complete periodic neighbour list and gives every edge its Cartesian
length. They are different claims. The earlier diagnostic that motivates this package moved a coordination-number probe
from 1.57 accuracy points over a geometry-blind control to 7.29 by adding distances and to about 10.2 by adding periodic
multiedges on top -- but it changed both at once, so it cannot say how much of the gain is the node feature and how much is
the topology. Here they are orthogonal switches and the study is a two-by-two, so the question is answerable.

WHY EVERY ADDITION STARTS AT ZERO

The node projection is bias-free and zero-initialised, and the appended edge columns are zeroed. So:

- ``message_graph=fc, feature_mode=none`` is the stock encoder, exactly, and loads a stock checkpoint;
- every other combination is *the same function* at initialisation, on the same topology;
- the layer weights are bit-identical across arms, because the replacement layers copy the baseline layers they replace.

A difference between two arms' learning curves is therefore something training found in the data, not a different starting
point, a different parameter count, or a different random draw.
"""

from enum import Enum
from typing import Any, Optional
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter

from omg.model.encoders.cspnet_full import CSPNetFull
from omg.model.model_utils import prop_indicator

from .features import DescriptorSpec, FeatureMode, gaussian_shell_profile, local_environment_descriptor
from .layers import DistanceCSPLayer
from .neighbors import (MAX_EDGES, SHIFT_CHUNK, Neighbors, constant_degree_radius, minimum_image_distance,
                        periodic_neighbors)


GEOMETRY_PROJECTION_KEY = "geometry_projection.weight"
"""
Name of the one parameter a baseline checkpoint cannot supply.

Loading a stock atomwise checkpoint must leave it at zero and must not be excused from noticing anything else is missing,
so it is named rather than covered by a blanket non-strict load.
"""


class MessageGraph(str, Enum):
    """
    Which topology and edge features message passing uses.

    A string enum so a configuration file can name one and the launcher can pass it through unparsed.
    """

    FC = "fc"
    """
    The baseline: one edge per ordered pair of atoms in a structure, offsets wrapped into the unit cell.

    Cannot represent periodic image multiplicity at all. In a two-atom caesium-chloride cell each atom's eight nearest
    neighbours are one site at eight images, and this graph offers a single edge for all of them.
    """

    FC_DISTANCE = "fc_distance"
    """
    The baseline topology, with the minimum-image Cartesian length of each edge added as a radial expansion.

    This exists to separate the two things ``periodic_distance`` changes at once. The baseline edge network receives the
    node features, the flattened Gram matrix of the lattice, and a *sinusoidal embedding* of the fractional offset; to
    recover a Cartesian length from that it has to invert the sinusoids and then form the quadratic form
    ``offset @ gram @ offset``, a bilinear interaction between two separately embedded inputs. That is a much harder
    function for a two-layer edge MLP than any pooling, and the project's own auxiliary classifier gained 7.29 points when
    handed Cartesian distances directly, which is evidence that the trunk does not form it well on its own.

    So this arm supplies the product and changes nothing else: same edges, same count, same cost. Against
    ``periodic_distance`` it isolates the topology, and against ``fc`` it isolates the distance channel. Without it, a
    ``periodic_distance`` win could not be attributed to either.

    Minimum image is the correct reduction *here and only here*: this graph carries one edge per ordered pair, so each
    pair has exactly one length to report. Applying it on the periodic graph would collapse the distinct images that graph
    exists to separate.
    """

    PERIODIC_DISTANCE = "periodic_distance"
    """
    A complete periodic neighbour list at constant expected degree, every edge carrying a radial expansion of its length.

    The two halves arrive together deliberately. ``SinusoidsEmbedding`` embeds ``sin`` and ``cos`` of integer multiples of
    the fractional offset, which are unchanged by an integer image shift, so the baseline edge feature is identical for
    every image of a pair. Without the length channel the extra edges would be indistinguishable copies.

    The radius follows each structure's own density rather than being fixed, so the degree matches the control's at every
    denoising time; ``constant_degree_radius`` documents why a fixed radius does not work on this probability path.
    """

    @property
    def carries_edge_distance(self) -> bool:
        """
        Return whether this graph appends a length expansion to every edge, and so needs the distance-aware layers.

        :return:
            True for the two graphs that carry lengths.
        :rtype: bool
        """
        return self in (MessageGraph.FC_DISTANCE, MessageGraph.PERIODIC_DISTANCE)

    @property
    def needs_neighbor_list(self) -> bool:
        """
        Return whether this graph has to enumerate periodic images.

        Only the periodic graph does. Keeping the two fully connected graphs off that path is what makes the fifth arm a
        free control rather than one that pays the neighbour builder's cost for a topology it does not use.

        :return:
            True for the periodic graph alone.
        :rtype: bool
        """
        return self is MessageGraph.PERIODIC_DISTANCE


class DirectGeometryCSPNet(CSPNetFull):
    """
    The baseline atomwise encoder with an invariant node descriptor and an optional periodic distance graph.

    :param feature_mode:
        Which descriptor blocks reach the node stream: ``none``, ``radial``, ``angular`` or ``both``.
        Defaults to "none", so that a configuration that forgets to set it reproduces the baseline rather than quietly
        running an experimental arm.
    :type feature_mode: str
    :param message_graph:
        Which topology and edge features message passing uses: ``fc``, ``fc_distance`` or ``periodic_distance``.
        Defaults to "fc", for the same reason.
    :type message_graph: str
    :param descriptor_cutoff:
        Radius of the neighbourhood the descriptor summarises, in Angstrom.
        Defaults to 6.0.
    :type descriptor_cutoff: float
    :param descriptor_shells:
        Number of Gaussian shell sums in the radial block.
        Defaults to 16.
    :type descriptor_shells: int
    :param descriptor_angular_order:
        Highest Legendre order in the angular block.
        Defaults to 4.
    :type descriptor_angular_order: int
    :param graph_degree:
        Expected number of neighbours per atom on the periodic message graph, which sets its radius per structure through
        ``constant_degree_radius``.
        Defaults to 32, matched to the fully connected control's own degree on MPTS-52: that split averages 25 atoms per
        structure and the control joins every ordered pair within a cell, which is 31.9 edges per atom weighted by atom.
        Matching it is what makes the graph factor a test of periodic multiplicity and true edge lengths rather than of
        one graph simply carrying more edges than the other, and it is also what keeps the arm affordable -- a fixed radius
        large enough to reach this degree on crystals reaches three times as far into the density at the noisy end of the
        baseline's path, which measured at 2.5 times the control's peak memory and failed Gate DG0.
    :type graph_degree: float
    :param graph_max_radius:
        Largest radius the message graph will use, in Angstrom, or None to share the descriptor's cutoff.
        Defaults to None, which is what lets one neighbour list serve both consumers and keeps every edge inside the span
        the fixed edge basis covers. It is a real bound and not only a safety net: a structure below about 28 cubic
        Angstrom per atom asks for more reach than 6 Angstrom, and on MPTS-52 that is 13 to 16 per cent of structures,
        which then get a fixed-radius graph of slightly lower degree -- the measured clean degree is 29.9 against the
        target of 32. Raising the bound would recover those few per cent at the price of enumerating a neighbour list at
        the larger radius on every structure, including at the dense noisy end where nothing needs it. The audit reports
        the bound fraction and Gate DG0 fails if it ever reaches a majority.
    :type graph_max_radius: Optional[float]
    :param edge_basis:
        Number of radial basis functions on each edge of the periodic graph.
        Defaults to 32, spanning zero to the radius bound, which places the centres under a fifth of an Angstrom apart --
        finer than the bond-length differences that distinguish coordination environments. The grid is fixed rather than
        scaled per structure, so a given absolute distance always produces the same coefficients.
    :type edge_basis: int
    :param neighbor_shift_chunk:
        Number of lattice images the neighbour builder evaluates per pass.
        Defaults to the builder's own value.
    :type neighbor_shift_chunk: int
    :param max_edges:
        Largest neighbour list tolerated before the builder raises.
        Defaults to the builder's own value.
    :type max_edges: int

    :raises ValueError:
        If a mode or graph name is unknown, or if the graph degree or radius bound is not positive.
    """

    def __init__(self, feature_mode: str = FeatureMode.NONE.value, message_graph: str = MessageGraph.FC.value,
                 descriptor_cutoff: float = 6.0, descriptor_shells: int = 16, descriptor_angular_order: int = 4,
                 graph_degree: float = 32.0, graph_max_radius: Optional[float] = None, edge_basis: int = 32,
                 neighbor_shift_chunk: int = SHIFT_CHUNK, max_edges: int = MAX_EDGES, **kwargs: Any) -> None:
        """Construct the encoder."""
        super().__init__(**kwargs)
        self.feature_mode = FeatureMode(feature_mode)
        self.message_graph = MessageGraph(message_graph)
        self.descriptor_spec = DescriptorSpec(cutoff=descriptor_cutoff, num_shells=descriptor_shells,
                                              max_angular_order=descriptor_angular_order)
        self.graph_degree = float(graph_degree)
        if not self.graph_degree > 0.0:
            raise ValueError(f"The graph degree must be positive, got {self.graph_degree}.")
        self.graph_max_radius = float(graph_max_radius) if graph_max_radius is not None else float(descriptor_cutoff)
        if not self.graph_max_radius > 0.0:
            raise ValueError(f"The graph radius bound must be positive, got {self.graph_max_radius}.")
        self.edge_basis = int(edge_basis)
        self.neighbor_shift_chunk = int(neighbor_shift_chunk)
        self.max_edges = int(max_edges)

        # One list, built at whichever radius is larger and narrowed per consumer, so that turning both factors on does
        # not enumerate the same periodic images twice. The graph's radius varies per structure but never exceeds its
        # bound, so this covers it.
        self.neighbor_cutoff = max(self.descriptor_spec.cutoff, self.graph_max_radius)

        self.geometry_projection = nn.Linear(self.descriptor_spec.dim, self.hidden_dim, bias=False)
        with torch.no_grad():
            self.geometry_projection.weight.zero_()
        # Not persistent: it is a restatement of feature_mode for the audit to read, and putting it in the state dict
        # would make a baseline checkpoint look as though it were missing a parameter.
        self.register_buffer("geometry_channel_mask", self.descriptor_spec.channel_mask(self.feature_mode),
                             persistent=False)

        if self.message_graph.carries_edge_distance:
            for index in range(self.num_layers):
                name = f"csp_layer_{index}"
                baseline = self._modules[name]
                replacement = DistanceCSPLayer(self.hidden_dim, self.act_fn, self.dis_emb, ln=self.ln, ip=self.ip,
                                               distance_dim=self.edge_basis)
                replacement.copy_baseline_weights(baseline)
                self._modules[name] = replacement

    @property
    def uses_neighbors(self) -> bool:
        """
        Return whether a forward pass needs a periodic neighbour list.

        The baseline combination must not pay for one, both because it is the cost reference the thirty per cent ceiling is
        measured against and because it has to stay an exact reproduction of the stock encoder.

        :return:
            True if either factor needs neighbours.
        :rtype: bool
        """
        return self.feature_mode.uses_geometry or self.message_graph.needs_neighbor_list

    def build_neighbors(self, frac_coords: torch.Tensor, lattices: torch.Tensor,
                        num_atoms: torch.Tensor) -> Neighbors:
        """
        Build the periodic neighbour list a forward pass will use.

        Public because the audit and the probes describe the same neighbourhoods the model sees, and a second
        implementation of that would be a second thing to keep in step.

        :param frac_coords:
            Fractional coordinates of shape ``(atoms, 3)``.
        :type frac_coords: torch.Tensor
        :param lattices:
            Lattices of shape ``(structures, 3, 3)``.
        :type lattices: torch.Tensor
        :param num_atoms:
            Number of atoms of every structure, of shape ``(structures,)``.
        :type num_atoms: torch.Tensor

        :return:
            The neighbour list.
        :rtype: Neighbors
        """
        return periodic_neighbors(frac_coords, lattices, num_atoms, self.neighbor_cutoff,
                                  shift_chunk=self.neighbor_shift_chunk, max_edges=self.max_edges)

    def _graph(self, neighbors: Optional[Neighbors], num_atoms: torch.Tensor, frac_coords: torch.Tensor,
               lattices: torch.Tensor, node2graph: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Return the edges, their fractional offsets, and their length expansion if the graph carries one.

        :param neighbors:
            Periodic neighbour list, or None when the fully connected graph is in use.
        :type neighbors: Optional[Neighbors]
        :param num_atoms:
            Number of atoms of every structure, of shape ``(structures,)``.
        :type num_atoms: torch.Tensor
        :param frac_coords:
            Fractional coordinates of shape ``(atoms, 3)``.
        :type frac_coords: torch.Tensor
        :param lattices:
            Lattices of shape ``(structures, 3, 3)``.
        :type lattices: torch.Tensor
        :param node2graph:
            Structure of every atom, of shape ``(atoms,)``.
        :type node2graph: torch.Tensor

        :return:
            Edges of shape ``(2, edges)``, fractional offsets of shape ``(edges, 3)``, and either the length expansion of
            shape ``(edges, edge_basis)`` or None.
        :rtype: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        """
        if self.message_graph is not MessageGraph.PERIODIC_DISTANCE:
            edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
            if self.message_graph is MessageGraph.FC:
                return edges, frac_diff, None
            # The baseline wraps the offset into [0, 1) for the sinusoids, which do not care. A length does, so the
            # nearest image is found here; on a one-edge-per-pair graph that is the only length the pair has. Searched
            # rather than folded, because a fold alone is the minimum image convention and these cells are not reduced.
            edge2graph = node2graph[edges[0]]
            distance = minimum_image_distance(frac_diff, lattices[edge2graph])
            # Built exactly as the periodic graph builds it, from the same basis grid and the same per-structure envelope,
            # so that the two arms differ in which edges exist and in nothing else. Pairs beyond the envelope are damped
            # to zero, which is the honest reading: the channel reports "far" and stops resolving.
            radius = constant_degree_radius(lattices, num_atoms, self.graph_degree, self.graph_max_radius)
            expansion = gaussian_shell_profile(distance, self.graph_max_radius, self.edge_basis,
                                               envelope=radius[edge2graph])
            return edges, frac_diff, expansion

        assert neighbors is not None
        radius = constant_degree_radius(lattices, num_atoms, self.graph_degree, self.graph_max_radius)
        inside = neighbors.within(radius)
        edges = torch.stack([inside.center, inside.neighbor], dim=0)
        # The offset keeps its image rather than being wrapped: wrapping is what would collapse the distinct neighbours
        # this graph exists to separate. The sinusoidal embedding downstream cannot see the image, which is exactly why
        # the length expansion below is not optional.
        expansion = gaussian_shell_profile(inside.distance, self.graph_max_radius, self.edge_basis,
                                           envelope=radius[inside.edge2graph])
        return edges, inside.offset, expansion

    def _forward(self, atom_types: torch.Tensor, frac_coords: torch.Tensor, lattices: torch.Tensor,
                 num_atoms: torch.Tensor, node2graph: torch.Tensor, t: torch.Tensor,
                 prop: Optional[torch.Tensor] = None) -> Data:
        """
        Run the encoder.

        Mirrors ``CSPNetFull._forward`` and adds exactly two things: the projected descriptor, once, immediately after the
        atom and time embedding, and the edge length expansion threaded into every layer. The duplication of the baseline's
        body is deliberate rather than clever -- there is no hook in the baseline to add a node term at that point -- and
        ``direct_geometry/tests/test_encoder.py`` asserts that the baseline combination reproduces the stock encoder output
        for output field, so a drift between the two bodies fails a test rather than biasing a study.

        :param atom_types:
            Atomic numbers of shape ``(atoms,)``.
        :type atom_types: torch.Tensor
        :param frac_coords:
            Fractional coordinates of shape ``(atoms, 3)``.
        :type frac_coords: torch.Tensor
        :param lattices:
            Lattices of shape ``(structures, 3, 3)``.
        :type lattices: torch.Tensor
        :param num_atoms:
            Number of atoms of every structure, of shape ``(structures,)``.
        :type num_atoms: torch.Tensor
        :param node2graph:
            Structure of every atom, of shape ``(atoms,)``.
        :type node2graph: torch.Tensor
        :param t:
            Embedded times of shape ``(structures, latent)``.
        :type t: torch.Tensor
        :param prop:
            Embedded conditioning property, or None.
            Defaults to None.
        :type prop: Optional[torch.Tensor]

        :return:
            The baseline's output fields, unchanged in name, shape and meaning.
        :rtype: torch_geometric.data.Data
        """
        neighbors = self.build_neighbors(frac_coords, lattices, num_atoms) if self.uses_neighbors else None
        edges, frac_diff, distance_features = self._graph(neighbors, num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]

        node_features = self.node_embedding(atom_types - self.species_shift)
        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = self.atom_latent_emb(torch.cat([node_features, t_per_atom], dim=1))

        if self.feature_mode.uses_geometry:
            descriptor = local_environment_descriptor(neighbors, node_features.shape[0], self.descriptor_spec,
                                                      self.feature_mode)
            node_features = node_features + self.geometry_projection(descriptor.to(node_features.dtype))

        indicator = prop_indicator(batch_size=len(num_atoms), p_uncond=0.2) if prop is not None else None
        extras = {} if distance_features is None else {"distance_features": distance_features}
        for index in range(self.num_layers):
            if prop is not None:
                node_features = self.adapters[index](node_features, prop, indicator, num_atoms)
            node_features = self._modules[f"csp_layer_{index}"](
                node_features, frac_coords, lattices, edges, edge2graph, frac_diff=frac_diff, **extras)

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        coord_b = self.coord_out(node_features)
        coord_eta = self.coord_out_2(node_features)
        graph_features = scatter(node_features, node2graph, dim=0, reduce="mean")

        if self.pred_scalar:
            return self.scalar_out(graph_features)

        lattice_b = self.lattice_out(graph_features).view(-1, 3, 3)
        lattice_eta = self.lattice_out_2(graph_features).view(-1, 3, 3)
        if self.ip:
            lattice_b = torch.einsum("bij,bjk->bik", lattice_b, lattices)
            lattice_eta = torch.einsum("bij,bjk->bik", lattice_eta, lattices)
        if self.pred_type:
            return Data(species_b=self.type_out(node_features), species_eta=self.type_out_2(node_features),
                        pos_b=coord_b, pos_eta=coord_eta, cell_b=lattice_b, cell_eta=lattice_eta)
        return Data(pos_b=coord_b, pos_eta=coord_eta, cell_b=lattice_b, cell_eta=lattice_eta)

    def load_baseline_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Load a stock ``CSPNetFull`` state dict, zero-padding the widened edge weights.

        The one weight whose shape differs is the first layer of each edge network, which gained columns for the length
        expansion. Those columns are filled with zeros, so a checkpoint trained without the feature is reproduced exactly
        and the feature starts as something training has to find useful.

        :param state_dict:
            A stock encoder's parameters.
        :type state_dict: dict[str, torch.Tensor]

        :raises ValueError:
            If a parameter is present with a shape that is not an accountable widening, or if anything other than the
            geometry projection is absent, or if the checkpoint carries parameters this encoder does not have.
        """
        prepared = dict(state_dict)
        for name, target in self.state_dict().items():
            if name not in prepared or prepared[name].shape == target.shape:
                continue
            source = prepared[name]
            widened = (source.dim() == 2 and target.dim() == 2 and source.shape[0] == target.shape[0]
                       and source.shape[1] < target.shape[1])
            if not widened:
                raise ValueError(f"Parameter {name} has shape {tuple(source.shape)} in the checkpoint but "
                                 f"{tuple(target.shape)} here, which is not a widening this encoder introduced.")
            padded = torch.zeros_like(target)
            padded[:, :source.shape[1]] = source.to(target.dtype)
            prepared[name] = padded

        missing, unexpected = self.load_state_dict(prepared, strict=False)
        unaccounted = [name for name in missing if name != GEOMETRY_PROJECTION_KEY]
        if unaccounted or unexpected:
            raise ValueError(f"The checkpoint does not describe this encoder. Missing {unaccounted}, unexpected "
                             f"{list(unexpected)}. Only {GEOMETRY_PROJECTION_KEY} may be absent, since a baseline run "
                             f"has no such parameter and it must stay at zero.")


def baseline_encoder_state(checkpoint: dict[str, Any], prefix: str = "model.encoder.") -> dict[str, torch.Tensor]:
    """
    Extract an encoder's parameters from a Lightning checkpoint.

    :param checkpoint:
        A loaded Lightning checkpoint.
    :type checkpoint: dict[str, Any]
    :param prefix:
        Prefix the encoder's parameters carry inside the checkpoint.
        Defaults to "model.encoder.": ``OMGLightning`` holds the ``Model`` as ``self.model`` and the ``Model`` holds the
        encoder as ``self.encoder``, so that is the path a saved atomwise run uses.
    :type prefix: str

    :raises KeyError:
        If the checkpoint has no state dict.
    :raises ValueError:
        If no parameter carries the prefix, which means the checkpoint is of a different shape than assumed rather than
        that the encoder happens to be empty.

    :return:
        The encoder's parameters, with the prefix removed.
    :rtype: dict[str, torch.Tensor]
    """
    if "state_dict" not in checkpoint:
        raise KeyError("The checkpoint has no 'state_dict' entry, so it is not a Lightning checkpoint.")
    state = {name[len(prefix):]: value for name, value in checkpoint["state_dict"].items() if name.startswith(prefix)}
    if not state:
        raise ValueError(f"No parameter in the checkpoint starts with {prefix!r}. Available prefixes include "
                         f"{sorted({name.split('.')[0] for name in checkpoint['state_dict']})}.")
    return state
