"""
Fixed, invariant descriptors of an atom's local environment.

WHAT IS COMPUTED, AND WHY IT IS THIS

Two blocks, both sums over the periodic neighbours of an atom, both weighted by a smooth cutoff so that a neighbour
entering or leaving the shell changes the descriptor continuously rather than in a step.

**Radial**, sixteen Gaussian shell sums normalised to a distribution over shells, plus the log of one plus the soft
coordination number. The normalisation is what makes the shells describe *shape* in the radial direction -- where the
neighbours sit -- and the unnormalised channel is what keeps *how many* there are, which the normalisation would otherwise
divide away. Both are needed: ``CSPNet`` aggregates its messages with a mean, so a count is exactly the thing the trunk
cannot recover on its own.

**Angular**, the rotationally invariant power spectrum through order four, normalised by the squared neighbour mass, plus
the log of one plus that mass. The order-``l`` component is

    ``A_l = sum_{j,k} w_j w_k P_l(u_j . u_k)``

over neighbour pairs, with ``P_l`` the Legendre polynomial and ``u`` the unit vectors. Written that way it is a sum over
pairs of neighbours and costs the square of the coordination number. It is not computed that way. Expanding ``P_l`` in
powers of the dot product and using

    ``sum_{j,k} w_j w_k (u_j . u_k)^m = sum_{a_1..a_m} ( sum_j w_j u_{j,a_1} ... u_{j,a_m} )^2``

turns each power into the squared norm of a single summed moment tensor, so the whole block is a scatter-add over *edges*
and costs the coordination number once. The moment tensors are fully symmetric, so only the distinct monomials are
accumulated -- three, six, ten and fifteen for orders one to four -- with multinomial weights restoring the full norm.

Each ``A_l`` is a squared norm of a spherical-harmonic sum, hence non-negative, and is bounded above by the squared mass,
so the normalised channels sit in ``[0, 1]`` by construction and no learned or dataset-dependent scaling is needed.

WHAT THE BLOCKS SEPARATE

The angular block is not decoration. A tetrahedral and a square-planar four-coordinate site have the same neighbour count
at the same radius, so their radial blocks are identical to the last bit; their order-two components are 0 and 1/4. A
tetrahedral and an octahedral site both have order-two component 0, and their order-four components are 0.259 and 0.583.
This is the property the whole pivot rests on, so it is asserted on hand-built polyhedra rather than assumed.

WHAT IS DELIBERATELY ABSENT

Species. The descriptor summarises geometry alone, so that a probe's gain over a geometry-blind control cannot be chemistry
leaking through the feature. The earlier coordination-geometry work spent a week on a number that turned out to be 73 per
cent chemistry, and the cheapest defence is a descriptor that has no way to encode it.
"""

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from math import factorial, pi
from typing import Optional, Union
import torch

from .neighbors import Neighbors


NORMALISATION_EPSILON = 1.0e-12
"""
Guard added to a denominator before dividing.

Chosen so that an atom with no neighbours yields exactly zero rather than a NaN: its numerators are exactly zero, so the
quotient is zero for any positive guard, and no branch on the coordination number is needed. Small enough that it never
perturbs an atom that does have neighbours.
"""


class FeatureMode(str, Enum):
    """
    Which descriptor blocks reach the network.

    A string enum so that a configuration file can name a mode and the launcher can pass it through unparsed.

    Every mode produces a descriptor of the same width, with the disabled blocks exactly zero, and every arm builds the
    same projection. That is what makes the arms comparable: they differ in the content of an input, not in the number of
    parameters or the shape of a tensor.
    """

    NONE = "none"
    """No geometry. The encoder is the baseline, and the projection multiplies a zero vector."""

    RADIAL = "radial"
    """Shell sums and the coordination channel only."""

    ANGULAR = "angular"
    """Power spectrum and the mass channel only."""

    BOTH = "both"
    """Everything."""

    @property
    def uses_radial(self) -> bool:
        """
        Return whether this mode enables the radial block.

        :return:
            True if the radial block is enabled.
        :rtype: bool
        """
        return self in (FeatureMode.RADIAL, FeatureMode.BOTH)

    @property
    def uses_angular(self) -> bool:
        """
        Return whether this mode enables the angular block.

        :return:
            True if the angular block is enabled.
        :rtype: bool
        """
        return self in (FeatureMode.ANGULAR, FeatureMode.BOTH)

    @property
    def uses_geometry(self) -> bool:
        """
        Return whether this mode needs a neighbour list at all.

        The ``none`` mode must not pay for one, both because it is the cost baseline the thirty per cent ceiling is
        measured against and because it has to remain an exact reproduction of the stock encoder.

        :return:
            True if any block is enabled.
        :rtype: bool
        """
        return self.uses_radial or self.uses_angular


@dataclass(frozen=True)
class DescriptorSpec:
    """
    The fixed shape of the descriptor.

    Frozen and free of trainable parameters, so that a descriptor computed today can be compared with one computed by the
    probe script or the audit without either having to agree about a learned state.

    :param cutoff:
        Radius of the neighbourhood summarised, in Angstrom.
        Defaults to 6.0, comfortably past a first coordination shell and inside the 7 Angstrom graph the baseline already
        pays for.
    :type cutoff: float
    :param num_shells:
        Number of Gaussian shell sums.
        Defaults to 16, which places the centres 0.4 Angstrom apart and so resolves a first from a second shell.
    :type num_shells: int
    :param max_angular_order:
        Highest Legendre order retained.
        Defaults to 4. Order two separates a tetrahedron from a square plane and order four separates a tetrahedron from
        an octahedron; beyond that the moment tensors grow and the returns do not.
    :type max_angular_order: int
    """

    cutoff: float = 6.0
    num_shells: int = 16
    max_angular_order: int = 4

    def __post_init__(self) -> None:
        """
        Check the specification is usable.

        :raises ValueError:
            If the cutoff is not positive, if fewer than two shells are asked for, or if the angular order is outside the
            range the moment expansion is derived for.
        """
        if not self.cutoff > 0.0:
            raise ValueError(f"The cutoff must be positive, got {self.cutoff}.")
        if self.num_shells < 2:
            raise ValueError(f"At least two shells are needed for a width to be defined, got {self.num_shells}.")
        if not 1 <= self.max_angular_order <= 8:
            raise ValueError(f"The angular order must lie between 1 and 8, got {self.max_angular_order}. Above 8 the "
                             f"moment tensors are larger than the neighbour lists they summarise.")

    @property
    def radial_dim(self) -> int:
        """
        Return the width of the radial block.

        :return:
            Number of radial channels: one per shell plus the coordination channel.
        :rtype: int
        """
        return self.num_shells + 1

    @property
    def angular_dim(self) -> int:
        """
        Return the width of the angular block.

        :return:
            Number of angular channels: one per Legendre order plus the mass channel.
        :rtype: int
        """
        return self.max_angular_order + 1

    @property
    def dim(self) -> int:
        """
        Return the total width of the descriptor.

        :return:
            Number of channels.
        :rtype: int
        """
        return self.radial_dim + self.angular_dim

    @property
    def shell_width(self) -> float:
        """
        Return the standard deviation of the shell Gaussians, in Angstrom.

        Set to the centre spacing, so neighbouring shells overlap near their half maxima and the expansion is smooth in
        distance rather than a set of disjoint bins.

        :return:
            Standard deviation.
        :rtype: float
        """
        return self.cutoff / (self.num_shells - 1)

    def channel_mask(self, mode: FeatureMode) -> torch.Tensor:
        """
        Return which channels a mode leaves enabled.

        :param mode:
            The feature mode.
        :type mode: FeatureMode

        :return:
            Boolean mask of shape ``(dim,)``.
        :rtype: torch.Tensor
        """
        return torch.cat([
            torch.full((self.radial_dim,), mode.uses_radial, dtype=torch.bool),
            torch.full((self.angular_dim,), mode.uses_angular, dtype=torch.bool),
        ])

    def channel_names(self) -> list[str]:
        """
        Return a name per channel, for the audit and the probe reports.

        :return:
            Channel names, in order.
        :rtype: list[str]
        """
        names = [f"shell_{index}" for index in range(self.num_shells)] + ["log_coordination"]
        names += [f"power_{order}" for order in range(1, self.max_angular_order + 1)] + ["log_pair_mass"]
        return names


@lru_cache(maxsize=None)
def legendre_coefficients(order: int) -> tuple[float, ...]:
    """
    Return the coefficients of the Legendre polynomial of an order, lowest power first.

    Built from Bonnet's recurrence ``(n + 1) P_{n+1}(x) = (2n + 1) x P_n(x) - n P_{n-1}(x)`` rather than written out, so
    that raising the angular order cannot silently disagree with a hardcoded table.

    :param order:
        Order of the polynomial.
    :type order: int

    :raises ValueError:
        If the order is negative.

    :return:
        Coefficients, lowest power first.
    :rtype: tuple[float, ...]
    """
    if order < 0:
        raise ValueError(f"The order must be non-negative, got {order}.")
    previous, current = [1.0], [0.0, 1.0]
    if order == 0:
        return tuple(previous)
    for degree in range(1, order):
        shifted = [0.0] + current
        scaled = [(2 * degree + 1) * value for value in shifted]
        for index, value in enumerate(previous):
            scaled[index] -= degree * value
        previous, current = current, [value / (degree + 1) for value in scaled]
    return tuple(current)


@lru_cache(maxsize=None)
def monomial_table(degree: int) -> tuple[tuple[tuple[int, int, int], ...], tuple[float, ...]]:
    """
    Return the distinct monomials of a given degree in three variables, with their multinomial multiplicities.

    A moment tensor of degree ``m`` is fully symmetric, so its squared norm over all ``3 ** m`` index tuples equals a
    weighted sum over the ``(m + 1)(m + 2) / 2`` distinct monomials, the weight being the number of index tuples that
    produce each. Accumulating the distinct monomials instead of the full tensor is a factor of five and a half at degree
    four, which is where the cost of this block lives.

    :param degree:
        Degree of the monomials.
    :type degree: int

    :return:
        Exponent triples and their multiplicities.
    :rtype: tuple[tuple[tuple[int, int, int], ...], tuple[float, ...]]
    """
    exponents, multiplicities = [], []
    for first in range(degree, -1, -1):
        for second in range(degree - first, -1, -1):
            third = degree - first - second
            exponents.append((first, second, third))
            multiplicities.append(factorial(degree) / (factorial(first) * factorial(second) * factorial(third)))
    return tuple(exponents), tuple(multiplicities)


def cosine_cutoff(distance: torch.Tensor, cutoff: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Return the smooth cutoff weight of every edge.

    Falls from one at zero separation to zero at the cutoff with a vanishing derivative there, so that a neighbour crossing
    the boundary changes the descriptor continuously. A hard cutoff would make the descriptor discontinuous in the atomic
    positions, which for a denoiser reading it at every integration step is a source of noise rather than of signal.

    :param distance:
        Edge lengths of shape ``(edges,)``.
    :type distance: torch.Tensor
    :param cutoff:
        Radius at which the weight reaches zero, in Angstrom: one radius, or a tensor broadcasting against the distances
        so that each edge can be damped at its own structure's radius.
    :type cutoff: Union[float, torch.Tensor]

    :return:
        Weights of shape ``(edges,)``, in ``[0, 1]``.
    :rtype: torch.Tensor
    """
    scaled = (distance / cutoff).clamp(max=1.0)
    return 0.5 * (torch.cos(pi * scaled) + 1.0)


def gaussian_shell_profile(distance: torch.Tensor, cutoff: float, count: int, width: Optional[float] = None,
                           envelope: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
    """
    Expand edge lengths over evenly spaced Gaussians, damped to nothing at the edge's own cutoff.

    Shared by the node descriptor's radial block and the message graph's edge features, so that the two cannot drift into
    describing distance differently. The envelope matters as much as the basis: an edge at the cutoff must contribute
    nothing, or the expansion is discontinuous where the neighbour list is.

    The grid and the envelope are separable because the message graph needs them to differ. Its radius follows each
    structure's density, but its basis must not: the whole point of the length channel is to state an *absolute*
    interatomic distance, which is the one thing the baseline's sinusoidal embedding of a fractional offset cannot express.
    A grid that moved with the structure would encode a ratio instead, and recovering the bond length from it would need
    the same composition with the cell that the trunk apparently fails to perform. So the centres stay on a fixed grid and
    only the damping follows the radius.

    :param distance:
        Edge lengths of shape ``(edges,)``.
    :type distance: torch.Tensor
    :param cutoff:
        Largest distance the Gaussian centres span, in Angstrom.
    :type cutoff: float
    :param count:
        Number of Gaussians.
    :type count: int
    :param width:
        Standard deviation, or None for the centre spacing.
        Defaults to None.
    :type width: Optional[float]
    :param envelope:
        Radius at which the expansion is damped to zero, or None to damp at the grid's own cutoff.
        Defaults to None.
    :type envelope: Optional[Union[float, torch.Tensor]]

    :return:
        Features of shape ``(edges, count)``.
    :rtype: torch.Tensor
    """
    centres = torch.linspace(0.0, cutoff, count, dtype=distance.dtype, device=distance.device)
    deviation = width if width is not None else cutoff / max(count - 1, 1)
    profile = torch.exp(-0.5 * ((distance.unsqueeze(-1) - centres) / deviation) ** 2)
    damping = cosine_cutoff(distance, cutoff if envelope is None else envelope)
    return profile * damping.unsqueeze(-1)


def _accumulate(values: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Sum edge quantities onto their centre atoms.

    :param values:
        Edge quantities of shape ``(edges, channels)``.
    :type values: torch.Tensor
    :param index:
        Centre atom of every edge, of shape ``(edges,)``.
    :type index: torch.Tensor
    :param num_nodes:
        Number of atoms.
    :type num_nodes: int

    :return:
        Per-atom sums of shape ``(num_nodes, channels)``.
    :rtype: torch.Tensor
    """
    out = torch.zeros((num_nodes, values.shape[-1]), dtype=values.dtype, device=values.device)
    return out.index_add(0, index, values)


def radial_block(neighbors: Neighbors, num_nodes: int, spec: DescriptorSpec) -> torch.Tensor:
    """
    Return the radial shell distribution and the coordination channel.

    :param neighbors:
        Neighbour list, already restricted to the descriptor cutoff.
    :type neighbors: Neighbors
    :param num_nodes:
        Number of atoms.
    :type num_nodes: int
    :param spec:
        Descriptor specification.
    :type spec: DescriptorSpec

    :return:
        Features of shape ``(num_nodes, spec.radial_dim)``.
    :rtype: torch.Tensor
    """
    weight = cosine_cutoff(neighbors.distance, spec.cutoff)
    profile = gaussian_shell_profile(neighbors.distance, spec.cutoff, spec.num_shells, spec.shell_width)

    shells = _accumulate(profile, neighbors.center, num_nodes)
    # A distribution over shells says where the neighbours are; the soft count says how many. Normalising discards the
    # count, so it is carried separately rather than left implicit.
    shells = shells / (shells.sum(dim=-1, keepdim=True) + NORMALISATION_EPSILON)
    coordination = _accumulate(weight.unsqueeze(-1), neighbors.center, num_nodes)
    return torch.cat([shells, torch.log1p(coordination)], dim=-1)


def angular_block(neighbors: Neighbors, num_nodes: int, spec: DescriptorSpec) -> torch.Tensor:
    """
    Return the normalised rotationally invariant power spectrum and the mass channel.

    Computed from summed moment tensors, so the cost is linear in the number of edges rather than quadratic in the
    coordination number. See the module docstring for the identity this rests on.

    :param neighbors:
        Neighbour list, already restricted to the descriptor cutoff.
    :type neighbors: Neighbors
    :param num_nodes:
        Number of atoms.
    :type num_nodes: int
    :param spec:
        Descriptor specification.
    :type spec: DescriptorSpec

    :return:
        Features of shape ``(num_nodes, spec.angular_dim)``.
    :rtype: torch.Tensor
    """
    weight = cosine_cutoff(neighbors.distance, spec.cutoff)
    direction = neighbors.vector / neighbors.distance.clamp(min=NORMALISATION_EPSILON).unsqueeze(-1)
    mass = _accumulate(weight.unsqueeze(-1), neighbors.center, num_nodes).squeeze(-1)

    # Power zero of the dot product is one, so its pair sum is the squared mass. Kept in the list because the Legendre
    # expansions of the even orders need it.
    powers = [mass ** 2]
    for degree in range(1, spec.max_angular_order + 1):
        exponents, multiplicities = monomial_table(degree)
        monomials = torch.stack([
            direction[:, 0] ** first * direction[:, 1] ** second * direction[:, 2] ** third
            for first, second, third in exponents], dim=-1)
        moments = _accumulate(monomials * weight.unsqueeze(-1), neighbors.center, num_nodes)
        factors = torch.tensor(multiplicities, dtype=moments.dtype, device=moments.device)
        powers.append((factors * moments ** 2).sum(dim=-1))

    scale = powers[0] + NORMALISATION_EPSILON
    spectrum = []
    for order in range(1, spec.max_angular_order + 1):
        coefficients = legendre_coefficients(order)
        component = sum(coefficient * powers[power] for power, coefficient in enumerate(coefficients) if coefficient)
        spectrum.append(component / scale)
    return torch.cat([torch.stack(spectrum, dim=-1), torch.log1p(powers[0]).unsqueeze(-1)], dim=-1)


def local_environment_descriptor(neighbors: Neighbors, num_nodes: int, spec: DescriptorSpec,
                                 mode: FeatureMode = FeatureMode.BOTH) -> torch.Tensor:
    """
    Return the per-atom descriptor, with the blocks a mode disables set exactly to zero.

    The width does not depend on the mode. Disabled blocks are zeroed rather than dropped so that every arm of the study
    builds the same projection with the same parameter count, and a difference between two arms is a difference in what
    the network was shown rather than in how large it was.

    :param neighbors:
        Neighbour list. Edges beyond the descriptor cutoff are dropped here, so a list built at a larger radius for the
        message graph can be passed straight in.
    :type neighbors: Neighbors
    :param num_nodes:
        Number of atoms.
    :type num_nodes: int
    :param spec:
        Descriptor specification.
    :type spec: DescriptorSpec
    :param mode:
        Which blocks to enable.
        Defaults to FeatureMode.BOTH.
    :type mode: FeatureMode

    :return:
        Descriptor of shape ``(num_nodes, spec.dim)``.
    :rtype: torch.Tensor
    """
    mode = FeatureMode(mode)
    dtype = neighbors.vector.dtype
    device = neighbors.vector.device
    inside = neighbors.within(spec.cutoff)

    if mode.uses_radial:
        radial = radial_block(inside, num_nodes, spec)
    else:
        radial = torch.zeros((num_nodes, spec.radial_dim), dtype=dtype, device=device)
    if mode.uses_angular:
        angular = angular_block(inside, num_nodes, spec)
    else:
        angular = torch.zeros((num_nodes, spec.angular_dim), dtype=dtype, device=device)
    return torch.cat([radial, angular], dim=-1)


def descriptor_statistics(descriptor: torch.Tensor, spec: DescriptorSpec,
                          names: Optional[list[str]] = None) -> dict[str, dict[str, float]]:
    """
    Summarise a descriptor batch channel by channel.

    Written for the audit rather than for training. A channel with zero variance is a channel the projection can never use,
    and a non-finite entry is a bug that would otherwise surface a thousand steps into a run as a NaN loss.

    :param descriptor:
        Descriptor of shape ``(atoms, spec.dim)``.
    :type descriptor: torch.Tensor
    :param spec:
        Descriptor specification.
    :type spec: DescriptorSpec
    :param names:
        Channel names, or None to take them from the specification.
        Defaults to None.
    :type names: Optional[list[str]]

    :return:
        Mean, standard deviation, extremes and finite fraction of every channel, keyed by channel name.
    :rtype: dict[str, dict[str, float]]
    """
    names = names if names is not None else spec.channel_names()
    values = descriptor.detach().double()
    finite = torch.isfinite(values)
    summary = {}
    for index, name in enumerate(names):
        column = values[:, index]
        good = column[finite[:, index]]
        summary[name] = {
            "mean": float(good.mean()) if good.numel() else float("nan"),
            "std": float(good.std(unbiased=False)) if good.numel() else float("nan"),
            "min": float(good.min()) if good.numel() else float("nan"),
            "max": float(good.max()) if good.numel() else float("nan"),
            "finite_fraction": float(finite[:, index].double().mean()),
        }
    return summary
