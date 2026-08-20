"""
Tests of the local-environment descriptor.

Three things have to hold for the descriptor to be worth measuring.

It has to be *invariant*: to the labelling of the atoms, to where the crystal sits, to how it is oriented, to which basis
the lattice is written in, and to how many times the cell is tiled. A descriptor that fails any of these is partly reading
the file format, and a probe would report information that the denoiser cannot use.

It has to be *computed correctly*: the angular block replaces an explicit sum over pairs of neighbours with squared norms
of summed moment tensors, and that identity is asserted against the brute-force pair sum rather than trusted.

And it has to *separate the environments the pivot claims to describe*. Two four-coordinate sites at the same radius have
identical radial blocks by construction, so if the angular block cannot tell a tetrahedron from a square plane then the
angular channels are decoration and the honest thing is to drop them.
"""

import math
import pytest
import torch
from scipy.special import eval_legendre

from direct_geometry.features import (DescriptorSpec, FeatureMode, angular_block, cosine_cutoff,
                                      descriptor_statistics, legendre_coefficients, local_environment_descriptor,
                                      monomial_table)
from direct_geometry.neighbors import periodic_neighbors
from .conftest import (BODY_CENTRED, CUBIC, OCTAHEDRAL, SKEWED, SQUARE_PLANAR, TETRAHEDRAL, cluster, random_batch,
                       random_structure, rotation, single, supercell)


SPEC = DescriptorSpec()
"""The production specification: 6 Angstrom, sixteen shells, angular order four."""


def describe(frac: torch.Tensor, cell: torch.Tensor, atoms: torch.Tensor,
             spec: DescriptorSpec = SPEC, mode: FeatureMode = FeatureMode.BOTH) -> torch.Tensor:
    """
    Build the neighbour list and the descriptor in one step.

    :param frac:
        Fractional coordinates of shape ``(atoms, 3)``.
    :type frac: torch.Tensor
    :param cell:
        Lattices of shape ``(structures, 3, 3)``.
    :type cell: torch.Tensor
    :param atoms:
        Atom counts of shape ``(structures,)``.
    :type atoms: torch.Tensor
    :param spec:
        Descriptor specification.
        Defaults to SPEC.
    :type spec: DescriptorSpec
    :param mode:
        Which blocks to enable.
        Defaults to FeatureMode.BOTH.
    :type mode: FeatureMode

    :return:
        Descriptor of shape ``(atoms, spec.dim)``.
    :rtype: torch.Tensor
    """
    neighbors = periodic_neighbors(frac, cell, atoms, spec.cutoff)
    return local_environment_descriptor(neighbors, frac.shape[0], spec, mode)


def test_the_descriptor_is_the_width_the_plan_specifies() -> None:
    """Seventeen radial channels and five angular ones. Named here so a change to either is a deliberate one."""
    assert (SPEC.radial_dim, SPEC.angular_dim, SPEC.dim) == (17, 5, 22)
    assert len(SPEC.channel_names()) == SPEC.dim


def test_the_legendre_coefficients_are_the_polynomials_they_claim_to_be() -> None:
    """
    The recurrence is checked against the closed forms, since every angular channel is a combination of these numbers.

    Built from Bonnet's recurrence rather than written out, so raising the angular order cannot silently disagree with a
    hardcoded table -- but then the recurrence itself has to be pinned to something.
    """
    assert legendre_coefficients(0) == (1.0,)
    assert legendre_coefficients(1) == (0.0, 1.0)
    assert legendre_coefficients(2) == pytest.approx((-0.5, 0.0, 1.5))
    assert legendre_coefficients(3) == pytest.approx((0.0, -1.5, 0.0, 2.5))
    assert legendre_coefficients(4) == pytest.approx((3 / 8, 0.0, -30 / 8, 0.0, 35 / 8))
    for order in range(6):
        for value in (-1.0, -0.4, 0.0, 0.3, 1.0):
            polynomial = sum(c * value ** power for power, c in enumerate(legendre_coefficients(order)))
            assert polynomial == pytest.approx(float(eval_legendre(order, value)))


def test_the_monomial_multiplicities_reproduce_the_full_tensor_norm() -> None:
    """
    Accumulating distinct monomials with multinomial weights must equal the squared norm of the full symmetric tensor.

    This is the optimisation the angular block rests on -- fifteen numbers instead of eighty-one at order four -- so it is
    checked against an explicit outer product rather than argued for.
    """
    generator = torch.Generator().manual_seed(0)
    vector = torch.randn(3, generator=generator)
    for degree in range(1, 5):
        full = vector
        for _ in range(degree - 1):
            full = torch.tensordot(full, vector, dims=0)
        exponents, multiplicities = monomial_table(degree)
        compact = sum(multiplicity * (vector[0] ** first * vector[1] ** second * vector[2] ** third) ** 2
                      for (first, second, third), multiplicity in zip(exponents, multiplicities))
        assert float(compact) == pytest.approx(float(full.pow(2).sum()))
        assert len(exponents) == (degree + 1) * (degree + 2) // 2


def test_the_power_spectrum_equals_the_brute_force_sum_over_neighbour_pairs() -> None:
    """
    The moment expansion is asserted against the definition it replaces.

    The block computes ``sum_{j,k} w_j w_k P_l(u_j . u_k)`` in time linear in the number of edges, by rewriting each power
    of the dot product as a squared norm of a summed moment tensor. The definition is quadratic in the coordination number.
    On a small cell they must agree exactly, and if they do not, every angular number this project reports is wrong.
    """
    frac, cell, atoms = random_structure(5, CUBIC, seed=5)
    neighbors = periodic_neighbors(frac, cell, atoms, SPEC.cutoff)
    computed = angular_block(neighbors, frac.shape[0], SPEC)

    for center in range(frac.shape[0]):
        selected = neighbors.center == center
        weight = cosine_cutoff(neighbors.distance[selected], SPEC.cutoff)
        unit = neighbors.vector[selected] / neighbors.distance[selected].unsqueeze(-1)
        cosines = unit @ unit.T
        outer = weight.unsqueeze(-1) * weight.unsqueeze(0)
        mass = float(weight.sum())
        for order in range(1, SPEC.max_angular_order + 1):
            brute = float((outer * torch.tensor(eval_legendre(order, cosines.numpy()))).sum())
            assert float(computed[center, order - 1]) == pytest.approx(brute / mass ** 2, abs=1.0e-10)
        assert float(computed[center, -1]) == pytest.approx(math.log1p(mass ** 2))


def test_the_power_spectrum_is_bounded_to_the_unit_interval() -> None:
    """
    Each channel is a squared spherical-harmonic norm over the squared mass, so it lies in ``[0, 1]`` by construction.

    Worth asserting because it is the reason no learned or dataset-derived normalisation layer is needed in front of the
    projection: the channels are already on a common, bounded scale, so a run cannot depend on statistics of the split it
    was fitted on.
    """
    frac, cell, atoms = random_batch([8, 6], cells=[CUBIC, SKEWED], seed=9)
    descriptor = describe(frac, cell, atoms)
    spectrum = descriptor[:, SPEC.radial_dim:SPEC.radial_dim + SPEC.max_angular_order]
    assert float(spectrum.min()) >= -1.0e-12
    assert float(spectrum.max()) <= 1.0 + 1.0e-12
    shells = descriptor[:, :SPEC.num_shells]
    assert torch.allclose(shells.sum(dim=-1), torch.ones(frac.shape[0]), atol=1.0e-10)


def test_the_descriptor_permutes_with_the_atoms() -> None:
    """Relabelling the atoms permutes the rows and changes nothing else."""
    frac, cell, atoms = random_structure(7, SKEWED, seed=15)
    order = torch.randperm(7, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(describe(frac, cell, atoms)[order], describe(frac[order], cell, atoms), atol=1.0e-12)


def test_the_descriptor_ignores_where_the_crystal_sits() -> None:
    """A rigid translation of every atom leaves every environment alone."""
    frac, cell, atoms = random_structure(6, SKEWED, seed=21)
    moved = frac + torch.tensor([0.31, -0.77, 1.4])
    assert torch.allclose(describe(frac, cell, atoms), describe(moved, cell, atoms), atol=1.0e-10)


def test_the_descriptor_ignores_how_the_crystal_is_oriented() -> None:
    """
    Rotating the lattice vectors rotates every edge and leaves every distance and every angle between edges.

    The channels are built from lengths and from dot products of unit vectors, so this is invariance by construction rather
    than by learning -- which is the point of choosing them over the raw offsets the trunk already receives.
    """
    frac, cell, atoms = random_structure(6, SKEWED, seed=25)
    turned = cell @ rotation([0.7, -1.3, 2.1])
    assert torch.allclose(describe(frac, cell, atoms), describe(frac, turned, atoms), atol=1.0e-10)


def test_the_descriptor_ignores_which_basis_the_lattice_is_written_in() -> None:
    """
    An integer basis change of determinant one describes the same crystal, so the descriptor must not move.

    A cell whose vectors are recombined has different lengths, different angles and a different Gram matrix -- which is
    what the trunk's own edge features are built from -- while the set of physical neighbours is untouched.
    """
    frac, cell, atoms = random_structure(5, SKEWED, seed=27)
    change = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=frac.dtype)
    assert float(torch.linalg.det(change)) == pytest.approx(1.0)
    rebased = (change @ cell[0]).unsqueeze(0)
    moved = frac @ torch.linalg.inv(change)
    assert torch.allclose(describe(frac, cell, atoms), describe(moved, rebased, atoms), atol=1.0e-10)


def test_the_descriptor_ignores_how_many_times_the_cell_is_tiled() -> None:
    """
    Every atom of a supercell has the environment its counterpart in the primitive cell had.

    This is the invariance a fully connected graph most conspicuously lacks: its edge count per atom *is* the number of
    atoms in the cell, so tiling by two along each axis changes what it sees by a factor of eight while changing no physics.
    """
    frac, cell, atoms = random_structure(3, CUBIC, seed=33)
    primitive = describe(frac, cell, atoms)
    tiled = describe(*supercell(frac, CUBIC, repeats=2))
    assert tiled.shape[0] == 24
    for index in range(3):
        # The first tile holds the original atoms in the original order.
        assert torch.allclose(primitive[index], tiled[index], atol=1.0e-10)


def test_a_site_with_no_neighbours_is_finite_and_exactly_zero() -> None:
    """
    An isolated atom must produce a finite descriptor without a branch on the coordination number.

    Not hypothetical: the base distribution places atoms uniformly at random in a cell, and early in denoising a site can
    be empty inside 6 Angstrom. A NaN there would poison the whole batch's gradient.
    """
    descriptor = describe(*cluster([]))
    assert bool(torch.isfinite(descriptor).all())
    assert bool((descriptor == 0.0).all())


def test_a_site_with_one_neighbour_is_finite() -> None:
    """One neighbour makes every angular channel a self-pair, which must be a finite one rather than a division by nothing."""
    descriptor = describe(*cluster([(1, 0, 0)]))
    assert bool(torch.isfinite(descriptor).all())
    spectrum = descriptor[0, SPEC.radial_dim:SPEC.radial_dim + SPEC.max_angular_order]
    # Every Legendre polynomial is one at unit argument, so a single neighbour saturates every normalised channel.
    assert torch.allclose(spectrum, torch.ones_like(spectrum), atol=1.0e-10)


def test_a_neighbour_at_the_cutoff_contributes_nothing() -> None:
    """
    The descriptor is continuous where the neighbour list is discontinuous.

    A neighbour is admitted or dropped by a hard comparison against the cutoff, so without the smooth envelope the
    descriptor would jump as an atom drifted across 6 Angstrom -- at every one of 210 integration steps.
    """
    assert float(cosine_cutoff(torch.tensor([SPEC.cutoff]), SPEC.cutoff)) == pytest.approx(0.0)
    without = describe(*cluster(OCTAHEDRAL))
    for gap in (1.0e-3, 1.0e-4, 1.0e-5):
        extra = cluster(OCTAHEDRAL + [(0.0, 0.0, 1.0)])
        frac, cell, atoms = extra
        # Push the extra neighbour out to just inside the cutoff, leaving the octahedron at 2 Angstrom.
        frac = frac.clone()
        frac[-1] = frac[0] + torch.tensor([0.0, 0.0, (SPEC.cutoff - gap) / float(cell[0, 2, 2])])
        assert torch.allclose(describe(frac, cell, atoms)[0], without[0], atol=1.0e-4), gap


def test_the_radial_block_reads_coordination_and_shell_radius() -> None:
    """
    More neighbours must raise the coordination channel, and a wider shell must move the distribution outwards.

    This is the minimum the radial block has to do, and it is the thing ``CSPNet``'s mean aggregation cannot do on its own:
    a count divided by an atom count that is not an input feature.
    """
    four = describe(*cluster(TETRAHEDRAL, radius=2.0))[0]
    six = describe(*cluster(OCTAHEDRAL, radius=2.0))[0]
    far = describe(*cluster(TETRAHEDRAL, radius=3.5))[0]
    coordination = SPEC.num_shells
    assert float(six[coordination]) > float(four[coordination])
    assert float(four[coordination]) == pytest.approx(math.log1p(4.0 * float(cosine_cutoff(torch.tensor([2.0]),
                                                                                          SPEC.cutoff))))
    # Same count, larger radius: the distribution's centre of mass moves out while the count channel barely moves.
    centres = torch.linspace(0.0, SPEC.cutoff, SPEC.num_shells)
    assert float((far[:coordination] * centres).sum()) > float((four[:coordination] * centres).sum()) + 1.0


def test_two_four_coordinate_shapes_share_a_radial_block_and_are_separated_by_the_angular_one() -> None:
    """
    The load-bearing test for the angular channels.

    A tetrahedron and a square plane have four neighbours at one radius each, so their radial blocks agree to the last bit
    and no radial descriptor can distinguish them. Their order-two power-spectrum components are 0 and 1/4, which are
    values that can be worked out by hand, so this is a check against arithmetic rather than against a previous run.
    """
    tetrahedral = describe(*cluster(TETRAHEDRAL))[0]
    planar = describe(*cluster(SQUARE_PLANAR))[0]
    assert torch.allclose(tetrahedral[:SPEC.radial_dim], planar[:SPEC.radial_dim], atol=1.0e-12)
    assert not torch.allclose(tetrahedral[SPEC.radial_dim:], planar[SPEC.radial_dim:], atol=1.0e-6)
    second = SPEC.radial_dim + 1
    assert float(tetrahedral[second]) == pytest.approx(0.0, abs=1.0e-10)
    assert float(planar[second]) == pytest.approx(0.25, abs=1.0e-10)


def test_the_octahedron_the_tetrahedron_and_the_body_centred_site_are_all_separated() -> None:
    """
    Order two alone is not enough, which is why the block runs to order four.

    A tetrahedron and an octahedron both have order-two component zero. Their order-four components are 0.2593 and 0.5833.
    A body-centred eight-fold site shares the tetrahedron's order-four value and is separated from it by order three and by
    coordination.
    """
    shapes = {name: describe(*cluster(directions))[0]
              for name, directions in (("tetrahedral", TETRAHEDRAL), ("octahedral", OCTAHEDRAL),
                                       ("body_centred", BODY_CENTRED))}
    fourth = SPEC.radial_dim + 3
    assert float(shapes["tetrahedral"][fourth]) == pytest.approx(0.259259, abs=1.0e-5)
    assert float(shapes["octahedral"][fourth]) == pytest.approx(0.583333, abs=1.0e-5)
    for first, second in (("tetrahedral", "octahedral"), ("tetrahedral", "body_centred"),
                          ("octahedral", "body_centred")):
        assert not torch.allclose(shapes[first][SPEC.radial_dim:], shapes[second][SPEC.radial_dim:], atol=1.0e-4), \
            f"{first} and {second} share an angular block"


@pytest.mark.parametrize("mode", list(FeatureMode))
def test_a_mode_zeroes_exactly_the_channels_it_disables(mode: FeatureMode) -> None:
    """
    Every mode returns the same width, with the disabled blocks exactly zero.

    That is what lets all four arms build the same projection with the same parameter count, so a difference between two
    arms is a difference in what the network was shown rather than in how large it was.
    """
    frac, cell, atoms = random_batch([6, 5], seed=37)
    descriptor = local_environment_descriptor(periodic_neighbors(frac, cell, atoms, SPEC.cutoff),
                                             frac.shape[0], SPEC, mode)
    assert descriptor.shape == (frac.shape[0], SPEC.dim)
    mask = SPEC.channel_mask(mode)
    assert bool((descriptor[:, ~mask] == 0.0).all())
    if mode.uses_geometry:
        assert float(descriptor[:, mask].abs().max()) > 0.0
    reference = local_environment_descriptor(periodic_neighbors(frac, cell, atoms, SPEC.cutoff),
                                            frac.shape[0], SPEC, FeatureMode.BOTH)
    # An enabled block is bit-identical to the same block in ``both``: the modes select channels, they do not rescale them.
    assert torch.equal(descriptor[:, mask], reference[:, mask])


def test_a_longer_neighbour_list_is_narrowed_rather_than_read() -> None:
    """
    One neighbour list built at the larger of the two radii serves both consumers, so the descriptor must drop the
    edges beyond its own cutoff itself. Otherwise turning the periodic graph on would silently widen the descriptor too,
    and the two factors would not be independent.
    """
    frac, cell, atoms = random_structure(6, CUBIC, seed=41)
    exact = local_environment_descriptor(periodic_neighbors(frac, cell, atoms, SPEC.cutoff), 6, SPEC)
    wider = local_environment_descriptor(periodic_neighbors(frac, cell, atoms, SPEC.cutoff + 3.0), 6, SPEC)
    assert torch.allclose(exact, wider, atol=1.0e-12)


def test_the_statistics_report_flags_a_dead_channel() -> None:
    """The audit's summary has to be able to say that a channel never varies, which is a channel the projection cannot use."""
    frac, cell, atoms = random_batch([6, 6], seed=43)
    summary = descriptor_statistics(describe(frac, cell, atoms), SPEC)
    assert set(summary) == set(SPEC.channel_names())
    assert all(entry["finite_fraction"] == 1.0 for entry in summary.values())
    assert summary["log_coordination"]["std"] > 0.0
    constant = descriptor_statistics(torch.ones((5, SPEC.dim)), SPEC)
    assert constant["shell_0"]["std"] == pytest.approx(0.0)


def test_the_descriptor_carries_no_species_information() -> None:
    """
    Geometry alone, by construction: the descriptor never receives the atomic numbers.

    Asserted rather than assumed because the earlier coordination-geometry work spent a week on a label that turned out to
    be mostly chemistry, and the probe in Phase 1 measures a gain over a chemistry-only floor. If species could leak
    through the feature, that gain would not mean what it says.
    """
    frac, cell, atoms = random_structure(6, CUBIC, seed=47)
    neighbors = periodic_neighbors(frac, cell, atoms, SPEC.cutoff)
    fields = set(vars(neighbors))
    assert "species" not in fields and "atom_types" not in fields
    assert local_environment_descriptor(neighbors, 6, SPEC).shape == (6, SPEC.dim)


def test_a_bad_specification_is_refused() -> None:
    """A cutoff of zero or a single shell has no width, and a silent fallback would describe nothing."""
    with pytest.raises(ValueError, match="cutoff must be positive"):
        DescriptorSpec(cutoff=0.0)
    with pytest.raises(ValueError, match="At least two shells"):
        DescriptorSpec(num_shells=1)
    with pytest.raises(ValueError, match="angular order"):
        DescriptorSpec(max_angular_order=0)
