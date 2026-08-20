"""
Tests of the periodic neighbour list.

Everything downstream is a sum over this list, so an error here is invisible: the descriptors stay finite, the model trains,
and the numbers describe a neighbourhood that is not the one in the crystal. The tests therefore assert exact degree and
exact image multisets against pymatgen rather than checking that the nearest distance looks about right, and they cover the
two defects that make the repository's existing builder unusable -- a hardcoded one-image enumeration and a cutoff replaced
by the cell's own interplanar spacing.
"""

from collections import Counter
import pytest
import torch
from pymatgen.core import Lattice, Structure

from direct_geometry.neighbors import (MAX_DEGREE, MAX_IMAGES_PER_DIMENSION, bounded_cutoff, constant_degree_radius,
                                       minimum_image_distance, periodic_neighbors, required_repetitions)
from .conftest import CUBIC, FLAT, SHORT, SKEWED, random_batch, random_structure, rotation, single, supercell


def reference_neighbors(frac: torch.Tensor, cell: torch.Tensor, cutoff: float) -> dict[int, Counter]:
    """
    Return pymatgen's neighbour list as a multiset of (neighbour index, image) per atom.

    :param frac:
        Fractional coordinates of shape ``(atoms, 3)``.
    :type frac: torch.Tensor
    :param cell:
        Lattice of shape ``(3, 3)``.
    :type cell: torch.Tensor
    :param cutoff:
        Largest neighbour distance, in Angstrom.
    :type cutoff: float

    :return:
        For every atom, a multiset of its neighbours keyed by index and image shift.
    :rtype: dict[int, Counter]
    """
    structure = Structure(Lattice(cell.numpy()), ["Si"] * frac.shape[0], frac.numpy())
    out = {}
    for index, shell in enumerate(structure.get_all_neighbors(cutoff)):
        out[index] = Counter((site.index, tuple(int(round(value)) for value in site.image)) for site in shell)
    return out


def computed_neighbors(frac: torch.Tensor, cell: torch.Tensor, cutoff: float) -> dict[int, Counter]:
    """
    Return this package's neighbour list in the same form.

    :param frac:
        Fractional coordinates of shape ``(atoms, 3)``.
    :type frac: torch.Tensor
    :param cell:
        Lattice of shape ``(3, 3)``.
    :type cell: torch.Tensor
    :param cutoff:
        Largest neighbour distance, in Angstrom.
    :type cutoff: float

    :return:
        For every atom, a multiset of its neighbours keyed by index and image shift.
    :rtype: dict[int, Counter]
    """
    neighbors = periodic_neighbors(*single(frac, cell), cutoff)
    out = {index: Counter() for index in range(frac.shape[0])}
    for center, neighbor, image in zip(neighbors.center.tolist(), neighbors.neighbor.tolist(),
                                       neighbors.image.tolist()):
        out[center][(neighbor, tuple(image))] += 1
    return out


@pytest.mark.parametrize("cell,name", [(CUBIC, "cubic"), (SKEWED, "skewed"), (SHORT, "short")])
@pytest.mark.parametrize("cutoff", [4.0, 6.0])
def test_the_neighbour_list_matches_pymatgen_image_for_image(cell: torch.Tensor, name: str, cutoff: float) -> None:
    """
    The neighbourhood is exactly pymatgen's, down to which periodic image each edge stands for.

    Distances alone would not catch the failure that matters. A builder that finds every distance once but misses the
    seven other images at that distance produces a plausible histogram and a coordination number that is out by a factor
    of eight, which is the defect the fully connected graph has and the reason this list exists.
    """
    frac, _, _ = random_structure(6, cell, seed=3)
    assert computed_neighbors(frac, cell, cutoff) == reference_neighbors(frac, cell, cutoff), name


def test_a_short_cell_needs_more_than_the_one_image_the_repository_builder_offers() -> None:
    """
    The enumeration is driven by the cutoff and the cell, not by a constant.

    ``radius_graph_pbc`` hardcodes ``max_rep = 1``. On this cell a 6 Angstrom cutoff reaches two cells out along every
    axis, so a fixed one-image enumeration cannot even in principle find the neighbourhood, and would not say so.
    """
    repetitions = required_repetitions(SHORT.unsqueeze(0), 6.0)
    assert torch.all(repetitions >= 2), repetitions
    frac, _, _ = random_structure(2, SHORT, seed=1)
    neighbors = periodic_neighbors(*single(frac, SHORT), 6.0)
    assert int(neighbors.image.abs().max()) >= 2
    assert computed_neighbors(frac, SHORT, 6.0) == reference_neighbors(frac, SHORT, 6.0)


def test_the_cutoff_is_the_one_that_was_asked_for() -> None:
    """
    The requested radius is used, not the cell's smallest interplanar spacing.

    ``radius_graph_pbc`` discards its ``radius`` argument and substitutes ``min_dist.min() + 0.01``, so its effective
    cutoff is a property of each cell. Two structures in one batch are then described at two different radii, and no
    descriptor built on them is comparable across a dataset.
    """
    frac, _, _ = random_structure(4, CUBIC, seed=2)
    longest = {cutoff: float(periodic_neighbors(*single(frac, CUBIC), cutoff).distance.max())
               for cutoff in (3.0, 5.0, 7.0)}
    for cutoff, reach in longest.items():
        assert reach <= cutoff + 1.0e-9
    # Strictly growing, so the argument is doing work rather than being clamped to something cell-derived.
    assert longest[3.0] < longest[5.0] < longest[7.0]


def test_rock_salt_multiplicities_are_the_textbook_ones() -> None:
    """
    Six unlike neighbours at half the lattice constant and twelve like ones at its face diagonal.

    A one-edge-per-pair graph offers each atom seven edges in this eight-atom cell, which is why coordination number was
    unreadable from it: the octahedron is six edges to three distinct sites at plus and minus images. The count here is the
    physical one, and every one of the eight atoms gets it.
    """
    constant = 4.2
    # The conventional face-centred cell: the two interpenetrating sublattices, four sites each.
    frac, cell, atoms = single(
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0],
                      [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]),
        torch.eye(3, dtype=torch.float64) * constant)
    # Past the face diagonal and short of the body diagonal at 3.64, so exactly the first two shells are in range.
    neighbors = periodic_neighbors(frac, cell, atoms, 3.2)
    for center in range(8):
        histogram = Counter(round(float(value), 6) for value in neighbors.distance[neighbors.center == center])
        assert histogram == {round(constant / 2, 6): 6, round(constant * 2 ** -0.5, 6): 12}


def test_caesium_chloride_multiplicities_are_the_textbook_ones() -> None:
    """Eight unlike neighbours at ``a sqrt(3) / 2`` and six like ones at ``a``, from a cell holding two atoms."""
    constant = 5.6
    frac, cell, atoms = single(torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]), torch.eye(3) * constant)
    neighbors = periodic_neighbors(frac, cell, atoms, 6.0)
    assert neighbors.degree(2).tolist() == [14, 14]
    histogram = Counter(round(float(value), 6) for value in neighbors.distance[neighbors.center == 0])
    assert histogram == {round(constant * 3 ** 0.5 / 2, 6): 8, round(constant, 6): 6}


def test_an_atom_is_its_own_neighbour_through_a_nonzero_image_but_never_through_the_zero_one() -> None:
    """
    The only edge excluded is the zero-length self edge.

    In a cell shorter than the cutoff an atom's own images are among its nearest neighbours, and dropping the diagonal pair
    to be rid of the self edge would drop them too. So the exclusion is by length, not by index.
    """
    frac, cell, atoms = single(torch.tensor([[0.3, 0.7, 0.1]]), torch.eye(3) * 4.0)
    neighbors = periodic_neighbors(frac, cell, atoms, 6.0)
    assert len(neighbors) > 0
    assert torch.all(neighbors.center == 0) and torch.all(neighbors.neighbor == 0)
    assert torch.all(neighbors.image.abs().sum(dim=-1) > 0)
    assert float(neighbors.distance.min()) == pytest.approx(4.0)


def test_the_reported_image_reconstructs_the_offset_exactly() -> None:
    """
    ``offset == frac[neighbour] + image - frac[centre]``, which is what every consumer assumes.

    The builder folds the offset before enumerating images, for a tight repetition bound, and undoes the fold in the
    reported image. If that bookkeeping were wrong the distances would still be right and the images would be off by one
    cell, which is exactly the kind of error that survives a distance-only test.
    """
    frac, cell, atoms = random_batch([5, 4], seed=7)
    neighbors = periodic_neighbors(frac, cell, atoms, 6.0)
    expected = frac[neighbors.neighbor] + neighbors.image.to(frac.dtype) - frac[neighbors.center]
    assert torch.allclose(neighbors.offset, expected, atol=1.0e-12)
    cartesian = torch.einsum("ek,ekj->ej", neighbors.offset, cell[neighbors.edge2graph])
    assert torch.allclose(neighbors.vector, cartesian, atol=1.0e-10)
    assert torch.allclose(neighbors.distance, cartesian.norm(dim=-1), atol=1.0e-10)


def test_the_graph_is_symmetric() -> None:
    """
    Every edge has its reverse, with the opposite image and the same length.

    Message passing aggregates onto the centre, so an asymmetric list would give an atom a different neighbourhood
    depending on which end of the pair it sat at, and the descriptor would not be a function of the environment.
    """
    frac, cell, atoms = random_batch([4, 6], seed=11)
    neighbors = periodic_neighbors(frac, cell, atoms, 5.0)
    forward = Counter(zip(neighbors.center.tolist(), neighbors.neighbor.tolist(),
                          map(tuple, neighbors.image.tolist())))
    reverse = Counter((neighbor, center, tuple(-value for value in image))
                      for center, neighbor, image in forward.elements())
    assert forward == reverse


def test_batching_a_structure_does_not_change_its_neighbourhood() -> None:
    """
    A structure's edges are the same whether it is alone or beside a cell needing a different image count.

    The builder groups structures by repetition triple precisely so that one short cell does not impose its image count on
    the rest of the batch. Grouping is an optimisation, and an optimisation that changed the answer would be a bug that
    only appeared at batch sizes above one.
    """
    frac, cell, atoms = random_batch([4, 3], cells=[CUBIC, SHORT], seed=13)
    batched = periodic_neighbors(frac, cell, atoms, 6.0)
    for index, (lower, upper) in enumerate([(0, 4), (4, 7)]):
        alone = periodic_neighbors(frac[lower:upper], cell[index:index + 1], atoms[index:index + 1], 6.0)
        selected = batched.edge2graph == index
        assert Counter(zip((batched.center[selected] - lower).tolist(), (batched.neighbor[selected] - lower).tolist(),
                           map(tuple, batched.image[selected].tolist()))) == \
               Counter(zip(alone.center.tolist(), alone.neighbor.tolist(), map(tuple, alone.image.tolist())))


def test_a_supercell_reproduces_every_environment() -> None:
    """
    Tiling a crystal changes the cell and not the physics, so every atom keeps its distance multiset.

    This is the invariance a descriptor most easily fails, because the number of atoms in the cell -- which is what a
    fully connected graph counts -- changes by a factor of eight while nothing physical does.
    """
    frac, cell, atoms = random_structure(3, CUBIC, seed=17)
    primitive = periodic_neighbors(frac, cell, atoms, 6.0)
    tiled = periodic_neighbors(*supercell(frac, CUBIC, repeats=2), 6.0)
    for index in range(3):
        original = sorted(round(float(value), 8) for value in primitive.distance[primitive.center == index])
        # The first three atoms of the supercell are the first tile, in the original order.
        copied = sorted(round(float(value), 8) for value in tiled.distance[tiled.center == index])
        assert original == copied


def test_a_singular_cell_is_refused_rather_than_inverted() -> None:
    """A flat cell has no volume and no defined images. Inverting it returns large numbers instead of complaining."""
    with pytest.raises(ValueError, match="singular"):
        required_repetitions(FLAT.unsqueeze(0), 6.0)


def test_a_cell_far_shorter_than_the_cutoff_is_refused_rather_than_truncated() -> None:
    """
    The safety bound raises. A truncated neighbour list is the failure mode this package cannot tolerate, because it
    produces finite, plausible descriptors of the wrong neighbourhood.
    """
    tiny = torch.eye(3).unsqueeze(0) * 0.05
    with pytest.raises(ValueError, match="safety bound"):
        required_repetitions(tiny, 6.0)
    assert required_repetitions(torch.eye(3).unsqueeze(0) * 6.0 / MAX_IMAGES_PER_DIMENSION, 6.0).max() \
        <= MAX_IMAGES_PER_DIMENSION


def test_a_normal_cell_keeps_the_cutoff_it_asked_for() -> None:
    """
    The cutoff bound is a no-op for anything a crystal could be.

    It exists for generated cells, and if it ever engaged on ordinary structures it would be quietly shrinking the
    neighbourhood the DG1 probes measured the descriptor's value at. Checked on the whole span of test cells, including
    the deliberately short and skewed ones.
    """
    for name, cell in (("cubic", CUBIC), ("skewed", SKEWED), ("short", SHORT)):
        bound = bounded_cutoff(cell.unsqueeze(0), torch.tensor([8]), 6.0)
        assert bound.item() == pytest.approx(6.0), name


def test_a_cell_too_flat_for_the_cutoff_gets_a_smaller_one_instead_of_an_exception() -> None:
    """
    A cell needing more images than the bound allows is served at a reduced radius, not refused.

    This is the failure that killed an overfit run at epoch 1,000: not a corrupted dataset, but a cell the *sampler*
    generated while integrating an undertrained model's cell velocity. A batch containing one of those still has to be
    scored, so the list has to be defined there.
    """
    thin = torch.diag(torch.tensor([6.0, 6.0, 0.35], dtype=torch.float64)).unsqueeze(0)
    with pytest.raises(ValueError, match="safety bound"):
        required_repetitions(thin, 6.0)

    reduced = bounded_cutoff(thin, torch.tensor([3]), 6.0)
    assert 0.0 < reduced.item() < 6.0
    assert int(required_repetitions(thin, reduced).max()) <= MAX_IMAGES_PER_DIMENSION

    frac, _, atoms = random_structure(3, thin[0], seed=5)
    neighbors = periodic_neighbors(frac, thin, atoms, 6.0)
    # Every edge respects the reduced radius, and the list is still exact for that radius: the reduction is a smaller
    # question honestly answered, not a truncated answer to the original one.
    assert float(neighbors.distance.max()) <= reduced.item() + 1.0e-9
    assert computed_neighbors(frac, thin[0], reduced.item()) == reference_neighbors(frac, thin[0], reduced.item())


def test_the_reduced_cutoff_is_applied_per_structure_and_not_across_the_batch() -> None:
    """
    One pathological cell in a batch must not shrink its neighbours' radii.

    A batch-wide reduction would make an arm's descriptor depend on what else happened to be batched with it, which is
    both wrong and untraceable -- the same structure would get different features in different epochs.
    """
    thin = torch.diag(torch.tensor([6.0, 6.0, 0.35], dtype=torch.float64))
    cells = torch.stack([CUBIC, thin])
    bound = bounded_cutoff(cells, torch.tensor([3, 3]), 6.0)
    assert bound[0].item() == pytest.approx(6.0)
    assert bound[1].item() < 6.0

    frac_a, _, _ = random_structure(3, CUBIC, seed=7)
    frac_b, _, _ = random_structure(3, thin, seed=8)
    atoms = torch.tensor([3, 3])
    batched = periodic_neighbors(torch.cat([frac_a, frac_b]), cells, atoms, 6.0)
    alone = periodic_neighbors(*single(frac_a, CUBIC), 6.0)
    assert int((batched.edge2graph == 0).sum()) == len(alone)


def test_a_singular_cell_is_still_refused_by_the_bound() -> None:
    """
    Reducing a cutoff cannot rescue a cell with no volume, and pretending otherwise would hide a collapsed model.

    The distinction the bound draws is between a cell that is extreme and one that is degenerate. The first is a
    legitimate point on a generated trajectory; the second means the predictions have gone to zero or to NaN, and a run
    that has reached that state should stop rather than train on it.
    """
    with pytest.raises(ValueError, match="singular|zero volume"):
        bounded_cutoff(FLAT.unsqueeze(0), torch.tensor([3]), 6.0)


def test_a_cell_too_small_for_the_cutoff_is_bounded_by_neighbourhood_size_not_only_image_count() -> None:
    """
    A small cell is bounded by degree, which the image-count bound does not do.

    These are separate pathologies. A cell can be flat, needing an unenumerable grid to cover the cutoff; or it can be
    small, needing a perfectly ordinary grid that happens to contain tens of thousands of atoms. Bounding only images
    turns the second case from an exception into an out-of-memory error, which is worse: a half-Angstrom cubic cell asks
    for 20,562 edges per atom, or a hundred and thirty million edges at the batch size these runs use.
    """
    small = (torch.eye(3, dtype=torch.float64) * 0.5).unsqueeze(0)
    atoms = torch.tensor([8])
    frac, _, _ = random_structure(8, small[0], seed=11)

    images_only = bounded_cutoff(small, atoms, 6.0, max_degree=float("inf"))
    both = bounded_cutoff(small, atoms, 6.0)
    assert int(required_repetitions(small, images_only).max()) <= MAX_IMAGES_PER_DIMENSION
    assert both.item() < images_only.item()

    neighbors = periodic_neighbors(frac, small, atoms, 6.0)
    assert len(neighbors) / 8 < 4.0 * MAX_DEGREE
    assert computed_neighbors(frac, small[0], both.item()) == reference_neighbors(frac, small[0], both.item())


def test_the_reduced_cutoff_survives_the_precision_the_runs_actually_use() -> None:
    """
    The reduced cutoff still respects the repetition bound after a float32 round trip.

    This is not a hypothetical. The bound is derived in double and returned in the caller's dtype; training runs in
    "32-true"; and a bound placed one part in a billion inside a strict inequality rounds straight back onto it in
    float32, at which point the enumeration asks for exactly the one extra repetition the reduction existed to avoid.
    That is what it did, and every test in this file runs in double, which is why nothing here caught it.

    Swept over a range of thicknesses rather than asserted at one, because the failure needs the boundary to be hit
    almost exactly and a single cell can miss it by luck.
    """
    for thickness in torch.linspace(0.05, 1.5, 60, dtype=torch.float64):
        cell = torch.diag(torch.tensor([6.0, 6.0, float(thickness)], dtype=torch.float64)).unsqueeze(0)
        atoms = torch.tensor([4])
        for dtype in (torch.float32, torch.float64):
            narrowed = cell.to(dtype)
            reduced = bounded_cutoff(narrowed, atoms, 6.0)
            assert reduced.dtype == dtype
            assert int(required_repetitions(narrowed, reduced).max()) <= MAX_IMAGES_PER_DIMENSION, \
                (float(thickness), dtype)


def test_the_neighbourhood_bound_never_engages_on_the_real_probability_path() -> None:
    """
    Neither bound touches a structure the baseline's own path actually visits.

    This is the assertion that makes the bounds safe to have: they are there for cells the *sampler* generates while
    integrating an undertrained model, and if they ever fired on ordinary interpolated structures they would be silently
    shrinking the neighbourhood the DG1 probes measured the descriptor's value at, and the reported +26.1 accuracy points
    would not describe the model being trained.
    """
    frac, cell, atoms = random_batch((6, 9, 4), (CUBIC, SKEWED, SHORT), seed=3)
    assert torch.allclose(bounded_cutoff(cell, atoms, 6.0), torch.full((3,), 6.0, dtype=cell.dtype))


def test_the_repetition_bound_accepts_a_cutoff_per_structure() -> None:
    """
    Repetition counts are computed from each structure's own cutoff.

    Needed because the reduction is per structure; a scalar-only bound would have to use the smallest cutoff in the batch
    and would enumerate the wrong grid for everything else.
    """
    cells = torch.stack([CUBIC, CUBIC])
    per_structure = torch.tensor([1.0, 5.0], dtype=torch.float64)
    repetitions = required_repetitions(cells, per_structure)
    assert int(repetitions[0].max()) < int(repetitions[1].max())
    assert torch.equal(repetitions[0], required_repetitions(CUBIC.unsqueeze(0), 1.0)[0])
    assert torch.equal(repetitions[1], required_repetitions(CUBIC.unsqueeze(0), 5.0)[0])

    with pytest.raises(ValueError, match="one cutoff per structure"):
        required_repetitions(cells, torch.tensor([1.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="must be positive"):
        required_repetitions(cells, torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_an_edge_budget_is_refused_rather_than_truncated() -> None:
    """The edge bound raises too, for the same reason: no silent ``max_neighbors`` truncation anywhere in this package."""
    frac, cell, atoms = random_structure(8, CUBIC, seed=19)
    with pytest.raises(ValueError, match="degenerate cell"):
        periodic_neighbors(frac, cell, atoms, 6.0, max_edges=4)


def test_chunking_the_image_grid_does_not_change_the_answer() -> None:
    """Peak memory is bounded by evaluating the images in blocks, and the block size must not be a parameter of the result."""
    frac, cell, atoms = random_batch([4, 5], cells=[SKEWED, SHORT], seed=23)
    reference = periodic_neighbors(frac, cell, atoms, 6.0, shift_chunk=1000)
    for chunk in (1, 7, 27):
        other = periodic_neighbors(frac, cell, atoms, 6.0, shift_chunk=chunk)
        assert len(other) == len(reference)
        assert sorted(round(float(value), 10) for value in other.distance) == \
               sorted(round(float(value), 10) for value in reference.distance)


def test_unwrapped_coordinates_describe_the_same_crystal() -> None:
    """
    Coordinates need not be wrapped into the unit cell.

    The interpolant's corrector keeps positions in ``[0, 1)``, but an integration step can leave them a hair outside and a
    caller should not have to know. Offsets are folded before the images are enumerated, so any representative works.
    """
    frac, cell, atoms = random_structure(5, SKEWED, seed=29)
    shifted = frac + torch.tensor([1.0, -2.0, 3.0])
    first = periodic_neighbors(frac, cell, atoms, 6.0)
    second = periodic_neighbors(shifted, cell, atoms, 6.0)
    assert sorted(round(float(value), 10) for value in first.distance) == \
           sorted(round(float(value), 10) for value in second.distance)


def test_the_minimum_image_helper_agrees_with_pymatgen_on_a_skewed_cell() -> None:
    """
    The fully connected control's distance is the true minimum image, which a fractional fold alone does not give.

    Kept in this package because the control has to be measured with the right number too: comparing an arm using real
    distances against a control using folded ones would confound the factor with a bug.
    """
    frac, _, _ = random_structure(6, SKEWED, seed=31)
    structure = Structure(Lattice(SKEWED.numpy()), ["Si"] * 6, frac.numpy())
    offsets, expected = [], []
    for i in range(6):
        for j in range(6):
            offsets.append((frac[j] - frac[i]) % 1.0)
            expected.append(structure.get_distance(i, j))
    computed = minimum_image_distance(torch.stack(offsets), SKEWED.unsqueeze(0).expand(36, 3, 3))
    assert torch.allclose(computed, torch.tensor(expected), atol=1.0e-9)


def test_the_minimum_image_helper_would_collapse_a_periodic_list_and_is_therefore_not_used_on_one() -> None:
    """
    Documents, as an assertion, why there are two distance conventions in this package.

    On a periodic list the image is already in the offset. Minimising over images there reports one distance for several
    distinct neighbours, which is precisely the multiplicity the list was built to supply.
    """
    cell = torch.eye(3).unsqueeze(0).expand(3, 3, 3) * 4.0
    offsets = torch.tensor([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
    imaged = torch.einsum("ek,ekj->ej", offsets, cell).norm(dim=-1)
    assert imaged.tolist() == pytest.approx([2.0, 2.0, 6.0])
    assert float(minimum_image_distance(offsets, cell)[2]) == pytest.approx(2.0)


def test_an_empty_neighbourhood_returns_an_empty_list_of_the_right_shapes() -> None:
    """A lone atom in a large cell has no neighbours, and the consumers must get empty tensors rather than a crash."""
    frac, cell, atoms = single(torch.tensor([[0.5, 0.5, 0.5]]), torch.eye(3) * 60.0)
    neighbors = periodic_neighbors(frac, cell, atoms, 6.0)
    assert len(neighbors) == 0
    assert neighbors.image.shape == (0, 3) and neighbors.vector.shape == (0, 3)
    assert neighbors.degree(1).tolist() == [0]


@pytest.mark.parametrize("degree", [12.0, 32.0, 60.0])
def test_the_density_scaled_radius_delivers_the_degree_it_promises(degree: float) -> None:
    """
    The point of the adaptive radius is that the degree it produces is the degree it was asked for.

    Checked against an actual neighbour count rather than against the algebra it was derived from, and on a randomly
    filled cell so that the expected-count argument applies. The tolerance is wide because a finite sample of a Poisson-ish
    count is noisy, but the systematic error the test is looking for -- a wrong power, a factor of two, a radius that
    tracks the cell length instead of the density -- is far larger than that.
    """
    frac, cell, atoms = random_structure(60, CUBIC, seed=101)
    radius = constant_degree_radius(cell, atoms, degree, max_radius=100.0)
    counted = len(periodic_neighbors(frac, cell, atoms, float(radius))) / int(atoms.sum())
    assert counted == pytest.approx(degree, rel=0.15)


def test_the_density_scaled_radius_holds_the_degree_as_the_cell_is_squeezed() -> None:
    """
    The defect this radius exists to fix: a fixed radius on a denser structure reaches more atoms.

    Squeezing the cell by a factor of two in volume is roughly the density change between the baseline's cell prior and
    MPTS-52's own cells, which is what made a fixed radius unaffordable at the noisy end of the path.
    """
    frac, cell, atoms = random_structure(40, CUBIC, seed=103)
    counts, fixed_counts = [], []
    for scale in (0.7, 0.8, 1.0, 1.25):
        scaled = cell * scale
        radius = constant_degree_radius(scaled, atoms, 32.0, max_radius=100.0)
        counts.append(len(periodic_neighbors(frac, scaled, atoms, float(radius))) / int(atoms.sum()))
        fixed_counts.append(len(periodic_neighbors(frac, scaled, atoms, 5.0)) / int(atoms.sum()))
    assert max(counts) / min(counts) < 1.2
    # The comparison that motivates the whole mechanism: at a fixed radius the same squeeze changes the degree severalfold.
    assert max(fixed_counts) / min(fixed_counts) > 2.0


def test_the_density_scaled_radius_is_invariant_where_the_physics_is() -> None:
    """
    Number density is unchanged by tiling, by rotating and by relabelling, so the radius must be too.

    Supercell invariance is the one that could plausibly fail, since both the volume and the atom count change by the tile
    count and only their ratio is preserved. If it failed, a supercell would get a different graph from the crystal it
    describes, and every invariance the descriptor tests establish would be undone by the topology.
    """
    frac, cell, atoms = random_structure(4, SKEWED, seed=107)
    reference = constant_degree_radius(cell, atoms, 32.0, max_radius=100.0)

    tiled_frac, tiled_cell, tiled_atoms = supercell(frac, SKEWED, repeats=2)
    assert float(constant_degree_radius(tiled_cell, tiled_atoms, 32.0, 100.0)) == pytest.approx(float(reference))

    turn = rotation([0.4, 0.9, 0.2]).to(cell.dtype)
    rotated = torch.einsum("bij,jk->bik", cell, turn.T)
    assert float(constant_degree_radius(rotated, atoms, 32.0, 100.0)) == pytest.approx(float(reference))

    # A left-handed cell has a negative determinant, and a radius built from a signed volume would come out complex.
    flipped = cell.clone()
    flipped[:, 0] = -flipped[:, 0]
    assert float(constant_degree_radius(flipped, atoms, 32.0, 100.0)) == pytest.approx(float(reference))


def test_the_density_scaled_radius_respects_its_bound_and_refuses_nonsense() -> None:
    """
    The bound is what keeps one neighbour list serving both consumers and every edge inside the fixed edge basis.

    Without it a structure whose cell was drawn very large would ask for a radius beyond the list that was built, and the
    graph would silently be the whole list rather than the requested degree.
    """
    _, cell, atoms = random_structure(2, CUBIC, seed=109)
    assert float(constant_degree_radius(cell * 10.0, atoms, 32.0, max_radius=6.0)) == pytest.approx(6.0)
    with pytest.raises(ValueError, match="degree must be positive"):
        constant_degree_radius(cell, atoms, 0.0, 6.0)
    with pytest.raises(ValueError, match="bound must be positive"):
        constant_degree_radius(cell, atoms, 32.0, 0.0)
    with pytest.raises(ValueError, match="zero volume"):
        constant_degree_radius(FLAT.unsqueeze(0), torch.tensor([2]), 32.0, 6.0)


def test_narrowing_a_list_per_structure_keeps_each_structure_to_its_own_radius() -> None:
    """
    ``within`` takes one radius per structure, and the edges it keeps must be each structure's own, not the batch's.

    An off-by-one in the ``edge2graph`` indexing would silently give one structure another's radius, which on a batch of
    similar cells would be almost invisible and would still corrupt every count the audit reports.
    """
    frac, cell, atoms = random_batch([4, 5], cells=[CUBIC, SKEWED], seed=113)
    built = periodic_neighbors(frac, cell, atoms, 6.0)
    radii = torch.tensor([3.0, 5.0], dtype=cell.dtype)
    narrowed = built.within(radii)
    for structure, radius in enumerate(radii.tolist()):
        selected = narrowed.edge2graph == structure
        assert bool((narrowed.distance[selected] <= radius).all())
        expected = int(((built.edge2graph == structure) & (built.distance <= radius)).sum())
        assert int(selected.sum()) == expected
    with pytest.raises(ValueError, match="cutoffs for a list"):
        built.within(torch.tensor([3.0]))
