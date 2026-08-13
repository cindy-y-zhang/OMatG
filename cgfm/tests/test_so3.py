"""Tests for the geodesic path on SO(3) and for the template symmetry that makes its target ambiguous."""

import numpy as np
import pytest
import torch
from cgfm.so3 import (angle_between, canonicalise, endpoint, exp_map, geodesic, hat, log_map, project, sample_uniform,
                      stabiliser, vee, velocity, vertex_distance)


OCTAHEDRON = 2.0 * np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                             [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
"""Six vertices on the axes, whose rotation group has order twenty-four."""

TETRAHEDRON = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
"""Four vertices on alternate cube corners, whose rotation group has order twelve."""

SQUARE = 2.0 * np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
"""Four coplanar vertices, whose rotation group has order eight."""

DUMBBELL = np.array([[0.0, 0.0, 1.6], [0.0, 0.0, -1.6]])
"""Two collinear vertices, stabilised by a continuous rotation about their own axis."""


def random_rotations(count: int, seed: int = 0) -> torch.Tensor:
    """
    Draw rotations from a fixed seed.

    :param count:
        Number of rotations.
    :type count: int
    :param seed:
        Seed of the random generator.
        Defaults to 0.
    :type seed: int

    :return:
        Rotation matrices of shape (count, 3, 3).
    :rtype: torch.Tensor
    """
    return sample_uniform(count, generator=torch.Generator().manual_seed(seed))


def assert_rotations(matrices: torch.Tensor) -> None:
    """
    Assert that every matrix of a batch is a proper rotation.

    :param matrices:
        Matrices of shape (..., 3, 3).
    :type matrices: torch.Tensor
    """
    identity = torch.eye(3, dtype=matrices.dtype).expand(matrices.shape)
    assert torch.allclose(matrices @ matrices.transpose(-1, -2), identity, atol=1e-10)
    assert torch.allclose(torch.linalg.det(matrices), torch.ones(matrices.shape[:-2], dtype=matrices.dtype), atol=1e-10)


def test_hat_and_vee_are_inverse():
    vectors = torch.randn(7, 3)
    assert torch.allclose(vee(hat(vectors)), vectors, atol=1e-12)


def test_hat_produces_a_cross_product():
    first, second = torch.randn(5, 3), torch.randn(5, 3)
    assert torch.allclose((hat(first) @ second[..., None])[..., 0], torch.cross(first, second, dim=-1), atol=1e-12)


def test_exp_map_produces_rotations():
    assert_rotations(exp_map(3.0 * torch.randn(11, 3)))


def test_exp_map_of_zero_is_the_identity():
    assert torch.allclose(exp_map(torch.zeros(4, 3)), torch.eye(3).expand(4, 3, 3), atol=1e-14)


def test_exp_map_is_finite_and_smooth_at_a_vanishing_angle():
    tiny = torch.tensor([[1e-14, 0.0, 0.0], [0.0, 1e-8, 0.0]], requires_grad=True)
    exp_map(tiny).sum().backward()
    assert torch.isfinite(tiny.grad).all()


def test_log_map_inverts_exp_map():
    vectors = torch.randn(23, 3)
    vectors = vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True) * torch.rand(23, 1) * 3.0
    assert torch.allclose(log_map(exp_map(vectors)), vectors, atol=1e-9)


def test_exp_map_inverts_log_map_for_uniform_rotations():
    rotations = random_rotations(64)
    assert torch.allclose(exp_map(log_map(rotations)), rotations, atol=1e-9)


def test_log_map_of_the_identity_is_zero():
    assert torch.allclose(log_map(torch.eye(3).expand(3, 3, 3)), torch.zeros(3, 3), atol=1e-12)


def test_log_map_handles_a_half_turn():
    axes = torch.randn(16, 3)
    axes = axes / torch.linalg.norm(axes, dim=-1, keepdim=True)
    rotations = exp_map(np.pi * axes)
    assert torch.allclose(exp_map(log_map(rotations)), rotations, atol=1e-8)
    assert torch.allclose(torch.linalg.norm(log_map(rotations), dim=-1), torch.full((16,), np.pi), atol=1e-6)


def test_log_map_is_continuous_approaching_a_half_turn():
    axis = torch.tensor([[0.3, -0.5, 0.81]])
    axis = axis / torch.linalg.norm(axis)
    below = log_map(exp_map((np.pi - 2e-3) * axis))
    above = log_map(exp_map((np.pi - 5e-4) * axis))
    assert torch.allclose(below / torch.linalg.norm(below), above / torch.linalg.norm(above), atol=1e-4)


def test_project_leaves_a_rotation_alone():
    rotations = random_rotations(8)
    assert torch.allclose(project(rotations), rotations, atol=1e-12)


def test_project_repairs_a_perturbed_rotation():
    rotations = random_rotations(8)
    repaired = project(rotations + 0.01 * torch.randn(8, 3, 3))
    assert_rotations(repaired)
    assert angle_between(rotations, repaired).max() < 0.05


def test_project_excludes_reflections():
    reflection = torch.diag(torch.tensor([1.0, 1.0, -1.0])).expand(2, 3, 3)
    assert_rotations(project(reflection))


def test_sample_uniform_produces_rotations():
    assert_rotations(random_rotations(256))


def test_sample_uniform_has_no_preferred_orientation():
    # The mean of the Haar measure on SO(3) is the zero matrix, so a biased sampler shows up as a non-vanishing mean.
    assert random_rotations(20000, seed=3).mean(dim=0).abs().max() < 0.02


def test_sample_uniform_is_reproducible():
    assert torch.equal(random_rotations(5, seed=1), random_rotations(5, seed=1))


def test_geodesic_hits_both_endpoints():
    start, end = random_rotations(9, seed=1), random_rotations(9, seed=2)
    assert torch.allclose(geodesic(start, end, torch.zeros(9)), start, atol=1e-10)
    assert torch.allclose(geodesic(start, end, torch.ones(9)), end, atol=1e-9)


def test_geodesic_stays_on_the_group():
    start, end = random_rotations(9, seed=1), random_rotations(9, seed=2)
    for time in (0.1, 0.5, 0.9):
        assert_rotations(geodesic(start, end, torch.full((9,), time)))


def test_geodesic_covers_the_angle_at_a_constant_rate():
    start, end = random_rotations(9, seed=1), random_rotations(9, seed=2)
    total = angle_between(start, end)
    for time in (0.25, 0.5, 0.75):
        travelled = angle_between(start, geodesic(start, end, torch.full((9,), time)))
        assert torch.allclose(travelled, time * total, atol=1e-8)


def test_velocity_is_constant_along_the_path():
    start, end = random_rotations(9, seed=1), random_rotations(9, seed=2)
    reference = velocity(start, end, torch.zeros(9))
    for time in (0.3, 0.6, 0.9):
        current = geodesic(start, end, torch.full((9,), time))
        assert torch.allclose(velocity(current, end, torch.full((9,), time)), reference, atol=1e-8)


def test_endpoint_inverts_velocity():
    start, end = random_rotations(9, seed=1), random_rotations(9, seed=2)
    times = torch.full((9,), 0.4)
    current = geodesic(start, end, times)
    assert torch.allclose(endpoint(current, velocity(current, end, times), times), end, atol=1e-9)


def test_angle_between_is_zero_for_equal_rotations():
    rotations = random_rotations(6)
    assert torch.allclose(angle_between(rotations, rotations), torch.zeros(6), atol=1e-7)


@pytest.mark.parametrize("offsets, order", [(OCTAHEDRON, 24), (TETRAHEDRON, 12), (SQUARE, 8)])
def test_stabiliser_finds_the_whole_rotation_group(offsets, order):
    assert len(stabiliser(offsets, ("O",) * len(offsets))) == order


def test_stabiliser_elements_map_the_template_onto_itself():
    group = stabiliser(OCTAHEDRON, ("O",) * 6)
    for rotation in group:
        rotated = OCTAHEDRON @ rotation.T
        assert np.max(np.min(np.linalg.norm(rotated[:, None, :] - OCTAHEDRON[None, :, :], axis=-1), axis=-1)) < 1e-8


def test_stabiliser_starts_from_the_identity():
    assert np.allclose(stabiliser(OCTAHEDRON, ("O",) * 6)[0], np.eye(3), atol=1e-8)


def test_stabiliser_respects_vertex_species():
    # Colouring one axis of the octahedron differently leaves only the rotations that keep that axis in place.
    group = stabiliser(OCTAHEDRON, ("F", "F", "O", "O", "O", "O"))
    assert len(group) == 8
    for rotation in group:
        assert np.allclose(np.abs(rotation @ OCTAHEDRON[0]), np.abs(OCTAHEDRON[0]), atol=1e-8)


def test_stabiliser_species_blind_recovers_the_full_octahedral_group():
    coloured = stabiliser(OCTAHEDRON, ("F", "F", "O", "O", "O", "O"), species_aware=True)
    blind = stabiliser(OCTAHEDRON, ("F", "F", "O", "O", "O", "O"), species_aware=False)
    assert len(coloured) == 8
    assert len(blind) == 24


def test_stabiliser_of_a_collinear_template_is_trivial():
    assert len(stabiliser(DUMBBELL, ("O", "O"))) == 1
    assert len(stabiliser(DUMBBELL[:1], ("O",))) == 1


def test_canonicalise_picks_the_nearest_representative():
    group = torch.from_numpy(stabiliser(OCTAHEDRON, ("O",) * 6)).expand(4, 24, 3, 3)
    reference = random_rotations(4, seed=5)
    target = reference @ torch.from_numpy(stabiliser(OCTAHEDRON, ("O",) * 6)[7]).expand(4, 3, 3)

    chosen = canonicalise(target, reference, group)
    assert torch.allclose(chosen, reference, atol=1e-6)


def test_canonicalise_shrinks_the_distance_to_the_reference():
    group = torch.from_numpy(stabiliser(OCTAHEDRON, ("O",) * 6)).expand(16, 24, 3, 3)
    reference, target = random_rotations(16, seed=6), random_rotations(16, seed=7)

    chosen = canonicalise(target, reference, group)
    assert (angle_between(reference, chosen) <= angle_between(reference, target) + 1e-9).all()
    # A group of twenty-four elements leaves no orientation further than about a quarter turn from a representative.
    assert angle_between(reference, chosen).max() < np.pi / 3.0


def test_canonicalise_is_a_no_op_for_a_trivial_group():
    group = torch.eye(3).expand(5, 1, 3, 3)
    reference, target = random_rotations(5, seed=8), random_rotations(5, seed=9)
    assert torch.allclose(canonicalise(target, reference, group), target, atol=1e-12)


def test_vertex_distance_vanishes_for_equal_orientations():
    rotations = random_rotations(4, seed=2)
    offsets = torch.from_numpy(OCTAHEDRON).expand(4, 6, 3)
    assert torch.allclose(vertex_distance(rotations, rotations, offsets, torch.ones(4, 6)), torch.zeros(4), atol=1e-12)


def test_vertex_distance_ignores_a_continuous_stabiliser():
    rotations = random_rotations(4, seed=2)
    about_the_axis = exp_map(torch.tensor([0.0, 0.0, 1.0]) * torch.rand(4, 1) * 6.0)
    offsets = torch.from_numpy(DUMBBELL).expand(4, 2, 3)

    equivalent = rotations @ about_the_axis
    assert angle_between(rotations, equivalent).min() > 0.1
    assert torch.allclose(vertex_distance(rotations, equivalent, offsets, torch.ones(4, 2)), torch.zeros(4), atol=1e-12)


def test_vertex_distance_sees_a_finite_stabiliser_that_canonicalise_then_removes():
    group = torch.from_numpy(stabiliser(OCTAHEDRON, ("O",) * 6))
    offsets = torch.from_numpy(OCTAHEDRON).expand(4, 6, 3)
    mask = torch.ones(4, 6)
    rotations = random_rotations(4, seed=2)
    equivalent = rotations @ group[7].expand(4, 3, 3)

    # The octahedron is a set of six labelled slots, and a four-fold turn moves what sits in each of them.
    assert (vertex_distance(rotations, equivalent, offsets, mask) > 1.0).all()
    chosen = canonicalise(equivalent, rotations, group.expand(4, len(group), 3, 3))
    assert torch.allclose(vertex_distance(rotations, chosen, offsets, mask), torch.zeros(4), atol=1e-10)


def test_vertex_distance_costs_nothing_for_a_block_with_no_vertices():
    offsets = torch.zeros(2, 6, 3)
    distance = vertex_distance(random_rotations(2, seed=2), random_rotations(2, seed=3), offsets, torch.zeros(2, 6))
    assert torch.allclose(distance, torch.zeros(2), atol=1e-14)


def test_vertex_distance_ignores_padded_slots():
    rotations, other = random_rotations(3, seed=2), random_rotations(3, seed=3)
    offsets = torch.from_numpy(OCTAHEDRON).expand(3, 6, 3).clone()
    mask = torch.ones(3, 6)
    mask[:, 4:] = 0.0

    padded = torch.cat([offsets, 5.0 * torch.randn(3, 2, 3)], dim=1)
    assert torch.allclose(vertex_distance(rotations, other, offsets, mask),
                          vertex_distance(rotations, other, padded, torch.cat([mask, torch.zeros(3, 2)], dim=1)),
                          atol=1e-12)


def test_vertex_distance_on_velocities_ignores_the_continuous_stabiliser():
    # A collinear template turns about its own axis without moving, so a velocity along that axis must cost nothing,
    # which is what lets coordination numbers one and two be trained without a special case.
    dumbbell = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]], dtype=torch.float64)
    axis = torch.tensor([[0.0, 0.0, 1.3]], dtype=torch.float64)
    zero = torch.zeros(1, 3, 3, dtype=torch.float64)
    assert torch.allclose(vertex_distance(hat(axis), zero, dumbbell.expand(1, 2, 3), torch.ones(1, 2)),
                          torch.zeros(1, dtype=torch.float64), atol=1e-14)


def test_vertex_distance_on_velocities_needs_no_time_weighting():
    # The endpoint recovered from a velocity is R_t exp((1 - t) V), so an endpoint loss fades as t approaches one while
    # the velocity loss does not. Both are computed here, and only the velocity loss is flat in t.
    offsets = torch.from_numpy(OCTAHEDRON).expand(1, 6, 3)
    mask = torch.ones(1, 6)
    truth, guess = torch.tensor([[0.3, -0.2, 0.5]]), torch.tensor([[0.1, 0.4, -0.3]])
    velocity_losses, endpoint_losses = [], []
    for t in (0.1, 0.5, 0.9, 0.99):
        current = exp_map(t * truth)
        velocity_losses.append(vertex_distance(hat(guess), hat(truth), offsets, mask).item())
        endpoint_losses.append(vertex_distance(current @ exp_map((1.0 - t) * guess),
                                               current @ exp_map((1.0 - t) * truth), offsets, mask).item())
    assert velocity_losses == pytest.approx([velocity_losses[0]] * 4)
    assert endpoint_losses[-1] < 0.02 * endpoint_losses[0]


def test_vertex_distance_grows_with_the_angle():
    axis = torch.tensor([[0.0, 0.0, 1.0]]).expand(3, 3)
    offsets = torch.from_numpy(TETRAHEDRON).expand(3, 4, 3)
    identity = torch.eye(3).expand(3, 3, 3)
    distances = [vertex_distance(identity, exp_map(angle * axis), offsets, torch.ones(3, 4))[0].item()
                 for angle in (0.05, 0.2, 0.5)]
    assert distances[0] < distances[1] < distances[2]
