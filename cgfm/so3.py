"""
Rotations for rigid-block flow matching: the geodesic path on SO(3), and the symmetry that makes its target ambiguous.

A block pose is a translation on the torus and an orientation in SO(3). The torus half is already in OMatG, as
``PeriodicLinearInterpolant`` and its corrector. The rotation half is not: nothing in the code base moves on a Lie
group. This module supplies the pieces, in the simplest form that is still correct, namely the geodesic path

    R_t = R_0 exp(t log(R_0^T R_1)),

whose velocity in the body frame is the constant ``log(R_0^T R_1)``. That is the SO(3) analogue of the straight line
that OMatG uses for the lattice, and it is what rigid-body flow matching for molecular and metal-organic crystals uses.

The part that is not standard bookkeeping is the target. A coordination polyhedron is symmetric, so the rotation that
places it is not unique: an octahedron looks the same under twenty-four rotations, and regressing towards an arbitrary
one of them injects an error the size of a whole rotation. The two kinds of symmetry need different handling, and
conflating them is a mistake worth stating.

- A **continuous** stabiliser only exists for a template whose vertices are collinear, which for real decompositions
  means the coordination numbers one and two, about a fifth of blocks. Every vertex lies on the symmetry axis, so the
  rotation about that axis moves no vertex at all. A loss written on where the vertices end up therefore cannot see it
  and needs no special case. A loss written on the rotation matrix would see it, and would be wrong.
- A **finite** stabiliser permutes vertices between template slots. A loss on where the vertices end up does see that,
  because slot k now holds what used to be in slot pi(k), so it has to be minimised over the group. The group is small
  and depends only on the template, so ``stabiliser`` enumerates it once per block type and ``canonicalise`` picks the
  representative closest to whatever the model currently believes.

Handling both is what ``vertex_distance`` and ``canonicalise`` are for. Together they mean the regression target for an
orientation is a point again rather than a set.
"""

from typing import Optional
import numpy as np
import torch


SMALL_ANGLE = 1e-6
"""
Rotation angle in radians below which the series expansions are used instead of the closed forms.

Both the exponential and the logarithm divide by a sine of the angle, which is exact in the limit and catastrophic just
short of it. The cutoff is far above double-precision round-off and far below any angle a flow takes a real step
through, so no path ever notices which branch it is on.
"""

STABILISER_TOLERANCE = 0.1
"""
Largest root-mean-square deviation, in Angstrom, at which a rotation counts as mapping a template onto itself.

Templates are averages of real and slightly distorted polyhedra, so an octahedron's four-fold axis reproduces it to
within a few hundredths rather than exactly. The tolerance has to sit above that and below the separation between
genuinely different vertices, which is of the order of an Angstrom.
"""


def hat(vector: torch.Tensor) -> torch.Tensor:
    """
    Map a rotation vector to the skew-symmetric matrix that generates the same rotation.

    :param vector:
        Rotation vectors of shape (..., 3).
    :type vector: torch.Tensor

    :return:
        Skew-symmetric matrices of shape (..., 3, 3).
    :rtype: torch.Tensor
    """
    zero = torch.zeros_like(vector[..., 0])
    return torch.stack([
        torch.stack([zero, -vector[..., 2], vector[..., 1]], dim=-1),
        torch.stack([vector[..., 2], zero, -vector[..., 0]], dim=-1),
        torch.stack([-vector[..., 1], vector[..., 0], zero], dim=-1)], dim=-2)


def vee(matrix: torch.Tensor) -> torch.Tensor:
    """
    Map a skew-symmetric matrix back to its rotation vector, which is the inverse of ``hat``.

    The antisymmetric part is taken rather than the upper entries alone, so that a matrix carrying round-off is handled
    as the nearest skew-symmetric one instead of arbitrarily.

    :param matrix:
        Matrices of shape (..., 3, 3).
    :type matrix: torch.Tensor

    :return:
        Rotation vectors of shape (..., 3).
    :rtype: torch.Tensor
    """
    skew = 0.5 * (matrix - matrix.transpose(-1, -2))
    return torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)


def exp_map(vector: torch.Tensor) -> torch.Tensor:
    """
    Exponentiate a rotation vector into a rotation matrix by Rodrigues' formula.

    :param vector:
        Rotation vectors of shape (..., 3), whose norm is the angle in radians.
    :type vector: torch.Tensor

    :return:
        Rotation matrices of shape (..., 3, 3).
    :rtype: torch.Tensor
    """
    angle = torch.linalg.norm(vector, dim=-1, keepdim=True)
    small = angle < SMALL_ANGLE
    # The angle is clamped before dividing so that the zero-angle entries, whose result is discarded, do not produce a
    # not-a-number that would poison the gradient of every entry through the backward pass.
    safe = torch.where(small, torch.ones_like(angle), angle)
    sine = torch.where(small, 1.0 - angle * angle / 6.0, torch.sin(safe) / safe)
    cosine = torch.where(small, 0.5 - angle * angle / 24.0, (1.0 - torch.cos(safe)) / (safe * safe))

    generator = hat(vector)
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device).expand(generator.shape)
    return identity + sine[..., None] * generator + cosine[..., None] * (generator @ generator)


def log_map(rotation: torch.Tensor) -> torch.Tensor:
    """
    Take the logarithm of a rotation matrix, giving the rotation vector of smallest norm that produces it.

    :param rotation:
        Rotation matrices of shape (..., 3, 3).
    :type rotation: torch.Tensor

    :return:
        Rotation vectors of shape (..., 3), with norm in [0, pi].
    :rtype: torch.Tensor
    """
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    angle = torch.arccos(torch.clamp(0.5 * (trace - 1.0), -1.0, 1.0))

    small = angle < SMALL_ANGLE
    # Near pi the antisymmetric part vanishes and carries no information, so the axis is read off the symmetric part
    # instead. Both branches are always evaluated, so both have to be finite everywhere.
    near_half_turn = angle > np.pi - 1e-3
    safe = torch.where(small | near_half_turn, torch.ones_like(angle), angle)
    scale = torch.where(small, 0.5 + angle * angle / 12.0, 0.5 * safe / torch.sin(safe))
    from_antisymmetric = scale[..., None] * vee(rotation) * 2.0

    # At a half turn the rotation is 2 a a^T - I, so the outer product of the axis with itself is readable off the
    # symmetric part while the antisymmetric part has vanished and cannot supply it.
    symmetric = 0.5 * (rotation + rotation.transpose(-1, -2))
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device).expand(symmetric.shape)
    outer = 0.5 * (symmetric + identity)
    axis = torch.sqrt(torch.clamp(torch.diagonal(outer, dim1=-2, dim2=-1), min=0.0))

    # An outer product fixes the axis only up to an overall sign, and the relative signs come from its off-diagonal
    # entries against the largest component, whose own sign is the one free choice.
    dominant = torch.argmax(axis, dim=-1, keepdim=True)
    column = torch.take_along_dim(outer, dominant[..., None].expand(outer.shape[:-1] + (1,)), dim=-1)[..., 0]
    signs = torch.where(column < 0.0, -torch.ones_like(column), torch.ones_like(column))
    axis = axis * signs.scatter(-1, dominant, torch.ones_like(column))
    norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
    axis = axis / torch.where(norm > 0.0, norm, torch.ones_like(norm))

    # Just short of a half turn the antisymmetric part is small but its direction is still the right one, so it settles
    # the overall sign that the outer product cannot. At exactly a half turn it is zero and either sign is correct.
    turning = vee(rotation)
    agreement = torch.sum(turning * axis, dim=-1, keepdim=True)
    from_symmetric = angle[..., None] * torch.where(agreement < 0.0, -axis, axis)

    return torch.where(near_half_turn[..., None], from_symmetric, from_antisymmetric)


def project(matrix: torch.Tensor) -> torch.Tensor:
    """
    Return the closest proper rotation to an arbitrary matrix.

    An integrator that takes finite steps drifts off the group, slowly but without bound, and everything downstream
    assumes it is on it. Projecting after every step costs one small singular value decomposition and removes the whole
    question. Reflections are excluded because the mirror image of a chiral polyhedron is a different environment.

    :param matrix:
        Matrices of shape (..., 3, 3).
    :type matrix: torch.Tensor

    :return:
        Rotation matrices of shape (..., 3, 3).
    :rtype: torch.Tensor
    """
    left, _, right = torch.linalg.svd(matrix)
    determinant = torch.linalg.det(left @ right)
    unit = torch.ones_like(determinant)
    return left @ torch.diag_embed(torch.stack([unit, unit, torch.sign(determinant)], dim=-1)) @ right


def geodesic(start: torch.Tensor, end: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Interpolate along the shortest path on SO(3) from one orientation to another.

    :param start:
        Orientations at time zero, of shape (..., 3, 3).
    :type start: torch.Tensor
    :param end:
        Orientations at time one, of shape (..., 3, 3).
    :type end: torch.Tensor
    :param t:
        Times of shape (...,), broadcast against the orientations.
    :type t: torch.Tensor

    :return:
        Orientations at the given times, of shape (..., 3, 3).
    :rtype: torch.Tensor
    """
    return start @ exp_map(t[..., None] * log_map(start.transpose(-1, -2) @ end))


def velocity(current: torch.Tensor, end: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute the body-frame velocity that carries an orientation to its endpoint along the geodesic.

    This is the regression target of the flow, and it is constant along a geodesic, so the value returned at any time
    on a path is the value returned at any other.

    :param current:
        Orientations at the given times, of shape (..., 3, 3).
    :type current: torch.Tensor
    :param end:
        Orientations at time one, of shape (..., 3, 3).
    :type end: torch.Tensor
    :param t:
        Times of shape (...,), which must be strictly below one.
    :type t: torch.Tensor

    :return:
        Body-frame velocities of shape (..., 3), such that the derivative of the path is ``current @ hat(velocity)``.
    :rtype: torch.Tensor
    """
    return log_map(current.transpose(-1, -2) @ end) / (1.0 - t)[..., None]


def endpoint(current: torch.Tensor, body_velocity: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Recover the endpoint a body-frame velocity is pointing at, which inverts ``velocity``.

    :param current:
        Orientations at the given times, of shape (..., 3, 3).
    :type current: torch.Tensor
    :param body_velocity:
        Body-frame velocities of shape (..., 3).
    :type body_velocity: torch.Tensor
    :param t:
        Times of shape (...,).
    :type t: torch.Tensor

    :return:
        Orientations at time one, of shape (..., 3, 3).
    :rtype: torch.Tensor
    """
    return current @ exp_map((1.0 - t)[..., None] * body_velocity)


def angle_between(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """
    Compute the geodesic distance between two orientations, which is the angle of the rotation between them.

    :param first:
        Orientations of shape (..., 3, 3).
    :type first: torch.Tensor
    :param second:
        Orientations of shape (..., 3, 3).
    :type second: torch.Tensor

    :return:
        Angles in radians of shape (...,), in [0, pi].
    :rtype: torch.Tensor
    """
    product = first.transpose(-1, -2) @ second
    trace = product[..., 0, 0] + product[..., 1, 1] + product[..., 2, 2]
    return torch.arccos(torch.clamp(0.5 * (trace - 1.0), -1.0, 1.0))


def sample_uniform(count: int, generator: Optional[torch.Generator] = None, device: Optional[torch.device] = None,
                   dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Draw orientations uniformly from SO(3), which is the base distribution the flow starts from.

    Sampling is by way of unit quaternions, whose uniform distribution is a normalised four-dimensional Gaussian. Doing
    it by orthogonalising a Gaussian matrix instead would be biased unless the signs of the decomposition are corrected,
    which is a step easy to omit and hard to notice.

    :param count:
        Number of orientations.
    :type count: int
    :param generator:
        Random generator, for reproducibility.
        Defaults to None.
    :type generator: Optional[torch.Generator]
    :param device:
        Device to place the result on.
        Defaults to None.
    :type device: Optional[torch.device]
    :param dtype:
        Floating point type of the result.
        Defaults to None.
    :type dtype: Optional[torch.dtype]

    :return:
        Rotation matrices of shape (count, 3, 3).
    :rtype: torch.Tensor
    """
    quaternion = torch.randn((count, 4), generator=generator, device=device, dtype=dtype)
    quaternion = quaternion / torch.linalg.norm(quaternion, dim=-1, keepdim=True)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1)], dim=-2)


def stabiliser(offsets: np.ndarray, species: tuple[str, ...],
               tolerance: float = STABILISER_TOLERANCE, species_aware: bool = True) -> np.ndarray:
    """
    Enumerate every rotation that maps a template onto itself, which is the group its orientation is ambiguous under.

    Candidates are not searched over: a rotation preserving the template is determined by where it sends two vertices
    that are not collinear with the centre, so trying every such pair against every same-element, same-radius
    destination generates the whole group in time quadratic in the vertex count. Enumerating vertex permutations
    instead would work for a tetrahedron and be hopeless for a twelve-vertex polyhedron.

    A collinear template, meaning coordination number one or two, is stabilised by a continuous rotation about its own
    axis. That subgroup is not returned and does not need to be: it moves no vertex, so nothing downstream that measures
    where vertices end up can tell its elements apart.

    :param offsets:
        Template vertex offsets from the centre, of shape (n, 3).
    :type offsets: numpy.ndarray
    :param species:
        Element symbols of the vertices, aligned with the offsets.
    :type species: tuple[str, ...]
    :param tolerance:
        Largest root-mean-square deviation at which a rotation counts as mapping the template onto itself, in Angstrom.
        Defaults to STABILISER_TOLERANCE.
    :type tolerance: float
    :param species_aware:
        Whether a destination vertex must match the source vertex's element. Coarse ``(centre, CN)`` templates are
        species-blind, so this is False for them.
        Defaults to True.
    :type species_aware: bool

    :return:
        The distinct rotations, of shape (S, 3, 3), always including the identity as the first element.
    :rtype: numpy.ndarray
    """
    identity = np.eye(3)[None, :, :]
    if len(offsets) < 2:
        return identity

    radii = np.linalg.norm(offsets, axis=-1)
    if np.linalg.matrix_rank(offsets, tol=1e-6) < 2:
        return identity

    # The reference pair is the first vertex and the first vertex that is not parallel to it, which is what makes the
    # rotation built from a candidate destination pair unique.
    primary = 0
    secondary = next((index for index in range(1, len(offsets))
                      if np.linalg.norm(np.cross(offsets[primary], offsets[index])) > 1e-6 * radii[primary]), None)
    if secondary is None:
        return identity

    def frame(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Build an orthonormal frame from two vectors that are not parallel."""
        axis = first / np.linalg.norm(first)
        planar = second - np.dot(second, axis) * axis
        planar = planar / np.linalg.norm(planar)
        return np.stack([axis, planar, np.cross(axis, planar)], axis=-1)

    source = frame(offsets[primary], offsets[secondary])
    labels = np.array(species) if species_aware else np.array(["X"] * len(offsets))
    rotations = []
    for first in range(len(offsets)):
        if labels[first] != labels[primary] or abs(radii[first] - radii[primary]) > tolerance:
            continue
        for second in range(len(offsets)):
            if second == first or labels[second] != labels[secondary]:
                continue
            if abs(radii[second] - radii[secondary]) > tolerance:
                continue
            # A rotation preserves the separation of two vertices, so a destination pair whose separation differs from
            # the reference pair's cannot be the image of it. Screening on the separation rather than on the dot
            # product keeps this comparison in Angstrom, which is what the tolerance is in; a dot product is in
            # Angstrom squared and would tighten or loosen the screen with the size of the polyhedron.
            if abs(np.linalg.norm(offsets[first] - offsets[second])
                   - np.linalg.norm(offsets[primary] - offsets[secondary])) > tolerance:
                continue
            if np.linalg.norm(np.cross(offsets[first], offsets[second])) < 1e-6 * radii[first]:
                continue

            candidate = frame(offsets[first], offsets[second]) @ source.T
            rotated = offsets @ candidate.T
            distances = np.linalg.norm(rotated[:, None, :] - offsets[None, :, :], axis=-1)
            distances[labels[:, None] != labels[None, :]] = np.inf
            if np.max(np.min(distances, axis=-1)) < tolerance:
                rotations.append(candidate)

    if not rotations:
        return identity
    group = np.array(rotations)
    keep = [0]
    for index in range(1, len(group)):
        if all(np.linalg.norm(group[index] - group[other]) > 1e-3 for other in keep):
            keep.append(index)
    group = group[keep]
    # The identity is always in the group and is put first so that a caller that ignores the rest still gets a rotation
    # rather than an arbitrary symmetry of it.
    order = np.argsort(-np.trace(group, axis1=-2, axis2=-1))
    return group[order]


def canonicalise(target: torch.Tensor, reference: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    """
    Replace an orientation by the symmetry-equivalent one that is the shortest turn away from where the path starts.

    A polyhedron's orientation is only defined up to its stabiliser, so the regression target is a set of rotations
    rather than one. Choosing one element of that set makes it a point again, and choosing the nearest element makes the
    geodesic to it the shortest of the paths that all end at the same physical polyhedron.

    The reference must be the orientation the path *starts* from, not the orientation at the current time. Canonicalising
    against a moving reference would let the target jump between stabiliser elements partway along, and the constant body
    velocity that the geodesic is defined by, and that the loss regresses, would no longer exist. Canonicalising once
    against the sample at time zero keeps one target and one velocity for the whole path.

    :param target:
        Orientations at time one, of shape (B, 3, 3).
    :type target: torch.Tensor
    :param reference:
        Orientations to measure closeness against, of shape (B, 3, 3), which is the orientation at time zero.
    :type reference: torch.Tensor
    :param group:
        Stabiliser of each block's template, of shape (B, S, 3, 3), padded with repeats of the identity for blocks
        whose stabiliser is smaller than the largest in the batch.
    :type group: torch.Tensor

    :return:
        The chosen representative of each target, of shape (B, 3, 3).
    :rtype: torch.Tensor
    """
    candidates = target[:, None, :, :] @ group
    angles = angle_between(reference[:, None, :, :].expand(candidates.shape), candidates)
    return torch.take_along_dim(candidates, torch.argmin(angles, dim=1)[:, None, None, None], dim=1)[:, 0]


def vertex_distance(predicted: torch.Tensor, target: torch.Tensor, offsets: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """
    Measure how differently two linear maps act on a template's vertices, per block.

    Scoring in vertex space rather than on the maps themselves does two things. It puts the error in Angstrom, so it is
    commensurate with the translation loss and with the quantity the match rate is computed from. And it is blind to a
    continuous stabiliser for free, because motion about the axis of a collinear template moves none of its vertices, so
    the fifth of blocks with coordination number one or two contribute no spurious error. A finite stabiliser permutes
    vertices between slots and is not handled here; ``canonicalise`` deals with that before this is called.

    The two maps are body velocities during training and orientations when reporting how far apart two poses are. The
    training loss is on the velocity, not on the endpoint, and the distinction matters. On the geodesic the body
    velocity is constant, so an endpoint recovered from a velocity is ``R_t exp((1 - t) V)`` and its sensitivity to the
    velocity vanishes as t approaches one: an endpoint loss on a velocity output would therefore have to be reweighted
    by ``(1 - t)`` squared to train the late part of the path at all. Scoring the velocity directly is that reweighting,
    exactly rather than approximately, and needs no time-dependent factor. It is well defined because the metric is
    unchanged by the current orientation: the vertices move at ``R_t V p``, and ``R_t`` is orthogonal.

    :param predicted:
        Predicted body velocities, or predicted orientations, of shape (B, 3, 3).
    :type predicted: torch.Tensor
    :param target:
        Target body velocities, or target orientations, of shape (B, 3, 3).
    :type target: torch.Tensor
    :param offsets:
        Template vertex offsets of each block, of shape (B, V, 3), padded to the largest vertex count in the batch.
    :type offsets: torch.Tensor
    :param mask:
        Whether each slot holds a vertex, of shape (B, V).
    :type mask: torch.Tensor

    :return:
        Mean squared vertex discrepancy of each block, of shape (B,), in square Angstrom or in square Angstrom per unit
        time. A block with no vertices contributes zero, which is what a singleton should cost.
    :rtype: torch.Tensor
    """
    separation = offsets @ (predicted - target).transpose(-1, -2)
    squared = torch.sum(separation * separation, dim=-1) * mask
    counts = mask.sum(dim=-1)
    return squared.sum(dim=-1) / torch.where(counts > 0, counts, torch.ones_like(counts))
