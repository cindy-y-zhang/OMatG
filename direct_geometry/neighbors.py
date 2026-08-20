"""
Complete periodic neighbour lists for batched crystals.

WHY THIS EXISTS RATHER THAN A CALL INTO ``omg``

``omg.model.encoders.diffcsp_copies.radius_graph_pbc`` is the only periodic neighbour builder in the repository and it
cannot be used here, for two independent reasons that are both silent:

- ``max_rep`` is hardcoded to one, so only the twenty-seven images with shifts in ``{-1, 0, 1}`` are ever considered. Any
  cell shorter than the cutoff in some direction loses neighbours, and the loss is worst exactly where the physics is
  densest.
- ``radius_real`` is recomputed as ``min_dist.min(dim=-1) + 0.01`` and the ``radius`` argument is discarded. The effective
  cutoff is therefore the smallest interplanar spacing of each cell rather than the one the caller asked for, so it varies
  from structure to structure and no two structures are described on the same footing.

A descriptor built on either defect would be a descriptor of the bug. The builder below asks for exactly the images the
cutoff needs, uses the cutoff it was given, and refuses rather than truncates when a safety bound is reached.

WHAT "COMPLETE" MEANS

One edge per periodic image, not one edge per pair of atoms. In a rock-salt cell an atom's six octahedral neighbours are
three distinct sites appearing at plus and minus images, and in a caesium-chloride cell the eight nearest neighbours are a
single site appearing at eight. A graph carrying one edge per pair cannot represent either, which is why the fully
connected graph's coordination numbers were unreadable. The only edge excluded here is the zero-length self edge; a
nonzero image of an atom onto itself is a real neighbour and is kept.
"""

import math
from dataclasses import dataclass
from typing import Optional, Union
import torch


MIN_DISTANCE = 1.0e-3
"""
Shortest edge kept, in Angstrom.

Its purpose is to drop the self edge, whose length is exactly zero, without special-casing indices: an atom's own zero
image is the only edge that can be that short in a physical structure. It doubles as protection for the angular block,
which divides by the edge length to form a unit vector. A thousandth of an Angstrom is four orders of magnitude below any
bond, so nothing physical is lost, and two atoms closer than this are degenerate for every purpose this package has.
"""

MAX_IMAGES_PER_DIMENSION = 8
"""
Largest number of lattice repetitions requested along one axis before the builder gives up.

A cutoff of 6 Angstrom on a physical crystal needs one or two, and a flat or collapsed cell needs hundreds. This bound is
therefore a corruption detector rather than a tuning knob, and it raises rather than truncating, because a truncated
neighbour list produces plausible numbers that are quietly wrong.
"""

MAX_EDGES = 80_000_000
"""
Largest number of edges the builder will return.

Sized to be unreachable for the batches this project trains on -- MPTS-52 at batch 256 with a 6 Angstrom cutoff produces
of order a million -- so that hitting it means a degenerate cell has been handed in, and the run should stop rather than
begin swapping.
"""

SHIFT_CHUNK = 27
"""
Number of lattice images evaluated per pass.

The candidate tensor is ``pairs by images by three``, so evaluating every image at once is what would make a large batch on
a short cell run out of memory. Chunking bounds the peak at the cost of a short Python loop, and twenty-seven is the
natural unit: it is the whole grid in the common case where one repetition suffices.
"""


@dataclass(frozen=True)
class Neighbors:
    """
    A periodic neighbour list, one entry per directed edge.

    Every edge stands for one periodic image of one atom in the neighbourhood of one other atom, so an atom pair may appear
    several times with different images and an atom may appear as its own neighbour through a nonzero image.

    The list is symmetric by construction: the shift grid is symmetric about zero and every ordered pair is enumerated, so
    ``(i, j, n)`` is present exactly when ``(j, i, -n)`` is, with the opposite vector and the same length.

    :param center:
        Atom each edge belongs to, of shape ``(edges,)``. This is the aggregation target.
    :type center: torch.Tensor
    :param neighbor:
        Atom each edge points at, of shape ``(edges,)``, before the image shift is applied.
    :type neighbor: torch.Tensor
    :param image:
        Integer lattice shift of every edge, of shape ``(edges, 3)``.
    :type image: torch.Tensor
    :param offset:
        Fractional offset of every edge, of shape ``(edges, 3)``, equal to the neighbour's fractional coordinate plus the
        image shift minus the centre's. Not wrapped into the unit cell, because the image is the point.
    :type offset: torch.Tensor
    :param vector:
        Cartesian offset of every edge, of shape ``(edges, 3)``, pointing from the centre to the imaged neighbour.
    :type vector: torch.Tensor
    :param distance:
        Length of every edge, of shape ``(edges,)``, in Angstrom.
    :type distance: torch.Tensor
    :param edge2graph:
        Structure every edge belongs to, of shape ``(edges,)``.
    :type edge2graph: torch.Tensor
    """

    center: torch.Tensor
    neighbor: torch.Tensor
    image: torch.Tensor
    offset: torch.Tensor
    vector: torch.Tensor
    distance: torch.Tensor
    edge2graph: torch.Tensor

    def __len__(self) -> int:
        """
        Return the number of edges.

        :return:
            Number of edges.
        :rtype: int
        """
        return int(self.center.numel())

    def within(self, cutoff: Union[float, torch.Tensor]) -> "Neighbors":
        """
        Return the sub-list of edges no longer than a cutoff, which may differ per structure.

        Exists so that one neighbour list built at the largest cutoff any consumer needs can serve them all, rather than
        building the same images twice at two radii. The per-structure form is what lets the message graph hold its degree
        constant across a batch whose cells differ in size by a factor of two.

        :param cutoff:
            Largest edge length kept, in Angstrom: either one radius for the whole batch, or a tensor of shape
            ``(structures,)`` giving each structure its own.
        :type cutoff: Union[float, torch.Tensor]

        :raises ValueError:
            If a tensor of cutoffs does not have one entry per structure this list was built over.

        :return:
            The sub-list.
        :rtype: Neighbors
        """
        if isinstance(cutoff, torch.Tensor) and cutoff.dim() > 0:
            structures = int(self.edge2graph.max()) + 1 if len(self) else 0
            if cutoff.numel() < structures:
                raise ValueError(f"Got {cutoff.numel()} cutoffs for a list spanning at least {structures} structures.")
            keep = self.distance <= cutoff.to(self.distance.dtype)[self.edge2graph]
        else:
            keep = self.distance <= cutoff
        return Neighbors(center=self.center[keep], neighbor=self.neighbor[keep], image=self.image[keep],
                         offset=self.offset[keep], vector=self.vector[keep], distance=self.distance[keep],
                         edge2graph=self.edge2graph[keep])

    def degree(self, num_nodes: int) -> torch.Tensor:
        """
        Return the number of edges at every atom.

        Reported by the audit rather than used by the descriptors, because a degree histogram is the cheapest way to see
        that an image enumeration has gone wrong: a cell that lost its images shows a suspiciously flat one.

        :param num_nodes:
            Number of atoms in the batch.
        :type num_nodes: int

        :return:
            Number of edges at every atom, of shape ``(num_nodes,)``.
        :rtype: torch.Tensor
        """
        counts = torch.zeros(num_nodes, dtype=torch.long, device=self.center.device)
        return counts.index_add_(0, self.center, torch.ones_like(self.center))


def required_repetitions(cell: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Return the number of lattice repetitions needed along each axis to cover a cutoff.

    The bound is exact rather than heuristic. Write an edge as ``r = sum_k (d_k + n_k) a_k`` for lattice vectors ``a_k``,
    integer shifts ``n_k`` and a fractional offset ``d`` folded into ``[-1/2, 1/2)``. The reciprocal vectors ``b_i``, the
    columns of the inverted cell, satisfy ``a_k . b_i = delta_ki``, so ``r . b_i = d_i + n_i`` and hence

        ``|r| >= |r . b_i| / |b_i| = |d_i + n_i| / |b_i|``.

    An edge shorter than the cutoff therefore has ``|d_i + n_i| <= cutoff |b_i|``, and with ``|d_i| <= 1/2`` this gives
    ``|n_i| <= 1/2 + cutoff |b_i|``. Taking the floor is tight: the next shift out is provably too long.

    Folding the offset first is what keeps this cheap. Without it the bound carries ``|d_i| < 1`` and every axis needs one
    more repetition, which at the common one-repetition size is 125 images where 27 suffice.

    :param cell:
        Lattices of shape ``(structures, 3, 3)``, rows being the lattice vectors.
    :type cell: torch.Tensor
    :param cutoff:
        Largest edge length to be resolved, in Angstrom.
    :type cutoff: float

    :raises ValueError:
        If the cutoff is not positive, or if a cell is singular to working precision and has no well-defined images.

    :return:
        Repetitions of shape ``(structures, 3)``.
    :rtype: torch.Tensor
    """
    scalar = not isinstance(cutoff, torch.Tensor)
    if scalar and not cutoff > 0.0:
        raise ValueError(f"The cutoff must be positive, got {cutoff}.")
    if cell.dim() != 3 or cell.shape[-2:] != (3, 3):
        raise ValueError(f"Expected lattices of shape (structures, 3, 3), got {tuple(cell.shape)}.")
    if not scalar:
        if cutoff.shape != (cell.shape[0],):
            raise ValueError(f"Expected one cutoff per structure, of shape {(cell.shape[0],)}, got "
                             f"{tuple(cutoff.shape)}.")
        if not bool((cutoff > 0.0).all()):
            raise ValueError("Every per-structure cutoff must be positive.")

    with torch.no_grad():
        reciprocal_norm = reciprocal_norms(cell)
        reach = cutoff if scalar else cutoff.double().to(reciprocal_norm.device).unsqueeze(-1)
        repetitions = torch.floor(0.5 + reach * reciprocal_norm).long()

    largest = int(repetitions.max()) if repetitions.numel() else 0
    if largest > MAX_IMAGES_PER_DIMENSION:
        raise ValueError(
            f"Covering a cutoff of {cutoff} Angstrom needs {largest} lattice repetitions along one axis, over the "
            f"safety bound of {MAX_IMAGES_PER_DIMENSION}. Pass the cutoff through 'bounded_cutoff' first if the caller "
            f"can be handed arbitrary cells; a truncated neighbour list would be silently wrong.")
    return repetitions


def reciprocal_norms(cell: torch.Tensor) -> torch.Tensor:
    """
    Return the length of each reciprocal lattice vector, which is the reciprocal of an interplanar spacing.

    Factored out because it is what both the repetition bound and the cutoff bound are computed from, and a second
    inversion with a slightly different degeneracy test is a second thing to be wrong.

    :param cell:
        Lattices of shape ``(structures, 3, 3)``, rows being the lattice vectors.
    :type cell: torch.Tensor

    :raises ValueError:
        If a cell is singular to working precision and has no well-defined images.

    :return:
        Reciprocal vector lengths of shape ``(structures, 3)``, in inverse Angstrom, at double precision.
    :rtype: torch.Tensor
    """
    with torch.no_grad():
        determinant = torch.linalg.det(cell.double())
        # A cell whose volume has collapsed has no finite image count, and inverting it would return large numbers rather
        # than raising. Scaled by the cell's own size so that the test means "flat" rather than "small".
        scale = cell.double().abs().amax(dim=(-2, -1)).clamp(min=1.0e-12)
        if bool((determinant.abs() <= 1.0e-8 * scale ** 3).any()):
            raise ValueError("At least one lattice is singular to working precision, so its periodic images are not "
                             "defined. This is a corrupted or collapsed cell rather than a tolerance to widen.")
        # Column i of the inverse is the reciprocal vector dual to lattice vector i, so the norm taken over rows.
        return torch.linalg.inv(cell.double()).norm(dim=1)


MAX_DEGREE = 512.0
"""
Largest expected number of neighbours per atom a descriptor neighbourhood is allowed to reach.

Sized from measurement, not taste. Gate DG0 reports the descriptor's degree over the whole probability path: 53 edges per
atom on crystals rising to 158 at the noisy end, where the prior's smaller cells make structures about twice as dense. This
bound is over three times that worst case, so it never engages on anything the path visits with a physical cell, and it
cuts in only on cells no crystal could be -- a half-Angstrom cubic cell asks for 20,562 edges per atom, which at a batch of
256 is a hundred and thirty million edges and an out-of-memory error rather than a result.
"""


def bounded_cutoff(cell: torch.Tensor, num_atoms: torch.Tensor, cutoff: float, max_degree: float = MAX_DEGREE,
                   max_repetitions: int = MAX_IMAGES_PER_DIMENSION) -> torch.Tensor:
    """
    Return the requested cutoff, reduced per structure where the neighbourhood it asks for is unbounded in practice.

    Two things are bounded, because either alone is insufficient. Bounding the image count stops a *flat* cell from
    needing an unenumerable grid, but leaves a *small* cell producing tens of thousands of edges per atom inside a
    perfectly legal grid. Bounding the degree stops that, but says nothing about the shape of the grid needed to find
    them. A cell can be pathological in either way independently, so both bounds are applied and the smaller wins.

    WHY AN ARBITRARY CELL HAS TO BE TOLERATED

    Nothing here controls the cells this sees. Two thirds of them come from the sampler: the denoiser is evaluated on
    ``cell_t`` while generating, and generation integrates the model's own predicted cell velocity over 210 Euler steps
    from a prior draw. An undertrained model drives that trajectory wherever it likes, and it visits cells no crystal
    would have -- very flat, very skewed, very small. Measured on the real path this is rare, seven repetitions being the
    worst in 716,800 interpolated draws, but rare is not never: an overfit run reached epoch 1,000 and then died in
    validation on a generated cell that needed nine.

    The baseline never notices, because the fully connected graph does not enumerate images at all. So this is a hazard
    that exists only for the periodic arms, and one that would otherwise show up as those arms mysteriously failing to
    finish long runs.

    WHY REDUCING THE CUTOFF IS THE HONEST ANSWER

    The alternatives are worse. Raising the bound accepts an unbounded image count, which is an out-of-memory error
    instead of an exception. Truncating the neighbour list drops real neighbours silently and breaks the invariances the
    descriptor is built on. Skipping the structure is not available, since it is part of a batch that has to be scored.

    A reduced cutoff is exact for what it does report: every neighbour within the *returned* radius is still found, and
    the radial envelope beyond it goes to zero smoothly. It is also the only reading that means anything on such a cell,
    where a 6 Angstrom ball contains thousands of images of a handful of atoms and the phrase "local environment" has
    stopped referring to anything. What it costs is that a pathological structure's descriptor summarises a smaller
    neighbourhood than a normal one's, which is why callers are expected to report how often this engages rather than
    treat it as invisible.

    :param cell:
        Lattices of shape ``(structures, 3, 3)``, rows being the lattice vectors.
    :type cell: torch.Tensor
    :param num_atoms:
        Number of atoms of every structure, of shape ``(structures,)``, which with the volume gives the density the
        degree bound needs.
    :type num_atoms: torch.Tensor
    :param cutoff:
        Cutoff requested, in Angstrom.
    :type cutoff: float
    :param max_degree:
        Largest expected neighbours per atom to allow.
        Defaults to the module's bound.
    :type max_degree: float
    :param max_repetitions:
        Largest number of repetitions per axis to allow.
        Defaults to the module's safety bound.
    :type max_repetitions: int

    :raises ValueError:
        If the cutoff, degree or repetition bound is not positive, or if a cell is singular.

    :return:
        Per-structure cutoffs of shape ``(structures,)``, no larger than the requested one, in the cell's own dtype.
    :rtype: torch.Tensor
    """
    if not cutoff > 0.0:
        raise ValueError(f"The cutoff must be positive, got {cutoff}.")
    if max_repetitions < 1:
        raise ValueError(f"At least one repetition must be allowed, got {max_repetitions}. Zero would restrict the list "
                         f"to the home cell and silently stop being a periodic neighbour list.")
    reciprocal_norm = reciprocal_norms(cell)
    # Inverting floor(0.5 + c |b|) <= R gives the *strict* bound c < (R + 0.5) / |b|, so a margin is needed rather than
    # optional. It has to survive a float32 round trip: this is computed in double but returned in the caller's dtype,
    # and training runs in "32-true", so a bound sitting one part in 10^9 inside the boundary rounds back out to it and
    # the enumeration then asks for one repetition more than the bound it was reduced to respect. One part in 10^5 is
    # two orders of magnitude clear of float32 epsilon and costs a hundredth of a milliangstrom of reach.
    by_images = (max_repetitions + 0.5) * (1.0 - 1.0e-5) / reciprocal_norm.amax(dim=-1)
    # The same sphere-counting relation the message graph's radius is derived from, used here as a ceiling rather than as
    # a target, so there is one implementation of it and not two.
    by_degree = constant_degree_radius(cell, num_atoms, max_degree, cutoff).double()
    return torch.minimum(by_images, by_degree).to(cell.dtype)


def constant_degree_radius(cell: torch.Tensor, num_atoms: torch.Tensor, degree: float,
                           max_radius: float) -> torch.Tensor:
    """
    Return the per-structure radius at which a periodic neighbour list has a given expected degree.

    WHY THE RADIUS IS NOT A CONSTANT

    A sphere of radius ``r`` in a structure of number density ``rho = N / V`` holds ``(4 pi / 3) r^3 rho`` neighbours on
    average, so a *fixed* radius gives a degree proportional to the density -- and the denoiser's input density is not
    fixed. The baseline's base distribution for the cell is a lognormal fit whose draws are systematically smaller than
    MPTS-52's own cells, so early in the trajectory structures are about twice as dense as crystals. Measured on the
    baseline path, a fixed 5 Angstrom graph carries 30 edges per atom on data and 91 at the noisy end. That is bad three
    times over: the cost is set by the noisy end where the geometry is meaningless, peak memory is 2.5 times the fully
    connected control's and fails the cost gate, and the network is handed a qualitatively different graph at each time.

    Solving the same relation for ``r`` instead fixes the degree and lets the radius follow the structure::

        r = (3 * degree / (4 pi))^(1/3) * (V / N)^(1/3)

    On MPTS-52 this is about 5.2 Angstrom on crystals and 4.4 at the noisy end, and the measured degree is constant to
    within a few per cent across the whole trajectory.

    WHY THIS IS STILL A LEGITIMATE GRAPH

    ``V / N`` is invariant under translation, rotation, atom permutation and choice of unit cell, and unchanged by tiling
    the cell into a supercell, so every invariance the fixed-radius graph has is preserved -- the tests assert this
    directly. It is also the closer analogue of the baseline it is compared against: the fully connected control joins
    every pair within a cell, so *its* degree is the structure size and does not scale with density either. Matching the
    degree is what lets a difference between the two graphs be read as periodic multiplicity and true edge lengths rather
    than as one graph simply having more edges.

    :param cell:
        Lattices of shape ``(structures, 3, 3)``.
    :type cell: torch.Tensor
    :param num_atoms:
        Number of atoms of every structure, of shape ``(structures,)``.
    :type num_atoms: torch.Tensor
    :param degree:
        Expected number of neighbours per atom.
    :type degree: float
    :param max_radius:
        Largest radius returned, in Angstrom. Bounds the enumeration for a structure whose cell has been drawn very large,
        and keeps every edge inside the range the edge basis covers.
    :type max_radius: float

    :raises ValueError:
        If the degree or the bound is not positive, or if a cell has collapsed to zero volume.

    :return:
        Radii of shape ``(structures,)``.
    :rtype: torch.Tensor
    """
    if not degree > 0.0:
        raise ValueError(f"The target degree must be positive, got {degree}.")
    if not max_radius > 0.0:
        raise ValueError(f"The radius bound must be positive, got {max_radius}.")
    volume = torch.linalg.det(cell.double()).abs()
    if bool((volume <= 0.0).any()):
        raise ValueError("At least one lattice has zero volume, so its number density is not defined.")
    coefficient = (3.0 * degree / (4.0 * math.pi)) ** (1.0 / 3.0)
    radius = coefficient * (volume / num_atoms.to(volume.dtype)) ** (1.0 / 3.0)
    return radius.clamp(max=max_radius).to(cell.dtype)


def _ordered_pairs(num_atoms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Enumerate every ordered pair of atoms within each structure of a batch.

    Includes the diagonal, because an atom's own nonzero periodic images are real neighbours and dropping the pair would
    drop them with it.

    :param num_atoms:
        Number of atoms of every structure, of shape ``(structures,)``.
    :type num_atoms: torch.Tensor

    :return:
        Centre indices, neighbour indices, and the structure of every pair, each of shape ``(pairs,)``.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    device = num_atoms.device
    counts = num_atoms.long()
    pairs_per_structure = counts * counts
    node_offset = torch.cumsum(counts, dim=0) - counts
    pair_offset = torch.cumsum(pairs_per_structure, dim=0) - pairs_per_structure

    pair2graph = torch.repeat_interleave(torch.arange(counts.numel(), device=device), pairs_per_structure)
    local = (torch.arange(int(pairs_per_structure.sum()), device=device)
             - torch.repeat_interleave(pair_offset, pairs_per_structure))
    width = torch.repeat_interleave(counts, pairs_per_structure)
    base = torch.repeat_interleave(node_offset, pairs_per_structure)
    return local // width + base, local % width + base, pair2graph


def _shift_grid(repetitions: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Build the symmetric grid of integer lattice shifts for one repetition triple.

    :param repetitions:
        Repetitions along each axis, of shape ``(3,)``.
    :type repetitions: torch.Tensor
    :param device:
        Device to build the grid on.
    :type device: torch.device

    :return:
        Shifts of shape ``(images, 3)``.
    :rtype: torch.Tensor
    """
    axes = [torch.arange(-int(rep), int(rep) + 1, device=device, dtype=torch.long) for rep in repetitions]
    # Explicit indexing, because the default is deprecated and warns on every call, which buried a real warning once.
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)


def periodic_neighbors(frac_coords: torch.Tensor, cell: torch.Tensor, num_atoms: torch.Tensor, cutoff: float,
                       shift_chunk: int = SHIFT_CHUNK, max_edges: int = MAX_EDGES,
                       min_distance: float = MIN_DISTANCE) -> Neighbors:
    """
    Build the complete periodic neighbour list of a batch of crystals within a cutoff.

    Every periodic image inside the cutoff becomes its own edge. Structures are grouped by how many lattice repetitions
    their own cell needs, so a batch holding one short cell does not pay that cell's image count on every structure, and
    within a group the images are evaluated in chunks so that peak memory is set by the chunk rather than by the batch.

    :param frac_coords:
        Fractional coordinates of shape ``(atoms, 3)``. Need not be wrapped into the unit cell: offsets are folded before
        the images are enumerated, so any representative works.
    :type frac_coords: torch.Tensor
    :param cell:
        Lattices of shape ``(structures, 3, 3)``, rows being the lattice vectors.
    :type cell: torch.Tensor
    :param num_atoms:
        Number of atoms of every structure, of shape ``(structures,)``.
    :type num_atoms: torch.Tensor
    :param cutoff:
        Largest edge length kept, in Angstrom.
    :type cutoff: float
    :param shift_chunk:
        Number of lattice images evaluated per pass.
        Defaults to SHIFT_CHUNK.
    :type shift_chunk: int
    :param max_edges:
        Largest number of edges tolerated before the builder raises.
        Defaults to MAX_EDGES.
    :type max_edges: int
    :param min_distance:
        Shortest edge kept, in Angstrom, which excludes the zero-length self edge.
        Defaults to MIN_DISTANCE.
    :type min_distance: float

    :raises ValueError:
        If the inputs are inconsistently shaped, if the chunk size is not positive, if a cell is singular, if a cell needs
        more repetitions than the safety bound allows, or if the edge count exceeds the safety bound.

    :return:
        The neighbour list.
    :rtype: Neighbors
    """
    if frac_coords.dim() != 2 or frac_coords.shape[-1] != 3:
        raise ValueError(f"Expected fractional coordinates of shape (atoms, 3), got {tuple(frac_coords.shape)}.")
    if int(num_atoms.sum()) != frac_coords.shape[0]:
        raise ValueError(f"The atom counts sum to {int(num_atoms.sum())} but {frac_coords.shape[0]} coordinates were "
                         f"given.")
    if cell.shape[0] != num_atoms.numel():
        raise ValueError(f"Got {cell.shape[0]} lattices for {num_atoms.numel()} structures.")
    if not shift_chunk > 0:
        raise ValueError(f"The chunk size must be positive, got {shift_chunk}.")
    if frac_coords.dtype != cell.dtype:
        # Refused rather than promoted. This is exactly how the upstream builder misbehaves -- it constructs its image
        # offsets with a hardcoded ``torch.float`` and multiplies them by the caller's lattice -- and a silent promotion
        # here would hide the same class of mistake behind a plausible answer at reduced precision.
        raise ValueError(f"The coordinates are {frac_coords.dtype} and the lattices are {cell.dtype}. They describe one "
                         f"structure and must share a dtype; promoting one silently is how a precision bug becomes a "
                         f"geometry bug.")

    # Reduced per structure where the requested cutoff would need an unbounded number of images. A no-op for anything a
    # crystal could be; see ``bounded_cutoff`` for the generated cells that make it necessary.
    effective = bounded_cutoff(cell, num_atoms, cutoff)
    repetitions = required_repetitions(cell, effective)
    center_all, neighbor_all, pair2graph_all = _ordered_pairs(num_atoms)
    cutoff_squared = (effective * effective).to(frac_coords.dtype)
    min_squared = min_distance * min_distance
    collected: list[tuple[torch.Tensor, ...]] = []
    total_edges = 0

    # One pass per distinct repetition triple in the batch. Physical batches hold one or two, so this loop is short, and
    # grouping is what keeps a single collapsed cell from imposing its image count on every other structure.
    for triple in torch.unique(repetitions, dim=0):
        in_group = (repetitions == triple).all(dim=1)
        pair_in_group = in_group[pair2graph_all]
        if not bool(pair_in_group.any()):
            continue
        center = center_all[pair_in_group]
        neighbor = neighbor_all[pair_in_group]
        pair2graph = pair2graph_all[pair_in_group]

        raw_offset = frac_coords[neighbor] - frac_coords[center]
        # Folding into a half-open unit interval is what makes the repetition bound tight. The fold is undone in the
        # reported image, so that ``offset == frac[neighbor] + image - frac[center]`` holds exactly as documented.
        fold = torch.round(raw_offset)
        folded = raw_offset - fold
        pair_cell = cell[pair2graph]
        shifts = _shift_grid(triple, frac_coords.device)

        for start in range(0, shifts.shape[0], shift_chunk):
            block = shifts[start:start + shift_chunk].to(folded.dtype)
            candidate = folded.unsqueeze(1) + block.unsqueeze(0)
            vector = torch.einsum("pik,pkj->pij", candidate, pair_cell)
            squared = vector.pow(2).sum(dim=-1)
            keep = (squared <= cutoff_squared[pair2graph].unsqueeze(1)) & (squared > min_squared)
            if not bool(keep.any()):
                continue
            pair_index, image_index = keep.nonzero(as_tuple=True)
            kept_vector = vector[pair_index, image_index]
            collected.append((
                center[pair_index],
                neighbor[pair_index],
                block[image_index].long() - fold[pair_index].long(),
                candidate[pair_index, image_index],
                kept_vector,
                squared[pair_index, image_index].clamp(min=min_squared).sqrt(),
                pair2graph[pair_index],
            ))
            total_edges += int(kept_vector.shape[0])
            if total_edges > max_edges:
                raise ValueError(
                    f"The neighbour list passed {max_edges} edges at a cutoff of {cutoff} Angstrom. Nothing this "
                    f"project trains on is that dense, so this is a degenerate cell rather than a bound to raise.")

    if not collected:
        empty_long = torch.zeros(0, dtype=torch.long, device=frac_coords.device)
        empty_vector = torch.zeros((0, 3), dtype=frac_coords.dtype, device=frac_coords.device)
        return Neighbors(center=empty_long, neighbor=empty_long.clone(),
                         image=torch.zeros((0, 3), dtype=torch.long, device=frac_coords.device),
                         offset=empty_vector, vector=empty_vector.clone(),
                         distance=torch.zeros(0, dtype=frac_coords.dtype, device=frac_coords.device),
                         edge2graph=empty_long.clone())

    return Neighbors(*(torch.cat(parts, dim=0) for parts in zip(*collected)))


def fold_to_unit_cell(offset: torch.Tensor) -> torch.Tensor:
    """
    Wrap a fractional offset into ``[0, 1)``.

    Used only to reproduce the baseline fully connected graph's convention, which carries one edge per pair of atoms with
    the image discarded. Applying it to a periodic neighbour list would throw away the image multiplicity that list exists
    to provide.

    :param offset:
        Fractional offsets of shape ``(edges, 3)``.
    :type offset: torch.Tensor

    :return:
        Wrapped offsets of the same shape.
    :rtype: torch.Tensor
    """
    return offset % 1.0


def minimum_image_distance(offset: torch.Tensor, cell: torch.Tensor,
                           repetitions: Optional[int] = 1) -> torch.Tensor:
    """
    Return the shortest distance over periodic images of a fractional offset that carries no image of its own.

    Correct only for the fully connected graph, where an edge stands for a pair of atoms rather than for one image of one
    atom. On a periodic neighbour list the image is already in the offset, and minimising over images there would collapse
    the distinct neighbours the list was built to separate, which is the failure this package is arranged around.

    A fractional fold alone is not the minimum image: in a skewed cell the nearest image can be one the fold moves away
    from, so the neighbouring shifts are enumerated and the shortest kept.

    :param offset:
        Fractional offsets of shape ``(edges, 3)``.
    :type offset: torch.Tensor
    :param cell:
        Lattice of every edge's structure, of shape ``(edges, 3, 3)``.
    :type cell: torch.Tensor
    :param repetitions:
        Number of shifts searched along each axis beyond the fold.
        Defaults to 1.
    :type repetitions: Optional[int]

    :return:
        Distances of shape ``(edges,)``.
    :rtype: torch.Tensor
    """
    folded = offset - torch.round(offset)
    span = torch.arange(-int(repetitions), int(repetitions) + 1, device=offset.device, dtype=offset.dtype)
    shifts = torch.stack(torch.meshgrid(span, span, span, indexing="ij"), dim=-1).reshape(-1, 3)
    candidate = folded.unsqueeze(1) + shifts.unsqueeze(0)
    return torch.einsum("pik,pkj->pij", candidate, cell).norm(dim=-1).min(dim=1).values
