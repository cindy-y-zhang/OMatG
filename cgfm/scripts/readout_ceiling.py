"""
Measure what it costs to recover a structure from block poses without being told which blocks share which atoms.

``cgfm.scripts.block_ceiling`` measures the ceiling of the rigid-block parameterization while handing the reconstruction
the true sharing graph. That graph is the one component of the parameterization a generative model cannot read off
anything: it is a bipartite incidence between centres and anions carrying periodic image labels, and generating it has no
published precedent for inorganic crystals. The measurement here asks whether it has to be generated at all.

A placed block does not only claim an atom, it says where the atom is, so blocks that share an atom place a vertex at
nearly the same point and the graph is recoverable from the geometry. The composition says how many atoms of each
element there must be, so the recovery is a clustering with a known count rather than a guess. If that costs little, the
discrete half of the model shrinks to a coordination number per centre and the build becomes tractable; if it costs a
lot, the sharing graph has to be generated and the project is a much larger one.

Both readouts are scored on identical poses, from the same noise draw, on the same block decomposition, so the
difference between them is the readout and nothing else. Everything is scored at two matching tolerances: the strict one
the existing rigid-block report used, and the one ``omg.omg_lightning`` actually validates under, which is the tolerance
any comparison against the atomwise baseline will be made at.

Usage:

    python -m cgfm.scripts.readout_ceiling --data omg/data/mpts_52/val.lmdb \\
        --template-data omg/data/mpts_52/train.lmdb --limit 1500 --template-limit 8000 --workers 8
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional
import json
import pickle
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, Structure as PymatgenStructure
from tqdm import tqdm
from omg.datamodule import StructureDataset
from ..blocks import Decomposition, Template, centre_elements, decompose, fit_templates, lookup_template
from ..readout import (LINKAGE_METHOD, Readout, derive_templates, displacement, oracle_placement, orphan_free,
                       read_out, read_out_with_graph, read_out_with_oracle_clusters)


DEFAULT_POSE_NOISE = ((0.0, 0.0), (0.05, 2.0), (0.10, 5.0), (0.20, 10.0), (0.40, 20.0))
"""Translation standard deviation in Angstrom and rotation standard deviation in degrees of the sensitivity curve."""

TOLERANCES = (("strict", 0.2, 0.3, 5.0), ("comparison", 0.3, 0.5, 10.0))
"""
Structure matching tolerances the reconstructions are scored under.

The strict set is the one the rigid-block ceiling report used. The comparison set is the one
``omg.omg_lightning.OMGLightning.on_validation_epoch_end`` computes the validation match rate with, and is therefore the
set any claim against the atomwise baseline has to be made at.
"""

READOUTS = ("graph", "oracle", "free", "bound")
"""
The four readouts compared, in decreasing order of how much they are told.

"graph" is handed the true sharing graph and is the ceiling of the parameterisation. "oracle" is handed only the answer
to the grouping question, by letting each vote join the true atom nearest it, so it is the ceiling of the clustering.
"free" recovers the grouping from the geometry and is what a sampler can actually do. "bound" additionally forbids one
block from predicting the same atom twice, which sounds like free information and measurably is not; it is reported so
that the finding stays visible rather than living in a comment.

The two middle columns are the point of the table. The drop from "graph" to "oracle" is what the votes themselves cost,
and the drop from "oracle" to "free" is what the clustering costs. Without them a shortfall of zero was being read as
evidence that the clustering was the only problem, which it never established.
"""


_FINE: dict[tuple, Template] = {}
"""Per-worker templates keyed by centre, coordination number and vertex composition."""

_COARSE: dict[tuple, Template] = {}
"""Per-worker templates keyed by centre and coordination number only."""

_POSE_NOISE: tuple[tuple[float, float], ...] = DEFAULT_POSE_NOISE
"""Per-worker pose noise levels."""

_LINKAGE: str = LINKAGE_METHOD
"""Per-worker agglomerative linkage rule."""

_TYPE_KEY_MODE: str = "centre-cn-ligands"
"""Per-worker block type mode."""

_DATASET: Optional[StructureDataset] = None
"""Per-worker dataset handle, since LMDB environments cannot be shared across processes."""

G1_TRANSLATION = 0.10
"""Pose translation noise, in Angstrom, at which G1 is scored."""

G1_ROTATION = 5.0
"""Pose rotation noise, in degrees, at which G1 is scored."""

G1_FREE_MATCH_RATE = 0.90
"""Minimum free-readout match rate at the comparison tolerance for G1 to pass."""

G2_TOLERANCE_POINTS = 3.0
"""Largest allowed gap, in percentage points, between capped coarse and fine free readout for G2 to pass."""


def _initialise_decomposer(file_path: str, type_key_mode: str) -> None:
    """
    Open the dataset inside a worker process for the decomposition pass.

    :param file_path:
        Path of the dataset split.
    :type file_path: str
    :param type_key_mode:
        How block types are defined.
    :type type_key_mode: str
    """
    global _DATASET, _TYPE_KEY_MODE
    _DATASET = StructureDataset(file_path=file_path, lazy_storage=True, niggli_reduce=False,
                                convert_to_fractional=True, floating_point_precision="64-true")
    _TYPE_KEY_MODE = type_key_mode


def _initialise_scorer(fine: dict[tuple, Template], coarse: dict[tuple, Template],
                       pose_noise: tuple[tuple[float, float], ...], linkage_method: str) -> None:
    """
    Install the shared configuration of the scoring pass inside a worker process.

    :param fine:
        Templates keyed by centre, coordination number and vertex composition.
    :type fine: dict[tuple, Template]
    :param coarse:
        Templates keyed by centre and coordination number only.
    :type coarse: dict[tuple, Template]
    :param pose_noise:
        Pose noise levels of the sensitivity curve.
    :type pose_noise: tuple[tuple[float, float], ...]
    :param linkage_method:
        Agglomerative linkage rule used to group vertex predictions into atoms.
    :type linkage_method: str
    """
    global _FINE, _COARSE, _POSE_NOISE, _LINKAGE
    _FINE, _COARSE, _POSE_NOISE, _LINKAGE = fine, coarse, pose_noise, linkage_method


def _decompose_structure(index: int) -> Optional[Decomposition]:
    """
    Decompose one structure of the split and repair it so its blocks match its centre atoms one for one.

    :param index:
        Index of the structure within the split.
    :type index: int

    :return:
        The decomposition, or None if the neighbour analysis failed.
    :rtype: Optional[Decomposition]
    """
    assert _DATASET is not None
    structure = _DATASET[index]
    identifier = str(structure.metadata.get("identifier", index))
    decomposition = decompose(structure.get_pymatgen_structure(), identifier=identifier,
                              type_key_mode=_TYPE_KEY_MODE)
    return None if decomposition is None else orphan_free(decomposition, type_key_mode=_TYPE_KEY_MODE)


def _as_pymatgen(lattice: np.ndarray, numbers: np.ndarray, coords: np.ndarray) -> PymatgenStructure:
    """
    Build a pymatgen structure from atomic numbers and Cartesian coordinates on a lattice.

    :param lattice:
        Cell vectors of shape (3, 3).
    :type lattice: numpy.ndarray
    :param numbers:
        Atomic numbers of shape (N,).
    :type numbers: numpy.ndarray
    :param coords:
        Cartesian coordinates of shape (N, 3).
    :type coords: numpy.ndarray

    :return:
        The structure.
    :rtype: pymatgen.core.Structure
    """
    return PymatgenStructure(lattice, [Element.from_Z(int(number)) for number in numbers], coords,
                             coords_are_cartesian=True)


def _score_structure(task: tuple[int, Decomposition]) -> dict:
    """
    Rebuild one structure at every pose noise level by both readouts and match every rebuild against the original.

    :param task:
        Index of the structure within the split, used to seed the pose noise, and its decomposition.
    :type task: tuple[int, Decomposition]

    :return:
        Match outcome, displacement and readout diagnostics of every noise level and every readout.
    :rtype: dict
    """
    index, decomposition = task
    matchers = {name: StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
                for name, ltol, stol, angle_tol in TOLERANCES}
    original = _as_pymatgen(decomposition.lattice, decomposition.numbers, decomposition.coords)
    templates = derive_templates(decomposition.numbers, _FINE, _COARSE)
    symbols = tuple(Element.from_Z(int(number)).symbol for number in decomposition.numbers)
    centres, _ = centre_elements(symbols)
    single_anion = len(set(symbols) - set(centres)) <= 1

    matched = {(name, readout): [] for name, *_ in TOLERANCES for readout in READOUTS}
    displacements = {readout: [] for readout in READOUTS}
    votes, spread, shortfall, uncovered = [], [], 0, 0

    for level, (translation_sigma, rotation_sigma) in enumerate(_POSE_NOISE):
        rng = np.random.default_rng(1_000_003 * index + level)
        placement, correspondences = oracle_placement(decomposition, templates, translation_sigma=translation_sigma,
                                                      rotation_sigma=rotation_sigma, rng=rng)
        rebuilt: dict[str, Readout] = {
            "graph": read_out_with_graph(placement, decomposition, templates, correspondences),
            "oracle": read_out_with_oracle_clusters(placement, decomposition, templates),
            "free": read_out(placement, templates, decomposition.numbers, method=_LINKAGE, refine=False),
            "bound": read_out(placement, templates, decomposition.numbers, method=_LINKAGE, refine=True)}

        for readout, result in rebuilt.items():
            displacements[readout].append(displacement(result, decomposition))
            structure = _as_pymatgen(decomposition.lattice, result.numbers, result.coords)
            for name, *_ in TOLERANCES:
                matched[(name, readout)].append(matchers[name].get_rms_dist(original, structure) is not None)
        votes.append(rebuilt["free"].votes_per_atom)
        spread.append(rebuilt["free"].cluster_spread)
        shortfall += rebuilt["free"].shortfall
        uncovered += rebuilt["oracle"].shortfall

    return {"matched": matched, "displacements": displacements, "votes": votes[0], "spread": spread[0],
            "shortfall": shortfall, "uncovered": uncovered, "single_anion": single_anion,
            "unknowns": sum(1 for number in decomposition.numbers) - len(placement.centre_numbers)}


def _run_decomposition(file_path: str, count: int, workers: int, type_key_mode: str,
                       description: str) -> tuple[list[Decomposition], int]:
    """
    Decompose a whole split in parallel.

    :param file_path:
        Path of the dataset split.
    :type file_path: str
    :param count:
        Number of structures to process from the start of the split.
    :type count: int
    :param workers:
        Number of worker processes.
    :type workers: int
    :param type_key_mode:
        How block types are defined.
    :type type_key_mode: str
    :param description:
        Label of the progress bar.
    :type description: str

    :return:
        The successful decompositions and the number of structures whose neighbour analysis failed.
    :rtype: tuple[list[Decomposition], int]
    """
    decompositions, failures = [], 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialise_decomposer,
                             initargs=(file_path, type_key_mode)) as executor:
        for decomposition in tqdm(executor.map(_decompose_structure, range(count), chunksize=16),
                                  total=count, desc=description, unit=" structures"):
            if decomposition is None:
                failures += 1
            else:
                decompositions.append(decomposition)
    return decompositions, failures


def _collect_instances(decompositions: list[Decomposition],
                       fine: bool) -> dict[tuple, list[tuple[np.ndarray, tuple[str, ...]]]]:
    """
    Group every observed polyhedron by its type, ready for template fitting.

    Keys are rebuilt from the block rather than taken from its type key, so that both granularities come out of a
    single decomposition pass and the fine table is available even though the blocks were typed coarsely.

    :param decompositions:
        Decompositions of the split the templates are fitted on.
    :type decompositions: list[Decomposition]
    :param fine:
        If True, types additionally carry the sorted vertex composition.
    :type fine: bool

    :return:
        Vertex offsets and vertex species of every block, grouped by type.
    :rtype: dict[tuple, list[tuple[numpy.ndarray, tuple[str, ...]]]]
    """
    instances = defaultdict(list)
    for decomposition in decompositions:
        for block in decomposition.blocks:
            key = (block.type_key[0], len(block.species))
            instances[key + (tuple(sorted(block.species)),) if fine else key].append((block.offsets, block.species))
    return instances


def _resolve_templates(evaluation: list[Decomposition], fine: dict[tuple, Template],
                       coarse: dict[tuple, Template]) -> tuple[dict[tuple, Template], Counter]:
    """
    Report how every block of the evaluation split gets its template under the composition-derived scheme.

    This does not choose the templates the scoring uses, which are derived per structure inside the workers. It exists
    to say how often the composition was enough to determine the sharper key, since that fraction is what the sharper
    key can possibly be worth.

    :param evaluation:
        Decompositions of the split being scored.
    :type evaluation: list[Decomposition]
    :param fine:
        Templates keyed by centre, coordination number and vertex composition.
    :type fine: dict[tuple, Template]
    :param coarse:
        Templates keyed by centre and coordination number only.
    :type coarse: dict[tuple, Template]

    :return:
        Every template that is actually used somewhere, and a count of how each block was resolved.
    :rtype: tuple[dict[tuple, Template], collections.Counter]
    """
    used, provenance = {}, Counter()
    for decomposition in evaluation:
        derived = derive_templates(decomposition.numbers, fine, coarse)
        symbols = tuple(Element.from_Z(int(number)).symbol for number in decomposition.numbers)
        centres, _ = centre_elements(symbols)
        determined = len(set(symbols) - set(centres)) <= 1
        for block in decomposition.blocks:
            key = (block.type_key[0], len(block.species))
            if len(block.ligands) == 0:
                provenance["singleton block"] += 1
                continue
            template, label = lookup_template(key, derived)
            used[key] = template
            sharp = key + (tuple(sorted(block.species)),) in fine
            if determined and sharp and label == "exact":
                provenance["composition-derived vertex composition"] += 1
            else:
                provenance[f"coarse lookup: {label}"] += 1
    return used, provenance


def _report_decomposition(decompositions: list[Decomposition], failures: int, count: int) -> None:
    """
    Print the statistics of the repaired block decomposition of the evaluation split.

    :param decompositions:
        The successful decompositions, after repair.
    :type decompositions: list[Decomposition]
    :param failures:
        Number of structures whose neighbour analysis failed.
    :type failures: int
    :param count:
        Number of structures attempted.
    :type count: int
    """
    atoms = np.array([len(d.coords) for d in decompositions])
    blocks = np.array([len(d.blocks) for d in decompositions])
    polyhedra = np.array([sum(1 for b in d.blocks if len(b.ligands) > 0) for d in decompositions])
    vertices = np.array([len(b.ligands) for d in decompositions for b in d.blocks if len(b.ligands) > 0])
    memberships = np.concatenate([np.bincount(np.concatenate([np.array([b.centre for b in d.blocks]),
                                                              *(b.ligands for b in d.blocks)]),
                                              minlength=len(d.coords)) for d in decompositions])
    anion_elements = []
    for decomposition in decompositions:
        symbols = tuple(Element.from_Z(int(number)).symbol for number in decomposition.numbers)
        centres, _ = centre_elements(symbols)
        anion_elements.append(len(set(symbols) - set(centres)))
    anions = np.array(anion_elements)
    atomwise_dof = 3 * atoms.sum() + 9 * len(decompositions)
    block_dof = 6 * polyhedra.sum() + 3 * (blocks.sum() - polyhedra.sum()) + 9 * len(decompositions)

    print(f"\nDecomposition of {len(decompositions)} structures, {failures} of {count} failed CrystalNN.")
    print(f"  atoms per structure               {atoms.mean():.2f}")
    print(f"  blocks per structure              {blocks.mean():.2f}, one per centre atom by construction")
    print(f"  of which hold no vertices         {(blocks - polyhedra).mean():.3f}")
    print(f"  vertices per polyhedron           {vertices.mean():.2f} (max {vertices.max()})")
    print(f"  blocks claiming each atom         {memberships.mean():.2f} "
          f"({100.0 * (memberships > 1).mean():.1f} per cent of atoms shared)")
    print(f"  degrees of freedom                {block_dof} against {atomwise_dof} atomwise, "
          f"a factor {atomwise_dof / block_dof:.2f}")
    print(f"  non-centre elements per structure {anions.mean():.2f} "
          f"({100.0 * (anions <= 1).mean():.1f} per cent hold at most one)")


def _report_templates(templates: dict[tuple, Template], provenance: Counter) -> None:
    """
    Print the statistics of the fitted templates, whose spread is the rigidity of a block type.

    :param templates:
        Template of every block type present in the evaluation split.
    :type templates: dict[tuple, Template]
    :param provenance:
        Count of how each block's template was resolved.
    :type provenance: collections.Counter
    """
    spreads = np.array([template.spread for template in templates.values()])
    counts = np.array([template.count for template in templates.values()])

    print(f"\nTemplates: {len(templates)} types used, fitted on the training split.")
    print(f"  rigidity, instance-weighted mean deviation from the type template   "
          f"{float(np.sum(spreads * counts) / np.sum(counts)):.3f} Angstrom")
    print(f"  rigidity, unweighted median over types                             {np.median(spreads):.3f} Angstrom")
    total = sum(provenance.values())
    for label, number in provenance.most_common():
        print(f"  blocks resolved by {label:<30} {number} ({100.0 * number / total:.1f} per cent)")


def _report_ceiling(results: list[dict], pose_noise: tuple[tuple[float, float], ...]) -> None:
    """
    Print the match rate of both readouts against pose precision, which is the comparison the whole script exists for.

    :param results:
        Per-structure outcomes.
    :type results: list[dict]
    :param pose_noise:
        The noise levels, as translation standard deviation in Angstrom and rotation standard deviation in degrees.
    :type pose_noise: tuple[tuple[float, float], ...]
    """
    print("\nReadout comparison. 'graph' is given the true sharing graph and is the ceiling of the parameterisation;")
    print("  'oracle' is given only the grouping, so it is the ceiling of the clustering; 'free' recovers the grouping")
    print("  itself; 'bound' adds the one-vote-per-block constraint, which costs rather than helps.")
    print("  Match rate in per cent at each matching tolerance, and mean per-atom displacement in Angstrom.\n")
    header = f"  {'translation':>11} {'rotation':>9} |"
    for name, ltol, stol, angle_tol in TOLERANCES:
        header += f" {f'{name} {ltol}/{stol}/{angle_tol}':>25} |"
    print(header + f" {'displacement':>25}")
    subheader = f"  {'(Angstrom)':>11} {'(degrees)':>9} |"
    for _ in TOLERANCES:
        subheader += "".join(f" {readout:>7}" for readout in READOUTS) + " |"
    print(subheader + "".join(f" {readout:>8}" for readout in READOUTS))

    for level, (translation_sigma, rotation_sigma) in enumerate(pose_noise):
        row = f"  {translation_sigma:>11.2f} {rotation_sigma:>9.1f} |"
        for name, *_ in TOLERANCES:
            for readout in READOUTS:
                row += f" {100.0 * float(np.mean([r['matched'][(name, readout)][level] for r in results])):>7.2f}"
            row += " |"
        for readout in READOUTS:
            row += f" {float(np.mean([result['displacements'][readout][level] for result in results])):>8.4f}"
        print(row)

    strata = (("composition fixes the vertex composition", [r for r in results if r["single_anion"]]),
              ("it does not, so templates stay coarse    ", [r for r in results if not r["single_anion"]]))
    print("\n  Stratified by whether the composition determines each block's vertex composition, at the comparison")
    print("  tolerance and the 0.10 Angstrom / 5 degree pose.\n")
    level = _pose_level(pose_noise, G1_TRANSLATION, G1_ROTATION)
    for label, subset in strata:
        if not subset:
            continue
        rates = "".join(f" {100.0 * float(np.mean([r['matched'][('comparison', readout)][level] for r in subset])):>7.2f}"
                        for readout in READOUTS)
        print(f"  {label} {len(subset):>5} structures |{rates}")

    print(f"\n  vertex predictions averaged into each non-centre atom, at oracle poses  "
          f"{float(np.mean([result['votes'] for result in results])):.2f}")
    print(f"  disagreement between them, at oracle poses                              "
          f"{float(np.mean([result['spread'] for result in results])):.4f} Angstrom")
    print(f"  votes short of the atom count, summed over all noise levels             "
          f"{sum(result['shortfall'] for result in results)}")
    unknowns = sum(result["unknowns"] for result in results) * len(pose_noise)
    print(f"  atoms that no vote landed nearest to, summed over all noise levels      "
          f"{sum(result['uncovered'] for result in results)} of {unknowns} "
          f"({100.0 * (1.0 - sum(r['uncovered'] for r in results) / max(unknowns, 1)):.2f} per cent covered)")


def _parse_pose_noise(specification: str) -> tuple[tuple[float, float], ...]:
    """
    Parse the pose noise levels of the sensitivity curve.

    :param specification:
        Comma-separated levels, each a translation standard deviation in Angstrom and a rotation standard deviation in
        degrees joined by a colon, for example "0:0,0.1:5".
    :type specification: str

    :return:
        The noise levels.
    :rtype: tuple[tuple[float, float], ...]

    :raises ValueError:
        If a level is not a colon-separated pair of non-negative numbers.
    """
    levels = []
    for entry in specification.split(","):
        parts = entry.split(":")
        if len(parts) != 2:
            raise ValueError(f"Pose noise level {entry!r} is not a translation and a rotation joined by a colon.")
        translation, rotation = float(parts[0]), float(parts[1])
        if translation < 0.0 or rotation < 0.0:
            raise ValueError(f"Pose noise level {entry!r} is negative.")
        levels.append((translation, rotation))
    return tuple(levels)


def _pose_level(pose_noise: tuple[tuple[float, float], ...], translation: float, rotation: float) -> int:
    """
    Return the index of a pose-noise level, or the nearest one if the exact pair is absent.

    :param pose_noise:
        The noise levels.
    :type pose_noise: tuple[tuple[float, float], ...]
    :param translation:
        Requested translation standard deviation in Angstrom.
    :type translation: float
    :param rotation:
        Requested rotation standard deviation in degrees.
    :type rotation: float

    :return:
        Index into ``pose_noise``.
    :rtype: int
    """
    for index, (sigma_t, sigma_r) in enumerate(pose_noise):
        if abs(sigma_t - translation) < 1.0e-9 and abs(sigma_r - rotation) < 1.0e-9:
            return index
    distances = [abs(sigma_t - translation) + abs(sigma_r - rotation) for sigma_t, sigma_r in pose_noise]
    return int(np.argmin(distances))


def _match_rate(results: list[dict], tolerance: str, readout: str, level: int) -> float:
    """
    Mean match rate of one readout at one tolerance and pose-noise level.

    :param results:
        Per-structure outcomes.
    :type results: list[dict]
    :param tolerance:
        Name of the matching tolerance set.
    :type tolerance: str
    :param readout:
        Name of the readout.
    :type readout: str
    :param level:
        Index of the pose-noise level.
    :type level: int

    :return:
        Match rate in [0, 1].
    :rtype: float
    """
    return float(np.mean([result["matched"][(tolerance, readout)][level] for result in results]))


def _cache_path(cache_dir: Path, split: str, type_key: str, kind: str) -> Path:
    """
    Return the cache path of one artifact.

    :param cache_dir:
        Directory holding cached decompositions and templates.
    :type cache_dir: Path
    :param split:
        Split name, for example "train" or "val".
    :type split: str
    :param type_key:
        Block type mode.
    :type type_key: str
    :param kind:
        Artifact kind, for example "decompositions" or "templates".
    :type kind: str

    :return:
        The path.
    :rtype: Path
    """
    return cache_dir / f"{split}.{type_key}.{kind}.pkl"


def _load_or_decompose(file_path: str, count: int, workers: int, type_key: str, description: str,
                       cache_path: Optional[Path]) -> tuple[list[Decomposition], int]:
    """
    Decompose a split, reading from a pickle cache when one exists.

    :param file_path:
        Path of the dataset split.
    :type file_path: str
    :param count:
        Number of structures to process from the start of the split.
    :type count: int
    :param workers:
        Number of worker processes.
    :type workers: int
    :param type_key:
        How block types are defined.
    :type type_key: str
    :param description:
        Label of the progress bar.
    :type description: str
    :param cache_path:
        Path of the pickle cache, or None to always recompute.
    :type cache_path: Optional[Path]

    :return:
        The successful decompositions and the number of structures whose neighbour analysis failed.
    :rtype: tuple[list[Decomposition], int]
    """
    if cache_path is not None and cache_path.exists():
        payload = pickle.loads(cache_path.read_bytes())
        if payload.get("file_path") == file_path and payload.get("count") == count and payload.get("type_key") == type_key:
            print(f"Loaded {len(payload['decompositions'])} decompositions from {cache_path}.")
            return payload["decompositions"], payload["failures"]
    decompositions, failures = _run_decomposition(file_path, count, workers, type_key, description)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps({
            "file_path": file_path, "count": count, "type_key": type_key,
            "decompositions": decompositions, "failures": failures}))
    return decompositions, failures


def _summarise(results: list[dict], pose_noise: tuple[tuple[float, float], ...], type_key: str,
               n_eval: int, failures: int, provenance: Counter) -> dict:
    """
    Build the compact machine-readable summary of a ceiling run.

    :param results:
        Per-structure outcomes.
    :type results: list[dict]
    :param pose_noise:
        Pose noise levels.
    :type pose_noise: tuple[tuple[float, float], ...]
    :param type_key:
        Block type mode scored.
    :type type_key: str
    :param n_eval:
        Number of evaluation structures attempted.
    :type n_eval: int
    :param failures:
        Number of structures whose neighbour analysis failed.
    :type failures: int
    :param provenance:
        Count of how each block's template was resolved.
    :type provenance: collections.Counter

    :return:
        JSON-serialisable summary, including the G1 statistic.
    :rtype: dict
    """
    level = _pose_level(pose_noise, G1_TRANSLATION, G1_ROTATION)
    free_rate = _match_rate(results, "comparison", "free", level)
    summary = {
        "type_key": type_key,
        "n_eval": n_eval,
        "n_scored": len(results),
        "failures": failures,
        "g1_pose": {"translation": G1_TRANSLATION, "rotation": G1_ROTATION},
        "g1": {
            "free_match_rate_comparison": free_rate,
            "threshold": G1_FREE_MATCH_RATE,
            "passed": bool(free_rate >= G1_FREE_MATCH_RATE),
        },
        "rates": {},
        "provenance": dict(provenance),
        "shortfall": int(sum(result["shortfall"] for result in results)),
        "uncovered": int(sum(result["uncovered"] for result in results)),
    }
    for index, (translation_sigma, rotation_sigma) in enumerate(pose_noise):
        key = f"{translation_sigma:.2f}A_{rotation_sigma:.1f}deg"
        summary["rates"][key] = {
            f"{tolerance}_{readout}": _match_rate(results, tolerance, readout, index)
            for tolerance, *_ in TOLERANCES
            for readout in READOUTS
        }
    return summary


def main() -> None:
    """Measure the cost of recovering a structure from block poses without the sharing graph, and print the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="split to score, for example omg/data/mpts_52/val.lmdb")
    parser.add_argument("--template-data", required=True, help="split to fit templates on, normally the training split")
    parser.add_argument("--limit", type=int, default=1500, help="number of structures to score")
    parser.add_argument("--template-limit", type=int, default=8000,
                        help="number of structures to fit templates from; templates converge well before this")
    parser.add_argument("--workers", type=int, default=8, help="number of worker processes")
    parser.add_argument("--type-key", choices=("centre-cn", "centre-cn-ligands"), default="centre-cn",
                        help="what a rigid template is shared across")
    parser.add_argument("--linkage", default=LINKAGE_METHOD,
                        help="agglomerative linkage rule used to group vertex predictions into atoms")
    parser.add_argument("--pose-noise", default=",".join(f"{t}:{r}" for t, r in DEFAULT_POSE_NOISE),
                        help="pose noise levels as translation:rotation pairs in Angstrom and degrees")
    parser.add_argument("--cache-dir", default=None,
                        help="directory holding cached decompositions and templates; created if missing")
    parser.add_argument("--json-out", default=None, help="write a compact machine-readable summary to this path")
    arguments = parser.parse_args()
    pose_noise = _parse_pose_noise(arguments.pose_noise)
    cache_dir = Path(arguments.cache_dir) if arguments.cache_dir else None

    def split_size(path: str, limit: int) -> int:
        probe = StructureDataset(file_path=path, lazy_storage=True, niggli_reduce=False, convert_to_fractional=True,
                                 floating_point_precision="64-true")
        size = min(limit, len(probe))
        del probe
        return size

    def cache_for(path: str, kind: str) -> Optional[Path]:
        if cache_dir is None:
            return None
        return _cache_path(cache_dir, Path(path).stem, arguments.type_key, kind)

    training, _ = _load_or_decompose(arguments.template_data,
                                     split_size(arguments.template_data, arguments.template_limit),
                                     arguments.workers, arguments.type_key, "Decomposing train",
                                     cache_for(arguments.template_data, "decompositions"))
    evaluation_count = split_size(arguments.data, arguments.limit)
    evaluation, failures = _load_or_decompose(arguments.data, evaluation_count, arguments.workers,
                                              arguments.type_key, "Decomposing eval ",
                                              cache_for(arguments.data, "decompositions"))

    template_cache = cache_for(arguments.template_data, "templates")
    if template_cache is not None and template_cache.exists():
        payload = pickle.loads(template_cache.read_bytes())
        fine, coarse = payload["fine"], payload["coarse"]
        print(f"Loaded templates from {template_cache}.")
    else:
        print("\nFitting templates.")
        fine = fit_templates(_collect_instances(training, fine=True), species_aware=True)
        coarse = fit_templates(_collect_instances(training, fine=False), species_aware=False)
        if template_cache is not None:
            template_cache.parent.mkdir(parents=True, exist_ok=True)
            template_cache.write_bytes(pickle.dumps({"fine": fine, "coarse": coarse}))
    templates, provenance = _resolve_templates(evaluation, fine, coarse)

    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_initialise_scorer,
                             initargs=(fine, coarse, pose_noise, arguments.linkage)) as executor:
        results = list(tqdm(executor.map(_score_structure, list(enumerate(evaluation)), chunksize=8),
                            total=len(evaluation), desc="Rebuilding    ", unit=" structures"))

    _report_decomposition(evaluation, failures, evaluation_count)
    _report_templates(templates, provenance)
    _report_ceiling(results, pose_noise)
    summary = _summarise(results, pose_noise, arguments.type_key, evaluation_count, failures, provenance)
    print(f"\nG1 free match rate at {G1_TRANSLATION:.2f} A / {G1_ROTATION:.1f} deg, comparison tolerance: "
          f"{100.0 * summary['g1']['free_match_rate_comparison']:.2f} per cent "
          f"({'pass' if summary['g1']['passed'] else 'fail'}, threshold {100.0 * G1_FREE_MATCH_RATE:.0f}).")
    if arguments.json_out:
        out_path = Path(arguments.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote {out_path}.")


if __name__ == "__main__":
    main()
