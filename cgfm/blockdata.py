"""
Leakage-safe rigid-block tables, datasets and batching.

Templates, CN validity masks and the sharing map are fitted on the training split only. Every other split is transformed
with the same centre rule, coordination cap, orphan repair and fallback policy. The on-disk table stores source
identifiers and a hash of those settings, and loading fails if either disagrees with the dataset it is paired with.

Rotations are stored as one valid representative plus the template stabiliser. The symmetry-equivalent target nearest
the sampled base orientation is chosen online, so the table is not canonicalised against a fixed reference.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import hashlib
import json
import pickle
import numpy as np
import torch
from pymatgen.core import Element
from torch_geometric.data import Dataset
from omg.datamodule import OMGData, OMGDataModule, StructureDataset
from omg.globals import MAX_ATOM_NUM
from .blocks import (CN_CLASSES, COORDINATION_CAP, Decomposition, Template, align, centre_elements, encode_coordination,
                     lookup_template, template_species_probs)
from .readout import derive_templates, orphan_free
from .so3 import stabiliser


MAX_VERTICES = COORDINATION_CAP
"""Largest number of vertices stored per block, equal to the coordination cap."""

MAX_STABILISER = 60
"""Largest finite rotation group stored per block, enough for an icosahedron."""

MAX_STRUCTURE_ATOMS = 52
"""Largest number of atoms in an MPTS-52 structure, used to pad graph-level composition tensors."""

SETTINGS_VERSION = 3
"""Version of the preprocessing settings recorded in the manifest."""


def default_settings(type_key_mode: str = "centre-cn") -> dict:
    """
    Return the preprocessing settings that every split of one experiment must share.

    :param type_key_mode:
        How block types are defined.
        Defaults to "centre-cn".
    :type type_key_mode: str

    :return:
        JSON-serialisable settings.
    :rtype: dict
    """
    return {
        "version": SETTINGS_VERSION,
        "type_key_mode": type_key_mode,
        "coordination_cap": COORDINATION_CAP,
        "exclude_centre_vertices": True,
        "orphan_free": True,
        "coarse_species_aware": False,
        "fine_species_aware": True,
        "composition_conditioned_fine_templates": True,
        "max_vertices": MAX_VERTICES,
        "max_stabiliser": MAX_STABILISER,
        "max_structure_atoms": MAX_STRUCTURE_ATOMS,
        "cn_classes": CN_CLASSES,
    }


def settings_hash(settings: dict) -> str:
    """
    Hash preprocessing settings so a table cannot silently be paired with a different policy.

    :param settings:
        Settings dictionary.
    :type settings: dict

    :return:
        Hex digest.
    :rtype: str
    """
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """
    Hash a source dataset file without loading it into memory.

    :param path:
        File to hash.
    :type path: pathlib.Path
    :param chunk_size:
        Bytes read per update.
        Defaults to eight MiB.
    :type chunk_size: int

    :return:
        SHA-256 hex digest.
    :rtype: str
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cartesian_to_fractional(coords: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates to fractional coordinates in [0, 1).

    :param coords:
        Cartesian coordinates of shape (N, 3).
    :type coords: numpy.ndarray
    :param lattice:
        Cell vectors of shape (3, 3), with row i the i-th lattice vector.
    :type lattice: numpy.ndarray

    :return:
        Fractional coordinates of shape (N, 3).
    :rtype: numpy.ndarray
    """
    return np.remainder(coords @ np.linalg.inv(lattice), 1.0)


def _pad_vertices(offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad vertex offsets to ``MAX_VERTICES`` and return the accompanying mask.

    :param offsets:
        Vertex offsets of shape (n, 3).
    :type offsets: numpy.ndarray

    :return:
        Padded offsets of shape (MAX_VERTICES, 3) and a boolean mask of shape (MAX_VERTICES,).
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    padded = np.zeros((MAX_VERTICES, 3), dtype=np.float64)
    mask = np.zeros(MAX_VERTICES, dtype=bool)
    count = min(len(offsets), MAX_VERTICES)
    if count:
        padded[:count] = offsets[:count]
        mask[:count] = True
    return padded, mask


def _pad_probs(probabilities: np.ndarray) -> np.ndarray:
    """
    Pad a per-slot element distribution to ``MAX_VERTICES``.

    :param probabilities:
        Distribution of shape (n, MAX_ATOM_NUM + 1).
    :type probabilities: numpy.ndarray

    :return:
        Padded distribution of shape (MAX_VERTICES, MAX_ATOM_NUM + 1).
    :rtype: numpy.ndarray
    """
    padded = np.zeros((MAX_VERTICES, MAX_ATOM_NUM + 1), dtype=np.float64)
    count = min(len(probabilities), MAX_VERTICES)
    if count:
        padded[:count] = probabilities[:count]
    return padded


def _pad_stabiliser(group: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Pad a finite rotation group with copies of the identity.

    :param group:
        Rotations of shape (S, 3, 3).
    :type group: numpy.ndarray

    :return:
        Padded group of shape (MAX_STABILISER, 3, 3) and the true group order.
    :rtype: tuple[numpy.ndarray, int]
    """
    padded = np.repeat(np.eye(3)[None, :, :], MAX_STABILISER, axis=0)
    order = min(len(group), MAX_STABILISER)
    if order:
        padded[:order] = group[:order]
    return padded, order


@dataclass
class TemplateLibrary:
    """
    Train-only templates, CN validity mask and the settings they were fitted under.

    :param coarse:
        Templates keyed by ``(centre, CN)``.
    :type coarse: dict[tuple, Template]
    :param fine:
        Templates keyed by ``(centre, CN, ligands)``.
    :type fine: dict[tuple, Template]
    :param cn_valid:
        Whether coordination number ``c`` was observed for atomic number ``Z``, of shape
        (MAX_ATOM_NUM + 1, CN_CLASSES) indexing ``[Z, c]``.
    :type cn_valid: numpy.ndarray
    :param settings:
        Preprocessing settings.
    :type settings: dict
    :param settings_digest:
        Hash of ``settings``.
    :type settings_digest: str
    """

    coarse: dict[tuple, Template]
    fine: dict[tuple, Template]
    cn_valid: np.ndarray
    settings: dict
    settings_digest: str

    def save(self, path: Path) -> None:
        """
        Write the library to a pickle.

        :param path:
            Destination path.
        :type path: Path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({
            "coarse": self.coarse, "fine": self.fine, "cn_valid": self.cn_valid,
            "settings": self.settings, "settings_digest": self.settings_digest}))

    @staticmethod
    def load(path: Path) -> "TemplateLibrary":
        """
        Read a library from a pickle.

        :param path:
            Path of the pickle.
        :type path: Path

        :return:
            The library.
        :rtype: TemplateLibrary
        """
        payload = pickle.loads(Path(path).read_bytes())
        return TemplateLibrary(coarse=payload["coarse"], fine=payload["fine"], cn_valid=payload["cn_valid"],
                               settings=payload["settings"], settings_digest=payload["settings_digest"])

    def check_settings(self, settings: dict) -> None:
        """
        Fail if this library was fitted under a different preprocessing policy.

        :param settings:
            Expected settings.
        :type settings: dict

        :raises ValueError:
            If the hash disagrees.
        """
        digest = settings_hash(settings)
        if digest != self.settings_digest:
            raise ValueError(
                "The template library was fitted under different preprocessing settings than the ones requested.")


def cn_validity_mask(coarse: dict[tuple, Template]) -> np.ndarray:
    """
    Build the per-centre-element CN mask from the train-only coarse vocabulary.

    :param coarse:
        Templates keyed by ``(centre, CN)``.
    :type coarse: dict[tuple, Template]

    :return:
        Boolean mask of shape (MAX_ATOM_NUM + 1, CN_CLASSES).
    :rtype: numpy.ndarray
    """
    valid = np.zeros((MAX_ATOM_NUM + 1, CN_CLASSES), dtype=bool)
    for key in coarse:
        number = Element(key[0]).Z
        valid[number, int(key[1])] = True
    for number in range(1, MAX_ATOM_NUM + 1):
        if not valid[number].any():
            valid[number, 0] = True
    return valid


@dataclass
class BlockTable:
    """
    Precomputed rigid-block representation of every structure of one split.

    Arrays that vary with the number of blocks are concatenated; ``block_offsets`` recovers one structure. Arrays that
    vary with the original atom count are concatenated; ``atom_offsets`` recovers them.

    :param identifiers:
        Material identifier of every structure.
    :type identifiers: numpy.ndarray
    :param n_blocks:
        Number of blocks of every structure.
    :type n_blocks: numpy.ndarray
    :param n_target_atoms:
        Number of original atoms of every structure.
    :type n_target_atoms: numpy.ndarray
    :param single_anion:
        Whether the composition determines the non-centre element.
    :type single_anion: numpy.ndarray
    :param fallback_count:
        Number of blocks of every structure that did not have an exact train-only template.
    :type fallback_count: numpy.ndarray
    :param cells:
        Lattice of every structure, of shape (S, 3, 3).
    :type cells: numpy.ndarray
    :param block_offsets:
        Start index of every structure within the concatenated block arrays, of shape (S + 1,).
    :type block_offsets: numpy.ndarray
    :param atom_offsets:
        Start index of every structure within the concatenated atom arrays, of shape (S + 1,).
    :type atom_offsets: numpy.ndarray
    :param centre_numbers:
        Atomic number of every block centre.
    :type centre_numbers: numpy.ndarray
    :param frac_pos:
        Fractional translation of every block.
    :type frac_pos: numpy.ndarray
    :param rotations:
        One valid target rotation of every block, of shape (total blocks, 3, 3).
    :type rotations: numpy.ndarray
    :param block_type:
        Encoded coordination number of every block.
    :type block_type: numpy.ndarray
    :param template_offsets:
        Padded template vertices of every block.
    :type template_offsets: numpy.ndarray
    :param template_mask:
        Whether each padded vertex slot is used.
    :type template_mask: numpy.ndarray
    :param template_probs:
        Padded per-slot element distribution of every block.
    :type template_probs: numpy.ndarray
    :param stabilizer:
        Padded finite rotation group of every block.
    :type stabilizer: numpy.ndarray
    :param stabilizer_size:
        True group order of every block.
    :type stabilizer_size: numpy.ndarray
    :param vote_atom:
        Local atom index that each template slot of each block votes for, or -1.
    :type vote_atom: numpy.ndarray
    :param centre_atom:
        Local atom index placed directly by each block translation.
    :type centre_atom: numpy.ndarray
    :param target_numbers:
        Atomic numbers of the original atoms.
    :type target_numbers: numpy.ndarray
    :param target_frac:
        Fractional coordinates of the original atoms.
    :type target_frac: numpy.ndarray
    :param settings_digest:
        Hash of the preprocessing settings.
    :type settings_digest: str
    """

    identifiers: np.ndarray
    n_blocks: np.ndarray
    n_target_atoms: np.ndarray
    single_anion: np.ndarray
    fallback_count: np.ndarray
    cells: np.ndarray
    block_offsets: np.ndarray
    atom_offsets: np.ndarray
    centre_numbers: np.ndarray
    frac_pos: np.ndarray
    rotations: np.ndarray
    block_type: np.ndarray
    template_offsets: np.ndarray
    template_mask: np.ndarray
    template_probs: np.ndarray
    stabilizer: np.ndarray
    stabilizer_size: np.ndarray
    vote_atom: np.ndarray
    centre_atom: np.ndarray
    target_numbers: np.ndarray
    target_frac: np.ndarray
    settings_digest: str

    def __len__(self) -> int:
        """Return the number of structures."""
        return len(self.n_blocks)

    def save(self, path: Path) -> None:
        """
        Write the table to a compressed numpy archive.

        :param path:
            Destination path.
        :type path: Path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, identifiers=self.identifiers, n_blocks=self.n_blocks, n_target_atoms=self.n_target_atoms,
            single_anion=self.single_anion, fallback_count=self.fallback_count, cells=self.cells,
            block_offsets=self.block_offsets, atom_offsets=self.atom_offsets, centre_numbers=self.centre_numbers,
            frac_pos=self.frac_pos, rotations=self.rotations, block_type=self.block_type,
            template_offsets=self.template_offsets, template_mask=self.template_mask,
            template_probs=self.template_probs, stabilizer=self.stabilizer, stabilizer_size=self.stabilizer_size,
            vote_atom=self.vote_atom, centre_atom=self.centre_atom,
            target_numbers=self.target_numbers, target_frac=self.target_frac,
            settings_digest=np.array(self.settings_digest))

    @staticmethod
    def load(path: Path) -> "BlockTable":
        """
        Read a table from a compressed numpy archive.

        :param path:
            Path of the archive.
        :type path: Path

        :return:
            The table.
        :rtype: BlockTable
        """
        with np.load(path, allow_pickle=False) as archive:
            return BlockTable(
                identifiers=archive["identifiers"], n_blocks=archive["n_blocks"].astype(np.int64),
                n_target_atoms=archive["n_target_atoms"].astype(np.int64),
                single_anion=archive["single_anion"].astype(bool),
                fallback_count=archive["fallback_count"].astype(np.int64), cells=archive["cells"],
                block_offsets=archive["block_offsets"].astype(np.int64),
                atom_offsets=archive["atom_offsets"].astype(np.int64),
                centre_numbers=archive["centre_numbers"].astype(np.int64), frac_pos=archive["frac_pos"],
                rotations=archive["rotations"], block_type=archive["block_type"].astype(np.int64),
                template_offsets=archive["template_offsets"], template_mask=archive["template_mask"].astype(bool),
                template_probs=archive["template_probs"], stabilizer=archive["stabilizer"],
                stabilizer_size=archive["stabilizer_size"].astype(np.int64),
                vote_atom=archive["vote_atom"].astype(np.int64),
                centre_atom=archive["centre_atom"].astype(np.int64),
                target_numbers=archive["target_numbers"].astype(np.int64), target_frac=archive["target_frac"],
                settings_digest=str(archive["settings_digest"]))

    def check_against(self, identifiers: Sequence[str], settings: dict, source: str) -> None:
        """
        Check that this table describes exactly the given structures under the given settings.

        :param identifiers:
            Material identifier of every structure of the dataset, in dataset order.
        :type identifiers: Sequence[str]
        :param settings:
            Expected preprocessing settings.
        :type settings: dict
        :param source:
            Description of the dataset, used in the error message.
        :type source: str

        :raises ValueError:
            If the table covers a different split or a different preprocessing policy.
        """
        digest = settings_hash(settings)
        if digest != self.settings_digest:
            raise ValueError(
                f"The block table paired with {source} was built under different preprocessing settings.")
        if len(identifiers) != len(self):
            raise ValueError(
                f"The block table covers {len(self)} structures but {source} holds {len(identifiers)}.")
        stored = np.asarray(self.identifiers)
        expected = np.asarray(identifiers, dtype=stored.dtype)
        mismatched = np.flatnonzero(stored != expected)
        if mismatched.size > 0:
            first = int(mismatched[0])
            raise ValueError(
                f"The block table disagrees with {source} at {mismatched.size} of {len(self)} structures, first at "
                f"index {first}: the file has '{self.identifiers[first]}' and the dataset has "
                f"'{identifiers[first]}'.")


def record_structure(decomposition: Decomposition, library: TemplateLibrary,
                     type_key_mode: str = "centre-cn") -> dict[str, np.ndarray]:
    """
    Convert one repaired decomposition into the tensors stored for a single structure.

    :param decomposition:
        The structure after ``orphan_free``.
    :type decomposition: Decomposition
    :param library:
        Train-only template library.
    :type library: TemplateLibrary
    :param type_key_mode:
        How block types are defined.
        Defaults to "centre-cn".
    :type type_key_mode: str

    :return:
        Arrays for this structure, including a scalar fallback count.
    :rtype: dict[str, numpy.ndarray]
    """
    symbols = tuple(Element.from_Z(int(number)).symbol for number in decomposition.numbers)
    centres, _ = centre_elements(symbols)
    single_anion = len(set(symbols) - set(centres)) <= 1
    frac_atoms = cartesian_to_fractional(decomposition.coords, decomposition.lattice)
    n_blocks = len(decomposition.blocks)
    centre_numbers = np.array([decomposition.numbers[block.centre] for block in decomposition.blocks], dtype=np.int64)
    frac_pos = frac_atoms[[block.centre for block in decomposition.blocks]]
    rotations = np.zeros((n_blocks, 3, 3), dtype=np.float64)
    block_type = np.zeros(n_blocks, dtype=np.int64)
    template_offsets = np.zeros((n_blocks, MAX_VERTICES, 3), dtype=np.float64)
    template_mask = np.zeros((n_blocks, MAX_VERTICES), dtype=bool)
    template_probs = np.zeros((n_blocks, MAX_VERTICES, MAX_ATOM_NUM + 1), dtype=np.float64)
    stabilizer = np.zeros((n_blocks, MAX_STABILISER, 3, 3), dtype=np.float64)
    stabilizer_size = np.zeros(n_blocks, dtype=np.int64)
    vote_atom = np.full((n_blocks, MAX_VERTICES), -1, dtype=np.int64)
    centre_atom = np.array([block.centre for block in decomposition.blocks], dtype=np.int64)
    fallback_count = 0
    placement_templates = (
        derive_templates(decomposition.numbers, library.fine, library.coarse)
        if type_key_mode == "centre-cn" else library.fine)

    for index, block in enumerate(decomposition.blocks):
        key = (block.type_key[0], int(block.type_key[1])) if type_key_mode == "centre-cn" else block.type_key
        template, provenance = lookup_template(key, placement_templates, fallback=library.coarse)
        if provenance != "exact":
            fallback_count += 1
        coordination = len(block.ligands)
        block_type[index] = encode_coordination(coordination)
        padded_offsets, mask = _pad_vertices(template.offsets)
        template_offsets[index] = padded_offsets
        template_mask[index] = mask
        template_probs[index] = _pad_probs(template_species_probs(template))
        group = stabiliser(template.offsets, template.species, species_aware=template.species_aware)
        padded_group, order = _pad_stabiliser(group)
        stabilizer[index] = padded_group
        stabilizer_size[index] = order
        if coordination == 0 or len(template.offsets) != coordination:
            rotations[index] = np.eye(3)
            continue
        rotation, correspondence, _ = align(
            template.offsets, block.offsets, template.species, block.species,
            species_aware=template.species_aware)
        rotations[index] = rotation
        for slot, vertex in enumerate(correspondence):
            vote_atom[index, slot] = int(block.ligands[vertex])

    return {
        "centre_numbers": centre_numbers, "frac_pos": frac_pos, "rotations": rotations, "block_type": block_type,
        "template_offsets": template_offsets, "template_mask": template_mask, "template_probs": template_probs,
        "stabilizer": stabilizer, "stabilizer_size": stabilizer_size, "vote_atom": vote_atom,
        "centre_atom": centre_atom,
        "target_numbers": decomposition.numbers.astype(np.int64), "target_frac": frac_atoms,
        "cell": decomposition.lattice.astype(np.float64), "single_anion": np.array(single_anion),
        "fallback_count": np.array(fallback_count, dtype=np.int64),
        "n_target_atoms": np.array(len(decomposition.numbers), dtype=np.int64),
    }


def table_from_records(identifiers: Sequence[str], records: Sequence[dict[str, np.ndarray]],
                       settings_digest: str) -> BlockTable:
    """
    Assemble a table from per-structure records.

    :param identifiers:
        Material identifier of every structure.
    :type identifiers: Sequence[str]
    :param records:
        Output of ``record_structure`` for every structure.
    :type records: Sequence[dict[str, numpy.ndarray]]
    :param settings_digest:
        Hash of the preprocessing settings.
    :type settings_digest: str

    :return:
        The table.
    :rtype: BlockTable
    """
    n_blocks = np.array([len(record["centre_numbers"]) for record in records], dtype=np.int64)
    n_target_atoms = np.array([int(record["n_target_atoms"]) for record in records], dtype=np.int64)
    return BlockTable(
        identifiers=np.asarray(identifiers, dtype=np.str_),
        n_blocks=n_blocks, n_target_atoms=n_target_atoms,
        single_anion=np.array([bool(record["single_anion"]) for record in records]),
        fallback_count=np.array([int(record["fallback_count"]) for record in records], dtype=np.int64),
        cells=np.stack([record["cell"] for record in records]),
        block_offsets=np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(n_blocks)]),
        atom_offsets=np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(n_target_atoms)]),
        centre_numbers=np.concatenate([record["centre_numbers"] for record in records]),
        frac_pos=np.concatenate([record["frac_pos"] for record in records]),
        rotations=np.concatenate([record["rotations"] for record in records]),
        block_type=np.concatenate([record["block_type"] for record in records]),
        template_offsets=np.concatenate([record["template_offsets"] for record in records]),
        template_mask=np.concatenate([record["template_mask"] for record in records]),
        template_probs=np.concatenate([record["template_probs"] for record in records]),
        stabilizer=np.concatenate([record["stabilizer"] for record in records]),
        stabilizer_size=np.concatenate([record["stabilizer_size"] for record in records]),
        vote_atom=np.concatenate([record["vote_atom"] for record in records]),
        centre_atom=np.concatenate([record["centre_atom"] for record in records]),
        target_numbers=np.concatenate([record["target_numbers"] for record in records]),
        target_frac=np.concatenate([record["target_frac"] for record in records]),
        settings_digest=settings_digest,
    )


def dataset_identifiers(dataset: StructureDataset) -> list[str]:
    """
    Read the material identifier of every structure of a split, in dataset order.

    :param dataset:
        The split to read.
    :type dataset: StructureDataset

    :return:
        Identifier of every structure.
    :rtype: list[str]
    """
    return [str(dataset[index].metadata.get("identifier", index)) for index in range(len(dataset))]


_BLOCK_FLOAT_FIELDS = (
    "cell", "pos", "rot", "template_offsets", "template_probs", "stabilizer", "target_frac",
)
"""Continuous fields that must follow ``trainer.precision``, matching atomwise ``Structure.to``."""


def _dataset_floating_dtype(dataset: StructureDataset) -> torch.dtype:
    """
    Return the coordinate dtype the wrapped StructureDataset was constructed with.

    :param dataset:
        The StructureDataset Lightning instantiated with ``floating_point_precision``.
    :type dataset: StructureDataset

    :return:
        The torch dtype applied to that dataset's coordinates.
    :rtype: torch.dtype

    :raises TypeError:
        If the dataset has no recorded torch precision.
    """
    dtype = getattr(dataset, "_torch_precision", None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError("The wrapped StructureDataset has no torch floating-point precision.")
    return dtype


def _to_block_data(record_slice: dict[str, np.ndarray], dtype: Optional[torch.dtype] = None) -> OMGData:
    """
    Convert one structure's block arrays into an ``OMGData`` graph whose nodes are blocks.

    On-disk tables stay float64. When ``dtype`` is set, every continuous field is cast so the graph matches the
    trainer precision, the same contract atomwise ``StructureDataset`` already honours.

    :param record_slice:
        Arrays of one structure, as stored in the table.
    :type record_slice: dict[str, numpy.ndarray]
    :param dtype:
        Torch floating-point dtype for continuous fields. ``None`` keeps the numpy dtypes.
        Defaults to None.
    :type dtype: Optional[torch.dtype]

    :return:
        The graph.
    :rtype: OMGData

    :raises ValueError:
        If the structure has more atoms than ``MAX_STRUCTURE_ATOMS``.
    """
    n_blocks = len(record_slice["centre_numbers"])
    n_atoms = int(record_slice["n_target_atoms"])
    if n_atoms > MAX_STRUCTURE_ATOMS:
        raise ValueError(f"Structure has {n_atoms} atoms, above the MPTS-52 cap of {MAX_STRUCTURE_ATOMS}.")
    data = OMGData()
    data.n_atoms = torch.tensor(n_blocks)
    data.species = torch.from_numpy(record_slice["centre_numbers"]).long()
    data.cell = torch.from_numpy(record_slice["cell"]).unsqueeze(0)
    data.pos = torch.from_numpy(record_slice["frac_pos"])
    data.pos_is_fractional = torch.tensor(True)
    data.property = {}
    data.rot = torch.from_numpy(record_slice["rotations"])
    data.block_type = torch.from_numpy(record_slice["block_type"]).long()
    data.template_offsets = torch.from_numpy(record_slice["template_offsets"])
    data.template_mask = torch.from_numpy(record_slice["template_mask"])
    data.template_probs = torch.from_numpy(record_slice["template_probs"])
    data.stabilizer = torch.from_numpy(record_slice["stabilizer"])
    data.stabilizer_size = torch.from_numpy(record_slice["stabilizer_size"]).long()
    data.vote_atom = torch.from_numpy(record_slice["vote_atom"]).long()
    data.centre_atom = torch.from_numpy(record_slice["centre_atom"]).long()
    data.n_target_atoms = torch.tensor(n_atoms)
    target_numbers = torch.zeros(1, MAX_STRUCTURE_ATOMS, dtype=torch.long)
    target_frac = torch.zeros(
        1, MAX_STRUCTURE_ATOMS, 3, dtype=torch.from_numpy(record_slice["target_frac"]).dtype)
    target_mask = torch.zeros(1, MAX_STRUCTURE_ATOMS, dtype=torch.bool)
    target_numbers[0, :n_atoms] = torch.from_numpy(record_slice["target_numbers"]).long()
    target_frac[0, :n_atoms] = torch.from_numpy(record_slice["target_frac"])
    target_mask[0, :n_atoms] = True
    data.target_numbers = target_numbers
    data.target_frac = target_frac
    data.target_mask = target_mask
    data.single_anion = torch.tensor(bool(record_slice["single_anion"]))
    data.fallback_count = torch.tensor(int(record_slice["fallback_count"]))
    if dtype is not None:
        for name in _BLOCK_FLOAT_FIELDS:
            value = getattr(data, name)
            if value.dtype != dtype:
                setattr(data, name, value.to(dtype))
    return data


class BlockDataset(Dataset):
    """
    Dataset that returns one block graph per structure of a split.

    :param dataset:
        StructureDataset holding the original crystals, used only to check identifiers.
    :type dataset: StructureDataset
    :param table:
        Precomputed block table of this split.
    :type table: BlockTable
    :param library:
        Train-only template library, stored so the Lightning module can read it from the datamodule.
    :type library: TemplateLibrary
    :param settings:
        Expected preprocessing settings.
    :type settings: dict
    """

    def __init__(self, dataset: StructureDataset, table: BlockTable, library: TemplateLibrary,
                 settings: dict) -> None:
        """Construct the dataset."""
        super().__init__()
        identifiers = dataset_identifiers(dataset)
        table.check_against(identifiers, settings, "the dataset the block table was paired with")
        library.check_settings(settings)
        self._dataset = dataset
        self._table = table
        self.library = library
        self._dtype = _dataset_floating_dtype(dataset)

    def len(self) -> int:
        """Return the number of structures."""
        return len(self._table)

    def get(self, idx: int) -> OMGData:
        """
        Return the block graph of one structure.

        :param idx:
            Index of the structure.
        :type idx: int

        :return:
            The block graph.
        :rtype: OMGData
        """
        table = self._table
        block_slice = slice(int(table.block_offsets[idx]), int(table.block_offsets[idx + 1]))
        atom_slice = slice(int(table.atom_offsets[idx]), int(table.atom_offsets[idx + 1]))
        return _to_block_data({
            "centre_numbers": table.centre_numbers[block_slice],
            "frac_pos": table.frac_pos[block_slice],
            "rotations": table.rotations[block_slice],
            "block_type": table.block_type[block_slice],
            "template_offsets": table.template_offsets[block_slice],
            "template_mask": table.template_mask[block_slice],
            "template_probs": table.template_probs[block_slice],
            "stabilizer": table.stabilizer[block_slice],
            "stabilizer_size": table.stabilizer_size[block_slice],
            "vote_atom": table.vote_atom[block_slice],
            "centre_atom": table.centre_atom[block_slice],
            "target_numbers": table.target_numbers[atom_slice],
            "target_frac": table.target_frac[atom_slice],
            "cell": table.cells[idx],
            "single_anion": table.single_anion[idx],
            "fallback_count": table.fallback_count[idx],
            "n_target_atoms": table.n_target_atoms[idx],
        }, dtype=self._dtype)


class BlockDataModule(OMGDataModule):
    """
    Data module whose graphs are blocks rather than atoms.

    :param train_dataset:
        StructureDataset for training.
    :type train_dataset: StructureDataset
    :param val_dataset:
        StructureDataset for validation.
    :type val_dataset: StructureDataset
    :param pred_dataset:
        StructureDataset for prediction.
    :type pred_dataset: StructureDataset
    :param block_dir:
        Directory holding ``train.npz``, ``val.npz``, ``test.npz``, ``templates.pkl`` and ``manifest.json``.
    :type block_dir: str
    """

    def __init__(self, train_dataset: StructureDataset, val_dataset: StructureDataset,
                 pred_dataset: StructureDataset, block_dir: str,
                 train_batch_size: Optional[int] = None, val_batch_size: Optional[int] = None,
                 pred_batch_size: Optional[int] = None, **kwargs: Any) -> None:
        """Construct the data module."""
        super().__init__(train_dataset=train_dataset, val_dataset=val_dataset, pred_dataset=pred_dataset,
                         train_batch_size=train_batch_size, val_batch_size=val_batch_size,
                         pred_batch_size=pred_batch_size, **kwargs)
        root = Path(block_dir)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No block manifest at {manifest_path}. Produce it with cgfm.scripts.precompute_blocks.")
        manifest = json.loads(manifest_path.read_text())
        settings = manifest["settings"]
        source_hashes = manifest.get("source_hashes")
        if not isinstance(source_hashes, dict):
            raise ValueError(
                f"The block manifest at {manifest_path} has no source hashes; rebuild the block tables.")
        datasets = {"train": train_dataset, "val": val_dataset, "test": pred_dataset}
        dtypes = {split: _dataset_floating_dtype(dataset) for split, dataset in datasets.items()}
        if len(set(dtypes.values())) != 1:
            raise ValueError(
                f"Train, val and test StructureDatasets disagree on floating-point precision: {dtypes}.")
        for split, dataset in datasets.items():
            source_path = getattr(dataset, "_file_path", None)
            if source_path is None:
                raise ValueError(f"Cannot identify the source file of the {split} dataset.")
            actual_hash = file_sha256(Path(source_path))
            expected_hash = source_hashes.get(split)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"The {split} source dataset does not match the file used to precompute the block tables.")
        library = TemplateLibrary.load(root / "templates.pkl")
        library.check_settings(settings)
        self.library = library
        self.settings = settings
        self.train_dataset = BlockDataset(train_dataset, BlockTable.load(root / "train.npz"), library, settings)
        self.val_dataset = BlockDataset(val_dataset, BlockTable.load(root / "val.npz"), library, settings)
        self.pred_dataset = BlockDataset(pred_dataset, BlockTable.load(root / "test.npz"), library, settings)
