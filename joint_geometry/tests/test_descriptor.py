"""Correctness tests for CN-RDF transforms and aligned endpoint tables."""

from pathlib import Path

import numpy as np
import pytest
import torch

from cgfm.blockdata import settings_hash
from direct_geometry.tests.conftest import CUBIC, SKEWED, random_structure, rotation, supercell
from joint_geometry.data import GeometryTable, within_element_permutation
from joint_geometry.descriptor import (
    DescriptorTransform,
    JointGeometryDescriptorSettings,
    clean_radial_descriptor,
)


def _raw(rows: int = 200) -> np.ndarray:
    generator = np.random.default_rng(4)
    shells = generator.dirichlet(np.ones(16), size=rows)
    coordination = generator.normal(2.0, 0.4, size=(rows, 1))
    return np.concatenate([shells, coordination], axis=1)


@pytest.mark.parametrize(
    ("representation", "dimension"),
    (("cn-rdf4", 4), ("cn-rdf8", 8), ("radial17", 17)),
)
def test_transform_has_requested_width_and_unit_scale(representation: str, dimension: int) -> None:
    settings = JointGeometryDescriptorSettings(representation=representation)
    transform = DescriptorTransform.fit(_raw(), settings)
    values = transform.transform(_raw())
    assert values.shape == (200, dimension)
    assert np.isfinite(values).all()
    assert np.allclose(values.mean(axis=0), 0.0, atol=2.0e-5)
    assert np.allclose(values.std(axis=0), 1.0, atol=5.0e-3)


@pytest.mark.parametrize("representation", ("cn-rdf4", "cn-rdf8", "radial17"))
def test_torch_transform_matches_numpy(representation: str) -> None:
    settings = JointGeometryDescriptorSettings(representation=representation)
    raw = _raw()
    transform = DescriptorTransform.fit(raw, settings)
    expected = transform.transform(raw)
    actual = transform.transform_tensor(torch.tensor(raw, dtype=torch.float64)).numpy()
    assert np.allclose(actual, expected, atol=2.0e-6)


def test_transform_round_trip(tmp_path: Path) -> None:
    transform = DescriptorTransform.fit(_raw(), JointGeometryDescriptorSettings())
    path = tmp_path / "transform.pkl"
    transform.save(path)
    loaded = DescriptorTransform.load(path)
    assert np.array_equal(loaded.transform(_raw()), transform.transform(_raw()))


def test_raw_descriptor_invariances_survive_compression() -> None:
    settings = JointGeometryDescriptorSettings()
    frac, cell, atoms = random_structure(5, SKEWED, seed=27)
    frac = frac.to(cell.dtype)
    raw = clean_radial_descriptor(frac, cell, atoms, settings)
    fit_rows = np.concatenate([raw.numpy(), _raw()])
    transform = DescriptorTransform.fit(fit_rows, settings)
    reference = transform.transform_tensor(raw)

    translated = clean_radial_descriptor(
        frac + torch.tensor([0.2, -0.4, 1.1], dtype=frac.dtype), cell, atoms, settings
    )
    turned = clean_radial_descriptor(
        frac, cell @ rotation([0.3, 0.7, -1.1]).to(cell.dtype), atoms, settings
    )
    assert torch.allclose(reference, transform.transform_tensor(translated), atol=1.0e-9)
    assert torch.allclose(reference, transform.transform_tensor(turned), atol=1.0e-9)

    primitive_frac, primitive_cell, primitive_atoms = random_structure(3, CUBIC, seed=33)
    primitive_frac = primitive_frac.to(primitive_cell.dtype)
    primitive = transform.transform_tensor(
        clean_radial_descriptor(primitive_frac, primitive_cell, primitive_atoms, settings)
    )
    tiled_frac, tiled_cell, tiled_atoms = supercell(primitive_frac, CUBIC, repeats=2)
    tiled = transform.transform_tensor(
        clean_radial_descriptor(tiled_frac, tiled_cell, tiled_atoms, settings)
    )
    assert torch.allclose(primitive, tiled[:3], atol=1.0e-9)


def test_geometry_table_detects_alignment_and_settings_errors(tmp_path: Path) -> None:
    settings = JointGeometryDescriptorSettings().as_dict()
    digest = settings_hash(settings)
    table = GeometryTable(
        identifiers=np.asarray(["a", "b"]),
        atom_offsets=np.asarray([0, 2, 3]),
        numbers=np.asarray([8, 14, 6]),
        values=np.zeros((3, 4), dtype=np.float32),
        settings_digest=digest,
    )
    path = tmp_path / "table.npz"
    table.save(path)
    loaded = GeometryTable.load(path)
    loaded.check_against(["a", "b"], [np.asarray([8, 14]), np.asarray([6])], settings, "fixture")
    with pytest.raises(ValueError, match="atom ordering"):
        loaded.check_against(["a", "b"], [np.asarray([14, 8]), np.asarray([6])], settings, "fixture")
    with pytest.raises(ValueError, match="different descriptor settings"):
        loaded.check_against(["a", "b"], [np.asarray([8, 14]), np.asarray([6])], {**settings, "cutoff": 5.0}, "fixture")


def test_within_element_shuffle_preserves_elements_and_is_not_identity() -> None:
    numbers = np.asarray([8, 14, 8, 6, 14, 8, 6])
    permutation = within_element_permutation(numbers, seed=3)
    assert np.array_equal(numbers[permutation], numbers)
    assert not np.array_equal(permutation, np.arange(len(numbers)))
