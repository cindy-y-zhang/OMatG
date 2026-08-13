"""Tests for leakage-safe block tables, datasets and PyG batching."""

import json
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch
from cgfm.blockdata import (MAX_STRUCTURE_ATOMS, MAX_VERTICES, TemplateLibrary, _to_block_data, cn_validity_mask,
                            default_settings, record_structure, settings_hash, table_from_records)
from cgfm.blocks import Block, Decomposition, Template, fit_templates
from cgfm.tests.test_readout import CHAIN, CHAIN_NUMBERS, CUBE, chain_blocks, hand_built
from cgfm.readout import orphan_free


def tiny_library() -> TemplateLibrary:
    """
    Build a one-type train-only library around a linear Cl-Na-Cl fragment.

    :return:
        The library.
    :rtype: TemplateLibrary
    """
    offsets = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    probabilities = np.zeros((2, 101))
    probabilities[:, 17] = 1.0
    template = Template(offsets=offsets, species=("Cl", "Cl"), count=4, spread=0.0,
                        species_probs=probabilities, species_aware=False)
    coarse = {("Na", 2): template, ("Na", 1): Template(offsets=offsets[:1], species=("Cl",), count=2, spread=0.0,
                                                       species_probs=probabilities[:1], species_aware=False)}
    settings = default_settings("centre-cn")
    return TemplateLibrary(coarse=coarse, fine={}, cn_valid=cn_validity_mask(coarse), settings=settings,
                           settings_digest=settings_hash(settings))


def test_settings_hash_changes_when_the_cap_changes():
    first = settings_hash(default_settings())
    other = default_settings()
    other["coordination_cap"] = 8
    assert settings_hash(other) != first


def test_cn_validity_mask_covers_observed_centre_cn_pairs():
    library = tiny_library()
    assert library.cn_valid[11, 2]
    assert library.cn_valid[11, 1]
    assert not library.cn_valid[11, 6]
    assert library.cn_valid[26, 0]


def test_precompute_keeps_observed_singleton_coordination_class():
    """CN=0 targets must not be masked when the same centre element also has polyhedra."""
    from cgfm.scripts.precompute_blocks import _collect_instances
    singleton = Decomposition(
        identifier="singleton", lattice=np.eye(3), coords=np.zeros((1, 3)), numbers=np.array([11]),
        blocks=(Block(centre=0, ligands=np.empty(0, dtype=np.int64), offsets=np.empty((0, 3)),
                      species=(), type_key=("Na", 0)),),
        num_singletons=1, centre_rule="elemental")
    instances = _collect_instances([singleton], fine=False)
    coarse = fit_templates(instances, species_aware=False)
    assert ("Na", 0) in coarse
    assert cn_validity_mask(coarse)[11, 0]


def test_crystalnn_failure_keeps_composition_defined_block_count(monkeypatch):
    """The fallback must not silently turn every ligand atom into a modelled block."""
    from pymatgen.core import Lattice, Structure
    from cgfm.scripts import precompute_blocks

    pymatgen = Structure(Lattice.cubic(4.0), ["Ti", "O", "O"],
                         [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])

    class Datum:
        metadata = {"identifier": "fallback"}

        @staticmethod
        def get_pymatgen_structure():
            return pymatgen

    monkeypatch.setattr(precompute_blocks, "_DATASET", [Datum()])
    monkeypatch.setattr(precompute_blocks, "_TYPE_KEY_MODE", "centre-cn")
    monkeypatch.setattr(precompute_blocks, "decompose", lambda *args, **kwargs: None)
    decomposition, identifier = precompute_blocks._decompose_structure(0)

    assert identifier == "fallback"
    assert decomposition is not None
    assert len(decomposition.blocks) == 1
    assert int(decomposition.numbers[decomposition.blocks[0].centre]) == 22
    assert sorted(decomposition.blocks[0].ligands.tolist()) == [1, 2]


def test_record_structure_uses_train_only_templates_and_encodes_cn():
    repaired = orphan_free(hand_built(CHAIN, CHAIN_NUMBERS, chain_blocks()))
    library = tiny_library()
    record = record_structure(repaired, library, type_key_mode="centre-cn")
    assert record["block_type"].min() >= 1
    assert record["frac_pos"].shape[0] == len(repaired.blocks)
    assert record["rotations"].shape == (len(repaired.blocks), 3, 3)
    assert record["vote_atom"].shape[1] == MAX_VERTICES
    assert np.array_equal(record["centre_atom"], [block.centre for block in repaired.blocks])
    assert int(record["n_target_atoms"]) == len(CHAIN)


def test_record_structure_uses_the_same_composition_conditioned_templates_as_readout():
    repaired = orphan_free(hand_built(CHAIN, CHAIN_NUMBERS, chain_blocks()))
    base = tiny_library()
    fine = {}
    for key, template in base.coarse.items():
        coordination = int(key[1])
        fine[(key[0], coordination, ("Cl",) * coordination)] = Template(
            offsets=1.25 * template.offsets, species=("Cl",) * coordination,
            count=template.count, spread=template.spread, species_aware=True)
    library = TemplateLibrary(
        coarse=base.coarse, fine=fine, cn_valid=base.cn_valid,
        settings=base.settings, settings_digest=base.settings_digest)

    record = record_structure(repaired, library, type_key_mode="centre-cn")

    for index, block in enumerate(repaired.blocks):
        coordination = len(block.ligands)
        expected = fine[("Na", coordination, ("Cl",) * coordination)].offsets
        assert np.allclose(record["template_offsets"][index, :coordination], expected)


def test_table_round_trip_and_identifier_check(tmp_path):
    repaired = orphan_free(hand_built(CHAIN, CHAIN_NUMBERS, chain_blocks()))
    library = tiny_library()
    record = record_structure(repaired, library)
    table = table_from_records(["mp-1"], [record], library.settings_digest)
    path = tmp_path / "train.npz"
    table.save(path)
    loaded = table.__class__.load(path)
    loaded.check_against(["mp-1"], library.settings, "test")
    with pytest.raises(ValueError):
        loaded.check_against(["mp-2"], library.settings, "test")
    with pytest.raises(ValueError):
        other = default_settings()
        other["coordination_cap"] = 4
        loaded.check_against(["mp-1"], other, "test")


def test_block_graph_batches_rotations_and_padded_compositions():
    repaired = orphan_free(hand_built(CHAIN, CHAIN_NUMBERS, chain_blocks()))
    library = tiny_library()
    record = record_structure(repaired, library)
    first = _to_block_data(record)
    second = _to_block_data(record)
    batch = Batch.from_data_list([first, second])
    assert batch.rot.shape == (int(first.n_atoms) * 2, 3, 3)
    assert batch.block_type.shape == (int(first.n_atoms) * 2,)
    assert batch.centre_atom.shape == (int(first.n_atoms) * 2,)
    assert batch.target_numbers.shape == (2, MAX_STRUCTURE_ATOMS)
    assert batch.target_frac.shape == (2, MAX_STRUCTURE_ATOMS, 3)
    assert batch.target_frac.dtype == batch.pos.dtype == batch.cell.dtype == batch.rot.dtype
    assert int(batch.n_target_atoms.sum()) == 2 * len(CHAIN)
    assert batch.template_offsets.shape[1] == MAX_VERTICES
    single = _to_block_data(record, dtype=torch.float32)
    assert single.rot.dtype == single.pos.dtype == single.cell.dtype == torch.float32
    assert single.template_offsets.dtype == single.stabilizer.dtype == torch.float32
    assert single.target_frac.dtype == single.template_probs.dtype == torch.float32


def _write_nacl_lmdb(path, count: int) -> None:
    """Write ``count`` copies of a rocksalt cell into an OMatG LMDB split."""
    import pickle
    import lmdb
    import torch
    from pymatgen.core import Lattice, Structure as PymatgenStructure
    structure = PymatgenStructure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"],
                                                  [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = torch.tensor(structure.lattice.matrix, dtype=torch.float64)
    pos = torch.tensor(structure.cart_coords, dtype=torch.float64)
    numbers = torch.tensor([site.specie.Z for site in structure], dtype=torch.int64)
    with (lmdb.Environment(str(path), subdir=False, map_size=int(1e8), lock=False) as environment,
          environment.begin(write=True) as transaction):
        for index in range(count):
            transaction.put(str(index).encode(), pickle.dumps({
                "cell": cell, "pos": pos, "atomic_numbers": numbers, "identifier": f"nacl-{index}"}))


def test_precompute_blocks_then_one_batch_train_and_readout(tmp_path):
    """Tiny smoke: train-only tables, one training loss, assignment-free readout of a generated-as-target pose."""
    import sys
    from omg.datamodule import StructureDataset
    from omg.model.heads.pass_through import PassThrough
    from omg.model.model import Model
    from omg.model.model_utils import SinusoidalTimeEmbeddings
    from omg.sampler.cell_distributions import MirrorCell
    from omg.sampler.position_distributions import UniformPositionDistribution
    from cgfm.block_encoder import BlockCSPNet
    from cgfm.block_lightning import _read_out_structure
    from cgfm.block_sampler import BlockSampler
    from cgfm.block_si import CoupledBlockInterpolants
    from cgfm.blockdata import BlockDataModule
    from cgfm.scripts import precompute_blocks

    data_dir = tmp_path / "data"
    block_dir = tmp_path / "blocks"
    for split, count in (("train", 4), ("val", 2), ("test", 2)):
        _write_nacl_lmdb(data_dir / f"{split}.lmdb", count)

    argv = sys.argv
    try:
        sys.argv = ["precompute_blocks", "--data-dir", str(data_dir), "--out-dir", str(block_dir),
                    "--workers", "1", "--type-key", "centre-cn"]
        precompute_blocks.main()
    finally:
        sys.argv = argv

    assert (block_dir / "train.npz").exists()
    assert (block_dir / "templates.pkl").exists()
    assert (block_dir / "manifest.json").exists()

    kwargs = dict(lazy_storage=True, niggli_reduce=False, convert_to_fractional=True,
                  floating_point_precision="64-true")

    def make_datamodule() -> BlockDataModule:
        return BlockDataModule(
            StructureDataset(file_path=str(data_dir / "train.lmdb"), **kwargs),
            StructureDataset(file_path=str(data_dir / "val.lmdb"), **kwargs),
            StructureDataset(file_path=str(data_dir / "test.lmdb"), **kwargs),
            block_dir=str(block_dir), batch_size=2, num_workers=0)

    manifest_path = block_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest["source_hashes"]) == {"train", "val", "test"}
    original_manifest = manifest_path.read_text()
    manifest["source_hashes"]["train"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="train source dataset"):
        make_datamodule()
    manifest_path.write_text(original_manifest)

    datamodule = make_datamodule()
    batch = next(iter(datamodule.train_dataloader()))
    assert batch.rot.dtype == batch.pos.dtype == batch.cell.dtype == torch.float64
    kwargs32 = dict(kwargs, floating_point_precision="32-true")
    datamodule32 = BlockDataModule(
        StructureDataset(file_path=str(data_dir / "train.lmdb"), **kwargs32),
        StructureDataset(file_path=str(data_dir / "val.lmdb"), **kwargs32),
        StructureDataset(file_path=str(data_dir / "test.lmdb"), **kwargs32),
        block_dir=str(block_dir), batch_size=2, num_workers=0)
    batch32 = next(iter(datamodule32.train_dataloader()))
    assert batch32.rot.dtype == batch32.pos.dtype == batch32.cell.dtype == torch.float32
    assert batch32.target_frac.dtype == batch32.template_offsets.dtype == torch.float32
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        encoder32 = BlockCSPNet(hidden_dim=32, latent_dim=16, num_layers=1, num_freqs=8, am_hidden_dim=8)
        encoder32.set_cn_mask(torch.from_numpy(datamodule32.library.cn_valid))
        model32 = Model(encoder=encoder32, head=PassThrough(), time_embedder=SinusoidalTimeEmbeddings(16))
        sampler32 = BlockSampler(UniformPositionDistribution(), MirrorCell(), cn_mode="oracle")
        si32 = CoupledBlockInterpolants(cn_mode="oracle", integration_time_steps=3, enable_progress_bar=False)
        losses32 = si32.losses(
            model32, torch.full((len(batch32.n_atoms),), 0.4), sampler32.sample_p_0(batch32), batch32)
        assert all(torch.isfinite(value) for value in losses32.values())
    finally:
        torch.set_default_dtype(previous)
    encoder = BlockCSPNet(hidden_dim=32, latent_dim=16, num_layers=1, num_freqs=8, am_hidden_dim=8)
    encoder.set_cn_mask(torch.from_numpy(datamodule.library.cn_valid))
    model = Model(encoder=encoder, head=PassThrough(), time_embedder=SinusoidalTimeEmbeddings(16))
    sampler = BlockSampler(UniformPositionDistribution(), MirrorCell(), cn_mode="oracle")
    si = CoupledBlockInterpolants(cn_mode="oracle", integration_time_steps=3, enable_progress_bar=False)
    losses = si.losses(model, torch.full((len(batch.n_atoms),), 0.4), sampler.sample_p_0(batch), batch)
    assert all(torch.isfinite(value) for value in losses.values())
    reference, generated, diagnostics = _read_out_structure(batch, batch, 0, datamodule.library)
    assert sorted(generated.get_atomic_numbers().tolist()) == sorted(reference.get_atomic_numbers().tolist())
    assert diagnostics["shortfall"] == 0
