"""
Lightning module for the rigid-block model.

Validation integrates block states, looks up train-only templates, runs assignment-free readout, and then calls the
same structure matcher the atomwise baseline uses. Training losses over poses are never compared to atomwise losses.
"""

from pathlib import Path
from typing import Any, Optional
import time
import numpy as np
import torch
from ase import Atoms
from ase.io import write
from pymatgen.core import Element
from omg.analysis import ValidAtoms, match_rmsds
from omg.datamodule import OMGData
from omg.omg_lightning import OMGLightning
from .block_si import consensus_positions
from .blocks import decode_coordination
from .periodic import Cell
from .readout import Placement, derive_templates, read_out
from .so3 import angle_between, canonicalise, vertex_distance


class BlockLightning(OMGLightning):
    """
    OMGLightning whose generated structures are recovered from block poses by assignment-free readout.

    :param consensus_weight:
        Weight of the differentiable consensus endpoint loss. Chosen on the 100-structure overfit gate and then frozen.
        Defaults to 0.0.
    :type consensus_weight: float
    """

    def __init__(self, *args: Any, consensus_weight: float = 0.0, **kwargs: Any) -> None:
        """Construct the Lightning module."""
        super().__init__(*args, **kwargs)
        if consensus_weight < 0.0:
            raise ValueError("consensus_weight must be non-negative.")
        self.consensus_weight = consensus_weight
        self._block_diagnostics: list[dict[str, float]] = []

    def setup(self, stage: Optional[str] = None) -> None:
        """Install the train-only CN mask on the encoder once the datamodule is available."""
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "library"):
            return
        library = datamodule.library
        encoder = self.model.encoder
        if hasattr(encoder, "set_cn_mask"):
            encoder.set_cn_mask(torch.from_numpy(library.cn_valid))
        self.template_library = library

    def training_step(self, x_1: OMGData) -> torch.Tensor:
        """Compute the coupled pose losses and, if configured, the consensus endpoint loss."""
        x_0 = self.sampler.sample_p_0(x_1).to(self.device)
        t = self.time_sampler(len(x_1.n_atoms)).to(self.device)
        if self.consensus_weight > 0.0 and hasattr(self.si, "losses_with_prediction"):
            losses, x_t, prediction = self.si.losses_with_prediction(self.model, t, x_0, x_1)
        else:
            losses = self.si.losses(self.model, t, x_0, x_1)
            x_t = prediction = None
        total_loss = torch.tensor(0.0, device=self.device)
        for loss_key in losses:
            losses[loss_key] = self._relative_si_costs[loss_key] * losses[loss_key]
            total_loss += losses[loss_key]
        if self.consensus_weight > 0.0:
            consensus = self._consensus_loss(x_0, x_1, t, x_t=x_t, prediction=prediction)
            losses["consensus_loss"] = self.consensus_weight * consensus
            total_loss = total_loss + losses["consensus_loss"]
        losses["loss_total"] = total_loss
        self.log_dict(losses, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                      batch_size=len(x_1.n_atoms))
        return total_loss

    def _consensus_loss(self, x_0: OMGData, x_1: OMGData, t: torch.Tensor,
                        x_t: Optional[OMGData] = None, prediction: Optional[OMGData] = None) -> torch.Tensor:
        """Average template votes at the predicted endpoint onto the known sharing map."""
        if x_t is None or prediction is None:
            x_t, _ = _interpolate_for_consensus(self.si, t, x_0, x_1)
            prediction = self.model(x_t, t)
        t_pos = t.repeat_interleave(x_1.n_atoms).unsqueeze(-1)
        t_rot = t.repeat_interleave(x_1.n_atoms)
        from omg.si.corrector import PeriodicBoundaryConditionsCorrector
        from .so3 import endpoint
        wrap = PeriodicBoundaryConditionsCorrector(0.0, 1.0)
        pos_1 = wrap.correct(x_t.pos + (1.0 - t_pos) * prediction.pos_b)
        rot_1 = endpoint(x_t.rot, prediction.rot_b, t_rot)
        cell_1 = x_t.cell + (1.0 - t[:, None, None]) * prediction.cell_b
        predicted, true = consensus_positions(
            pos_1, rot_1, cell_1, x_1.template_offsets, x_1.template_mask, x_1.vote_atom, x_1.centre_atom,
            x_1.target_frac, x_1.target_mask, x_1.n_target_atoms, x_1.batch)
        residual = predicted - true
        return residual.pow(2).mean()

    def on_validation_epoch_start(self) -> None:
        """Reset generated structures and per-structure diagnostics."""
        super().on_validation_epoch_start()
        self._block_diagnostics = []

    def validation_step(self, x_1: OMGData) -> torch.Tensor:
        """Integrate block states, read them out to atoms, and log the pose loss."""
        batch_size = len(x_1.n_atoms)
        x_0 = self.sampler.sample_p_0(x_1).to(self.device)
        x_1_cpu = x_1.clone().to("cpu")
        gen = self.si.integrate(x_0, self.model, save_intermediate=False).to("cpu")
        library = getattr(self, "template_library", None)
        if library is None:
            datamodule = getattr(self.trainer, "datamodule", None)
            library = datamodule.library if datamodule is not None else None
        for index in range(batch_size):
            reference, generated, diagnostics = _read_out_structure(
                gen, x_1_cpu, index, library, compute_match=False)
            self.reference_atoms.append(reference)
            self.generated_atoms.append(generated)
            self._block_diagnostics.append(diagnostics)

        t = self.time_sampler(len(x_1.n_atoms)).to(self.device)
        losses = self.si.losses(self.model, t, x_0, x_1)
        total_loss = torch.tensor(0.0, device=self.device)
        logged = {}
        for loss_key in list(losses):
            logged[f"val_{loss_key}"] = self._relative_si_costs[loss_key] * losses[loss_key]
            total_loss = total_loss + logged[f"val_{loss_key}"]
        logged["val_loss_total"] = total_loss
        self.log_dict(logged, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return total_loss

    def on_validation_epoch_end(self) -> None:
        """Match once in parallel, then log aggregate and block-specific diagnostics."""
        if not self._block_diagnostics:
            return
        generated = ValidAtoms.get_valid_atoms(
            self.generated_atoms, desc="Validating generated structures", skip_validation=True, number_cpus=1)
        reference = ValidAtoms.get_valid_atoms(
            self.reference_atoms, desc="Validating reference structures", skip_validation=True, number_cpus=1)
        match_rate, mean_rmsd, _, _, rmsds, _, corr_rmsd, _ = match_rmsds(
            generated, reference, ltol=0.3, stol=0.5, angle_tol=10.0, number_cpus=self.number_cpus,
            enable_progress_bar=True)
        self.log("match_rate", float(match_rate), sync_dist=True)
        self.log("mean_rmsd", float(mean_rmsd), sync_dist=True)
        self.log("corr_rmsd", float(corr_rmsd), sync_dist=True)

        keys = ("cn_accuracy", "all_cns_correct", "translation_error", "rotation_error", "vertex_error",
                "votes_per_atom", "shortfall", "fallback_use")
        for key in keys:
            values = [row[key] for row in self._block_diagnostics]
            self.log(key, float(np.mean(values)), sync_dist=True)
        for name, predicate in (("single_anion", True), ("mixed_anion", False)):
            subset = [rmsd is not None for rmsd, row in zip(rmsds, self._block_diagnostics)
                      if row["single_anion"] is predicate]
            if subset:
                self.log(f"match_rate_{name}", float(np.mean(subset)), sync_dist=True)
        if self.store_validation_structures_path is not None:
            filename = Path(self.store_validation_structures_path)
            epoch_filename = filename.with_stem(
                f"{filename.stem}_epoch_{self.current_epoch:04d}_step_{self.global_step:06d}")
            epoch_filename_ref = epoch_filename.with_stem(epoch_filename.stem + "_ref")
            if epoch_filename.exists():
                epoch_filename.unlink()
            if epoch_filename_ref.exists():
                epoch_filename_ref.unlink()
            write(epoch_filename, self.generated_atoms, append=True)
            write(epoch_filename_ref, self.reference_atoms, append=True)

    def predict_step(self, x: OMGData) -> OMGData:
        """Integrate, read out to atoms, and write XYZ files of the recovered crystals."""
        x_0 = self.sampler.sample_p_0(x).to(self.device)
        gen = self.si.integrate(x_0, self.model, save_intermediate=False).to("cpu")
        x_cpu = x.clone().to("cpu")
        library = getattr(self, "template_library", None)
        if library is None:
            datamodule = getattr(self.trainer, "datamodule", None)
            library = datamodule.library if datamodule is not None else None
        atoms = []
        for index in range(len(x_cpu.n_atoms)):
            _, generated, _ = _read_out_structure(gen, x_cpu, index, library, compute_match=False)
            atoms.append(generated)
        filename = (Path(self.generation_xyz_filename) if self.generation_xyz_filename is not None
                    else Path(f"{time.strftime('%Y%m%d-%H%M%S')}.xyz"))
        filename.parent.mkdir(parents=True, exist_ok=True)
        from ase.io import write
        write(filename, atoms, append=True)
        return gen


def _interpolate_for_consensus(si, t, x_0, x_1):
    """Interpolate to ``x_t`` using the coupled interpolant's private helper when available."""
    if hasattr(si, "_interpolate"):
        return si._interpolate(t, x_0, x_1), None
    raise TypeError("Consensus loss requires CoupledBlockInterpolants.")


def _structure_slice(batch: OMGData, index: int) -> slice:
    """Return the node slice of one structure in a batch."""
    return slice(int(batch.ptr[index]), int(batch.ptr[index + 1]))


def _read_out_structure(gen: OMGData, target: OMGData, index: int, library,
                        compute_match: bool = True) -> tuple[Atoms, Atoms, dict[str, float]]:
    """
    Recover one crystal from generated block poses and score block-level diagnostics against the target.

    :param gen:
        Generated block batch on CPU.
    :type gen: OMGData
    :param target:
        Target block batch on CPU.
    :type target: OMGData
    :param index:
        Structure index within the batch.
    :type index: int
    :param library:
        Train-only template library, or None to use the templates stored on the target graph.
    :type library: object
    :param compute_match:
        Whether to run a structure matcher immediately. Validation and prediction disable this so matching can be
        parallelised once per epoch, or omitted when the result is unused.
        Defaults to True.
    :type compute_match: bool

    :return:
        Reference atoms, generated atoms, and a diagnostics dictionary.
    :rtype: tuple[ase.Atoms, ase.Atoms, dict[str, float]]
    """
    nodes = _structure_slice(gen, index)
    cell = gen.cell[index].detach().numpy()
    centre_numbers = gen.species[nodes].detach().numpy()
    frac = gen.pos[nodes].detach().numpy()
    rotations = gen.rot[nodes].detach().numpy()
    block_type = gen.block_type[nodes].detach().numpy()
    cartesian = frac @ cell
    coordinations = []
    type_keys = []
    fallback = 0
    for number, token in zip(centre_numbers, block_type):
        token = int(token)
        if token == 0:
            raise RuntimeError("Block integration left a masked coordination token at readout.")
        coordination = decode_coordination(token)
        coordinations.append(coordination)
        key = (Element.from_Z(int(number)).symbol, coordination)
        type_keys.append(key)
        if library is not None and key not in library.coarse:
            fallback += 1
    if library is not None:
        target_numbers = target.target_numbers[index][target.target_mask[index]].detach().numpy()
        templates = derive_templates(target_numbers, library.fine, library.coarse)
    else:
        target_numbers = target.target_numbers[index][target.target_mask[index]].detach().numpy()
        templates = {}
    placement = Placement(lattice=cell, centre_numbers=centre_numbers, centre_coords=cartesian,
                          rotations=rotations, type_keys=tuple(type_keys))
    result = read_out(placement, templates, target_numbers,
                      fallback=library.coarse if library is not None else None)
    generated = Atoms(numbers=result.numbers, positions=result.coords, cell=cell, pbc=(1, 1, 1))
    true_frac = target.target_frac[index][target.target_mask[index]].detach().numpy()
    true_numbers = target.target_numbers[index][target.target_mask[index]].detach().numpy()
    reference = Atoms(numbers=true_numbers, scaled_positions=true_frac, cell=target.cell[index].detach().numpy(),
                      pbc=(1, 1, 1))

    true_type = target.block_type[nodes].detach().numpy()
    cn_correct = block_type == true_type
    periodic = Cell.of(cell)
    translation = float(np.linalg.norm(periodic.minimum_image(cartesian - (target.pos[nodes].detach().numpy() @ cell)),
                                       axis=-1).mean()) if len(cartesian) else 0.0
    true_rot = target.rot[nodes]
    group = target.stabilizer[nodes]
    chosen = canonicalise(true_rot, gen.rot[nodes], group)
    rotation_error = float(angle_between(gen.rot[nodes], chosen).mean()) if len(cartesian) else 0.0
    vertex_error = float(vertex_distance(
        gen.rot[nodes], chosen, target.template_offsets[nodes], target.template_mask[nodes]).mean())
    diagnostics = {
        "cn_accuracy": float(cn_correct.mean()) if len(cn_correct) else 1.0,
        "all_cns_correct": float(bool(cn_correct.all())) if len(cn_correct) else 1.0,
        "translation_error": translation,
        "rotation_error": rotation_error,
        "vertex_error": vertex_error,
        "votes_per_atom": float(result.votes_per_atom),
        "shortfall": float(result.shortfall),
        "fallback_use": float(fallback) / max(len(type_keys), 1),
        "single_anion": bool(target.single_anion[index]),
    }
    if compute_match:
        from pymatgen.analysis.structure_matcher import StructureMatcher
        from pymatgen.core import Lattice, Structure as PymatgenStructure
        matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10.0)
        reference_structure = PymatgenStructure(
            Lattice(reference.cell), reference.get_chemical_symbols(), reference.get_positions(),
            coords_are_cartesian=True)
        generated_structure = PymatgenStructure(
            Lattice(generated.cell), generated.get_chemical_symbols(), generated.get_positions(),
            coords_are_cartesian=True)
        diagnostics["matched"] = float(matcher.fit(reference_structure, generated_structure) is True)
    return reference, generated, diagnostics
