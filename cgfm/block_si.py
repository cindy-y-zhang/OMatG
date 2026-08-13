"""
Coupled CN / SE(3) / lattice interpolants that evaluate the network once per batch and once per Euler step.

OMatG's ``StochasticInterpolants`` re-forwards the encoder for every field. A 210-step validation of a four-field
block model would then spend most of a 30-hour allocation on repeated identical forwards. This collection interpolates
every field, calls the model once, and splits the prediction into translation, rotation, cell and CN losses.

Rotations move on the geodesic. The regression target is the body-frame velocity scored in vertex space, which is
blind to a continuous stabiliser and, after ``canonicalise`` against the sample at time zero, a point rather than a
set. Discrete CN uses a 13-class masked flow; oracle mode replaces that interpolant with the identity.
"""

from typing import Callable, Dict, List, Tuple, Union
import torch
import torch.nn.functional as functional
from torch.distributions import Categorical
from torch_geometric.data import Data
from torch_scatter import scatter_mean
from omg.globals import BIG_TIME, SMALL_TIME
from omg.si.corrector import PeriodicBoundaryConditionsCorrector
from omg.si.discrete_flow_matching_mask import DiscreteFlowMatchingMask
from omg.si.interpolants import LinearInterpolant, PeriodicLinearInterpolant
from omg.si.single_stochastic_interpolant_identity import SingleStochasticInterpolantIdentity
from omg.si.stochastic_interpolants import StochasticInterpolants
from omg.utils import DataField, reshape_t
from tqdm import trange
from .blocks import CN_CLASSES
from .so3 import canonicalise, exp_map, geodesic, hat, project, velocity, vertex_distance


class DiscreteFlowMatchingMaskN(DiscreteFlowMatchingMask):
    """
    Masked discrete flow matching over a configurable number of classes rather than ``MAX_ATOM_NUM``.

    :param n_classes:
        Number of real classes, excluding the mask token 0.
        Defaults to CN_CLASSES.
    :type n_classes: int
    :param noise:
        Parameter scaling the noise added during integration.
        Defaults to 0.0.
    :type noise: float
    """

    def __init__(self, n_classes: int = CN_CLASSES, noise: float = 0.0) -> None:
        """Construct the interpolant."""
        super().__init__(noise=noise)
        self._n_classes = n_classes

    def loss(self, model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
             t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor,
             batch_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the cross-entropy loss against the unmasked class indices."""
        assert x_0.shape == x_1.shape
        assert torch.all(x_0 == self._mask_index)
        assert torch.all(x_1 != self._mask_index)
        pred = model_function(x_t)[0]
        assert pred.shape == (x_0.shape[0], self._n_classes)
        return {"loss": functional.cross_entropy(pred, x_1 - 1)}

    def integrate(self, model_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  x_t: torch.Tensor, time: torch.Tensor, time_step: torch.Tensor,
                  batch_indices: torch.Tensor, final_step: bool = False) -> torch.Tensor:
        """Take one discrete unmasking step from the predicted class distribution."""
        x_1_probs = functional.softmax(model_function(time, x_t)[0], dim=-1)
        x_1_probs = x_1_probs.reshape((-1, self._n_classes))
        x_1 = Categorical(x_1_probs).sample() + 1
        will_unmask = torch.rand(x_t.shape, device=x_t.device) < (time_step * (1.0 + self._noise * time) / (1.0 - time))
        will_unmask = will_unmask * (x_t == self._mask_index)
        if final_step:
            will_unmask = (x_t == self._mask_index)
        x_t = x_t.clone()
        x_t[will_unmask] = x_1[will_unmask]
        if self._noise > 0.0 and not final_step:
            will_mask = torch.rand(x_t.shape, device=x_t.device) < time_step * self._noise
            will_mask = will_mask * (x_t != self._mask_index)
            x_t[will_mask] = self._mask_index
        return x_t


def _broadcast_time(t: torch.Tensor, n_atoms: torch.Tensor, data_field: DataField) -> torch.Tensor:
    """
    Reshape a per-structure time tensor to a data field, using a per-block scalar for rotations.

    :param t:
        Times of shape (batch,).
    :type t: torch.Tensor
    :param n_atoms:
        Block counts of shape (batch,).
    :type n_atoms: torch.Tensor
    :param data_field:
        Field whose tensor the times must match, except ``rot`` which uses a per-block scalar.
    :type data_field: DataField

    :return:
        Times broadcast against that field.
    :rtype: torch.Tensor
    """
    if data_field == DataField.rot:
        return t.repeat_interleave(n_atoms)
    return reshape_t(t, n_atoms, data_field)


def consensus_positions(frac_pos: torch.Tensor, rotations: torch.Tensor, cell: torch.Tensor,
                        template_offsets: torch.Tensor, template_mask: torch.Tensor, vote_atom: torch.Tensor,
                        centre_atom: torch.Tensor, target_frac: torch.Tensor, target_mask: torch.Tensor,
                        n_target_atoms: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Average template votes onto the known sharing map, returning predicted and true Cartesian atom positions.

    :param frac_pos:
        Fractional block translations of shape (sum M, 3).
    :type frac_pos: torch.Tensor
    :param rotations:
        Block orientations of shape (sum M, 3, 3).
    :type rotations: torch.Tensor
    :param cell:
        Lattices of shape (B, 3, 3).
    :type cell: torch.Tensor
    :param template_offsets:
        Padded template vertices of shape (sum M, V, 3).
    :type template_offsets: torch.Tensor
    :param template_mask:
        Whether each slot holds a vertex, of shape (sum M, V).
    :type template_mask: torch.Tensor
    :param vote_atom:
        Local atom index of every slot, or -1, of shape (sum M, V).
    :type vote_atom: torch.Tensor
    :param centre_atom:
        Local atom index placed directly by each block translation, of shape (sum M,).
    :type centre_atom: torch.Tensor
    :param target_frac:
        Padded true fractional coordinates of shape (B, A, 3).
    :type target_frac: torch.Tensor
    :param target_mask:
        Whether each padded atom slot is used, of shape (B, A).
    :type target_mask: torch.Tensor
    :param n_target_atoms:
        Original atom count of every structure, of shape (B,).
    :type n_target_atoms: torch.Tensor
    :param batch:
        Structure index of every block, of shape (sum M,).
    :type batch: torch.Tensor

    :return:
        Predicted Cartesian coordinates of every real atom, concatenated, and the true Cartesian coordinates.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    block_lattices = cell[batch]
    centres = torch.matmul(frac_pos.unsqueeze(-2), block_lattices).squeeze(-2)
    vertices = centres.unsqueeze(1) + torch.matmul(template_offsets, rotations.transpose(-1, -2))
    valid = template_mask & (vote_atom >= 0)
    atom_offset = torch.zeros(len(n_target_atoms) + 1, dtype=torch.long, device=frac_pos.device)
    atom_offset[1:] = torch.cumsum(n_target_atoms, dim=0)
    global_atom = vote_atom + atom_offset[batch][:, None]
    n_total = int(n_target_atoms.sum())
    residual_sum = torch.zeros((n_total, 3), dtype=frac_pos.dtype, device=frac_pos.device)
    counts = torch.zeros((n_total, 1), dtype=frac_pos.dtype, device=frac_pos.device)
    target_cart = torch.matmul(target_frac, cell)

    def minimum_image_residual(predicted: torch.Tensor, expected: torch.Tensor,
                               lattices: torch.Tensor) -> torch.Tensor:
        """Return differentiable shortest-image Cartesian residuals in the supplied lattices."""
        delta = predicted - expected
        fractional = torch.matmul(delta.unsqueeze(-2), torch.linalg.inv(lattices)).squeeze(-2)
        fractional = fractional - torch.round(fractional).detach()
        return torch.matmul(fractional.unsqueeze(-2), lattices).squeeze(-2)

    centre_global = centre_atom + atom_offset[batch]
    centre_true = target_cart[batch, centre_atom]
    centre_residual = minimum_image_residual(centres, centre_true, block_lattices)
    residual_sum.index_add_(0, centre_global, centre_residual)
    counts.index_add_(
        0, centre_global, torch.ones((len(centre_global), 1), dtype=frac_pos.dtype, device=frac_pos.device))

    flat_atom = global_atom[valid]
    flat_vertex = vertices[valid]
    slot_batch = batch[:, None].expand_as(vote_atom)[valid]
    flat_local_atom = vote_atom[valid]
    vertex_true = target_cart[slot_batch, flat_local_atom]
    vertex_residual = minimum_image_residual(flat_vertex, vertex_true, cell[slot_batch])
    residual_sum.index_add_(0, flat_atom, vertex_residual)
    counts.index_add_(
        0, flat_atom, torch.ones((len(flat_atom), 1), dtype=frac_pos.dtype, device=frac_pos.device))

    true = target_cart.reshape(-1, 3)[target_mask.reshape(-1)]
    predicted = true + residual_sum / counts.clamp(min=1.0)
    return predicted, true


class CoupledBlockInterpolants(StochasticInterpolants):
    """
    Joint interpolant over centre species, coordination number, translation, rotation and lattice.

    :param cn_mode:
        ``"oracle"`` copies the target CN into the base state; ``"joint"`` masks it.
    :type cn_mode: str
    :param integration_time_steps:
        Number of Euler steps used at inference.
    :type integration_time_steps: int
    :param pos_annealing_factor:
        Multiplier of ``(1 + factor * t)`` applied to predicted translation velocities at inference.
        Defaults to 0.0.
    :type pos_annealing_factor: float
    :param rot_annealing_factor:
        The same multiplier for body-frame rotation velocities.
        Defaults to 0.0.
    :type rot_annealing_factor: float
    :param cell_annealing_factor:
        The same multiplier for lattice velocities.
        Defaults to 0.0.
    :type cell_annealing_factor: float
    :param discrete_noise:
        Noise parameter of the masked CN flow.
        Defaults to 0.0.
    :type discrete_noise: float
    :param enable_progress_bar:
        Whether to show an integration progress bar.
        Defaults to True.
    :type enable_progress_bar: bool

    :raises ValueError:
        If ``cn_mode`` is not ``oracle`` or ``joint``, or the step count is not positive.
    """

    def __init__(self, cn_mode: str = "joint", integration_time_steps: int = 210,
                 pos_annealing_factor: float = 0.0, rot_annealing_factor: float = 0.0,
                 cell_annealing_factor: float = 0.0, discrete_noise: float = 0.0,
                 enable_progress_bar: bool = True) -> None:
        """Construct the coupled interpolants."""
        if cn_mode not in ("oracle", "joint"):
            raise ValueError(f"Unknown cn_mode {cn_mode!r}.")
        if integration_time_steps <= 0:
            raise ValueError("The number of integration time steps must be positive.")
        self.cn_mode = cn_mode
        self._integration_time_steps = integration_time_steps
        self._pos_annealing_factor = pos_annealing_factor
        self._rot_annealing_factor = rot_annealing_factor
        self._cell_annealing_factor = cell_annealing_factor
        self._enable_progress_bar = enable_progress_bar
        self._species = SingleStochasticInterpolantIdentity()
        self._block_type = (SingleStochasticInterpolantIdentity() if cn_mode == "oracle"
                            else DiscreteFlowMatchingMaskN(n_classes=CN_CLASSES, noise=discrete_noise))
        self._pos = PeriodicLinearInterpolant()
        self._cell = LinearInterpolant()
        self._pos_corrector = PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)
        self._interpolants = {
            "species": self._species,
            "block_type": self._block_type,
        }
        super().__init__(
            stochastic_interpolants=(self._species, self._block_type),
            data_fields=("species", "block_type"),
            integration_time_steps=integration_time_steps,
            enable_progress_bar=enable_progress_bar,
        )

    def loss_keys(self) -> List[str]:
        """
        Return the loss keys this collection logs.

        :return:
            Loss keys.
        :rtype: List[str]
        """
        keys = [f"species_{key}" for key in self._species.loss_keys()]
        keys.extend(f"block_type_{key}" for key in self._block_type.loss_keys())
        keys.extend(["pos_loss_b", "rot_loss_b", "cell_loss_b"])
        return keys

    def get_stochastic_interpolant(self, data_field: str) -> object:
        """
        Return the interpolant associated with a named field.

        :param data_field:
            Field name.
        :type data_field: str

        :return:
            The interpolant.
        :rtype: object

        :raises ValueError:
            If the field is unknown.
        """
        name = data_field.lower()
        if name == "species":
            return self._species
        if name == "block_type":
            return self._block_type
        if name in ("pos", "cell", "rot"):
            return self
        raise ValueError(f"Unknown data field {data_field!r}.")

    def get_integration_time_steps(self) -> int:
        """Return the number of Euler steps."""
        return self._integration_time_steps

    def _canonical_target(self, x_0: Data, x_1: Data) -> torch.Tensor:
        """Replace each target rotation by the stabiliser element nearest the sample at time zero."""
        return canonicalise(x_1.rot, x_0.rot, x_1.stabilizer)

    def _interpolate(self, t: torch.Tensor, x_0: Data, x_1: Data) -> Data:
        """Interpolate every field, canonicalising rotations against the base sample."""
        n_atoms = x_0.n_atoms
        x_t = x_0.clone()
        target_rot = self._canonical_target(x_0, x_1)
        t_species = _broadcast_time(t, n_atoms, DataField.species)
        t_type = _broadcast_time(t, n_atoms, DataField.block_type)
        t_pos = _broadcast_time(t, n_atoms, DataField.pos)
        t_cell = _broadcast_time(t, n_atoms, DataField.cell)
        t_rot = _broadcast_time(t, n_atoms, DataField.rot)
        x_t.species, _ = self._species.interpolate(t_species, x_0.species, x_1.species, x_0.batch)
        x_t.block_type, _ = self._block_type.interpolate(t_type, x_0.block_type, x_1.block_type, x_0.batch)
        x_t.pos = self._pos.interpolate(t_pos, x_0.pos, x_1.pos)
        x_t.cell = self._cell.interpolate(t_cell, x_0.cell, x_1.cell)
        x_t.rot = geodesic(x_0.rot, target_rot, t_rot)
        x_t.rot_target = target_rot
        x_t.t_rot = t_rot
        x_t.t_pos = t_pos
        x_t.t_cell = t_cell
        return x_t

    def losses(self, model_function: Callable[[Data, torch.Tensor], Data], t: torch.Tensor, x_0: Data,
               x_1: Data) -> dict[str, torch.Tensor]:
        """
        Evaluate the network once and return every field's loss.

        :param model_function:
            Model returning velocity and CN fields given ``x_t`` and times ``t``.
        :type model_function: Callable
        :param t:
            Times of shape (batch,).
        :type t: torch.Tensor
        :param x_0:
            Base sample.
        :type x_0: torch_geometric.data.Data
        :param x_1:
            Data sample.
        :type x_1: torch_geometric.data.Data

        :return:
            Mapping from loss key to scalar.
        :rtype: dict[str, torch.Tensor]
        """
        losses, _, _ = self.losses_with_prediction(model_function, t, x_0, x_1)
        return losses

    def losses_with_prediction(
            self, model_function: Callable[[Data, torch.Tensor], Data], t: torch.Tensor, x_0: Data,
            x_1: Data) -> tuple[dict[str, torch.Tensor], Data, Data]:
        """
        Return losses together with the shared interpolated state and network prediction.

        The extra values let endpoint objectives reuse the same expensive encoder evaluation instead of forwarding the
        same training batch a second time.
        """
        x_t = self._interpolate(t, x_0, x_1)
        prediction = model_function(x_t, t)
        losses = {}
        species_t = t.repeat_interleave(x_0.n_atoms)
        for key in self._species.loss_keys():
            losses[f"species_{key}"] = prediction.pos_b.sum() * 0.0
        for key, value in self._block_type.loss(
                lambda _x: (prediction.block_type_b, prediction.block_type_b),
                species_t, x_0.block_type, x_1.block_type, x_t.block_type,
                torch.zeros_like(x_0.block_type), x_0.batch).items():
            losses[f"block_type_{key}"] = value

        expected_pos = self._pos.interpolate_derivative(x_t.t_pos, x_0.pos, x_1.pos)
        mean_velocity = torch.index_select(scatter_mean(expected_pos, x_0.batch, dim=0), 0, x_0.batch)
        expected_pos = expected_pos - mean_velocity
        pred_pos = prediction.pos_b
        losses["pos_loss_b"] = torch.mean(pred_pos ** 2) - 2.0 * torch.mean(pred_pos * expected_pos)

        expected_cell = self._cell.interpolate_derivative(x_t.t_cell, x_0.cell, x_1.cell)
        pred_cell = prediction.cell_b
        losses["cell_loss_b"] = torch.mean(pred_cell ** 2) - 2.0 * torch.mean(pred_cell * expected_cell)

        true_body = velocity(x_t.rot, x_t.rot_target, x_t.t_rot)
        losses["rot_loss_b"] = vertex_distance(
            hat(prediction.rot_b), hat(true_body), x_1.template_offsets, x_1.template_mask).mean()
        return losses, x_t, prediction

    def integrate(self, x_0: Data, model_function: Callable[[Data, torch.Tensor], Data],
                  save_intermediate: bool = False) -> Union[Data, Tuple[Data, List[Data]]]:
        """
        Integrate every field with one network evaluation per Euler step.

        :param x_0:
            Base sample.
        :type x_0: torch_geometric.data.Data
        :param model_function:
            Model returning velocity and CN fields.
        :type model_function: Callable
        :param save_intermediate:
            If True, also return the trajectory.
            Defaults to False.
        :type save_intermediate: bool

        :return:
            The sample at time one, and optionally the intermediate states.
        :rtype: torch_geometric.data.Data
        """
        times = torch.linspace(
            SMALL_TIME, BIG_TIME, self._integration_time_steps + 1, device=x_0.pos.device)
        x_t = x_0.clone()
        intermediates = [x_t.clone()] if save_intermediate else None
        for t_index in trange(1, len(times), desc="Integrating", disable=not self._enable_progress_bar):
            time = times[t_index - 1]
            step = times[t_index] - time
            batch_time = time.repeat(len(x_t.n_atoms))
            prediction = model_function(x_t, batch_time)
            pos_scale = 1.0 + self._pos_annealing_factor * time
            rot_scale = 1.0 + self._rot_annealing_factor * time
            cell_scale = 1.0 + self._cell_annealing_factor * time
            x_t.pos = self._pos_corrector.correct(x_t.pos + step * pos_scale * prediction.pos_b)
            x_t.cell = x_t.cell + step * cell_scale * prediction.cell_b
            x_t.rot = project(x_t.rot @ exp_map(step * rot_scale * prediction.rot_b))
            type_model = lambda _time, _x: (prediction.block_type_b, prediction.block_type_b)
            if self.cn_mode == "joint":
                x_t.block_type = self._block_type.integrate(
                    type_model, x_t.block_type, time, step, x_t.batch,
                    final_step=t_index == self._integration_time_steps)
            else:
                x_t.block_type = self._block_type.integrate(
                    type_model, x_t.block_type, time, step, x_t.batch)
            if save_intermediate:
                intermediates.append(x_t.clone())
        if save_intermediate:
            return x_t, intermediates
        return x_t
