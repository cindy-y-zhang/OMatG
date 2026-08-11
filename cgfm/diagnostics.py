"""
Diagnostics for the coarse-to-fine arms.

Minimising the flow-matching loss over the interpolant rewards a learned grouping for making the conditional velocity
easy to predict, not for making the generative path good. The cheapest way to do that is to shrink the within-group
residual, and the argmin of that residual at a fixed number of groups is plain geometric clustering. The learned arm
may therefore reproduce the k-medoids baseline instead of discovering anything.

That outcome is worth reporting, but only if it can be told apart from success, so these statistics are recorded during
training:

- the group-size distribution, singleton fraction and spatial extent of the groups;
- the fine-energy fraction ||(I - S) d||^2 / ||d||^2, which is exactly what the shortcut shrinks;
- the adjusted Rand index of the learned partition against both precomputed partitions: against periodic k-medoids it
  detects collapse onto geometric clustering directly, and against the CrystalNN coordination shells it measures the
  effect the project exists to test;
- the assignment entropy and temperature, which say how close the soft assignment is to a genuine partition.

Reported training losses are not comparable across arms, because different probability paths have different conditional
velocity distributions. Model selection is therefore on validation match rate, never on loss.
"""

from typing import Any, Dict, Optional
import lightning
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from .graph import atom_group_extents, dense_positions, pair_distances
from .grouping import REFERENCE_METHODS, reference_field


def _pair_count(counts: torch.Tensor) -> torch.Tensor:
    """
    Count unordered pairs within groups of the given sizes.

    :param counts:
        Group sizes.
    :type counts: torch.Tensor

    :return:
        Number of unordered pairs, elementwise.
    :rtype: torch.Tensor
    """
    return counts * (counts - 1.0) / 2.0


def adjusted_rand_index(labels_a: torch.Tensor, labels_b: torch.Tensor, batch: torch.Tensor,
                        num_structures: int) -> torch.Tensor:
    """
    Compute the adjusted Rand index between two partitions of every structure, averaged over the batch.

    :param labels_a:
        Structure-local labels of the first partition, of shape (N,).
    :type labels_a: torch.Tensor
    :param labels_b:
        Structure-local labels of the second partition, of shape (N,).
    :type labels_b: torch.Tensor
    :param batch:
        Structure index of every atom, of shape (N,).
    :type batch: torch.Tensor
    :param num_structures:
        Number of structures B in the batch.
    :type num_structures: int

    :return:
        Scalar mean adjusted Rand index.
    :rtype: torch.Tensor
    """
    size_a = int(labels_a.max()) + 1 if labels_a.numel() > 0 else 1
    size_b = int(labels_b.max()) + 1 if labels_b.numel() > 0 else 1
    joint = batch * (size_a * size_b) + labels_a * size_b + labels_b
    contingency = torch.bincount(joint, minlength=num_structures * size_a * size_b).to(torch.float64)
    contingency = contingency.reshape(num_structures, size_a, size_b)

    pairs_joint = _pair_count(contingency).sum(dim=(1, 2))
    pairs_a = _pair_count(contingency.sum(dim=2)).sum(dim=1)
    pairs_b = _pair_count(contingency.sum(dim=1)).sum(dim=1)
    total = contingency.sum(dim=(1, 2))
    pairs_total = _pair_count(total)

    expected = pairs_a * pairs_b / pairs_total.clamp_min(1.0)
    maximum = 0.5 * (pairs_a + pairs_b)
    denominator = maximum - expected
    # A degenerate denominator means both partitions are trivial in the same way, which is perfect agreement.
    score = torch.where(denominator.abs() < 1.0e-12, torch.ones_like(denominator),
                        (pairs_joint - expected) / denominator.masked_fill(denominator.abs() < 1.0e-12, 1.0))
    return score.mean()


def group_statistics(assignment: torch.Tensor, batch: torch.Tensor, num_structures: int,
                     x_1: Optional[Data] = None) -> Dict[str, float]:
    """
    Summarise a batch of atom-to-group assignments.

    :param assignment:
        Row-stochastic assignment of shape (N, K).
    :type assignment: torch.Tensor
    :param batch:
        Structure index of every atom, of shape (N,).
    :type batch: torch.Tensor
    :param num_structures:
        Number of structures B in the batch.
    :type num_structures: int
    :param x_1:
        Batch of clean target structures, used for spatial extents and the reference partition when available.
    :type x_1: Optional[torch_geometric.data.Data]

    :return:
        Mapping from diagnostic name to scalar value.
    :rtype: Dict[str, float]
    """
    with torch.no_grad():
        labels = assignment.argmax(dim=1)
        num_columns = assignment.shape[1]
        flat = batch * num_columns + labels
        sizes = torch.bincount(flat, minlength=num_structures * num_columns).reshape(num_structures, num_columns)
        occupied = sizes > 0

        stats: Dict[str, float] = {
            "cg_groups_per_structure": float(occupied.sum(dim=1).to(torch.float64).mean()),
            "cg_group_size_mean": float(sizes[occupied].to(torch.float64).mean()),
            "cg_group_size_max": float(sizes.max()),
            "cg_singleton_fraction": float((sizes[occupied] == 1).to(torch.float64).mean()),
        }

        probabilities = assignment.clamp_min(1.0e-12)
        stats["cg_assignment_entropy"] = float(-(assignment * probabilities.log()).sum(dim=1).mean())

        if x_1 is not None and getattr(x_1, "pos", None) is not None:
            frac_dense, mask = dense_positions(x_1.pos, batch, num_structures)
            distances = pair_distances(frac_dense, x_1.cell, mask)
            labels_dense, _ = to_dense_batch(labels, batch, batch_size=num_structures, fill_value=-1)
            extents = atom_group_extents(distances, labels_dense, mask)
            stats["cg_group_extent_mean"] = float(extents.mean())
            stats["cg_group_extent_max"] = float(extents.max())

        # Against k-medoids this detects collapse onto plain geometric clustering, which is what the flow-matching
        # objective rewards by itself; against the coordination shells it is the research question.
        for method in REFERENCE_METHODS:
            reference = getattr(x_1, reference_field(method), None) if x_1 is not None else None
            if reference is not None:
                stats[f"cg_ari_vs_{method}"] = float(
                    adjusted_rand_index(labels, reference.reshape(-1).long(), batch, num_structures))

    return stats


class CoarseToFineDiagnostics(lightning.pytorch.callbacks.Callback):
    """
    Log the grouping diagnostics that the coarse-to-fine interpolant records during training.

    :param data_field:
        Data field whose stochastic interpolant carries the diagnostics.
        Defaults to "pos".
    :type data_field: str
    """

    def __init__(self, data_field: str = "pos") -> None:
        """Construct the diagnostics callback."""
        super().__init__()
        self._data_field = data_field

    def _interpolant(self, pl_module: lightning.LightningModule) -> Any:
        """
        Return the stochastic interpolant of the tracked data field.

        :param pl_module:
            The Lightning module being trained.
        :type pl_module: lightning.LightningModule

        :return:
            The stochastic interpolant, or None if the module does not have one.
        :rtype: Any
        """
        si = getattr(pl_module, "si", None)
        return si.get_stochastic_interpolant(self._data_field) if si is not None else None

    def on_train_batch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule, outputs: Any,
                           batch: Any, batch_idx: int) -> None:
        """
        Log any diagnostics recorded since the previous batch.

        :param trainer:
            The Lightning trainer.
        :type trainer: lightning.Trainer
        :param pl_module:
            The Lightning module being trained.
        :type pl_module: lightning.LightningModule
        :param outputs:
            Outputs of the training step.
        :type outputs: Any
        :param batch:
            The training batch.
        :type batch: Any
        :param batch_idx:
            Index of the training batch.
        :type batch_idx: int
        """
        interpolant = self._interpolant(pl_module)
        recorded = getattr(interpolant, "last_diagnostics", None)
        if not recorded:
            return
        grouping = interpolant.get_grouping()
        temperature = getattr(grouping, "temperature", None)
        if temperature is not None:
            recorded = dict(recorded, cg_temperature=float(temperature))
        pl_module.log_dict(recorded, on_step=False, on_epoch=True, batch_size=len(batch.n_atoms))
        interpolant.last_diagnostics = {}


class GroupingTemperatureSchedule(lightning.pytorch.callbacks.Callback):
    """
    Linearly anneal the membership temperature of a learned grouping towards a hard partition.

    Annealing matters for two reasons. The coarse and fine schedules are exactly t +/- eta s(t) only when the group-mean
    operator is idempotent, which holds for hard partitions, and the descriptive comparison against coordination
    environments is only meaningful for a genuine partition.

    :param start:
        Temperature at the beginning of training.
        Defaults to 1.0.
    :type start: float
    :param end:
        Temperature reached at the end of the annealing window.
        Defaults to 0.1.
    :type end: float
    :param anneal_epochs:
        Number of epochs over which the temperature is annealed. Non-positive keeps the temperature at start.
        Defaults to 200.
    :type anneal_epochs: int
    :param data_field:
        Data field whose stochastic interpolant holds the grouping.
        Defaults to "pos".
    :type data_field: str

    :raises ValueError:
        If either temperature is not positive.
    """

    def __init__(self, start: float = 1.0, end: float = 0.1, anneal_epochs: int = 200,
                 data_field: str = "pos") -> None:
        """Construct the temperature schedule callback."""
        super().__init__()
        if start <= 0.0 or end <= 0.0:
            raise ValueError("Temperatures must be positive.")
        self._start = float(start)
        self._end = float(end)
        self._anneal_epochs = int(anneal_epochs)
        self._data_field = data_field

    def on_train_epoch_start(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        """
        Set the grouping temperature for the epoch that is about to start.

        :param trainer:
            The Lightning trainer.
        :type trainer: lightning.Trainer
        :param pl_module:
            The Lightning module being trained.
        :type pl_module: lightning.LightningModule
        """
        si = getattr(pl_module, "si", None)
        if si is None:
            return
        grouping = si.get_stochastic_interpolant(self._data_field).get_grouping()
        if grouping is None or not hasattr(grouping, "temperature"):
            return
        if self._anneal_epochs <= 0:
            grouping.temperature = self._start
            return
        progress = min(1.0, trainer.current_epoch / self._anneal_epochs)
        grouping.temperature = self._start + progress * (self._end - self._start)
