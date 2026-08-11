"""
Coarse-to-fine stochastic interpolant for fractional coordinates.

Let d = Log_{x_0}(x_1) be the periodic displacement from the base sample to the data sample, that is, the minimum-image
separation that OMatG's periodic corrector already computes, and let Delta = (2 S - I) d be the centred difference
between the coarse (group-mean) and fine (within-group) parts of d. The path is

    x_t = wrap( (1 - t) x_0 + t x_1 + eta s(t) Delta ),
    u_t = (x_1 - x_0) + eta s'(t) Delta,

with the bump s(t) = t^p (1 - t)^q vanishing at both endpoints. Because s(0) = s(1) = 0 the marginals at t = 0 and
t = 1 are exactly those of the atomwise baseline, so this is a genuine reparameterisation of the conditional path and
not a change of the modelled distribution.

For an idempotent S (any hard partition) this is algebraically identical to running the coarse part of the displacement
on the schedule a(t) = t + eta s(t) and the within-group part on b(t) = t - eta s(t): group centroids reach their
targets ahead of the internal geometry. Setting eta = 0 removes the extra term exactly, recovering OMatG's
PeriodicLinearInterpolant bit for bit.

Two deliberate departures from the stock SingleStochasticInterpolant:

- The velocity loss is the explicit mean squared error rather than OMatG's expansion that drops the constant
  E||u_t||^2. The two have identical gradients with respect to the denoiser, but the dropped term is not constant once
  the grouping is learned: its absence would leave the grouping network with the unbounded gradient of -2 E[b . u_t],
  which is minimised by inflating ||u_t|| along the model prediction rather than by finding better groups.
- Only ODE training without a latent variable is supported, matching the reference crystal-structure-prediction
  configuration. The antithetic and score-based branches would each need their own coarse-to-fine derivation.
"""

from typing import Any, Callable, Dict, Optional
import torch
from torch_geometric.data import Data
from omg.globals import SMALL_TIME, BIG_TIME
from omg.si.interpolants import PeriodicLinearInterpolant
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from .blur import coarse_fine_delta, fine_energy_fraction, structure_mean
from .diagnostics import group_statistics
from .grouping import Grouping


class CoarseToFineStochasticInterpolant(SingleStochasticInterpolant):
    """
    Periodic linear interpolant with an added coarse-to-fine bump along the group-mean direction.

    :param eta:
        Strength of the coarse-to-fine bump. Zero reproduces the atomwise periodic linear path exactly.
    :type eta: float
    :param grouping:
        Producer of atom-to-group assignments, or None for the atomwise baseline. Must be None if and only if eta is
        zero.
    :type grouping: Optional[Grouping]
    :param bump_power_start:
        Exponent p of the bump s(t) = t^p (1 - t)^q. Raising it delays the coarse-to-fine crossover.
        Defaults to 1.0.
    :type bump_power_start: float
    :param bump_power_end:
        Exponent q of the bump s(t) = t^p (1 - t)^q. Raising it advances the coarse-to-fine crossover.
        Defaults to 1.0.
    :type bump_power_end: float
    :param integrator_kwargs:
        Optional keyword arguments for the odeint function of torchdiffeq.
    :type integrator_kwargs: Optional[dict]
    :param correct_center_of_mass_motion:
        Whether to remove the per-structure mean velocity before computing the loss.
        Defaults to False.
    :type correct_center_of_mass_motion: bool
    :param velocity_annealing_factor:
        During inference, predicted velocities are multiplied by (1 + velocity_annealing_factor * t).
        Defaults to 0.0.
    :type velocity_annealing_factor: float
    :param diagnostics_every:
        Record grouping diagnostics on every n-th loss evaluation. Zero disables them.
        Defaults to 50.
    :type diagnostics_every: int

    :raises ValueError:
        If eta is zero but a grouping is given, or eta is non-zero but no grouping is given.
        If the bump exponents are not positive.
        If eta and the bump exponents make a coarse or fine schedule non-monotonic on [SMALL_TIME, BIG_TIME].
    """

    requires_aux = True

    def __init__(self, eta: float, grouping: Optional[Grouping] = None, bump_power_start: float = 1.0,
                 bump_power_end: float = 1.0, integrator_kwargs: Optional[dict[str, Any]] = None,
                 correct_center_of_mass_motion: bool = False, velocity_annealing_factor: float = 0.0,
                 diagnostics_every: int = 50) -> None:
        """Construct the coarse-to-fine stochastic interpolant."""
        super().__init__(interpolant=PeriodicLinearInterpolant(), gamma=None, epsilon=None,
                         differential_equation_type="ODE", integrator_kwargs=integrator_kwargs,
                         correct_center_of_mass_motion=correct_center_of_mass_motion,
                         velocity_annealing_factor=velocity_annealing_factor)
        if eta == 0.0 and grouping is not None:
            raise ValueError("A grouping has no effect at eta = 0. Pass grouping=None for the atomwise baseline arm.")
        if eta != 0.0 and grouping is None:
            raise ValueError("A non-zero eta requires a grouping.")
        if bump_power_start <= 0.0 or bump_power_end <= 0.0:
            raise ValueError("The bump exponents must be positive so that the bump vanishes at both endpoints.")
        self._eta = float(eta)
        self._grouping = grouping
        self._bump_power_start = float(bump_power_start)
        self._bump_power_end = float(bump_power_end)
        self._diagnostics_every = int(diagnostics_every)
        self._check_monotonic_schedules()
        # Single-slot cache of Delta, written by interpolate and consumed by the loss. See _pop_cached_delta.
        self._cached_delta: Optional[tuple[int, torch.Size, torch.Tensor]] = None
        self._loss_calls = 0
        self.last_diagnostics: Dict[str, float] = {}

    def _bump(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the bump s(t) = t^p (1 - t)^q.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Values of the bump.
        :rtype: torch.Tensor
        """
        return t ** self._bump_power_start * (1.0 - t) ** self._bump_power_end

    def _bump_derivative(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the time derivative of the bump s(t) = t^p (1 - t)^q.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivatives of the bump.
        :rtype: torch.Tensor
        """
        p, q = self._bump_power_start, self._bump_power_end
        return (p * t ** (p - 1.0) * (1.0 - t) ** q) - (q * t ** p * (1.0 - t) ** (q - 1.0))

    def _check_monotonic_schedules(self) -> None:
        """
        Check that the coarse schedule a(t) = t + eta s(t) and the fine schedule b(t) = t - eta s(t) both increase.

        Both derivatives are 1 -/+ eta s'(t), so the condition is |eta s'(t)| < 1 over the sampled time interval. For
        the default bump this is exactly eta < 1.

        :raises ValueError:
            If either schedule is non-monotonic anywhere on [SMALL_TIME, BIG_TIME].
        """
        if self._eta == 0.0:
            return
        t = torch.linspace(SMALL_TIME, BIG_TIME, 10001, dtype=torch.float64)
        worst = float((self._eta * self._bump_derivative(t)).abs().max())
        if worst >= 1.0:
            raise ValueError(
                f"eta={self._eta} with bump exponents ({self._bump_power_start}, {self._bump_power_end}) makes the "
                f"coarse or fine schedule non-monotonic (max |eta s'(t)| = {worst:.4f}, must be below 1).")

    def _delta(self, t: torch.Tensor, displacement: torch.Tensor, batch_indices: torch.Tensor,
               aux: Optional[Data]) -> torch.Tensor:
        """
        Compute the centred coarse-minus-fine displacement for the batch.

        :param t:
            Times in [0,1], only used for its shape and dtype in the atomwise case.
        :type t: torch.Tensor
        :param displacement:
            Per-atom displacement d of shape (N, 3).
        :type displacement: torch.Tensor
        :param batch_indices:
            Structure index of every atom, of shape (N,).
        :type batch_indices: torch.Tensor
        :param aux:
            Batch of clean target structures carrying whatever the grouping needs.
        :type aux: Optional[torch_geometric.data.Data]

        :return:
            Centred Delta of shape (N, 3), or an exact zero scalar for the atomwise baseline.
        :rtype: torch.Tensor

        :raises ValueError:
            If a grouping is configured but the batch of clean target structures was not passed through.
        """
        if self._grouping is None:
            return torch.zeros((), dtype=displacement.dtype, device=displacement.device)
        if aux is None:
            raise ValueError(
                "The coarse-to-fine interpolant needs the clean target structures. This is passed automatically by "
                "omg.si.stochastic_interpolants.StochasticInterpolants when requires_aux is True.")
        assignment = self._grouping.assignment(aux)
        num_structures = int(aux.n_atoms.shape[0])
        return coarse_fine_delta(displacement, assignment, batch_indices, num_structures)

    def _pop_cached_delta(self, x_1: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Consume the Delta computed by the immediately preceding interpolate call for the same batch.

        StochasticInterpolants.losses interpolates every data field first and only then loops over the fields again to
        evaluate their losses, so nothing re-enters this instance between its own interpolate and loss calls. Caching
        matters for the learned grouping, where recomputing would build the grouping network's autograd graph twice.
        The cache is keyed on the identity of the target tensor so that a stale entry can never be consumed.

        :param x_1:
            Points from p_1, used as the cache key.
        :type x_1: torch.Tensor

        :return:
            The cached Delta, or None if the cache does not hold an entry for this batch.
        :rtype: Optional[torch.Tensor]
        """
        cached, self._cached_delta = self._cached_delta, None
        if cached is None:
            return None
        key, shape, delta = cached
        if key != x_1.data_ptr() or shape != x_1.shape:
            return None
        return delta

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, batch_indices: torch.Tensor,
                    aux: Optional[Data] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolate between the base points x_0 and the data points x_1 at times t along the coarse-to-fine path.

        :param t:
            Times in [0,1], broadcast to the shape of the coordinates.
        :type t: torch.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor
        :param batch_indices:
            Structure index of every atom.
        :type batch_indices: torch.Tensor
        :param aux:
            Batch of clean target structures carrying whatever the grouping needs.
        :type aux: Optional[torch_geometric.data.Data]

        :return:
            Interpolated points x_t, and the zero latent variable that this interpolant does not use.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        assert x_0.shape == x_1.shape
        assert self._check_t(t)
        corrector = self._corrector
        x_0prime = corrector.correct(x_0)
        x_1prime = corrector.unwrap(x_0prime, x_1)
        delta = self._delta(t, x_1prime - x_0prime, batch_indices, aux)
        self._cached_delta = (x_1.data_ptr(), x_1.shape, delta)
        # Written in the same form as the stock interpolant so that the eta = 0 path is bit-for-bit identical: the
        # added term is then exactly the float zero, which leaves the baseline expression unchanged.
        base = self._interpolant.alpha(t) * x_0prime + self._interpolant.beta(t) * x_1prime
        x_t = corrector.correct(base + self._eta * self._bump(t) * delta)
        return x_t, torch.zeros_like(x_0)

    def expected_velocity(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, batch_indices: torch.Tensor,
                          aux: Optional[Data] = None, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the conditional velocity u_t of the coarse-to-fine path.

        :param t:
            Times in [0,1], broadcast to the shape of the coordinates.
        :type t: torch.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor
        :param batch_indices:
            Structure index of every atom.
        :type batch_indices: torch.Tensor
        :param aux:
            Batch of clean target structures carrying whatever the grouping needs.
        :type aux: Optional[torch_geometric.data.Data]
        :param delta:
            Precomputed centred Delta, recomputed from the grouping if None.
        :type delta: Optional[torch.Tensor]

        :return:
            Conditional velocity of shape (N, 3).
        :rtype: torch.Tensor
        """
        corrector = self._corrector
        x_0prime = corrector.correct(x_0)
        x_1prime = corrector.unwrap(x_0prime, x_1)
        if delta is None:
            delta = self._delta(t, x_1prime - x_0prime, batch_indices, aux)
        base = self._interpolant.alpha_dot(t) * x_0prime + self._interpolant.beta_dot(t) * x_1prime
        return base + self._eta * self._bump_derivative(t) * delta

    def _ode_loss(self, model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                  t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor,
                  batch_indices: torch.Tensor, aux: Optional[Data] = None) -> Dict[str, torch.Tensor]:
        """
        Compute the flow-matching loss of the coarse-to-fine path.

        :param model_function:
            Model function returning the velocity field b and the denoiser eta given the positions x_t.
        :type model_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        :param t:
            Times in [0,1], broadcast to the shape of the coordinates.
        :type t: torch.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor
        :param x_t:
            Interpolated points.
        :type x_t: torch.Tensor
        :param z:
            Unused latent variable.
        :type z: torch.Tensor
        :param batch_indices:
            Structure index of every atom.
        :type batch_indices: torch.Tensor
        :param aux:
            Batch of clean target structures carrying whatever the grouping needs.
        :type aux: Optional[torch_geometric.data.Data]

        :return:
            Dictionary holding the velocity loss under the key 'loss_b'.
        :rtype: Dict[str, torch.Tensor]
        """
        assert x_0.shape == x_1.shape
        delta = self._pop_cached_delta(x_1)
        velocity = self.expected_velocity(t, x_0, x_1, batch_indices, aux=aux, delta=delta)
        num_structures = int(aux.n_atoms.shape[0]) if aux is not None else int(batch_indices.max()) + 1
        if self._correct_center_of_mass_motion:
            velocity = velocity - structure_mean(velocity, batch_indices, num_structures)[batch_indices]
        pred_b = model_function(x_t)[0]
        self._record_diagnostics(x_0, x_1, batch_indices, num_structures, delta, aux)
        return {"loss_b": torch.mean((pred_b - velocity) ** 2)}

    def _record_diagnostics(self, x_0: torch.Tensor, x_1: torch.Tensor, batch_indices: torch.Tensor,
                            num_structures: int, delta: Optional[torch.Tensor], aux: Optional[Data]) -> None:
        """
        Refresh the grouping diagnostics that the CoarseToFineDiagnostics callback logs.

        The learned grouping can lower the flow-matching loss by shrinking the within-group residual instead of by
        finding better groups, and the argmin of that residual at fixed group count is plain geometric clustering.
        These statistics are what distinguishes the two outcomes, so they are collected during training rather than
        reconstructed afterwards.

        :param x_0:
            Points from p_0.
        :type x_0: torch.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: torch.Tensor
        :param batch_indices:
            Structure index of every atom.
        :type batch_indices: torch.Tensor
        :param num_structures:
            Number of structures in the batch.
        :type num_structures: int
        :param delta:
            Centred Delta of the batch, if it is available.
        :type delta: Optional[torch.Tensor]
        :param aux:
            Batch of clean target structures carrying whatever the grouping needs.
        :type aux: Optional[torch_geometric.data.Data]
        """
        self._loss_calls += 1
        if self._grouping is None or self._diagnostics_every <= 0 or aux is None:
            return
        if self._loss_calls % self._diagnostics_every != 0:
            return
        with torch.no_grad():
            corrector = self._corrector
            x_0prime = corrector.correct(x_0)
            displacement = corrector.unwrap(x_0prime, x_1) - x_0prime
            assignment = self._grouping.assignment(aux)
            stats = group_statistics(assignment, batch_indices, num_structures, aux)
            stats["cg_fine_energy_fraction"] = float(
                fine_energy_fraction(displacement, assignment, batch_indices, num_structures))
            if delta is not None and delta.ndim == 2:
                stats["cg_delta_norm_ratio"] = float(
                    delta.norm() / displacement.norm().clamp_min(1.0e-12))
            self.last_diagnostics = stats

    def get_eta(self) -> float:
        """
        Return the strength of the coarse-to-fine bump.

        :return:
            The value of eta.
        :rtype: float
        """
        return self._eta

    def get_grouping(self) -> Optional[Grouping]:
        """
        Return the grouping used to build the coarse-to-fine paths.

        :return:
            The grouping, or None for the atomwise baseline.
        :rtype: Optional[Grouping]
        """
        return self._grouping
