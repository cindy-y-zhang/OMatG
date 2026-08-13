"""
Construction of the four experimental arms.

The experiment only supports a conclusion if the arms are identical in everything except the probability path for
fractional coordinates. Expressing each arm as a full stochastic-interpolant list in its own configuration file would
make that a property of four files staying in sync by hand, which it would not. Instead every arm instantiates this
class, which builds all three interpolants itself and exposes exactly two knobs that are allowed to differ: which
grouping is used, and how strong the coarse-to-fine bump is.

The species and cell interpolants and the integrator are therefore literally the same objects across arms. Changing an
arm cannot silently change anything else.

The position velocity annealing factor is the one exception: it is a knob of the sampler rather than of the path, and it
is exposed because the tuned default is only correct for the atomwise arm. The bump raises the terminal slope of the
within-group schedule to 1 + eta, so a factor tuned at eta = 0 overshoots by that ratio on the component that carries
most of the displacement. Arms compared against each other must either share this factor or each be reported at its own
tuned value, and which of the two is being done has to be stated.

The four arms are:

- atomwise: no grouping, eta = 0, which reproduces OMatG's periodic linear path bit for bit;
- kmedoids: precomputed periodic geometric clusters;
- shells: precomputed CrystalNN coordination shells;
- learned: the anchor-and-membership grouping network trained through the flow-matching loss.

The two precomputed arms build the same object here and differ only in which group file the data module loads, so the
arm name is carried by the run directory and the data configuration rather than by this class.
"""

from typing import Any, Optional
from omg.si.interpolants import LinearInterpolant
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from omg.si.single_stochastic_interpolant_identity import SingleStochasticInterpolantIdentity
from omg.si.stochastic_interpolants import StochasticInterpolants
from .grouper import build_grouper
from .interpolant import CoarseToFineStochasticInterpolant


POSITION_VELOCITY_ANNEALING_FACTOR = 10.182659004291072
"""
Velocity annealing factor for fractional coordinates, from OMatG's tuned crystal-structure-prediction configuration
omg/conf_examples/csp_linear_ode_mp_20.yaml.
"""

CELL_VELOCITY_ANNEALING_FACTOR = 1.824475401606087
"""
Velocity annealing factor for the cell, from the same OMatG configuration.
"""

INTEGRATION_TIME_STEPS = 210
"""
Number of Euler steps used for generation, from the same OMatG configuration.
"""


class ArmStochasticInterpolants(StochasticInterpolants):
    """
    Collection of stochastic interpolants for one experimental arm.

    :param grouping_kind:
        One of "none", "precomputed" or "learned".
    :type grouping_kind: str
    :param eta:
        Strength of the coarse-to-fine bump. Must be zero when grouping_kind is "none" and non-zero otherwise.
        Defaults to 0.0.
    :type eta: float
    :param bump_power_start:
        Exponent p of the bump s(t) = t^p (1 - t)^q.
        Defaults to 1.0.
    :type bump_power_start: float
    :param bump_power_end:
        Exponent q of the bump s(t) = t^p (1 - t)^q.
        Defaults to 1.0.
    :type bump_power_end: float
    :param grouper_kwargs:
        Keyword arguments for the grouping network, used only when grouping_kind is "learned".
        Defaults to None.
    :type grouper_kwargs: Optional[dict]
    :param diagnostics_every:
        Record grouping diagnostics on every n-th loss evaluation.
        Defaults to 50.
    :type diagnostics_every: int
    :param integration_time_steps:
        Number of Euler steps used for generation.
        Defaults to INTEGRATION_TIME_STEPS.
    :type integration_time_steps: int
    :param velocity_annealing_factor:
        Inference-only factor that multiplies predicted position velocities by (1 + factor * t) during generation. The
        default was tuned for the atomwise path, where the terminal schedule slope is one; a non-zero eta raises the
        terminal slope of the within-group schedule to 1 + eta, so the tuned value overshoots there.
        Defaults to POSITION_VELOCITY_ANNEALING_FACTOR.
    :type velocity_annealing_factor: float
    :param enable_progress_bar:
        Whether the integrator shows a progress bar.
        Defaults to True.
    :type enable_progress_bar: bool
    """

    def __init__(self, grouping_kind: str, eta: float = 0.0, bump_power_start: float = 1.0,
                 bump_power_end: float = 1.0, grouper_kwargs: Optional[dict[str, Any]] = None,
                 diagnostics_every: int = 50, integration_time_steps: int = INTEGRATION_TIME_STEPS,
                 enable_progress_bar: bool = True,
                 velocity_annealing_factor: float = POSITION_VELOCITY_ANNEALING_FACTOR) -> None:
        """Construct the collection of stochastic interpolants for one arm."""
        position_interpolant = CoarseToFineStochasticInterpolant(
            eta=eta, grouping=build_grouper(grouping_kind, **(grouper_kwargs or {})),
            bump_power_start=bump_power_start, bump_power_end=bump_power_end,
            integrator_kwargs={"method": "euler"}, correct_center_of_mass_motion=True,
            velocity_annealing_factor=velocity_annealing_factor, diagnostics_every=diagnostics_every)
        cell_interpolant = SingleStochasticInterpolant(
            interpolant=LinearInterpolant(), gamma=None, epsilon=None, differential_equation_type="ODE",
            integrator_kwargs={"method": "euler"}, correct_center_of_mass_motion=False,
            velocity_annealing_factor=CELL_VELOCITY_ANNEALING_FACTOR)
        super().__init__(
            stochastic_interpolants=[SingleStochasticInterpolantIdentity(), position_interpolant, cell_interpolant],
            data_fields=["species", "pos", "cell"], integration_time_steps=integration_time_steps,
            enable_progress_bar=enable_progress_bar)
