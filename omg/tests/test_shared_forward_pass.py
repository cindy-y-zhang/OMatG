"""
Regression tests for shared stochastic-interpolant model evaluations.

Every ordinary data field asks the model for a prediction at the same
interpolated state. These tests pin that work to one forward pass and verify
that sharing it does not change the objective, including with the fourth
joint-geometry field.
"""

import pytest
import torch
from torch_geometric.data import Data

from omg.si.interpolants import LinearInterpolant, PeriodicLinearInterpolant
from omg.si.gamma import LatentGammaSqrt
from omg.si.epsilon import VanishingEpsilon
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from omg.si.single_stochastic_interpolant_identity import SingleStochasticInterpolantIdentity
from omg.si.stochastic_interpolants import StochasticInterpolants
from omg.utils import DataField, reshape_t


N_STRUCTURES = 4
ATOMS_PER_STRUCTURE = 3


def _batch(seed: int) -> tuple[Data, Data, torch.Tensor]:
    """Build a small deterministic pair of endpoints plus times, in double precision for exact comparisons."""
    generator = torch.Generator().manual_seed(seed)
    n_atoms = torch.full((N_STRUCTURES,), ATOMS_PER_STRUCTURE, dtype=torch.long)
    batch = torch.repeat_interleave(torch.arange(N_STRUCTURES), ATOMS_PER_STRUCTURE)
    total_atoms = N_STRUCTURES * ATOMS_PER_STRUCTURE
    species = torch.randint(1, 90, (total_atoms,), generator=generator)
    x_0 = Data(
        pos=torch.rand(total_atoms, 3, generator=generator, dtype=torch.float64),
        cell=torch.rand(N_STRUCTURES, 3, 3, generator=generator, dtype=torch.float64),
        species=species.clone(), batch=batch, n_atoms=n_atoms)
    x_1 = Data(
        pos=torch.rand(total_atoms, 3, generator=generator, dtype=torch.float64),
        cell=torch.rand(N_STRUCTURES, 3, 3, generator=generator, dtype=torch.float64),
        species=species.clone(), batch=batch, n_atoms=n_atoms)
    t = torch.rand(N_STRUCTURES, generator=generator, dtype=torch.float64) * 0.98 + 0.01
    return x_0, x_1, t


class _RecordingModel:
    """
    Stand-in for the encoder that counts its calls and keeps the states it was asked about.

    The predictions are deliberately functions of the input, so a prediction taken at the wrong state would change the
    losses rather than pass unnoticed.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.seen_pos = []
        self.seen_cell = []
        self.seen_geometry = []

    def __call__(self, x: Data, t: torch.Tensor) -> Data:
        self.calls += 1
        self.seen_pos.append(x.pos.clone())
        self.seen_cell.append(x.cell.clone())
        result = Data(
            pos_b=2.0 * x.pos, pos_eta=torch.zeros_like(x.pos),
            cell_b=3.0 * x.cell, cell_eta=torch.zeros_like(x.cell),
            species_b=torch.zeros_like(x.pos), species_eta=torch.zeros_like(x.pos))
        if hasattr(x, "geometry"):
            self.seen_geometry.append(x.geometry.clone())
            result.geometry_b = 4.0 * x.geometry
            result.geometry_eta = torch.zeros_like(x.geometry)
        return result


def _csp_interpolants(correct_center_of_mass_motion: bool = False) -> StochasticInterpolants:
    """The crystal-structure-prediction configuration: species carried through, positions and lattice as ODEs."""
    return StochasticInterpolants(
        stochastic_interpolants=[
            SingleStochasticInterpolantIdentity(),
            SingleStochasticInterpolant(
                interpolant=PeriodicLinearInterpolant(), gamma=None, epsilon=None,
                differential_equation_type="ODE", integrator_kwargs={"method": "euler"},
                correct_center_of_mass_motion=correct_center_of_mass_motion),
            SingleStochasticInterpolant(
                interpolant=LinearInterpolant(), gamma=None, epsilon=None,
                differential_equation_type="ODE", integrator_kwargs={"method": "euler"}),
        ],
        data_fields=["species", "pos", "cell"], integration_time_steps=5, enable_progress_bar=False)


@pytest.mark.parametrize("correct_center_of_mass_motion", [False, True])
def test_csp_losses_run_the_model_exactly_once(correct_center_of_mass_motion):
    """Three data fields, one forward pass, and the states it saw are the interpolated states."""
    si = _csp_interpolants(correct_center_of_mass_motion)
    x_0, x_1, t = _batch(seed=0)
    model = _RecordingModel()

    losses = si.losses(model, t, x_0, x_1)

    assert model.calls == 1
    expected_x_t, _ = si._interpolate(t, x_0, x_1)
    assert torch.equal(model.seen_pos[0], expected_x_t.pos)
    assert torch.equal(model.seen_cell[0], expected_x_t.cell)
    assert set(losses) == {"species_loss", "pos_loss_b", "cell_loss_b"}


def test_joint_geometry_losses_run_the_model_exactly_once():
    """The geometry field shares the same encoder call as positions and cell."""
    base = _csp_interpolants()
    si = StochasticInterpolants(
        stochastic_interpolants=[
            *base._stochastic_interpolants,
            SingleStochasticInterpolant(
                interpolant=LinearInterpolant(),
                gamma=None,
                epsilon=None,
                differential_equation_type="ODE",
                integrator_kwargs={"method": "euler"},
            ),
        ],
        data_fields=["species", "pos", "cell", "geometry"],
        integration_time_steps=5,
        enable_progress_bar=False,
    )
    x_0, x_1, t = _batch(seed=9)
    generator = torch.Generator().manual_seed(10)
    x_0.geometry = torch.randn(
        (N_STRUCTURES * ATOMS_PER_STRUCTURE, 4),
        generator=generator,
        dtype=torch.float64,
    )
    x_1.geometry = torch.randn(
        (N_STRUCTURES * ATOMS_PER_STRUCTURE, 4),
        generator=generator,
        dtype=torch.float64,
    )
    model = _RecordingModel()

    losses = si.losses(model, t, x_0, x_1)

    expected_x_t, _ = si._interpolate(t, x_0, x_1)
    assert model.calls == 1
    assert torch.equal(model.seen_geometry[0], expected_x_t.geometry)
    assert set(losses) == {
        "species_loss",
        "pos_loss_b",
        "cell_loss_b",
        "geometry_loss_b",
    }


@pytest.mark.parametrize("correct_center_of_mass_motion", [False, True])
def test_shared_forward_pass_gives_the_hand_computed_losses(correct_center_of_mass_motion):
    """
    The objective is unchanged, checked against the closed form rather than against the old code.

    The ODE loss is ``mean(b ** 2) - 2 mean(b . u)`` for the conditional velocity ``u``, which for a linear interpolant
    is the endpoint difference. The prediction is a fixed multiple of the interpolated state, so both terms are known.
    """
    si = _csp_interpolants(correct_center_of_mass_motion)
    x_0, x_1, t = _batch(seed=1)
    model = _RecordingModel()

    losses = si.losses(model, t, x_0, x_1)

    x_t, _ = si._interpolate(t, x_0, x_1)
    for field, factor in ((DataField.pos, 2.0), (DataField.cell, 3.0)):
        reshaped_t = reshape_t(t, x_0.n_atoms, field)
        interpolant = si.get_stochastic_interpolant(field.name)
        velocity = interpolant._interpolant.interpolate_derivative(
            reshaped_t, x_0[field.name], x_1[field.name])
        if correct_center_of_mass_motion and field is DataField.pos:
            mean_velocity = torch.zeros_like(velocity).index_add_(
                0, x_0.batch, velocity) / ATOMS_PER_STRUCTURE
            velocity = velocity - mean_velocity[x_0.batch]
        prediction = factor * x_t[field.name]
        expected = torch.mean(prediction ** 2) - 2.0 * torch.mean(prediction * velocity)
        assert losses[f"{field.name}_loss_b"].item() == pytest.approx(expected.item(), rel=1e-12, abs=1e-12)


def test_the_antithetic_path_still_evaluates_the_perturbed_state():
    """
    The cache must not be applied where the loss perturbs the state it asks about.

    An SDE field with an antithetic gamma asks for predictions at ``x_t + gamma z`` and at ``x_t - gamma z``. Only the
    first is the interpolated state, so exactly one of the two calls may be shared and the other must be a real
    forward pass.
    """
    si = StochasticInterpolants(
        stochastic_interpolants=[
            SingleStochasticInterpolant(
                interpolant=LinearInterpolant(), gamma=LatentGammaSqrt(0.1), epsilon=VanishingEpsilon(c=0.1),
                differential_equation_type="SDE", integrator_kwargs={"method": "euler"}),
        ],
        data_fields=["pos"], integration_time_steps=5, enable_progress_bar=False)
    x_0, x_1, t = _batch(seed=2)
    model = _RecordingModel()

    si.losses(model, t, x_0, x_1)

    assert model.calls == 2
    assert not torch.equal(model.seen_pos[0], model.seen_pos[1])
