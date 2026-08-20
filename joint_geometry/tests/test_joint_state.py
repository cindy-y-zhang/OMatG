"""Tests for the fourth stochastic-interpolant field and its encoder paths."""

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from direct_geometry.encoder import DirectGeometryCSPNet, baseline_encoder_state
from direct_geometry.tests.test_encoder import SMALL, inputs
from joint_geometry.checkpoint import FixedTrainingSeed, PairedInferenceSeed
from joint_geometry.encoder import JointGeometryCSPNet
from joint_geometry.interpolants import OracleGeometryInterpolants
from joint_geometry.sampler import JointGeometrySampler, OracleGeometrySampler
from omg.sampler import Sampler
from omg.si.interpolants import LinearInterpolant
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from omg.si.stochastic_interpolants import StochasticInterpolants
from omg.utils import DataField, reshape_t


def linear_si() -> SingleStochasticInterpolant:
    return SingleStochasticInterpolant(
        interpolant=LinearInterpolant(),
        gamma=None,
        epsilon=None,
        differential_equation_type="ODE",
        integrator_kwargs={"method": "euler"},
        velocity_annealing_factor=0.0,
        correct_center_of_mass_motion=False,
    )


def encoder_inputs():
    atom_types, frac, cells, atoms, node2graph, time = inputs()
    return atom_types, frac.double(), cells.double(), atoms, node2graph, time.double()


def test_geometry_time_broadcast_uses_runtime_width() -> None:
    time = torch.tensor([0.2, 0.8])
    atoms = torch.tensor([2, 3])
    actual = reshape_t(time, atoms, DataField.geometry, (5, 7))
    assert actual.shape == (5, 7)
    assert torch.equal(actual[:2], torch.full((2, 7), 0.2))
    assert torch.equal(actual[2:], torch.full((3, 7), 0.8))


def test_geometry_time_broadcast_requires_runtime_shape() -> None:
    with pytest.raises(ValueError, match="field_shape is required"):
        reshape_t(torch.tensor([0.4]), torch.tensor([2]), DataField.geometry)


def test_base_interpolant_trains_and_integrates_geometry_as_a_normal_field() -> None:
    atoms = torch.tensor([2])
    common = {
        "pos": torch.zeros((2, 3)),
        "cell": torch.eye(3).unsqueeze(0),
        "species": torch.ones(2, dtype=torch.long),
        "n_atoms": atoms,
        "batch": torch.zeros(2, dtype=torch.long),
    }
    x_0 = Data(**common, geometry=torch.zeros((2, 4)))
    x_1 = Data(**common, geometry=torch.ones((2, 4)))
    collection = StochasticInterpolants(
        stochastic_interpolants=[linear_si()],
        data_fields=["geometry"],
        integration_time_steps=5,
        enable_progress_bar=False,
    )

    def model(state: Data, time: torch.Tensor) -> Data:
        return Data(geometry_b=torch.ones_like(state.geometry), geometry_eta=torch.zeros_like(state.geometry))

    losses = collection.losses(model, torch.tensor([0.4]), x_0, x_1)
    assert set(losses) == {"geometry_loss_b"}
    assert float(losses["geometry_loss_b"]) == -1.0
    generated = collection.integrate(x_0, model)
    assert torch.allclose(generated.geometry, torch.full((2, 4), 0.998), atol=1.0e-6)


def test_structure_and_geometry_receive_the_same_sampled_time() -> None:
    atoms = torch.tensor([2])
    common = {
        "cell": torch.eye(3).unsqueeze(0),
        "species": torch.ones(2, dtype=torch.long),
        "n_atoms": atoms,
        "batch": torch.zeros(2, dtype=torch.long),
    }
    x_0 = Data(**common, pos=torch.zeros((2, 3)), geometry=torch.zeros((2, 4)))
    x_1 = Data(**common, pos=torch.ones((2, 3)), geometry=torch.ones((2, 4)))
    collection = StochasticInterpolants(
        stochastic_interpolants=[linear_si(), linear_si()],
        data_fields=["pos", "geometry"],
        integration_time_steps=3,
        enable_progress_bar=False,
    )
    seen = {}

    def model(state: Data, time: torch.Tensor) -> Data:
        seen["pos"] = state.pos.detach().clone()
        seen["geometry"] = state.geometry.detach().clone()
        return Data(
            pos_b=torch.ones_like(state.pos),
            pos_eta=torch.zeros_like(state.pos),
            geometry_b=torch.ones_like(state.geometry),
            geometry_eta=torch.zeros_like(state.geometry),
        )

    collection.losses(model, torch.tensor([0.25]), x_0, x_1)
    assert torch.equal(seen["pos"], torch.full((2, 3), 0.25))
    assert torch.equal(seen["geometry"], torch.full((2, 4), 0.25))


@pytest.mark.parametrize("graph", ("fc", "fc_distance", "periodic_distance"))
@pytest.mark.parametrize("geometry_dimension", (4, 17))
def test_joint_encoder_starts_with_exact_backbone_structural_outputs(
    graph: str,
    geometry_dimension: int,
) -> None:
    torch.manual_seed(5)
    backbone = DirectGeometryCSPNet(feature_mode="none", message_graph=graph, **SMALL).double()
    torch.manual_seed(99)
    joint = JointGeometryCSPNet(
        geometry_dimension=geometry_dimension,
        geometry_input=True,
        message_graph=graph,
        **SMALL,
    ).double()
    joint.load_baseline_state_dict(backbone.state_dict())
    atom_types, frac, cells, atoms, node2graph, time = encoder_inputs()
    geometry = torch.randn(
        (int(atoms.sum()), geometry_dimension),
        dtype=torch.float64,
    )
    expected = backbone._forward(atom_types, frac, cells, atoms, node2graph, time)
    actual = joint._forward(atom_types, frac, cells, atoms, node2graph, geometry, time)
    for field in ("species_b", "species_eta", "pos_b", "pos_eta", "cell_b", "cell_eta"):
        assert torch.equal(expected[field], actual[field]), field
    assert actual.geometry_b.shape == geometry.shape
    assert torch.equal(actual.geometry_eta, torch.zeros_like(geometry))


def test_real_atomwise_checkpoint_import_accounts_only_for_joint_parameters() -> None:
    path = Path("cgfm/centre_runs/atomwise/checkpoints/best_match_rate.ckpt")
    if not path.is_file():
        pytest.skip("local baseline checkpoint is not part of portable source bundles")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = JointGeometryCSPNet(
        geometry_dimension=4,
        geometry_input=True,
        message_graph="periodic_distance",
    )
    model.load_baseline_state_dict(baseline_encoder_state(checkpoint))
    assert torch.equal(
        model.joint_geometry_projection.weight,
        torch.zeros_like(model.joint_geometry_projection.weight),
    )


def test_joint_input_and_output_paths_receive_gradients() -> None:
    torch.manual_seed(7)
    model = JointGeometryCSPNet(
        geometry_dimension=4,
        geometry_input=True,
        message_graph="fc",
        **SMALL,
    ).double()
    atom_types, frac, cells, atoms, node2graph, time = encoder_inputs()
    geometry = torch.randn((int(atoms.sum()), 4), dtype=torch.float64)
    output = model._forward(atom_types, frac, cells, atoms, node2graph, geometry, time)
    (output.pos_b.square().mean() + output.geometry_b.square().mean()).backward()
    assert model.joint_geometry_projection.weight.grad is not None
    assert float(model.joint_geometry_projection.weight.grad.abs().sum()) > 0.0
    assert model.geometry_out.weight.grad is not None
    assert float(model.geometry_out.weight.grad.abs().sum()) > 0.0


def test_head_only_control_is_independent_of_geometry_input() -> None:
    model = JointGeometryCSPNet(
        geometry_dimension=4,
        geometry_input=False,
        message_graph="fc",
        **SMALL,
    ).double()
    atom_types, frac, cells, atoms, node2graph, time = encoder_inputs()
    first = model._forward(
        atom_types, frac, cells, atoms, node2graph,
        torch.zeros((int(atoms.sum()), 4), dtype=torch.float64), time
    )
    second = model._forward(
        atom_types, frac, cells, atoms, node2graph,
        torch.randn((int(atoms.sum()), 4), dtype=torch.float64), time
    )
    for field in ("species_b", "pos_b", "cell_b", "geometry_b"):
        assert torch.equal(first[field], second[field])
    first.geometry_b.square().mean().backward()
    assert model.joint_geometry_projection.weight.grad is None
    assert model.geometry_out.weight.grad is not None
    assert float(model.geometry_out.weight.grad.abs().sum()) > 0.0


def test_head_dummy_and_real_arms_have_identical_parameter_counts() -> None:
    counts = []
    for geometry_input in (False, True, True):
        model = JointGeometryCSPNet(
            geometry_dimension=4,
            geometry_input=geometry_input,
            message_graph="fc",
            **SMALL,
        )
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert counts[0] == counts[1] == counts[2]


class MirrorSampler(Sampler):
    def sample_p_0(self, x_1):
        return x_1.clone("pos", "cell", "species", "n_atoms", "batch", "ptr")


class LeakySampler(MirrorSampler):
    def sample_p_0(self, x_1):
        result = super().sample_p_0(x_1)
        result.geometry_target = x_1.geometry.clone()
        return result


def test_joint_sampler_replaces_clean_geometry_with_independent_noise() -> None:
    target = Data(
        pos=torch.zeros((3, 3)),
        cell=torch.eye(3).unsqueeze(0),
        species=torch.ones(3, dtype=torch.long),
        n_atoms=torch.tensor([3]),
        batch=torch.zeros(3, dtype=torch.long),
        ptr=torch.tensor([0, 3]),
        geometry=torch.full((3, 4), 9.0),
    )
    torch.manual_seed(12)
    sampled = JointGeometrySampler(MirrorSampler(), geometry_dimension=4).sample_p_0(target)
    assert sampled.geometry.shape == (3, 4)
    assert not torch.equal(sampled.geometry, target.geometry)
    assert abs(float(sampled.geometry.mean())) < 1.0
    assert not hasattr(sampled, "geometry_oracle_target")
    assert not hasattr(sampled, "geometry_target")


def test_joint_sampler_rejects_clean_target_fields() -> None:
    target = Data(
        pos=torch.zeros((2, 3)),
        cell=torch.eye(3).unsqueeze(0),
        species=torch.ones(2, dtype=torch.long),
        n_atoms=torch.tensor([2]),
        batch=torch.zeros(2, dtype=torch.long),
        ptr=torch.tensor([0, 2]),
        geometry=torch.ones((2, 4)),
    )
    sampler = JointGeometrySampler(LeakySampler(), geometry_dimension=4)
    with pytest.raises(ValueError, match="forbidden clean-target fields"):
        sampler.sample_p_0(target)


def test_paired_inference_draw_overrides_callback_seed(monkeypatch) -> None:
    monkeypatch.setenv("JG_INFERENCE_DRAW", "4")
    assert PairedInferenceSeed(seed=0).seed == 4


def test_fixed_training_seed_replays_the_same_draw() -> None:
    callback = FixedTrainingSeed(seed=17)
    callback.on_train_batch_start(None, None, None, 0)
    first = torch.rand(4)
    callback.on_train_batch_start(None, None, None, 0)
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_oracle_integration_follows_the_known_descriptor_path() -> None:
    target = Data(
        pos=torch.zeros((2, 3)),
        cell=torch.eye(3).unsqueeze(0),
        species=torch.ones(2, dtype=torch.long),
        n_atoms=torch.tensor([2]),
        batch=torch.zeros(2, dtype=torch.long),
        ptr=torch.tensor([0, 2]),
        geometry=torch.full((2, 4), 3.0),
    )
    torch.manual_seed(1)
    x_0 = OracleGeometrySampler(MirrorSampler(), geometry_dimension=4).sample_p_0(target)
    base = x_0.geometry.clone()
    collection = OracleGeometryInterpolants(
        stochastic_interpolants=[linear_si()],
        data_fields=["geometry"],
        integration_time_steps=7,
        enable_progress_bar=False,
    )

    def wrong_model(state: Data, time: torch.Tensor) -> Data:
        return Data(
            geometry_b=torch.zeros_like(state.geometry),
            geometry_eta=torch.zeros_like(state.geometry),
        )

    generated = collection.integrate(x_0, wrong_model)
    expected = base + 0.998 * (target.geometry - base)
    assert torch.allclose(generated.geometry, expected, atol=1.0e-6)
