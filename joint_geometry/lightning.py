"""Lightning module for joint structural and geometry flow matching."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from ase import Atoms

from omg.datamodule import OMGData
from omg.omg_lightning import OMGLightning
from omg.sampler.minimum_permutation_distance import correct_for_minimum_permutation_distance
from omg.utils import DataField, reshape_t

from .descriptor import DescriptorTransform, transformed_clean_descriptor


TARGET_GRADIENT_RATIO = 0.0707
GRADIENT_RATIO_BAND = (0.05, 0.10)
CALIBRATION_BATCHES = 20
CALIBRATION_INTERVAL = 500


class JointGeometryLightning(OMGLightning):
    """
    Append geometry velocity matching without rescaling OMatG's tuned losses.

    The stochastic-interpolant cost for ``geometry_loss_b`` must be zero.  This
    class reads the unweighted term and adds it with a separately calibrated
    coefficient after constructing the unchanged structural objective.
    """

    def __init__(
        self,
        *args,
        geometry_weight: Optional[float] = None,
        **kwargs,
    ) -> None:
        if geometry_weight is not None and geometry_weight <= 0.0:
            raise ValueError("geometry_weight must be positive when supplied.")
        super().__init__(*args, **kwargs)
        if "geometry_loss_b" not in self._relative_si_costs:
            raise ValueError("JointGeometryLightning requires a geometry stochastic-interpolant field.")
        if self._relative_si_costs["geometry_loss_b"] != 0.0:
            raise ValueError(
                "geometry_loss_b must have zero base cost; its calibrated loss is appended separately."
            )
        structural_sum = sum(
            value
            for key, value in self._relative_si_costs.items()
            if key != "geometry_loss_b"
        )
        if not np.isclose(structural_sum, 1.0):
            raise ValueError("The unchanged structural loss weights must sum to one.")
        self._geometry_weight = geometry_weight
        self._calibrated = geometry_weight is not None
        self._calibration_ratios: list[float] = []
        self.descriptor_transform: DescriptorTransform | None = None

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["joint_geometry_calibration"] = {
            "weight": self._geometry_weight,
            "calibrated": self._calibrated,
            "ratios": self._calibration_ratios,
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        calibration = checkpoint.get("joint_geometry_calibration")
        if calibration is None:
            return
        self._geometry_weight = calibration.get("weight")
        self._calibrated = bool(calibration.get("calibrated", False))
        self._calibration_ratios = list(calibration.get("ratios", []))

    def setup(self, stage: Optional[str] = None) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        transform = getattr(datamodule, "transform", None) if datamodule is not None else None
        if transform is not None:
            if not isinstance(transform, DescriptorTransform):
                raise TypeError("The geometry datamodule exposes an invalid descriptor transform.")
            encoder_dimension = getattr(self.model.encoder, "geometry_dimension", None)
            if encoder_dimension != transform.dimension:
                raise ValueError(
                    f"The encoder expects {encoder_dimension} geometry channels but the artifacts "
                    f"contain {transform.dimension}."
                )
            self.descriptor_transform = transform

    def _trunk_parameters(self) -> list[torch.nn.Parameter]:
        excluded = (
            "geometry_out",
            "joint_geometry_projection",
            "geometry_projection",
            "coord_out",
            "lattice_out",
            "type_out",
        )
        return [
            parameter
            for name, parameter in self.model.encoder.named_parameters()
            if parameter.requires_grad and not any(name.startswith(prefix) for prefix in excluded)
        ]

    def _gradient_ratio(
        self, flow_loss: torch.Tensor, geometry_loss: torch.Tensor
    ) -> Optional[float]:
        if not flow_loss.requires_grad or not geometry_loss.requires_grad:
            return None
        parameters = self._trunk_parameters()
        if not parameters:
            return None
        flow_norm = _gradient_norm(flow_loss, parameters)
        geometry_norm = _gradient_norm(geometry_loss, parameters)
        if flow_norm is None or geometry_norm is None or geometry_norm <= 0.0:
            return None
        return flow_norm / geometry_norm

    def _geometry_weight_for(
        self, flow_loss: torch.Tensor, geometry_loss: torch.Tensor
    ) -> torch.Tensor:
        if self._calibrated:
            if self.global_step % CALIBRATION_INTERVAL == 0:
                ratio = self._gradient_ratio(flow_loss, geometry_loss)
                if ratio is not None and ratio > 0.0:
                    achieved = float(self._geometry_weight) / ratio
                    self.log(
                        "geometry_gradient_ratio",
                        achieved,
                        on_step=True,
                        on_epoch=False,
                        sync_dist=True,
                    )
                    if not GRADIENT_RATIO_BAND[0] <= achieved <= GRADIENT_RATIO_BAND[1]:
                        self.print(
                            f"step {self.global_step}: geometry gradient share {achieved:.4f} "
                            f"is outside {GRADIENT_RATIO_BAND}."
                        )
            return torch.tensor(float(self._geometry_weight), device=self.device)

        ratio = self._gradient_ratio(flow_loss, geometry_loss)
        if ratio is None:
            return torch.tensor(0.0, device=self.device)
        self._calibration_ratios.append(ratio)
        # The monitored share is weight * ||grad L_geometry|| / ||grad L_flow||
        # = weight / ratio.  Calibrate that quantity directly; using the
        # arithmetic mean of ``ratio`` biases the achieved share upward when
        # gradient norms vary substantially between batches.
        self._geometry_weight = TARGET_GRADIENT_RATIO / float(
            np.mean([1.0 / value for value in self._calibration_ratios])
        )
        if len(self._calibration_ratios) >= CALIBRATION_BATCHES:
            self._calibrated = True
            self.print(
                f"Calibrated geometry weight to {self._geometry_weight:.6g} over "
                f"{len(self._calibration_ratios)} batches."
            )
        return torch.tensor(float(self._geometry_weight), device=self.device)

    def _joint_losses(
        self,
        x_0: OMGData,
        x_1: OMGData,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        time = self.time_sampler(len(x_1.n_atoms)).to(self.device)
        captured: dict[str, torch.Tensor] = {}

        def model_function(state: OMGData, model_time: torch.Tensor):
            result = self.model(state, model_time)
            if "state" not in captured:
                captured["state"] = state.geometry
                captured["time"] = model_time
                captured["velocity"] = result.geometry_b
            return result

        raw = self.si.losses(model_function, time, x_0, x_1)
        if "geometry_loss_b" not in raw:
            raise KeyError("The stochastic interpolants returned no geometry_loss_b.")
        geometry_loss = raw["geometry_loss_b"]
        flow_loss = sum(
            self._relative_si_costs[key] * value
            for key, value in raw.items()
            if key != "geometry_loss_b"
        )
        weight = self._geometry_weight_for(flow_loss, geometry_loss)
        total = flow_loss + weight * geometry_loss

        target_velocity = x_1.geometry - x_0.geometry
        velocity_error = (
            captured["velocity"] - target_velocity
        ).square().mean(dim=tuple(range(1, target_velocity.dim())))
        geometry_mse = velocity_error.mean()
        denoised = captured["state"] + (
            1.0
            - reshape_t(
                captured["time"],
                x_1.n_atoms,
                DataField.geometry,
                captured["state"].shape,
            )
        ) * captured["velocity"]
        endpoint_mse = (denoised - x_1.geometry).square().mean()
        logged = {
            **{
                key: (self._relative_si_costs[key] * value).detach()
                for key, value in raw.items()
                if key != "geometry_loss_b"
            },
            "geometry_loss_b": geometry_loss.detach(),
            "geometry_velocity_mse": geometry_mse.detach(),
            "geometry_clean_estimate_mse": endpoint_mse.detach(),
            "geometry_weight": weight.detach(),
            "loss_total": total,
        }
        atom_time = captured["time"].repeat_interleave(x_1.n_atoms)
        endpoint_error = (denoised - x_1.geometry).square().mean(
            dim=tuple(range(1, x_1.geometry.dim()))
        )
        for lower_index in range(5):
            lower = lower_index / 5.0
            upper = (lower_index + 1) / 5.0
            selected = (atom_time >= lower) & (
                atom_time <= upper if lower_index == 4 else atom_time < upper
            )
            denominator = selected.sum().clamp(min=1)
            label = f"t{int(100 * lower):02d}_{int(100 * upper):03d}"
            logged[f"geometry_velocity_mse_{label}"] = (
                velocity_error[selected].sum() / denominator
            ).detach()
            logged[f"geometry_clean_estimate_mse_{label}"] = (
                endpoint_error[selected].sum() / denominator
            ).detach()
            logged[f"geometry_atoms_{label}"] = selected.sum().float().detach()
        return total, logged

    def training_step(self, x_1: OMGData) -> torch.Tensor:
        x_0 = self.sampler.sample_p_0(x_1).to(self.device)
        if self.use_min_perm_dist:
            correct_for_minimum_permutation_distance(
                x_0, x_1, self._pos_corrector, switch_species=False
            )
        total, losses = self._joint_losses(x_0, x_1)
        self.log_dict(
            losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(x_1.n_atoms),
        )
        return total

    def validation_step(self, x_1: OMGData) -> torch.Tensor:
        batch_size = len(x_1.n_atoms)
        x_0 = self.sampler.sample_p_0(x_1).to(self.device)
        generated = None
        if self._validation_metric in (
            self.ValidationMetric.MATCH_RATE,
            self.ValidationMetric.METRE,
            self.ValidationMetric.DNG_EVAL,
        ):
            x_1_cpu = x_1.clone().to("cpu")
            for index in range(batch_size):
                lower, upper = x_1_cpu.ptr[index], x_1_cpu.ptr[index + 1]
                self.reference_atoms.append(
                    Atoms(
                        numbers=x_1_cpu.species[lower:upper],
                        scaled_positions=x_1_cpu.pos[lower:upper],
                        cell=x_1_cpu.cell[index],
                        pbc=(1, 1, 1),
                    )
                )
            generated = self.si.integrate(x_0, self.model, save_intermediate=False)
            if self.descriptor_transform is not None:
                try:
                    recomputed = transformed_clean_descriptor(
                        generated.pos,
                        generated.cell,
                        generated.n_atoms,
                        self.descriptor_transform,
                    )
                    consistency = (generated.geometry - recomputed).square().mean()
                    descriptor_failure = torch.tensor(0.0, device=self.device)
                except (ValueError, RuntimeError) as error:
                    self.print(f"Generated endpoint descriptor failed: {error}")
                    consistency = torch.tensor(float("inf"), device=self.device)
                    descriptor_failure = torch.tensor(1.0, device=self.device)
                self.log(
                    "val_geometry_endpoint_consistency",
                    consistency,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
                self.log(
                    "val_geometry_descriptor_failure",
                    descriptor_failure,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
            generated.to("cpu")
            for index in range(batch_size):
                lower, upper = generated.ptr[index], generated.ptr[index + 1]
                self.generated_atoms.append(
                    Atoms(
                        numbers=generated.species[lower:upper],
                        scaled_positions=generated.pos[lower:upper],
                        cell=generated.cell[index],
                        pbc=(1, 1, 1),
                    )
                )
        elif self._validation_metric != self.ValidationMetric.LOSS:
            raise ValueError(f"Unsupported validation metric {self._validation_metric}.")

        if self.use_min_perm_dist:
            correct_for_minimum_permutation_distance(
                x_0, x_1, self._pos_corrector, switch_species=False
            )
        total, losses = self._joint_losses(x_0, x_1)
        validation_losses = {
            (key if key.startswith("val_") else f"val_{key}"): value
            for key, value in losses.items()
        }
        self.log_dict(
            validation_losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return total


def _gradient_norm(
    loss: torch.Tensor, parameters: list[torch.nn.Parameter]
) -> Optional[float]:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
        create_graph=False,
    )
    present = [gradient for gradient in gradients if gradient is not None]
    if not present:
        return None
    return float(
        torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(gradient) for gradient in present])
        )
    )
