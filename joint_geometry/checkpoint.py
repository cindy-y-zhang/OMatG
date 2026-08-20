"""Strict warm-start helpers for paired local geometry experiments."""

from pathlib import Path
import random
import json
import os
import time

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer

from direct_geometry.encoder import baseline_encoder_state


def load_backbone_checkpoint(
    encoder: torch.nn.Module,
    checkpoint_path: str,
    prefix: str = "model.encoder.",
) -> None:
    """Load only baseline encoder weights through the encoder's audited importer."""
    importer = getattr(encoder, "load_baseline_state_dict", None)
    if importer is None:
        raise TypeError(f"{type(encoder).__name__} has no strict baseline checkpoint importer.")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"No baseline checkpoint at {path}.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    importer(baseline_encoder_state(checkpoint, prefix=prefix))


class LoadBackboneCheckpoint(Callback):
    """Warm-start an encoder once while leaving every optimizer fresh."""

    def __init__(
        self,
        checkpoint_path: str,
        prefix: str = "model.encoder.",
    ) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.prefix = prefix
        self._loaded = False

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._loaded:
            return
        load_backbone_checkpoint(
            pl_module.model.encoder,
            self.checkpoint_path,
            prefix=self.prefix,
        )
        self._loaded = True


class PairedInferenceSeed(Callback):
    """Reset all sampler RNGs before each validation batch for paired priors."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        override = os.environ.get("JG_INFERENCE_DRAW")
        self.seed = int(override) if override is not None else int(seed)

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        value = self.seed + 1_000_003 * int(batch_idx)
        random.seed(value)
        np.random.seed(value % (2**32))
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)


class FixedTrainingSeed(Callback):
    """Replay one base draw and time sample in the memorization diagnostic."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = int(seed)

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed % (2**32))
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


class FinalCheckpoint(Callback):
    """Save exactly one resumable checkpoint after a bounded screen."""

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        log_dir = getattr(trainer.logger, "log_dir", None)
        root = Path(log_dir) if log_dir is not None else Path(trainer.default_root_dir)
        destination = root / "checkpoints" / "last.ckpt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(destination, weights_only=False)


class ResourceMetrics(Callback):
    """Record wall time and accelerator peak allocation for a bounded run."""

    def __init__(self, filename: str = "RESOURCE.json") -> None:
        super().__init__()
        self.filename = filename
        self.started: float | None = None

    def _start(self) -> None:
        self.started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _write(self, trainer: Trainer) -> None:
        if self.started is None:
            return
        log_dir = getattr(trainer.logger, "log_dir", None)
        root = Path(log_dir) if log_dir is not None else Path(trainer.default_root_dir)
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "elapsed_seconds": time.perf_counter() - self.started,
            "cuda_peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            ),
            "global_step": trainer.global_step,
        }
        (root / self.filename).write_text(json.dumps(payload, indent=2))

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._start()

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._write(trainer)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._start()

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._write(trainer)
