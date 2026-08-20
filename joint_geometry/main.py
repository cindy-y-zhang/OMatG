"""Lightning CLI for joint geometry-state crystal flow matching."""

import argparse

from omg.omg_cli import OMGCLI
from omg.omg_trainer import OMGTrainer

from .data import GeometryDataModule
from .lightning import JointGeometryLightning


def main() -> None:
    OMGCLI(
        model_class=JointGeometryLightning,
        datamodule_class=GeometryDataModule,
        trainer_class=OMGTrainer,
        save_config_callback=None,
        parser_kwargs={
            "formatter_class": argparse.RawDescriptionHelpFormatter,
            "description": "Joint structure and local-geometry flow matching for OMatG.",
        },
    )


if __name__ == "__main__":
    main()
