"""Lightning CLI for motif-census conditioned crystal flow matching."""

import argparse

from omg.omg_cli import OMGCLI
from omg.omg_lightning import OMGLightning
from omg.omg_trainer import OMGTrainer

from .data import CensusDataModule


def main() -> None:
    OMGCLI(
        model_class=OMGLightning,
        datamodule_class=CensusDataModule,
        trainer_class=OMGTrainer,
        save_config_callback=None,
        parser_kwargs={
            "formatter_class": argparse.RawDescriptionHelpFormatter,
            "description": "Global motif-census conditioning for OMatG crystal generation.",
        },
    )


if __name__ == "__main__":
    main()
