"""
Command line interface for the rigid-block model.

OMatG's own entry point pins the model and data module classes, so this module re-runs the same CLI with the block
Lightning module and data module.

Usage:

    python -m cgfm.block_main fit --config cgfm/configs/block_oracle_mpts52.yaml
    python -m cgfm.block_main validate --config ... --ckpt_path ...
    python -m cgfm.block_main predict --config ... --ckpt_path ... --model.generation_xyz_filename generated.xyz
"""

import argparse
from omg.omg_cli import OMGCLI
from omg.omg_trainer import OMGTrainer
from .block_lightning import BlockLightning
from .blockdata import BlockDataModule


def main() -> None:
    """Run the rigid-block command line interface."""
    OMGCLI(model_class=BlockLightning, datamodule_class=BlockDataModule, trainer_class=OMGTrainer,
           save_config_callback=None,
           parser_kwargs={"formatter_class": argparse.RawDescriptionHelpFormatter,
                          "description": "Rigid-block flow matching for crystal structure prediction."})


if __name__ == "__main__":
    main()
