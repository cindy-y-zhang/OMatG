"""
Command line interface for the coarse-to-fine arms.

OMatG's own entry point pins the model and data module classes, so this module re-runs the same CLI with the two
subclasses that the experiment needs. Every subcommand and every argument link of the OMatG CLI is inherited unchanged,
so anything documented for the omg command works here as well.

Usage mirrors the omg command, for example:

    python -m cgfm.main fit --config cgfm/configs/base_mpts52.yaml --config cgfm/configs/arm_shells.yaml
    python -m cgfm.main predict --config ... --ckpt_path ... --model.generation_xyz_filename generated.xyz
    python -m cgfm.main csp_metrics --config ... --xyz_file generated.xyz
"""

import argparse
from omg.omg_cli import OMGCLI
from omg.omg_trainer import OMGTrainer
from .data import CGDataModule
from .lightning import CGLightning


def main() -> None:
    """Run the coarse-to-fine command line interface."""
    OMGCLI(model_class=CGLightning, datamodule_class=CGDataModule, trainer_class=OMGTrainer,
           save_config_callback=None,
           parser_kwargs={"formatter_class": argparse.RawDescriptionHelpFormatter,
                          "description": "Coarse-to-fine flow matching for crystal structure prediction."})


if __name__ == "__main__":
    main()
