#!/usr/bin/env python
__author__ =    'Christos Margadji'
__credits__ =   'Sebastian Pattinson'
__copyright__ = '2024, University of Cambridge, Computer-aided Manufacturing Group'
__email__ =     'cm2161@cam.ac.uk'

import argparse
import re
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# local dependencies
from src.dataset.data_module import H5PYModule
from src.interpol import Interpol


def to_namespace(d):
    if isinstance(d, dict):
        return argparse.Namespace(**{k: to_namespace(v) for k, v in d.items()})
    return d


def natural_key(path):
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", path.name)
    ]


def load_config(config_path):
    with open(config_path, "r") as file:
        return to_namespace(yaml.safe_load(file))


def run_config(config, config_path):
    print(f"Running sweep config: {config_path}")

    data = H5PYModule(config=config)
    model = Interpol(config=config)
    validation_fr = getattr(config.data, "validation_fr", None)
    checkpoint_monitor = "val_loss" if validation_fr is not None else "loss"
    checkpoint_filename = (
        f"{config.name}_" + "{epoch:02d}-{val_loss:.2f}"
        if validation_fr is not None
        else f"{config.name}_" + "{epoch:02d}-{loss:.2f}"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_monitor,
        filename=checkpoint_filename,
        save_top_k=1,
    )

    logger = CSVLogger("logs/train", name=Path(config_path).stem)

    trainer = pl.Trainer(
        deterministic=False,
        devices=config.training.gpus if config.training.gpus > 0 else 1,
        accelerator="gpu" if config.training.gpus > 0 else "cpu",
        max_epochs=config.training.epochs,
        strategy="ddp_find_unused_parameters_false",
        logger=logger,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback],
        limit_train_batches=1.0,
        limit_val_batches=1.0 if validation_fr is not None else 0.0,
        precision="16-mixed",
        num_sanity_val_steps=0,  # Optional: skips sanity check to save time, can be removed
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all configs in a sweep folder.")
    parser.add_argument(
        "--sweep-dir",
        default="sweep",
        help="Directory containing YAML configs to run one by one.",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    config_paths = sorted(sweep_dir.glob("*.yaml"), key=natural_key)

    if not config_paths:
        raise FileNotFoundError(f"No YAML configs found in {sweep_dir}")

    for config_path in config_paths:
        config = load_config(config_path)
        run_config(config, config_path)
