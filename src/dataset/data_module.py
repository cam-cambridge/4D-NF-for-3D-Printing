#!/usr/bin/env python
__author__ =    'Christos Margadji'
__credits__ =   'Sebastian Pattinson'
__copyright__ = '2024, University of Cambridge, Computer-aided Manufacturing Group'
__email__ =     'cm2161@cam.ac.uk'

from PIL import ImageFile
from torch.utils.data import DataLoader
from src.dataset.dataset import DirDataset, ThreeDimDataset, H5PYDataset
from pytorch_lightning import LightningDataModule

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DirDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.train_dataset = DirDataset(config=self.config)
            self.val_dataset = DirDataset(config=self.config)

        if stage == "test" or stage is None:
            self.test_dataset = DirDataset(config=self.config)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch,
            num_workers=16,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
        )

class ThreeDimModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = ThreeDimDataset(config=self.config)
            self.val_dataset = ThreeDimDataset(config=self.config)

        if stage == "test" or stage is None:
            self.test_dataset = ThreeDimDataset(config=self.config)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch,
            num_workers=16,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
        )
    

class H5PYModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        validation_fr = getattr(self.config.data, "validation_fr", None)
        if stage == "fit" or stage is None:
            self.train_dataset = H5PYDataset(config=self.config, split="train")
            self.val_dataset = (
                H5PYDataset(config=self.config, split="val")
                if validation_fr is not None
                else None
            )

        if stage == "test" or stage is None:
            self.test_dataset = H5PYDataset(config=self.config, split="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch,
            num_workers=16,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
        )