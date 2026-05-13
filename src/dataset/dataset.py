#!/usr/bin/env python
__author__ =    'Christos Margadji'
__credits__ =   'Sebastian Pattinson'
__copyright__ = '2024, University of Cambridge, Computer-aided Manufacturing Group'
__email__ =     'cm2161@cam.ac.uk'

import os
import cv2
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

class DirDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data_root = config.data.dataroot
        self.dim_reduction = config.data.dim_reduction
        self.image_paths = sorted([os.path.join(self.data_root, f) for f in os.listdir(self.data_root) if f.endswith('.jpg')], key=self.sort_func)
        self.images = []
        for image_path in tqdm(self.image_paths, desc="Loading images"):
            self.images.append(self.load(image_path))

    def __len__(self):
        return len(self.images)*self.images[0].shape[0]*self.images[0].shape[1]

    def __getitem__(self, idx):
        image_idx = idx // (self.new_height * self.new_width)
        pixel_idx = idx % (self.new_height * self.new_width)
        pixel_x = pixel_idx // self.new_width
        pixel_y = pixel_idx % self.new_width
        pixel_RGB= self.images[image_idx][pixel_x][pixel_y]
        normalised_x, normalised_y, normalised_t= self.domain_normalisation(pixel_x, pixel_y, image_idx) 

        return torch.tensor([normalised_x, normalised_y, normalised_t]).float(), torch.tensor(pixel_RGB).float()/255

    def sort_func(self, image_path):
        filename = os.path.basename(image_path)
        digit = int(''.join(filter(str.isdigit, filename)))
        return digit

    def domain_normalisation(self, pixel_x, pixel_y, image_idx):
        norm_x= pixel_y/self.new_width
        norm_y= pixel_x/self.new_height
        norm_t= image_idx/(len(self.images)-1)
        return norm_x, norm_y, norm_t

    def load(self, image_path):
        image = cv2.imread(image_path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.new_height = int(image.shape[0] / self.dim_reduction)
        self.new_width = int(image.shape[1] / self.dim_reduction)
        resized_image = cv2.resize(image, (self.new_width, self.new_height))
        return resized_image
    

class ThreeDimDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data_root = config.data.dataroot
        self.dim_reduction = config.data.dim_reduction

        self.volume_paths = [os.path.join(self.data_root, f) for f in os.listdir(self.data_root) if f.endswith('.png')]
        self.volumes = {}
        for volume_path in tqdm(self.volume_paths, desc="Loading volumes"):
            self.volumes.update({self.extract_parameter(volume_path): self.load(volume_path)})
        self.generate_points()
        self.measure_normalisation_boundaries()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample= self.domain_normalisation(self.data[idx])
        x= sample[:4]
        y= sample[4:]
        return torch.tensor(x).float(), torch.tensor(y) .float()

    def load(self, volume_dir, patch=600):
        ct_scan = cv2.imread(volume_dir)
        ct_scan = cv2.cvtColor(ct_scan, cv2.COLOR_BGR2GRAY)
        num_patches_x = ct_scan.shape[0] // patch
        num_patches_y = ct_scan.shape[1] // patch
        stacked_patches = np.zeros((num_patches_x * num_patches_y, patch, patch), dtype=np.uint8)
        for i in range(num_patches_x):
            for j in range(num_patches_y):
                patch_data = ct_scan[i*patch:(i+1)*patch, j*patch:(j+1)*patch]
                stacked_patches[i*num_patches_y + j] = patch_data
        return stacked_patches

    def extract_parameter(self, name):
        return int(''.join(filter(str.isdigit, name)))

    def generate_points(self):
        self.data = []
        for key, volume in self.volumes.items():
            volume=volume[::3,::3,::3]
            Z, Y, X = volume.shape
            z, y, x = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing='ij')
            points = np.stack((x.flatten(), y.flatten(), z.flatten(), np.full(volume.size, key), volume.flatten()), axis=1)
            self.data.append(points)
        self.data = np.concatenate(self.data, axis=0)

    def measure_normalisation_boundaries(self):
        self.min_vals = np.min(self.data, axis=0)
        self.max_vals = np.max(self.data, axis=0)
        print(self.min_vals, self.max_vals)

    def domain_normalisation(self, point):
        normalized_point = (point - self.min_vals) / (self.max_vals - self.min_vals)
        return normalized_point

class H5PYDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.data_root = config.data.dataroot
        self.split = split
        with h5py.File(self.data_root,'r') as dataset:
            self.dataset = np.array(dataset["dataset"])

        validation_fr = getattr(self.config.data, "validation_fr", None)
        if validation_fr is not None:
            validation_mask = np.isclose(self.dataset[:, 3], validation_fr)
            if self.split == "train":
                self.dataset = self.dataset[~validation_mask]
            elif self.split in {"val", "test"}:
                self.dataset = self.dataset[validation_mask]
            else:
                raise ValueError(f"Unknown H5PYDataset split: {self.split}")

        self.dataset = self.dataset[::self.config.data.dim_reduction]

        if len(self.dataset) == 0:
            raise ValueError(
                f"No samples found for split '{self.split}'"
                f" with validation_fr={validation_fr}"
            )

    def __len__(self):
        size = self.dataset.shape[0]
        return size

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        x = self.domain_normalisation(sample[:4])
        y = sample[4:]

        return torch.tensor(x).float(), torch.tensor(y).float()

    def domain_normalisation(self, point):
        point = np.array(point)
        min_vals = np.array([0, 0, 0, 45])
        max_vals = np.array([299, 299, 337, 280])
        normalized_point = (point - min_vals) / (max_vals - min_vals)
        return normalized_point
