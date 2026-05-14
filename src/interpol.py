#!/usr/bin/env python
__author__ =    'Christos Margadji'
__credits__ =   'Sebastian Pattinson'
__copyright__ = '2024, University of Cambridge, Computer-aided Manufacturing Group'
__email__ =     'cm2161@cam.ac.uk'

import numpy as np
import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pyDOE import lhs

# local dependencies
from src.utils import Logger
from src.sine import SineLayer

class Interpol(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.max_epochs = self.config.training.epochs
        self.lr = self.config.training.learning_rate
        self.gdir = self.config.training.gdir
        self.gpus = self.config.training.gpus
        self.chamfer_sdf_threshold = getattr(self.config.data, "chamfer_sdf_threshold", None) or 0.01
        self.chamfer_max_points = getattr(self.config.data, "chamfer_max_points", None) or 5000
        self.render_config = getattr(self.config, "render", None)
        self.ground_truth_render_data = None
        self.ground_truth_render_cache = {}

        self.net = []
        self.net.append(SineLayer(config.model.inputs, config.model.hidden, 
                                  is_first=True, omega_0= config.model.first_omega_0))

        for _ in range(config.model.n_hidden):
            self.net.append(SineLayer(config.model.hidden, config.model.hidden, 
                                      is_first=False, omega_0=30.))

        if config.model.outermost_linear:
            final_linear = nn.Linear(config.model.hidden, config.model.outputs)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / config.model.hidden) / config.model.hidden_omega_0, 
                                              np.sqrt(6 / config.model.hidden) / config.model.hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(config.model.hidden, config.model.outputs, 
                                      is_first=False, omega_0= config.model.hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
        self.save_hyperparameters()

    def get_render_values(self, key, default):
        if self.render_config is None:
            return default

        values = getattr(self.render_config, key, default)
        if values is None:
            return []
        if isinstance(values, (int, float)):
            return [float(values)]

        return [float(value) for value in values]

    def forward(self, space, params):
        input= torch.concat([space, params], axis=1)
        output = self.net(input)
        return output

    def predict_from_input(self, X: torch.Tensor) -> torch.Tensor:
        spatial_features, params = X[:, :3], X[:, 3:]
        return self.forward(spatial_features, params)

    def configure_optimizers(self):
        optimizer = torch.optim.Rprop(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.training.epochs)  # T_max is the number of epochs
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def reconstruction_loss(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        Y_hat_recon = self.predict_from_input(X)
        return F.mse_loss(Y, Y_hat_recon)

    def collect_chamfer_points(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        Y_hat: torch.Tensor
    ) -> None:
        spatial_features = X[:, :3].detach()
        true_surface_mask = torch.abs(Y.squeeze(-1)) <= self.chamfer_sdf_threshold
        pred_surface_mask = torch.abs(Y_hat.squeeze(-1)) <= self.chamfer_sdf_threshold

        if torch.any(true_surface_mask):
            self.true_chamfer_points.append(spatial_features[true_surface_mask])
        if torch.any(pred_surface_mask):
            self.pred_chamfer_points.append(spatial_features[pred_surface_mask])

    def sample_chamfer_points(self, points):
        if not points:
            return None

        points = torch.cat(points, dim=0)
        if len(points) > self.chamfer_max_points:
            indices = torch.randperm(len(points), device=points.device)[:self.chamfer_max_points]
            points = points[indices]

        return points

    def chamfer_distance(self, pred_points, true_points):
        pred_to_true = torch.cdist(pred_points, true_points).min(dim=1).values.square().mean()
        true_to_pred = torch.cdist(true_points, pred_points).min(dim=1).values.square().mean()
        return pred_to_true + true_to_pred

    def loss_function(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss function, including reconstruction loss and optional regularization.

        Parameters:
        - X (torch.Tensor): Input tensor of shape [batch_size, input_dim].
          First part contains spatial features, second part contains parameters.
        - Y (torch.Tensor): Ground truth tensor of shape [batch_size, output_dim].

        Returns:
        - torch.Tensor: Total loss (reconstruction + regularization).
        """
        # Compute reconstruction loss
        reconstruction_loss = self.reconstruction_loss(X, Y)

        # Initialize regularization term
        regularization_term = torch.tensor(0.0, device=X.device)

        if self.gdir > 0:
            # Generate Latin Hypercube Samples
            lhs_samples = torch.tensor(lhs(4, samples=10000), dtype=torch.float, device=X.device)
            space, params = lhs_samples[:, :-1], lhs_samples[:, -1:].requires_grad_(True)

            # Forward pass for regularization term
            Y_hat_regularization = self.forward(space, params)

            # Compute gradient of Y_hat w.r.t. parameters
            dS_dp = grad(
                outputs=Y_hat_regularization,
                inputs=params,
                grad_outputs=torch.ones_like(Y_hat_regularization),
                create_graph=True
            )[0]

            # Compute the regularization term (squared gradient norm)
            regularization_term = torch.norm(dS_dp, p=2) ** 2

        # Compute total loss
        total_loss = reconstruction_loss + self.gdir * regularization_term

        # Logging (only on rank 0 if distributed training is enabled)
        if self.training and getattr(self, 'global_rank', 0) == 0:
            self.logging_device.log("reconstruction_loss", reconstruction_loss.item())
            self.logging_device.log("regularization_term", (self.gdir * regularization_term).item())

        self.log(
            "loss",
            total_loss*1000,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            reduce_fx="mean",
            sync_dist=True,
        )

        return total_loss

    def on_fit_start(self):
        if self.global_rank==0:
            self.logging_device= Logger(
                batch=self.config.training.batch, 
                log_dir=self.trainer.logger.log_dir,
                v_num=self.trainer.logger.version,
            )

    def training_step(self, batch, batch_idx):
        input, output = batch
        loss = self.loss_function(input, output)

        if self.global_rank==0 and batch_idx%1==0 and batch_idx!=0:
            self.logging_device.report_running_mean(plot=False)

        return loss

    def on_validation_epoch_start(self):
        self.pred_chamfer_points = []
        self.true_chamfer_points = []

    def validation_step(self, batch, batch_idx):
        input, output = batch
        prediction = self.predict_from_input(input)
        loss = F.mse_loss(output, prediction)
        self.collect_chamfer_points(input, output, prediction)
        self.log(
            "val_loss",
            loss*1000,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            reduce_fx="mean",
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_end(self):
        pred_points = self.sample_chamfer_points(self.pred_chamfer_points)
        true_points = self.sample_chamfer_points(self.true_chamfer_points)

        if pred_points is None or true_points is None:
            if self.global_rank == 0:
                print(
                    "Skipping val_chamfer_distance because no predicted or true "
                    "surface points were found. Increase data.chamfer_sdf_threshold."
                )
            return

        chamfer_distance = self.chamfer_distance(pred_points, true_points)
        self.log(
            "val_chamfer_distance",
            chamfer_distance,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

    def on_train_epoch_end(self, unused=None):
        current_lr = self.optimizers().param_groups[0]["lr"]
        if self.global_rank == 0:
            print(f"Current Learning Rate (opt1): {current_lr}")
            self.log_rendered_sdf_slices()

    def should_log_renders(self):
        if self.render_config is None:
            return False

        enabled = getattr(self.render_config, "enabled", False)
        if not enabled:
            return False

        every_n_epochs = max(1, int(getattr(self.render_config, "every_n_epochs", 1)))
        return (self.current_epoch + 1) % every_n_epochs == 0

    def render_output_dir(self):
        log_dir = getattr(self.trainer.logger, "log_dir", None)
        if log_dir is None:
            return None

        output_dir = os.path.join(log_dir, "renders")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @torch.no_grad()
    def predict_render_slice(self, z_value, flowrate, image_size, batch_size):
        device = self.device
        x = torch.linspace(0, 299, image_size, device=device)
        y = torch.linspace(0, 299, image_size, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        x_norm = xx.flatten() / 299
        y_norm = yy.flatten() / 299
        z_norm = torch.full_like(x_norm, z_value / 337)
        flowrate_norm = torch.full_like(x_norm, (flowrate - 45) / (280 - 45))
        render_input = torch.stack([x_norm, y_norm, z_norm, flowrate_norm], dim=1)

        predictions = []
        for start in range(0, len(render_input), batch_size):
            predictions.append(self.predict_from_input(render_input[start:start + batch_size]))

        return torch.cat(predictions, dim=0).reshape(image_size, image_size).detach().float().cpu().numpy()

    def ground_truth_dataset(self):
        if self.ground_truth_render_data is None:
            with h5py.File(self.config.data.dataroot, "r") as dataset:
                self.ground_truth_render_data = np.array(dataset["dataset"])

        return self.ground_truth_render_data

    def ground_truth_render_slice(self, z_value, flowrate, image_size):
        cache_key = (float(z_value), float(flowrate), int(image_size))
        if cache_key in self.ground_truth_render_cache:
            return self.ground_truth_render_cache[cache_key]

        dataset = self.ground_truth_dataset()
        flowrates = np.unique(dataset[:, 3])
        ground_truth_flowrate = flowrates[np.abs(flowrates - flowrate).argmin()]
        flowrate_mask = np.isclose(dataset[:, 3], ground_truth_flowrate)

        z_values = np.unique(dataset[flowrate_mask, 2])
        ground_truth_z = z_values[np.abs(z_values - z_value).argmin()]
        slice_data = dataset[flowrate_mask & np.isclose(dataset[:, 2], ground_truth_z)]

        image_sum = np.zeros((image_size, image_size), dtype=np.float32)
        image_count = np.zeros((image_size, image_size), dtype=np.float32)
        x_idx = np.clip(np.rint(slice_data[:, 0] / 299 * (image_size - 1)).astype(int), 0, image_size - 1)
        y_idx = np.clip(np.rint(slice_data[:, 1] / 299 * (image_size - 1)).astype(int), 0, image_size - 1)

        np.add.at(image_sum, (y_idx, x_idx), slice_data[:, 4])
        np.add.at(image_count, (y_idx, x_idx), 1)

        image = np.full((image_size, image_size), np.nan, dtype=np.float32)
        populated_pixels = image_count > 0
        image[populated_pixels] = image_sum[populated_pixels] / image_count[populated_pixels]

        result = image, ground_truth_z, ground_truth_flowrate
        self.ground_truth_render_cache[cache_key] = result
        return result

    def plot_sdf_slice(self, ax, image, title, vmin, vmax):
        plotted_image = ax.imshow(
            image,
            cmap="coolwarm",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        finite_values = image[np.isfinite(image)]
        if len(finite_values) > 0 and finite_values.min() <= 0 <= finite_values.max():
            ax.contour(image, levels=[0], colors="black", linewidths=0.5)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return plotted_image

    def log_rendered_sdf_slices(self):
        if not self.should_log_renders():
            return

        output_dir = self.render_output_dir()
        if output_dir is None:
            return

        z_values = self.get_render_values("z_values", [84, 168, 252])
        flowrates = self.get_render_values("flowrates", [45, 100, 180, 280])
        if not z_values or not flowrates:
            return

        image_size = int(getattr(self.render_config, "image_size", 150))
        batch_size = int(getattr(self.render_config, "batch_size", 65536))
        sdf_clip = getattr(self.render_config, "sdf_clip", None)
        vmin = -float(sdf_clip) if sdf_clip is not None else None
        vmax = float(sdf_clip) if sdf_clip is not None else None
        include_ground_truth = getattr(self.render_config, "include_ground_truth", True)
        columns_per_z = 2 if include_ground_truth else 1

        was_training = self.training
        self.eval()

        fig, axes = plt.subplots(
            len(flowrates),
            len(z_values) * columns_per_z,
            figsize=(3 * len(z_values) * columns_per_z, 3 * len(flowrates)),
            squeeze=False,
            constrained_layout=True,
        )

        image = None
        try:
            for row, flowrate in enumerate(flowrates):
                for col, z_value in enumerate(z_values):
                    prediction = self.predict_render_slice(z_value, flowrate, image_size, batch_size)
                    pred_col = col * columns_per_z
                    image = self.plot_sdf_slice(
                        axes[row][pred_col],
                        prediction,
                        f"Pred z={z_value:g}, FR={flowrate:g}",
                        vmin,
                        vmax,
                    )

                    if include_ground_truth:
                        ground_truth, ground_truth_z, ground_truth_flowrate = self.ground_truth_render_slice(
                            z_value,
                            flowrate,
                            image_size,
                        )
                        image = self.plot_sdf_slice(
                            axes[row][pred_col + 1],
                            ground_truth,
                            f"GT z={ground_truth_z:g}, FR={ground_truth_flowrate:g}",
                            vmin,
                            vmax,
                        )

            title = "Rendered SDF and ground-truth slices" if include_ground_truth else "Rendered SDF slices"
            fig.suptitle(f"{title}, epoch {self.current_epoch + 1}")
            if image is not None:
                fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)

            output_path = os.path.join(output_dir, f"render_epoch_{self.current_epoch + 1:04d}.png")
            fig.savefig(output_path, dpi=160)
        finally:
            plt.close(fig)
            if was_training:
                self.train()
