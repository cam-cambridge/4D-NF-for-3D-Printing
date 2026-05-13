#!/usr/bin/env python
__author__ =    'Christos Margadji'
__credits__ =   'Sebastian Pattinson'
__copyright__ = '2024, University of Cambridge, Computer-aided Manufacturing Group'
__email__ =     'cm2161@cam.ac.uk'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import pytorch_lightning as pl
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
                v_num=self.trainer.logger.version
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