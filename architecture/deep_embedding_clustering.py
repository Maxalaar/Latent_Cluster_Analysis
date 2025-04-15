import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.autoencoder import Autoencoder
from architecture.loss.distance_centroid_loss import DistanceCentroidLoss
from architecture.loss.kmeans import Kmeans


class DeepEmbeddingClustering(Autoencoder):
    def __init__(
        self,
        input_shape,
        output_shape,
        activation_function_class=nn.LeakyReLU,
        encoder_configuration=[64, 32, 16],
        lantent_space_size=8,
        decoder_configuration=[16, 32, 64],
        leaning_rate=1e-3,
        number_cluster: int = 2,
        alpha: float = 1.0,
        reconstruction_loss_coefficient: float = 1.0,
        clustering_loss_coefficient: float = 1.0,
    ):
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            activation_function_class=activation_function_class,
            encoder_configuration=encoder_configuration,
            lantent_space_size=lantent_space_size,
            decoder_configuration=decoder_configuration,
            leaning_rate=leaning_rate,
        )
        self.embeddings = None

        self.alpha = alpha
        self.reconstruction_loss_coefficient = reconstruction_loss_coefficient
        self.clustering_loss_coefficient = clustering_loss_coefficient

        self.number_cluster = number_cluster
        self.centroids = nn.Parameter(torch.randn(self.number_cluster, lantent_space_size))

    def forward(self, x):
        self.embeddings = self.encode(x)
        x_hat = self.decoder(self.embeddings)
        return x_hat

    def loss_function(self, x_hat, x):
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')

        # q_{i,j}
        difference = self.embeddings[:, None, :] - self.centroids[None, :, :]
        euclidean_distance = torch.norm(difference, p=2, dim=-1)
        squared_distance = torch.pow(euclidean_distance, 2)

        power = -(self.alpha + 1) / 2
        value_actual_distribution = (1 + (squared_distance / self.alpha))
        value_actual_distribution_powered = torch.pow(value_actual_distribution, power)
        value_actual_distribution_sum = value_actual_distribution_powered.sum(dim=1)
        actual_distribution = value_actual_distribution_powered / value_actual_distribution_sum.unsqueeze(1)

        # p_{i,j}
        soft_cluster_frequencies = actual_distribution.sum(dim=0)
        value_target_distribution = torch.pow(actual_distribution, 2) / soft_cluster_frequencies
        value_target_distribution_sum = value_target_distribution.sum(dim=1)
        target_distribution = value_target_distribution / value_target_distribution_sum.unsqueeze(1)

        # KL
        clustering_loss = torch.sum(target_distribution * torch.log(target_distribution / actual_distribution))

        self.log('clustering_loss', clustering_loss, on_step=True, on_epoch=True)
        self.log('reconstruction_loss', reconstruction_loss, on_step=True, on_epoch=True)
        return self.reconstruction_loss_coefficient * reconstruction_loss + self.clustering_loss_coefficient * clustering_loss

    def step(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        return loss