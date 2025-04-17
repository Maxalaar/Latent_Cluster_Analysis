import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.autoencoder import Autoencoder
from architecture.loss.distance_centroid_loss import DistanceCentroidLoss
from architecture.loss.kmeans import Kmeans

class ClusteringAutoencoder(Autoencoder):
    def __init__(
        self,
        input_shape,
        output_shape,
        activation_function_class=nn.LeakyReLU,
        encoder_configuration=[64, 32, 16],
        latent_space_size=8,
        decoder_configuration=[16, 32, 64],
        leaning_rate=1e-3,
        number_cluster: int = 2,
        margin_between_clusters: float = 1.0,
        number_centroids_repulsion: int = 1,
        memory_size: int = 0,
        reconstruction_loss_coefficient: float = 1.0,
        clustering_loss_coefficient: float = 1.0,
    ):
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            activation_function_class=activation_function_class,
            encoder_configuration=encoder_configuration,
            latent_space_size=latent_space_size,
            decoder_configuration=decoder_configuration,
            leaning_rate=leaning_rate,
        )
        self.embeddings = None

        self.reconstruction_loss_coefficient = reconstruction_loss_coefficient
        self.clustering_loss_coefficient = clustering_loss_coefficient

        self.kmeans = Kmeans(
            number_cluster=number_cluster,
            memory_size=memory_size,
            logger=self.log,
        )

        self.distance_centroid_loss = DistanceCentroidLoss(
            margin_between_clusters=margin_between_clusters,
            number_centroids_repulsion=number_centroids_repulsion,
            logger=self.log,
        )

    def forward(self, x):
        self.embeddings = self.encode(x)
        x_hat = self.decoder(self.embeddings)
        return x_hat

    def loss_function(self, x_hat, x):
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')
        clustering_result = self.kmeans.forward(embeddings=self.embeddings)
        clustering_loss = self.distance_centroid_loss(
            embeddings=self.embeddings,
            **clustering_result,
        )
        self.log('reconstruction_loss', reconstruction_loss, on_step=True, on_epoch=True)
        return self.reconstruction_loss_coefficient * reconstruction_loss + self.clustering_loss_coefficient * clustering_loss

    def step(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        return loss
