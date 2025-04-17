import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta

from architecture.autoencoder import Autoencoder
from architecture.loss.kmeans import Kmeans

class DeepEmbeddingClustering(Autoencoder):
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
        alpha: float = 1.0,
        reconstruction_loss_coefficient: float = 1.0,
        clustering_loss_coefficient: float = 1.0,
        clustering_loss_delay=timedelta(minutes=0),
        max_saved_embeddings: int = 10_000,
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
        self.latent_space_size = latent_space_size
        self.embeddings = None
        self.alpha = alpha
        self.reconstruction_loss_coefficient = reconstruction_loss_coefficient
        self.clustering_loss_coefficient = clustering_loss_coefficient
        self.start_time = datetime.now()
        self.clustering_loss_delay = clustering_loss_delay

        self.kullback_leibler_divergence = nn.KLDivLoss(size_average=False)

        self.number_cluster = number_cluster
        self.centroids = nn.Parameter(torch.randn(self.number_cluster, self.latent_space_size, device='cuda'))
        self.embedding_buffer = []
        self.max_saved_embeddings = max_saved_embeddings
        self.initialisation_centroid_done = False
        self.kmeans = Kmeans(number_cluster=self.number_cluster)

    def forward(self, x):
        self.embeddings = self.encode(x)

        if datetime.now() - self.start_time < self.clustering_loss_delay:
            self._update_embedding_buffer(self.embeddings.detach())

        x_hat = self.decoder(self.embeddings)
        return x_hat

    def _update_embedding_buffer(self, new_embeddings: torch.Tensor):
        """
        Add new embeddings to the circular buffer.
        """
        self.embedding_buffer.append(new_embeddings)
        total = sum(e.shape[0] for e in self.embedding_buffer)
        while total > self.max_saved_embeddings:
            removed = self.embedding_buffer.pop(0)
            total -= removed.shape[0]

    def initialize_centroids_with_kmeans(self):
        """
        Use the most recent embeddings in the buffer to initialize centroids using KMeans.
        """
        print()
        print('Initialize Centroids')
        if len(self.embedding_buffer) > 0:
            print('Use Kmeans')
            all_embeddings = torch.cat(self.embedding_buffer, dim=0)
            self.centroids = nn.Parameter(self.kmeans(all_embeddings)['centroids'])
        del self.embedding_buffer
        self.initialisation_centroid_done = True

    def loss_function(self, x_hat, x):
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')
        clustering_loss = 0.0

        if datetime.now() - self.start_time >= self.clustering_loss_delay:
            if not self.initialisation_centroid_done:
                self.initialize_centroids_with_kmeans()

            norm_squared = torch.sum((self.embeddings.unsqueeze(1) - self.centroids) ** 2, 2)
            numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
            power = float(self.alpha + 1) / 2
            numerator = numerator ** power
            actual_distribution = numerator / torch.sum(numerator, dim=1, keepdim=True)

            weight = (actual_distribution ** 2) / torch.sum(actual_distribution, 0)
            target_distribution = ((weight.t() / torch.sum(weight, 1)).t()).detach()

            clustering_loss = self.kullback_leibler_divergence(actual_distribution.log(), target_distribution) / actual_distribution.shape[0]

        self.log('clustering_loss', clustering_loss, on_step=True, on_epoch=True)
        self.log('reconstruction_loss', reconstruction_loss, on_step=True, on_epoch=True)
        return self.reconstruction_loss_coefficient * reconstruction_loss + self.clustering_loss_coefficient * clustering_loss

    def step(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        return loss
