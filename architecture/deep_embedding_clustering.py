import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta

from architecture.autoencoder import Autoencoder
from architecture.loss.kmeans_module import KMeansModule
from sklearn.mixture import GaussianMixture

from sklearn.cluster import KMeans, SpectralClustering


# https://arxiv.org/abs/1511.06335
class DeepEmbeddingClustering(Autoencoder):
    def __init__(
        self,
        input_shape,
        output_shape,
        activation_function_class=nn.LeakyReLU,
        encoder_configuration=[64, 32, 16],
        latent_space_size=8,
        decoder_configuration=[16, 32, 64],
        learning_rate=1e-3,
        number_clusters: int = 2,
        alpha: float = 1.0,
        reconstruction_loss_coefficient: float = 1.0,
        clustering_loss_coefficient: float = 1.0,
        clustering_loss_start_epoch: int = 0,
        max_saved_embeddings: int = 10_000,
    ):
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            activation_function_class=activation_function_class,
            encoder_configuration=encoder_configuration,
            latent_space_size=latent_space_size,
            decoder_configuration=decoder_configuration,
            learning_rate=learning_rate,
        )
        self.latent_space_size = latent_space_size
        self.embeddings = None
        self.alpha = alpha
        self.reconstruction_loss_coefficient = reconstruction_loss_coefficient
        self.clustering_loss_coefficient = clustering_loss_coefficient
        self.clustering_loss_start_epoch = clustering_loss_start_epoch

        self.kullback_leibler_divergence = nn.KLDivLoss(size_average=False)

        self.number_clusters = number_clusters
        self.centroids = nn.Parameter(torch.randn(self.number_clusters, self.latent_space_size, device='cuda'))
        self.embedding_buffer = []
        self.max_saved_embeddings = max_saved_embeddings
        self.initialisation_centroid_done = False
        self.kmeans = KMeansModule(number_cluster=self.number_clusters)

    def forward(self, x):
        self.embeddings = self.encode(x)

        if self.current_epoch < self.clustering_loss_start_epoch:
            self._update_embedding_buffer(self.embeddings.detach())

        x_hat = self.decoder(self.embeddings)
        return x_hat

    def _update_embedding_buffer(self, new_embeddings: torch.Tensor):
        self.embedding_buffer.append(new_embeddings)
        total = sum(e.shape[0] for e in self.embedding_buffer)
        while total > self.max_saved_embeddings:
            removed = self.embedding_buffer.pop(0)
            total -= removed.shape[0]

    def initialize_centroids_with_clustering(self):
        print("\nInitialize Centroids")
        if len(self.embedding_buffer) > 0:
            print("Use Clustering")
            all_embeddings = torch.cat(self.embedding_buffer, dim=0)
            clustering_function = KMeans(
                n_clusters=self.number_clusters,
                init='k-means++',
                n_init=10,
                max_iter=500,
                tol=1e-5,
                algorithm='elkan',
                random_state=42
            )
            # clustering_function = GaussianMixture(
            #     n_components=10,
            #     random_state=42,
            #     n_init=30,
            # )
            # clustering_function.cluster_centers_ = clustering_function.means_
            # clustering_function.cluster_centers_ = clustering_function.means_

            clustering_function.fit(all_embeddings.cpu().detach().numpy())
            centroids_tensor = torch.from_numpy(clustering_function.cluster_centers_).to(self.device)
            with torch.no_grad():
                self.centroids.copy_(centroids_tensor)
        del self.embedding_buffer

        # new_optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # self.trainer.optimizers = [new_optimizer]

        self.initialisation_centroid_done = True

    def get_centroids_with_clustering(self):
        all_embeddings = torch.cat(self.embedding_buffer, dim=0)
        return self.kmeans(all_embeddings)['centroids']

    def loss_function(self, x_hat, x):
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')
        clustering_loss = 0.0

        if self.current_epoch >= self.clustering_loss_start_epoch:
            if not self.initialisation_centroid_done:
                self.initialize_centroids_with_clustering()

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

        return (
            self.reconstruction_loss_coefficient * reconstruction_loss
            + self.clustering_loss_coefficient * clustering_loss
        )

    def clustering(self, number_cluster, embedding):
        if self.number_clusters == number_cluster:
            return KMeans(
                n_clusters=self.number_clusters,
                init=self.centroids.detach().cpu().numpy(),
                n_init=1,
                max_iter=1,
            ).fit_predict(embedding)

