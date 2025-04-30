import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from architecture.autoencoder import Autoencoder


class IterativeClusteringAutoencoder(Autoencoder):
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
            learning_rate=learning_rate,
        )
        self.embeddings = None
        self.number_clusters = number_clusters
        self.initialization_centroids = False
        self.centroids = torch.nn.Parameter(torch.randn(self.number_clusters, self.latent_space_size), requires_grad=False)

        self.reconstruction_loss_coefficient = reconstruction_loss_coefficient
        self.clustering_loss_coefficient = clustering_loss_coefficient

    def forward(self, x):
        self.embeddings = self.encode(x)
        x_hat = self.decoder(self.embeddings)
        return x_hat

    def loss_function(self, x_hat, x):
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')

        if not self.initialization_centroids:
            random_index = torch.randperm(self.embeddings.size(0))[:self.number_clusters]
            self.centroids.copy_(self.embeddings[random_index].detach())
            self.initialization_centroids = True

        # Calculate distances and assignments
        distances = torch.cdist(self.embeddings, self.centroids)
        min_indices = torch.argmin(distances, dim=1)

        # Correct indexing for distance retrieval
        reference_distances = distances[torch.arange(len(self.embeddings)), min_indices]
        clustering_loss = torch.mean(torch.pow(reference_distances, 2))

        # Update centroids safely
        new_centroids = []
        for cluster_idx in range(self.number_clusters):
            mask = (min_indices == cluster_idx)
            cluster_samples = self.embeddings[mask]

            if len(cluster_samples) == 0:  # Handle empty clusters
                new_centroid = self.centroids[cluster_idx].detach()
            else:
                new_centroid = torch.mean(cluster_samples, dim=0).detach()

            new_centroids.append(new_centroid)

        # self.centroids = torch.stack(new_centroids)
        self.centroids.data = torch.stack(new_centroids).to(self.centroids.device)

        # Logging metrics
        self.log('reconstruction_loss', reconstruction_loss, on_step=True, on_epoch=True)
        self.log('clustering_loss', clustering_loss, on_step=True, on_epoch=True)

        return self.reconstruction_loss_coefficient * reconstruction_loss + self.clustering_loss_coefficient * clustering_loss

    def step(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        return loss

    # def clustering(self, number_clusters, embedding):
    #     if self.number_clusters == number_clusters:
    #         return KMeans(
    #             n_clusters=self.number_clusters,
    #             init=self.centroids.detach().cpu().numpy(),
    #             n_init=1,
    #             max_iter=1,
    #         ).fit_predict(embedding)