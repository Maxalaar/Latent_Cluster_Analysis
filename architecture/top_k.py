import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.autoencoder import Autoencoder
from architecture.loss.distance_centroid_loss import DistanceCentroidLoss
from architecture.loss.kmeans import Kmeans

# https://arxiv.org/pdf/2406.04093
class TopKAutoencoder(Autoencoder):
    def __init__(
        self,
        input_shape,
        output_shape,
        activation_function_class=nn.LeakyReLU,
        encoder_configuration=[64, 32, 16],
        latent_space_size=8,
        decoder_configuration=[16, 32, 64],
        leaning_rate=1e-3,
        number_activation=1,
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
        self.number_activation = number_activation

    def encode(self, x):
        embeddings = self.encoder(x)
        if self.number_activation < embeddings.size(1):  # Check if k is less than embedding dimension
            # Get the top k values and their indices for each sample
            values, indices = torch.topk(embeddings, k=self.number_activation, dim=1)
            # Create a binary mask where top k elements are 1
            mask = torch.zeros_like(embeddings)
            mask.scatter_(1, indices, 1.0)
            # Apply the mask to zero out non-top k elements
            masked_embeddings = embeddings * mask
        else:
            # If k is equal or larger, keep all elements
            masked_embeddings = embeddings
        return masked_embeddings

    def loss_function(self, x_hat, x):
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')
        self.log('reconstruction_loss', reconstruction_loss, on_step=True, on_epoch=True)
        return reconstruction_loss

    def step(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        return loss
