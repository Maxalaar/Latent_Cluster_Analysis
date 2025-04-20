import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from architecture.autoencoder import Autoencoder
from architecture.create_dense_architecture import create_dense_architecture


class SoftmaxAutoencoder(Autoencoder):
    def __init__(
        self,
        input_shape,
        output_shape,
        activation_function_class=nn.LeakyReLU,
        encoder_configuration=[64, 32, 16],
        latent_space_size=8,
        decoder_configuration=[16, 32, 64],
        leaning_rate=1e-3,
        reconstruction_loss_coefficient: float = 1.0,
        softmax_loss_loss_coefficient: float = 1.0,
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

        input_size = np.prod(input_shape)
        self.encoder = create_dense_architecture(
            flatten_input=True,
            input_dimension=input_size,
            configuration_hidden_layers=encoder_configuration,
            output_dimension=latent_space_size,
            activation_function_class=activation_function_class,
            # final_activation_function_class=nn.Sigmoid,
        )

        self.embeddings = None
        self.reconstruction_loss_coefficient: float = reconstruction_loss_coefficient
        self.softmax_loss_loss_coefficient: float = softmax_loss_loss_coefficient

    def encode(self, x):
        embeddings = self.encoder(x)
        # embeddings = embeddings / embeddings.sum(dim=1, keepdim=True)
        return embeddings
        # return F.softmax(self.encoder(x), dim=1)

    def forward(self, x):
        self.embeddings = self.encode(x)
        x_hat = self.decoder(self.embeddings)
        return x_hat

    def loss_function(self, x_hat, x):
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')

        actual_distribution_temperature = 1.0
        target_distribution_temperature = 1.0

        actual_distribution = F.softmax(self.embeddings / actual_distribution_temperature, dim=1)


        # target_distribution = F.softmax(self.embeddings / target_distribution_temperature, dim=1).detach()
        target_distribution = F.softmax(self.embeddings / target_distribution_temperature, dim=1).detach()

        kullback_leibler_divergence_loss = F.kl_div(actual_distribution.log(), target_distribution, reduction='batchmean')


        # print()
        print(actual_distribution.mean(0))
        # print(self.embeddings.mean(1))
        # print(self.embeddings[:5, :])

        self.log('reconstruction_loss', reconstruction_loss, on_step=True, on_epoch=True)
        self.log('kullback_leibler_divergence_loss', kullback_leibler_divergence_loss, on_step=True, on_epoch=True)

        return 1.0 * reconstruction_loss + 0.001 * kullback_leibler_divergence_loss

    def step(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        return loss