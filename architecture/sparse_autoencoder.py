import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.autoencoder import Autoencoder
from architecture.create_dense_architecture import create_dense_architecture

# https://github.com/AntonP999/Sparse_autoencoder
class SparseAutoencoder(Autoencoder):
    def __init__(
            self,
            input_shape,
            output_shape,
            activation_function_class=nn.LeakyReLU,
            encoder_configuration=[64, 32, 16],
            latent_space_size=8,
            decoder_configuration=[16, 32, 64],
            learning_rate=1e-3,
            reconstruction_loss_coefficient=1.0,
            sparsity_loss_coefficient=1.0,
            rho: float = 0.05,
    ):
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            activation_function_class=activation_function_class,
            encoder_configuration=encoder_configuration,
            latent_space_size=latent_space_size,
            decoder_configuration=decoder_configuration,
            leaning_rate=learning_rate,
        )

        input_size = np.prod(input_shape)
        output_size = np.prod(output_shape)

        self.encoder = create_dense_architecture(
            flatten_input=True,
            input_dimension=input_size,
            configuration_hidden_layers=encoder_configuration,
            output_dimension=latent_space_size,
            activation_function_class=activation_function_class,
            final_activation_function_class=nn.Sigmoid,
        )
        self.decoder = create_dense_architecture(
            input_dimension=latent_space_size,
            configuration_hidden_layers=decoder_configuration,
            output_dimension=output_size,
            activation_function_class=activation_function_class,
            output_shape=output_shape,
            final_activation_function_class=nn.Tanh,
        )

        self.size_average=True
        self.embeddings = None
        self.data_rho = None

        self.reconstruction_loss_coefficient = reconstruction_loss_coefficient
        self.sparsity_loss_coefficient = sparsity_loss_coefficient
        self.rho = rho

    def forward(self, x):
        self.embeddings = self.encode(x)
        self.data_rho = self.embeddings.mean(0)
        x_hat = self.decoder(self.embeddings)
        return x_hat

    def loss_function(self, x_hat, x):
        # Loss de reconstruction
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')

        dkl = - self.rho * torch.log(self.data_rho) - (1 - self.rho) * torch.log(1 - self.data_rho)  # calculates KL divergence
        if self.size_average:
            rho_loss = dkl.mean()
        else:
            rho_loss = dkl.sum()
        sparsity_loss = rho_loss

        # Logging
        self.log('reconstruction_loss', reconstruction_loss, on_step=True, on_epoch=True)
        self.log('mean_non_zero', (self.embeddings != 0).sum(dim=1).float().mean(), on_step=True, on_epoch=True)
        self.log('sparsity_loss', sparsity_loss, on_step=True, on_epoch=True)

        return self.reconstruction_loss_coefficient * reconstruction_loss + self.sparsity_loss_coefficient * sparsity_loss