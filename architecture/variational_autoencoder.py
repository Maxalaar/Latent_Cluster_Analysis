import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.autoencoder import Autoencoder
from architecture.create_dense_architecture import create_dense_architecture


class VariationalAutoencoder(Autoencoder):
    def __init__(
        self,
        input_shape,
        output_shape,
        activation_function_class=nn.LeakyReLU,
        encoder_configuration=[64, 32, 16],
        lantent_space_size=8,
        decoder_configuration=[16, 32, 64],
        leaning_rate=1e-3,
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

        input_size = np.prod(input_shape)
        self.encoder = create_dense_architecture(
            flatten_input=True,
            input_dimension=input_size,
            configuration_hidden_layers=encoder_configuration[:-1],
            output_dimension=encoder_configuration[-1],
            activation_function_class=activation_function_class,
        )

        self.mu_layer = nn.Sequential(activation_function_class(), nn.Linear(encoder_configuration[-1], lantent_space_size))
        self.logvar_layer = nn.Sequential(activation_function_class(), nn.Linear(encoder_configuration[-1], lantent_space_size))
        self.mu = None
        self.logvar = None

    def encode(self, x):
        x = self.encoder(x)
        self.mu = self.mu_layer(x)
        self.logvar = self.logvar_layer(x)
        z = self.reparameterize()
        return z

    def reparameterize(self):
        epsilon = torch.randn_like(self.logvar)
        return self.mu + torch.sqrt(self.logvar.exp()) * epsilon

    def forward(self, x):
        x = self.encode(x)
        # z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(x)
        return x_hat

    def loss_function(self, x_hat, x):
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        return recon_loss + kl_div

    def step(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        return loss
