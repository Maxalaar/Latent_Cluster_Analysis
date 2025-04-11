import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.autoencoder import Autoencoder


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

        self.mu_layer = nn.Sequential(activation_function_class(), nn.Linear(encoder_configuration[-1], lantent_space_size))
        self.logvar_layer = nn.Sequential(activation_function_class(), nn.Linear(encoder_configuration[-1], lantent_space_size))

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        epsilon = torch.randn_like(logvar)
        return mu + torch.sqrt(logvar.exp()) * epsilon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def loss_function(self, x_hat, x, mu, logvar):
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def step(self, batch):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat, mu, logvar = self.forward(x)
        loss = self.loss_function(x_hat, x, mu, logvar)
        return loss
