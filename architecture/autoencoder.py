import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np


from architecture.create_dense_architecture import create_dense_architecture


class Autoencoder(pl.LightningModule):
    def __init__(
            self,
            input_shape,
            output_shape,
            activation_function_class=nn.LeakyReLU,
            encoder_configuration=[64, 32, 16],
            latent_space_size=8,
            decoder_configuration=[16, 32, 64],
            leaning_rate=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        input_size = np.prod(input_shape)
        output_size = np.prod(output_shape)
        self.leaning_rate = leaning_rate

        self.encoder = create_dense_architecture(
            flatten_input=True,
            input_dimension=input_size,
            configuration_hidden_layers=encoder_configuration,
            output_dimension=latent_space_size,
            activation_function_class=activation_function_class,
        )
        self.decoder = create_dense_architecture(
            input_dimension=latent_space_size,
            configuration_hidden_layers=decoder_configuration,
            output_dimension=output_size,
            activation_function_class=activation_function_class,
            output_shape=output_shape,
            final_activation_function_class=nn.Sigmoid,
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def loss_function(self, x_hat, x):
        reconstruction_loss = F.mse_loss(x_hat, x)
        self.log('reconstruction_loss', reconstruction_loss, on_step=True, on_epoch=True)
        return reconstruction_loss

    def step(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("validation_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.leaning_rate)
