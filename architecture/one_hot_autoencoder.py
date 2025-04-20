import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from architecture.autoencoder import Autoencoder
from architecture.create_dense_architecture import create_dense_architecture


class OneHotAutoencoder(Autoencoder):
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
            final_activation_function_class=nn.Sigmoid,
        )

        self.embeddings = None
        self.reconstruction_loss_coefficient: float = reconstruction_loss_coefficient
        self.softmax_loss_loss_coefficient: float = softmax_loss_loss_coefficient

    def encode(self, x):
        embeddings = self.encoder(x)
        embeddings = embeddings / embeddings.sum(dim=1, keepdim=True)
        return embeddings
        # return F.softmax(self.encoder(x), dim=1)

    def forward(self, x):
        self.embeddings = self.encode(x)
        x_hat = self.decoder(self.embeddings)
        return x_hat

    def loss_function(self, x_hat, x):
        reconstruction_loss = 0.0
        one_hot_loss = 0.0
        balance_loss = 0.0


        reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')

        current_balance = self.embeddings.mean(dim=0)
        # max_indices = torch.argmax(self.embeddings - current_balance.detach(), dim=1)

        # one_hot_target = F.one_hot(max_indices, num_classes=self.latent_space_size).float()
        target_balance = torch.ones((self.latent_space_size,), dtype=torch.float32).to(self.device) / self.latent_space_size

        # one_hot_loss = F.mse_loss(self.embeddings, one_hot_target)
        balance_loss = F.mse_loss(current_balance, target_balance)

        print()
        # print('mean : ' + str(one_hot_target.mean(dim=0)))
        print(current_balance)
        print(self.embeddings[:5,:])

        print('reconstruction_loss: ' + str(reconstruction_loss))
        print('balance_loss: ' + str(balance_loss))
        # print('one_hot_loss: ' + str(one_hot_loss))


        self.log('reconstruction_loss', reconstruction_loss, on_step=True, on_epoch=True)
        self.log('balance_loss', balance_loss, on_step=True, on_epoch=True)
        # self.log('one_hot_loss', one_hot_loss, on_step=True, on_epoch=True)

        # kullback_leibler_divergence_loss = F.kl_div(self.embeddings, target_distribution, reduction='batchmean')    # .log()
        # self.log('softmax_loss', one_hot_loss, on_step=True, on_epoch=True)
        # self.log('reconstruction_loss', reconstruction_loss, on_step=True, on_epoch=True)

        return 1.0 * reconstruction_loss + 0.0 * one_hot_loss + 1.0 * balance_loss

    def step(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        return loss