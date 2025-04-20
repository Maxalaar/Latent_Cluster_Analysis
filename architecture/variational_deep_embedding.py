import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from architecture.autoencoder import Autoencoder
from architecture.create_dense_architecture import create_dense_architecture

# https://arxiv.org/abs/1611.05148
# https://github.com/klovbe/vade-pytorch
class VariationalDeepEmbedding(Autoencoder):
    def __init__(
        self,
        input_shape,
        output_shape,
        activation_function_class=nn.LeakyReLU,
        encoder_configuration=[64, 32, 16],
        latent_space_size=8,
        decoder_configuration=[16, 32, 64],
        leaning_rate=1e-3,
        number_cluster=10,
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
        input_size = np.prod(input_shape)
        self.encoder = create_dense_architecture(
            flatten_input=True,
            input_dimension=input_size,
            configuration_hidden_layers=encoder_configuration[:-1],
            output_dimension=encoder_configuration[-1],
            activation_function_class=activation_function_class,
        )
        self.number_cluster = number_cluster

        self.mu_layer = nn.Sequential(activation_function_class(), nn.Linear(encoder_configuration[-1], latent_space_size))
        self.log_layer = nn.Sequential(activation_function_class(), nn.Linear(encoder_configuration[-1], latent_space_size))

        self.pi_ = nn.Parameter(torch.FloatTensor(self.number_cluster, ).fill_(1) / self.number_cluster, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(self.number_cluster, latent_space_size).fill_(0), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(self.number_cluster, latent_space_size).fill_(0), requires_grad=True)

        self.z_mu = None
        self.z_sigma2_log = None

    def encode(self, x):
        x = self.encoder(x)
        self.z_mu = self.mu_layer(x)
        self.z_sigma2_log = self.log_layer(x)
        z = self.reparameterize()
        return z

    def reparameterize(self):
        return torch.randn_like(self.z_mu)*torch.exp(self.z_sigma2_log/2)+self.z_mu

    def forward(self, x):
        self.embeddings = self.encode(x)
        x_hat = self.decoder(self.embeddings)
        return x_hat

    def loss_function(self, x_hat, x):
        det = 1e-10

        L_rec = F.binary_cross_entropy(x_hat, x)

        Loss = L_rec * x.size(1)

        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c

        z = torch.randn_like(self.z_mu) * torch.exp(self.z_sigma2_log / 2) + self.z_mu
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)) + det

        yita_c = yita_c / (yita_c.sum(1).view(-1, 1))  # batch_size*Clusters

        Loss += 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2_c.unsqueeze(0) +
                                                              torch.exp(
                                                                  self.z_sigma2_log.unsqueeze(1) - log_sigma2_c.unsqueeze(
                                                                      0)) +
                                                              (self.z_mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(
                                                                  2) / torch.exp(log_sigma2_c.unsqueeze(0)), 2), 1))

        Loss -= torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c)), 1)) + 0.5 * torch.mean(
            torch.sum(1 + self.z_sigma2_log, 1))

        reconstruction_loss = L_rec

        self.log('reconstruction_loss', reconstruction_loss, on_step=True, on_epoch=True)
        return Loss

    def gaussian_pdfs_log(self, x, mus, log_sigma2s):
        G = []
        for c in range(self.number_cluster):
            G.append(self.gaussian_pdf_log(x, mus[c:c + 1, :], log_sigma2s[c:c + 1, :]).view(-1, 1))
        return torch.cat(G, 1)

    def step(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        return loss

    @staticmethod
    def gaussian_pdf_log(x, mu, log_sigma2):
        return -0.5 * (torch.sum(np.log(np.pi * 2) + log_sigma2 + (x - mu).pow(2) / torch.exp(log_sigma2), 1))