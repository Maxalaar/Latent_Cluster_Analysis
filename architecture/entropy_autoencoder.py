import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from architecture.autoencoder import Autoencoder
from architecture.create_dense_architecture import create_dense_architecture


def normalized_entropy(probs):
    """
    Entropie normalisée entre 0 et 1, en base naturelle (nats).
    - `probs` : Tenseur de probabilités (shape quelconque, somme à 1 sur la dernière dimension).
    """
    K = probs.shape[-1]  # Nombre de classes
    H = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # Entropie en nats
    H_max = torch.log(torch.tensor(K, dtype=torch.float))  # H_max = ln(K)
    return H / H_max  # Normalisation entre [0, 1]

def distribution_projection(embeddings):
    return F.gumbel_softmax(embeddings, tau=1000.0, hard=False)

    # return embeddings

    # return F.softmax(embeddings)

    # absolut_embedding = embeddings.abs()
    # normalize_embedding = absolut_embedding / absolut_embedding.sum(dim=1).unsqueeze(dim=1)  #.detach()
    # return normalize_embedding

    # positive_values = embeddings.abs()
    # sum_values = torch.sum(positive_values, dim=-1, keepdim=True)
    # normalized_values = positive_values / sum_values
    # return normalized_values

    # positive_values = F.relu(embeddings)

    # positive_values = embeddings
    # sum_values = torch.sum(positive_values, dim=-1, keepdim=True)
    # normalized_values = positive_values / sum_values
    # return normalized_values

    # positive_values = F.sigmoid(embeddings)
    # sum_values = torch.sum(positive_values, dim=-1, keepdim=True)
    # normalized_values = positive_values / sum_values
    # return normalized_values

    # embeddings = F.sigmoid(embeddings)
    # embeddings = embeddings / (embeddings.sum(dim=1).unsqueeze(dim=1))  #.detach()
    # return embeddings

# positive_values = F.softplus(embeddings)
    # sum_values = torch.sum(positive_values, dim=-1, keepdim=True)
    # normalized_values = positive_values / sum_values
    # return normalized_values


class EntropyAutoencoder(Autoencoder):
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
            configuration_hidden_layers=encoder_configuration,
            output_dimension=latent_space_size,
            activation_function_class=activation_function_class,
            # final_activation_function_class=nn.ReLU,
            # final_activation_function_class=nn.Sigmoid,
            # final_activation_function_class=nn.LeakyReLU,
            # final_activation_function_class=nn.Softmax,
        )

        self.embeddings = None
        self.softmax = nn.Softmax(dim=1)

    def encode(self, x):
        return distribution_projection(self.encoder(x))
        # return self.encoder(x)

    def forward(self, x):
        x = self.encode(x)
        self.embeddings = x
        x = self.decode(x)
        return x

    def loss_function(self, x_hat, x):
        reconstruction_loss = F.mse_loss(x_hat, x)

        # intra_distribution = distribution_projection(self.embeddings)
        intra_distribution = self.embeddings

        intra_entropy = normalized_entropy(intra_distribution)
        intra_loss = intra_entropy.mean()

        extra_distribution = intra_distribution.mean(0)
        # extra_distribution = distribution_projection(intra_distribution.mean(0))

        extra_entropy = normalized_entropy(extra_distribution)
        extra_loss = 1 - extra_entropy

        print()
        print('embeddings : ' + str(self.embeddings[:5]))
        print('intra_distribution : ' + str(intra_distribution[:5]))
        print('extra_distribution : ' + str(extra_distribution))
        print('argmax : ' + str(torch.argmax(intra_distribution, dim=1)[:5]))
        print()
        print('intra_loss : ' + str(intra_loss))
        print('extra_loss : ' + str(extra_loss))
        print('reconstruction_loss : ' + str(reconstruction_loss))

        self.log('intra_loss', intra_loss, on_step=True, on_epoch=True)
        self.log('extra_loss', extra_loss, on_step=True, on_epoch=True)
        self.log('reconstruction_loss', reconstruction_loss, on_step=True, on_epoch=True)

        # loss = 1.0 * reconstruction_loss + 0.02 * intra_loss + 1.0 * extra_loss
        # loss = 1.0 * reconstruction_loss + 0.004 * (1.0 * intra_loss + 0.0 * extra_loss)

        loss = 1.0 * reconstruction_loss + 0.000001 * (1.0 * intra_loss + 1.0 * extra_loss)
        return loss

        # return 1.0 * reconstruction_loss + 0.001 * intra_loss + 1.0 * extra_loss
