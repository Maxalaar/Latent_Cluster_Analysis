from datetime import timedelta

import torch.nn as nn
from torchvision.datasets import MNIST

from architecture.autoencoder import Autoencoder
from architecture.variational_autoencoder import VariationalAutoencoder
from utilities.training_configuration import TrainingConfiguration

autoencoder_10_minutes = TrainingConfiguration(
    experiment_name='autoencoder_10_minutes',
    dataset_class=MNIST,  # MNIST, CIFAR10
    model_class=Autoencoder,
    maximum_training_time=timedelta(minutes=10),
    checkpoint_interval_time=timedelta(minutes=2),
    number_model_to_train=6,
)

variational_autoencoder_10_minutes = TrainingConfiguration(
    experiment_name='variational_autoencoder_10_minutes',
    dataset_class=MNIST,  # MNIST, CIFAR10
    model_class=VariationalAutoencoder,
    maximum_training_time=timedelta(minutes=10),
    checkpoint_interval_time=timedelta(minutes=2),
    number_model_to_train=6,
)

debug = TrainingConfiguration(
    experiment_name='debug',
    dataset_class=MNIST,
    model_class=VariationalAutoencoder,
    maximum_training_time=timedelta(minutes=0.5),
    checkpoint_interval_time=timedelta(minutes=0.25),
    number_model_to_train=6,
)

maxalaar = TrainingConfiguration(
    experiment_name='maxalaar',
    dataset_class=MNIST,  # MNIST, CIFAR10
    model_class=VariationalAutoencoder,
    model_configuration={
        'kullback_leibler_divergence_loss_coefficient': 0.01,
    },
    maximum_training_time=timedelta(minutes=2),
    checkpoint_interval_time=timedelta(minutes=2),
    number_model_to_train=6,
)

samy_2d = TrainingConfiguration(
    experiment_name='samy_2d',
    dataset_class=MNIST,  # MNIST, CIFAR10
    model_class=VariationalAutoencoder,
    model_configuration={
        'kullback_leibler_divergence_loss_coefficient': 0.01,
        'lantent_space_size': 2,
        'activation_function_class': nn.ReLU,
    },
    maximum_training_time=timedelta(minutes=2),
    checkpoint_interval_time=timedelta(minutes=2),
    number_model_to_train=1,
)