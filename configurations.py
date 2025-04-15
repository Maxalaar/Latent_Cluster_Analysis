from datetime import timedelta

import torch.nn as nn
from torchvision.datasets import MNIST

from architecture.autoencoder import Autoencoder
from architecture.clustering_autoencoder import ClusteringAutoencoder
from architecture.deep_embedding_clustering import DeepEmbeddingClustering
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

clustering_autoencoder_10_minutes = TrainingConfiguration(
    experiment_name='new_clustering_autoencoder_10_minutes',
    dataset_class=MNIST,  # MNIST, CIFAR10
    model_class=ClusteringAutoencoder,
    model_configuration={
        'number_cluster': 10,
        'margin_between_clusters': 3,
        'number_centroids_repulsion': 3,
        'memory_size': 10_000,
        'clustering_loss_coefficient': 0.1,
    },
    maximum_training_time=timedelta(minutes=10),
    checkpoint_interval_time=timedelta(minutes=2),
    number_model_to_train=6,
)

brendan = TrainingConfiguration(
    experiment_name='brendan',
    dataset_class=MNIST,  # MNIST, CIFAR10
    model_class=DeepEmbeddingClustering,
    model_configuration={
        'number_cluster': 10,
        'clustering_loss_coefficient': 100,
    },
    maximum_training_time=timedelta(minutes=5),
    checkpoint_interval_time=timedelta(minutes=0.5),
    number_model_to_train=6,
)

clustering_autoencoder_60_minutes = TrainingConfiguration(
    experiment_name='clustering_autoencoder_60_minutes',
    dataset_class=MNIST,  # MNIST, CIFAR10
    model_class=ClusteringAutoencoder,
    model_configuration={
        'number_cluster': 10,
        'margin_between_clusters': 3,
        'number_centroids_repulsion': 3,
        'memory_size': 10_000,
        'clustering_loss_coefficient': 0.1,
    },
    maximum_training_time=timedelta(minutes=60),
    checkpoint_interval_time=timedelta(minutes=2),
    number_model_to_train=6,
)

debug = TrainingConfiguration(
    experiment_name='debug',
    dataset_class=MNIST,
    model_class=DeepEmbeddingClustering,
    model_configuration={
        'number_cluster': 10,
        'reconstruction_loss_coefficient': 0.0,
        'clustering_loss_coefficient': 1.0,
        'lantent_space_size': 2,
    },
    maximum_training_time=timedelta(minutes=5),
    checkpoint_interval_time=timedelta(minutes=1),
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