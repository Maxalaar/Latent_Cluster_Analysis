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

no_pretrain_deep_embedding_clustering_autoencoder_10_minutes = TrainingConfiguration(
    experiment_name='no_pretrain_deep_embedding_clustering_autoencoder_10_minutes',
    dataset_class=MNIST,  # MNIST, CIFAR10
    model_class=DeepEmbeddingClustering,
    model_configuration={
        'number_cluster': 10,
    },
    maximum_training_time=timedelta(minutes=10),
    checkpoint_interval_time=timedelta(minutes=2),
    number_model_to_train=6,
)

deep_embedding_clustering_autoencoder_20_minutes = TrainingConfiguration(
    experiment_name='pretrain_deep_embedding_clustering_autoencoder_20_minutes',
    dataset_class=MNIST,  # MNIST, CIFAR10
    model_class=DeepEmbeddingClustering,
    model_configuration={
        'number_cluster': 10,
        'clustering_loss_delay': timedelta(minutes=10),
    },
    maximum_training_time=timedelta(minutes=20),
    checkpoint_interval_time=timedelta(minutes=2),
    number_model_to_train=2,
    batch_size=2048,
)

deep_embedding_clustering_autoencoder_120_minutes = TrainingConfiguration(
    experiment_name='deep_embedding_clustering_autoencoder_120_minutes',
    dataset_class=MNIST,
    model_class=DeepEmbeddingClustering,
    model_configuration={
        'number_cluster': 10,
        'reconstruction_loss_coefficient': 1.0,
        'clustering_loss_coefficient': 1.0,
        'encoder_configuration': [64, 32, 16],
        'latent_space_size': 8,
        'decoder_configuration': [16, 32, 64],
        'leaning_rate': 1e-4,
        'clustering_loss_delay': timedelta(minutes=20),
    },
    batch_size=2048,
    maximum_training_time=timedelta(minutes=120),
    checkpoint_interval_time=timedelta(minutes=2),
    number_model_to_train=5,
)


# Non Classical
debug = TrainingConfiguration(
    experiment_name='debug',
    dataset_class=MNIST,
    model_class=DeepEmbeddingClustering,
    model_configuration={
        'number_cluster': 10,
        'reconstruction_loss_coefficient': 0.5,
        'clustering_loss_coefficient': 1.0,
        'encoder_configuration': [64, 32],
        'latent_space_size': 16,
        'decoder_configuration': [32, 64],
        'clustering_loss_delay': timedelta(minutes=5),
        # 'leaning_rate': 1e-4,
    },
    batch_size=512,
    maximum_training_time=timedelta(minutes=20),
    checkpoint_interval_time=timedelta(minutes=2),
    number_model_to_train=1,
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
        'latent_space_size': 2,
        'activation_function_class': nn.ReLU,
    },
    maximum_training_time=timedelta(minutes=2),
    checkpoint_interval_time=timedelta(minutes=2),
    number_model_to_train=1,
)