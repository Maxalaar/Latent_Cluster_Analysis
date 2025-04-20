from datetime import timedelta

import torch.nn as nn
from torchvision.datasets import MNIST

from architecture.autoencoder import Autoencoder
from architecture.clustering_autoencoder import ClusteringAutoencoder
from architecture.deep_embedding_clustering import DeepEmbeddingClustering
from architecture.entropy_autoencoder import EntropyAutoencoder
from architecture.one_hot_autoencoder import OneHotAutoencoder
from architecture.softmax_autoencoder import SoftmaxAutoencoder
from architecture.sparse_autoencoder import SparseAutoencoder
from architecture.top_k import TopKAutoencoder
from architecture.variational_autoencoder import VariationalAutoencoder
from architecture.variational_deep_embedding import VariationalDeepEmbedding
from utilities.training_configuration import TrainingConfiguration

autoencoder_10_minutes = TrainingConfiguration(
    experiment_name='autoencoder_10_minutes',
    dataset_class=MNIST,  # MNIST, CIFAR10
    model_class=Autoencoder,
    maximum_training_epochs=300,
    # maximum_training_time=timedelta(minutes=10),
    checkpoint_interval_time=timedelta(minutes=2),
    number_model_to_train=2,
)

variational_autoencoder_10_minutes = TrainingConfiguration(
    experiment_name='variational_autoencoder_10_minutes',
    dataset_class=MNIST,  # MNIST, CIFAR10
    model_class=VariationalAutoencoder,
    maximum_training_time=timedelta(minutes=10),
    checkpoint_interval_time=timedelta(minutes=2),
    number_model_to_train=6,
)

# clustering_autoencoder_10_minutes = TrainingConfiguration(
#     experiment_name='clustering_autoencoder_10_minutes',
#     dataset_class=MNIST,  # MNIST, CIFAR10
#     model_class=ClusteringAutoencoder,
#     model_configuration={
#         'number_cluster': 10,
#         'margin_between_clusters': 3,
#         'number_centroids_repulsion': 3,
#         'memory_size': 10_000,
#         'clustering_loss_coefficient': 0.1,
#     },
#     maximum_training_time=timedelta(minutes=10),
#     checkpoint_interval_time=timedelta(minutes=2),
#     number_model_to_train=6,
# )

clustering_autoencoder_10_minutes = TrainingConfiguration(
    experiment_name='clustering_autoencoder_10_minutes',
    dataset_class=MNIST,  # MNIST, CIFAR10
    model_class=ClusteringAutoencoder,
    model_configuration={
        'number_cluster': 10,
        'margin_between_clusters': 3,
        'number_centroids_repulsion': 3,
        'memory_size': 10_000,
        'clustering_loss_coefficient': 0.01,
        'clustering_loss_epoch': 300,
    },
    maximum_training_epochs=800,
    # maximum_training_time=timedelta(minutes=10),
    checkpoint_interval_time=timedelta(minutes=2),
    number_model_to_train=1,
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
    experiment_name='deep_embedding_clustering_autoencoder_20_minutes_v2',
    dataset_class=MNIST,
    model_class=DeepEmbeddingClustering,
    model_configuration={
        'number_cluster': 10,
        'reconstruction_loss_coefficient': 1.0,
        'clustering_loss_coefficient': 1.0,
        'encoder_configuration': [500, 500, 2000],
        'latent_space_size': 10,
        'decoder_configuration': [2000, 500, 500],
        'clustering_loss_start_epoch': 300,
        # 'leaning_rate': 1e-4,
    },
    batch_size=1024,
    maximum_training_epochs=300,
    checkpoint_interval_time=timedelta(minutes=2),
    number_model_to_train=3,
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

top_k_autoencoder = TrainingConfiguration(
    experiment_name='top_2_autoencoder',
    dataset_class=MNIST,
    model_class=TopKAutoencoder,
    model_configuration={
        'encoder_configuration': [64, 32, 16],
        'latent_space_size': 10,
        'decoder_configuration': [16, 32, 64],
        'number_activation': 2,
        # 'leaning_rate': 5e-4,
    },
    maximum_training_epochs=-1,
    batch_size=2048,
    checkpoint_interval_time=timedelta(minutes=2),
    number_model_to_train=1,
)

sparse_autoencoder = TrainingConfiguration(
    experiment_name='sparse_autoencoder_v5',
    dataset_class=MNIST,
    model_class=SparseAutoencoder,
    model_configuration={
        'encoder_configuration': [64, 32],
        'latent_space_size': 32,
        'decoder_configuration': [32, 64],
        'rho': 0.005,
    },
    batch_size=128,
    maximum_training_epochs=100,
    checkpoint_interval_time=timedelta(minutes=1),
    number_model_to_train=2,
)


# Non Classical
# debug = TrainingConfiguration(
#     experiment_name='one_hot_autoencoder',
#     dataset_class=MNIST,
#     model_class=OneHotAutoencoder,
#     model_configuration={
#         'encoder_configuration': [64, 32],
#         'latent_space_size': 10,
#         'decoder_configuration': [32, 64],
#     },
#     batch_size=2048 * 2,
#     maximum_training_epochs=-1,
#     checkpoint_interval_time=timedelta(minutes=1),
#     number_model_to_train=2,
# )

# debug = TrainingConfiguration(
#     experiment_name='softmax_autoencoder',
#     dataset_class=MNIST,
#     model_class=SoftmaxAutoencoder,
#     model_configuration={
#         'encoder_configuration': [64, 32],
#         'latent_space_size': 10,
#         'decoder_configuration': [32, 64],
#     },
#     batch_size=128,
#     maximum_training_epochs=-1,
#     checkpoint_interval_time=timedelta(minutes=1),
#     number_model_to_train=2,
# )

# debug = TrainingConfiguration(
#     experiment_name='variational_deep_embedding',
#     dataset_class=MNIST,
#     model_class=VariationalDeepEmbedding,
#     model_configuration={
#         'encoder_configuration': [64, 32],
#         'latent_space_size': 10,
#         'decoder_configuration': [32, 64],
#     },
#     batch_size=128,
#     maximum_training_epochs=-1,
#     checkpoint_interval_time=timedelta(minutes=1),
#     number_model_to_train=2,
# )

debug = TrainingConfiguration(
    experiment_name='entropy_autoencoder',
    dataset_class=MNIST,
    model_class=EntropyAutoencoder,
    model_configuration={
        'encoder_configuration': [64, 32],
        'latent_space_size': 10,
        'decoder_configuration': [32, 64],
        'leaning_rate': 1e-4,
    },
    batch_size=64,
    maximum_training_epochs=-1,
    checkpoint_interval_time=timedelta(minutes=0.1),
    number_model_to_train=2,
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
        'softmax_loss_loss_coefficient': 0.01,
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
        'softmax_loss_loss_coefficient': 0.01,
        'latent_space_size': 2,
        'activation_function_class': nn.ReLU,
    },
    maximum_training_time=timedelta(minutes=2),
    checkpoint_interval_time=timedelta(minutes=2),
    number_model_to_train=1,
)