from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from architecture.autoencoder import Autoencoder
from utilities.compare_reconstruction import compare_reconstruction
from utilities.get_checkpoint_paths import get_checkpoint_paths
from utilities.mean_adjusted_rand_index import mean_adjusted_rand_index
from utilities.training_configuration import TrainingConfiguration

from utilities.tsne import tsne
from sklearn.cluster import KMeans


if __name__ == '__main__':
    repository_path = Path('/home/malaarabiou/Programming_Projects/Pycharm_Projects/Latent_Cluster_Analysis/experiments/autoencoder_10_minutes')
    checkpoint_paths = get_checkpoint_paths(repository_path)
    number_sample = 10_000
    cluster_number = 8
    clustering_function = KMeans(n_clusters=cluster_number)
    use_tsne = False
    use_clustering = True
    use_adjusted_rand_index = True
    use_compare_reconstruction = True

    dataset_class = MNIST
    dataset = dataset_class(
        root='./dataset',
        train=False,
        download=False,
        transform=ToTensor(),
    )

    # Select random indices
    indices = torch.randperm(len(dataset))[:number_sample]
    subset = Subset(dataset, indices)
    data_loader = DataLoader(subset, batch_size=number_sample, shuffle=False)

    # Load samples
    samples = next(iter(data_loader))[0]  # Take the images from the batch

    cluster_label_list = list()

    for checkpoint_path in checkpoint_paths:
        training_configuration: TrainingConfiguration = TrainingConfiguration().load(checkpoint_path.parents[1] / 'training_configuration.yaml')
        model: Autoencoder = training_configuration.model_class.load_from_checkpoint(checkpoint_path)

        # Encode samples
        with torch.no_grad():
            code = model.encode(samples.to(model.device))
            code = code.cpu().numpy()

        if use_compare_reconstruction:
            compare_reconstruction(
                model=model,
                samples=samples,
                path=checkpoint_path.parents[1],
            )

        if use_clustering or use_adjusted_rand_index:
            cluster_label = clustering_function.fit_predict(code)

        if use_tsne:
            tsne(code, checkpoint_path.parents[1])

        if use_adjusted_rand_index:
            cluster_label_list.append(cluster_label)

    if use_adjusted_rand_index:
        print()
        print('cluster_number: ' + str(cluster_number))
        print('mean_adjusted_rand_index: ' + str(mean_adjusted_rand_index(cluster_label_list)))