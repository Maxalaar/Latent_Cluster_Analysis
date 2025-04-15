import warnings

import torch
import torch.nn as nn

try:
    from cuml import KMeans
    from cuml.metrics.cluster import silhouette_score
except ImportError:
    print('cuml is not installed')


class Kmeans(nn.Module):
    def __init__(
            self,
            number_cluster: int,
            memory_size: int = 0,
            logger=None,
    ):
        super(Kmeans, self).__init__()
        self.number_cluster = number_cluster
        self.max_memory_size = memory_size
        self.logger = logger
        self.kmeans = KMeans(n_clusters=self.number_cluster)
        self.silhouette_score = silhouette_score
        self.memory = None  # Memory for previous embeddings

    def forward(self, embeddings):
        device = embeddings.device

        # Add new embeddings to memory
        if self.memory is None:
            self.memory = embeddings.detach()
        else:
            self.memory = torch.cat((self.memory, embeddings.detach()), dim=0)
            max_data_size = self.max_memory_size + embeddings.size(0)

            # Limit memory size to the maximum allowed
            if self.memory.size(0) > max_data_size:
                self.memory = self.memory[-max_data_size:]

        # Perform clustering on the combined memory and new embeddings
        all_embeddings = self.memory
        cluster_labels_all = torch.tensor(
            self.kmeans.fit_predict(all_embeddings), device=device
        )
        centroids = torch.tensor(self.kmeans.cluster_centers_, device=device)

        # Identify cluster labels only for the new embeddings
        new_labels = cluster_labels_all[-embeddings.size(0):]

        return {'cluster_labels': new_labels, 'centroids': centroids}


