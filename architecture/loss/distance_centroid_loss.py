from typing import Optional

import torch
import torch.nn as nn


class DistanceCentroidLoss(nn.Module):
    def __init__(
            self,
            margin_between_clusters: float,
            number_centroids_repulsion: Optional[int] = None,
            logger=None,
    ):
        super(DistanceCentroidLoss, self).__init__()
        self.margin_between_clusters: float = margin_between_clusters
        self.number_centroids_repulsion: Optional[int] = number_centroids_repulsion
        self.logger = logger

    def forward(self, embeddings: torch.Tensor, cluster_labels: torch.Tensor, centroids: torch.Tensor):
        device = embeddings.device
        unique_cluster_labels = torch.unique(cluster_labels)
        number_cluster_label_in_batch = len(unique_cluster_labels)
        attraction_loss = torch.tensor(0.0, device=device)
        repulsion_loss = torch.tensor(0.0, device=device)
        distance_intra_cluster = 0.0

        # Compute attraction and repulsion loss
        for i in list(unique_cluster_labels):
            current_centroid = centroids[i].unsqueeze(dim=0)
            ixd_points_current_cluster = torch.nonzero(cluster_labels == i).squeeze(dim=1)
            if ixd_points_current_cluster.size(0) > 0:
                points_current_cluster = embeddings[ixd_points_current_cluster]
                other_centroids = torch.cat([centroids[:i], centroids[i + 1:]])

                # Attraction Term
                attraction_distances = torch.cdist(points_current_cluster, current_centroid)
                attraction_loss += torch.mean(attraction_distances ** 2)

                # Repulsion Term
                if number_cluster_label_in_batch > 1:
                    repulsion_distances = torch.cdist(points_current_cluster, other_centroids)
                    if self.number_centroids_repulsion is not None:
                        number_centroids_repulsion = min(number_cluster_label_in_batch - 1, self.number_centroids_repulsion)
                        closest_distances, _ = torch.topk(repulsion_distances, k=number_centroids_repulsion, dim=1, largest=False)
                    else:
                        closest_distances, _ = torch.topk(repulsion_distances, k=number_cluster_label_in_batch-1, dim=1, largest=False)
                    repulsion_loss += torch.mean(torch.nn.functional.relu((self.margin_between_clusters - closest_distances) ** 2))

                    # Intra-cluster distance
                    distance_intra_cluster += torch.mean(torch.norm(points_current_cluster - current_centroid, dim=1)).item()

        total_loss = (attraction_loss + repulsion_loss) / number_cluster_label_in_batch

        # Logging
        if self.logger:
            matrix_distance_centroids = torch.cdist(centroids, centroids).triu(diagonal=1)
            distance_centroids = matrix_distance_centroids[matrix_distance_centroids != 0]
            self.logger('number_cluster_label_in_batch', number_cluster_label_in_batch, on_epoch=True)
            self.logger('average_distance_centroids', torch.mean(distance_centroids).item(), on_epoch=True)
            self.logger('min_distance_centroids', distance_centroids.min().item(), on_epoch=True)
            self.logger('max_distance_centroids', distance_centroids.max().item(), on_epoch=True)
            self.logger('mean_distance_intra_cluster', distance_intra_cluster, on_epoch=True)
            self.logger('clustering_attraction_loss', attraction_loss.item(), on_epoch=True)
            self.logger('clustering_repulsion_loss', repulsion_loss.item(), on_epoch=True)
            self.logger('clustering_total_loss', total_loss.item(), on_epoch=True)

        return total_loss
