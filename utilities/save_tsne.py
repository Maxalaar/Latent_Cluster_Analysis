import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from pathlib import Path


def save_tsne(projection_2d, path: Path = None, cluster_label = None, name: str = 't-SNE'):
    plt.figure()

    if cluster_label is not None:
        # Utiliser les labels pour colorier les points
        scatter = plt.scatter(projection_2d[:, 0], projection_2d[:, 1], c=cluster_label, cmap='viridis')
        # Ajouter une légende pour les couleurs
        legend = plt.colorbar(scatter)
        legend.set_label('Cluster Label')
    else:
        plt.scatter(projection_2d[:, 0], projection_2d[:, 1])

    plt.title('t-SNE des embeddings')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')

    if path is not None:
        # path.parent.mkdir(parents=True, exist_ok=True)
        file_path = path / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_path, bbox_inches='tight')
        print(f"t-SNE plot saved to {file_path}")
    else:
        plt.show()

    plt.close()