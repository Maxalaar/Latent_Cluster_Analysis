import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from pathlib import Path

def tsne(embeddings: torch.Tensor, path: Path = None, cluster_label = None):
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure()

    if cluster_label is not None:
        # Utiliser les labels pour colorier les points
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_label, cmap='viridis')
        # Ajouter une légende pour les couleurs
        legend = plt.colorbar(scatter)
        legend.set_label('Cluster Label')
    else:
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1])

    plt.title('t-SNE des embeddings')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')

    if path is not None:
        # path.parent.mkdir(parents=True, exist_ok=True)
        file_path = path / 'tsne.png'
        plt.savefig(file_path, bbox_inches='tight')
        print(f"t-SNE plot saved to {file_path}")
    else:
        plt.show()

    plt.close()
