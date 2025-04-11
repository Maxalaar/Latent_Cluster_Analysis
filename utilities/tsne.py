import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path


def tsne(embeddings: torch.Tensor, path: Path = None):
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure()
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.title('t-SNE des embeddings')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')

    if path is not None:
        # path.parent.mkdir(parents=True, exist_ok=True)
        file_path = path / 'tsne'
        plt.savefig(file_path, bbox_inches='tight')
        print(f"t-SNE plot saved to {file_path}")
    else:
        plt.show()

    plt.close()