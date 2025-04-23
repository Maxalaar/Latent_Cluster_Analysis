import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from pathlib import Path

def tsne(embeddings: torch.Tensor):
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(embeddings)
    return tsne_results