from sklearn.metrics import adjusted_rand_score
from itertools import combinations
import numpy as np


def mean_adjusted_rand_index(cluster_labels_list):
    """
    Computes the mean Adjusted Rand Index between all pairs
    of cluster label vectors in the list.

    Args:
        cluster_labels_list (list of list or np.array): List of clustering label vectors

    Returns:
        float: Mean Adjusted Rand Index (ARI)
    """
    n = len(cluster_labels_list)
    if n < 2:
        return 1.0  # Trivial case: only one clustering

    scores = []
    for a, b in combinations(cluster_labels_list, 2):
        score = adjusted_rand_score(a, b)
        scores.append(score)

    return np.mean(scores)