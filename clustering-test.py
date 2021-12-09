"""
Sources:
    https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.datasets import make_blobs

def find_closest_centroid(data, centers):
    ind = np.zeros((data.shape[0],))

    # Calculate Euclidean distance between data point and each centroid
    for i, x in enumerate(data):
        dist = np.fromiter((np.linalg.norm(x-i) for i in centers), float)
        min = np.argmin(dist)
        ind[i] = min

    return ind

def find_clusters(data, n_clusters, rseed=2):
    """Performs k-means clustering.

    Parameters:
        data (array): dataset to create clusters from.
        n_clusters (int): number of clusters to form.
        rseed (int): random seed to intialize random number generator.

    Returns:
        centers (array): contains centroid of each cluster.
        labels (array): contains cluster assignment of each data point.
    """
    # Randomly initialize n centroids
    rng = np.random.RandomState(rseed)
    i = rng.permutation(data.shape[0])[:n_clusters]
    centers = data[i]

    while True:
        # Assign each point to its closest centroid
        labels = find_closest_centroid(data, centers)

        # Compute the new centroid (mean) of each cluster
        new_centers = np.zeros(centers.shape)

        for i in range(n_clusters):
            new_center = data[labels == i].mean(0)
            new_centers[i] = new_center

        print("New centers")
        print(new_centers)

        # Check for convergence
        if np.all(centers == new_centers):
            break

        centers = new_centers
        print("Again!")

    return centers, labels

# Generate dataset containing four distinct blots
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
centers, labels = find_clusters(X, 4)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

# Compute the sum of the squared error (SSE) to evaluate the cluster assignments
