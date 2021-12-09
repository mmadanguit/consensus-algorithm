"""
Sources:
    https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

class Cluster:

    def __init__(self, data, n_clusters):
        self.data = data
        self.n_clusters = n_clusters

    def find_closest_centroid(self, centers):
        """Calculates the Euclidean distance between each data point and each centroid.

        Parameters:
            data (array): dataset to create clusters from.
            centers (array): centroid of each cluster.

        Returns:
            ind (array): contains cluster assignment of each data point.
        """
        ind = np.zeros((self.data.shape[0],))

        for i, x in enumerate(self.data):
            dist = np.fromiter((np.linalg.norm(x-i) for i in centers), float)
            min = np.argmin(dist)
            ind[i] = min

        return ind

    def find_clusters(self, rseed, max_iter=300):
        """Performs k-means clustering.

        Parameters:
            data (array): dataset to create clusters from.
            n_clusters (int): number of clusters to form.
            max_iter (int): maximum number of iterations of the algorithm for a single run.
            rseed (int): random seed to intialize random number generator.

        Returns:
            centers (array): contains centroid of each cluster.
            labels (array): contains cluster assignment of each data point.
        """
        # Randomly initialize n centroids
        rng = np.random.RandomState(rseed)
        i = rng.permutation(self.data.shape[0])[:self.n_clusters]
        centers = self.data[i]

        iter = 0
        while iter <= max_iter:
            # Assign each point to its closest centroid
            labels = self.find_closest_centroid(centers)

            # Compute the new centroid (mean) of each cluster
            new_centers = np.zeros(centers.shape)

            for i in range(self.n_clusters):
                new_center = self.data[labels == i].mean(0)
                new_centers[i] = new_center

            # Check for convergence
            if np.all(centers == new_centers):
                break

            centers = new_centers
            iter += 1

        return centers, labels

    def compute_sse(self, centers, labels):
        """Compute the sum of the squared error (SSE) to evaluate the cluster assignments."""
        sse = 0
        for i in range(centers.shape[0]):
            error = np.fromiter((np.linalg.norm(x-centers[i]) for x in self.data[labels == i]), float)
            sse += sum(error)

        return sse

    def find_clusters_opt(self, n_init=10):
        """Run k-means clustering algorithm multiple times to optimize for SSE.

        Parameters:
            n_init (int): number of times the k-means algorithm will run with different centroid seeds.
        """
        centers_opt = 0
        labels_opt = 0
        min_sse = 1000

        for i in range(n_init):
            centers, labels = self.find_clusters(i)
            sse = self.compute_sse(centers, labels)
            if sse < min_sse:
                centers_opt = centers
                labels_opt = labels

        return centers_opt, labels_opt

if __name__ == "__main__":
    # Generate dataset containing four distinct blots
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    cluster = Cluster(X, 4)
    centers, labels = cluster.find_clusters_opt()
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.show()
