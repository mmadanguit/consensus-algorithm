"""
Sources:
    https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from pca import *
import sys

class Cluster:

    def __init__(self, data):
        self.data = data # Dataset to create clusters from
        self.n_clusters = 4 # Number of clusters to form

    def find_closest_centroid(self, centers):
        """Assigns each data point to the centroid closest to it.

        Parameters:
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

    def find_clusters(self, n_clusters, rseed, max_iter=300):
        """Performs k-means clustering.

        Parameters:
            n_clusters(int): number of clusters to form
            rseed (int): random seed to intialize random number generator.
            max_iter (int): maximum number of iterations of the algorithm for a single run.

        Returns:
            centers (array): contains centroid of each cluster.
            labels (array): contains cluster assignment of each data point.
        """
        # Randomly initialize n centroids
        rng = np.random.RandomState(rseed)
        i = rng.permutation(self.data.shape[0])[:n_clusters]
        centers = self.data[i]

        iter = 0
        while iter <= max_iter:
            # Assign each point to its closest centroid
            labels = self.find_closest_centroid(centers)

            # Compute the new centroid (mean) of each cluster
            new_centers = np.zeros(centers.shape)

            for i in range(n_clusters):
                new_center = self.data[labels == i].mean(0)
                new_centers[i] = new_center

            # Check for convergence
            if np.all(centers == new_centers):
                break

            centers = new_centers
            iter += 1

        return centers, labels

    def compute_sse(self, centers, labels):
        """Computes the sum of the squared error (SSE) to evaluate the cluster assignments."""
        sse = 0
        for i in range(centers.shape[0]):
            error = np.fromiter((np.linalg.norm(x-centers[i]) for x in self.data[labels == i]), float)
            sse += sum(error)

        return sse

    def find_clusters_opt(self, n_clusters, n_init=10):
        """Runs k-means clustering algorithm multiple times to optimize for SSE.

        Parameters:
            n_clusters (int): # Number of clusters to form
            n_init (int): number of times the k-means algorithm will run with different centroid seeds.
        """
        centers_opt = 0
        labels_opt = 0
        min_sse = 1000

        for i in range(n_init):
            centers, labels = self.find_clusters(n_clusters, i)
            sse = self.compute_sse(centers, labels)
            if sse < min_sse:
                centers_opt = centers
                labels_opt = labels
                min_sse = sse

        return centers_opt, labels_opt, min_sse

    def find_n_clusters(self, max_clusters=10):
        """Find the appropriate number of clusters based on the elbow point.

        Parameters:
            max_clusters (int): maximum number of clusters to test
        """
        n_clusters = []
        sse = []

        for i in range(1, max_clusters+1):
            print(f'Testing {i} clusters')
            centers, labels, sse_i = self.find_clusters_opt(i)
            n_clusters.append(i)
            sse.append(sse_i)

        return n_clusters, sse

if __name__ == "__main__":
    filename = sys.argv[1]
    res = pca.reduce_dimensions(filename)

    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    cluster = Cluster(res)

    n_clusters, sse = cluster.find_n_clusters()
    plt.plot(n_clusters, sse)
    plt.show()

    n = input("Enter the number of clusters: ")

    centers, labels, sse = cluster.find_clusters_opt(int(n))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.show()
