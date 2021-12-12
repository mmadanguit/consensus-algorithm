"""
Sources:
    https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
"""

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import numpy as np
import pandas as pd
import sys

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import csv


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

def reduce_dimensions(csv):
    # scale the data so the mean is 0 and standard deviation is 1
    data_scaled = StandardScaler().fit_transform(csv)

    # find the covariance matrix that gives the relative variance of the data
    data_scaled_transform = data_scaled.T
    cov_matrix = np.cov(data_scaled_transform)
    # we can see that it's symmetrical across the diagonal

    # now decompose the covariance matrix into eigenvalue and eigenvectors
    values, vectors = np.linalg.eig(cov_matrix)

    # the first 3 eigenvectors in our array are our three principal components!
    principal_1 = data_scaled.dot(vectors.T[0])
    principal_2 = data_scaled.dot(vectors.T[1])
    principal_3 = data_scaled.dot(vectors.T[3])

    res = pd.DataFrame(principal_1, columns=['PC1'])
    res['PC2'] = principal_2
    res['PC3'] = principal_3

    return res


def plot_pca(dataframe):
    ax = plt.axes(projection='3d')
    ax.plot3D(dataframe['PC1'], dataframe['PC2'], dataframe['PC3'], '.')
    plt.show()


def get_opinion_groups(filename):
    df = pd.read_csv(filename)
    res = reduce_dimensions(df) # reduce to 3 dimensions

    res_nd = res.to_numpy()

    # res_nd, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    cluster = Cluster(res_nd)

    n_clusters, sse = cluster.find_n_clusters() # find elbow point visually

    n = 3 # input number of clusters indicated by elbow point

    centers, labels, sse = cluster.find_clusters_opt(int(n)) # find clusters

    df['GROUP'] = labels
    df.to_csv(r'~/Documents/ResponseMatrix_Mod.csv')


    return df

def get_consensus(filename):
    df = pd.read_csv(filename)
    nd = df.to_numpy()

    group0 = []
    group1 = []
    group2 = []

    group0_agree = [0]*17
    group0_disagree = [0]*17
    group0_pass = [0]*17

    group1_agree = [0]*17
    group1_disagree = [0]*17
    group1_pass = [0]*17

    group2_agree = [0]*17
    group2_disagree = [0]*17
    group2_pass = [0]*17

    consensus = [0]*17

    for i in range(0, 73):
        row = nd[i,:]

        group = nd[i,17]

        if int(group) == 0:
            group0.append(row)
        if int(group) == 1:
            group1.append(row)
        if int(group) == 2:
            group2.append(row)

    for i in group0:
        for j in range(16):
           if i[j] == 0.0:
               group0_disagree[j] = group0_disagree[j] + 1
           if i[j] == 1.0:
               group0_agree[j] = group0_agree[j] + 1
           if i[j] == 0.5:
               group0_pass[j] = group0_pass[j] + 1

    for i in group1:
        for j in range(16):
           if i[j] == 0.0:
               group1_disagree[j] = group1_disagree[j] + 1
           if i[j] == 1.0:
               group1_agree[j] = group1_agree[j] + 1
           if i[j] == 0.5:
               group1_pass[j] = group1_pass[j] + 1

    for i in group2:
        for j in range(16):
           if i[j] == 0.0:
               group2_disagree[j] = group2_disagree[j] + 1
           if i[j] == 1.0:
               group2_agree[j] = group2_agree[j] + 1
           if i[j] == 0.5:
               group2_pass[j] = group2_pass[j] + 1


    group0_pass_avg = [(i / len(group0))*100 for i in group0_pass]
    group0_agree_avg = [(i / len(group0))*100 for i in group0_agree]
    group0_disagree_avg = [(i / len(group0))*100 for i in group0_disagree]

    group1_pass_avg = [(i / len(group1))*100 for i in group1_pass]
    group1_agree_avg = [(i / len(group1))*100 for i in group1_agree]
    group1_disagree_avg = [(i / len(group1))*100 for i in group1_disagree]

    group2_pass_avg = [(i / len(group2))*100 for i in group2_pass]
    group2_agree_avg = [(i / len(group2))*100 for i in group2_agree]
    group2_disagree_avg = [(i / len(group2))*100 for i in group2_disagree]

    for i in range(16):
        consensus[i] = abs(((group0_agree_avg[i]+group1_agree_avg[i]+group2_agree_avg[i])/3) - 50)












    # calculate percentage agreement for each question in all three get_opinion_groups

    # find average of three percentages for eac statement

    # take absolute value of subtraction from 50 for each average


if __name__ == "__main__":
    filename = sys.argv[1]
    #res = reduce_dimensions(filename) # reduce to 3 dimensions

    #df = get_opinion_groups(filename)
    get_consensus(filename)
    #res_nd = res.to_numpy()

    # res_nd, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    #cluster = Cluster(res_nd)

    #n_clusters, sse = cluster.find_n_clusters() # find elbow point visually
    #plt.plot(n_clusters, sse)
    #plt.show()

    #n = input("Enter the number of clusters: ") # input number of clusters indicated by elbow point

    #centers, labels, sse = cluster.find_clusters_opt(int(n))
    # find clusters

    #fig = plt.figure() # plot clusters IN 3D !
    #ax = fig.add_subplot(111, projection='3d')
    #p3d = ax.scatter(res_nd[:, 0], res_nd[:, 1],res_nd[:, 2], s=50, c=labels, cmap='viridis')
    #plt.show()
