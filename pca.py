import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import numpy as np
import pandas as pd
import sys

import cluster

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import csv

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

if __name__ == "__main__":
    res = reduce_dimensions(filename) # reduce to 3 dimensions
    res_nd = res.to_numpy()

    df = get_opinion_groups(filename)

    #res_nd, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
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
