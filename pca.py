import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import sys


def reduce_dimensions(filename):
    df = pd.read_csv(filename)

    # scale the data so the mean is 0 and standard deviation is 1
    data_scaled = StandardScaler().fit_transform(df)

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


if __name__ == "__main__":
    filename = sys.argv[1]
    print(filename)

    res = reduce_dimensions(filename)
    plot_pca(res)
