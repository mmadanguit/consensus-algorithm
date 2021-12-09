import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

df = pd.read_csv('ResponseMatrix.csv')
print(df.head())

# scale the data so the mean is 0 and standard deviation is 1
data_scaled = StandardScaler().fit_transform(df)
print(data_scaled[:5])

# find the covariance matrix that gives the relative variance of the data
data_scaled_transform = data_scaled.T
cov_matrix = np.cov(data_scaled_transform)
print(cov_matrix[:5])
# we can see that it's symmetrical across the diagonal

# now decompose the covariance matrix into eigenvalue and eigenvectors
values, vectors = np.linalg.eig(cov_matrix)
print(values[:5])
print(vectors[:5])

# the first 3 eigenvectors in our array are our three principal components!
principal_1 = data_scaled.dot(vectors.T[0])
principal_2 = data_scaled.dot(vectors.T[1])

res = pd.DataFrame(principal_1, columns=['PC1'])
res['PC2'] = principal_2
print(res.head())

plt.scatter(res['PC1'], res['PC2'])
plt.show()
