# consensus-algorithm

This is the repository for our Discrete Math 2021 Final Project, based on work by groups like [Pol.is](https://pol.is/home), that explores graph clustering in applications of alternative social media algorithms that optimize for consensus rather than conflict. 

We perform PCA and k-means clustering on survey data that we collected here at Olin. All of this data can be found in the `data` folder. 

We use `pca.py` to reduce the dimensionality of our survey data. We then perform k-means clustering in `cluster.py` to group our participants into opinion groups. `opinion-groups.py` finds the statements that best define each opinion groups as well as the statements that are areas of consensus for each opinion group.  

