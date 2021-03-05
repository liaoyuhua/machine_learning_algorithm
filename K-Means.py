# import required modules
import numpy as np
from sklearn import datasets
import pandas as pd
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt

# sample data set
# In fact, this data set is dict-type, so we need to transform it.
# To make it convenient to display the result of clustering, we choose only two features and combine them.
iris = datasets.load_iris()
iris.keys()
data = pd.DataFrame(iris['data'][:, [0, 1]], columns=['sepal length', 'sepal width'])


# implement k-means clustering algorithm
def kmeans(data, k, max_iterations, random_seed=1):
    np.random.seed(random_seed)
    init_ind = np.random.choice(len(data), k)
    centroids = data.iloc[init_ind, :]
    dist = cdist(data, centroids)
    clusters = np.array([np.argmin(i) for i in dist])
    for i in range(max_iterations):
        centroids = []
        for j in range(k):
            temp = np.array(data[clusters == j].mean())
            centroids.append(temp)
        dist = cdist(data, centroids)
        clusters = np.array([np.argmin(i) for i in dist])
        return clusters


# visualization
# sns.scatterplot(data.iloc[:,0], data.iloc[:,1])
clusters = kmeans(data, 4, 100)
for i in np.unique(clusters):
    plt.scatter(data.iloc[clusters == i, 0], data.iloc[clusters == i, 1], label=i)
plt.legend()
plt.show()
