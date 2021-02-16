# import required modules
import numpy as np
from sklearn import datasets
import pandas as pd
import math

# sample data set
# In fact, this data set is dict-type, so we need to transform it.
# To make it convenient to display the result of clustering, we choose only two features and combine them.
iris = datasets.load_iris()
iris.keys()
data = pd.DataFrame(iris['data'][:, [0, 1]], columns=['sepal length', 'sepal width'])


# implement k-means clustering algorithm
def EuclidDistance(x, y):
    return math.sqrt(sum(pow((x - y), 2)))


def AllocateCluster(data, centroid, k):
    cluster = []
    sse = []
    for i in range(len(data)):
        temp_dist = []
        for j in range(k):
            temp_dist.append(EuclidDistance(data.loc[i, :], centroid.loc[j, :]))
        cluster.append(np.nanargmin(temp_dist))
        sse.append(sum(pow(min(temp_dist) - centroid.iloc[np.nanargmin(temp_dist), :], 2)))
    return cluster, sse


def FindCentroid(data, cluster_list, k):
    dim = data.shape[1]
    centroid_mat = np.zeros((k, dim))
    for i in range(k):
        ind = np.where(cluster_list == i)
        centroid_mat[i, :] = np.mean(data.iloc(ind))
    return pd.DataFrame(centroid_mat)


def Kmeans(data, k, max_iteration, min_sse_lifting):
    """

    :param data: data set to implement clustering algorithm
    :param k: number of clusters
    :param max_iteration: maximum number of iteration
    :param min_sse_lifting: threshold of lift of SSE between current iteration and last iteration
    :return:
    """
    init_centroid = data.sample(n=k)
    cluster, last_sse = AllocateCluster(data, init_centroid)[0], AllocateCluster(data, init_centroid)[1]
    iteration = 1
    sse_lifting = min_sse_lifting + 1
    while iteration <= max_iteration and sse_lifting > min_sse_lifting:
        centroid = FindCentroid(data, cluster, k)
        cluster, current_sse = AllocateCluster(data, centroid, k)[0], AllocateCluster(data, centroid, k)[1]
        sse_lifting = current_sse - last_sse
        last_sse = current_sse
        iteration += 1
    return cluster