import numpy as np
import pandas as pd
from sklearn import datasets
from matplotlib import pyplot as plt

# data set
iris = datasets.load_iris()
iris.keys()
data = pd.DataFrame(iris['data'][:, [0, 1]], columns=['sepal length', 'sepal width'])


# implementing DBSCAN from scratch
class dbscan:
    def __init__(self, df, eps, minpts, distance_type):
        self.df = np.array(df)
        self.label = np.array(len(self.df) * [-1])
        self.eps = eps
        self.minpts = minpts
        self.distance_type = distance_type
        self.cluster = 0
        self.noise = 0

    def distance(self, point1, point2):
        if self.distance_type == 'euclidean':
            return np.sqrt(sum(pow(point1 - point2, 2)))
        elif self.distance_type == 'manhattan':
            return sum(abs(e1 - e2) for e1, e2 in zip(point1, point2))
        else:
            return np.sqrt(sum(pow(point1 - point2, 2)))

    def regionQuery(self, instance):
        neighbours = []
        for i in range(len(self.df)):
            tmp = self.df.iloc[i]
            if self.distance(instance, tmp) <= self.eps:
                neighbours.append(i)
        return neighbours

    def expandCluster(self, point, clusterId):
        neighbours = self.regionQuery(self.iloc[point], clusterId)
        if neighbours < self.minpts:
            self.label[point] = self.noise
            return False
        else:
            self.label[point] = clusterId
            for i in neighbours:
                if self.label[i] == -1:
                    self.label[i] == self.cluster

    def fit(self):

        for i

# visualization
