'''
This is a demo/test project to train a XGBoost model for iris data set.
Structure:
    1. load dataset: iris
    2. exploratory data analysis
    3.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import missingno
import xgboost as xgb
import seaborn as sns

# data
raw_data = datasets.load_iris()
data = raw_data['data']
feature_names = raw_data['feature_names']
data_set = pd.DataFrame(data, columns=feature_names)
target = raw_data['target']

# EDA
# basic analysis
data_set.shape  # (150, 4)
data_set.describe()
# missing values
data_set.isnull().sum()  # no missing values
# missingno.matrix(data_set)
# distribution of labels
sns.countplot(target)  # balanced labels
# correlation analysis
sns.heatmap(data_set.corr())  # some features are highly correlated

# split dataset into train and test set
ind = np.random.rand(len(data_set)) < 0.8
train_set_origin = data_set[ind]
test_set_origin = data_set[~ind]
# reset index
train_set = train_set_origin.reset_index(drop=True)
test_set = test_set_origin.reset_index(drop=True)
# train label and test label
train_label = target[ind]
test_label = target[~ind]

# Train model
xgb_train = xgb.DMatrix(train_set, label=train_label)
params = {
    "objective":"multi:softmax",
    "eta": 0.1,
    "max_depth": 5,
    "num_class": 3
}
num_round = 50
watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]
xgb_test = xgb.DMatrix(test_set, label=test_label)
xgb_model = xgb.train(params, xgb_train, num_round, watchlist)