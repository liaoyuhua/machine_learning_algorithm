import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets

# build data set
n = 1000
X = tf.random.uniform([n,5],minval=-10,maxval=10)
w0 = tf.constant([[2.0],[-3.0],[5.8],[-0.9],[1.4]])
b0 = tf.constant([[3.0]])
Y = X@w0 + b0 + tf.random.normal([n,1],mean = 0.0,stddev= 4.0)

# keras: linear regression
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(2,)))
model.summary()
model.compile(
    optimizer='adam',
    loss='mse'
)
history = model.fit(X,Y,epochs=1000)
pred = model.predict(X)
plt.scatter(x=Y, y=pred)

# mlp
model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,), activation='relu'),
                             tf.keras.layers.Dense(8, input_shape=(10,),activation='relu'),
                             tf.keras.layers.Dense(1)])
model.summary()
model.compile(
    optimizer='adam',
    loss='mse'
)
model.fit(X, Y, epochs=1000)

